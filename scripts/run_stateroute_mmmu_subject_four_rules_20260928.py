#!/usr/bin/env python3
"""Replace TinyKNN's dataset profile groups with MMMU-30 subject states.

This is a fixed-rule ablation. The four heldout11 switch rules, Ridge score
thresholds, cost cap and K2 fallback are inherited unchanged. Only the group
used for expert profiles and same-group neighbors changes to the Qwen3-VL
MMMU subject prototype assignment. Full14 keeps the common accuracy action.

Run train, freeze, predict, evaluate for each --protocol full14/heldout11.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import run_stateroute_mmmu_subject_20260928 as subject
from scripts import run_stateroute_tiny_knn_20260928 as tiny
from src.next_round_x.runtime import digest, read_json, record, write_csv, write_json

PROTO = ROOT / "outputs/stateroute_mmmu_subject_20260928"
GUARD = ROOT / "outputs/stateroute_tiny_k2_guard_11_3_20260928"
CAP = ROOT / "outputs/stateroute_tiny_mixguard_costcap_11_3_20260928"
RANK_SAFE = ROOT / "outputs/stateroute_tiny_mixguard_11_3_20260928"
ACCURACY = ROOT / "outputs/stateroute_tiny_mixguard_accuracy_11_3_20260928"
METHODS = ("SubjectRule1-RankSafe-Guard10", "SubjectRule2-Accuracy-Guard10",
           "SubjectRule3-CostCap-Guard10", "SubjectRule4-CostCap-Guard25")
FULL14_METHOD = "SubjectAccuracy14"
SUBJECT_N = 30
QUALITY_KAPPA = 64.
COST_KAPPA = 128.
NEIGHBOR_TEMPERATURE = .08


def paths(protocol: str) -> tuple[Path, Path]:
    if protocol not in ("full14", "heldout11"):
        raise ValueError(protocol)
    stem = "full14" if protocol == "full14" else "11_3"
    return (ROOT / f"outputs/stateroute_mmmu_subject_four_rules_{stem}_20260928",
            ROOT / f"reports/stateroute_mmmu_subject_four_rules_{stem}_20260928")


def say(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def state_profile(y: np.ndarray, c: np.ndarray, states: np.ndarray):
    """The original TinyKNN smoothing, with 30 subject groups."""
    if y.shape != c.shape or y.shape[1] != tiny.N_EXPERTS or len(states) != len(y):
        raise ValueError("source outcome and state alignment changed")
    count = np.bincount(states, minlength=SUBJECT_N)
    if len(count) != SUBJECT_N:
        raise ValueError("invalid subject state")
    global_y = y.mean(axis=0, dtype=np.float64)
    global_c = c.mean(axis=0, dtype=np.float64)
    median_c = float(np.median(c[c > 0]))
    sums_y = np.zeros((SUBJECT_N, tiny.N_EXPERTS), dtype=np.float64)
    sums_c = np.zeros_like(sums_y)
    np.add.at(sums_y, states, y)
    np.add.at(sums_c, states, c)
    quality = (sums_y + QUALITY_KAPPA * global_y) / (count[:, None] + QUALITY_KAPPA)
    state_cost = (sums_c + COST_KAPPA * global_c) / (count[:, None] + COST_KAPPA) / median_c
    global_cost = global_c / median_c
    return quality, state_cost, global_cost, count


def weighted_neighbors(sim: np.ndarray, labels: np.ndarray, k: int):
    """Softmax cosine average over up to k available neighbors."""
    take = min(k, sim.shape[1])
    if not take:
        raise ValueError("empty neighbor set")
    ix = np.argpartition(sim, -take, axis=1)[:, -take:]
    vals = np.take_along_axis(sim, ix, axis=1)
    order = np.argsort(-vals, axis=1)
    ix = np.take_along_axis(ix, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    weights = np.exp((vals - vals[:, :1]) / NEIGHBOR_TEMPERATURE)
    near = labels[ix]
    return ((near * weights[:, :, None]).sum(axis=1) /
            weights.sum(axis=1)[:, None]).astype(np.float32)


def neighbor_quality(query: np.ndarray, bank: np.ndarray, bank_y: np.ndarray,
                     bank_state: np.ndarray, states: np.ndarray):
    """Global top64 and predicted-subject top32, with sparse-state fallback.

    For 1..31 state examples, use all available examples; for an empty state,
    use global top32. This avoids fabricating repeated neighbors.
    """
    n = len(query)
    if bank_y.shape != (len(bank), tiny.N_EXPERTS) or len(bank_state) != len(bank):
        raise ValueError("neighbor bank alignment changed")
    if len(bank) < 64:
        raise ValueError("global bank too small")
    all64 = np.empty((n, tiny.N_EXPERTS), dtype=np.float32)
    same32 = np.empty_like(all64)
    by_state = [np.flatnonzero(bank_state == i) for i in range(SUBJECT_N)]
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        sim = np.asarray(query[start:stop] @ bank.T, dtype=np.float32)
        all64[start:stop] = weighted_neighbors(sim, bank_y, 64)
        for state in np.unique(states[start:stop]):
            rows = np.flatnonzero(states[start:stop] == state)
            columns = by_state[int(state)]
            if len(columns):
                same32[start + rows] = weighted_neighbors(sim[rows][:, columns], bank_y[columns], 32)
            else:
                same32[start + rows] = weighted_neighbors(sim[rows], bank_y, 32)
    return all64, same32


def candidates(model, bank, xt: np.ndarray, available: np.ndarray, states: np.ndarray):
    all64, same32 = neighbor_quality(
        xt, bank, model["bank_y"], model["bank_state"], states)
    base = np.asarray(model["quality"][states], dtype=np.float32)
    qr = .8 * base + .2 * all64
    qa = .8 * base + .2 * same32
    q14 = .6 * base + .4 * same32
    qr[~available] = base[~available]
    qa[~available] = base[~available]
    q14[~available] = base[~available]
    state_cost = model["state_cost"][states]
    global_cost = model["global_cost"]
    rank = (qr - .06 * state_cost).argmax(axis=1).astype(np.int8)
    accuracy = (qa - .025 * global_cost).argmax(axis=1).astype(np.int8)
    full14_accuracy = (q14 - .025 * state_cost).argmax(axis=1).astype(np.int8)
    rows = np.arange(len(states))
    gain = qa[rows, accuracy] - qa[rows, rank]
    delta_cost = global_cost[accuracy] - global_cost[rank]
    return rank, accuracy, full14_accuracy, gain, delta_cost


def fixed_rules(maxscore, rank, accuracy, gain, delta_cost, available, k2, settings):
    """Apply the four previously frozen switching/guard configurations."""
    safe = settings["rank_safe"]
    oriented = settings["accuracy"]
    guard = settings["guard"]
    cap = settings["cap"]
    different = (accuracy != rank) & available
    u = gain - oriented["price_weight"] * delta_cost
    switch1 = ((maxscore >= safe["confidence_threshold"]) &
               (gain - safe["price_weight"] * delta_cost > safe["gain_margin"]) & different)
    switch2 = ((maxscore >= oriented["confidence_threshold"]) &
               (u > oriented["gain_margin"]) & different)
    switch3 = switch2 & (delta_cost <= cap)
    guard10 = (maxscore < guard["0.1"]) & available
    guard25 = (maxscore < guard["0.25"]) & available
    switches = np.stack((switch1, switch2, switch3, switch3))
    gates = np.stack((guard10, guard10, guard10, guard25))
    mixed = np.where(switches, accuracy[None, :], rank[None, :])
    actions = np.where(gates, k2[None, :], mixed).astype(np.int8)
    return actions, switches, gates, u


def settings():
    safe = read_json(RANK_SAFE / "SOURCE_SELECTION.json")["selected"]
    oriented = read_json(ACCURACY / "SOURCE_SELECTION.json")["selected"]
    cap = read_json(CAP / "SOURCE_SELECTION.json")["selected"]["cost_delta_cap"]
    guard = read_json(GUARD / "SOURCE_SELECTION.json")["thresholds"]
    return dict(rank_safe=safe, accuracy=oriented, cap=cap, guard=guard)


def train(protocol: str):
    out, _ = paths(protocol)
    if (out / "SOURCE_SELECTION.json").exists():
        raise FileExistsError("subject-state model already trained")
    proto_meta = read_json(PROTO / "PROTOTYPES.json")
    subject.require_records(proto_meta["sources"] + [proto_meta["artifact"]])
    with np.load(PROTO / "PROTOTYPES.npz", allow_pickle=False) as proto:
        centers = proto["centers"].copy()
        if proto["subjects"].tolist() != subject.SUBJECTS or centers.shape != (30, 2048):
            raise RuntimeError("MMMU prototype contract changed")
    data, xq, xt, available, fit, names, _, tiny_records = tiny.source_inputs(protocol)
    base, _ = tiny.paths(protocol)
    with np.load(base / "MODEL.npz", allow_pickle=False) as prior:
        if prior["bank_uids"].tolist() != data["uids"][fit].tolist():
            raise RuntimeError("source-fit bank UID order changed")
        if not np.array_equal(prior["bank_y"], data["y"][fit]):
            raise RuntimeError("source-fit expert outcome matrix changed")
    states, sim, margin = subject.assign_subjects(xq[fit], centers)
    quality, state_cost, global_cost, count = state_profile(data["y"][fit], data["c"][fit], states)
    if count.sum() != len(fit):
        raise RuntimeError("subject state assignment incomplete")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "MODEL.npz", subjects=np.array(subject.SUBJECTS), centers=centers,
                        quality=quality.astype(np.float32), state_cost=state_cost.astype(np.float32),
                        global_cost=global_cost.astype(np.float32), bank_state=states.astype(np.int8),
                        bank_y=data["y"][fit].astype(np.uint8), support=count)
    info = dict(
        protocol=protocol, method="fixed-rule MMMU subject-state substitution",
        subject_assignment="Qwen3-VL 2048-d query feature, cosine nearest of 30 MMMU official-train centroids",
        subject_prototype_n=735, source_fit_n=len(fit), source_dataset_n=len(names),
        source_fit_subject_counts=dict(zip(subject.SUBJECTS, map(int, count))),
        sparse_subjects_lt32=int((count < 32).sum()), empty_subjects=int((count == 0).sum()),
        mean_subject_cosine=float(sim.mean()), mean_subject_margin=float(margin.mean()),
        smoothing=dict(quality_kappa=QUALITY_KAPPA, cost_kappa=COST_KAPPA),
        rank_candidate="0.8 subject quality + 0.2 all-source cosine top64; lambda 0.06; subject cost",
        accuracy_candidate="0.8 subject quality + 0.2 same-subject cosine top32; lambda 0.025; global cost",
        full14_candidate="0.6 subject quality + 0.4 same-subject cosine top32; lambda 0.025; subject cost",
        sparse_neighbor_policy="use min(32, state support); if zero use global top32",
        tiny_missing_policy="use subject profile without local neighbor correction",
        switch_thresholds_and_k2="reuse previously frozen heldout11 source-calibrated rules unchanged",
        ridge_score_role="original TinyLLaVA dataset Ridge max raw decision score used for gating only",
        source_outcomes_used_only_from_source_fit=True, test_dataset_name_used_for_action=False,
        mmmu_train_inputs_and_subject_labels_used_in_heldout11=True,
        mmmu_test_expert_outcomes_used_in_heldout11=False,
        model=record(out / "MODEL.npz"), prototypes=record(PROTO / "PROTOTYPES.npz"),
        base_model=record(base / "MODEL.npz"), base_bank=record(base / "BANK_FEATURES.npy"),
        tiny_features=[record(p) for p in tiny_records])
    write_json(out / "SOURCE_SELECTION.json", info)
    say(stage="train", protocol=protocol, source_fit_n=len(fit), nonempty_subjects=int((count > 0).sum()),
        sparse_subjects_lt32=info["sparse_subjects_lt32"])


def freeze(protocol: str):
    out, _ = paths(protocol)
    if (out / "MODEL_LOCK.json").exists():
        raise FileExistsError("subject-state model already frozen")
    info = read_json(out / "SOURCE_SELECTION.json")
    subject.require_records([info[k] for k in ("model", "prototypes", "base_model", "base_bank")])
    subject.require_records(info["tiny_features"])
    base, _ = tiny.paths(protocol)
    files = [Path(__file__), out / "SOURCE_SELECTION.json", out / "MODEL.npz",
             PROTO / "PROTOTYPES.npz", PROTO / "PROTOTYPES.json",
             base / "MODEL.npz", base / "BANK_FEATURES.npy", base / "MODEL_LOCK.json",
             ROOT / "data/matrices/cost_bounds.json"]
    if protocol == "heldout11":
        files += [GUARD / "PREDICTIONS.npz", GUARD / "PREDICTION_RECEIPT.json",
                  GUARD / "SOURCE_SELECTION.json", RANK_SAFE / "SOURCE_SELECTION.json",
                  ACCURACY / "SOURCE_SELECTION.json", CAP / "SOURCE_SELECTION.json"]
    files += [ROOT / item["path"] for item in info["tiny_features"]]
    files = list(dict.fromkeys(files))
    write_json(out / "MODEL_LOCK.json", dict(
        status="SUBJECT_STATES_AND_FIXED_RULES_FROZEN_BEFORE_TEST_EVALUATION",
        protocol=protocol, methods=list(METHODS if protocol == "heldout11" else (FULL14_METHOD,)),
        rules=settings() if protocol == "heldout11" else None,
        reused_historical_thresholds=True, source_only_expert_profiles=True,
        mmmu_ood_contamination="MMMU official train inputs and subject labels define prototypes; 3-OOD is MMMU-assisted, not fully unseen"
        if protocol == "heldout11" else None,
        historical_test_exposure="exploratory replay; previous test results visible before this ablation",
        cost_scope="cached expert USD proxy only; excludes Qwen3-VL, TinyLLaVA, MiniLM and router computation",
        files=[record(p) for p in files]))
    say(stage="freeze", protocol=protocol, files=len(files))


def verify_lock(protocol: str):
    out, _ = paths(protocol)
    lock = read_json(out / "MODEL_LOCK.json")
    if lock["protocol"] != protocol:
        raise RuntimeError("model protocol changed")
    subject.require_records(lock["files"])
    return lock


def predict(protocol: str):
    out, _ = paths(protocol)
    if (out / "PREDICTION_RECEIPT.json").exists():
        raise FileExistsError("subject-state predictions already frozen")
    from src.next_round_x.read_fence import install
    denied = install()
    lock = verify_lock(protocol)
    uids, views = tiny.test_uids(protocol)
    xq, overlay, _ = subject.qwen_features(uids, allow_multi=True)
    xt, available, _ = tiny.tiny_features(uids)
    if len(overlay) != 10 or int((~available).sum()) != 10:
        raise RuntimeError("test feature coverage changed")
    base, _ = tiny.paths(protocol)
    with np.load(out / "MODEL.npz", allow_pickle=False) as model, \
         np.load(base / "MODEL.npz", allow_pickle=False) as prior:
        states, sim, margin = subject.assign_subjects(xq, model["centers"])
        bank = np.load(base / "BANK_FEATURES.npy", mmap_mode="r", allow_pickle=False)
        if not np.array_equal(prior["bank_y"], model["bank_y"]):
            raise RuntimeError("frozen neighbor outcomes changed")
        rank, accuracy, full14_accuracy, gain, delta_cost = candidates(
            model, bank, xt, available, states)
        if protocol == "full14":
            actions = full14_accuracy[None, :]
            methods = np.array([FULL14_METHOD])
            maxscore = np.zeros(len(uids), dtype=np.float32)
            switches = np.zeros((1, len(uids)), dtype=bool)
            gates = np.zeros_like(switches)
            u = np.zeros(len(uids), dtype=np.float32)
        else:
            maxscore = (xt @ prior["tiny_coef"].T + prior["tiny_intercept"]).max(axis=1)
            with np.load(GUARD / "PREDICTIONS.npz", allow_pickle=False) as frozen:
                if frozen["uids"].tolist() != uids or not np.allclose(
                        maxscore, frozen["maxscore"], atol=1e-4, rtol=0):
                    raise RuntimeError("frozen K2/Ridge test alignment changed")
                k2 = frozen["k2_actions"].astype(np.int8)
            actions, switches, gates, u = fixed_rules(
                maxscore, rank, accuracy, gain, delta_cost, available, k2, lock["rules"])
            methods = np.array(METHODS)
    if actions.shape != (len(methods), 6108) or (actions < 0).any() or (actions >= tiny.N_EXPERTS).any():
        raise RuntimeError("invalid test expert actions")
    path = out / "PREDICTIONS.npz"
    np.savez_compressed(path, uids=np.array(uids), views=views, methods=methods, actions=actions,
                        subjects=np.array(subject.SUBJECTS), states=states, cosine_similarity=sim,
                        cosine_margin=margin, rank_candidate=rank, accuracy_candidate=accuracy,
                        gain=gain.astype(np.float32), cost_delta=delta_cost.astype(np.float32),
                        utility=u.astype(np.float32), maxscore=maxscore.astype(np.float32),
                        use_accuracy=switches, k2_gate=gates, tiny_available=available)
    write_json(out / "PREDICTION_RECEIPT.json", dict(
        kind="subject_routing_final_evaluation_unlock", target_split="test",
        predictions_frozen=True, method_selection_complete=True, approved=True,
        approval_source="user requested MMMU subject-state experiment with four frozen switch rules",
        prediction_path=str(path.relative_to(ROOT)), prediction_sha256=digest(path),
        model_lock=record(out / "MODEL_LOCK.json"), test_dataset_name_used_for_action=False,
        labels_or_actual_cost_used=False, test_outcomes_opened=0, test_fit_performed=False,
        denied_attempts=denied))
    say(stage="predict", protocol=protocol, n=len(uids),
        switch_n=switches.sum(axis=1).astype(int).tolist(),
        k2_gate_n=gates.sum(axis=1).astype(int).tolist())


def evaluate(protocol: str):
    out, report = paths(protocol)
    if (report / "RESULTS.json").exists():
        raise FileExistsError("subject-state experiment already evaluated")
    from src.subject_routing.final_evaluation import materialize_scoped_test_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit, evaluate_final_unlock
    lock = verify_lock(protocol)
    path = out / "PREDICTIONS.npz"
    receipt_path = out / "PREDICTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    if digest(path) != receipt["prediction_sha256"] or digest(out / "MODEL_LOCK.json") != receipt["model_lock"]["sha256"]:
        raise RuntimeError("frozen prediction/model binding changed")
    uids, expected_views = tiny.test_uids(protocol)
    with np.load(path, allow_pickle=False) as frozen:
        if frozen["uids"].tolist() != uids or frozen["methods"].tolist() != lock["methods"]:
            raise RuntimeError("test UID/method order changed")
        views = frozen["views"].copy()
        actions = frozen["actions"].astype(np.int64)
        switches = frozen["use_accuracy"].copy()
        gates = frozen["k2_gate"].copy()
        states = frozen["states"].copy()
    if not np.array_equal(views, expected_views):
        raise RuntimeError("test view alignment changed")
    audit = UIDAccessAudit.create(Operation.FINAL_EVALUATE, uids)
    with evaluate_final_unlock(receipt_path) as capability:
        labels, provenance = materialize_scoped_test_labels(
            uids, allowed_test_uids=uids, prediction_sha256=digest(path),
            prediction_receipt_sha256=digest(receipt_path), audit=audit,
            unlock_capability=capability, **subject.label_args())
    write_json(out / "OUTCOME_PROVENANCE.json", dict(materialization=provenance, events=audit.events))
    y, c = labels.matrix.y, labels.matrix.c
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    datasets = np.array([uid.split("/", 1)[0] for uid in uids])
    summary, per_dataset = [], []
    view_names = ("14-ID",) if protocol == "full14" else ("11-ID", "3-OOD")
    for j, method in enumerate(lock["methods"]):
        for view in view_names:
            mask = views == view
            summary.append(dict(method=method, view=view,
                                accuracy_switch_n=int(switches[j, mask].sum()),
                                k2_gate_n=int(gates[j, mask].sum()),
                                **tiny.metrics(y[mask], c[mask], actions[j, mask], bounds)))
        for dataset in sorted(set(datasets)):
            mask = datasets == dataset
            per_dataset.append(dict(method=method, dataset=dataset,
                                    view=str(views[np.flatnonzero(mask)[0]]),
                                    accuracy_switch_n=int(switches[j, mask].sum()),
                                    k2_gate_n=int(gates[j, mask].sum()),
                                    **tiny.metrics(y[mask], c[mask], actions[j, mask], bounds)))
    report.mkdir(parents=True, exist_ok=True)
    write_csv(report / "summary.csv", summary)
    write_csv(report / "per_dataset.csv", per_dataset)
    counts = []
    source_counts = read_json(out / "SOURCE_SELECTION.json")["source_fit_subject_counts"]
    for i, name in enumerate(subject.SUBJECTS):
        counts.append(dict(subject=name, mmmu_train_prototype_n=int(
            read_json(PROTO / "PROTOTYPES.json")["support"][name]),
            source_fit_n=source_counts[name],
            **{f"{view}_test_n": int((states[views == view] == i).sum()) for view in view_names}))
    write_csv(report / "per_subject.csv", counts)
    write_json(report / "RESULTS.json", dict(
        protocol=protocol, summary=summary, source_selection=record(out / "SOURCE_SELECTION.json"),
        model_lock=record(out / "MODEL_LOCK.json"), prediction=record(path),
        original_four_rule_thresholds_unchanged=True,
        mmmu_ood_is_not_fully_unseen=protocol == "heldout11",
        historical_test_exposure=lock["historical_test_exposure"],
        cost_scope=lock["cost_scope"]))
    say(stage="evaluate", protocol=protocol, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("full14", "heldout11"), required=True)
    parser.add_argument("--stage", choices=("train", "freeze", "predict", "evaluate"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=12):
        {"train": train, "freeze": freeze, "predict": predict,
         "evaluate": evaluate}[args.stage](args.protocol)


if __name__ == "__main__":
    main()
