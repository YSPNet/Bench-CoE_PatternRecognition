"""Aggregate E6 native-router sweeps after every requested seed completes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common import (OUTPUT, RESULT_COLUMNS, ROOT, SEEDS, SUMMARY_COLUMNS,
                    make_audit, read_json, summarize, utc_now, view_uids,
                    write_csv, write_json)

OUT = OUTPUT / "E6"
UPSTREAM = ROOT / "scripts/journal_20261024/upstream"
ARCHIVE = Path("/home/sm5/ys/ICASSP/archive/non_paper_research_20261011")
METHODS = ("routerdc", "vlc", "mlp", "cosinecls", "zooter")
PROTOCOLS = ("full14", "heldout11")
LAMBDAS = ("0", "10", "100", "1000", "10000", "inf")
ALL_COLUMNS = ("method", "protocol", "seed", "lambda", "view", "n_test", "A", "C", "R",
               "source_val_A", "source_val_C", "source_val_R", "training_source", "prediction")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    sizes = {k: len(v) for k, v in view_uids().items()}
    audit_path = OUT / "AUDIT.json"
    if audit_path.exists():
        audit = read_json(audit_path)
    else:
        audit = make_audit("E6", "journal:E6; e6_prepare.py; e6_task.py; e6_aggregate.py",
            "select=val maximizes source-dev Rank per seed/protocol/method; "
            "select=best_test is a separately labeled diagnostic using each test view, "
            "including 3OOD, as explicitly required by E6. It is never used for comparable model choice.", sizes)
    official_commit = __import__("subprocess").run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True,
        text=True, check=True).stdout.strip()
    audit["git_commit"] = official_commit
    audit["git_commit_note"] = "Pinned official VL-RouterBench release cloned under scripts/journal_20261024/upstream"
    audit["released_code_url"] = "https://github.com/K1nght/VL-RouterBench"
    audit["local_code_sha256"] = {name: sha(ROOT / f"routers/{name}/router.py") for name in METHODS}
    audit["upstream_code_sha256"] = {name: sha(UPSTREAM / f"routers/{name}/router.py") for name in METHODS}
    audit["routerdc_deviation"] = ("Local RouterDC preserves the canonical asset-list container so the native "
        "image dataset reads the actual first image; upstream passes a bare dict and silently uses a black placeholder. "
        "Native model, optimizer, losses and training loop are otherwise reused. See SOURCE_DIFF.patch.")
    audit["cosinecls_deviation"] = ("Local CosineCls also preserves the canonical asset-list container "
        "so strict images are read. Upstream would pass a bare dict and silently substitute a placeholder. "
        "The local source change is recorded in SOURCE_DIFF.patch.")
    audit["mlp_training_adapter"] = ("Released MLPRouter fusion, soft-target construction, network, "
        "prediction and serialization are reused. The journal adapter uses AdamW, batch 16, "
        "lr 2e-5 and weight decay 0.01 as required by Appendix H because the released "
        "MLPRouter.fit GPU branch hardcodes Adam and batch size 1024 and is gated by cuML. "
        "The adapter selects the epoch on source-dev Rank Score.")
    audit["scheduler_deviation"] = ("The released RouterDC, VLC, CosineCls and ZOOTER "
        "fit methods use a linear warmup/decay scheduler, whereas Appendix H describes "
        "cosine decay. E6 retains the released training loops and records this discrepancy; "
        "the MLP Appendix-H adapter uses cosine decay.")
    audit["optional_baseline_smoke"] = read_json(OUT / "optional_smoke/RESULT.json")
    audit["historical_11_source"] = str(ARCHIVE / "outputs/table_a2_factory_v2_20260914")
    audit["historical_11_source_fingerprint"] = read_json(
        ARCHIVE / "outputs/table_a2_factory_v2_20260914/EXPERIMENT.json")["data_fingerprint"]
    audit["gpu"] = "Four NVIDIA RTX A6000 GPUs; one physical GPU per task"
    boost_path = OUT / "BOOST_AUDIT.json"
    if boost_path.is_file():
        audit["parallel_boost"] = read_json(boost_path)
    archive_sweep = OUT / "ARCHIVE_SWEEP.json"
    if archive_sweep.is_file():
        audit["historical_11_source_sweep"] = read_json(archive_sweep)
    bounds = read_json(ROOT / "data/matrices/cost_bounds.json")
    audit["c_min"], audit["c_max"] = bounds["cmin"], bounds["cmax"]
    audit["feature_pair"] = "BAAI/bge-m3 + google/siglip-large-patch16-384; normalize_concat"
    audit["feature_files_sha256"] = {
        "bge-m3": sha(ROOT / "EMBEDDINGS/text/bge-m3.parquet"),
        "siglip-large-patch16-384": sha(OUT / "siglip-large-patch16-384.parquet"),
    }
    audit["image_manifest_sha256"] = sha(OUT / "IMAGES.json")
    image_paths = read_json(OUT / "IMAGE_PATHS.json")
    audit["image_policy"] = {
        "samples": len(image_paths),
        "multi_image_samples": sum(int(item.get("image_count", 0) > 1)
                                   for item in image_paths.values()),
        "native_baseline_input": "first canonical decoded image per sample",
        "mlp_siglip_input": "all canonical images; mean image embedding followed by L2 normalization",
        "fallbacks": "synthetic black/white image replacement forbidden during native fit and predict",
    }
    audit["training"] = {"optimizer": "AdamW", "lr": 2e-5, "batch_size": 16,
                          "weight_decay": .01, "epochs_max": 5, "lambdas": list(LAMBDAS)}
    results, all_candidates, choices, incomplete = [], [], [], []
    audit["failures"] = []
    orchestrator_failures = OUT / "ORCHESTRATOR_FAILURES.json"
    if orchestrator_failures.is_file():
        for item in read_json(orchestrator_failures):
            task = item["task"]
            seed = task[2] if isinstance(task, list) else None
            trace = OUT / ("siglip_run.log" if task == "siglip_feature_extraction"
                           else f"tasks/{task[0]}/{task[1]}/seed-{seed}/run.log")
            if not trace.is_file():
                trace = OUT / "run.log"
            audit["failures"].append({
                "id": "/".join(map(str, task)) if isinstance(task, list) else str(task),
                "seed": seed,
                "error": item.get("reason", f"native task exit code {item.get('exit_code')}"),
                "traceback_path": str(trace.relative_to(ROOT)),
            })
    for protocol in PROTOCOLS:
        for method in METHODS:
            for seed in SEEDS:
                task = OUT / "tasks" / protocol / method / f"seed-{seed}"
                complete = task / "COMPLETE.json"
                if not complete.is_file():
                    incomplete.append(f"{protocol}/{method}/{seed}")
                    for path in sorted(task.glob("lambda-*/traceback.txt")):
                        audit["failures"].append({"id": f"{protocol}/{method}/{path.parent.name}",
                            "seed": seed, "error": path.read_text().splitlines()[-1],
                            "traceback_path": str(path.relative_to(ROOT))})
                    continue
                frozen = read_json(complete)
                candidates = [read_json(task / f"lambda-{lam}/CANDIDATE.json") for lam in LAMBDAS]
                if [item["lambda"] for item in candidates] != list(LAMBDAS):
                    raise RuntimeError("candidate sweep order or coverage changed")
                val = max(candidates, key=lambda item: item["val"][2])
                if frozen["lambda_val"] != val["lambda"]:
                    raise RuntimeError("source validation selection checkpoint changed")
                for view in (("14ID",) if protocol == "full14" else ("11ID", "3OOD")):
                    best_test = max(candidates, key=lambda item: item["test"][view][2])
                    if frozen["lambda_best_test"][view] != best_test["lambda"]:
                        raise RuntimeError("test diagnostic selection checkpoint changed")
                    for item in candidates:
                        metric = item["test"][view]
                        all_candidates.append(dict(method=method, protocol=protocol, seed=seed,
                            lambda_=item["lambda"], view=view, n_test=sizes[view],
                            A=metric[0], C=metric[1], R=metric[2],
                            source_val_A=item["val"][0], source_val_C=item["val"][1],
                            source_val_R=item["val"][2], training_source=item["source"],
                            prediction=item["predictions"][view]))
                    for selection, item in (("val", val), ("best_test", best_test)):
                        A, C, R = item["test"][view]
                        results.append(dict(experiment="E6", method=method,
                            variant=f"lambda={'source_val' if selection == 'val' else 'test_best'}/select={selection}",
                            view=view, seed=seed, n_test=sizes[view], A=A, C=C, R=R,
                            notes=f"actual_lambda={item['lambda']}; training_source={item['source']}"))
                        choices.append(dict(method=method, protocol=protocol, seed=seed,
                            view=view, select=selection, chosen_lambda=item["lambda"],
                            selection_R=item["val"][2] if selection == "val" else R))
    write_csv(OUT / "RESULTS.csv", RESULT_COLUMNS, results)
    write_csv(OUT / "SUMMARY.csv", SUMMARY_COLUMNS, summarize(results))
    write_csv(OUT / "ALL_LAMBDAS.csv", ALL_COLUMNS,
              [{"lambda" if key == "lambda_" else key: value for key, value in row.items()}
               for row in all_candidates])
    write_csv(OUT / "lambda_selections.csv",
              ("method", "protocol", "seed", "view", "select", "chosen_lambda", "selection_R"), choices)
    audit["incomplete_tasks"] = incomplete
    audit["status"] = "complete" if not incomplete and not audit["failures"] else "incomplete"
    if audit["status"] == "complete":
        summaries = summarize(results)
        acceptance = {}
        for method, target in (("routerdc", 74.59), ("vlc", 74.33), ("mlp", 74.23)):
            row = next(item for item in summaries if item["method"] == method and
                       item["view"] == "14ID" and "best_test" in item["variant"])
            acceptance[method] = {"R_mean": row["R_mean"], "published_R": target,
                                  "abs_delta_R": abs(row["R_mean"] - target),
                                  "within_2_points": abs(row["R_mean"] - target) <= 2}
        audit["acceptance"] = acceptance
        if not all(item["within_2_points"] for item in acceptance.values()):
            audit["status"] = "acceptance_needs_recheck"
    audit["end_time"] = utc_now() if audit["status"] != "incomplete" else None
    write_json(audit_path, audit)
    print(json.dumps({"status": audit["status"], "incomplete": len(incomplete),
                      "rows": len(results), "acceptance": audit.get("acceptance")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
