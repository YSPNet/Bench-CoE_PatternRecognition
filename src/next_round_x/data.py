"""Frozen input cache and source-scoped materialization. No global Y loader."""
from __future__ import annotations
import json
import numpy as np
from .runtime import ROOT, OUT, digest, read_json, record, write_json, canonical

SPLIT_PATH = ROOT / "artifacts/subject_routing/manifests/splits.json"
POOL_PATH = ROOT / "artifacts/subject_routing/manifests/model_pool.json"


def manifests():
    return read_json(SPLIT_PATH), read_json(POOL_PATH)


def role_uids(protocol, role):
    splits, _ = manifests()
    env = splits["environments"]["id14" if protocol == "full14" else "ood11_3"]
    mapped = role if protocol == "full14" else {"train":"train_seen", "dev":"dev_seen", "id":"id_test", "ood":"ood_test"}[role]
    return list(env["roles"][mapped]["ordered_sample_ids"])


class FeatureCache:
    def __init__(self):
        pointer = read_json(ROOT / "outputs/bench_coe_query/features/qwen3vl/LATEST_FULL.json")
        self.root = ROOT / pointer["cache_root"] / "merged"
        self.manifest = read_json(self.root / "manifest.json")
        for name, key in (("features.npy","features_sha256"),("index.jsonl","index_sha256")):
            if digest(self.root/name) != self.manifest[key]:
                raise ValueError("frozen feature binding changed")
        rows = [json.loads(line) for line in (self.root/"index.jsonl").read_text().splitlines()]
        self.positions = {r["sample_id"]:i for i,r in enumerate(rows)}
        if len(self.positions) != len(rows):
            raise ValueError("duplicate feature UID")
        self.features = np.load(self.root/"features.npy",mmap_mode="r",allow_pickle=False)
        if self.features.shape != (len(rows),2048):
            raise ValueError("frozen feature shape mismatch")

    def rows(self, uids):
        x = np.zeros((len(uids),2048),dtype=np.float32)
        available = np.array([uid in self.positions for uid in uids])
        indices = np.flatnonzero(available)
        x[indices] = self.features[[self.positions[uids[i]] for i in indices]]
        if not np.isfinite(x).all(): raise ValueError("nonfinite features")
        return x, available

    def binding(self):
        return dict(kind="generic_frozen_features",fitted=False, dimension=2048,
                    feature_sha256=self.manifest["features_sha256"],index_sha256=self.manifest["index_sha256"])


def source_labels(protocol, uids, *, role="train"):
    """The called legacy reader predicates per-dataset Parquet and memmaps only scope.

    For dev, use the legacy development reader after writing a committed dummy
    input-only action set; dev outcomes never become ordinary training labels.
    """
    if role != "train":
        raise ValueError("this materializer only permits official source train")
    from src.subject_routing.profiling import materialize_scoped_training_labels
    from src.subject_routing.leakage_guard import Operation, UIDAccessAudit
    from scripts.subject_routing_v2.run_b00_query_teacher import _common_label_arguments
    splits,pool=manifests()
    allowed=role_uids(protocol,"train")
    if not set(uids) <= set(allowed): raise ValueError("source scope violation")
    args=_common_label_arguments(experts=pool["canonical_order"],model_pool=pool,split_manifest=splits,
        matrix_audit=read_json(ROOT/"artifacts/subject_routing/audits/P02_matrix_semantics.json"))
    labels, provenance=materialize_scoped_training_labels(uids,allowed_train_uids=allowed,
        audit=UIDAccessAudit.create(Operation.TRAIN,uids),**args)
    return np.asarray(labels.matrix.y,dtype=np.float32),np.asarray(labels.matrix.c,dtype=np.float64),provenance.to_dict()


def prepare_source():
    from src.subject_routing.final_evaluation import load_scoped_grouping_records
    cache=FeatureCache()
    splits,pool=manifests()
    paths=[]
    for protocol in ("full14","heldout11"):
        uids=role_uids(protocol,"train")
        datasets=sorted(set(uid.split("/",1)[0] for uid in uids))
        # All inner folds, selection and independent calibration are inside
        # official train. Official dev remains available only for final R3
        # checkpoint selection using its original algorithm.
        _, group_audit=load_scoped_grouping_records(ROOT,uids,
            uid_to_dataset={uid:uid.split("/",1)[0] for uid in uids},allowed_datasets=datasets)
        groups=np.array([group_audit["uid_to_group"][uid] for uid in uids])
        # 15% groups for independent calibration/certification, kept out of
        # every head, cost statistic, clustering and source model selection.
        buckets=np.array([int(canonical({"protocol":protocol,"group":str(g),"salt":"calibration20260913"})[:8],16)%20 for g in groups])
        roles=np.where(buckets==0,"cert",np.where(buckets<=2,"qcal","fit"))
        # Remaining fit groups are assigned to three disjoint folds.
        fold=np.array([int(canonical({"protocol":protocol,"group":str(g),"salt":"oof20260913"})[:8],16)%3 for g in groups])
        y,c,prov=source_labels(protocol,uids)
        x,available=cache.rows(uids)
        d=OUT/"source"/protocol
        d.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(d/"labels.npz",uids=np.array(uids),y=y,c=c)
        np.save(d/"features.npy",x,allow_pickle=False)
        np.savez_compressed(d/"input_metadata.npz",uids=np.array(uids),groups=groups,role=roles,fold=fold,
                            available=available,dataset=np.array([uid.split("/",1)[0] for uid in uids]))
        foldrows=[]
        for k in range(3):
            train=(roles=="fit")&(fold!=k)
            valid=(roles=="fit")&(fold==k)
            assert not set(groups[train])&set(groups[valid])
            foldrows.append(dict(fold=k,fit_uids=np.array(uids)[train].tolist(),evaluation_uids=np.array(uids)[valid].tolist(),
                                 train_n=int(train.sum()),evaluation_n=int(valid.sum())))
        write_json(d/"FOLDS.json",dict(protocol=protocol,folds=foldrows,qcal_uids=np.array(uids)[roles=="qcal"].tolist(),
            cert_uids=np.array(uids)[roles=="cert"].tolist(),official_split="train only",group_audit=group_audit))
        manifest=dict(protocol=protocol,kind="source_labels",fitted=True,fit_uids=uids,ancestors=[cache.binding()],
            y_binary_complete=True,all_wrong_retained=True,source_label_provenance=prov,
            files=[record(d/name) for name in ("labels.npz","features.npy","input_metadata.npz","FOLDS.json")],
            official_train_n=len(uids),split_sha256=digest(SPLIT_PATH),expert_order=pool["canonical_order"])
        write_json(d/"MANIFEST.json",manifest)
        paths.extend([d/"MANIFEST.json",d/"FOLDS.json"])
        print(json.dumps(dict(protocol=protocol,n=len(uids),roles={r:int((roles==r).sum()) for r in ("fit","qcal","cert")},missing_features=int((~available).sum())),ensure_ascii=False),flush=True)
    return paths


def load_source(protocol):
    d=OUT/"source"/protocol
    m=read_json(d/"MANIFEST.json")
    from .protocol import validate_ancestry
    validate_ancestry(m,protocol,set(role_uids(protocol,"train")))
    for item in m["files"]:
        if digest(ROOT/item["path"]) != item["sha256"]: raise ValueError("source artifact changed")
    labels=np.load(d/"labels.npz",allow_pickle=False)
    meta=np.load(d/"input_metadata.npz",allow_pickle=False)
    if not np.array_equal(labels["uids"],meta["uids"]): raise ValueError("source alignment")
    return dict(x=np.load(d/"features.npy",mmap_mode="r"),y=labels["y"],c=labels["c"],**{k:meta[k] for k in meta.files})
