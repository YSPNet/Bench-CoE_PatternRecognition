from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


METHODS = (
    "oracle", "strongest", "cheapest", "knn", "prknn", "ovr", "kmeans",
    "linear", "mlp", "cosinecls", "routerdc", "zooter", "vlc",
)
FEATURE_METHODS = frozenset(("knn", "prknn", "ovr", "kmeans", "linear", "mlp"))
END_TO_END = frozenset(("cosinecls", "routerdc", "zooter", "vlc"))
STOCHASTIC = frozenset(("linear", "mlp", "cosinecls", "routerdc", "zooter", "vlc"))
HELD_OUT = ("MMMU_DEV_VAL", "MathVision_MINI", "ChartQA_TEST")
ROLES = ("train_seen", "dev_seen", "id_test", "ood_test")
DEFAULTS = {
    "schema_version": 1,
    "protocol_manifest": "outputs/bench_coe_query/ood/protocol/HELD_OUT_PROTOCOL.json",
    "output_dir": "outputs/table_a2",
    "methods": list(METHODS),
    "seeds": [0, 1, 2, 3, 4],
    "lambdas": [0, 10, 100, 1000, 10000, "inf"],
    "features": {
        "root": "EMBEDDINGS", "text_encoder": "BAAI/bge-m3",
        "vision_encoder": "facebook/dinov2-base", "fusion_method": "normalize_concat",
        "text_weight": 0.5, "batch_size": 32,
        "multi_image_pooling": "mean",
    },
    "training": {
        "max_epochs": 5, "learning_rate": 2e-5, "batch_size": 16,
        "weight_decay": 0.01, "patience": 2,
    },
    "runtime": {"device": "cuda", "offline": True, "verbose": 1},
    "method_parameters": {
        "knn": {"n_neighbors": 5}, "prknn": {"n_neighbors": 7},
        "ovr": {}, "kmeans": {}, "linear": {},
        "mlp": {"hidden_sizes": [2048, 1024]},
        "cosinecls": {"lambda_sample_sample": 0.0}, "routerdc": {},
        "zooter": {"max_length": 128},
        "vlc": {"model_type": "lxmert"},
    },
    "rank_score_beta": 0.1,
}


class ExperimentError(ValueError):
    pass


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        result = copy.deepcopy(DEFAULTS)
    else:
        with Path(path).open(encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle)
        if not isinstance(supplied, dict):
            raise ExperimentError("configuration must be a mapping")
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            raise ExperimentError(f"unknown configuration keys: {sorted(unknown)}")
        result = merge(DEFAULTS, supplied)
    if result["schema_version"] != 1:
        raise ExperimentError("unsupported configuration schema")
    if (not result["methods"] or len(set(result["methods"])) != len(result["methods"])
            or set(result["methods"]) - set(METHODS)):
        raise ExperimentError("methods must be unique canonical method names")
    seeds = result["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ExperimentError("seeds must be unique nonnegative integers")
    normalized = [lambda_tag(x) for x in result["lambdas"]]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ExperimentError("lambdas must be nonempty and unique")
    result["lambdas"] = normalized
    if result["runtime"]["device"] not in ("cpu", "cuda"):
        raise ExperimentError("device must be cpu or cuda")
    if type(result["runtime"]["offline"]) is not bool:
        raise ExperimentError("runtime.offline must be a boolean")
    for key in ("max_epochs", "batch_size", "patience"):
        if type(result["training"][key]) is not int or result["training"][key] <= 0:
            raise ExperimentError(f"training.{key} must be a positive integer")
    if result["training"]["learning_rate"] <= 0 or result["training"]["weight_decay"] < 0:
        raise ExperimentError("invalid learning rate or weight decay")
    feature = result["features"]
    if type(feature["batch_size"]) is not int or feature["batch_size"] <= 0:
        raise ExperimentError("features.batch_size must be a positive integer")
    if feature["fusion_method"] not in ("concat", "average", "weighted_average", "normalize_concat", "only_text", "only_image"):
        raise ExperimentError("unknown feature fusion method")
    if not 0 <= feature["text_weight"] <= 1:
        raise ExperimentError("features.text_weight must be between 0 and 1")
    if feature["multi_image_pooling"] != "mean":
        raise ExperimentError("multi-image features use the native mean-and-normalize pooling")
    for key in ("root", "text_encoder", "vision_encoder"):
        if not isinstance(feature[key], str) or not feature[key]:
            raise ExperimentError(f"features.{key} must be a nonempty string")
    if result["rank_score_beta"] < 0:
        raise ExperimentError("rank_score_beta must be nonnegative")
    return result


def lambda_value(value) -> float:
    import math
    parsed = float(value)
    if math.isnan(parsed) or parsed < 0:
        raise ExperimentError("lambda must be nonnegative, or inf")
    return parsed


def lambda_tag(value) -> str:
    import math
    parsed = lambda_value(value)
    return "inf" if math.isinf(parsed) else format(parsed, ".12g")


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()
