"""Query-only contracts and revision-aware purchased observations.

The evaluator runs in a separate process. These types are API checks, not an OS
security sandbox; the prediction worker additionally denies outcome file reads.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Mapping
import numpy as np
from .runtime import canonical


class Schema(str, Enum):
    FIXED_17 = "FIXED_17"
    PROBE_AUGMENTED = "PROBE_AUGMENTED"
    EXPANDED_POOL = "EXPANDED_POOL"
    TOOL_AUGMENTED = "TOOL_AUGMENTED"


@dataclass(frozen=True)
class QueryInput:
    query_id: str
    features: np.ndarray
    available: bool

    @classmethod
    def from_mapping(cls, value: Mapping):
        allowed = {"query_id", "features", "available"}
        if set(value) != allowed:
            raise ValueError(f"policy schema forbids extra/missing fields: {set(value) ^ allowed}")
        x = np.asarray(value["features"], dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("features must be a finite numeric vector")
        if not isinstance(value["available"], bool):
            raise ValueError("available must be boolean")
        x = x.copy()
        x.flags.writeable = False
        return cls(str(value["query_id"]), x, value["available"])


@dataclass(frozen=True)
class Action:
    query_id: str
    model: str
    revision: str
    prompt_sha256: str
    image_representation_sha256: str
    decoding_sha256: str
    schema: str = Schema.FIXED_17.value

    def __post_init__(self):
        Schema(self.schema)
        if not all(isinstance(v, str) and v for v in asdict(self).values()):
            raise ValueError("all action identity fields must be nonempty")

    @property
    def key(self):
        return canonical(asdict(self))


class PurchaseLedger:
    def __init__(self):
        self.committed = set()
        self.purchased = {}
        self.legacy_usd_proxy = 0.0
        self.extra_compute = []
        self.build_compute = []

    def commit(self, action: Action):
        self.committed.add(action.key)
        return action.key

    def accept(self, action: Action, *, observed_action: Action, observation,
               legacy_usd_proxy: float | None, cpu_seconds=0.0, gpu_seconds=0.0):
        if action.key not in self.committed:
            raise ValueError("commit required before opening evaluator observation")
        if action != observed_action:
            raise ValueError("new view/prompt/weights cannot reuse old outcomes")
        if action.key in self.purchased:
            return self.purchased[action.key]
        if legacy_usd_proxy is not None:
            if not np.isfinite(legacy_usd_proxy) or legacy_usd_proxy < 0:
                raise ValueError("invalid expert charge")
            self.legacy_usd_proxy += legacy_usd_proxy
        if any(not np.isfinite(v) or v < 0 for v in (cpu_seconds, gpu_seconds)):
            raise ValueError("invalid measured compute")
        self.extra_compute.append(dict(action=action.key, cpu_seconds=cpu_seconds,
                                       gpu_seconds=gpu_seconds, dollar_conversion=None))
        self.purchased[action.key] = observation
        return observation

    def observation(self, action: Action):
        if action.key not in self.purchased:
            raise ValueError("unpaid observation")
        return self.purchased[action.key]


def validate_ancestry(manifest: Mapping, protocol: str, allowed_fit_uids: set[str]):
    """Recursively validate every fitted ancestor; frozen generic features can share."""
    if manifest.get("fitted", True):
        if manifest.get("protocol") != protocol:
            raise ValueError("cross-protocol fitted ancestor")
        fit = manifest.get("fit_uids")
        if not isinstance(fit, list) or not set(fit) <= allowed_fit_uids:
            raise ValueError("fit ancestry outside allowed source scope")
    elif manifest.get("kind") not in {"generic_frozen_weights", "generic_frozen_features"}:
        raise ValueError("unfitted sharing requires a generic frozen input")
    for parent in manifest.get("ancestors", []):
        validate_ancestry(parent, protocol, allowed_fit_uids)


def decide(p, cost_hat, *, lambda_cost, kappa, available=None, fallback=0):
    from routers.bench_coe_query.cost_policies import utility_policy
    actions = utility_policy(p, cost_hat, lambda_cost=float(lambda_cost), kappa=float(kappa))
    if available is not None:
        actions[~np.asarray(available, dtype=bool)] = int(fallback)
    return actions

