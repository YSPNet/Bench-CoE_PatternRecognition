"""Deterministic group-aware inner cross-fitting over frozen train UIDs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class FoldIntegrityError(ValueError):
    """Raised when an inner fold violates train, group, or role honesty."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _UnionFind:
    def __init__(self, values: Sequence[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


@dataclass(frozen=True, slots=True)
class PublicGroupingRecord:
    uid: str
    dataset: str
    structural_asset_digest: str
    exact_prompt_sha256: str


def connected_groups(
    ordered_uids: Sequence[str], records: Mapping[str, PublicGroupingRecord]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    ids = tuple(str(uid) for uid in ordered_uids)
    if len(ids) != len(set(ids)) or set(ids) != set(records):
        raise FoldIntegrityError("group records must exactly cover unique ordered train UIDs")
    union = _UnionFind(ids)
    first_by_key: dict[tuple[str, str], str] = {}
    for uid in ids:
        record = records[uid]
        for kind, digest in (
            ("asset", record.structural_asset_digest),
            ("prompt", record.exact_prompt_sha256),
        ):
            key = (kind, digest)
            previous = first_by_key.setdefault(key, uid)
            union.union(previous, uid)
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for uid in ids:
        members_by_root[union.find(uid)].append(uid)
    uid_to_group: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for members in members_by_root.values():
        # members were appended while traversing `ids`, so official order is
        # already preserved without an O(N x groups) rescan.
        ordered_members = members
        group_id = f"grp-{canonical_sha256(ordered_members)[:24]}"
        groups[group_id] = ordered_members
        uid_to_group.update({uid: group_id for uid in ordered_members})
    return uid_to_group, dict(sorted(groups.items()))


def assign_groups(
    groups: Mapping[str, Sequence[str]],
    records: Mapping[str, PublicGroupingRecord],
    *,
    fold_count: int,
    seed: int,
) -> dict[str, int]:
    if fold_count < 3:
        raise FoldIntegrityError("at least three folds are required")
    datasets = sorted({record.dataset for record in records.values()})
    total_by_dataset = Counter(record.dataset for record in records.values())
    target_by_dataset = {
        dataset: total_by_dataset[dataset] / fold_count for dataset in datasets
    }
    target_total = len(records) / fold_count
    fold_dataset = [Counter() for _ in range(fold_count)]
    fold_total = [0] * fold_count
    group_counts = {
        group_id: Counter(records[uid].dataset for uid in members)
        for group_id, members in groups.items()
    }
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (-len(groups[group_id]), canonical_sha256([seed, group_id])),
    )
    assignment: dict[str, int] = {}
    for group_id in ordered_groups:
        counts = group_counts[group_id]
        size = len(groups[group_id])
        tie_order = sorted(
            range(fold_count),
            key=lambda fold: canonical_sha256([seed, group_id, fold]),
        )

        def objective(fold: int) -> tuple[float, int]:
            dataset_penalty = sum(
                (
                    (fold_dataset[fold][dataset] + counts[dataset])
                    / max(target_by_dataset[dataset], 1.0)
                )
                ** 2
                for dataset in datasets
                if counts[dataset]
            )
            total_penalty = ((fold_total[fold] + size) / max(target_total, 1.0)) ** 2
            return dataset_penalty + 0.2 * total_penalty, tie_order.index(fold)

        selected = min(range(fold_count), key=objective)
        assignment[group_id] = selected
        fold_total[selected] += size
        fold_dataset[selected].update(counts)
    if set(assignment) != set(groups):
        raise FoldIntegrityError("not all groups received a fold")
    return assignment


def build_rotations(
    ordered_uids: Sequence[str],
    uid_to_group: Mapping[str, str],
    group_to_fold: Mapping[str, int],
    *,
    fold_count: int,
) -> list[dict[str, Any]]:
    ids = tuple(str(uid) for uid in ordered_uids)
    rotations = []
    for rotation in range(fold_count):
        check_fold = rotation
        profile_fold = (rotation + 1) % fold_count
        role_ids = {
            "discovery_uids": [
                uid
                for uid in ids
                if group_to_fold[uid_to_group[uid]] not in {check_fold, profile_fold}
            ],
            "profile_uids": [
                uid for uid in ids if group_to_fold[uid_to_group[uid]] == profile_fold
            ],
            "check_uids": [
                uid for uid in ids if group_to_fold[uid_to_group[uid]] == check_fold
            ],
        }
        role_sets = {name: set(values) for name, values in role_ids.items()}
        if any(
            role_sets[left].intersection(role_sets[right])
            for left, right in (
                ("discovery_uids", "profile_uids"),
                ("discovery_uids", "check_uids"),
                ("profile_uids", "check_uids"),
            )
        ):
            raise FoldIntegrityError("inner roles overlap")
        if set().union(*role_sets.values()) != set(ids):
            raise FoldIntegrityError("inner roles do not cover train universe")
        scopes = {
            "allowed_feature_fit_uids": role_ids["discovery_uids"],
            "allowed_label_fit_uids": role_ids["discovery_uids"],
            "pca_fit_uids": role_ids["discovery_uids"],
            "teacher_sampling_uids": role_ids["discovery_uids"],
            "profile_uids": role_ids["profile_uids"],
            "capability_table_fit_uids": role_ids["profile_uids"],
            "check_uids": role_ids["check_uids"],
            "pseudo_label_recipient_uids": role_ids["check_uids"],
        }
        rotations.append(
            {
                "rotation": rotation,
                "check_fold": check_fold,
                "profile_fold": profile_fold,
                "discovery_folds": [
                    fold
                    for fold in range(fold_count)
                    if fold not in {check_fold, profile_fold}
                ],
                "roles": {
                    name: {
                        "count": len(values),
                        "ordered_uid_sha256": canonical_sha256(values),
                        "ordered_uids": values,
                    }
                    for name, values in role_ids.items()
                },
                "component_scopes": {
                    name: {
                        "count": len(values),
                        "ordered_uid_sha256": canonical_sha256(values),
                        "ordered_uids": values,
                    }
                    for name, values in scopes.items()
                },
                "actual_ratios": {
                    name.removesuffix("_uids"): len(values) / len(ids)
                    for name, values in role_ids.items()
                },
            }
        )
    return rotations


def validate_rotations(
    ordered_uids: Sequence[str],
    uid_to_group: Mapping[str, str],
    group_to_fold: Mapping[str, int],
    rotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ids = tuple(str(uid) for uid in ordered_uids)
    check_frequency = Counter()
    profile_frequency = Counter()
    discovery_frequency = Counter()
    group_members: dict[str, set[str]] = defaultdict(set)
    for uid in ids:
        group_members[uid_to_group[uid]].add(uid)
    for rotation in rotations:
        roles = rotation["roles"]
        discovery = set(roles["discovery_uids"]["ordered_uids"])
        profile = set(roles["profile_uids"]["ordered_uids"])
        check = set(roles["check_uids"]["ordered_uids"])
        if discovery & profile or discovery & check or profile & check:
            raise FoldIntegrityError("rotation roles overlap")
        if discovery | profile | check != set(ids):
            raise FoldIntegrityError("rotation roles do not cover train")
        check_frequency.update(check)
        profile_frequency.update(profile)
        discovery_frequency.update(discovery)
        for group_id in {uid_to_group[uid] for uid in check}:
            if not group_members[group_id].issubset(check):
                raise FoldIntegrityError("dependency group crosses a role boundary")
        scopes = rotation["component_scopes"]
        if set(scopes["allowed_label_fit_uids"]["ordered_uids"]) & check:
            raise FoldIntegrityError("predicted check sample entered supervised partition fit")
        if set(scopes["capability_table_fit_uids"]["ordered_uids"]) & check:
            raise FoldIntegrityError("predicted check sample entered capability fit")
    if set(check_frequency.values()) != {1} or set(profile_frequency.values()) != {1}:
        raise FoldIntegrityError("every train UID must be check/profile exactly once")
    if set(discovery_frequency.values()) != {len(rotations) - 2}:
        raise FoldIntegrityError("discovery frequency differs from rotation design")
    fold_sizes = Counter(group_to_fold[uid_to_group[uid]] for uid in ids)
    return {
        "passed": True,
        "train_uids": len(ids),
        "groups": len(set(uid_to_group.values())),
        "fold_sizes": {str(fold): fold_sizes[fold] for fold in sorted(fold_sizes)},
        "every_uid_check_once": True,
        "every_uid_profile_once": True,
        "every_uid_discovery_count": len(rotations) - 2,
        "groups_never_split": True,
        "check_excluded_from_upstream_supervised_fit": True,
    }
