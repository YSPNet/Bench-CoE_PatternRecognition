"""Deterministic row-modulo feature sharding helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int


def context_from_environment(world_size_override: int | None = None) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    detected_world = int(os.environ.get("WORLD_SIZE", "1"))
    world_size = detected_world if world_size_override is None else int(world_size_override)
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid distributed context rank={rank}, world_size={world_size}")
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank)


def shard_rows(rows: Iterable[int], *, rank: int, world_size: int) -> tuple[int, ...]:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    return tuple(int(row) for row in rows if int(row) % world_size == rank)


def validate_exact_sharding(rows: Iterable[int], world_size: int) -> bool:
    expected = tuple(int(row) for row in rows)
    observed = tuple(
        sorted(row for rank in range(world_size) for row in shard_rows(expected, rank=rank, world_size=world_size))
    )
    return observed == tuple(sorted(expected)) and len(observed) == len(set(observed))
