"""Leakage-aware subject-routing V2 research utilities.

The V2 namespace is intentionally separate from the frozen
``artifacts/subject_routing`` experiment.  New code may read the historical
artifacts for audit purposes, but it must never mutate them.
"""

from ._io import canonical_sha256, load_json, sha256_file

__all__ = ["canonical_sha256", "load_json", "sha256_file"]
