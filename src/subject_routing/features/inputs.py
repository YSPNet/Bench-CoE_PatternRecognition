"""Explicit failure accounting for text/image feature inputs."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Sequence

from PIL import Image, UnidentifiedImageError

from ..data_contracts import InputRecord


@dataclass(frozen=True, slots=True)
class FeatureInputFailure:
    row: int
    code: str
    detail: str


def validate_modality_inputs(
    records: Sequence[InputRecord],
    *,
    require_text: bool,
    require_vision: bool,
    max_text_characters: int,
    image_policy: str = "exactly_one",
) -> tuple[FeatureInputFailure, ...]:
    """Validate every row and return failures; callers may not drop them."""

    failures: list[FeatureInputFailure] = []
    for row, record in enumerate(records):
        if require_text and not record.question.strip():
            failures.append(FeatureInputFailure(row, "empty_text", "question is empty"))
        elif require_text and len(record.question) > max_text_characters:
            failures.append(
                FeatureInputFailure(
                    row,
                    "overlong_text",
                    f"deterministic tokenizer truncation required above {max_text_characters} characters",
                )
            )
        if not require_vision:
            continue
        if not record.images:
            failures.append(FeatureInputFailure(row, "missing_image", "no image bytes"))
            continue
        if image_policy == "exactly_one" and len(record.images) != 1:
            failures.append(
                FeatureInputFailure(
                    row,
                    "multi_image",
                    f"strict adapter received {len(record.images)} ordered images",
                )
            )
            continue
        for image_index, payload in enumerate(record.images):
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                failures.append(
                    FeatureInputFailure(
                        row,
                        "corrupt_image",
                        f"ordered image {image_index} failed decode: {type(exc).__name__}",
                    )
                )
                break
    return tuple(failures)
