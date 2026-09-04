from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class InputSample:
    """Typed boundary record accepted by extraction and hybrid planning."""

    image: np.ndarray
    segmentation: np.ndarray
    source_name: str
    source_image_path: str | None = None
    source_segmentation_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def coerce_input_sample(value) -> InputSample:
    if isinstance(value, InputSample):
        return value
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        raise TypeError(
            "A dataloader item must be InputSample or (image, segmentation, source_name[, paths...])."
        )

    image, segmentation, source_name, *extra = value
    source_image_path = None
    source_segmentation_path = None
    if len(extra) >= 2 and all(isinstance(item, (str, os.PathLike)) for item in extra[:2]):
        source_image_path = os.fspath(extra[0])
        source_segmentation_path = os.fspath(extra[1])

    return InputSample(
        image=np.asarray(image),
        segmentation=np.asarray(segmentation),
        source_name=str(source_name),
        source_image_path=source_image_path,
        source_segmentation_path=source_segmentation_path,
    )


def iter_input_samples(dataloader: Iterable) -> Iterator[InputSample]:
    typed_iterator = getattr(dataloader, "iter_input_samples", None)
    if callable(typed_iterator):
        yield from typed_iterator()
        return
    for item in dataloader:
        yield coerce_input_sample(item)
