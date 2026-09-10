"""Dataset adapter for the MVTec AD 2 example integration."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from hybrid_sample_generator.domain.input_sample import InputSample
from examples.image_2d.image_dataloader import ensure_chw
from examples.image_2d.image_dataloader import _load_image_array as load_image_array
from examples.mvtec_ad2.records import MVTecAD2Sample


class MVTecAD2Dataloader:
    """
    Iterator adapter for MVTec AD 2 samples.

    The HybridSampleGeneration framework expects tuples of
    (image_array, segmentation_array, basename). Positive MVTec AD 2 samples
    provide real masks. Good/negative control samples do not have masks, so this
    adapter creates zero masks for them.
    """

    def __init__(self, samples: Sequence[MVTecAD2Sample]) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("MVTecAD2Dataloader requires at least one sample.")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        for sample in self.samples:
            img = ensure_chw(load_image_array(str(sample.image_path))).astype(np.float32, copy=False)

            if sample.mask_path is None:
                seg = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
            else:
                seg = ensure_chw(load_image_array(str(sample.mask_path))).astype(np.float32, copy=False)
                if seg.shape[0] > 1:
                    seg = seg[:1]
                seg = np.where(seg > 0, 1.0, 0.0).astype(np.float32, copy=False)

            yield img, seg, sample.sample_id

    def iter_input_samples(self) -> Iterator[InputSample]:
        for sample in self.samples:
            img = ensure_chw(load_image_array(str(sample.image_path))).astype(
                np.float32, copy=False
            )
            if sample.mask_path is None:
                seg = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
            else:
                seg = ensure_chw(load_image_array(str(sample.mask_path))).astype(
                    np.float32, copy=False
                )
                seg = np.where(seg[:1] > 0, 1.0, 0.0).astype(np.float32, copy=False)
            yield InputSample(
                image=img,
                segmentation=seg,
                source_name=sample.sample_id,
                source_image_path=str(sample.image_path),
                source_segmentation_path=(
                    None if sample.mask_path is None else str(sample.mask_path)
                ),
                metadata={"split": sample.split, "label": sample.label},
            )
