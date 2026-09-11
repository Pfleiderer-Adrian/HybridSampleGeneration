"""Shared fake components and sample fixtures for pipeline tests."""

import numpy as np

from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.fusion.interfaces import FusionOutput


class _FakeGenerator:
    def generate(self, sample, **_kwargs):
        image = np.asarray(sample["img"], dtype=np.float32)
        noise = np.random.normal(0.0, 0.01, image.shape).astype(np.float32)
        return image + noise, np.asarray(sample["ori_mask"], dtype=np.uint8)


class _FakeFusionBackend:
    def warmup(self, *_args, **_kwargs):
        return self

    def fuse(self, sample, control_img, position, **_kwargs):
        image = np.asarray(control_img).copy()
        segmentation = np.zeros_like(image, dtype=np.uint8)
        spatial_shape = image.shape[1:]
        center = [
            min(max(int(round(value * size)), 0), size - 1)
            for value, size in zip(position, spatial_shape)
        ]
        slices = tuple(slice(max(value - 1, 0), min(value + 1, size)) for value, size in zip(center, spatial_shape))
        image[(slice(None), *slices)] += 0.5
        segmentation[(slice(None), *slices)] = 1
        return FusionOutput(
            image=image,
            segmentation=segmentation,
            roi=np.asarray(sample["synth_anomaly"]),
            roi_mask=np.asarray(sample["tgt_mask"]),
        )


def _anomaly_samples():
    samples = []
    for sample_index in range(2):
        image = np.zeros((1, 32, 32), dtype=np.float32)
        segmentation = np.zeros_like(image, dtype=np.uint8)
        first = 4 + sample_index
        second = 22 - sample_index
        image[:, first:first + 4, first:first + 4] = 0.4 + sample_index * 0.1
        image[:, second:second + 4, second:second + 4] = 0.8
        segmentation[:, first:first + 4, first:first + 4] = 1
        segmentation[:, second:second + 4, second:second + 4] = 1
        samples.append(InputSample(image, segmentation, f"anomaly-{sample_index}"))
    return samples


def _control_samples():
    return [
        InputSample(
            np.full((1, 32, 32), 0.2 + index * 0.1, dtype=np.float32),
            np.zeros((1, 32, 32), dtype=np.uint8),
            f"control-{index}",
        )
        for index in range(2)
    ]
