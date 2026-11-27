# Quick start

This script runs the complete pipeline on a small generated dataset: gray noise
images, half of them with a bright blob as defect. It needs no data files and
finishes within a minute on a GPU; expect a few minutes on a CPU.

```python
import numpy as np

from hybrid_sample_generator.configuration import Configuration
from hybrid_sample_generator.evaluation.service import evaluate_study
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator


def toy_samples(count=40, size=96, seed=0):
    """Yield (image, segmentation, source_name); every second sample has a defect."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    for index in range(count):
        image = 0.5 + 0.05 * rng.standard_normal((1, size, size))
        mask = np.zeros((1, size, size), dtype=np.uint8)
        if index % 2 == 0:
            cy, cx = rng.integers(24, size - 24, size=2)
            radius = rng.integers(6, 12)
            blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            image[0][blob] += 0.3
            mask[0][blob] = 1
        yield image.astype(np.float32), mask, f"sample-{index:03d}"


config = Configuration("quick-start", study_folder="results/quick-start")
config.extraction.anomaly_size = (1, 32, 32)
config.model.set_model("cVAE_ConvNeXt_2D")
config.training.num_trials = 1
config.training.epochs = 20
config.generation.variants_per_real_anomaly = 3
config.matching.hybrids_per_original = 2

generator = HybridDataGenerator(config)
summary = generator.ingest_dataset(toy_samples())
generator.extract_anomalies()
generator.train_generator()
generator.generate_synthetic_anomalies()
generator.plan_hybrid_samples()
generator.materialize_hybrid_samples()
config.save_config_file()

hybrids = generator.datasets.hybrid_samples(numpy_mode=True)
print(len(hybrids), hybrids[0]["img"].shape, hybrids[0]["segmentation"].shape)

evaluate_study(config)
```

The script prints the progress of every stage and ends with the number and
shape of the hybrids, `40 (1, 96, 96) (1, 96, 96)`, followed by the evaluation
summary. The study folder `results/quick-start/` then contains the
configuration, the database, the trained model and all arrays; see
[data model and storage](../concepts/study-storage.md).

## What happened

1. `ingest_dataset()` stored all 40 originals: 20 anomalous ones and 20
   controls with an empty mask.
2. `extract_anomalies()` cut out the 20 defects as real anomalies.
3. `train_generator()` trained a mask-conditioned VAE on them in one Optuna
   trial and loaded it.
4. `generate_synthetic_anomalies()` created three variants per real anomaly.
5. `plan_hybrid_samples()` chose, for each of the two hybrids per control, a
   synthetic anomaly and a position by template matching.
6. `materialize_hybrid_samples()` fused the anomalies into the controls and
   wrote the hybrid images and masks.
7. `evaluate_study()` compared every synthetic anomaly with its real parent.

Each stage is described in the [pipeline overview](../guides/pipeline.md).
The values here are chosen for speed; the defaults train for up to 1000 epochs
in 10 trials.

## Inspect the study

Open the [study browser](../guides/visualization.md) for the finished study:

```bash
python -m hybrid_sample_generator.visualization results/quick-start
```

## Use your own data

Replace `toy_samples()` with any iterable that yields
`(image, segmentation, source_name)` with channel-first arrays, and set
`config.extraction.anomaly_size` to the channel count and cutout size of your
data. [Input data](../guides/input-data.md) describes the requirements.

The repository contains loaders for image folders and NIfTI volumes. Install
their dependencies with `python -m pip install -e ".[examples]"`, set the data
paths in the example's `main.py` and run it from the repository root:

```bash
python -m examples.image_2d.main
python -m examples.nifti_3d.main
```

The [MVTec AD 2 benchmark](../mvtec-ad2/index.md) is a complete application
with grouped data splits, several synthetic anomaly sources and a downstream
segmentation model:

```bash
python -m examples.mvtec_ad2.run --category can
```

## Next steps

- [Continue or repeat](../guides/continuing-studies.md) single stages of a
  study.
- Tune [training](../guides/training.md),
  [hybrid planning](../guides/matching.md) and [fusion](../guides/fusion.md).
- Look up parameters in the [configuration reference](../configuration/index.md).
