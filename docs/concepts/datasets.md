# Repository-backed datasets

`generator.datasets` is a `StudyDatasets` factory. Each call creates a
short-lived PyTorch `Dataset` view over the records of one entity type. The
views read records from the study database and arrays from the artifact store;
they do not scan folders or align files by basename. A view lists its records
when it is created, so create a new view after a pipeline stage has changed
the study.

| Factory method | Records | Filters |
| --- | --- | --- |
| `original_samples()` | ingested originals | `has_anomaly`, `is_annotated` |
| `real_anomalies()` | extracted real anomalies | – |
| `synthetic_anomalies()` | synthetic variants | – |
| `hybrid_samples()` | materialized hybrids | – |

## Reading samples

Every item is a dictionary. `return_artifacts` selects its keys:

```python
hybrids = generator.datasets.hybrid_samples(
    return_artifacts=("img", "segmentation", "hybrid_sample_id"),
    numpy_mode=True,
)

for sample in hybrids:
    image = sample["img"]                # (C, H, W) or (C, D, H, W)
    mask = sample["segmentation"]
    hybrid_id = sample["hybrid_sample_id"]
```

| View | Available keys | Default keys |
| --- | --- | --- |
| `original_samples()` | `img`, `ori_mask`, `fname`, `original_sample_id`, `has_anomaly`, `is_annotated`, `metadata`, `record` | `img`, `ori_mask`, `fname`, `original_sample_id` |
| `real_anomalies()` | `img`, `ori_mask`, `anomaly_roi`, `anomaly_roi_mask`, `anomaly_meta`, `fname`, `real_anomaly_id`, `record` | `img`, `fname` |
| `synthetic_anomalies()` | `synth_anomaly`, `tgt_mask`, `anomaly_roi`, `anomaly_roi_mask`, `anomaly_meta`, `fname`, `synthetic_anomaly_id`, `real_anomaly_id`, `record` | all except `record` |
| `hybrid_samples()` | `img`, `segmentation`, `fname`, `hybrid_sample_id`, `record` | `img`, `segmentation`, `hybrid_sample_id` |

`record` is the immutable [record](study-storage.md) of the item. For
originals, `fname` is the `source_name`; for all other views it is the record
ID. `ori_mask` of an unannotated original is `None`. In the synthetic view,
`anomaly_roi`, `anomaly_roi_mask` and `anomaly_meta` belong to the real parent.

`load_sample_by_id(record_id)` returns the item of one record.

## Options

All views accept the same loading options:

| Option | Default | Effect |
| --- | --- | --- |
| `numpy_mode` | `False` | `True` returns NumPy arrays as stored; `False` returns tensors of `dtype`. |
| `dtype` | `torch.float32` | tensor dtype when `numpy_mode` is `False`. |
| `load_to_ram` | `False` | `True` loads all selected arrays when the view is created; `False` reads each array on access. |
| `transform` | `None` | callable applied to `img`. |

## Filtering originals and hybrids

```python
controls = generator.datasets.original_samples(has_anomaly=False)
annotated_controls = generator.datasets.original_samples(
    has_anomaly=False, is_annotated=True
)
```

`hybrid_samples()` returns only materialized hybrids (status `generated`).
Planned or failed hybrids have no images, so reading them through the view
raises an error; list their records through the repository instead:

```python
failed = generator.repository.list_hybrid_samples(status="failed")
for hybrid in failed:
    print(hybrid.id, hybrid.error)
```

The view does not apply
[postprocessing](../guides/postprocessing.md#using-decisions-downstream)
decisions.

## Without a generator

The views only need the repository and the artifact store, for example in a
separate training process:

```python
from hybrid_sample_generator.configuration import load_config_file
from hybrid_sample_generator.datasets.study_datasets import StudyDatasets
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.study_repository import StudyRepository

config = load_config_file("results/study-01/configuration.json")
paths = config.study.paths
datasets = StudyDatasets(
    StudyRepository(paths.artifact_database),
    ArtifactStore(paths.study_folder),
)
hybrids = datasets.hybrid_samples()
```
