# Input data

The pipeline accepts channel-first arrays:

- 2D: `(C, H, W)`
- 3D: `(C, D, H, W)`

A single dataloader yields all originals: annotated anomaly sources and normal
controls. Controls use an empty segmentation; unannotated samples may use
`None`. A dataloader may yield the compact tuple
`(image, segmentation, source_name)`.
For unambiguous source identity and provenance, yield `InputSample` records or
implement `iter_input_samples()`:

```python
from hybrid_sample_generator.domain.input_sample import InputSample

yield InputSample(
    image=image,
    segmentation=mask,
    source_name="sample-001",
    source_image_path="/dataset/images/sample-001.png",
    source_segmentation_path="/dataset/masks/sample-001.png",
)
```

Each `source_name` must be unique within an import, even when samples have
different `source_image_path` values. Resolved source identities must also be
unique. A positive segmentation marks an anomalous original; an empty mask marks
an annotated control, and `None` marks an unannotated control.

An annotated mask must have the same spatial shape as its image and either one
channel or the same channel count as the image. The spatial dimensions in
`config.extraction.anomaly_size` must match the input data; tuple order is
`(C, H, W)` for 2D and `(C, D, H, W)` for 3D.

The bundled image, NIfTI and MVTec AD 2 loaders expose this typed boundary.
`ingest_dataset()` validates, classifies and snapshots the complete supplied
dataset on each call, replacing the previous input catalog and derived records.
All later phases select their inputs from the repository and never iterate the
original dataloader again.
