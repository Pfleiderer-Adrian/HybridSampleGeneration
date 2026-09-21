# Data model and study storage

Study metadata and relationships are stored in `artifacts.sqlite`. NumPy arrays
remain normal files below `artifacts/`; the database stores paths relative to
the study folder.

A **record** is a small, structured description of one study entity, such as an
original sample, a real or synthetic anomaly, a hybrid sample, or a placement.
Records contain identifiers, metadata, artifact paths, and links to related
records; the image and segmentation arrays themselves remain separate `.npy`
files. In Python, these records are immutable dataclasses defined in
`hybrid_sample_generator/domain/records.py` and persisted in `artifacts.sqlite`.

```text
OriginalSample  1 ── 0..n  RealAnomaly  1 ── 0..n  SyntheticAnomaly
     1 │                                                     │ 1
       └── 0..n  HybridSample  1 ── 1..n  Placement  0..n ──┘
```

A placement is an independent record. It identifies one synthetic anomaly,
one hybrid sample, an insertion order, a matching method and score, and an
explicit normalized center position. Positions use `(y, x)` for 2D and
`(z, y, x)` for 3D. This removes the old one-to-one and filename-based
relationship between anomalies and generated samples.

Database constraints enforce unique component/variant/order combinations and
foreign-key integrity. Original IDs are deterministic hashes of the resolved
absolute `source_image_path`, falling back to `source_name` when no path is
provided. Derived IDs use parent IDs and component, variant or placement-order
indices. Relationships are stored explicitly rather than inferred from artifact
filenames. Changing the source path, or the fallback name, changes its ID.

One study has this layout:

```text
study/
  configuration.json
  artifacts.sqlite
  <study_name>.db                  # Optuna trials and model checkpoint references
  trained_models/
  artifacts/
    original_samples/<id>/{image,segmentation}.npy
    real_anomalies/<id>/{image,segmentation,roi_image,roi_segmentation}.npy
    synthetic_anomalies/<id>/{image,segmentation}.npy
    hybrid_samples/<id>/{image,segmentation}.npy
    placements/<id>/{roi_image,roi_segmentation}.npy
  evaluation_results/
```

Files appear as their pipeline phases run. Unannotated originals have no
segmentation artifact; hybrid images and masks are written during materialization.
Placement ROI images and masks are optional backend outputs. Loading a trained
generator requires its Optuna database and the referenced model checkpoint.
