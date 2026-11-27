# Glossary

Original (original sample)
:   An image or volume supplied by the dataloader and stored during ingest.
    Every original is either anomalous or a control.

Anomalous original
:   An original whose segmentation mask contains at least one positive pixel.
    Real anomalies are extracted from anomalous originals.

Control
:   An original without anomaly: its mask is empty (annotated control) or
    missing (unannotated control). Controls are the usual targets into which
    synthetic anomalies are placed.

Real anomaly
:   One anomaly extracted from an anomalous original, by default one per
    connected component of the mask. It consists of a cutout, its mask and an
    ROI.

Cutout
:   The anomaly crop used to train the generator: the masked anomaly, scaled
    down if necessary, center-padded to `config.extraction.anomaly_size` and
    intensity-normalized.

ROI (region of interest)
:   A crop of the unmodified original around a real anomaly, including
    surrounding context. Planning uses it as the template for template
    matching; evaluation compares it with the fused placement ROI.

Synthetic anomaly (synthetic variant)
:   A generated variant of one real anomaly, consisting of an image and a
    target mask in cutout size. The real anomaly is its parent.

Sibling variants
:   Synthetic variants that share the same real parent.

Target mask
:   The mask of a synthetic variant. Mask-conditioned models generate the
    image for a transformed copy of the parent's mask; unconditioned models
    derive the mask from the generated image.

Hybrid (hybrid sample)
:   A target original with one or more synthetic anomalies fused into it,
    together with the resulting label mask.

Placement
:   The assignment of one synthetic anomaly to one hybrid at a normalized
    center position, with an insertion order and, for template matching, a
    score.

Hybrid plan
:   All hybrid and placement records written by `plan_hybrid_samples()`,
    before any image is produced.

Materialization
:   Executing the hybrid plan: fusing every placement and writing the hybrid
    images and masks.

Fusion
:   Blending one synthetic anomaly into its target original, using the
    classical alpha-blending backend or the Poisson backend.

Study
:   One run of the pipeline with its configuration, database, trained models
    and artifacts in a single study folder.

Record
:   The database entry of one study entity (original, real anomaly, synthetic
    anomaly, hybrid or placement). Arrays are stored as separate files.

Artifact
:   An array file belonging to a record, such as an image, a mask or an ROI.

Trial
:   One Optuna training run of the generator with one sampled set of model
    parameters.
