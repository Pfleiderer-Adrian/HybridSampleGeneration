# Pipeline stages

After [ingesting input data](input-data.md), the pipeline extracts real anomalies, generates variants, plans hybrid samples, and fuses them into target originals. The [quick start](../getting-started/quick-start.md) shows the calls in order.

## Extraction

Extraction finds connected components in the positive segmentation, crops each
component, downscales it only when it exceeds the configured target size, and
center-pads it to `config.extraction.anomaly_size`. By default, one scale is
used for every spatial axis so the aspect ratio is preserved. The original ROI, mask,
normalized source center, scale factors and normalization metadata are retained
with the resulting `RealAnomaly` record.

The principal settings are:

- `config.extraction.separate_components`: extract connected components as
  separate real anomalies. When disabled, the positive mask is handled as one
  region.
- `config.extraction.min_coverage_ratio`: discard components smaller than this
  fraction of the target spatial cutout area/volume. The default is `0.05`.
- `config.extraction.add_background_noise`: add a small noise floor to otherwise
  constant cutout background.
- `config.extraction.preserve_aspect_ratio`: use one scale for every spatial
  axis before padding. Set it to `False` to stretch axes independently.
- `config.extraction.normalization`: `"z-score"` (mean/std),
  `"zscore_median"` (median/MAD), or `None`.
- `config.extraction.roi.fixed_size`: fixed spatial ROI size, or `None` for a
  dynamic ROI.
- `config.extraction.roi.min_padding` and `padding_ratio`: for a dynamic ROI,
  its size on each axis is the anomaly extent plus the larger of the absolute
  padding and proportional padding.
- `config.extraction.roi.min_size`: scalar or per-axis lower bound for a dynamic
  ROI.

ROI tuples contain spatial axes only: `(H, W)` for 2D and `(D, H, W)` for 3D.

## Synthetic variants

`config.generation.variants_per_real_anomaly` controls how many children are
generated for every `RealAnomaly`. Each child has its own deterministic ID,
variant index, seed, image and target mask. Feedback generation is bounded by
`config.generation.feedback.max_attempts`.

## Hybrid planning

- `hybrids_per_original`: requested number of hybrid variants per eligible
  target original.
- `anomalies_per_hybrid`: target placement count in each hybrid.
- `max_anomalies_per_hybrid_deviation`: deterministic random deviation around
  the placement count.
- `reuse_synthetic_across_hybrids`: whether the same synthetic ID may be used
  by more than one hybrid.
- `allow_sibling_variants_in_same_hybrid`: whether variants with the same real
  parent may occur together in one hybrid.
- `intensity_weight` and `gradient_weight`: weights for template matching.
- `seed`: reproducibility seed owned by the matching phase.

`local`, `global`, `batchwise` and `fixed_from_extraction_control_fusion` target
originals with `has_anomaly=False`. `fixed_from_extraction_anomaly_fusion` targets
anomalous originals. Only real anomalies with synthetic variants are candidates.
A hybrid can contain fewer placements than requested; if no eligible placement
is found, that hybrid is omitted entirely.

`local` assigns real anomaly ROIs sequentially across hybrids and controls.
It searches the full control image only for the next ROI with an eligible
synthetic variant, trying another ROI if the match is invalid or overlaps an
existing placement. Matching stops as soon as the requested placement count is
reached. Each hybrid tries at most one pass through the ROI pool; unused ROIs
are not loaded or matched. The ROI sequence restarts on each planning run.

`global` evaluates all real anomaly ROIs for each control and selects placements
in descending match-score order. `batchwise` evaluates and ranks only a seeded
subset of at most `batch_size` ROIs per control.

All three modes prepare control and ROI gradients on demand and reuse them
within the planning run. Pair results, including rejected pairs, are cached in
SQLite by matcher signature. Repeated planning with unchanged inputs and weights
reuses evaluated pairs, including when changing modes; new pairs are computed
only as needed. `fixed_from_extraction_control_fusion` reuses source centers on
arbitrary controls;
`fixed_from_extraction_anomaly_fusion` joins originals and real anomalies by
foreign key and places variants back at their extraction positions.

## Classical fusion

`config.fusion.parameters` is the selected backend's parameter dataclass.
Configure its fields directly; the former `set_fusion_params(...)` wrapper is
removed:

```python
config.fusion.set_backend("classical")  # stable default backend

config.fusion.parameters.sq = 0.1
config.fusion.parameters.steepness_factor = 5.0
config.fusion.parameters.upsampling_factor = 2
config.fusion.parameters.dilation_size = 1
config.fusion.parameters.shave_pixels = 0
config.fusion.parameters.max_alpha = 0.9  # default: classical backend
config.fusion.parameters.fusion_variation = False

config.validate()
```

The registry creates the matching dataclass and validates parameter types,
ranges and backend compatibility. Validation also runs when saving/loading a
configuration and creating a backend. JSON stores backend parameters directly
under `fusion.parameters`; unknown parameter names are rejected.

The classical backend crops the generated anomaly to its target mask, restores
its saved extraction scale, matches its intensity to the target context and
alpha-blends it at the planned normalized center. It returns the fused image, a
label mask in control coordinates and optional placement ROI artifacts. Multiple
placements are materialized in their stored order and their label masks are
combined.

Important classical parameters include:

- `max_alpha`, `sq`, `steepness_factor` and `upsampling_factor`, which control
  the maximum anomaly contribution and the distance-transform alpha falloff.
- `fusion_use_sobel_for_alpha_mask`, `sobel_threshold`, `dilation_size` and
  `shave_pixels`, which enable and tune the optional edge-refined alpha path.
- `fusion_variation` plus `alpha_variation`, `sq_variation`,
  `steepness_variation` and `selected_confidence`, which sample blending
  parameters per placement.
- `fusion_normalization_border_width`: `None` disables fusion-time intensity
  normalization, `-1` uses the whole control, `0` uses the available fallback
  context, and a positive value uses a local ring around the target mask.
- `fusion_restore_anomaly_bg_relation`, `fusion_relation_mode`,
  `fusion_relation_norm_classes_separately` and
  `fusion_relation_min_context_size`, which control whether the original
  anomaly/context relation is restored and how multiclass context is estimated.
- `fusion_keep_bg`, `fusion_bg_value`, `fusion_relative_bg_threshold` and
  `fusion_bg_exterior_only`, which can preserve detected control-background
  pixels unchanged.

Local normalization uses robust median/IQR context statistics. Relation mode
`delta` preserves the original median difference; `ratio` preserves the median
ratio and is intended for strictly positive intensities away from zero. If a
local or class-specific ring contains too few values, the backend falls back to
available target-mask-outside context; if that is still insufficient, the scope
is left unnormalized.

## Poisson fusion

Select gradient-domain blending as a separate backend:

```python
config.fusion.set_backend("poisson")
config.fusion.parameters.guidance_mode = "source"  # or "mixed"
config.fusion.parameters.solver_rtol = 1e-5
config.fusion.parameters.solver_max_iterations = 2000
config.validate()
```

Poisson fusion supports channel-first 2D `(C,H,W)` and true volumetric 3D
`(C,D,H,W)` data. It solves over a 4-neighbor grid in 2D and a 6-neighbor grid
in 3D. `source` guidance preserves anomaly gradients. `mixed` guidance selects
the stronger source or control gradient independently for each channel and
edge, which can retain important control-image boundaries.

Generated pixels outside the target mask are removed before placement but are
not used as Poisson boundary guidance. The corresponding control-image values
provide the exterior source baseline, preventing artificial gradients from the
cleaned anomaly background. A one-pixel or one-voxel control halo supplies
Dirichlet boundary values around the placed mask.

!!! warning "3D runtime and memory"
    True 3D Poisson blending constructs a sparse system over all target-mask
    voxels and solves it once per image channel. It can require substantially
    more computation time and memory than classical alpha blending. The first
    3D fusion call on each backend instance emits a `RuntimeWarning` containing
    the mask voxel and channel counts.

The complete parameter list is available in the
