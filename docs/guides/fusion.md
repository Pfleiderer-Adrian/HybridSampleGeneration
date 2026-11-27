# Fusion

`generator.materialize_hybrid_samples()` executes the stored
[hybrid plan](matching.md): every placement is fused into its target original
and the resulting image and label mask are written to the study. Two backends
are available:

| Backend | Method | Data |
| --- | --- | --- |
| `classical` (default) | intensity matching and alpha blending with a distance-based falloff | 2D and 3D |
| `poisson` | gradient-domain blending | 2D and 3D; considerably slower in 3D |

`config.fusion.parameters` is the selected backend's parameter dataclass.
`config.fusion.set_backend(name)` selects a backend and resets the parameters
to its defaults. The registry creates the matching dataclass and validates parameter types,
ranges and backend compatibility. Validation also runs when saving/loading a
configuration and creating a backend. JSON stores backend parameters directly
under `fusion.parameters`; unknown parameter names are rejected.

Both backends share the intensity normalization, anomaly/context relation and
background settings (`fusion_normalization_border_width`, `fusion_relation_*`,
`fusion_keep_bg` and related parameters) described below.

Materialization processes every planned hybrid. A hybrid whose fusion fails is
stored with status `failed` and its error message. After all hybrids are
processed, a `RuntimeError` lists the failures; pass
`materialize_hybrid_samples(raise_on_error=False)` to get the successfully
generated hybrids back without the exception.

An anomaly that is larger than its target image after restoring the extraction
scale cannot be fused: the hybrid fails with an error that names both shapes.
This usually indicates a target mask that covers far more of the cutout than
the mask of the real anomaly, for example a mask derived with a
`generation.background_threshold` that is too low.

## Classical fusion

```python
config.fusion.set_backend("classical")  # default backend

config.fusion.parameters.sq = 0.1
config.fusion.parameters.steepness_factor = 5.0
config.fusion.parameters.upsampling_factor = 2
config.fusion.parameters.dilation_size = 1
config.fusion.parameters.shave_pixels = 0
config.fusion.parameters.max_alpha = 0.9
config.fusion.parameters.fusion_variation = False
# Optional final pretrained harmonization step:
config.fusion.parameters.image_harmonization = "PCTNet"  # None, "PCTNet", or "LBM"

config.validate()
```

The classical backend crops the generated anomaly to its target mask, restores
its saved extraction scale, matches its intensity to the target context and
alpha-blends it at the planned normalized center. It returns the fused image, a
label mask in control coordinates and optional placement ROI artifacts. Multiple
placements are materialized in their stored order and their label masks are
combined.

Important classical parameters include:

- `max_alpha`, `sq`, `steepness_factor` and `upsampling_factor`, which control
  the maximum anomaly contribution and the distance-transform alpha falloff.
  The blending weight rises from the mask border inward and reaches
  `max_alpha` at `1 / steepness_factor` of the largest border distance inside
  the anomaly. A low `steepness_factor` gives a soft transition but fades out
  thin parts of an anomaly that also has a thick part; the default of `12`
  keeps them visible. A `max_alpha` below the default of `1.0` reduces the
  contrast of the whole anomaly by the same share.
- `segmentation_min_alpha`, which removes target-mask pixels whose anomaly
  contribution is below the configured threshold; `0.0` preserves the full
  target mask.
- `image_harmonization`, which optionally runs pretrained PCTNet or LBM after
  classical blending. It supports 2D samples with one or three channels and
  requires the separately installed `libcom` dependency and a CUDA GPU. One-channel
  samples are converted through RGB and restored to grayscale. Output outside
  the current placement mask is discarded, preserving the control image and
  previously fused anomalies. `None` keeps harmonization disabled.
- `fusion_use_sobel_for_alpha_mask`, `sobel_threshold`, `dilation_size` and
  `shave_pixels`, which enable and tune the optional edge-refined alpha path.
- `fusion_variation` plus `alpha_variation`, `sq_variation`,
  `steepness_variation` and `selected_confidence`, which sample blending
  parameters per placement.
- `fusion_normalization_border_width`: `None` disables fusion-time intensity
  normalization, `-1` uses the whole control, `0` uses the available fallback
  context, and a positive value uses a local ring of that width around the
  target mask. The default is `8`: a ring of only a few pixels mostly covers
  the rim of the anomaly and underestimates its contrast to the surroundings.
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

The complete parameter list is available in the
[classical fusion reference](../configuration/reference/fusion-classical.md).

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
[Poisson fusion reference](../configuration/reference/fusion-poisson.md).
