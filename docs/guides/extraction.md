# Extraction

`generator.extract_anomalies()` turns every annotated anomaly of the ingested
anomalous originals into a `RealAnomaly` record with a normalized cutout, its
mask and an ROI of the surrounding image. It needs no dataloader: the originals
come from the study repository.

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
  fraction of the target spatial cutout area/volume. The default is `0.01`.
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
- `config.extraction.roi.preserve_aspect_ratio`: expand a dynamic ROI without
  shrinking any axis until it follows the spatial aspect ratio of the complete
  image sample. This setting is ignored when `fixed_size` is configured.

ROI tuples contain spatial axes only: `(H, W)` for 2D and `(D, H, W)` for 3D;
an optional leading channel entry is also accepted and ignored for ROI sizing.

A ROI that would extend past the image border keeps its size and is shifted
inside the image, so the anomaly is no longer in its center. Extraction stores
this shift as `roi_center_offset` (anomaly center minus ROI center, in pixels
per axis) in the metadata of the real anomaly.
[Hybrid planning](matching.md#template-matching) uses it to place anomalies
from the border at the border again.

The complete parameter list is available in the
[extraction reference](../configuration/reference/extraction.md).
