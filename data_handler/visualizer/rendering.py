from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from synthesizer.ArtifactStore import ArtifactStore


@dataclass(frozen=True)
class DisplayPlane:
    image: np.ndarray
    depth: int
    slice_index: int
    mode: str


@dataclass(frozen=True)
class Marker:
    position: tuple[float, ...]
    label: str
    color: str = "#ffd166"
    selected: bool = False


@dataclass(frozen=True)
class PanelSpec:
    title: str
    image_path: str | None = None
    mask_path: str | None = None
    reference_path: str | None = None
    image: np.ndarray | None = None
    reference: np.ndarray | None = None
    markers: tuple[Marker, ...] = ()
    detail: str = ""


class ArrayCache:
    """Small LRU cache for arrays selected in the GUI."""

    def __init__(self, artifact_store: ArtifactStore, max_items: int = 16) -> None:
        self.artifact_store = artifact_store
        self.max_items = max(1, int(max_items))
        self._values: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str | None) -> np.ndarray | None:
        if not path:
            return None
        cached = self._values.pop(path, None)
        if cached is None:
            if not self.artifact_store.exists(path):
                return None
            cached = self.artifact_store.load_array(path, mmap_mode="r")
        self._values[path] = cached
        while len(self._values) > self.max_items:
            self._values.popitem(last=False)
        return cached

    def clear(self) -> None:
        self._values.clear()


def display_plane(
    array: np.ndarray,
    *,
    slice_index: int = 0,
    channel: str | int = "auto",
) -> DisplayPlane:
    """Convert channel-first 2D/3D data into a Matplotlib display plane."""
    array = np.asarray(array)
    requested_slice = max(0, int(slice_index))

    if array.ndim == 4:
        depth = int(array.shape[1])
        used_slice = min(requested_slice, max(depth - 1, 0))
        frame = array[:, used_slice]
        return DisplayPlane(
            _channel_first_frame(frame, channel), depth, used_slice, "C,D,H,W"
        )
    if array.ndim == 3:
        return DisplayPlane(
            _channel_first_frame(array, channel), 1, 0, "C,H,W"
        )
    if array.ndim == 2:
        return DisplayPlane(array, 1, 0, "H,W")
    raise ValueError(
        f"Cannot display shape {array.shape}; expected H,W / C,H,W / C,D,H,W."
    )


def display_mask_plane(
    array: np.ndarray,
    *,
    slice_index: int = 0,
) -> DisplayPlane:
    array = np.asarray(array)
    requested_slice = max(0, int(slice_index))
    if array.ndim == 4:
        depth = int(array.shape[1])
        used_slice = min(requested_slice, max(depth - 1, 0))
        return DisplayPlane(
            np.max(array[:, used_slice], axis=0), depth, used_slice, "C,D,H,W"
        )
    if array.ndim == 3:
        return DisplayPlane(np.max(array, axis=0), 1, 0, "C,H,W")
    if array.ndim == 2:
        return DisplayPlane(array, 1, 0, "H,W")
    raise ValueError(
        f"Cannot display mask shape {array.shape}; expected H,W / C,H,W / C,D,H,W."
    )


def normalize_for_display(
    image: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    contrast: float = 1.0,
) -> np.ndarray:
    """Map arbitrary finite intensity ranges to [0, 1] using one shared window."""
    image = np.asarray(image, dtype=np.float32)
    reference = image if reference is None else np.asarray(reference, dtype=np.float32)
    finite = reference[np.isfinite(reference)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low, high = np.percentile(finite, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high):
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    center = (float(low) + float(high)) / 2.0
    half_width = (float(high) - float(low)) / (2.0 * max(float(contrast), 1e-6))
    window_low = center - half_width
    window_high = center + half_width
    normalized = (image - window_low) / max(window_high - window_low, 1e-12)
    return np.clip(
        np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )


def render_panel(
    axis,
    spec: PanelSpec,
    cache: ArrayCache,
    *,
    slice_index: int,
    contrast: float,
    channel: str | int,
    show_mask: bool,
    mask_opacity: float,
    grayscale_cmap: str = "gray",
) -> tuple[int, str]:
    """Render one panel and return its depth and a compact status string."""
    axis.clear()
    axis.set_axis_off()
    image_array = spec.image if spec.image is not None else cache.get(spec.image_path)
    if image_array is None:
        _placeholder(axis, spec.title, spec.detail or "Artifact not available")
        return 1, "missing"

    try:
        plane = display_plane(image_array, slice_index=slice_index, channel=channel)
        reference_array = (
            spec.reference
            if spec.reference is not None
            else cache.get(spec.reference_path)
        )
        reference_plane = None
        if reference_array is not None:
            reference_plane = display_plane(
                reference_array,
                slice_index=plane.slice_index,
                channel=channel,
            ).image
        display = normalize_for_display(
            plane.image,
            reference=reference_plane,
            contrast=contrast,
        )
        if display.ndim == 3 and display.shape[-1] in (3, 4):
            axis.imshow(display, aspect="equal")
        else:
            axis.imshow(display, cmap=grayscale_cmap, vmin=0, vmax=1, aspect="equal")

        depth = plane.depth
        mask_status = ""
        if show_mask and spec.mask_path:
            mask_array = cache.get(spec.mask_path)
            if mask_array is not None:
                mask_plane = display_mask_plane(
                    mask_array, slice_index=plane.slice_index
                )
                depth = max(depth, mask_plane.depth)
                mask, label_count, foreground = prepare_mask(mask_plane.image)
                if mask.shape == display.shape[:2]:
                    cmap, norm = mask_colormap(
                        label_count,
                        background_alpha=0.0,
                        foreground_alpha=mask_opacity,
                    )
                    axis.imshow(
                        mask,
                        cmap=cmap,
                        norm=norm,
                        interpolation="nearest",
                        aspect="equal",
                    )
                    mask_status = f", mask={foreground:.1f}%"
                else:
                    mask_status = f", mask shape mismatch {mask.shape}"

        _draw_markers(axis, spec.markers, display.shape[:2], plane)
        axis.set_title(spec.title, fontsize=9, pad=7)
        status = (
            f"{plane.mode}, shape={tuple(np.asarray(image_array).shape)}, "
            f"slice={plane.slice_index + 1}/{plane.depth}{mask_status}"
        )
        return depth, status
    except Exception as exc:
        _placeholder(axis, spec.title, f"Could not render artifact:\n{exc}")
        return 1, f"error: {exc}"


def prepare_mask(mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    raw = np.nan_to_num(np.asarray(mask), nan=0.0, posinf=0.0, neginf=0.0)
    rounded = np.rint(raw).astype(np.int64)
    labels_raw = np.where(rounded > 0, rounded, 0)
    positive_labels = [int(value) for value in np.unique(labels_raw) if value > 0]
    display = np.zeros(labels_raw.shape, dtype=np.int32)
    for display_label, source_label in enumerate(positive_labels, start=1):
        display[labels_raw == source_label] = display_label
    foreground = 100.0 * np.count_nonzero(display) / max(display.size, 1)
    return display, len(positive_labels) + 1, float(foreground)


def mask_colormap(
    label_count: int,
    *,
    background_alpha: float = 1.0,
    foreground_alpha: float = 1.0,
):
    base = [
        "#000000",
        "#ffd166",
        "#06d6a0",
        "#118ab2",
        "#ef476f",
        "#a78bfa",
        "#f97316",
        "#22c55e",
        "#e879f9",
    ]
    label_count = max(1, int(label_count))
    if label_count > len(base):
        tab20 = plt.get_cmap("tab20")
        base.extend(
            matplotlib.colors.to_hex(tab20(index % tab20.N))
            for index in range(label_count - len(base))
        )
    colors = [
        (
            *matplotlib.colors.to_rgb(color),
            float(background_alpha if index == 0 else foreground_alpha),
        )
        for index, color in enumerate(base[:label_count])
    ]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(
        np.arange(-0.5, label_count + 0.5, 1.0), label_count
    )
    return cmap, norm


def _channel_first_frame(frame: np.ndarray, channel: str | int) -> np.ndarray:
    channels = int(frame.shape[0])
    if channels <= 0:
        raise ValueError("Image has no channels.")
    channel_key = str(channel).strip().lower()
    if channel_key in {"auto", "rgb"} and channels in (3, 4):
        return np.moveaxis(frame[:3], 0, -1)
    if channel_key == "rgb" and channels < 3:
        channel_key = "0"
    if channel_key == "auto":
        channel_index = 0
    else:
        try:
            channel_index = int(channel_key)
        except ValueError as exc:
            raise ValueError(f"Unknown channel selection {channel!r}.") from exc
    channel_index = min(max(channel_index, 0), channels - 1)
    return frame[channel_index]


def _draw_markers(
    axis,
    markers: tuple[Marker, ...],
    image_shape: tuple[int, int],
    plane: DisplayPlane,
) -> None:
    height, width = image_shape
    for marker in markers:
        if len(marker.position) == 3:
            marker_slice = float(marker.position[0]) * max(plane.depth - 1, 1)
            if abs(marker_slice - plane.slice_index) > 0.75:
                continue
            y, x = marker.position[1:]
        else:
            y, x = marker.position
        size = 150 if marker.selected else 75
        axis.scatter(
            [float(x) * width],
            [float(y) * height],
            s=size,
            facecolors="none",
            edgecolors=marker.color,
            linewidths=2.2 if marker.selected else 1.3,
        )
        axis.annotate(
            marker.label,
            (float(x) * width, float(y) * height),
            color=marker.color,
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )


def _placeholder(axis, title: str, detail: str) -> None:
    axis.imshow(np.full((12, 12), 0.94), cmap="gray", vmin=0, vmax=1)
    axis.set_title(title, fontsize=9, pad=7)
    axis.text(
        0.5,
        0.5,
        detail,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="#555555",
        fontsize=8,
        wrap=True,
    )
