import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib


def _select_gui_backend(prefer: str = "tk") -> str:
    """Select and activate a Matplotlib GUI backend (QtAgg or TkAgg)."""

    prefer = (prefer or "").lower().strip()

    def try_qt() -> bool:
        try:
            matplotlib.use("QtAgg", force=True)
            # Validate that a Qt binding is available
            try:
                import PyQt6  # noqa: F401
            except Exception:
                try:
                    import PySide6  # noqa: F401
                except Exception:
                    try:
                        import PyQt5  # noqa: F401
                    except Exception:
                        import PySide2  # noqa: F401
            return True
        except Exception:
            return False

    def try_tk() -> bool:
        try:
            matplotlib.use("TkAgg", force=True)
            import tkinter  # noqa: F401

            return True
        except Exception:
            return False

    if prefer == "qt":
        if try_qt():
            return "QtAgg"
        if try_tk():
            return "TkAgg"
    else:
        if try_tk():
            return "TkAgg"
        if try_qt():
            return "QtAgg"

    raise RuntimeError(
        "No Matplotlib GUI backend available. Install either a Qt binding (PyQt/PySide) or tkinter."
    )


def _normalize_exts(exts: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for e in exts:
        e = (e or "").strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.append(e.lower())
    return tuple(dict.fromkeys(out))  # unique, keep order


def _index_folder(folder: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    """Return mapping: basename -> fullpath for the allowed extensions."""
    # Fallback behavior: if folder does not exist, return empty mapping
    # so the GUI can show placeholders instead of crashing.
    if not os.path.isdir(folder):
        return {}

    mapping: Dict[str, str] = {}
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        base, ext = os.path.splitext(name)
        if ext.lower() not in exts:
            continue
        # Keep first occurrence per basename
        mapping.setdefault(base, full)
    return mapping


def _load_array(path: str) -> np.ndarray:
    """Load .npy or .npz into a numpy array."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        z = np.load(path)
        if isinstance(z, np.lib.npyio.NpzFile):
            keys = list(z.keys())
            if not keys:
                raise ValueError(f"Empty npz: {path}")
            return z[keys[0]]
        return z
    # Fallback: try np.load
    return np.load(path)


def _robust_window_params(vol: np.ndarray) -> Tuple[float, float]:
    """Return (center, half0) from robust percentiles over the volume."""
    v = vol.astype(np.float32, copy=False)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.percentile(finite, [2, 98])
        if float(lo) == float(hi):
            lo = float(finite.min())
            hi = float(finite.max())
            if lo == hi:
                hi = lo + 1.0
    center = 0.5 * (float(lo) + float(hi))
    half0 = 0.5 * (float(hi) - float(lo))
    if half0 <= 0:
        half0 = 1.0
    return center, half0


def _window_limits(center: float, half0: float, contrast: float) -> Tuple[float, float]:
    contrast = max(float(contrast), 1e-6)
    half = half0 / contrast
    vmin = center - half
    vmax = center + half
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _set_slider_range(slider, vmin: float, vmax: float, val: float):
    """Update a Matplotlib Slider's range (works for standard backends)."""
    slider.valmin = float(vmin)
    slider.valmax = float(vmax)
    try:
        slider.ax.set_xlim(float(vmin), float(vmax))
    except Exception:
        pass
    try:
        slider.valstep = 1  # depth is integer
    except Exception:
        pass
    slider.set_val(val)


def _folder_tag(folder: str) -> str:
    """Return '/parent/folder' for the given path."""
    folder = os.path.normpath(folder)
    name = os.path.basename(folder)
    parent = os.path.basename(os.path.dirname(folder))
    if parent:
        return f"/{parent}/{name}"
    return f"/{name}"


@dataclass
class View:
    label: str
    source: str
    ax: any
    im: any
    s_depth: any
    s_contrast: any
    vol: Optional[np.ndarray] = None  # (D,H,W)
    center: float = 0.0
    half0: float = 1.0

    def set_volume(self, vol_dhw: np.ndarray):
        self.vol = vol_dhw.astype(np.float32, copy=False)
        self.center, self.half0 = _robust_window_params(self.vol)

    @property
    def D(self) -> int:
        if self.vol is None:
            return 0
        return int(self.vol.shape[0])

    def render(self):
        if self.vol is None or self.D <= 0:
            self.im.set_data(np.zeros((10, 10), dtype=np.float32))
            self.im.set_clim(0.0, 1.0)
            self.ax.set_title(f"{self.label}\n{self.source} | (no data)")
            return

        d = int(self.s_depth.val)
        d = max(0, min(d, self.D - 1))
        contrast = float(self.s_contrast.val)
        vmin, vmax = _window_limits(self.center, self.half0, contrast)

        self.im.set_data(self.vol[d])
        self.im.set_clim(vmin, vmax)
        # NOTE: basename replaced by "/parent/folder"
        self.ax.set_title(
            f"{self.label}\n{self.source} | d={d}/{self.D - 1} | c={contrast:.2f}",
            fontsize=10,
        )


def visualize_three_folders(
    folder1: str,
    folder2: str,
    folder3: str,
    channel: int = 0,
    cmap: str = "gray",
    backend_preference: str = "tk",
    exts: Sequence[str] = (".npy",),
    labels: Optional[Sequence[str]] = (
        "ROI cutout Area",
        "Real Anomaly cutout",
        "Generated Synth. Anomaly",
    ),
):
    """GUI: show 3 volumes side-by-side (one per folder) with per-view depth/contrast sliders.

    Volumes are *paired by basename*. If a folder or a particular file is missing,
    the corresponding panel shows a fallback placeholder text instead of crashing.

    Above each image we show: '/parent/folder | depth | contrast' (instead of the basename).
    """

    _select_gui_backend(prefer=backend_preference)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    exts_n = _normalize_exts(exts)

    idx1 = _index_folder(folder1, exts_n)
    idx2 = _index_folder(folder2, exts_n)
    idx3 = _index_folder(folder3, exts_n)

    # Use UNION (not intersection) so we can still browse sets even if one folder/file is missing.
    all_names = sorted(set(idx1.keys()) | set(idx2.keys()) | set(idx3.keys()))
    if not all_names:
        # No files anywhere: still open GUI with placeholders.
        all_names = ["(no files found)"]

    if labels is None or len(labels) != 3:
        labels = ["Folder 1", "Folder 2", "Folder 3"]

    folders = [folder1, folder2, folder3]
    sources = [_folder_tag(f) for f in folders]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    try:
        fig.canvas.manager.set_window_title("3-Volume Set Viewer")
    except Exception:
        pass

    plt.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.28, wspace=0.02)

    views: List[View] = []
    for i, ax in enumerate(axs):
        ax.set_axis_off()
        im = ax.imshow(np.zeros((10, 10), dtype=np.float32), cmap=cmap, vmin=0.0, vmax=1.0)

        # Align sliders under each image axes
        l, b, w, h = ax.get_position().bounds
        ax_depth = fig.add_axes([l, 0.16, w, 0.03])
        ax_contrast = fig.add_axes([l, 0.11, w, 0.03])

        s_depth = Slider(ax_depth, "Depth (D)", 0, 1, valinit=0, valstep=1)
        s_contrast = Slider(ax_contrast, "Contrast", 0.2, 5.0, valinit=1.0)

        view = View(
            label=str(labels[i]),
            source=str(sources[i]),
            ax=ax,
            im=im,
            s_depth=s_depth,
            s_contrast=s_contrast,
        )
        views.append(view)

    state = {"set_idx": 0}

    def load_set(set_idx: int):
        set_idx = int(set_idx) % len(all_names)
        basename = all_names[set_idx]

        idxs = [idx1, idx2, idx3]

        for v, idx_map, folder in zip(views, idxs, folders):
            path = idx_map.get(basename)

            # Missing folder or missing file for this basename -> show placeholder
            if path is None or not os.path.isfile(path):
                v.vol = None
                v.center, v.half0 = 0.0, 1.0

                # Keep sliders usable (but inert)
                _set_slider_range(v.s_depth, 0, 1, 0)
                v.s_contrast.set_val(1.0)

                v.im.set_data(np.zeros((10, 10), dtype=np.float32))
                v.im.set_clim(0.0, 1.0)

                if not os.path.isdir(folder):
                    msg = f"(missing folder)"
                else:
                    msg = f"(missing file: {basename})"
                v.ax.set_title(f"{v.label}\n{v.source}\n{msg}", fontsize=10)
                continue

            # Have a file -> load normally
            arr = _load_array(path)
            if arr.ndim != 4:
                raise ValueError(f"Expected (C,D,H,W) in {path}, got {arr.shape}")
            C, D, H, W = arr.shape
            if not (0 <= channel < C):
                raise ValueError(f"Channel {channel} out of range for {path}: C={C}")

            vol = arr[channel]  # (D,H,W)
            v.set_volume(vol)
            d0 = v.D // 2
            _set_slider_range(v.s_depth, 0, max(v.D - 1, 0), d0)
            v.s_contrast.set_val(1.0)
            v.render()

        # You can keep basename here (it helps browsing sets). If you want it removed too, tell me.
        fig.suptitle(f"Set {set_idx + 1}/{len(all_names)}: {basename}", fontsize=12)
        fig.canvas.draw_idle()

        state["set_idx"] = set_idx

    def on_any_slider(_):
        for v in views:
            # Keep placeholder titles as-is for missing views
            if v.vol is None or v.D <= 0:
                continue
            v.render()
        fig.canvas.draw_idle()

    for v in views:
        v.s_depth.on_changed(on_any_slider)
        v.s_contrast.on_changed(on_any_slider)

    def on_scroll(event):
        # Scroll should only affect the view whose axes is under the cursor
        for v in views:
            if event.inaxes == v.ax:
                if v.vol is None or v.D <= 0:
                    break
                d = int(v.s_depth.val)
                step = 1 if event.button == "up" else -1
                new_d = max(0, min(d + step, v.D - 1))
                v.s_depth.set_val(new_d)
                break

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Next Set button
    ax_btn = fig.add_axes([0.445, 0.03, 0.11, 0.06])
    btn_next = Button(ax_btn, "Next Set")

    def on_next(_event):
        load_set(state["set_idx"] + 1)

    btn_next.on_clicked(on_next)

    # Load first set on start
    load_set(0)

    import matplotlib.pyplot as plt

    plt.show(block=True)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize 3 matched volumes (C,D,H,W) from 3 folders side-by-side. "
            "Files are matched by identical basename across folders."
        )
    )
    parser.add_argument("folder", help="Result Folder")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to display (default: 0)")
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap (default: gray)")
    parser.add_argument("--backend", choices=["tk", "qt"], default="tk", help="Preferred GUI backend")
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".npy"],
        help="File extensions to consider (default: .npy). Example: --exts .npy .npz",
    )

    args = parser.parse_args()

    anomaly_folder = os.path.join(args.folder, "anomaly_data")
    roi_folder = os.path.join(args.folder, "anomaly_roi_data")
    synth_folder = os.path.join(args.folder, "synth_anomaly_data")

    visualize_three_folders(
        roi_folder,
        anomaly_folder,
        synth_folder,
        channel=args.channel,
        cmap=args.cmap,
        backend_preference=args.backend,
        exts=args.exts,
    )


if __name__ == "__main__":
    main()
