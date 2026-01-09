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
    return np.load(path)


def _robust_window_params(arr: np.ndarray) -> Tuple[float, float]:
    """Return (center, half0) from robust percentiles over the array (2D/3D/RGB)."""
    v = arr.astype(np.float32, copy=False)
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
        slider.valstep = 1
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
    ax_depth: any
    ax_contrast: any
    s_depth: any
    s_contrast: any

    # Data holders (mutually exclusive per mode)
    vol3d: Optional[np.ndarray] = None      # (D,H,W)
    img2d: Optional[np.ndarray] = None      # (H,W)
    vol3d_rgb: Optional[np.ndarray] = None  # (D,H,W,3)
    img2d_rgb: Optional[np.ndarray] = None  # (H,W,3)

    mode: str = "none"  # "3d_gray", "2d_gray", "3d_rgb", "2d_rgb", "none"
    center: float = 0.0
    half0: float = 1.0

    def _set_depth_visible(self, visible: bool):
        self.ax_depth.set_visible(bool(visible))
        # Move contrast slider up if depth hidden
        l, _, w, h = self.ax_contrast.get_position().bounds
        y = 0.11 if visible else 0.16
        self.ax_contrast.set_position([l, y, w, h])

    def set_placeholder(self):
        self.mode = "none"
        self.vol3d = None
        self.img2d = None
        self.vol3d_rgb = None
        self.img2d_rgb = None
        self.center, self.half0 = 0.0, 1.0
        self._set_depth_visible(False)

    def set_volume_3d_gray(self, vol_dhw: np.ndarray):
        self.mode = "3d_gray"
        self.vol3d = vol_dhw.astype(np.float32, copy=False)
        self.img2d = None
        self.vol3d_rgb = None
        self.img2d_rgb = None
        self.center, self.half0 = _robust_window_params(self.vol3d)
        self._set_depth_visible(True)

    def set_image_2d_gray(self, img_hw: np.ndarray):
        self.mode = "2d_gray"
        self.img2d = img_hw.astype(np.float32, copy=False)
        self.vol3d = None
        self.vol3d_rgb = None
        self.img2d_rgb = None
        self.center, self.half0 = _robust_window_params(self.img2d)
        self._set_depth_visible(False)

    def set_volume_3d_rgb(self, vol_dhw3: np.ndarray):
        self.mode = "3d_rgb"
        self.vol3d_rgb = vol_dhw3.astype(np.float32, copy=False)
        self.vol3d = None
        self.img2d = None
        self.img2d_rgb = None
        self.center, self.half0 = _robust_window_params(self.vol3d_rgb)
        self._set_depth_visible(True)

    def set_image_2d_rgb(self, img_hw3: np.ndarray):
        self.mode = "2d_rgb"
        self.img2d_rgb = img_hw3.astype(np.float32, copy=False)
        self.vol3d = None
        self.img2d = None
        self.vol3d_rgb = None
        self.center, self.half0 = _robust_window_params(self.img2d_rgb)
        self._set_depth_visible(False)

    @property
    def D(self) -> int:
        if self.vol3d is not None:
            return int(self.vol3d.shape[0])
        if self.vol3d_rgb is not None:
            return int(self.vol3d_rgb.shape[0])
        return 0

    def _normalize_rgb(self, rgb: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        denom = (vmax - vmin) if (vmax != vmin) else 1.0
        out = (rgb - vmin) / denom
        return np.clip(out, 0.0, 1.0)

    def render(self):
        if self.mode == "none":
            return

        contrast = float(self.s_contrast.val)
        vmin, vmax = _window_limits(self.center, self.half0, contrast)

        if self.mode == "3d_gray":
            d = max(0, min(int(self.s_depth.val), self.D - 1))
            self.im.set_data(self.vol3d[d])
            self.im.set_clim(vmin, vmax)
            self.ax.set_title(
                f"{self.label}\n{self.source} | d={d}/{self.D - 1} | c={contrast:.2f}",
                fontsize=10,
            )
            return

        if self.mode == "2d_gray":
            self.im.set_data(self.img2d)
            self.im.set_clim(vmin, vmax)
            self.ax.set_title(
                f"{self.label}\n{self.source} | c={contrast:.2f}",
                fontsize=10,
            )
            return

        if self.mode == "3d_rgb":
            d = max(0, min(int(self.s_depth.val), self.D - 1))
            rgb = self._normalize_rgb(self.vol3d_rgb[d], vmin, vmax)
            self.im.set_data(rgb)  # (H,W,3) => RGB
            self.ax.set_title(
                f"{self.label}\n{self.source} | d={d}/{self.D - 1} | c={contrast:.2f}",
                fontsize=10,
            )
            return

        if self.mode == "2d_rgb":
            rgb = self._normalize_rgb(self.img2d_rgb, vmin, vmax)
            self.im.set_data(rgb)
            self.ax.set_title(
                f"{self.label}\n{self.source} | c={contrast:.2f}",
                fontsize=10,
            )
            return


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
    """
    GUI: show 3 matched arrays side-by-side (one per folder) with per-view sliders.

    Supported shapes:
      - (C,D,H,W): if C==3 => RGB slices, else => grayscale of selected --channel
      - (C,H,W):   if C==3 => RGB,        else => grayscale of selected --channel

    Depth slider is shown ONLY for (C,D,H,W) cases.
    """

    _select_gui_backend(prefer=backend_preference)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    exts_n = _normalize_exts(exts)

    idx1 = _index_folder(folder1, exts_n)
    idx2 = _index_folder(folder2, exts_n)
    idx3 = _index_folder(folder3, exts_n)

    all_names = sorted(set(idx1.keys()) | set(idx2.keys()) | set(idx3.keys()))
    if not all_names:
        all_names = ["(no files found)"]

    if labels is None or len(labels) != 3:
        labels = ["Folder 1", "Folder 2", "Folder 3"]

    folders = [folder1, folder2, folder3]
    sources = [_folder_tag(f) for f in folders]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    try:
        fig.canvas.manager.set_window_title("3-Array Set Viewer")
    except Exception:
        pass

    plt.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.28, wspace=0.02)

    views: List[View] = []
    for i, ax in enumerate(axs):
        ax.set_axis_off()

        # Start with a grayscale placeholder; RGB updates via set_data(H,W,3) later.
        im = ax.imshow(np.zeros((10, 10), dtype=np.float32), cmap=cmap, vmin=0.0, vmax=1.0)

        # Sliders under each image
        l, b, w, h = ax.get_position().bounds
        ax_depth = fig.add_axes([l, 0.16, w, 0.03])
        ax_contrast = fig.add_axes([l, 0.11, w, 0.03])

        s_depth = Slider(ax_depth, "Depth (D)", 0, 1, valinit=0, valstep=1)
        s_contrast = Slider(ax_contrast, "Contrast", 0.2, 5.0, valinit=1.0)

        v = View(
            label=str(labels[i]),
            source=str(sources[i]),
            ax=ax,
            im=im,
            ax_depth=ax_depth,
            ax_contrast=ax_contrast,
            s_depth=s_depth,
            s_contrast=s_contrast,
        )
        v._set_depth_visible(False)
        views.append(v)

    state = {"set_idx": 0}

    def load_set(set_idx: int):
        set_idx = int(set_idx) % len(all_names)
        basename = all_names[set_idx]

        idxs = [idx1, idx2, idx3]

        for v, idx_map, folder in zip(views, idxs, folders):
            path = idx_map.get(basename)

            # Missing folder or file -> placeholder
            if path is None or not os.path.isfile(path):
                v.set_placeholder()
                v.s_contrast.set_val(1.0)
                _set_slider_range(v.s_depth, 0, 1, 0)

                v.im.set_data(np.zeros((10, 10), dtype=np.float32))
                v.im.set_clim(0.0, 1.0)

                msg = "(missing folder)" if not os.path.isdir(folder) else f"(missing file: {basename})"
                v.ax.set_title(f"{v.label}\n{v.source}\n{msg}", fontsize=10)
                continue

            arr = _load_array(path)

            if arr.ndim == 4:
                # (C,D,H,W)
                C, D, H, W = arr.shape
                if C == 3:
                    vol_rgb = np.transpose(arr[:3], (1, 2, 3, 0))  # (D,H,W,3)
                    v.set_volume_3d_rgb(vol_rgb)
                    d0 = v.D // 2
                    _set_slider_range(v.s_depth, 0, max(v.D - 1, 0), d0)
                    v.s_contrast.set_val(1.0)
                    v.render()
                else:
                    if not (0 <= channel < C):
                        raise ValueError(f"Channel {channel} out of range for {path}: C={C}")
                    vol = arr[channel]  # (D,H,W)
                    v.set_volume_3d_gray(vol)
                    d0 = v.D // 2
                    _set_slider_range(v.s_depth, 0, max(v.D - 1, 0), d0)
                    v.s_contrast.set_val(1.0)
                    v.render()

            elif arr.ndim == 3:
                # (C,H,W)
                C, H, W = arr.shape
                if C == 3:
                    img_rgb = np.transpose(arr[:3], (1, 2, 0))  # (H,W,3)
                    v.set_image_2d_rgb(img_rgb)
                    _set_slider_range(v.s_depth, 0, 1, 0)  # hidden
                    v.s_contrast.set_val(1.0)
                    v.render()
                else:
                    if not (0 <= channel < C):
                        raise ValueError(f"Channel {channel} out of range for {path}: C={C}")
                    img = arr[channel]  # (H,W)
                    v.set_image_2d_gray(img)
                    _set_slider_range(v.s_depth, 0, 1, 0)  # hidden
                    v.s_contrast.set_val(1.0)
                    v.render()
            else:
                raise ValueError(
                    f"Expected (C,D,H,W) or (C,H,W) in {path}, got shape {arr.shape}"
                )

        fig.suptitle(f"Set {set_idx + 1}/{len(all_names)}: {basename}", fontsize=12)
        fig.canvas.draw_idle()
        state["set_idx"] = set_idx

    def on_any_slider(_):
        for v in views:
            if v.mode == "none":
                continue
            v.render()
        fig.canvas.draw_idle()

    for v in views:
        v.s_depth.on_changed(on_any_slider)
        v.s_contrast.on_changed(on_any_slider)

    def on_scroll(event):
        # Scroll affects only the view whose image axes is under cursor, and only if it's 3D.
        for v in views:
            if event.inaxes == v.ax:
                if v.mode not in ("3d_gray", "3d_rgb") or v.D <= 0:
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

    load_set(0)

    import matplotlib.pyplot as plt
    plt.show(block=True)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize 3 matched arrays from 3 folders side-by-side. "
            "Supports (C,D,H,W) and (C,H,W). If C==3 => RGB, if C!=3 => grayscale channel."
        )
    )
    parser.add_argument("folder", help="Result Folder")
    parser.add_argument("--channel", type=int, default=0, help="Channel index for grayscale (default: 0)")
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap for grayscale (default: gray)")
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
