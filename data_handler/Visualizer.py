import argparse
import numpy as np
import matplotlib


def _select_gui_backend(prefer: str = "tk") -> str:
    """
    Select and activate a Matplotlib GUI backend.

    Parameters
    ----------
    prefer : str, optional
        Preferred backend family:
        - "qt": try QtAgg first (requires a Qt binding such as PyQt6/PySide6/PyQt5/PySide2)
        - "tk": try TkAgg first (requires tkinter)
        Default is "tk" (most robust on Windows without extra installs).

    Returns
    -------
    str
        The name of the activated backend ("QtAgg" or "TkAgg").

    Raises
    ------
    RuntimeError
        If no GUI backend can be activated (neither Qt nor Tk is available).
    """
    prefer = (prefer or "").lower().strip()

    def try_qt() -> bool:
        try:
            matplotlib.use("QtAgg", force=True)
            # Validate that a Qt binding is actually importable
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
    else:  # prefer tk by default
        if try_tk():
            return "TkAgg"
        if try_qt():
            return "QtAgg"

    raise RuntimeError(
        "No Matplotlib GUI backend available.\n"
        "Install either:\n"
        "  - a Qt binding (PyQt6 / PySide6 / PyQt5 / PySide2), or\n"
        "  - tkinter (for TkAgg).\n"
        "Also consider using Python 3.11/3.12 on Windows for best wheel availability."
    )


def visualize_npy_volume(
    filepath: str,
    channel: int = 0,
    cmap: str = "gray",
    backend_preference: str = "tk",
):
    """
    Open a GUI window and interactively visualize a 3D volume stored in a .npy file.

    The input file must contain a 4D NumPy array of shape (C, D, H, W), where:
    - C: number of channels
    - D: depth (slice index you will scroll/slide through)
    - H: height
    - W: width

    The viewer displays a single channel as a stack of 2D slices (D, H, W).
    You can:
    - slide through depth (D) using a mouse-draggable slider
    - adjust contrast using a mouse-draggable slider (windowing around a robust percentile range)
    - scroll with the mouse wheel over the image to move through slices

    Parameters
    ----------
    filepath : str
        Path to the .npy file containing a NumPy array with shape (C, D, H, W).
    channel : int, optional
        Channel index to display (0 <= channel < C). Default is 0.
    cmap : str, optional
        Matplotlib colormap name used for rendering. Default is "gray".
    backend_preference : str, optional
        Which GUI backend family to prefer:
        - "tk" (default): TkAgg, usually works on Windows without extra packages
        - "qt": QtAgg, requires a Qt binding (PyQt/PySide)
        The function will fall back to the other if the preferred backend is unavailable.

    Returns
    -------
    matplotlib.figure.Figure
        The created Matplotlib figure instance.

    Raises
    ------
    ValueError
        If the loaded array does not have 4 dimensions (C, D, H, W),
        or if the channel index is out of range.
    RuntimeError
        If no GUI backend is available (neither Tk nor Qt can be loaded).
    """
    backend = _select_gui_backend(prefer=backend_preference)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    arr = np.load(filepath)

    if arr.ndim != 4:
        raise ValueError(f"Expected array with shape (C, D, H, W), got: {arr.shape}")

    C, D, H, W = arr.shape
    if not (0 <= channel < C):
        raise ValueError(f"channel must be in [0, {C - 1}], got: {channel}")

    # Display one channel as (D, H, W)
    vol = arr[channel].astype(np.float32, copy=False)

    # Robust contrast base window using percentiles across the whole volume
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.percentile(finite, [2, 98])
        if lo == hi:
            lo = float(finite.min())
            hi = float(finite.max() if finite.max() != finite.min() else finite.min() + 1.0)

    center = 0.5 * (float(lo) + float(hi))
    half0 = 0.5 * (float(hi) - float(lo))

    def window_limits(contrast: float) -> tuple[float, float]:
        """
        Compute (vmin, vmax) for imshow from a contrast factor.

        contrast > 1.0 -> narrower window (more contrast)
        contrast < 1.0 -> wider window (less contrast)
        """
        half = half0 / max(float(contrast), 1e-6)
        vmin = center - half
        vmax = center + half
        if vmin == vmax:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    d0 = D // 2
    contrast0 = 1.0
    vmin0, vmax0 = window_limits(contrast0)

    fig, ax = plt.subplots(figsize=(7, 7))
    try:
        fig.canvas.manager.set_window_title(f"NPY Volume Viewer ({backend})")
    except Exception:
        pass

    plt.subplots_adjust(bottom=0.22)

    im = ax.imshow(vol[d0], cmap=cmap, vmin=vmin0, vmax=vmax0)
    ax.set_title(f"Channel={channel} | Slice d={d0}/{D - 1} | Contrast={contrast0:.2f}")
    ax.set_axis_off()

    # Slider areas
    ax_depth = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_contrast = fig.add_axes([0.15, 0.07, 0.7, 0.03])

    s_depth = Slider(ax_depth, "Depth (D)", 0, D - 1, valinit=d0, valstep=1)
    s_contrast = Slider(ax_contrast, "Contrast", 0.2, 5.0, valinit=contrast0)

    def update(_):
        d = int(s_depth.val)
        contrast = float(s_contrast.val)
        vmin, vmax = window_limits(contrast)

        im.set_data(vol[d])
        im.set_clim(vmin, vmax)
        ax.set_title(f"Channel={channel} | Slice d={d}/{D - 1} | Contrast={contrast:.2f}")
        fig.canvas.draw_idle()

    s_depth.on_changed(update)
    s_contrast.on_changed(update)

    def on_scroll(event):
        # Only scroll when the mouse is over the image axes
        if event.inaxes != ax:
            return
        d = int(s_depth.val)
        step = 1 if event.button == "up" else -1
        s_depth.set_val(min(max(d + step, 0), D - 1))

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    plt.show(block=True)
    return fig


def main():
    """
    CLI entry point.

    Inputs
    ------
    - path (positional): Path to the .npy file (optional; if omitted, you will be prompted)
    - --channel: Channel index to display (default: 0)
    - --backend: Preferred GUI backend family: "tk" or "qt" (default: "tk")

    Output
    ------
    Opens an interactive GUI window for visualization.
    """
    parser = argparse.ArgumentParser(description="Interactive .npy volume viewer for arrays shaped (C, D, H, W).")
    parser.add_argument("path", nargs="?", help="Path to the .npy file")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to display (default: 0)")
    parser.add_argument("--backend", choices=["tk", "qt"], default="tk", help="Preferred GUI backend (default: tk)")
    args = parser.parse_args()

    path = args.path
    if not path:
        path = input("Enter .npy file path: ").strip().strip('"').strip("'")

    visualize_npy_volume(path, channel=args.channel, backend_preference=args.backend)


if __name__ == "__main__":
    main()
