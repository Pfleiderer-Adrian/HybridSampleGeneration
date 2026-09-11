"""Reusable Matplotlib image-grid widget for study artifacts."""

from __future__ import annotations

from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from hybrid_sample_generator.visualization.rendering import ArrayCache, PanelSpec, render_panel


class ImageGrid(ttk.Frame):
    def __init__(
        self,
        master,
        cache: ArrayCache,
        *,
        rows: int,
        columns: int,
        on_depth_changed=None,
    ) -> None:
        super().__init__(master)
        self.cache = cache
        self.on_depth_changed = on_depth_changed
        self.specs: tuple[PanelSpec, ...] = ()
        self.slice_index = 0
        self.contrast = 1.0
        self.channel: str | int = "auto"
        self.show_mask = True
        self.mask_opacity = 0.45
        self._views = {}
        self._full_views = {}
        self._pan_axis = None

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5.2 * columns, 4.1 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        self.axes = tuple(axes.flatten())
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._scroll_callback = None
        navigation = ttk.Frame(self, padding=(4, 4))
        navigation.grid(row=1, column=0, sticky="ew")
        ttk.Button(navigation, text="Reset zoom", command=self.reset_view).pack(side="left")
        ttk.Label(
            navigation,
            text="Wheel: zoom · Drag: pan · Double-click: reset · Shift+wheel: slice",
            wraplength=600,
        ).pack(side="left", padx=8)

    def set_scroll_callback(self, callback) -> None:
        self._scroll_callback = callback

    def set_specs(self, specs) -> None:
        self._on_release(None)
        self._views.clear()
        self._full_views.clear()
        self.specs = tuple(specs)
        self.render()

    def render(self) -> None:
        self._on_release(None)
        depths = [1]
        statuses = []
        for index, axis in enumerate(self.axes):
            if index >= len(self.specs):
                axis.clear()
                axis.set_axis_off()
                continue
            depth, status = render_panel(
                axis,
                self.specs[index],
                self.cache,
                slice_index=self.slice_index,
                contrast=self.contrast,
                channel=self.channel,
                show_mask=self.show_mask,
                mask_opacity=self.mask_opacity,
            )
            self._full_views[axis] = (axis.get_xlim(), axis.get_ylim())
            if axis in self._views:
                xlim, ylim = self._views[axis]
                axis.set_xlim(xlim)
                axis.set_ylim(ylim)
            depths.append(depth)
            statuses.append(f"{self.specs[index].title}: {status}")
        max_depth = max(depths)
        self.slice_index = min(self.slice_index, max_depth - 1)
        if self.on_depth_changed:
            self.on_depth_changed(max_depth, " | ".join(statuses))
        self.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        axis = event.inaxes
        if axis not in self._full_views:
            return
        step = getattr(event, "step", 0)
        if not step:
            return
        if "shift" in (getattr(event, "key", None) or "").lower():
            if self._scroll_callback is not None:
                self._scroll_callback(1 if step > 0 else -1)
            return
        if event.xdata is None or event.ydata is None:
            return
        self._on_release(None)
        factor = 1.2 ** (-max(-10, min(10, step)))
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        full_xlim, _ = self._full_views[axis]
        relative_width = abs((xlim[1] - xlim[0]) / (full_xlim[1] - full_xlim[0]))
        factor = min(max(relative_width * factor, 0.001), 4.0) / relative_width
        axis.set_xlim(tuple(event.xdata + (x - event.xdata) * factor for x in xlim))
        axis.set_ylim(tuple(event.ydata + (y - event.ydata) * factor for y in ylim))
        self._remember_view(axis)

    def _on_press(self, event) -> None:
        axis = event.inaxes
        if axis not in self._full_views or event.button != 1:
            return
        self._on_release(None)
        if event.dblclick:
            self.reset_view(axis)
            return
        self._pan_axis = axis
        axis.start_pan(event.x, event.y, 1)

    def _on_motion(self, event) -> None:
        if self._pan_axis is not None and event.x is not None and event.y is not None:
            self._pan_axis.drag_pan(1, None, event.x, event.y)
            self._remember_view(self._pan_axis)

    def _on_release(self, _event) -> None:
        if self._pan_axis is not None:
            self._pan_axis.end_pan()
            self._pan_axis = None

    def _remember_view(self, axis) -> None:
        self._views[axis] = (axis.get_xlim(), axis.get_ylim())
        self.canvas.draw_idle()

    def reset_view(self, axis=None) -> None:
        self._on_release(None)
        axes = self.axes if axis is None else (axis,)
        for target in axes:
            self._views.pop(target, None)
            if target in self._full_views:
                xlim, ylim = self._full_views[target]
                target.set_xlim(xlim)
                target.set_ylim(ylim)
        self.canvas.draw_idle()

    def close(self) -> None:
        self._on_release(None)
        plt.close(self.figure)
