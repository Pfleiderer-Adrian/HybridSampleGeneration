"""Reusable base widget for browsing study entities and images."""

from __future__ import annotations

import json
import tkinter as tk
from dataclasses import asdict, is_dataclass
from tkinter import ttk

from hybrid_sample_generator.visualization.rendering import ArrayCache
from hybrid_sample_generator.visualization.widgets.image_grid import ImageGrid


class EntityBrowserTab(ttk.Frame):
    """Three-column base: selection, image grid and record details."""

    def __init__(
        self,
        master,
        *,
        cache: ArrayCache,
        rows: int,
        columns: int,
    ) -> None:
        super().__init__(master)
        self.cache = cache
        self.search_query = ""
        self._updating_slice = False

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.side = ttk.Frame(self, padding=8, width=300)
        self.side.grid(row=0, column=0, sticky="nsew")
        self.side.grid_propagate(False)

        self.content = ttk.Frame(self, padding=(0, 8, 0, 8))
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)
        self.image_grid = ImageGrid(
            self.content,
            cache,
            rows=rows,
            columns=columns,
            on_depth_changed=self._on_depth_changed,
        )
        self.image_grid.grid(row=0, column=0, sticky="nsew")
        self.image_grid.set_scroll_callback(self.change_slice)

        self.details_frame = ttk.Frame(self, padding=8, width=340)
        self.details_frame.grid(row=0, column=2, sticky="nsew")
        self.details_frame.grid_propagate(False)
        ttk.Label(self.details_frame, text="Information", font=("Arial", 10, "bold")).pack(
            anchor="w"
        )
        self.details = tk.Text(
            self.details_frame,
            width=42,
            wrap="word",
            relief="flat",
            padx=4,
            pady=6,
        )
        self.details.pack(fill="both", expand=True)
        self.details.configure(state="disabled")

        self.mask_var = tk.BooleanVar(value=True)
        self.opacity_var = tk.DoubleVar(value=0.45)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.channel_var = tk.StringVar(value="auto")
        self.slice_var = tk.DoubleVar(value=0)
        self.slice_label_var = tk.StringVar(value="Slice 1 / 1")
        self.status_var = tk.StringVar(value="")

    def build_display_controls(self, parent=None) -> None:
        parent = self.side if parent is None else parent
        ttk.Separator(parent).pack(fill="x", pady=8)
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Button(row, text="◀", width=4, command=self.previous).pack(side="left")
        ttk.Button(row, text="Next ▶", command=self.next).pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )

        ttk.Checkbutton(
            parent,
            text="Mask overlays",
            variable=self.mask_var,
            command=self.render,
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(parent, text="Mask opacity").pack(anchor="w", pady=(6, 0))
        ttk.Scale(
            parent,
            from_=0.05,
            to=0.9,
            variable=self.opacity_var,
            command=lambda _value: self.render(),
        ).pack(fill="x")

        channel_row = ttk.Frame(parent)
        channel_row.pack(fill="x", pady=(8, 0))
        ttk.Label(channel_row, text="Channel").pack(side="left")
        channel = ttk.Combobox(
            channel_row,
            textvariable=self.channel_var,
            values=("auto", "rgb", "0", "1", "2", "3"),
            state="readonly",
            width=8,
        )
        channel.pack(side="right")
        channel.bind("<<ComboboxSelected>>", lambda _event: self.render())

        contrast_row = ttk.Frame(parent)
        contrast_row.pack(fill="x", pady=(8, 0))
        ttk.Label(contrast_row, text="Contrast").pack(side="left")
        ttk.Button(contrast_row, text="Reset", command=self.reset_contrast).pack(
            side="right"
        )
        ttk.Scale(
            parent,
            from_=0.2,
            to=5.0,
            variable=self.contrast_var,
            command=lambda _value: self.render(),
        ).pack(fill="x")

        ttk.Label(parent, textvariable=self.slice_label_var).pack(
            anchor="w", pady=(8, 0)
        )
        self.slice_scale = ttk.Scale(
            parent,
            from_=0,
            to=0,
            variable=self.slice_var,
            command=self._slice_changed,
        )
        self.slice_scale.pack(fill="x")
        self.slice_scale.state(["disabled"])
        ttk.Label(
            parent,
            textvariable=self.status_var,
            wraplength=275,
            foreground="#555555",
        ).pack(anchor="w", fill="x", pady=(8, 0))

    def set_details(self, *sections: tuple[str, object]) -> None:
        chunks = []
        for title, value in sections:
            if value is None:
                continue
            if is_dataclass(value):
                value = asdict(value)
            chunks.append(title)
            chunks.append("=" * len(title))
            chunks.append(json.dumps(value, ensure_ascii=False, indent=2, default=str))
            chunks.append("")
        self.details.configure(state="normal")
        self.details.delete("1.0", tk.END)
        self.details.insert("1.0", "\n".join(chunks) or "No selection")
        self.details.configure(state="disabled")

    def render(self) -> None:
        self.image_grid.slice_index = int(round(self.slice_var.get()))
        self.image_grid.contrast = float(self.contrast_var.get())
        self.image_grid.channel = self.channel_var.get()
        self.image_grid.show_mask = bool(self.mask_var.get())
        self.image_grid.mask_opacity = float(self.opacity_var.get())
        self.image_grid.render()

    def change_slice(self, delta: int) -> None:
        maximum = int(float(self.slice_scale.cget("to")))
        new_value = min(
            max(int(round(self.slice_var.get())) + int(delta), 0), maximum
        )
        if new_value != int(round(self.slice_var.get())):
            self.slice_var.set(new_value)
            self.render()

    def reset_contrast(self) -> None:
        self.contrast_var.set(1.0)
        self.render()

    def _slice_changed(self, _value=None) -> None:
        if not self._updating_slice:
            self.render()

    def _on_depth_changed(self, depth: int, status: str) -> None:
        current = min(int(round(self.slice_var.get())), max(depth - 1, 0))
        self._updating_slice = True
        self.slice_scale.configure(to=max(depth - 1, 0))
        self.slice_var.set(current)
        if depth <= 1:
            self.slice_scale.state(["disabled"])
        else:
            self.slice_scale.state(["!disabled"])
        self.slice_label_var.set(f"Slice {current + 1} / {depth}")
        self.status_var.set(status)
        self._updating_slice = False

    def set_search(self, query: str) -> None:
        self.search_query = query.strip().lower()
        self.refresh()

    def previous(self) -> None:
        pass

    def next(self) -> None:
        pass

    def refresh(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.image_grid.close()
