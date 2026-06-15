import argparse
import os
import json
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from synthesizer.StudyPaths import StudyPaths


def _mousewheel_units(event) -> int:
    """Normalize Tk mouse wheel events across Windows, macOS, and Linux."""
    if getattr(event, "delta", 0):
        delta = event.delta
        if abs(delta) >= 120:
            return int(-1 * (delta / 120))
        return -1 if delta > 0 else 1
    if getattr(event, "num", None) == 4:
        return -1
    if getattr(event, "num", None) == 5:
        return 1
    return 0


def _is_descendant(widget, ancestor) -> bool:
    """Return True when widget is inside ancestor in the Tk widget tree."""
    while widget is not None:
        if widget == ancestor:
            return True
        widget = widget.master
    return False


def _paths(config) -> StudyPaths:
    """Return the StudyPaths instance exposed by a configuration object."""
    get_paths = getattr(config, "get_paths", None)
    if callable(get_paths):
        paths = get_paths()
    else:
        paths = getattr(config, "paths", None)
    if not isinstance(paths, StudyPaths):
        raise TypeError("Visualizer requires config.get_paths() to return a StudyPaths instance.")
    return paths


class OutlierGUI:
    def __init__(self, root, config, embedded: bool = False):
        self.root = root
        self.embedded = embedded
        if not self.embedded:
            self.root.title("Outlier Viewer")
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.config = config
        self.paths = _paths(config)

        self.synth_anomaly_dir = self.paths.synth_anomaly_data
        self.synth_roi_dir = self.paths.synth_roi_data
        self.anomaly_dir = self.paths.anomaly_data
        self.anomaly_roi_dir = self.paths.anomaly_roi_data
        self.ghs_dir = self.paths.generated_images_npy
        self.ghs_seg_dir = self.paths.generated_segmentations_npy
        self.anomaly_transformations = _load_anomaly_transformations(self.paths.anomaly_transformations_file)

        self.metric_stats = {}

        self.hierarchy = defaultdict(list)
        self.metric_map = self.build_metric_sample_map()

        self.filtered_hierarchy = {}
        self.sorted_controls = []
        self.flat_list = []

        self.current_index = 0
        self.current_slice = 0

        self._setup_responsive_sizes()

        self.build_ui()
        self.update_filter()
        self.root.focus_set()

    def _setup_responsive_sizes(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        base_width = 1920
        base_height = 1080
        tk_scale = 1.0
        try:
            tk_scale = float(self.root.tk.call('tk', 'scaling'))
        except Exception:
            tk_scale = 1.0

        scale_x = screen_width / (base_width * max(1.0, tk_scale))
        scale_y = screen_height / (base_height * max(1.0, tk_scale))
        scale = min(scale_x, scale_y)
        scale = max(0.7, min(1.5, scale))

        self.font_label_bold = ('Arial', max(9, int(10 * scale)), 'bold')
        self.font_label = ('Arial', max(8, int(10 * scale)))
        self.font_button = ('Arial', max(8, int(10 * scale)))
        self.font_small = ('Arial', max(7, int(8 * scale)), 'italic')
        self.font_info = ('Arial', max(8, int(10 * scale)))

        self.info_text_height = max(8, int(10 * scale))
        self.info_text_width = max(25, int(30 * scale))

    def _on_canvas_resize(self, event):
        if hasattr(self, 'axs'):
            fig_width = self.fig.get_figwidth()
            scale = max(0.8, min(1.5, fig_width / 12.0))

            for ax in self.axs:
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(max(8, int(10 * scale)))
                ax.title.set_fontsize(max(10, int(12 * scale)))

            self.fig.canvas.draw_idle()

    def _background_color(self):
        for widget in (self.root, getattr(self.root, "master", None)):
            if widget is None:
                continue
            for option in ("bg", "background"):
                try:
                    return widget.cget(option)
                except tk.TclError:
                    pass
        try:
            return ttk.Style(self.root).lookup("TFrame", "background") or "#f0f0f0"
        except Exception:
            return "#f0f0f0"

    def on_closing(self):
        plt.close('all')
        if not self.embedded:
            self.root.quit()
            self.root.destroy()

    def build_metric_sample_map(self):
        metric_map = defaultdict(lambda: defaultdict(dict))
        temp_values = defaultdict(list)
        anomaly_to_controls = defaultdict(list)

        if os.path.exists(self.synth_roi_dir):
            for control_name in os.listdir(self.synth_roi_dir):
                control_path = os.path.join(self.synth_roi_dir, control_name)
                if os.path.isdir(control_path):
                    anomalies = [f for f in os.listdir(control_path) if f.endswith('.npy')]
                    self.hierarchy[control_name] = anomalies
                    for a in anomalies:
                        anomaly_to_controls[a].append(control_name)

        csv_path = self.paths.metric_diffs_csv

        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    if row[0].lower() == "sample_name":
                        continue

                    sample_id = row[0]
                    try:
                        data_dict = json.loads(row[2])
                    except json.JSONDecodeError:
                        continue

                    is_roi = any(isinstance(v, dict) for v in data_dict.values())

                    if is_roi:
                        control_name = sample_id
                        if control_name not in self.hierarchy:
                            if control_name + '.png' in self.hierarchy:
                                control_name += '.png'
                            elif control_name + '.npy' in self.hierarchy:
                                control_name += '.npy'
                            else:
                                base = control_name.replace('.png', '').replace('.npy', '')
                                if base in self.hierarchy:
                                    control_name = base

                        for anomaly_name, metrics in data_dict.items():
                            for metric_name, val in metrics.items():
                                metric_map[metric_name][control_name][anomaly_name] = float(val)
                                temp_values[metric_name].append(float(val))

                                if anomaly_name not in self.hierarchy[control_name]:
                                    self.hierarchy[control_name].append(anomaly_name)
                                    anomaly_to_controls[anomaly_name].append(control_name)
                    else:
                        anomaly_name = sample_id

                        if anomaly_name not in anomaly_to_controls:
                            if anomaly_name + '.npy' in anomaly_to_controls:
                                anomaly_name += '.npy'
                            elif anomaly_name + '.png' in anomaly_to_controls:
                                anomaly_name += '.png'

                        associated_controls = anomaly_to_controls.get(anomaly_name, [])

                        for control_name in associated_controls:
                            for metric_name, val in data_dict.items():
                                metric_map[metric_name][control_name][anomaly_name] = float(val)
                                temp_values[metric_name].append(float(val))

        except FileNotFoundError:
            pass

        for metric, values in temp_values.items():
            if values:
                self.metric_stats[metric] = {'min': min(values), 'max': max(values)}

        return metric_map

    def build_ui(self):
        bg = self._background_color()
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_container)
        main_container.add(left_frame, weight=1)

        scrollbar = tk.Scrollbar(left_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = tk.Canvas(left_frame, yscrollcommand=scrollbar.set, bg=bg)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)


        control_frame = tk.Frame(canvas, bg=bg)
        canvas_window = canvas.create_window((0, 0), window=control_frame, anchor="nw")

        # Update scroll region when frame changes
        def on_frame_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())

        control_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        # Mouse wheel scrolling stays scoped to the left control panel.
        def on_mousewheel(event):
            widget = self.root.winfo_containing(event.x_root, event.y_root)
            if widget is None:
                widget = getattr(event, "widget", None)
            if widget is None or not _is_descendant(widget, left_frame):
                return
            widget_class = widget.winfo_class() if widget is not None else ""
            if widget_class in ("Treeview", "TTreeview", "Text"):
                return
            units = _mousewheel_units(event)
            if units:
                canvas.yview_scroll(units, "units")
                return "break"

        self.root.bind("<MouseWheel>", on_mousewheel, add="+")
        self.root.bind("<Button-4>", on_mousewheel, add="+")
        self.root.bind("<Button-5>", on_mousewheel, add="+")

        # Filter section
        tk.Label(control_frame, text="Filter & Sort by:", font=self.font_label_bold, bg=bg).pack(anchor="w", padx=5, pady=(5, 0))
        self.metric_vars = {}
        for metric in sorted(self.metric_map.keys()):
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(control_frame, text=metric, variable=var, command=self.update_filter, font=self.font_label, bg=bg)
            cb.pack(anchor="w", padx=10)
            self.metric_vars[metric] = var

        tk.Label(control_frame, text="Outlier Threshold (Top %):", font=self.font_label_bold, bg=bg).pack(anchor="w", padx=5, pady=(10, 0))
        self.outlier_slider = tk.Scale(control_frame, from_=0, to=10, resolution=.1, orient=tk.HORIZONTAL,
                                       command=lambda _: self.update_filter(), font=self.font_label, bg=bg)
        self.outlier_slider.set(1)
        self.outlier_slider.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Label(control_frame, text="Controls / Anomalies:", font=self.font_label_bold, bg=bg).pack(anchor="w", padx=5, pady=(10, 0))
        self.list_frame = tk.Frame(control_frame, height=150, bg=bg)
        self.list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.scrollbar_tree = tk.Scrollbar(self.list_frame)
        self.scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(self.list_frame, yscrollcommand=self.scrollbar_tree.set, selectmode="browse", height=8)
        self.tree.heading("#0", text="Items", anchor="w")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_tree.config(command=self.tree.yview)
        self.tree.bind('<<TreeviewSelect>>', self.on_treeview_select)

        contrast_header_frame = tk.Frame(control_frame, bg=bg)
        contrast_header_frame.pack(fill=tk.X, padx=5, pady=(10, 0))
        tk.Label(contrast_header_frame, text="Contrast:", font=self.font_label_bold, bg=bg).pack(side=tk.LEFT)
        tk.Button(contrast_header_frame, text="reset", command=self.reset_contrast, font=self.font_small,
                  relief=tk.FLAT, padx=2, pady=0, cursor="hand2").pack(side=tk.LEFT, padx=5)

        self.contrast_slider = tk.Scale(control_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                                        command=lambda _: self.update_display(), font=self.font_label, bg=bg)
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(fill=tk.X, pady=(0, 10))

        tk.Button(control_frame, text="Prev Sample (<-)", command=self.prev_sample).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Next Sample (->)", command=self.next_sample).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Slice - (Down)", command=self.prev_slice).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Slice + (Up)", command=self.next_slice).pack(fill=tk.X, pady=2)

        self.info_text = tk.Text(control_frame, height=10, width=30, bg=control_frame.cget("bg"), relief=tk.FLAT, font=("Arial", 10))
        self.info_text.pack(pady=10, fill=tk.BOTH, expand=True, anchor="w")
        self.info_text.tag_configure("active", foreground="black", font=("Arial", 10, "bold"))
        self.info_text.tag_configure("inactive", foreground="gray")
        self.info_text.tag_configure("header", font=("Arial", 10, "italic"))

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        self.del_btn = tk.Button(button_frame, text="DELETE", command=self.delete_current_sample,
                                 bg="#ffcccc", font=('Arial', 10, 'bold'), pady=5)
        self.del_btn.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 5), constrained_layout=True)
        self.axs = self.axs.flatten()
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        main_container.add(self.canvas_widget, weight=4)

        # Connect resize event for responsive figure scaling
        self.canvas.mpl_connect('resize_event', self._on_canvas_resize)

        self.canvas_widget.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas_widget.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas_widget.bind("<Button-5>", self.on_mouse_wheel)
        self.root.bind("<Left>", lambda e: self.prev_sample())
        self.root.bind("<Right>", lambda e: self.next_sample())
        self.root.bind("<Up>", lambda e: self.next_slice())
        self.root.bind("<Down>", lambda e: self.prev_slice())

    def update_filter(self):
        active_metrics = [m for m, v in self.metric_vars.items() if v.get()]
        threshold_pct = float(self.outlier_slider.get())

        self.filtered_hierarchy = defaultdict(list)
        control_scores = {}

        if not active_metrics:
            for m in self.metric_map:
                for c, a_dict in self.metric_map[m].items():
                    for a in a_dict:
                        if a not in self.filtered_hierarchy[c]:
                            self.filtered_hierarchy[c].append(a)
            for c in self.filtered_hierarchy:
                control_scores[c] = 0
        else:
            outlier_anomalies = []

            for m in active_metrics:
                all_vals = []
                for c_dict in self.metric_map[m].values():
                    all_vals.extend(c_dict.values())

                if not all_vals:
                    outlier_anomalies.append(set())
                    continue

                cutoff_percentile = max(0.0, 100.0 - threshold_pct)
                cutoff_value = np.percentile(all_vals, cutoff_percentile)

                m_outliers = set()
                for c, a_dict in self.metric_map[m].items():
                    for a, val in a_dict.items():
                        if val >= cutoff_value:
                            m_outliers.add((c, a))
                outlier_anomalies.append(m_outliers)

            intersection = set.intersection(*outlier_anomalies) if outlier_anomalies else set()

            anomaly_scores = {}
            for c, a in intersection:
                norm_sum = 0
                for m in active_metrics:
                    val = self.metric_map[m].get(c, {}).get(a, 0)
                    m_min, m_max = self.metric_stats[m]['min'], self.metric_stats[m]['max']
                    norm_val = (val - m_min) / (m_max - m_min) if m_max > m_min else 1.0
                    norm_sum += norm_val
                score = norm_sum / len(active_metrics)
                anomaly_scores[(c, a)] = score

            for (c, a), score in anomaly_scores.items():
                self.filtered_hierarchy[c].append(a)
                if c not in control_scores or score > control_scores[c]:
                    control_scores[c] = score

        self.sorted_controls = sorted(self.filtered_hierarchy.keys(), key=lambda x: control_scores.get(x, 0), reverse=True)

        self.flat_list = []
        for c in self.sorted_controls:
            self.flat_list.append(("control", c))
            self.filtered_hierarchy[c].sort()
            for a in self.filtered_hierarchy[c]:
                self.flat_list.append(("anomaly", c, a))

        if self.current_index >= len(self.flat_list):
            self.current_index = max(0, len(self.flat_list) - 1)

        self.update_treeview()
        self.update_display()

    def update_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.tree_item_mapping = {}

        flat_idx = 0
        for c in self.sorted_controls:
            parent_id = self.tree.insert("", tk.END, text=c, open=True)
            self.tree_item_mapping[flat_idx] = parent_id
            self.tree_item_mapping[parent_id] = flat_idx
            flat_idx += 1

            for a in self.filtered_hierarchy[c]:
                child_id = self.tree.insert(parent_id, tk.END, text=a)
                self.tree_item_mapping[flat_idx] = child_id
                self.tree_item_mapping[child_id] = flat_idx
                flat_idx += 1

        self._sync_treeview_selection()

    def _sync_treeview_selection(self):
        if self.flat_list and self.current_index in self.tree_item_mapping:
            item_id = self.tree_item_mapping[self.current_index]
            self.tree.selection_set(item_id)
            self.tree.focus(item_id)
            self.tree.see(item_id)

    def on_treeview_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        item_id = selection[0]

        if item_id in self.tree_item_mapping:
            new_idx = self.tree_item_mapping[item_id]
            if new_idx != self.current_index:
                self.current_index = new_idx
                self.current_slice = 0
                self.update_display()

    def reset_contrast(self):
        self.contrast_slider.set(1.0)
        self.update_display()

    def next_sample(self):
        if self.current_index < len(self.flat_list) - 1:
            self.current_index += 1
            self.current_slice = 0
            self._sync_treeview_selection()
            self.update_display()

    def prev_sample(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.current_slice = 0
            self._sync_treeview_selection()
            self.update_display()

    def next_slice(self):
        self.current_slice += 1
        self.update_display()

    def prev_slice(self):
        if self.current_slice > 0:
            self.current_slice -= 1
            self.update_display()

    def on_mouse_wheel(self, event):
        delta = getattr(event, "delta", 0) or 0
        num = getattr(event, "num", None)

        if num == 4 or delta > 0:
            self.next_slice()
        elif num == 5 or delta < 0:
            self.prev_slice()

    def _remove_if_exists(self, path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _remove_anomaly_from_hierarchy(self, control, anomaly):
        if control in self.hierarchy and anomaly in self.hierarchy[control]:
            self.hierarchy[control].remove(anomaly)
        if control in self.hierarchy and not self.hierarchy[control]:
            del self.hierarchy[control]

        for m in self.metric_map:
            if control in self.metric_map[m] and anomaly in self.metric_map[m][control]:
                del self.metric_map[m][control][anomaly]

    def _delete_files_for_anomaly(self, control, anomaly):
        targets = [
            os.path.join(self.synth_roi_dir, control, anomaly),
            os.path.join(self.anomaly_dir, anomaly),
            os.path.join(self.anomaly_roi_dir, anomaly),
            os.path.join(self.synth_anomaly_dir, anomaly)
        ]
        for path in targets:
            self._remove_if_exists(path)

        self._remove_anomaly_from_hierarchy(control, anomaly)

    def _delete_files_for_control(self, control):
        anomalies_to_delete = list(self.hierarchy.get(control, []))
        for a in anomalies_to_delete:
            roi_path = os.path.join(self.synth_roi_dir, control, a)
            self._remove_if_exists(roi_path)

            self._remove_anomaly_from_hierarchy(control, a)

        control_roi_dir = os.path.join(self.synth_roi_dir, control)
        if os.path.exists(control_roi_dir):
            try:
                os.rmdir(control_roi_dir)
            except OSError:
                pass

        targets = [
            os.path.join(self.ghs_dir, control),
            os.path.join(self.ghs_dir, control.replace('.png', '.npy')),
            os.path.join(self.ghs_seg_dir, control),
            os.path.join(self.ghs_seg_dir, control.replace('.png', '.npy'))
        ]
        for path in targets:
            self._remove_if_exists(path)

    def _show_anomaly_delete_dialog(self, control, anomaly):
        dialog = tk.Toplevel(self.root)
        dialog.title("Delete options")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=f"What data should be removed for '{anomaly}'?", font=('Arial', 10, 'bold')).pack(pady=10, padx=20)

        var_real = tk.BooleanVar(value=False)
        var_synth_roi = tk.BooleanVar(value=False)
        var_synth_anom = tk.BooleanVar(value=False)
        var_all = tk.BooleanVar(value=False)

        tk.Checkbutton(dialog, text="Real Anomaly (VAE input) + real ROI", variable=var_real).pack(anchor='w', padx=20)
        tk.Checkbutton(dialog, text="Synthetic ROI (just this fusion)", variable=var_synth_roi).pack(anchor='w', padx=20)
        tk.Checkbutton(dialog, text="Synthetic Anomaly + all its ROIs (may affect other fusions)", variable=var_synth_anom).pack(anchor='w', padx=20)
        tk.Checkbutton(dialog, text="Hybrid Sample + all ROIs inside", variable=var_all).pack(anchor='w', padx=20, pady=(10, 0))

        def execute_delete():
            deleted_anything = False

            if var_all.get():
                self._delete_files_for_control(control)
                deleted_anything = True

            if var_real.get():
                self._remove_if_exists(os.path.join(self.anomaly_dir, anomaly))
                self._remove_if_exists(os.path.join(self.anomaly_roi_dir, anomaly))
                deleted_anything = True

            if var_synth_roi.get():
                self._remove_if_exists(os.path.join(self.synth_roi_dir, control, anomaly))
                deleted_anything = True

            if var_synth_anom.get():
                self._remove_if_exists(os.path.join(self.synth_anomaly_dir, anomaly))
                controls_to_check = list(self.hierarchy.keys())
                for c in controls_to_check:
                    if anomaly in self.hierarchy.get(c, []):
                        roi_path = os.path.join(self.synth_roi_dir, c, anomaly)
                        self._remove_if_exists(roi_path)
                        self._remove_anomaly_from_hierarchy(c, anomaly)
                deleted_anything = True

            if deleted_anything:
                self._remove_anomaly_from_hierarchy(control, anomaly)

            dialog.destroy()

            if deleted_anything:
                self.update_filter()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="Delete", bg="#ffcccc", font=('Arial', 10, 'bold'), command=execute_delete).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", font=('Arial', 10), command=dialog.destroy).pack(side=tk.LEFT)

        self.root.wait_window(dialog)

    def delete_current_sample(self):
        if not self.flat_list:
            return
        item = self.flat_list[self.current_index]

        if item[0] == "control":
            control = item[1]
            if not messagebox.askyesno("Delete Control", f"Do you want to delete the Hybrid Sample '{control}' with all its ROIs?"):
                return
            self._delete_files_for_control(control)
            self.update_filter()
        else:
            _, control, anomaly = item
            self._show_anomaly_delete_dialog(control, anomaly)

    def _get_fallback_path(self, base_dir, filename):
        p = os.path.join(base_dir, filename)
        if os.path.exists(p):
            return p
        p_npy = os.path.join(base_dir, filename.replace('.png', '.npy'))
        if os.path.exists(p_npy):
            return p_npy
        p_append = p + '.npy'
        if os.path.exists(p_append):
            return p_append
        return p

    def update_display(self):
        for ax in self.axs:
            ax.clear()
            ax.axis("off")

        if not self.flat_list:
            self.axs[0].set_title("No samples found")
            self.canvas.draw()
            return

        item = self.flat_list[self.current_index]
        contrast = float(self.contrast_slider.get())

        if item[0] == "control":
            control = item[1]
            self.fig.suptitle(f"Control: {control}", fontsize=12, fontweight='bold')

            ghs_path = self._get_fallback_path(self.ghs_dir, control)
            ghs_seg_path = self._get_fallback_path(self.ghs_seg_dir, control)

            paths = [
                (ghs_path, "Generated Hybrid Sample", None, None),
                (ghs_seg_path, "Generated Hybrid Segmentation", None, None),
                (None, "", None, None),
                (None, "", None, None)
            ]
        else:
            _, control, anomaly = item
            self.fig.suptitle(f"{anomaly} in {control}", fontsize=12, fontweight='bold')

            anomaly_meta = _get_anomaly_meta(self.anomaly_transformations, anomaly)
            synth_roi_path = os.path.join(self.synth_roi_dir, control, anomaly)
            real_roi_path = os.path.join(self.anomaly_roi_dir, anomaly)
            paths = [
                (os.path.join(self.synth_anomaly_dir, anomaly), "synth_anomaly_data", anomaly_meta, None),
                (synth_roi_path, "synth_roi_data", None, None),
                (os.path.join(self.anomaly_dir, anomaly), "anomaly_data", anomaly_meta, real_roi_path),
                (real_roi_path, "anomaly_roi_data", None, None)
            ]

        loaded_data = []
        max_slices = 0
        for p, title, meta, window_path in paths:
            if p and os.path.exists(p):
                try:
                    arr = np.load(p)
                    arr, denormalized = _denormalize_array_for_display(arr, meta)
                    window_arr = arr
                    if denormalized and window_path and os.path.exists(window_path):
                        window_arr = np.load(window_path)
                    img, depth, curr_slice, _mode = _display_plane(
                        arr, slice_index=self.current_slice, channel=0
                    )
                    max_slices = max(max_slices, depth)
                    display_title = f"{title}\nSlice {curr_slice}" if depth > 1 else title
                    loaded_data.append((img, display_title, p, None, window_arr))
                except Exception as exc:
                    loaded_data.append((None, title, p, str(exc), None))
            else:
                loaded_data.append((None, title, p, None, None))

        if self.current_slice >= max_slices and max_slices > 0:
            self.current_slice = max_slices - 1

        for i, (img, title, path, error, window_arr) in enumerate(loaded_data):
            if not title:
                continue

            if img is None:
                if error:
                    self.axs[i].text(
                        0.5, 0.5, f"Could not load:\n{path}\n\n{error}",
                        ha="center", va="center", transform=self.axs[i].transAxes,
                        fontsize=8, wrap=True,
                    )
                else:
                    self.axs[i].text(
                        0.5, 0.5, f"Not found:\n{path}",
                        ha="center", va="center", transform=self.axs[i].transAxes,
                        fontsize=8, wrap=True,
                    )
                self.axs[i].set_title(title, fontsize=9)
                continue

            img_display = _normalize_for_display(img, window_arr if window_arr is not None else img, contrast)

            self.axs[i].set_title(title, fontsize=10, pad=12)

            if img_display.ndim == 3 and img_display.shape[-1] == 1:
                self.axs[i].imshow(img_display[:, :, 0], cmap="gray", vmin=0, vmax=1, aspect='equal')
            elif img_display.ndim == 3:
                self.axs[i].imshow(img_display, aspect='equal')
            else:
                self.axs[i].imshow(img_display, cmap="gray", vmin=0, vmax=1, aspect='equal')

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert(tk.END, f"Selected: {self.current_index+1} / {len(self.flat_list)}\n", "header")
        self.info_text.insert(tk.END, "=" * 30 + "\n\n", "header")

        active = [m for m, v in self.metric_vars.items() if v.get()]

        if item[0] == "control":
            self.info_text.insert(tk.END, f"Anomalies: {len(self.filtered_hierarchy.get(item[1], []))}\n\n", "active")
        else:
            control, anomaly = item[1], item[2]
            self.info_text.insert(tk.END, f"Control:\n{control}\n\n", "active")
            self.info_text.insert(tk.END, f"Anomaly:\n{anomaly}\n\n", "active")
            self.info_text.insert(tk.END, "Metrics:\n", "header")
            self.info_text.insert(tk.END, "-" * 20 + "\n", "header")

            for m in sorted(self.metric_map.keys()):
                if control in self.metric_map[m] and anomaly in self.metric_map[m][control]:
                    val = self.metric_map[m][control][anomaly]
                    line = f"{m}:\n  {val:.4f}\n\n"
                    self.info_text.insert(tk.END, line, "active" if m in active else "inactive")

        self.info_text.config(state=tk.DISABLED)
        self.canvas.draw()

def run_outlier_gui(config):
    root = tk.Tk()
    app = OutlierGUI(root, config)
    root.mainloop()

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


def _load_anomaly_transformations(transformations_file: str) -> Dict[str, dict]:
    if not os.path.isfile(transformations_file):
        return {}
    try:
        with open(transformations_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_anomaly_meta(transformations: Dict[str, dict], filename: str) -> Optional[dict]:
    base = os.path.basename(str(filename or ""))
    stem, _ = os.path.splitext(base)
    for key in (base, stem, stem + ".npy"):
        meta = transformations.get(key)
        if isinstance(meta, dict):
            return meta
    return None


def _denormalize_array_for_display(arr: np.ndarray, meta: Optional[dict]) -> Tuple[np.ndarray, bool]:
    if not meta:
        return arr, False

    norm_type = meta.get("norm_type")
    if norm_type == "zscore":
        mean = meta.get("norm_mean")
        std = meta.get("norm_std")
        if mean is None or std is None:
            return arr, False
        return arr.astype(np.float32, copy=False) * float(std) + float(mean), True

    if norm_type == "zscore_median":
        median = meta.get("norm_median")
        mad = meta.get("norm_mad")
        if median is None or mad is None:
            return arr, False
        return arr.astype(np.float32, copy=False) * float(mad) + float(median), True

    if norm_type is None:
        return arr, True

    return arr, False


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


def visualize_folders(
    folders: Sequence[str],
    channel: int = 0,
    cmap: str = "gray",
    backend_preference: str = "tk",
    exts: Sequence[str] = (".npy",),
    labels: Optional[Sequence[str]] = None,
    window_title: str = "Array Set Viewer",
):
    """
    GUI: show N matched arrays side-by-side (one per folder) with per-view sliders.

    Supported shapes (per file; can differ between folders):
      - (C,D,H,W): if C==3 => RGB slices, else => grayscale of selected --channel
      - (C,H,W):   if C==3 => RGB,        else => grayscale of selected --channel

    Depth slider is shown ONLY for (C,D,H,W) cases.

    Notes:
      - Files are matched by basename (filename without extension) across all folders.
      - Missing files/folders are displayed as placeholders (per view).
    """

    if not folders:
        raise ValueError("folders must contain at least one folder path.")

    _select_gui_backend(prefer=backend_preference)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    exts_n = _normalize_exts(exts)

    idx_maps = [_index_folder(f, exts_n) for f in folders]
    all_names = sorted(set().union(*[set(m.keys()) for m in idx_maps]))
    if not all_names:
        all_names = ["(no files found)"]

    n = len(folders)
    if labels is None or len(labels) != n:
        labels = [f"Folder {i+1}" for i in range(n)]

    sources = [_folder_tag(f) for f in folders]

    fig, axs = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axs = [axs]
    try:
        fig.canvas.manager.set_window_title(window_title)
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

        for v, idx_map, folder in zip(views, idx_maps, folders):
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

    plt.show(block=True)
    return fig


def visualize_four_folders(
    folder1: str,
    folder2: str,
    folder3: str,
    folder4: str,
    channel: int = 0,
    cmap: str = "gray",
    backend_preference: str = "tk",
    exts: Sequence[str] = (".npy",),
    labels: Optional[Sequence[str]] = (
        "ROI cutout Area",
        "Real Anomaly cutout",
        "Generated Synth. Anomaly",
        "Fusioned Hybrid Sample",
    ),
):
    """
    GUI: show 4 matched arrays side-by-side (one per folder) with per-view sliders.

    Supported shapes:
      - (C,D,H,W): if C==3 => RGB slices, else => grayscale of selected --channel
      - (C,H,W):   if C==3 => RGB,        else => grayscale of selected --channel

    Depth slider is shown ONLY for (C,D,H,W) cases.
    """
    return visualize_folders(
        folders=[folder1, folder2, folder3, folder4],
        channel=channel,
        cmap=cmap,
        backend_preference=backend_preference,
        exts=exts,
        labels=labels,
        window_title="4-Array Set Viewer",
    )


def _list_npy_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        name for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(".npy")
    )


def _build_file_list(parent, height: int, on_select):
    """Create the shared file list used by the array visualizer tabs."""
    list_frame = ttk.Frame(parent)
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
    file_list = tk.Listbox(list_frame, width=34, height=height, exportselection=False)
    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=file_list.yview)
    file_list.configure(yscrollcommand=scrollbar.set)
    file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    file_list.bind("<<ListboxSelect>>", on_select)
    return file_list


def _add_unique(items: List[str], value: str):
    if value and value not in items:
        items.append(value)


def _name_candidates(filename: str, extensions: Sequence[str] = (".npy",)) -> List[str]:
    base = os.path.basename(str(filename or ""))
    stem, ext = os.path.splitext(base)
    candidates: List[str] = []
    _add_unique(candidates, base)
    if stem:
        for extension in extensions:
            extension = extension if extension.startswith(".") else "." + extension
            _add_unique(candidates, stem + extension)
        _add_unique(candidates, stem)
        for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            _add_unique(candidates, stem + extension)
    return candidates


def _resolve_file_by_name(folder: str, filename: str, extensions: Sequence[str] = (".npy",)) -> Tuple[Optional[str], str]:
    base = os.path.basename(str(filename or ""))
    stem, ext = os.path.splitext(base)
    expected_name = base if ext else stem + (extensions[0] if extensions else "")
    expected = os.path.join(folder, expected_name)

    if not os.path.isdir(folder):
        return None, expected

    for candidate in _name_candidates(filename, extensions):
        path = os.path.join(folder, candidate)
        if os.path.isfile(path):
            return path, expected

    target_stem = stem or os.path.splitext(expected_name)[0]
    for name in _list_npy_files(folder):
        if os.path.splitext(name)[0] == target_stem:
            return os.path.join(folder, name), expected
    return None, expected


def _resolve_dir_by_name(folder: str, dirname: str) -> Tuple[Optional[str], str]:
    base = os.path.basename(str(dirname or ""))
    expected = os.path.join(folder, base)
    if not os.path.isdir(folder):
        return None, expected

    for candidate in _name_candidates(base, extensions=(".npy",)):
        path = os.path.join(folder, candidate)
        if os.path.isdir(path):
            return path, expected

    target_stem = os.path.splitext(base)[0]
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isdir(path) and os.path.splitext(name)[0] == target_stem:
            return path, expected
    return None, expected


def _is_channel_axis(length: int) -> bool:
    return 1 <= int(length) <= 8


def _chw_to_image(frame: np.ndarray, channel: int = 0) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame
    channels = int(frame.shape[0])
    if channels in (3, 4):
        return np.moveaxis(frame[:3], 0, -1)
    channel = max(0, min(int(channel), channels - 1))
    return frame[channel]


def _hwc_to_image(frame: np.ndarray, channel: int = 0) -> np.ndarray:
    frame = np.asarray(frame)
    channels = int(frame.shape[-1])
    if channels in (3, 4):
        return frame[..., :3]
    channel = max(0, min(int(channel), channels - 1))
    return frame[..., channel]


def _display_plane(arr: np.ndarray, slice_index: int = 0, channel: int = 0) -> Tuple[np.ndarray, int, int, str]:
    arr = np.asarray(arr)
    slice_index = max(0, int(slice_index))

    if arr.ndim == 4:
        if _is_channel_axis(arr.shape[0]):
            depth = int(arr.shape[1])
            used = min(slice_index, max(depth - 1, 0))
            return _chw_to_image(arr[:, used, :, :], channel), depth, used, "C,D,H,W"
        if _is_channel_axis(arr.shape[-1]):
            depth = int(arr.shape[0])
            used = min(slice_index, max(depth - 1, 0))
            return _hwc_to_image(arr[used], channel), depth, used, "D,H,W,C"
        raise ValueError(f"Unsupported 4D shape {arr.shape}; expected channel-first or channel-last data.")

    if arr.ndim == 3:
        if _is_channel_axis(arr.shape[0]) and arr.shape[1] > 8 and arr.shape[2] > 8:
            return _chw_to_image(arr, channel), 1, 0, "C,H,W"
        if _is_channel_axis(arr.shape[-1]) and arr.shape[0] > 8 and arr.shape[1] > 8:
            return _hwc_to_image(arr, channel), 1, 0, "H,W,C"
        depth = int(arr.shape[0])
        used = min(slice_index, max(depth - 1, 0))
        return arr[used], depth, used, "D,H,W"

    if arr.ndim == 2:
        return arr, 1, 0, "H,W"

    raise ValueError(f"Unsupported shape {arr.shape}; expected 2D, 3D, or 4D array.")


def _display_mask_plane(arr: np.ndarray, slice_index: int = 0) -> Tuple[np.ndarray, int, int, str]:
    arr = np.asarray(arr)
    slice_index = max(0, int(slice_index))

    if arr.ndim == 4:
        if _is_channel_axis(arr.shape[0]):
            depth = int(arr.shape[1])
            used = min(slice_index, max(depth - 1, 0))
            return np.max(arr[:, used, :, :], axis=0), depth, used, "C,D,H,W"
        if _is_channel_axis(arr.shape[-1]):
            depth = int(arr.shape[0])
            used = min(slice_index, max(depth - 1, 0))
            return np.max(arr[used], axis=-1), depth, used, "D,H,W,C"
        raise ValueError(f"Unsupported 4D mask shape {arr.shape}; expected channel-first or channel-last data.")

    if arr.ndim == 3:
        if _is_channel_axis(arr.shape[0]) and arr.shape[1] > 8 and arr.shape[2] > 8:
            return np.max(arr, axis=0), 1, 0, "C,H,W"
        if _is_channel_axis(arr.shape[-1]) and arr.shape[0] > 8 and arr.shape[1] > 8:
            return np.max(arr, axis=-1), 1, 0, "H,W,C"
        depth = int(arr.shape[0])
        used = min(slice_index, max(depth - 1, 0))
        return arr[used], depth, used, "D,H,W"

    if arr.ndim == 2:
        return arr, 1, 0, "H,W"

    raise ValueError(f"Unsupported mask shape {arr.shape}; expected 2D, 3D, or 4D array.")


def _normalize_for_display(image: np.ndarray, reference: np.ndarray, contrast: float) -> np.ndarray:
    center, half0 = _robust_window_params(np.asarray(reference))
    vmin, vmax = _window_limits(center, half0, contrast)
    denom = (vmax - vmin) if vmax != vmin else 1.0
    out = (np.asarray(image, dtype=np.float32) - vmin) / denom
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


def _mask_cmap_and_norm(label_count: int):
    base_colors = [
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
    label_count = max(int(label_count), 1)
    if label_count <= len(base_colors):
        colors = base_colors[:label_count]
    else:
        tab20 = plt.get_cmap("tab20")
        colors = base_colors + [
            matplotlib.colors.to_hex(tab20(i % tab20.N))
            for i in range(label_count - len(base_colors))
        ]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(
        np.arange(-0.5, label_count + 0.5, 1.0),
        label_count,
    )
    return cmap, norm


def _prepare_mask_display(mask: np.ndarray) -> Tuple[np.ndarray, str]:
    raw = np.asarray(mask)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    labels_raw = np.rint(raw).astype(np.int64, copy=False)
    labels_raw = np.where(labels_raw > 0, labels_raw, 0)

    labels = np.unique(labels_raw)
    positive_labels = [int(value) for value in labels if value > 0]

    display = np.zeros(labels_raw.shape, dtype=np.int32)
    for idx, label in enumerate(positive_labels, start=1):
        display[labels_raw == label] = idx

    label_text_values = [int(value) for value in labels[:8]]
    label_suffix = "..." if labels.size > 8 else ""
    fg_pct = 100.0 * (float(np.count_nonzero(labels_raw > 0)) / max(labels_raw.size, 1))
    stats = f"labels={label_text_values}{label_suffix} | fg={fg_pct:.1f}%"
    return display, stats


def _label_count_for_display(image: np.ndarray) -> int:
    if image.size == 0:
        return 1
    return int(np.nanmax(image)) + 1


def _draw_placeholder(ax, title: str, detail: str):
    ax.clear()
    ax.axis("off")
    ax.imshow(np.full((12, 12), 0.94, dtype=np.float32), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10, pad=8)
    ax.text(
        0.5, 0.5, detail,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=8, color="#444444", wrap=True,
    )


def _array_item(
    title: str,
    path: Optional[str],
    expected_path: Optional[str] = None,
    meta: Optional[dict] = None,
    window_path: Optional[str] = None,
) -> Dict[str, object]:
    """Build a display item while keeping optional metadata out of the common path."""
    item: Dict[str, object] = {
        "title": title,
        "path": path,
        "expected_path": expected_path if expected_path is not None else path,
        "is_mask": "mask" in title.lower(),
    }
    if meta is not None:
        item["meta"] = meta
    if window_path is not None:
        item["window_path"] = window_path
    return item


def _render_array_panel(ax, item: Dict[str, object], slice_index: int, contrast: float,
                        channel: int = 0, cmap: str = "gray") -> int:
    title = str(item.get("title") or "")
    path = item.get("path")
    expected_path = item.get("expected_path") or path or ""

    if not path or not os.path.isfile(path):
        _draw_placeholder(ax, title, f"Not found:\n{expected_path}")
        return 1

    try:
        arr = _load_array(path)
        is_mask = bool(item.get("is_mask"))
        mask_stats = ""
        if item.get("is_mask"):
            image, depth, used_slice, mode = _display_mask_plane(arr, slice_index=slice_index)
            image, mask_stats = _prepare_mask_display(image)
        else:
            arr, denormalized = _denormalize_array_for_display(arr, item.get("meta"))
            window_reference = arr
            window_path = item.get("window_path")
            if denormalized and window_path and os.path.isfile(window_path):
                window_reference = _load_array(window_path)
            image, depth, used_slice, mode = _display_plane(arr, slice_index=slice_index, channel=channel)
            image = _normalize_for_display(image, window_reference, contrast)
    except Exception as exc:
        _draw_placeholder(ax, title, f"Could not load:\n{path}\n\n{exc}")
        return 1

    ax.clear()
    ax.axis("off")
    if is_mask:
        mask_cmap, mask_norm = _mask_cmap_and_norm(_label_count_for_display(image))
        ax.imshow(image, cmap=mask_cmap, norm=mask_norm, interpolation="nearest", aspect="equal")
    elif image.ndim == 3 and image.shape[-1] == 1:
        ax.imshow(image[:, :, 0], cmap=cmap, vmin=0, vmax=1, aspect="equal")
    elif image.ndim == 3:
        ax.imshow(image, aspect="equal")
    else:
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    slice_info = f" | slice {used_slice}/{depth - 1}" if depth > 1 else ""
    mask_info = f" | {mask_stats}" if mask_stats else ""
    ax.set_title(
        f"{title}\n{os.path.basename(path)} | shape={tuple(arr.shape)} | {mode}{slice_info}{mask_info}",
        fontsize=9,
        pad=8,
    )
    return max(int(depth), 1)


class _ArrayTabBase(ttk.Frame):
    def __init__(self, master, config, channel: int = 0, cmap: str = "gray"):
        super().__init__(master)
        self.config = config
        self.paths = _paths(config)
        self.anomaly_transformations = _load_anomaly_transformations(self.paths.anomaly_transformations_file)
        self.channel = int(channel)
        self.cmap = cmap
        self.slice_var = tk.DoubleVar(value=0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.status_var = tk.StringVar(value="")
        self._updating_controls = False

    def _build_controls(self, parent):
        ttk.Label(parent, text="Slice").pack(anchor="w", pady=(12, 0))
        self.slice_scale = ttk.Scale(parent, from_=0, to=0, variable=self.slice_var, command=self._on_slice_changed)
        self.slice_scale.pack(fill=tk.X)
        self.slice_label = ttk.Label(parent, text="0 / 0")
        self.slice_label.pack(anchor="w")

        ttk.Label(parent, text="Contrast").pack(anchor="w", pady=(12, 0))
        self.contrast_scale = ttk.Scale(
            parent, from_=0.2, to=5.0, variable=self.contrast_var, command=self._on_contrast_changed
        )
        self.contrast_scale.pack(fill=tk.X)
        ttk.Label(parent, textvariable=self.status_var, wraplength=260, foreground="#555555").pack(
            anchor="w", fill=tk.X, pady=(12, 0)
        )

    def _refresh_file_list(self, folder: str):
        """Reload the primary .npy list and select the first available file."""
        self.files = _list_npy_files(folder)
        self.file_list.delete(0, tk.END)
        for name in self.files:
            self.file_list.insert(tk.END, name)

        if self.files:
            self.file_list.selection_set(0)
            self.current_filename = self.files[0]
            self.status_var.set(f"{len(self.files)} files")
        else:
            self.current_filename = None
            self.status_var.set(f"No .npy files found in: {folder}")

    def _render_items(self, axes, items: Sequence[Dict[str, object]]) -> int:
        """Render a group of array panels and return the largest slice count."""
        slice_index = int(round(self.slice_var.get()))
        contrast = float(self.contrast_var.get())
        max_slices = 1
        for ax, item in zip(axes, items):
            max_slices = max(
                max_slices,
                _render_array_panel(
                    ax, item, slice_index, contrast,
                    channel=self.channel, cmap=self.cmap,
                ),
            )
        return max_slices

    def _on_slice_changed(self, _value=None):
        if not self._updating_controls:
            self.render()

    def _on_contrast_changed(self, _value=None):
        if not self._updating_controls:
            self.render()

    def _sync_slice_control(self, max_slices: int):
        max_index = max(int(max_slices) - 1, 0)
        current = max(0, min(int(round(self.slice_var.get())), max_index))
        self._updating_controls = True
        self.slice_scale.configure(to=max_index)
        self.slice_var.set(current)
        self.slice_label.configure(text=f"{current} / {max_index}")
        if max_index == 0:
            self.slice_scale.state(["disabled"])
        else:
            self.slice_scale.state(["!disabled"])
        self._updating_controls = False

    def _on_scroll(self, event):
        step = 1 if getattr(event, "button", "") == "up" else -1
        max_index = int(float(self.slice_scale.cget("to")))
        new_value = max(0, min(int(round(self.slice_var.get())) + step, max_index))
        if new_value != int(round(self.slice_var.get())):
            self.slice_var.set(new_value)
            self.render()

    def render(self):
        raise NotImplementedError


class AnomalyGenerationTab(_ArrayTabBase):
    def __init__(self, master, config, channel: int = 0, cmap: str = "gray"):
        super().__init__(master, config, channel=channel, cmap=cmap)
        self.anomaly_dir = self.paths.anomaly_data
        self.anomaly_roi_dir = self.paths.anomaly_roi_data
        self.synth_anomaly_dir = self.paths.synth_anomaly_data
        self.generated_mask_dir = self.paths.anomaly_tgt_mask_data
        self.files: List[str] = []
        self.current_filename: Optional[str] = None
        self._build_ui()
        self.refresh_files()

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side = ttk.Frame(self, padding=8)
        side.grid(row=0, column=0, sticky="ns")
        ttk.Label(side, text="anomaly_data", font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(side, text=self.anomaly_dir, wraplength=260).pack(anchor="w", fill=tk.X)

        self.file_list = _build_file_list(side, height=22, on_select=self._on_file_selected)

        ttk.Button(side, text="Refresh", command=self.refresh_files).pack(fill=tk.X, pady=(8, 0))
        self._build_controls(side)

        content = ttk.Frame(self, padding=(4, 8, 8, 8))
        content.grid(row=0, column=1, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        self.axs = self.axs.flatten()
        self.canvas = FigureCanvasTkAgg(self.fig, master=content)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def refresh_files(self):
        self._refresh_file_list(self.anomaly_dir)
        self.render()

    def _on_file_selected(self, _event=None):
        selection = self.file_list.curselection()
        if not selection:
            return
        self.current_filename = self.files[selection[0]]
        self.slice_var.set(0)
        self.render()

    def render(self):
        selected = self.current_filename
        if selected is None:
            items = [
                _array_item("Extracted anomaly", None, self.anomaly_dir),
                _array_item("Extracted ROI", None, self.anomaly_roi_dir),
                _array_item("Generated anomaly", None, self.synth_anomaly_dir),
                _array_item("Generated mask", None, self.generated_mask_dir),
            ]
            self.fig.suptitle("anomaly generation", fontsize=13, fontweight="bold")
        else:
            anomaly_path, anomaly_expected = _resolve_file_by_name(self.anomaly_dir, selected)
            roi_path, roi_expected = _resolve_file_by_name(self.anomaly_roi_dir, selected)
            synth_path, synth_expected = _resolve_file_by_name(self.synth_anomaly_dir, selected)
            mask_path, mask_expected = _resolve_file_by_name(self.generated_mask_dir, selected)
            anomaly_meta = _get_anomaly_meta(self.anomaly_transformations, selected)
            items = [
                _array_item(
                    "Extracted anomaly", anomaly_path, anomaly_expected,
                    meta=anomaly_meta, window_path=roi_path,
                ),
                _array_item("Extracted ROI", roi_path, roi_expected),
                _array_item(
                    "Generated anomaly", synth_path, synth_expected,
                    meta=anomaly_meta, window_path=roi_path,
                ),
                _array_item("Generated mask", mask_path, mask_expected),
            ]
            self.fig.suptitle(f"anomaly generation: {selected}", fontsize=13, fontweight="bold")

        max_slices = self._render_items(self.axs, items)
        self._sync_slice_control(max_slices)
        self.canvas.draw_idle()


class FusedAnomalyTab(_ArrayTabBase):
    def __init__(self, master, config, channel: int = 0, cmap: str = "gray"):
        super().__init__(master, config, channel=channel, cmap=cmap)
        self.hybrid_images_dir = self.paths.generated_images_npy
        self.hybrid_segmentations_dir = self.paths.generated_segmentations_npy
        self.synth_roi_dir = self.paths.synth_roi_data
        self.anomaly_dir = self.paths.anomaly_data
        self.anomaly_roi_dir = self.paths.anomaly_roi_data
        self.files: List[str] = []
        self.roi_files: List[str] = []
        self.current_filename: Optional[str] = None
        self.current_roi_dir: Optional[str] = None
        self.expected_roi_dir: str = os.path.join(self.synth_roi_dir, "")
        self.roi_var = tk.StringVar(value="")
        self._build_ui()
        self.refresh_files()

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side = ttk.Frame(self, padding=8)
        side.grid(row=0, column=0, sticky="ns")
        ttk.Label(side, text="images_npy", font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(side, text=self.hybrid_images_dir, wraplength=260).pack(anchor="w", fill=tk.X)

        self.file_list = _build_file_list(side, height=18, on_select=self._on_file_selected)

        ttk.Button(side, text="Refresh", command=self.refresh_files).pack(fill=tk.X, pady=(8, 0))
        self._build_controls(side)

        content = ttk.Frame(self, padding=(4, 8, 8, 8))
        content.grid(row=0, column=1, sticky="nsew")
        content.rowconfigure(0, weight=2)
        content.rowconfigure(2, weight=1)
        content.columnconfigure(0, weight=1)

        self.fig_hybrid, top_axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        self.ax_hybrid, self.ax_hybrid_mask = top_axs
        self.hybrid_canvas = FigureCanvasTkAgg(self.fig_hybrid, master=content)
        self.hybrid_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.hybrid_canvas.mpl_connect("scroll_event", self._on_scroll)

        roi_bar = ttk.Frame(content, padding=(0, 6, 0, 6))
        roi_bar.grid(row=1, column=0, sticky="ew")
        ttk.Label(roi_bar, text="ROI").pack(side=tk.LEFT)
        self.roi_combo = ttk.Combobox(roi_bar, textvariable=self.roi_var, state="readonly", width=48)
        self.roi_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self.roi_combo.bind("<<ComboboxSelected>>", self._on_roi_selected)

        self.fig_bottom, bottom_axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        self.ax_roi, self.ax_anomaly = bottom_axs
        self.bottom_canvas = FigureCanvasTkAgg(self.fig_bottom, master=content)
        self.bottom_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        self.bottom_canvas.mpl_connect("scroll_event", self._on_scroll)

    def refresh_files(self):
        self._refresh_file_list(self.hybrid_images_dir)
        self._refresh_roi_files()
        self.render()

    def _on_file_selected(self, _event=None):
        selection = self.file_list.curselection()
        if not selection:
            return
        self.current_filename = self.files[selection[0]]
        self.slice_var.set(0)
        self._refresh_roi_files()
        self.render()

    def _on_roi_selected(self, _event=None):
        self.slice_var.set(0)
        self.render()

    def _refresh_roi_files(self):
        selected = self.current_filename
        if selected is None:
            self.current_roi_dir = None
            self.expected_roi_dir = self.synth_roi_dir
            self.roi_files = []
        else:
            self.current_roi_dir, self.expected_roi_dir = _resolve_dir_by_name(self.synth_roi_dir, selected)
            self.roi_files = _list_npy_files(self.current_roi_dir) if self.current_roi_dir else []

        self.roi_combo.configure(values=self.roi_files)
        if self.roi_files:
            self.roi_var.set(self.roi_files[0])
            self.roi_combo.state(["!disabled"])
        else:
            self.roi_var.set("")
            self.roi_combo.state(["disabled"])

    def render(self):
        selected = self.current_filename
        roi_name = self.roi_var.get()

        if selected is None:
            top_items = [
                _array_item("Hybrid sample", None, self.hybrid_images_dir),
                _array_item("Hybrid mask", None, self.hybrid_segmentations_dir),
            ]
            bottom_items = [
                _array_item("Selected synthetic ROI", None, self.synth_roi_dir),
                _array_item("Matched anomaly", None, self.anomaly_dir),
            ]
            self.fig_hybrid.suptitle("fused anomaly", fontsize=13, fontweight="bold")
            self.fig_bottom.suptitle("")
        else:
            hybrid_path, hybrid_expected = _resolve_file_by_name(self.hybrid_images_dir, selected)
            hybrid_mask_path, hybrid_mask_expected = _resolve_file_by_name(self.hybrid_segmentations_dir, selected)

            if roi_name and self.current_roi_dir:
                roi_path = os.path.join(self.current_roi_dir, roi_name)
                roi_expected = os.path.join(self.expected_roi_dir, roi_name)
                anomaly_path, anomaly_expected = _resolve_file_by_name(self.anomaly_dir, roi_name)
                anomaly_roi_path, _anomaly_roi_expected = _resolve_file_by_name(self.anomaly_roi_dir, roi_name)
                anomaly_meta = _get_anomaly_meta(self.anomaly_transformations, roi_name)
            else:
                roi_path = None
                roi_expected = self.expected_roi_dir
                anomaly_path = None
                anomaly_expected = os.path.join(self.anomaly_dir, "<selected_roi>.npy")
                anomaly_roi_path = None
                anomaly_meta = None

            top_items = [
                _array_item("Hybrid sample", hybrid_path, hybrid_expected),
                _array_item("Hybrid mask", hybrid_mask_path, hybrid_mask_expected),
            ]
            bottom_items = [
                _array_item("Selected synthetic ROI", roi_path, roi_expected),
                _array_item(
                    "Matched anomaly", anomaly_path, anomaly_expected,
                    meta=anomaly_meta, window_path=anomaly_roi_path,
                ),
            ]
            roi_suffix = f" | ROI: {roi_name}" if roi_name else " | no ROI found"
            self.fig_hybrid.suptitle(f"fused anomaly: {selected}", fontsize=13, fontweight="bold")
            self.fig_bottom.suptitle(roi_suffix.lstrip(" | "), fontsize=11)

        max_slices = max(
            self._render_items((self.ax_hybrid, self.ax_hybrid_mask), top_items),
            self._render_items((self.ax_roi, self.ax_anomaly), bottom_items),
        )
        self._sync_slice_control(max_slices)
        self.hybrid_canvas.draw_idle()
        self.bottom_canvas.draw_idle()


class HybridDataGeneratorVisualizer:
    def __init__(self, root, config, channel: int = 0, cmap: str = "gray"):
        self.root = root
        self.config = config
        self.root.title("HybridDataGenerator Visualizer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.anomaly_tab = AnomalyGenerationTab(self.notebook, config, channel=channel, cmap=cmap)
        self.notebook.add(self.anomaly_tab, text="anomaly generation")

        self.fused_tab = FusedAnomalyTab(self.notebook, config, channel=channel, cmap=cmap)
        self.notebook.add(self.fused_tab, text="fused anomaly")

        self.evaluation_tab = ttk.Frame(self.notebook)
        if os.path.isdir(_paths(config).evaluation_results):
            self.evaluation_gui = OutlierGUI(self.evaluation_tab, config, embedded=True)
            self.notebook.add(self.evaluation_tab, text="evaluation")
        else:
            self.notebook.add(self.evaluation_tab, text="evaluation", state="disabled")

    def on_closing(self):
        plt.close("all")
        self.root.quit()
        self.root.destroy()


def run_hybrid_visualizer(config, channel: int = 0, cmap: str = "gray"):
    root = tk.Tk()
    root.tk.call("tk", "scaling", 2.0)
    HybridDataGeneratorVisualizer(root, config, channel=channel, cmap=cmap)
    root.mainloop()


def run_hybrid_visualizer_for_folder(study_folder: str, channel: int = 0, cmap: str = "gray"):
    class FolderConfig:
        def __init__(self, folder: str):
            study_name = os.path.basename(os.path.normpath(folder)) or "study"
            self.paths = StudyPaths(folder, study_name)

        def get_paths(self) -> StudyPaths:
            return self.paths

    config = FolderConfig(study_folder)
    run_hybrid_visualizer(config, channel=channel, cmap=cmap)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize HybridDataGenerator outputs in three tabs: anomaly generation, fused anomaly, and evaluation."
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

    _select_gui_backend(prefer=args.backend)
    run_hybrid_visualizer_for_folder(args.folder, channel=args.channel, cmap=args.cmap)


if __name__ == "__main__":
    main()
