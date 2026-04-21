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

class OutlierGUI:
    def __init__(self, root, config):
        self.root = root
        self.root.title("Outlier Viewer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.config = config

        self.synth_anomaly_dir = os.path.join(config.study_folder, "synth_anomaly_data")
        self.synth_roi_dir = os.path.join(config.study_folder, "synth_roi_data")
        self.anomaly_dir = os.path.join(config.study_folder, "anomaly_data")
        self.anomaly_roi_dir = os.path.join(config.study_folder, "anomaly_roi_data")
        
        self.ghs_dir = os.path.join(config.study_folder, "generated_hybrid_samples", "images_npy")
        self.ghs_seg_dir = os.path.join(config.study_folder, "generated_hybrid_samples", "segmentations_npy")

        self.metric_stats = {}
        
        self.hierarchy = defaultdict(list)
        self.metric_map = self.build_metric_sample_map()
        
        self.filtered_hierarchy = {}
        self.sorted_controls = []
        self.flat_list = []
        
        self.current_index = 0
        self.current_slice = 0

        self.build_ui()
        self.update_filter()
        self.root.focus_set()

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        os._exit(0)

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

        csv_path = os.path.join(self.config.study_folder, "evaluation_results", "metric_diffs.csv")
        
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
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Label(control_frame, text="Filter & Sort by:", font=('Arial', 10, 'bold')).pack(anchor="w")
        self.metric_vars = {}
        for metric in sorted(self.metric_map.keys()):
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(control_frame, text=metric, variable=var, command=self.update_filter)
            cb.pack(anchor="w")
            self.metric_vars[metric] = var

        tk.Label(control_frame, text="Outlier Threshold (Top %):", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        self.outlier_slider = tk.Scale(control_frame, from_=0, to=10, resolution=.1, orient=tk.HORIZONTAL, 
                                       command=lambda _: self.update_filter())
        self.outlier_slider.set(1)
        self.outlier_slider.pack(fill=tk.X, pady=(0, 5))

        self.list_frame = tk.Frame(control_frame)
        self.list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.scrollbar = tk.Scrollbar(self.list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(self.list_frame, yscrollcommand=self.scrollbar.set, selectmode="browse")
        self.tree.heading("#0", text="Controls / Anomalies", anchor="w")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.tree.yview)
        self.tree.bind('<<TreeviewSelect>>', self.on_treeview_select)

        contrast_header_frame = tk.Frame(control_frame)
        contrast_header_frame.pack(fill=tk.X, pady=(10, 0))
        tk.Label(contrast_header_frame, text="Contrast:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        tk.Button(contrast_header_frame, text="reset", command=self.reset_contrast, font=('Arial', 8, 'italic'),
                  relief=tk.FLAT, padx=2, pady=0, cursor="hand2").pack(side=tk.LEFT, padx=5)

        self.contrast_slider = tk.Scale(control_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                        command=lambda _: self.update_display())
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(fill=tk.X, pady=(0, 10))

        tk.Button(control_frame, text="Prev Sample (←)", command=self.prev_sample).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Next Sample (→)", command=self.next_sample).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Slice - (↓)", command=self.prev_slice).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Slice + (↑)", command=self.next_slice).pack(fill=tk.X, pady=2)

        self.info_text = tk.Text(control_frame, height=10, width=30, bg=self.root.cget("bg"), relief=tk.FLAT, font=("Arial", 10))
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
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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
        if event.num == 4 or event.delta > 0:
            self.next_slice()
        elif event.num == 5 or event.delta < 0:
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
            self.fig.suptitle(f"Control: {control}", fontsize=14, fontweight='bold', y=.995)
            
            ghs_path = self._get_fallback_path(self.ghs_dir, control)
            ghs_seg_path = self._get_fallback_path(self.ghs_seg_dir, control)
            
            paths = [
                (ghs_path, "Generated Hybrid Sample"),
                (ghs_seg_path, "Generated Hybrid Segmentation"),
                (None, ""),
                (None, "")
            ]
        else:
            _, control, anomaly = item
            self.fig.suptitle(f"{anomaly} in {control}", fontsize=14, fontweight='bold', y=.995)

            paths = [
                (os.path.join(self.synth_anomaly_dir, anomaly), "synth_anomaly_data"),
                (os.path.join(self.synth_roi_dir, control, anomaly), "synth_roi_data"),
                (os.path.join(self.anomaly_dir, anomaly), "anomaly_data"),
                (os.path.join(self.anomaly_roi_dir, anomaly), "anomaly_roi_data")
            ]

        loaded_data = []
        max_slices = 0
        for p, title in paths:
            if p and os.path.exists(p):
                arr = np.load(p)
                if arr.ndim == 4:
                    max_slices = max(max_slices, arr.shape[1])
                    curr_slice = min(self.current_slice, arr.shape[1] - 1)
                    img = arr[:, curr_slice, :, :]
                    img = np.transpose(img, (1, 2, 0))
                    display_title = f"{title}\nSlice {curr_slice}"
                elif arr.ndim == 3:
                    img = np.transpose(arr, (1, 2, 0))
                    display_title = title
                else:
                    img = arr
                    display_title = title
                loaded_data.append((img, display_title))
            else:
                loaded_data.append((None, title))

        if self.current_slice >= max_slices and max_slices > 0:
            self.current_slice = max_slices - 1

        for i, (img, title) in enumerate(loaded_data):
            if not title:
                continue
            
            if img is None:
                self.axs[i].set_title(f"{title}\nNOT FOUND", fontsize=9)
                continue

            img_float = img.astype(np.float32)
            i_min, i_max = np.min(img_float), np.max(img_float)
            img_norm = (img_float - i_min) / (i_max - i_min) if i_max > i_min else img_float - i_min
            img_display = np.clip(img_norm * contrast, 0, 1)

            self.axs[i].set_title(title, fontsize=10, pad=10)
            
            if img_display.ndim == 3 and img_display.shape[-1] == 1:
                self.axs[i].imshow(img_display[:, :, 0], cmap="gray", vmin=0, vmax=1, aspect='equal')
            elif img_display.ndim == 3:
                self.axs[i].imshow(img_display, aspect='equal')
            else:
                self.axs[i].imshow(img_display, cmap="gray", vmin=0, vmax=1, aspect='equal')

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert(tk.END, f"Selected: {self.current_index+1} / {len(self.flat_list)}\n\n", "header")
        
        active = [m for m, v in self.metric_vars.items() if v.get()]
        
        if item[0] == "control":
            self.info_text.insert(tk.END, f"Anomalies in this control: {len(self.filtered_hierarchy.get(item[1], []))}\n", "active")
        else:
            control, anomaly = item[1], item[2]
            for m in sorted(self.metric_map.keys()):
                if control in self.metric_map[m] and anomaly in self.metric_map[m][control]:
                    val = self.metric_map[m][control][anomaly]
                    line = f"{m}: {val:.4f}\n"
                    self.info_text.insert(tk.END, line, "active" if m in active else "inactive")
                
        self.info_text.config(state=tk.DISABLED)
        self.canvas.draw()

def run_outlier_gui(config):
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 2.0)
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

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize 4 matched arrays from 4 folders side-by-side. "
            "Supports (C,D,H,W) and (C,H,W). If C==3 => RGB, if C!=3 => grayscale channel. The 4th view is loaded from the fixed subfolder 'generated_hybrid_samples'."
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
    fusion_folder = os.path.join(args.folder, os.path.join("generated_hybrid_samples", "images_npy"))

    visualize_four_folders(
        roi_folder,
        anomaly_folder,
        synth_folder,
        fusion_folder,
        channel=args.channel,
        cmap=args.cmap,
        backend_preference=args.backend,
        exts=args.exts,
    )


if __name__ == "__main__":
    main()
