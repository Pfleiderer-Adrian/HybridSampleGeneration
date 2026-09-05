from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from data_handler.visualizer.queries import StudyBrowserModel


SUMMARY_FIELDS = (
    ("originals", "Originals"),
    ("anomalous_originals", "Anomalous"),
    ("controls", "Controls"),
    ("real_anomalies", "Real anomalies"),
    ("synthetic_anomalies", "Synthetic variants"),
    ("hybrids", "Hybrid samples"),
    ("placements", "Placements"),
    ("match_candidates", "Cached matches"),
    ("evaluation_pairs", "Evaluation pairs"),
)


class OverviewTab(ttk.Frame):
    def __init__(self, master, model: StudyBrowserModel, *, configuration_path=None):
        super().__init__(master, padding=12)
        self.model = model
        self.configuration_path = configuration_path
        self.value_vars = {name: tk.StringVar(value="0") for name, _ in SUMMARY_FIELDS}
        self.status_var = tk.StringVar(value="")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        cards = ttk.Frame(self)
        cards.grid(row=0, column=0, sticky="ew")
        for index, (name, label) in enumerate(SUMMARY_FIELDS):
            card = ttk.LabelFrame(cards, text=label, padding=8)
            card.grid(row=index // 5, column=index % 5, sticky="nsew", padx=4, pady=4)
            ttk.Label(
                card,
                textvariable=self.value_vars[name],
                font=("Arial", 18, "bold"),
            ).pack()
            cards.columnconfigure(index % 5, weight=1)

        lower = ttk.PanedWindow(self, orient="horizontal")
        lower.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        health = ttk.LabelFrame(lower, text="Study health", padding=8)
        config = ttk.LabelFrame(lower, text="Configuration", padding=8)
        lower.add(health, weight=1)
        lower.add(config, weight=2)

        self.health_tree = ttk.Treeview(
            health,
            columns=("value",),
            show="tree headings",
            height=12,
        )
        self.health_tree.heading("#0", text="Check")
        self.health_tree.heading("value", text="Value")
        self.health_tree.column("#0", width=220)
        self.health_tree.column("value", width=170)
        self.health_tree.pack(fill="both", expand=True)
        ttk.Label(health, textvariable=self.status_var, wraplength=390).pack(
            anchor="w", fill="x", pady=(8, 0)
        )

        self.config_text = tk.Text(config, wrap="none", relief="flat")
        self.config_text.pack(fill="both", expand=True)
        self.config_text.configure(state="disabled")
        self.refresh()

    def refresh(self) -> None:
        summary = self.model.summary()
        for name, _label in SUMMARY_FIELDS:
            self.value_vars[name].set(str(summary[name]))

        self.health_tree.delete(*self.health_tree.get_children())
        rows = (
            ("Generated hybrids", f"{summary['generated']} / {summary['hybrids']}"),
            ("Planned hybrids", summary["planned"]),
            ("Failed hybrids", summary["failed"]),
            ("Placements per hybrid", _ratio(summary["placements"], summary["hybrids"])),
            (
                "Synthetic variants per real",
                _ratio(summary["synthetic_anomalies"], summary["real_anomalies"]),
            ),
        )
        for label, value in rows:
            self.health_tree.insert("", "end", text=label, values=(value,))

        missing = self.model.missing_artifacts()
        self.health_tree.insert(
            "",
            "end",
            text="Missing artifact references",
            values=(len(missing),),
        )
        self.status_var.set(
            "Repository and artifact references are consistent."
            if not missing
            else (
                f"{len(missing)} referenced artifacts are missing; "
                "inspect the Data structure tab."
            )
        )
        self._load_configuration()

    def set_search(self, _query: str) -> None:
        return

    def close(self) -> None:
        return

    def _load_configuration(self) -> None:
        value = "No saved configuration found."
        path = Path(self.configuration_path) if self.configuration_path else None
        if path and path.is_file():
            try:
                value = json.dumps(
                    json.loads(path.read_text(encoding="utf-8")),
                    ensure_ascii=False,
                    indent=2,
                )
            except (OSError, json.JSONDecodeError) as exc:
                value = f"Could not read {path}:\n{exc}"
        self.config_text.configure(state="normal")
        self.config_text.delete("1.0", tk.END)
        self.config_text.insert("1.0", value)
        self.config_text.configure(state="disabled")


def _ratio(numerator: int, denominator: int) -> str:
    return "–" if not denominator else f"{numerator / denominator:.2f}"
