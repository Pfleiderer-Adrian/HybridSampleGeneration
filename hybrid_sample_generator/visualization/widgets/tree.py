"""Tree-view construction helpers."""

from tkinter import ttk


def insert_tree_scrollbars(parent, *, columns=(), headings=(), height=20):
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    tree = ttk.Treeview(
        frame,
        columns=tuple(columns),
        show="tree headings" if columns else "tree",
        height=height,
        selectmode="browse",
    )
    tree.grid(row=0, column=0, sticky="nsew")
    vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    horizontal.grid(row=1, column=0, sticky="ew")
    tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    for name, heading in zip(columns, headings):
        tree.heading(name, text=heading)
    return tree
