"""Generate and verify the searchable configuration reference.

Types, defaults, allowed values and explanations all come from
``Configuration.schema()``, i.e. from the ``param(...)`` declarations on the
configuration dataclasses. This script only adds page-level prose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_sample_generator.configuration.parameters import iter_parameters, range_text
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.generation.registry import MODEL_REGISTRY
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "configuration" / "reference"

TITLES = {
    "study": "Study",
    "extraction": "Extraction",
    "augmentation": "Augmentation",
    "generation": "Generation",
    "matching": "Matching",
    "postprocessing": "Postprocessing",
    "training": "Training",
    "evaluation": "Evaluation",
}


def _display(value) -> str:
    if isinstance(value, dict) and len(value) > 5:
        return f"{len(value)} metrics with min=None and max=None"
    return repr(value)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _type_text(entry) -> str:
    kind = entry["type"]
    if kind == "choice":
        names = sorted({type(choice).__name__ for choice in entry["choices"]})
        text = " | ".join(names)
    elif kind == "list":
        text = f"list[{_type_text(entry['items'])}]"
    elif kind == "map":
        text = "dict"
    elif kind == "any_of":
        text = " | ".join(_type_text(option) for option in entry["options"])
    elif kind == "any":
        text = "Any"
    else:
        text = kind
    return f"{text} | None" if entry.get("nullable") else text


def _allowed_text(entry) -> str:
    kind = entry["type"]
    if kind == "choice":
        values = [repr(choice) for choice in entry["choices"]]
        if entry.get("nullable"):
            values.append("None")
        return ", ".join(f"`{_cell(value)}`" for value in values)
    if kind in ("int", "float"):
        limits = range_text(entry)
        return f"`{limits}`" if limits else ""
    if kind == "list":
        parts = []
        low, high = entry.get("min_length"), entry.get("max_length")
        if low is not None or high is not None:
            parts.append(f"{low} elements" if low == high else f"{low} to {high} elements")
        items = _allowed_text(entry["items"])
        if items:
            parts.append(f"each {items}")
        return ", ".join(parts)
    if kind == "any_of":
        return " or ".join(text for text in map(_allowed_text, entry["options"]) if text)
    return ""


def _rows(group, prefix: str):
    rows = []
    for path, entry in iter_parameters(group, prefix):
        if "doc" not in entry:
            raise ValueError(f"Missing configuration description: {path}")
        if entry.get("required"):
            default = "required at construction"
        elif "default_label" in entry:
            default = entry["default_label"]
        else:
            default = _display(entry["default"])
        rows.append((path, _type_text(entry), default, _allowed_text(entry), entry["doc"]))
    return rows


def _table(rows, *, search=None):
    headers = ["Parameter", "Type", "Default", "Allowed values", "Meaning"]
    if search is not None:
        headers.append("Default search")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for path, kind, default, allowed, description in rows:
        cells = [
            f"`config.{path}`",
            f"`{_cell(kind)}`",
            f"`{_cell(default)}`",
            allowed or "—",
            description,
        ]
        if search is not None:
            name = path.rsplit(".", 1)[1]
            distribution = getattr(search, "_distributions", {}).get(name)
            cells.append(f"`{distribution!r}`" if distribution is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render() -> dict[Path, str]:
    config = Configuration("example", study_folder="results/example")
    schema = Configuration.schema()["sections"]
    pages = {}
    general = [name for name in schema if name not in ("model", "fusion")]
    if set(general) != set(TITLES):
        raise ValueError(
            f"Sections without a page title: {sorted(set(general) - set(TITLES))}; "
            f"obsolete titles: {sorted(set(TITLES) - set(general))}"
        )
    for slug in general:
        title = TITLES[slug]
        rows = _rows(schema[slug], slug)
        extra = ""
        if slug == "augmentation":
            generator = TransformGenerator.from_config(
                config.augmentation.mask_transforms,
                anomaly_size=config.extraction.anomaly_size,
                background_threshold=config.generation.background_threshold,
                seed=0,
            )
            transform_lines = [
                "| Transform | Default probability | Effective default parameters |",
                "| --- | --- | --- |",
            ]
            for name in generator.transform_params:
                probabilities = (
                    generator.local_transform_probs if name.startswith("local_")
                    else generator.global_transform_probs
                )
                transform_lines.append(
                    f"| `{name}` | `{probabilities.get(name, 0)}` | "
                    f"`{_display(generator.transform_params[name])}` |"
                )
            extra = (
                "\nMasks use nearest-neighbor interpolation; jointly transformed images use "
                "linear interpolation. `mask_transform_probs` and `mask_transform_params` "
                "accept global or local transform names and class IDs, and are applied only "
                "when `use_mask_transform=True`. With mask transformations enabled, the "
                "default anomaly size "
                f"`{config.extraction.anomaly_size}` gives these effective values; "
                "global elastic parameters change with `anomaly_size`.\n\n"
                "## Transform defaults\n\n"
                + "\n".join(transform_lines)
            )
        if slug == "training":
            extra = (
                "\n## Nested training settings\n\n"
                + "\n".join(
                    f"- `config.training.{name}[{key!r}]`: {item['doc']}"
                    + (f" Allowed: {_allowed_text(item)}." if _allowed_text(item) else "")
                    for name in ("early_stopping", "lr_scheduler")
                    for key, item in schema["training"]["fields"][name]["keys"].items()
                )
                + "\n\nTheir defaults are shown in the table above.\n"
            )
        if slug == "postprocessing":
            extra = (
                "\nRun with `generator.postprocess_hybrid_samples()` after materialization; "
                "see the postprocessing guide. Configuration files without this section "
                "load with postprocessing disabled.\n\n"
                "Any failed rule marks the entire hybrid as failed quality checking. "
                "Measurements refer to the union of the final materialized anomaly masks, "
                "not individual ROIs. Empty estimated object masks stop evaluation rather "
                "than silently rejecting all samples. Object overlap requires a healthy "
                "original partner. Threshold masks are heuristics, not semantic ground truth. "
                "Multiple defect types are not classified automatically; type-specific rules "
                "are not yet supported.\n"
            )
        if slug == "evaluation":
            extra = "\n## Defaults\n\n`outlier_thresholds` contains `min: None` and `max: None` for Contrast, Homogeneity, Energy, Correlation, their four `roi_` variants, Volume, D-center, H-center, and W-center.\n"
        pages[OUTPUT / f"{slug}.md"] = f"# {title}\n\n{_table(rows)}\n{extra}\n"
    for name, variant in sorted(schema["model"]["variants"].items()):
        spec = MODEL_REGISTRY[name]
        rows = _rows(variant["parameters"], "model.parameters")
        search = spec.build_search_space(spec.build_configuration())
        pages[OUTPUT / f"model-{name}.md"] = (
            f"# {name}\n\n"
            f"Select with `config.model.set_model({name!r})`. The table shows the "
            "actual defaults for this model variant. A dash in Default search means "
            "the parameter stays fixed unless you assign a compatible distribution "
            "through `config.model.search`.\n\n"
            f"{_table(rows, search=search)}\n"
        )

    for name, variant in sorted(schema["fusion"]["variants"].items()):
        rows = _rows(variant["parameters"], "fusion.parameters")
        pages[OUTPUT / f"fusion-{name}.md"] = (
            f"# Fusion: {name}\n\n"
            f"Select with `config.fusion.set_backend({name!r})`.\n\n"
            f"{_table(rows)}\n"
        )
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated pages differ")
    args = parser.parse_args()
    pages = _render()
    existing = set(OUTPUT.glob("*.md"))
    stale = existing - set(pages)
    changed = [path for path, content in pages.items() if not path.exists() or path.read_text() != content]
    if args.check:
        if changed or stale:
            for path in sorted(changed + list(stale)):
                print(f"Outdated configuration reference: {path.relative_to(ROOT)}")
            return 1
        print(f"Configuration reference current: {len(pages)} pages")
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, content in pages.items():
        path.write_text(content)
    for path in stale:
        path.unlink()
    print(f"Generated {len(pages)} configuration reference pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
