"""One explicit vocabulary for the MVTec workflow."""

from collections.abc import Iterable

GENERATOR_STEPS = (
    "ingest", "extract", "train_generator", "load_generator",
    "generate_synthetic", "plan", "materialize",
)
STEP_ORDER = (*GENERATOR_STEPS, "export", "train_downstream", "evaluate_downstream")
DEFAULT_GENERATION_STEPS = tuple(step for step in GENERATOR_STEPS if step != "load_generator")

GENERATE_HYBRIDS = DEFAULT_GENERATION_STEPS
TRAIN_DOWNSTREAM = ("train_downstream", "evaluate_downstream")
FULL_EXPERIMENT = (*GENERATE_HYBRIDS, "export", *TRAIN_DOWNSTREAM)


def normalize_steps(steps: str | Iterable[str] | None) -> tuple[str, ...]:
    selected = DEFAULT_GENERATION_STEPS if steps is None else ((steps,) if isinstance(steps, str) else tuple(steps))
    if not selected:
        raise ValueError("steps must contain at least one pipeline step.")
    for step in selected:
        if not isinstance(step, str):
            raise TypeError("Pipeline step names must be strings.")
        if step not in STEP_ORDER:
            raise ValueError(f"Unknown step {step!r}. Allowed steps: {', '.join(STEP_ORDER)}")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate pipeline steps are not allowed.")
    if "train_generator" in selected and "load_generator" in selected:
        raise ValueError("Training and loading a generator are mutually exclusive.")
    ordered = tuple(step for step in STEP_ORDER if step in selected)
    if tuple(selected) != ordered:
        raise ValueError(f"Steps must follow pipeline order: {', '.join(ordered)}")
    return ordered
