"""Selection and validation of MVTec generation steps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


PREPARE_USECASE_KEYS = {
    "include_public_good_controls",
    "save_path",
    "results_root",
}
DEPRECATED_GENERATION_FLAGS = ("run_evaluation", "visualize_evaluation")
DEFAULT_GENERATION_STEPS = (
    "ingest_dataset",
    "extract_anomalies",
    "train_generator",
    "generate_synthetic_anomalies",
    "plan_hybrid_samples",
    "materialize_hybrid_samples",
    "save_config",
)
GENERATION_STEP_ORDER = (
    "ingest_dataset",
    "extract_anomalies",
    "train_generator",
    "load_generator",
    "generate_synthetic_anomalies",
    "plan_hybrid_samples",
    "materialize_hybrid_samples",
    "save_config",
)
GENERATION_STEP_ALIASES = {
    "all": "all",
    "default": "all",
    "ingest": "ingest_dataset",
    "ingest_dataset": "ingest_dataset",
    "extract": "extract_anomalies",
    "extract_anomalies": "extract_anomalies",
    "train": "train_generator",
    "train_generator": "train_generator",
    "load_generator": "load_generator",
    "load_model": "load_generator",
    "generate_synth": "generate_synthetic_anomalies",
    "generate_synthetic_anomalies": "generate_synthetic_anomalies",
    "synth": "generate_synthetic_anomalies",
    "plan": "plan_hybrid_samples",
    "plan_hybrid_samples": "plan_hybrid_samples",
    "materialize": "materialize_hybrid_samples",
    "materialize_hybrid_samples": "materialize_hybrid_samples",
    "save": "save_config",
    "save_config": "save_config",
}
GENERATION_STEP_CONFLICTS = (
    ("train_generator", "load_generator"),
)


def _pop_prepare_kwargs(kwargs: dict) -> dict:
    return {key: kwargs.pop(key) for key in list(kwargs) if key in PREPARE_USECASE_KEYS}


def _reject_deprecated_generation_flags(kwargs: dict) -> None:
    deprecated_values = {key: kwargs.pop(key) for key in DEPRECATED_GENERATION_FLAGS if key in kwargs}
    enabled = [key for key, value in deprecated_values.items() if value]
    if enabled:
        raise ValueError(
            "Evaluation and visualization are now downstream steps. "
            "Run run_evaluation_for_all_usecases(...) and "
            "visualize_evaluation_for_all_usecases(...) after generation instead, "
            "or use run_evaluation_and_visualization_for_all_usecases(...)."
        )


def _reject_unknown_kwargs(kwargs: Mapping[str, object]) -> None:
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")


def _normalize_generation_steps(
    steps: str | Iterable[str] | None,
    *,
    train_generator: bool,
    load_existing_generator: bool,
    generate_synthetic_anomalies: bool,
    plan_hybrids: bool,
) -> tuple[str, ...]:
    if steps is None:
        selected_steps = _default_generation_steps(
            train_generator=train_generator,
            load_existing_generator=load_existing_generator,
            generate_synthetic_anomalies=generate_synthetic_anomalies,
            plan_hybrids=plan_hybrids,
        )
    else:
        selected_steps = _canonical_generation_steps(steps)

    _validate_generation_step_conflicts(selected_steps)
    return selected_steps


def _default_generation_steps(
    *,
    train_generator: bool,
    load_existing_generator: bool,
    generate_synthetic_anomalies: bool,
    plan_hybrids: bool,
) -> tuple[str, ...]:
    steps = []

    if generate_synthetic_anomalies:
        if train_generator and load_existing_generator:
            raise ValueError("Training and loading a generator are mutually exclusive.")
        steps.extend(["ingest_dataset", "extract_anomalies"])
        if train_generator:
            steps.append("train_generator")
        elif load_existing_generator:
            steps.append("load_generator")
        else:
            raise ValueError("Either train_generator or load_existing_generator must be True.")
        steps.append("generate_synthetic_anomalies")

    if plan_hybrids:
        steps.append("plan_hybrid_samples")

    steps.extend(["materialize_hybrid_samples", "save_config"])
    return tuple(steps)


def _canonical_generation_steps(steps: str | Iterable[str]) -> tuple[str, ...]:
    raw_steps = [steps] if isinstance(steps, str) else list(steps)
    if not raw_steps:
        raise ValueError("steps must contain at least one pipeline step.")

    canonical_steps = []
    for step in raw_steps:
        if not isinstance(step, str):
            raise TypeError(f"Pipeline step names must be strings, got {type(step).__name__}.")
        step_key = step.strip().lower().replace("-", "_").replace(" ", "_")
        canonical_step = GENERATION_STEP_ALIASES.get(step_key)
        if canonical_step is None:
            allowed_steps = ", ".join(sorted(GENERATION_STEP_ALIASES))
            raise ValueError(f"Unknown generation step {step!r}. Allowed steps: {allowed_steps}")
        if canonical_step == "all":
            return DEFAULT_GENERATION_STEPS
        canonical_steps.append(canonical_step)

    canonical_set = set(canonical_steps)
    return tuple(step for step in GENERATION_STEP_ORDER if step in canonical_set)


def _validate_generation_step_conflicts(steps: Sequence[str]) -> None:
    step_set = set(steps)
    for left, right in GENERATION_STEP_CONFLICTS:
        if left in step_set and right in step_set:
            raise ValueError(f"Pipeline steps {left!r} and {right!r} cannot be used together.")


def _needs_generator_loaded(steps: Sequence[str]) -> bool:
    step_set = set(steps)
    return (
        "generate_synthetic_anomalies" in step_set
        and "train_generator" not in step_set
        and "load_generator" not in step_set
    )
