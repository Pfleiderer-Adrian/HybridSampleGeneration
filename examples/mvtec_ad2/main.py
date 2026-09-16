"""Run a new experiment using settings.py and the shared/category presets."""

from examples.mvtec_ad2.pipeline import FULL_EXPERIMENT, run_new_experiment
from examples.mvtec_ad2.settings import EXPERIMENT

# Select GENERATE_HYBRIDS or an explicit step tuple for a partial workflow.
STEPS = FULL_EXPERIMENT


def main() -> None:
    for result in run_new_experiment(EXPERIMENT, steps=STEPS):
        print(f"Study: {result.study_folder}; downstream run: {result.downstream_run_folder}")


if __name__ == "__main__":
    main()
