"""Run a new experiment using settings.py and the shared/category presets."""

from examples.mvtec_ad2.pipeline import FULL_EXPERIMENT, run_new_experiment
from examples.mvtec_ad2.settings import Experiment, MVTECAD2_ROOT, MVTECAD2_SAVE, SplitConfiguration

# Select GENERATE_HYBRIDS or an explicit step tuple for a partial workflow.
STEPS = FULL_EXPERIMENT

def main() -> None:

    experiment = Experiment(
        dataset_root=MVTECAD2_ROOT,
        output_root=MVTECAD2_SAVE,
        categories=("can"),
        split=SplitConfiguration(test_enabled=True, test_fraction=0.2, validation_fraction=0.2),
    )

    for result in run_new_experiment(experiment, steps=STEPS):
        print(f"Study: {result.study_folder}; downstream run: {result.downstream_run_folder}")

if __name__ == "__main__":
    main()
