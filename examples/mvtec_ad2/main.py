"""Executable MVTec AD 2 example workflow."""

from examples.mvtec_ad2.runner import (
    run_evaluation_and_visualization_for_all_usecases,
    run_hybrid_sample_generation_for_all_usecases,
)
from examples.mvtec_ad2.settings import MVTECAD2_ROOT, MVTECAD2_SAVE


def main() -> None:
    categories = ("can",)
    run_hybrid_sample_generation_for_all_usecases(
        root=MVTECAD2_ROOT,
        categories=categories,
        no_of_trials=1,
        steps=(
            #"ingest",
            #"extract",
            #"train",
            #"generate_synth",
            #"plan",
            "materialize",
            "save",
        ),
        generator_trial_id=-2,
        save_path=MVTECAD2_SAVE,
    )
    run_evaluation_and_visualization_for_all_usecases(
        root=MVTECAD2_ROOT,
        categories=categories,
        save_path=MVTECAD2_SAVE,
    )


if __name__ == "__main__":
    main()
