"""Command-line entry points for new experiments and saved studies."""

import argparse
from dataclasses import replace
from pathlib import Path

from examples.mvtec_ad2.pipeline import FULL_EXPERIMENT, run_existing_studies, run_new_experiment
from examples.mvtec_ad2.review import review_studies
from examples.mvtec_ad2.settings import EXPERIMENT
from examples.mvtec_ad2.steps import STEP_ORDER


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create new category studies")
    generate.add_argument("--dataset-root", type=Path, default=EXPERIMENT.dataset_root)
    generate.add_argument("--output-root", type=Path, default=EXPERIMENT.output_root)
    generate.add_argument("--categories", nargs="+", default=EXPERIMENT.categories)
    generate.add_argument("--steps", nargs="+", choices=STEP_ORDER, default=FULL_EXPERIMENT)
    resume = commands.add_parser("continue", help="Run explicit steps on saved studies")
    resume.add_argument("--study-folder", type=Path, nargs="+", required=True)
    resume.add_argument("--steps", nargs="+", choices=STEP_ORDER, required=True)
    review = commands.add_parser("review", help="View or evaluate generator artifacts")
    review.add_argument("--study-folder", type=Path, nargs="+", required=True)
    review.add_argument("--actions", nargs="+", choices=("evaluate_generator", "visualize"), default=("visualize",))
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved downstream run")
    evaluate.add_argument("--study-folder", type=Path, required=True)
    evaluate.add_argument("--run-id", required=True)
    return parser


def main(argv=None) -> None:
    args = create_parser().parse_args(argv)
    if args.command == "generate":
        experiment = replace(EXPERIMENT, dataset_root=args.dataset_root,
                             output_root=args.output_root, categories=tuple(args.categories))
        results = run_new_experiment(experiment, steps=args.steps)
    elif args.command == "continue":
        results = run_existing_studies(args.study_folder, steps=args.steps)
    elif args.command == "evaluate":
        results = run_existing_studies([args.study_folder], steps=("evaluate_downstream",),
                                      downstream_run_id=args.run_id)
    else:
        review_studies(args.study_folder, actions=args.actions)
        return
    for result in results:
        print(f"Study: {result.study_folder}; downstream run: {result.downstream_run_folder}")


if __name__ == "__main__":
    main()
