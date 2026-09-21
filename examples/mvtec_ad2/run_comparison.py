"""Run all paired DRAEM experiments, or validate their grouped splits."""
import argparse
from pathlib import Path
from examples.mvtec_ad2.common import create_downstream_configuration
from examples.mvtec_ad2.settings import DATASET_ROOT
from .comparison.configuration import CATEGORIES, ComparisonConfiguration
from .comparison.runner import run_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--categories', nargs='+', default=['all'], choices=['all', *CATEGORIES])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-root', type=Path, default=DATASET_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Validate and persist splits without preparing hybrids or training.')
    parser.add_argument('--epochs', type=int, help='Override DRAEM epochs for both variants.')
    parser.add_argument('--texture-root', type=str, help='Shared DTD texture directory.')
    args = parser.parse_args()
    downstream = create_downstream_configuration()
    downstream.seed = args.seed
    downstream.data.normal_fraction = .5
    downstream.data.hybrid_fraction = .5
    if args.epochs is not None:
        downstream.training.epochs = args.epochs
    if args.texture_root is not None:
        downstream.data.texture_root = args.texture_root
    categories = CATEGORIES if args.categories == ['all'] else tuple(args.categories)
    config = ComparisonConfiguration(categories=categories, seed=args.seed, downstream=downstream)
    run_comparison(config, args.dataset_root, args.output, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
