"""AnomalyDiffusion pairs for MVTec AD 2 (profile of the V3 submissions: 2x, last 50k, AAR).

Run in the isolated environment created by setup.sh (see README.md):

    run.sh fetch-weights --output <model.ckpt>
    run.sh prepare  --category can --split <split_manifest.json> --output <inputs>
    run.sh train    --inputs <inputs> --checkpoint <model.ckpt>
    run.sh generate --inputs <inputs> --checkpoint <model.ckpt> --hybrid-study <study folder>

``generate`` writes ``<inputs>/generated/manifest.json`` for the downstream
mixtures ``real_anomalydiffusion*``.
"""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-weights", help="download and verify the base LDM checkpoint")
    fetch.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="defect windows from the split's train/validation defects")
    prepare.add_argument("--category", required=True)
    prepare.add_argument("--split", type=Path, required=True, help="split_manifest.json of the hybrid study")
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="appearance training (GPU)")
    train.add_argument("--inputs", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True, help="base LDM model.ckpt")
    train.add_argument("--output", type=Path, help="default: <inputs>/training")
    train.add_argument("--steps", type=int, default=None, help="default: 50000")
    generate = commands.add_parser("generate", help="pair generation (GPU)")
    generate.add_argument("--inputs", type=Path, required=True)
    generate.add_argument("--checkpoint", type=Path, required=True, help="base LDM model.ckpt")
    generate.add_argument("--hybrid-study", type=Path, required=True, help="study folder with artifacts.sqlite")
    generate.add_argument("--hybrid-ids", type=Path,
                          help="JSON with hybrid IDs, a provenance list or a downstream split_manifest.json")
    generate.add_argument("--training", type=Path, help="default: <inputs>/training")
    generate.add_argument("--output", type=Path, help="default: <inputs>/generated")
    generate.add_argument("--count", type=int, default=None, help="default: 1000")
    args = parser.parse_args(argv)

    if args.command == "fetch-weights":
        from .weights import fetch_weights
        print(fetch_weights(args.output))
    elif args.command == "prepare":
        from .prepare import prepare as run_prepare
        print(run_prepare(args.category, args.split, args.output))
    elif args.command == "train":
        from .training import STEPS, train as run_train
        print(run_train(args.inputs, args.checkpoint, args.output, steps=args.steps or STEPS))
    else:
        from .generation import COUNT, generate as run_generate
        print(run_generate(args.inputs, args.checkpoint, args.hybrid_study, args.output, training=args.training,
                           hybrid_ids=args.hybrid_ids, count=args.count or COUNT))


if __name__ == "__main__":
    main()
