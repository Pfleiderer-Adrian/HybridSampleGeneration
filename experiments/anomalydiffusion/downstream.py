"""Run a new source mixture with unchanged frozen V3 downstream settings."""
import argparse
import json
from pathlib import Path

from hybrid_sample_generator.configuration.root import load_config_file
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream

from .prepare import sha256
from .mixtures import PROFILES, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.inputs / "protocol.json").read_text())
    split = json.loads((args.inputs / "split_manifest.json").read_text())
    generator_path = Path(protocol["hybrid_study"]) / "configuration.json"
    if sha256(generator_path) != protocol["hybrid_configuration_sha256"]:
        raise ValueError("The selected frozen hybrid study configuration has changed.")
    generator_config = load_config_file(generator_path)
    config = DownstreamConfiguration.load(args.inputs / f"{args.variant}.json")
    quotas, rotation = profile(args.variant)
    mixed = quotas.get("hybrid", 0) > 0
    if config.data.source_quotas != quotas or config.data.source_quota_rotation != rotation:
        raise ValueError("Source quotas differ from the declared variant protocol.")
    if Path(config.data.anomalydiffusion_manifest).resolve() != (args.inputs / "generated/manifest.json").resolve():
        raise ValueError("Both variants must use the same prepared AnomalyDiffusion cache.")
    reference = DownstreamConfiguration.load(Path(protocol["reference_run"]) / "configuration.json")
    actual, expected = config.to_dict(), reference.to_dict()
    # Only the synthetic source mixture and its cache location may change.
    allowed = ("source_quotas", "source_quota_rotation", "hybrid_fraction", "anomalydiffusion_manifest")
    for key in allowed:
        actual["data"].pop(key, None)
        expected["data"].pop(key, None)
    if actual != expected:
        raise ValueError("Downstream configuration differs from the frozen V3 reference.")
    hybrid_ids = None
    if mixed:
        provenance_path = args.inputs / "hybrid_provenance.json"
        if sha256(provenance_path) != protocol["hybrid_provenance_sha256"]:
            raise ValueError("Frozen hybrid selection has changed.")
        provenance = json.loads(provenance_path.read_text())
        hybrid_ids = [r["hybrid_id"] for r in provenance]
        if not hybrid_ids:
            raise ValueError("Frozen V3 hybrid pool is empty.")
    run = train_downstream(generator_config, split, config, output_root=args.output, hybrid_ids=hybrid_ids)
    evaluate_downstream(run, split)


if __name__ == "__main__":
    main()
