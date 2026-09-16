"""MVTec study and experiment configuration types."""

from dataclasses import dataclass, field
import json
from pathlib import Path

from hybrid_sample_generator.configuration.root import Configuration as GeneratorConfiguration
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration


class Configuration(GeneratorConfiguration):
    """MVTec-specific configuration; the library remains downstream-independent."""

    def __init__(self, *args, **kwargs):
        self.downstream = DownstreamConfiguration()
        super().__init__(*args, **kwargs)

    def validate(self):
        super().validate()
        self.downstream.validate()

    def to_dict(self):
        values = super().to_dict()
        values["downstream"] = self.downstream.to_dict()
        return values

    @classmethod
    def from_dict(cls, values):
        config = super().from_dict(values)
        config.downstream = DownstreamConfiguration.from_dict(values.get("downstream", {}))
        config.validate()
        return config


def load_config_file(path):
    """Load the example-specific extension as well as generator settings."""
    return Configuration.from_dict(json.loads(Path(path).read_text()))


@dataclass
class SplitConfiguration:
    """Dataset-wide split, fixed before any generator training."""

    test_enabled: bool = True
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    seed: int = 42

    def validate(self):
        if type(self.test_enabled) is not bool or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Split requires boolean test_enabled and nonnegative integer seed.")
        for value in (self.test_fraction, self.validation_fraction):
            if type(value) not in (float, int) or not 0 <= value < 1:
                raise ValueError("Split fractions must be in [0, 1).")
        test = self.test_fraction if self.test_enabled else 0
        if self.validation_fraction <= 0 or test + self.validation_fraction >= 1:
            raise ValueError("Reserve positive training and validation fractions.")
        if self.test_enabled and test <= 0:
            raise ValueError("test_fraction must be positive when testing is enabled.")


@dataclass(frozen=True)
class Experiment:
    """Inputs for new studies; model settings live in presets.py."""

    dataset_root: Path
    output_root: Path
    categories: tuple[str, ...]
    split: SplitConfiguration = field(default_factory=SplitConfiguration)
