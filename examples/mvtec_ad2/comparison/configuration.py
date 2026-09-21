"""Shared settings for a complete paired experiment."""
from dataclasses import dataclass, field
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.splits import SplitConfiguration

CATEGORIES = ('can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts')

@dataclass
class ComparisonConfiguration:
    categories: tuple[str, ...] = CATEGORIES
    seed: int = 42
    test_fraction: float = .2
    validation_fraction: float = .2
    test_positive_fraction: float = .25
    downstream: DownstreamConfiguration = field(default_factory=DownstreamConfiguration)

    def validate(self):
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ValueError('Select unique categories.')
        if set(self.categories) - set(CATEGORIES):
            raise ValueError('Unknown category.')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Seed must be nonnegative.')
        if not 0 < self.test_positive_fraction < 1:
            raise ValueError('Test positive fraction must be between zero and one.')
        SplitConfiguration(self.test_fraction, self.validation_fraction, self.seed).validate()
        if self.test_fraction == 0:
            raise ValueError("A nonempty test split is required.")
        self.downstream.validate()
