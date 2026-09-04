from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from fusion_backend.fusion_configuration import FusionSettings
from fusion_backend.fusion_registry import registered_fusion_backend_names
from generation_models.model_configuration import GeneratorModelConfiguration
from generation_models.model_registry import get_model_spec, registered_model_names
from synthesizer.configuration.augmentation import AugmentationConfiguration
from synthesizer.configuration.evaluation import EvaluationConfiguration
from synthesizer.configuration.extraction import ExtractionConfiguration
from synthesizer.configuration.generation import GenerationConfiguration
from synthesizer.configuration.matching import MatchingConfiguration
from synthesizer.configuration.study import StudyConfiguration
from synthesizer.configuration.training import TrainingConfiguration


ALLOWED_MODELS = registered_model_names()
ALLOWED_FUSION_BACKENDS = registered_fusion_backend_names()


class Configuration:
    """Root configuration composed of small, domain-specific sections.

    This object contains requested pipeline behavior only. Generated entities,
    relationships and anomaly metadata live in the study repository.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self,
        study_name: str,
        model_name: str,
        anomaly_size,
        save_path=None,
        *,
        study_folder=None,
    ) -> None:
        if model_name not in ALLOWED_MODELS:
            raise ValueError(
                f"Model {model_name!r} is not supported. Currently supported: {ALLOWED_MODELS}"
            )
        if save_path is not None and study_folder is not None:
            raise ValueError("Use either save_path or study_folder, not both.")

        if study_folder is None:
            root = os.getcwd() if save_path is None else os.fspath(save_path)
            study_folder = os.path.join(root, "results", study_name)

        model_spec = get_model_spec(model_name)
        self.schema_version = self.SCHEMA_VERSION
        self.study = StudyConfiguration(name=study_name, folder=study_folder)
        self.extraction = ExtractionConfiguration(anomaly_size=tuple(anomaly_size))
        self.augmentation = AugmentationConfiguration()
        self.generation = GenerationConfiguration()
        self.matching = MatchingConfiguration(seed=self.study.seed)
        self.training = TrainingConfiguration()
        self.evaluation = EvaluationConfiguration()
        self.model = GeneratorModelConfiguration(
            name=model_name,
            parameters=model_spec.build_configuration(int(anomaly_size[0])),
            uses_masks=model_spec.uses_masks,
        )
        self.fusion = FusionSettings.for_backend("classical")
        self.validate()

    def validate(self) -> None:
        if self.model.name not in ALLOWED_MODELS:
            raise ValueError(f"Unknown model {self.model.name!r}.")
        if self.fusion.backend not in ALLOWED_FUSION_BACKENDS:
            raise ValueError(f"Unknown fusion backend {self.fusion.backend!r}.")
        model_spec = get_model_spec(self.model.name)
        if self.model.uses_masks != model_spec.uses_masks:
            raise ValueError(
                f"model.uses_masks={self.model.uses_masks} conflicts with model {self.model.name!r}."
            )
        self.extraction.validate()
        expected_channels = int(self.extraction.anomaly_size[0])
        for bound_name, parameters in (
            ("min", self.model.parameters.min),
            ("max", self.model.parameters.max),
        ):
            configured_channels = parameters.get("in_channels")
            if configured_channels is not None and int(configured_channels) != expected_channels:
                raise ValueError(
                    f"model.parameters.{bound_name}.in_channels={configured_channels} conflicts with "
                    f"extraction.anomaly_size channels={expected_channels}."
                )
        self.augmentation.validate()
        self.generation.validate()
        self.matching.validate()
        self.training.validate()
        self.evaluation.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _json_compatible(
            {
                "schema_version": self.schema_version,
                "study": asdict(self.study),
                "extraction": asdict(self.extraction),
                "augmentation": self.augmentation.to_dict(),
                "generation": asdict(self.generation),
                "matching": asdict(self.matching),
                "training": self.training.to_dict(),
                "evaluation": asdict(self.evaluation),
                "model": self.model.to_dict(),
                "fusion": self.fusion.to_dict(),
            }
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Configuration":
        required = {
            "schema_version",
            "study",
            "extraction",
            "augmentation",
            "generation",
            "matching",
            "training",
            "evaluation",
            "model",
            "fusion",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(
                "Configuration uses an unsupported schema; missing sections: "
                + ", ".join(sorted(missing))
            )
        if values["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported configuration schema {values['schema_version']!r}; "
                f"expected {cls.SCHEMA_VERSION}."
            )

        study_values = dict(values["study"])
        extraction_values = dict(values["extraction"])
        model_values = dict(values["model"])
        config = cls(
            study_values["name"],
            model_values["name"],
            extraction_values["anomaly_size"],
            study_folder=study_values["folder"],
        )
        config.schema_version = values["schema_version"]
        config.study = StudyConfiguration(**study_values)
        config.extraction = ExtractionConfiguration.from_dict(extraction_values)
        config.augmentation = AugmentationConfiguration.from_dict(values["augmentation"])
        config.generation = GenerationConfiguration.from_dict(values["generation"])
        config.matching = MatchingConfiguration(**values["matching"])
        config.training = TrainingConfiguration.from_dict(values["training"])
        config.evaluation = EvaluationConfiguration.from_dict(values["evaluation"])
        config.model = GeneratorModelConfiguration.from_dict(model_values)
        config.fusion = FusionSettings.from_dict(values["fusion"])
        config.validate()
        return config

    def save_config_file(self) -> str:
        """Validate and write this configuration as plain JSON."""
        json_path = Path(self.study.paths.configuration_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
            file.write("\n")
        return str(json_path)


def load_config_file(json_path) -> Configuration:
    """Load the current, section-based configuration schema from JSON."""
    with open(json_path, "r", encoding="utf-8") as file:
        values = json.load(file)
    return Configuration.from_dict(values)


def _json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
