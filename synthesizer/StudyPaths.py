import os
from dataclasses import dataclass


@dataclass
class StudyPaths:
    """
    Central path layout for all artifacts belonging to one study.

    Users should choose the study root once via Configuration(..., save_path=...).
    Pipeline code should use these managed paths instead of accepting ad-hoc
    output folders per step, because later steps and visualizers rely on this
    folder structure.
    """

    study_folder: str
    study_name: str
    layout_version: int = 2

    def __post_init__(self):
        self.study_folder = os.path.normpath(os.fspath(self.study_folder))

    def _join(self, *parts):
        return os.path.join(self.study_folder, *parts)

    @property
    def configuration_file(self):
        return self._join("configuration.json")

    @property
    def artifact_database(self):
        return self._join("artifacts.sqlite")

    @property
    def artifacts(self):
        return self._join("artifacts")

    @property
    def original_samples(self):
        return self._join("artifacts", "original_samples")

    @property
    def real_anomalies(self):
        return self._join("artifacts", "real_anomalies")

    @property
    def synthetic_anomalies(self):
        return self._join("artifacts", "synthetic_anomalies")

    @property
    def hybrid_samples(self):
        return self._join("artifacts", "hybrid_samples")

    @property
    def placements(self):
        return self._join("artifacts", "placements")

    @property
    def optuna_db_file(self):
        return self._join(f"{self.study_name}.db")

    @property
    def optuna_storage_url(self):
        return "sqlite:///" + str(self.optuna_db_file)

    @property
    def trained_models(self):
        return self._join("trained_models")

    @property
    def trained_fusion_backends(self):
        return self._join("trained_fusion_backends")

    @property
    def generated_images(self):
        return self._join("exports", "images")

    @property
    def generated_segmentations(self):
        return self._join("exports", "segmentations")

    @property
    def evaluation_results(self):
        return self._join("evaluation_results")

    @property
    def metric_diffs_csv(self):
        return os.path.join(self.evaluation_results, "metric_diffs.csv")

    @property
    def glcm_cutout_difference_histograms(self):
        return os.path.join(self.evaluation_results, "glcm_cutout_difference_histograms.png")

    @property
    def volume_cutout_difference_histograms(self):
        return os.path.join(self.evaluation_results, "volume_cutout_difference_histograms.png")

    @property
    def glcm_roi_difference_histograms(self):
        return os.path.join(self.evaluation_results, "glcm_roi_difference_histograms.png")
