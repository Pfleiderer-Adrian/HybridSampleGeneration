import os
import shutil
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
    layout_version: int = 1

    def __post_init__(self):
        self.study_folder = os.path.normpath(os.fspath(self.study_folder))

    def _join(self, *parts):
        return os.path.join(self.study_folder, *parts)

    @staticmethod
    def normalize_path(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def assert_inside_study_folder(self, folder):
        study_folder = self.normalize_path(self.study_folder)
        folder = self.normalize_path(folder)
        try:
            common_path = os.path.commonpath([study_folder, folder])
        except ValueError as exc:
            raise ValueError(f"Artifact folder is outside the study folder: {folder}") from exc
        if common_path != study_folder:
            raise ValueError(f"Artifact folder is outside the study folder: {folder}")

    @staticmethod
    def folder_has_contents(folder):
        if not os.path.isdir(folder):
            return False
        with os.scandir(folder) as entries:
            return any(entries)

    def confirm_and_clear_artifact_dirs(self, *folders):
        for folder in folders:
            self.assert_inside_study_folder(folder)
            if os.path.exists(folder) and not os.path.isdir(folder):
                raise NotADirectoryError(f"Expected artifact folder, found file: {folder}")

        non_empty_folders = [folder for folder in folders if self.folder_has_contents(folder)]
        if non_empty_folders:
            print("The following artifact folders are not empty and must be cleared before continuing:")
            for folder in non_empty_folders:
                print(f"  - {folder}")
            try:
                answer = input("Delete folder contents and continue? [y/N]: ").strip().lower()
            except EOFError as exc:
                raise RuntimeError("Cannot confirm deletion in a non-interactive command line. Aborting.") from exc

            if answer not in {"y", "yes", "j", "ja"}:
                raise RuntimeError("Aborted by user. No artifact folders were cleared.")

        for folder in folders:
            if os.path.isdir(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

    @property
    def configuration_file(self):
        return self._join("configuration.json")

    @property
    def anomaly_transformations_file(self):
        return self._join("anomaly_transformations.json")

    @property
    def matching_dict_file(self):
        return self._join("matching_dict.csv")

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
    def anomaly_data(self):
        return self._join("anomaly_data")

    @property
    def anomaly_roi_data(self):
        return self._join("anomaly_roi_data")

    @property
    def anomaly_mask_data(self):
        return self._join("anomaly_mask_data")

    @property
    def anomaly_tgt_mask_data(self):
        return self._join("anomaly_tgt_mask_data")

    @property
    def synth_anomaly_data(self):
        return self._join("synth_anomaly_data")

    @property
    def synth_anomaly_mask_data(self):
        return self._join("synth_anomaly_mask_data")
    
    @property
    def anomaly_mask_roi_data(self):
        return self._join("anomaly_mask_roi_data")

    @property
    def synth_roi_data(self):
        return self._join("synth_roi_data")
    
    @property
    def synth_roi_mask_data(self):
        return self._join("synth_roi_mask_data")

    @property
    def generated_hybrid_samples(self):
        return self._join("generated_hybrid_samples")

    @property
    def generated_images_npy(self):
        return os.path.join(self.generated_hybrid_samples, "images_npy")

    @property
    def generated_segmentations_npy(self):
        return os.path.join(self.generated_hybrid_samples, "segmentations_npy")

    @property
    def generated_images(self):
        return os.path.join(self.generated_hybrid_samples, "images")

    @property
    def generated_segmentations(self):
        return os.path.join(self.generated_hybrid_samples, "segmentations")

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
