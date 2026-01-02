from __future__ import annotations
import os.path
from typing import Iterator
import numpy as np
from typing import Tuple
import optuna
import pandas as pd
import torch

from data_handler.AnomalyDataset3D import AnomalyDataset3D, save_numpy_as_npy
from models.VAE_ResNet_3D import ResNetVAE3D, Config
from synthesizer.Anomaly_Extraction import crop_and_center_anomaly_3d
from synthesizer.Configuration import Configuration
from synthesizer.Fusion3D import fusion3d, create_matching_dict
from synthesizer.Trainer import optimize


class HybridDataGenerator:
    """
    Wraps the entire hybrid data generation workflow in a single class.

    Supported input shapes obtained from an external iterator/dataloader:
      - 2D: (C, H, W)
      - 3D: (C, D, H, W)

    Requirements:
      - All images and masks must have exactly the same shape.
      - ndim must be 3 (C + H + W) or 4 (C + D + H + W).
      - dataloader must yield: image (np.ndarray), mask (np.ndarray), basename (str).
    """

    def __init__(self, config: Configuration) -> None:
        """
        Initialize the hybrid data generator with a configuration.

        Inputs
        ------
        config:
            Configuration object containing paths, study name, anomaly size, fusion params, thresholds, etc.

        Outputs
        -------
        None
            Initializes internal state (datasets/model start as None).
        """
        self._config = config
        self._anomaly_dataset = None
        self._synth_anomaly_dataset = None
        self._model = None

    def extract_anomalies(self, sample_dataloader:Iterator[Tuple[np.ndarray, np.ndarray, str]], save_folder=None, save_folder_roi=None):
        """
        Extract anomaly cutouts (and anomaly ROIs) from samples that contain anomalies.

        This method:
          1) Iterates over the provided dataloader (img, seg, basename).
          2) Skips samples without anomalies (empty segmentation).
          3) Crops and centers anomaly cutouts and ROI cutouts.
          4) Saves anomaly cutouts to disk for training.
          5) Saves ROI cutouts to disk for matching.
          6) Stores transformation meta-data per anomaly in the config and persists it.
          7) Loads the anomaly dataset from disk (self.load_anomalies).

        Inputs
        ------
        sample_dataloader:
            Iterator yielding tuples (img, seg, basename):
              - img: np.ndarray, the image/volume
              - seg: np.ndarray, the anomaly mask (same shape as img)
              - basename: str, unique sample identifier used for filenames with extension (e.g. .nii)
        save_folder:
            Folder for saving anomaly cutouts (.npy). If None:
            "<study_folder>/anomaly_data"
        save_folder_roi:
            Folder for saving anomaly ROI cutouts (.npy). If None:
            "<study_folder>/anomaly_roi_data"

        Outputs
        -------
        None
            Side effects: writes .npy files, updates config transformation file, sets self._anomaly_dataset.
        """
        if save_folder is None:
            save_folder = os.path.join(self._config.study_folder, "anomaly_data")
        if save_folder_roi is None:
            save_folder_roi = os.path.join(self._config.study_folder, "anomaly_roi_data")
        os.makedirs(save_folder, exist_ok=True)
        for img, seg, basename in sample_dataloader:
            # check if sample has no anomaly
            if not np.any(seg):
                # continue if sample contains no anomaly
                continue
            anomalies, anomalies_roi = crop_and_center_anomaly_3d(img, seg, self._config.anomaly_size)

            i = 0
            # save anomaly cutout for training
            for anomaly_sample, meta_data in anomalies:
                save_numpy_as_npy(anomaly_sample, os.path.join(save_folder, basename+"_"+str(i)+".npy"), overwrite=True)
                self._config.add_anomaly_transformation(basename+"_"+str(i)+".npy", meta_data)
                i = i + 1
            i = 0
            # save roi of anomaly for matching
            for roi_sample in anomalies_roi:
                save_numpy_as_npy(roi_sample, os.path.join(save_folder_roi, basename+"_"+str(i)+".npy"), overwrite=True)
                i = i + 1
        self._config.save_anomaly_transformations()
        self.load_anomalies(anomaly_folder=save_folder)

    def load_anomalies(self, anomaly_folder=None):
        """
        Load previously extracted anomaly cutouts into an AnomalyDataset3D and
        load their transformation meta-data from config.

        Inputs
        ------
        anomaly_folder:
            Folder containing anomaly .npy files.
            If None: "<study_folder>/anomaly_data"

        Outputs
        -------
        None
            Side effects: sets self._anomaly_dataset and loads config transformations into memory.
        """
        if anomaly_folder is None:
            anomaly_folder = os.path.join(self._config.study_folder, "anomaly_data")
        self._anomaly_dataset = AnomalyDataset3D(
            anomaly_folder,  # oder samples=[...]
            return_filename=True,
            load_to_ram=True,
            dtype=torch.float32,
        )
        self._config.load_anomaly_transformations()

    def train_generator(self, no_of_trails):
        """
        Train/optimize the generator model using Optuna via `optimize(...)`,
        then load the resulting model via `load_generator()`.

        Inputs
        ------
        no_of_trails:
            Number of Optuna trials to run (hyperparameter optimization / training runs).

        Outputs
        -------
        None
            Side effects: writes Optuna study DB, writes model weights, sets self._model.
        """
        if self._anomaly_dataset is None:
            raise ValueError(f"No Anomalies Loaded: Run extract_anomalies or load_anomalies first")
        else:
            optimize(no_of_trails, self._config, self._anomaly_dataset)

        self.load_generator()

    def load_generator(self, path_to_db_file=None, trial_id=-1):
        """
        Load a trained generator model from an Optuna study database.

        Inputs
        ------
        path_to_db_file:
            Path to the SQLite DB file containing the Optuna study.
            If None: "<study_folder>/<study_name>.db"
        trial_id:
            Which trial to load:
              - -1: load best trial (study.best_trial)
              - otherwise: load trial with matching ts.number

        Outputs
        -------
        None
            Side effects: sets self._model and loads its weights.
        """
        if path_to_db_file is None:
            path_to_db_model = "sqlite:///" + str(os.path.join(self._config.study_folder, self._config.study_name + ".db"))  # Speicherort der Datenbank
        else:
            path_to_db_model = "sqlite:///" + path_to_db_file

        study = optuna.load_study(
            study_name=self._config.study_name,  # Name der Studie
            storage=path_to_db_model
        )
        print(study.user_attrs)

        t = None
        if trial_id == -1:
            t = study.best_trial
        else:
            all_trials = study.get_trials()
            for ts in all_trials:
                if ts.number == trial_id:
                    t = ts
                    break

        # load model from study
        params = t.user_attrs['params']
        anomaly_size = t.user_attrs['anomaly_size']
        model_name = t.user_attrs['model_name']
        if model_name == "VAE_ResNet_3D":
            self._model = ResNetVAE3D(anomaly_size[0], Config(**params))
        self._model.warmup(self._config.anomaly_size)
        self._model.load_state_dict(torch.load(t.user_attrs['model_path']))

    def generate_synth_anomalies(self, save_folder=None):
        """
        Generate synthetic anomalies for each anomaly in the loaded anomaly dataset.

        For each (img, basename) in anomaly dataset:
          - run model.generate_synth_sample(img)
          - save output as .npy under the same basename

        Inputs
        ------
        save_folder:
            Output folder for synthetic anomaly .npy files.
            If None: "<study_folder>/synth_anomaly_data"

        Outputs
        -------
        None
            Side effects: writes synthetic .npy files and sets self._synth_anomaly_dataset via load_synth_anomalies().
        """
        if self._anomaly_dataset is None:
            raise ValueError(f"No Anomalies Loaded: Run extract_anomalies or load_anomalies first")
        if self._model is None:
            raise ValueError(f"No Model Loaded: Run train_generator or load_generator first")
        if save_folder is None:
            save_folder = os.path.join(self._config.study_folder, "synth_anomaly_data")
        os.makedirs(save_folder, exist_ok=True)

        for img, basename in self._anomaly_dataset:
            syn_anomaly_sample = self._model.generate_synth_sample(img)
            save_numpy_as_npy(syn_anomaly_sample, str(os.path.join(save_folder, basename)))

        self.load_synth_anomalies(save_folder)

    def load_synth_anomalies(self, synth_anomaly_folder=None, transformation_file=None):
        """
        Load synthetic anomalies from disk and load the anomaly transformations file.

        Inputs
        ------
        synth_anomaly_folder:
            Folder containing synthetic anomaly .npy files.
            If None: "<study_folder>/synth_anomaly_data"
        transformation_file:
            Path to anomaly transformations file.
            If None: "<study_folder>/anomaly_transformations.json"

        Outputs
        -------
        None
            Side effects: sets self._synth_anomaly_dataset and loads transformation meta-data into config.
        """
        if synth_anomaly_folder is None:
            synth_anomaly_folder = os.path.join(self._config.study_folder, "synth_anomaly_data")
        if transformation_file is None:
            transformation_file = os.path.join(self._config.study_folder, "anomaly_transformations.json")

        self._config.load_anomaly_transformations(transformation_file)
        self._synth_anomaly_dataset = AnomalyDataset3D(
            synth_anomaly_folder,  # oder samples=[...]
            return_filename=True,
            load_to_ram=True,
            dtype=torch.float32,
        )

    def create_matching_dict(self, control_samples_dataloader:Iterator[Tuple[np.ndarray, np.ndarray, str]], matching_routine="local", roi_folder=None, csv_file_path=None):
        """
        Create a matching dictionary between control samples and anomaly ROI samples.

        The matching result is written to a CSV with columns:
          - control
          - anomaly
          - position_factor

        Then `load_matching_dict(...)` is called to load it into `self._config.matching_dict`.

        Inputs
        ------
        control_samples_dataloader:
            Iterator yielding (img, seg, basename) for control samples.
            (seg can be present even for controls; matching function decides how to use it.)
        matching_routine:
            One of: "local", "global", "fixed_from_extraction"
              - "local": match ith control with ith ROI (sequential) and compute best template position (resource middle - O(n)!)
              - "global": for each control, search over all ROI and pick best similarity (resource heavy - O(n2)!!)
              - "fixed_from_extraction": sequential mapping, take centroid from extraction metadata (resource lite)
        roi_folder:
            Folder containing ROI .npy files (saved during extract_anomalies()).
            If None: "<study_folder>/anomaly_roi_data"
        csv_file_path:
            Output path for the matching CSV.
            If None: "<study_folder>/matching_dict.csv"

        Outputs
        -------
        None
            Side effects: writes CSV and populates self._config.matching_dict.
        """
        if self._synth_anomaly_dataset is None:
            raise ValueError(f"No Synth Anomalies Loaded: Run generate_synth_anomalies or load_synth_anomalies first")
        if roi_folder is None:
            roi_folder = os.path.join(self._config.study_folder, "anomaly_roi_data")
        if csv_file_path is None:
            csv_file_path = os.path.join(self._config.study_folder, "matching_dict.csv")
        _roi_dataset = AnomalyDataset3D(
            roi_folder,
            return_filename=True,
            load_to_ram=True,
            dtype=torch.float32,
            numpy_mode=True
        )

        _data = create_matching_dict(control_samples_dataloader, _roi_dataset, self._config, matching_routine=matching_routine)
        df_detection = pd.DataFrame(_data, columns=["control", "anomaly", "position_factor"])
        df_detection.to_csv(csv_file_path, sep=',', encoding='utf-8', index=False)

        self.load_matching_dict(csv_file_path)

    def load_matching_dict(self, csv_file_path=None):
        """
        Load a matching CSV into `self._config.matching_dict`.

        Inputs
        ------
        csv_file_path:
            Path to the matching CSV file.
            If None: "<study_folder>/matching_dict.csv"

        Outputs
        -------
        None
            Side effects: updates self._config.matching_dict.
        """
        if csv_file_path is None:
            csv_file_path = os.path.join(self._config.study_folder, "matching_dict.csv")
        self._config.load_matching_csv(csv_file_path)


    def fusion_synth_anomalies(self, control_samples_array, basename_of_control_sample) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse a synthetic anomaly into a control sample according to the loaded matching dict.

        The method:
          - finds the matched anomaly basename + position factor for the control sample
          - loads the corresponding synthetic anomaly volume
          - looks up the scale factor from stored transformations
          - calls fusion3d(...) to obtain (fused_image, fused_segmentation)

        Inputs
        ------
        control_samples_array:
            The control image/volume as a numpy array (shape must match what fusion3d expects).
        basename_of_control_sample:
            Key used to look up the matched anomaly and fusion position in `self._config.matching_dict`.

        Outputs
        -------
        img:
            np.ndarray, the fused image/volume.
        seg:
            np.ndarray, the produced segmentation mask of the inserted anomaly (same spatial shape as img).
        """
        if self._synth_anomaly_dataset is None:
            raise ValueError(f"No Synth Anomalies Loaded: Run generate_synth_anomalies or load_synth_anomalies first")
        if len(self._config.matching_dict) < 1:
            raise ValueError(f"No Matching Dict Loaded: Run create_matching_dict or load_matching_dict first")

        anomaly_basename = self._config.matching_dict[basename_of_control_sample]["anomaly"]
        fusion_position = self._config.matching_dict[basename_of_control_sample]["position_factor"]
        synth_anomaly_image = self._synth_anomaly_dataset.load_numpy_by_basename(anomaly_basename)
        img, seg = fusion3d(control_samples_array, synth_anomaly_image, self._config.syn_anomaly_transformations[anomaly_basename]["scale_factor"], fusion_position, self._config.fusion_mask_params)

        return img, seg