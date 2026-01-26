from __future__ import annotations
import os.path
from typing import Iterator
import numpy as np
from typing import Tuple
import optuna
import pandas as pd
import torch

from data_handler import InlineDataset
from data_handler.AnomalyDataset import AnomalyDataset, save_numpy_as_npy

from models.VAE_Diffusion_inpaint_2D import DefectPatchGenerator, ModelConfig
from synthesizer.functions_2D.Fusion2D import fusion2d
from synthesizer.InpaintConfiguration import InpaintConfiguration
from synthesizer.functions_3D.Fusion3D import fusion3d
from synthesizer.functions_2D.Matching2D import create_matching_dict2d
from synthesizer.functions_3D.Matching3D import create_matching_dict3d
from synthesizer.Trainer import optimize


class InpaintDataGenerator:
    """
    Wraps the entire Inpaint data generation workflow in a single class.

    Supported input shapes obtained from an external iterator/dataloader:
      - 2D: (C, H, W)
      - 3D: (C, D, H, W)

    Requirements:
      - All images and masks must have exactly the same shape.
      - ndim must be 3 (C + H + W) or 4 (C + D + H + W).
      - dataloader must yield: image (np.ndarray), mask (np.ndarray), basename (str).
    """

    def __init__(self, config: InpaintConfiguration) -> None:
        """
        Initialize the Inpaint data generator with a configuration.

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
        self._model = None

    def _log_step(self, message: str) -> None:
        print(f"[InpaintDataGenerator] {message}")

    def extract_and_save(
        self,
        sample_dataloader: Iterator[Tuple[np.ndarray, np.ndarray, str]],
        save_folder=None,
        save_folder_roi=None,
        rng=None,
    ):
        """
        Extract anomaly cutouts (and anomaly ROIs) from samples that contain anomalies.

        This method:
          1) Iterates over the provided dataloader (img, seg, basename).
          2) Skips samples without anomalies (empty segmentation).
          4) Saves samples as .npy to disk for training.
          6) Stores class distributon meta-data per sample in the config and persists it.
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
        random_offset:
            If True, apply random spatial offsets to anomaly cutouts after resize+pad.
        rng:
            Optional numpy random Generator for reproducible offsets.
        Note:
            Normalization is controlled via config.normalization and config.normalization_eps.

        Outputs
        -------
        None
            Side effects: writes .npy files, updates config transformation file, sets self._anomaly_dataset.
        """
        self._log_step("Step 1/9: Prepare samples for training.")
        if save_folder is None:
            save_folder = os.path.join(self._config.study_folder, "train_data")    
        os.makedirs(save_folder, exist_ok=True)
        img_folder = os.path.join(save_folder, "images")
        seg_folder = os.path.join(save_folder, "segmentations")
        ori_seg_folder = os.path.join(seg_folder, "original")
        binary_seg_folder = os.path.join(seg_folder, "binary")
        global_labels = set()


        for img, seg, basename in sample_dataloader:
            # check if sample has no anomaly
            if not np.any(seg):
                # continue if sample contains no anomaly
                continue

            if not (img.ndim == 3 or img.ndim == 4):
                raise ValueError(f"Unexpected shape: {img.shape}, Supported: (C, H, W) or (C, D, H, W)")

            # save as npy
            save_numpy_as_npy(img, os.path.join(img_folder, basename+".npy"), overwrite=True)
            save_numpy_as_npy(seg, os.path.join(ori_seg_folder, basename+".npy"), overwrite=True)
            seg_bin = (seg > 0).astype(np.uint8)
            save_numpy_as_npy(seg_bin, os.path.join(binary_seg_folder, basename+".npy"), overwrite=True)

            # one mask per *present* label (excluding 0)
            labels = np.unique(seg)
            labels = labels[labels != 0]
            for label in labels:
                global_labels.add(label)

            masks = {int(k): (seg == k).astype(np.uint8) for k in labels}
            
            for mask in masks:
                i = np.max(mask)
                class_seg_folder = os.path.join(seg_folder, str(i))
                os.makedirs(class_seg_folder, exist_ok=True)
                save_numpy_as_npy(mask, os.path.join(class_seg_folder, basename+".npy"), overwrite=True)

            self._config.add_sample_metadata(basename, labels)

        self._config.save_metadata()
        self._config.save_config_file()

    def load_data(self, data_folder=None, segmentation="binary"):
        """
        Load previously extracted anomaly cutouts into an InlineDataset3D and
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
        self._log_step("Step 2/9: Loading anomaly dataset.")
        if data_folder is None:
            data_folder = os.path.join(self._config.study_folder, "train_data")
        img_folder = os.path.join(data_folder, "images")
        seg_folder = os.path.join(data_folder, segmentation)
        _dataset = InlineDataset(
            img_folder,
            seg_folder,
            return_filename=True,
            load_to_ram=True,
            dtype=torch.float32,
        )
        self._config.load_anomaly_transformations()
        return _dataset

    def train_base_generator(self, no_of_trails):
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
        self._log_step("Step 3/9: Training generator model (with Optuna optimization).")
        _dataset = self.load_data("binary")
        if self._dataset is None:
            raise ValueError(f"No Anomalies Loaded: Run extract_anomalies or load_anomalies first")
        else:
            optimize(no_of_trails, self._config, _dataset)


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
        self._log_step("Step 4/9: Loading trained generator model.")
        if path_to_db_file is None:
            path_to_db_model = "sqlite:///" + str(os.path.join(self._config.study_folder, self._config.study_name + ".db"))  # Speicherort der Datenbank
        else:
            path_to_db_model = "sqlite:///" + path_to_db_file

        study = optuna.load_study(
            study_name=self._config.study_name,  # Name der Studie
            storage=path_to_db_model
        )

        t = None
        if trial_id == -1:
            t = study.best_trial
        else:
            all_trials = study.get_trials()
            for ts in all_trials:
                if ts.number == trial_id:
                    t = ts
                    break
        print()
        print("Loaded Model Number: "+str(t.number))
        print(t.user_attrs)

        # load model from study
        params = t.user_attrs['params']
        model_name = t.user_attrs['model_name']
        if model_name == "VAE_Diffusion_inpaint_3D":
            # ToDo
            pass
        elif model_name == "VAE_Diffusion_inpaint_2D":
            self._model = DefectPatchGenerator(ModelConfig(**params))
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self._model.warmup(self._config.anomaly_size)
        self._model.load_state_dict(torch.load(t.user_attrs['model_path']))
    

    def finetune_classes(self, class_name):
        if self._model is None:
            raise ValueError(f"No model loaded. Train and load base model first. Exit!")
        pass
    """
        #todo
        #for class in classes:
        #    self.load_data(segmentation=class_name)
        



        
        #todo set model train mode to -> finetune
        #refactor model save path

        for class_number in range(1,self._config.num_classes+1):
            _dataset = self.load_data(str(class_number))
            optimize(no_of_trails, self._config, _dataset)
        #todo set model train mode to -> finetune
        
   """

    def generate_synth_samples(self, save_folder=None):
        #todo
        pass
