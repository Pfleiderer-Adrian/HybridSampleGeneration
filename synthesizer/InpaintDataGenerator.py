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

from models.VAE_ConvNeXt_2D import ConvNeXtVAE2D, Config as ConvNeXtVAE2D_Config
from models.VAE_ConvNeXt_3D import ConvNeXtVAE3D, Config as ConvNeXtVAE3D_Config
from models.VAE_Diffusion_inpaint_2D import DefectPatchGenerator, ModelConfig
from models.VAE_ResNet_2D import ResNetVAE2D, Config as ResNetVAE2D_Config
from models.VAE_ResNet_3D import ResNetVAE3D, Config as ResNetVAE3D_Config
from synthesizer.functions_2D.Anomaly_Extraction2D import crop_and_center_anomaly_2d
from synthesizer.functions_2D.Fusion2D import fusion2d
from synthesizer.functions_3D.Anomaly_Extraction3D import crop_and_center_anomaly_3d
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
            masks = {int(k): (seg == k).astype(np.uint8) for k in labels}
            
            for mask in masks:
                i = np.max(mask)
                class_seg_folder = os.path.join(seg_folder, str(i))
                os.makedirs(class_seg_folder, exist_ok=True)
                save_numpy_as_npy(mask, os.path.join(class_seg_folder, basename+".npy"), overwrite=True)

            self._config.add_sample_metadata(basename, labels)

        self._config.save_metadata()

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
    
    def finetune_classes(self, no_of_trails):
        pass
        #todo set model train mode to -> finetune
        #refactor model save path
        for class_number in range(1,self._config.num_classes+1):
            _dataset = self.load_data(str(class_number))
            optimize(no_of_trails, self._config, _dataset)
        #todo set model train mode to -> finetune
        
   
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
        anomaly_size = t.user_attrs['anomaly_size']
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
        self._log_step("Step 5/9: Generating synthetic anomalies.")
        if self._anomaly_dataset is None:
            raise ValueError(f"No Anomalies Loaded: Run extract_anomalies or load_anomalies first")
        if self._model is None:
            raise ValueError(f"No Model Loaded: Run train_generator or load_generator first")
        if save_folder is None:
            save_folder = os.path.join(self._config.study_folder, "synth_anomaly_data")
        os.makedirs(save_folder, exist_ok=True)

        for img, basename in self._anomaly_dataset:
            syn_anomaly_sample = self._model.generate_synth_sample(img, clamp_01=self._config.clamp01_output)
            save_numpy_as_npy(syn_anomaly_sample, str(os.path.join(save_folder, basename)), overwrite=True)

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
        self._log_step("Step 6/9: Loading synthetic anomalies and transformation metadata.")
        if synth_anomaly_folder is None:
            synth_anomaly_folder = os.path.join(self._config.study_folder, "synth_anomaly_data")
        if transformation_file is None:
            transformation_file = os.path.join(self._config.study_folder, "anomaly_transformations.json")

        self._config.load_anomaly_transformations(transformation_file)
        self._synth_anomaly_dataset = AnomalyDataset(
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
        self._log_step("Step 7/9: Creating matching dictionary between controls and anomalies.")
        if self._synth_anomaly_dataset is None:
            raise ValueError(f"No Synth Anomalies Loaded: Run generate_synth_anomalies or load_synth_anomalies first")
        if roi_folder is None:
            roi_folder = os.path.join(self._config.study_folder, "anomaly_roi_data")
        if csv_file_path is None:
            csv_file_path = os.path.join(self._config.study_folder, "matching_dict.csv")
        _roi_dataset = AnomalyDataset(
            roi_folder,
            return_filename=True,
            load_to_ram=True,
            dtype=torch.float32,
            numpy_mode=True
        )
        img = _roi_dataset.__getitem__(0)[0]
        if img.ndim == 3:
            _data = create_matching_dict2d(control_samples_dataloader, _roi_dataset, self._config, matching_routine=matching_routine, anomaly_duplicates=True)
        elif img.ndim == 4:
            _data = create_matching_dict3d(control_samples_dataloader, _roi_dataset, self._config, matching_routine=matching_routine, anomaly_duplicates=True)
        else:
            raise ValueError(f"Unexpected shape: {img.shape}, Supported: (C, H, W) or (C, D, H, W)")

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
        self._log_step("Step 8/9: Loading matching dictionary.")
        if csv_file_path is None:
            csv_file_path = os.path.join(self._config.study_folder, "matching_dict.csv")
        self._config.load_matching_csv(csv_file_path)


    def fusion_synth_anomalies(self, control_samples_array, basename_of_control_sample, save_npy=True, save_path=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse a synthetic anomaly into a control sample according to the loaded matching dict.

        The method:
          - finds the matched anomaly basename + position factor for the control sample
          - loads the corresponding synthetic anomaly volume
          - looks up the scale factor from stored transformations
          - calls fusion3d/2d(...) to obtain (fused_image, fused_segmentation)

        Inputs
        ------
        control_samples_array:
            The control image/volume as a numpy array (shape must match what fusion3d or fusion2d expects).
        basename_of_control_sample:
            Key used to look up the matched anomaly and fusion position in `self._config.matching_dict`.
        save_npy
            if True: saves generated hybrid images and segmentations as .npy -> for visualizer / debugging
        save_path:
            path to folder for npy files - only wenn save_npy is True

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

        anomaly_meta = self._config.syn_anomaly_transformations.get(anomaly_basename, {})

        # Warn if user enabled normalization but transformations do not include normalization metadata.
        if self._config.normalization is not None:
            norm_type = anomaly_meta.get("norm_type", None)
            if norm_type is None:
                self._log_step(
                    f"WARNING: config.normalization={self._config.normalization!r} but "
                    f"no normalization metadata found for {anomaly_basename!r}. "
                    f"Re-extract anomalies to populate norm_* fields (and re-generate synth anomalies)."
                )

        if control_samples_array.ndim == 3:
            img, seg = fusion2d(
                control_samples_array,
                synth_anomaly_image,
                anomaly_meta,
                fusion_position,
                self._config,
            )
        elif control_samples_array.ndim == 4:
            img, seg = fusion3d(
                control_samples_array,
                synth_anomaly_image,
                anomaly_meta,
                fusion_position,
                self._config,
            )
        else:
            raise ValueError(f"Unexpected shape: {control_samples_array.shape}, Supported: (C, H, W) or (C, D, H, W)")

        if save_npy:
            if save_path is None:
                save_path = os.path.join(self._config.study_folder, "generated_hybrid_samples")
            img_folder = os.path.join(save_path,"images_npy")
            seg_folder = os.path.join(save_path,"segmentations_npy")
            os.makedirs(img_folder, exist_ok=True)
            os.makedirs(seg_folder, exist_ok=True)

            img_path = os.path.join(img_folder, anomaly_basename)
            seg_path = os.path.join(seg_folder, anomaly_basename)
            save_numpy_as_npy(img, img_path, overwrite=True)
            save_numpy_as_npy(seg, seg_path, overwrite=True)

        return img, seg
    
