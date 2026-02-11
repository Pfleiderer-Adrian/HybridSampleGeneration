from __future__ import annotations
import os.path
import shutil
from typing import Iterator
import numpy as np
from typing import Tuple
import optuna
import pandas as pd
import torch
from tqdm import tqdm

from data_handler.AnomalyDataset import AnomalyDataset, save_numpy_as_npy

from models.model_loader import model_loader
from synthesizer.functions_2D.Anomaly_Extraction2D import crop_and_center_anomaly_2d
from synthesizer.functions_2D.Fusion2D import fusion2d
from synthesizer.functions_3D.Anomaly_Extraction3D import crop_and_center_anomaly_3d
from synthesizer.Configuration import Configuration
from synthesizer.functions_3D.Fusion3D import fusion3d
from synthesizer.Matching import center_foreground_com, combine_binary_masks, create_matching_dictionary, crop_border, ssim_01, template_matching
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

    def _log_step(self, message: str) -> None:
        print(f"[HybridDataGenerator] {message}")

    def extract_anomalies(
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
        self._log_step("Step 1/9: Extracting anomaly cutouts and ROI samples.")
        if save_folder is None:
            save_folder = os.path.join(self._config.study_folder, "anomaly_data")
        if save_folder_roi is None:
            save_folder_roi = os.path.join(self._config.study_folder, "anomaly_roi_data")
        if os.path.exists(save_folder):    
            shutil.rmtree(save_folder)
        if os.path.exists(save_folder_roi):
            shutil.rmtree(save_folder_roi)
        os.makedirs(save_folder_roi, exist_ok=True)
        os.makedirs(save_folder, exist_ok=True)
        for img, seg, basename in sample_dataloader:
            # check if sample has no anomaly
            if not np.any(seg):
                # continue if sample contains no anomaly
                continue

            if img.ndim == 3:
                anomalies, anomalies_roi = crop_and_center_anomaly_2d(
                    img,
                    seg,
                    self._config,
                    self._config.anomaly_size,
                    random_offset=self._config.random_offset,
                    rng=rng,
                    normalization=self._config.normalization,
                    normalization_eps=self._config.normalization_eps,
                )
            elif img.ndim == 4:
                anomalies, anomalies_roi = crop_and_center_anomaly_3d(
                    img,
                    seg,
                    self._config,
                    self._config.anomaly_size,
                    random_offset=self._config.random_offset,
                    rng=rng,
                    normalization=self._config.normalization,
                    normalization_eps=self._config.normalization_eps,
                )
            else:
                raise ValueError(f"Unexpected shape: {img.shape}, Supported: (C, H, W) or (C, D, H, W)")

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
        self._log_step("Step 2/9: Loading anomaly dataset.")
        if anomaly_folder is None:
            anomaly_folder = os.path.join(self._config.study_folder, "anomaly_data")
        self._anomaly_dataset = AnomalyDataset(
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
        self._log_step("Step 3/9: Training generator model (with Optuna optimization).")
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
        if trial_id == -2:
            all_trials = study.get_trials()
            i = 0
            for ts in all_trials:
                if ts.number > i:
                    i=i+1
                    t = ts
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
        self._model = model_loader(model_name, params) 
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
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder, exist_ok=True)


        # use feedback system to generate similar anomalies
        if self._config.use_feedback:
            self._anomaly_dataset.numpy_mode = True
            bad_anomalies = []

            for img, basename in tqdm(self._anomaly_dataset):
                #syn_anomaly_sample = self._model.generate_synth_sample(img, clamp_01=self._config.clamp01_output)

                best = -1
                best_image = None
                syn_anomaly_sample = None
                i = 0
                while best < self._config.feedback_threshold:
                    if self._config.prior_sampling:
                        syn_anomaly_sample = self._model.generate_synth_sample_prior(clamp_01=self._config.clamp01_output, out_hw=self._config.anomaly_size[1:])
                    else:
                        syn_anomaly_sample = self._model.generate_synth_sample(img, clamp_01=self._config.clamp01_output)

                    if best_image is None:
                        best_image = syn_anomaly_sample


                    if syn_anomaly_sample.shape != img.shape:
                        raise ValueError(str(syn_anomaly_sample.shape)+"vs"+str(img.shape))

                    if self._config.random_offset:
                        _background_threshold = self._config.background_threshold
                        if _background_threshold is None:
                            _background_threshold = np.min(syn_anomaly_sample)+0.01
                        syn_anomaly_sample, _, _, _ = center_foreground_com(syn_anomaly_sample, _background_threshold, largest_only=True)
                        img, _, _, _ = center_foreground_com(img, _background_threshold)
                    similarity_score = ssim_01(img, syn_anomaly_sample)

                    if similarity_score > best:
                        best = similarity_score
                        best_image = syn_anomaly_sample
                        print("New best Score: "+str(best))
                    
                    if i % 100 == 0:
                        if self._config.feedback_threshold > 0.15:
                            self._config.feedback_threshold = self._config.feedback_threshold * self._config.threshold_relaxation_factor
                    i = i + 1
                if(best < 0.25):
                    bad_anomalies.append(basename)
                print("Generated "+str(i)+" anomalies and save at threshold: "+str(best))
                save_numpy_as_npy(best_image, str(os.path.join(save_folder, basename)), overwrite=True)
            print("Summary")
            print("No-of bad Anomalies:"+ str(len(bad_anomalies)))
            for name in bad_anomalies:
                print(name)

        # standard generation without feedback
        else:
            for img, basename in tqdm(self._anomaly_dataset):
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

    def create_matching_dict(self, control_samples_dataloader:Iterator[Tuple[np.ndarray, np.ndarray, str]], roi_folder=None, csv_file_path=None):
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
        if img.ndim in [3, 4]:
            _data = create_matching_dictionary(control_samples_dataloader, _roi_dataset, self._config, matching_routine=self._config.matching_routine, anomaly_duplicates=self._config.anomaly_duplicates)
        else:
            raise ValueError(f"Unexpected shape: {img.shape}, Supported: (C, H, W) or (C, D, H, W)")

        df_detection = pd.DataFrame(_data, columns=["control", "anomaly_list"])
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


    def fusion_synth_anomalies(self, control_samples_array, basename_of_control_sample, base_mask=None, save_npy=True, save_path=None) -> Tuple[np.ndarray, np.ndarray]:
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

        if base_mask is not None:
            if base_mask.shape != control_samples_array.shape:
                raise ValueError(f"base_mask shape {base_mask.shape} does not match control_samples_array shape {control_samples_array.shape}")
            else:
                seg_final = base_mask.copy()
        else:
            seg_final = np.zeros_like(control_samples_array)

        anomalies = self._config.matching_dict[basename_of_control_sample]
        img = control_samples_array.copy()

        if anomalies is None or len(anomalies) == 0:
            print(f"No matched anomaly found for control sample {basename_of_control_sample} in matching dict.")

        for anomaly_basename, fusion_position in anomalies:

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
                    img,
                    synth_anomaly_image,
                    anomaly_meta,
                    fusion_position,
                    self._config,
                )
            elif control_samples_array.ndim == 4:
                img, seg = fusion3d(
                    img,
                    synth_anomaly_image,
                    anomaly_meta,
                    fusion_position,
                    self._config,
                )
            else:
                raise ValueError(f"Unexpected shape: {img.shape}, Supported: (C, H, W) or (C, D, H, W)")
            
            if seg_final is None:
                seg_final = seg
            else:
                seg_final = combine_binary_masks(seg_final, seg, mode = "or")

        if save_npy:
            if save_path is None:
                save_path = os.path.join(self._config.study_folder, "generated_hybrid_samples")
            img_folder = os.path.join(save_path,"images_npy")
            seg_folder = os.path.join(save_path,"segmentations_npy")
            os.makedirs(img_folder, exist_ok=True)
            os.makedirs(seg_folder, exist_ok=True)

            img_path = os.path.join(img_folder, basename_of_control_sample)
            seg_path = os.path.join(seg_folder, basename_of_control_sample)
            save_numpy_as_npy(img, img_path, overwrite=True)
            save_numpy_as_npy(seg_final, seg_path, overwrite=True)

        return img, seg_final
