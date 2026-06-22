from __future__ import annotations
import os.path
from typing import Iterator
import numpy as np
from typing import Tuple
import optuna
import pandas as pd
import torch
from tqdm import tqdm

from data_handler.AnomalyDataset import AnomalyDataset, save_numpy_as_npy

from generation_models.model_registry import get_model_spec
from fusion_backend.fusion_registry import get_fusion_backend_spec
from synthesizer.mask_manipulation import TransformGenerator
from synthesizer.functions_2D.Anomaly_Extraction2D import crop_and_center_anomaly_2d
from synthesizer.functions_3D.Anomaly_Extraction3D import crop_and_center_anomaly_3d
from synthesizer.Configuration import Configuration
from synthesizer.Matching import center_foreground_com, combine_label_masks, create_matching_dictionary, crop_border, ssim_01, template_matching
from synthesizer.Trainer import optimize
from synthesizer.Evaluation import evaluation_pipeline
from data_handler.Visualizer import run_hybrid_visualizer


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
        self._fusion_backend = None

    def _log_step(self, message: str) -> None:
        print(f"[HybridDataGenerator] {message}")

    def extract_anomalies(
        self,
        sample_dataloader: Iterator[Tuple[np.ndarray, np.ndarray, str]],
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
        Notes
        -----
            Saved anomaly cutouts are always centered. Dynamic random offset
            augmentation is applied later in the training dataloader.
            Normalization is controlled via config.normalization and config.normalization_eps.

        Outputs
        -------
        None
            Side effects: writes .npy files, updates config transformation file, sets self._anomaly_dataset.
        """
        self._log_step("Step 1/9: Extracting anomaly cutouts and ROI samples.")
        paths = self._config.get_paths()
        anomaly_folder = paths.anomaly_data
        anomaly_roi_folder = paths.anomaly_roi_data
        anomaly_mask_folder = paths.anomaly_mask_data
        anomaly_tgt_mask_folder = paths.anomaly_tgt_mask_data
        anomaly_mask_roi_folder = paths.anomaly_mask_roi_data

        paths.confirm_and_clear_artifact_dirs(
            anomaly_folder,
            anomaly_roi_folder,
            anomaly_mask_folder,
            anomaly_tgt_mask_folder,
            anomaly_mask_roi_folder
        )
        self._config.syn_anomaly_transformations = {}

        for img, seg, basename in sample_dataloader:
            # check if sample has no anomaly
            if not np.any(seg):
                # continue if sample contains no anomaly
                continue

            if img.ndim == 3:
                anomalies, anomalies_roi, org_masks, roi_masks = crop_and_center_anomaly_2d(
                    img,
                    seg,
                    self._config,
                    self._config.anomaly_size,
                    normalization=self._config.normalization,
                    normalization_eps=self._config.normalization_eps,
                )
            elif img.ndim == 4:
                anomalies, anomalies_roi, org_masks, roi_masks = crop_and_center_anomaly_3d(
                    img,
                    seg,
                    self._config,
                    self._config.anomaly_size,
                    normalization=self._config.normalization,
                    normalization_eps=self._config.normalization_eps,
                )
            else:
                raise ValueError(f"Unexpected shape: {img.shape}, Supported: (C, H, W) or (C, D, H, W)")

            # save anomaly cutout for training
            for i, (anomaly_sample, meta_data) in enumerate(anomalies):
                artifact_name = basename + "_" + str(i) + ".npy"
                save_numpy_as_npy(
                    anomaly_sample,
                    os.path.join(anomaly_folder, artifact_name),
                    overwrite=True,
                )
                self._config.add_anomaly_transformation(artifact_name, meta_data)

            # save roi of anomaly for matching
            for i, roi_sample in enumerate(anomalies_roi):
                artifact_name = basename + "_" + str(i) + ".npy"
                save_numpy_as_npy(
                    roi_sample,
                    os.path.join(anomaly_roi_folder, artifact_name),
                    overwrite=True,
                )

            # save mask of anomaly
            for i, mask_sample in enumerate(org_masks):
                artifact_name = basename + "_" + str(i) + ".npy"
                save_numpy_as_npy(
                    mask_sample,
                    os.path.join(anomaly_mask_folder, artifact_name),
                    overwrite=True,
                )

            # save mask of roi
            for i, mask_sample in enumerate(roi_masks):
                artifact_name = basename + "_" + str(i) + ".npy"
                save_numpy_as_npy(
                    mask_sample,
                    os.path.join(anomaly_mask_roi_folder, artifact_name),
                    overwrite=True,
                )

        self._config.save_anomaly_transformations()
        self.load_anomalies()

    def load_anomalies(self):
        """
        Load previously extracted anomaly cutouts into an AnomalyDataset and
        load their transformation meta-data from config.
        If conditional model: calculate num_anomaly_classes if necessary and sets model param.

        Outputs
        -------
        None
            Side effects: sets self._anomaly_dataset and loads config transformations into memory.
        """
        self._log_step("Step 2/9: Loading anomaly dataset.")
        paths = self._config.get_paths()

        self._anomaly_dataset = AnomalyDataset(
            paths,
            return_artifacts=self._config.model_params.input_artefacts,
            load_to_ram=True,
            dtype=torch.float32,
        )

        if self._config.conditional:
            if self._config.num_anomaly_classes is None:
                max_class_val = 0
                for sample in self._anomaly_dataset:
                    mask = sample["ori_mask"]
                    max_class_val = max(max_class_val, int(mask.max().item()))
                self._config.num_anomaly_classes = max_class_val

            self._config.model_params.set_model_param(
                "num_anomaly_classes",
                self._config.num_anomaly_classes,
            )

        self._config.load_anomaly_transformations()

    def train_generator(self, no_of_trials):
        """
        Train/optimize the generator model using Optuna via `optimize(...)`,
        then load the resulting model via `load_generator()`.

        Inputs
        ------
        no_of_trials:
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
            optimize(no_of_trials, self._config, self._anomaly_dataset)

        if no_of_trials > 1:
            self.load_generator(trial_id=-1)  # load best trial
        else:
            self.load_generator(trial_id=-2)  # load last trial

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
              - -2: load last trial
              - otherwise: load trial with matching ts.number

        Outputs
        -------
        None
            Side effects: sets self._model and loads its weights.
        """
        self._log_step("Step 4/9: Loading trained generator model.")
        if path_to_db_file is None:
            path_to_db_model = self._config.get_paths().optuna_storage_url
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
            i = -1
            for ts in all_trials:
                if ts.number > i:
                    i=i+1
                    t = ts
        if trial_id >= 0:
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
        self._model = get_model_spec(model_name).build(params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.warmup(
            self._config.anomaly_size,
            device=device,
            dtype=self._config.training_dtype,
            config=self._config,
        )
        model_path = t.user_attrs['model_path']
        self._model.load_checkpoint(model_path)

    def load_fusion_backend(self, fusion_backend_checkpoint=None):
        """
        Initialize the configured fusion backend from the current configuration.

        Outputs
        -------
        None
            Side effect: sets self._fusion_backend and optionally loads its checkpoint.
        """
        self._log_step("Step 5/9: Loading fusion backend.")
        backend_params = {"fusion_params": self._config.fusion_params}
        self._fusion_backend = get_fusion_backend_spec(self._config.fusion_backend).build(backend_params)

        if fusion_backend_checkpoint:
            self._fusion_backend.load_checkpoint(fusion_backend_checkpoint)
        elif self._config.fusion_backend_checkpoint:
            self._fusion_backend.load_checkpoint(self._config.fusion_backend_checkpoint)

    def generate_synth_anomalies(self):
        """
        Generate synthetic anomalies for each anomaly in the loaded anomaly dataset.

        For each anomaly sample:
          - run model.generate(sample, mode="prior"|"posterior")
          - save output as .npy under sample["fname"]

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
        paths = self._config.get_paths()
        synth_anomaly_folder = paths.synth_anomaly_data
        tgt_mask_folder = paths.anomaly_tgt_mask_data
        paths.confirm_and_clear_artifact_dirs(synth_anomaly_folder)
        os.makedirs(tgt_mask_folder, exist_ok=True)
        self._anomaly_dataset.numpy_mode = True
        target_mask_generator = TransformGenerator.from_config(self._config)
        generation_mode = "prior" if self._config.prior_sampling else "posterior"


        # use feedback system to generate similar anomalies
        if self._config.use_feedback:
            bad_anomalies = []

            for sample in tqdm(self._anomaly_dataset):
                img = sample["img"]
                basename = sample["fname"]

                best = -1
                best_image = None
                best_mask = None
                syn_anomaly_sample = None
                syn_anomaly_mask = None
                i = 0
                while best < self._config.feedback_threshold:
                    syn_anomaly_sample, syn_anomaly_mask = self._model.generate(
                        sample,
                        mode=generation_mode,
                        variation_strength=self._config.variation_strength,
                        clamp_01=self._config.clamp01_output,
                        target_mask_generator=target_mask_generator,
                    )

                    if best_image is None:
                        best_image = syn_anomaly_sample
                        best_mask = syn_anomaly_mask

                    if syn_anomaly_sample.shape != img.shape:
                        raise ValueError(str(syn_anomaly_sample.shape)+"vs"+str(img.shape))

                    """               
                    if self._config.random_offset:
                        _background_threshold = self._config.background_threshold
                        if _background_threshold is None:
                            _background_threshold = np.min(syn_anomaly_sample)+0.01
                        syn_anomaly_sample, _, _, _ = center_foreground_com(syn_anomaly_sample, _background_threshold, largest_only=True)
                        img, _, _, _ = center_foreground_com(img, _background_threshold) """
                    similarity_score = ssim_01(img, syn_anomaly_sample)

                    if similarity_score > best:
                        best = similarity_score
                        best_image = syn_anomaly_sample
                        best_mask = syn_anomaly_mask
                        print("New best Score: "+str(best))

                    if i % 100 == 0:
                        if self._config.feedback_threshold > 0.15:
                            self._config.feedback_threshold = self._config.feedback_threshold * self._config.threshold_relaxation_factor
                    i = i + 1
                if(best < 0.25):
                    bad_anomalies.append(basename)
                print("Generated "+str(i)+" anomalies and save at threshold: "+str(best))
                save_numpy_as_npy(best_image, str(os.path.join(synth_anomaly_folder, basename)), overwrite=True)
                save_numpy_as_npy(best_mask, str(os.path.join(tgt_mask_folder, basename)), overwrite=True)
            print("Summary")
            print("No-of bad Anomalies:"+ str(len(bad_anomalies)))
            for name in bad_anomalies:
                print(name)

        # standard generation without feedback
        else:
            for sample in tqdm(self._anomaly_dataset):
                basename = sample["fname"]
                syn_anomaly_sample, syn_anomaly_mask = self._model.generate(
                    sample,
                    mode=generation_mode,
                    variation_strength=self._config.variation_strength,
                    clamp_01=self._config.clamp01_output,
                    target_mask_generator=target_mask_generator,
                )
                save_numpy_as_npy(syn_anomaly_sample, str(os.path.join(synth_anomaly_folder, basename)), overwrite=True)
                save_numpy_as_npy(syn_anomaly_mask, str(os.path.join(tgt_mask_folder, basename)), overwrite=True)

        self.load_synth_anomalies()



    def load_synth_anomalies(self, transformation_file=None):
        """
        Load synthetic anomalies from disk and load the anomaly transformations file.

        Inputs
        ------
        transformation_file:
            Path to anomaly transformations file.
            If None: "<study_folder>/anomaly_transformations.json"

        Outputs
        -------
        None
            Side effects: sets self._synth_anomaly_dataset and loads transformation meta-data into config.
        """
        self._log_step("Step 6/9: Loading synthetic anomalies and transformation metadata.")
        paths = self._config.get_paths()
        if transformation_file is None:
            transformation_file = paths.anomaly_transformations_file

        self._config.load_anomaly_transformations(transformation_file)
        self._synth_anomaly_dataset = AnomalyDataset(
            paths,
            return_artifacts=("synth_anomaly", "fname"),
            index_artifact="synth_anomaly",
            load_to_ram=True,
            dtype=torch.float32,
        )

    def create_matching_dict(self, control_samples_dataloader:Iterator[Tuple[np.ndarray, np.ndarray, str]]):
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
        Outputs
        -------
        None
            Side effects: writes CSV and populates self._config.matching_dict.
        """
        self._log_step("Step 7/9: Creating matching dictionary between controls and anomalies.")
        if self._synth_anomaly_dataset is None:
            raise ValueError(f"No Synth Anomalies Loaded: Run generate_synth_anomalies or load_synth_anomalies first")
        paths = self._config.get_paths()
        csv_file_path = paths.matching_dict_file
        _roi_dataset = AnomalyDataset(
            paths,
            return_artifacts=("anomaly_roi", "fname"),
            index_artifact="anomaly_roi",
            load_to_ram=True,
            dtype=torch.float32,
            numpy_mode=True
        )
        img = _roi_dataset[0]["anomaly_roi"]
        if img.ndim in [3, 4]:
            _data = create_matching_dictionary(control_samples_dataloader, _roi_dataset, self._config, matching_routine=self._config.matching_routine, anomaly_duplicates=self._config.anomaly_duplicates)
        else:
            raise ValueError(f"Unexpected shape: {img.shape}, Supported: (C, H, W) or (C, D, H, W)")

        df_detection = pd.DataFrame(_data, columns=["control", "anomaly_list"])
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
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
            csv_file_path = self._config.get_paths().matching_dict_file
        self._config.load_matching_csv(csv_file_path)


    def fusion_synth_anomalies(self, control_samples_array, basename_of_control_sample, base_mask=None, save_npy=True) -> Tuple[np.ndarray, np.ndarray]:
        self._log_step("Step 9/9: Fusing synthetic anomalies.")
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
            seg_final = np.zeros_like(control_samples_array, dtype=np.uint8)

        anomalies = self._config.matching_dict[basename_of_control_sample]
        img = control_samples_array.copy()
        if self._fusion_backend is None:
            self.load_fusion_backend()
        self._fusion_backend.warmup(img.shape, config=self._config)

        if anomalies is None or len(anomalies) == 0:
            print(f"No matched anomaly found for control sample {basename_of_control_sample} in matching dict.")

        for anomaly_basename, fusion_position in anomalies:

            synth_anomaly_image = self._synth_anomaly_dataset.load_numpy_by_basename(
                anomaly_basename,
                artifact="synth_anomaly",
            )

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

            target_mask = self._synth_anomaly_dataset.load_numpy_by_basename(
                anomaly_basename,
                artifact="tgt_mask",
            )

            fusion_output = self._fusion_backend.fuse(
                img,
                synth_anomaly_image,
                anomaly_meta,
                fusion_position,
                target_mask=target_mask,
                config=self._config,
            )
            img = fusion_output.image
            seg = fusion_output.segmentation
            roi = fusion_output.roi
            roi_mask = fusion_output.roi_mask
            
            if seg_final is None:
                seg_final = seg
            else:
                seg_final = combine_label_masks(seg_final, seg, overwrite=True)

            if roi is None:
                print(f"Warning: roi is None for {anomaly_basename} in {basename_of_control_sample}. (Empty synthetic segmentation) => skip fusion")
            else:
                roi = np.array(roi, dtype=np.float32)
                roi_path = os.path.join(
                    self._config.get_paths().synth_roi_data,
                    basename_of_control_sample,
                    anomaly_basename,
                )
                os.makedirs(os.path.dirname(roi_path), exist_ok=True)
                save_numpy_as_npy(roi, roi_path, overwrite=True)

                if roi_mask is not None:
                    roi_mask_path = os.path.join(
                        self._config.get_paths().synth_roi_mask_data,
                        basename_of_control_sample,
                        anomaly_basename,
                    )
                    os.makedirs(os.path.dirname(roi_mask_path), exist_ok=True)
                    save_numpy_as_npy(roi_mask, roi_mask_path, overwrite=True)

        if save_npy:
            paths = self._config.get_paths()
            img_folder = paths.generated_images_npy
            seg_folder = paths.generated_segmentations_npy
            os.makedirs(img_folder, exist_ok=True)
            os.makedirs(seg_folder, exist_ok=True)

            img_path = os.path.join(img_folder, basename_of_control_sample)
            seg_path = os.path.join(seg_folder, basename_of_control_sample)
            save_numpy_as_npy(img, img_path, overwrite=True)
            save_numpy_as_npy(seg_final, seg_path, overwrite=True)

        return img, seg_final
    
    def run_evaluation_pipeline(self, sample_dataloader: Iterator[Tuple[np.ndarray, np.ndarray, str]]):
        self._log_step("Evaluation 1/2: Starting evaluation pipeline.")
        evaluation_pipeline(sample_dataloader, self._config)

    def visualize_evaluation_results(self):
        self._log_step("Evaluation 2/2: Starting visualization of evaluation results.")
        run_hybrid_visualizer(self._config)
