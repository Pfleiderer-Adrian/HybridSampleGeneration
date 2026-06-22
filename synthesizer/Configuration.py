import ast
import csv
import json

import numpy as np
import os
import jsonpickle

from generation_models.model_configuration import ModelConfiguration
from generation_models.model_registry import get_model_spec, registered_model_names
from fusion_backend.fusion_configuration import FusionConfiguration
from fusion_backend.fusion_registry import get_fusion_backend_spec, registered_fusion_backend_names
from synthesizer.StudyPaths import StudyPaths

ALLOWED_MODELS = registered_model_names()
ALLOWED_FUSION_BACKENDS = registered_fusion_backend_names()


# creates a new interactive config object/file for the data generator
class Configuration:
    """
    Central configuration object for the hybrid data generation pipeline.

    Stores:
      - project/study parameters (study name, paths)
      - synthesizer parameters (anomaly size, fusion parameters, thresholds)
      - training parameters (epochs, lr, early stopping, etc.)
      - Optuna hyperparameter search space (min/max model configs)
      - metadata artifacts (matching dictionary, anomaly transformation metadata)

    This object can be serialized/deserialized via jsonpickle.
    """
    _IMMUTABLE_FIELDS = {"model_name", "anomaly_size", "study_name", "save_path"}

    def __setattr__(self, name, value):
        if (
            name in self._IMMUTABLE_FIELDS
            and self.__dict__.get("_immutable_fields_locked", False)
            and name in self.__dict__
        ):
            raise AttributeError(
                f"{name} is fixed at Configuration initialization and cannot be changed afterwards."
            )
        if name == "model_params":
            value = ModelConfiguration.from_value(value)
        if name == "fusion_params":
            value = FusionConfiguration.from_value(value)
        super().__setattr__(name, value)

    def __init__(self, study_name, model_name, anomaly_size, save_path=None) -> None:
        """
        Create a new configuration instance.

        Inputs
        ------
        study_name:
            Name of the experiment/study. Used to create results directory and Optuna DB name.
        model_name:
            Model identifier string. Must be in ALLOWED_MODELS.
        anomaly_size:
            Target anomaly cutout size (used in extraction and model warmup).
            Typically a tuple/list like (C, D, H, W) / (C, H, W) or similar depending on your pipeline.

        Outputs
        -------
        None
            Initializes fields with defaults and sets model hyperparameter search space.
        """

        self._immutable_fields_locked = False

        # project parameter
        self.study_name = study_name
        if model_name not in ALLOWED_MODELS:
            raise ValueError(f"Model {model_name} is not supported. \nCurrently Supported models: {ALLOWED_MODELS}")
        model_spec = get_model_spec(model_name)
        self.model_name = model_name
        self.config_name = None
        if save_path is None:
            study_folder = os.path.join(os.path.join(os.getcwd(),"results"), study_name)
        else:
            study_folder = os.path.join(os.path.join(save_path,"results"), study_name)
        self.study_folder = os.path.normpath(study_folder)
        self.paths = StudyPaths(self.study_folder, self.study_name)
        # rng for persitence
        self.rng = np.random.default_rng(42)


        # anomaly extraction parameter
        self.anomaly_size = anomaly_size
        self.separated_anomaly = True

        # mask augmentation parameter
        self.use_default_mask_transforms = False
        # override dicts for changing transform probabilities and parameters manually
        self.mask_transform_probs = {}
        self.mask_transform_params = {}

        self.mask_transform_priorities = None   # dominant class first; default: present classes sorted by id

        # Random offsets are applied dynamically during training augmentation.
        # Persisted anomaly cutouts stay centered, which keeps later fusion stable.
        self.random_offset = True
        self.random_offset_max_fraction = 1.0
        self.random_offset_foreground_threshold_rel = 0.001
        self.add_bg_noise = True


        self.conditional = model_spec.uses_masks
        self.num_anomaly_classes = None # will be set in HDG (load_anomalies)

        # synthesizer parameter
        self.prior_sampling = False
        # Controls how strongly the latent sample is perturbed during posterior or prior generation.
        self.variation_strength = 1.0
        self.min_anomaly_percentage = 0.05
        self.min_pad = (20, 20, 20)    # just use first two values for 2d
        self.pad_ratio = (0.5, 0.5, 0.5)
        self.clamp01_output = False
        self.normalization = "z-score"
        self.normalization_eps = 1e-6
        self.matching_dict= {}
        self.syn_anomaly_transformations = {}
        # relative threshold for creating target mask (try 0.01 or 0.1)
        self.background_threshold = 0.01

        # feedback system
        self.use_feedback = False
        self.feedback_threshold = 0.8
        self.threshold_relaxation_factor = 0.9

        # matching parameter
        self.matching_routine = "fixed_from_extraction_anomaly_fusion"
        self.anomaly_duplicates = False

        self.fusions_per_control = 1  # for local and global matching
        self.max_fusions_per_control_deviation = 0
        self.matching_intensity_weight = 0.5
        self.matching_gradient_weight = 0.5

        self.fusion_backend = "classical"
        self.fusion_backend_checkpoint = None
        self.fusion_params = get_fusion_backend_spec(self.fusion_backend).build_configuration()

        # global training parameter, fixed during training
        self.val_ratio = 0.2
        self.batch_size = 64
        self.epochs = 3000
        self.lr = 1e-3
        self.log_every = None
        self.training_dtype = None
        self.grad_clip_norm = None
        self.monitor_metric = None
        self.early_stopping = True
        self.early_stopping_params = {
            "patience": 2000,
            "delta": 0.0001
        }
        self.lr_scheduler = True
        self.lr_scheduler_params = {
            "patience": 1000,
            "factor": 0.1,
            "threshold": 1e-5,
        }

        self.model_params = model_spec.build_configuration(anomaly_size[0])

        # set to None for variable roi size
        self.fixed_roi_size = None
        # define min size for every dimension to improve template matching for small anomalies
        self.min_roi_size = 0

        # evaluation parameter
        # (optional) absolute thresholds
        self.custom_outlier_thresholds = {
            "Contrast": {"min": None, "max": None},
            "Homogeneity": {"min": None, "max": None},
            "Energy": {"min": None, "max": None},
            "Correlation": {"min": None, "max": None},

            "roi_Contrast": {"min": None, "max": None},
            "roi_Homogeneity": {"min": None, "max": None},
            "roi_Energy": {"min": None, "max": None},
            "roi_Correlation": {"min": None, "max": None},

            "Volume": {"min": None, "max": None},
            "D-center": {"min": None, "max": None},
            "H-center": {"min": None, "max": None},
            "W-center": {"min": None, "max": None},
        }

        self._immutable_fields_locked = True

    def get_paths(self) -> StudyPaths:
        """
        Return the managed study path layout.

        This method keeps the StudyPaths object aligned with the current
        study folder and name.
        """
        study_folder = os.path.normpath(os.fspath(self.study_folder))
        if (
            not isinstance(self.paths, StudyPaths)
            or self.paths.study_folder != study_folder
            or self.paths.study_name != self.study_name
        ):
            self.paths = StudyPaths(study_folder, self.study_name)
        return self.paths

    # save config as JSON
    def save_config_file(self, overwrite=False):
        """
        Serialize and save the entire Configuration object to a JSON file using jsonpickle.

        Inputs
        ------
        overwrite:
            If True and file exists, remove it first.

        Outputs
        -------
        None
            Side effect: writes JSON to disk.
        """
        json_string = jsonpickle.encode(self, indent=0)
        json_path = self.get_paths().configuration_file
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        if os.path.exists(json_path):
            with open(json_path, 'w', encoding='utf-8') as fi:
                fi.write(json_string)
        else:
            #os.chmod(json_path, 0o644)
            with open(json_path, 'w', encoding='utf-8') as fi:
                fi.write(json_string)

    # add anomaly transformation entry to config
    def add_anomaly_transformation(self, name, params):
        """
        Register anomaly transformation metadata in memory.

        Typically called during anomaly extraction to store scale/position information.

        Inputs
        ------
        name:
            Filename/basename of the anomaly cutout (key).
        params:
            Arbitrary transformation metadata (value), e.g. {"scale_factor": ..., ...}.

        Outputs
        -------
        None
            Side effect: updates self.syn_anomaly_transformations[name] = params.
        """
        self.syn_anomaly_transformations[name] = params

    def load_anomaly_transformations(self, json_path=None):
        """
        Load anomaly transformation metadata from JSON into `self.syn_anomaly_transformations`.

        Inputs
        ------
        json_path:
            Path to transformation JSON file.
            If None: "<study_folder>/anomaly_transformations.json"

        Outputs
        -------
        None
            Side effect: overwrites self.syn_anomaly_transformations with loaded content.
        """
        if json_path is None:
            json_path = self.get_paths().anomaly_transformations_file
        with open(json_path, "r", encoding="utf-8") as f:
            self.syn_anomaly_transformations = json.load(f)

    def save_anomaly_transformations(self):
        """
        Save `self.syn_anomaly_transformations` to JSON.

        Outputs
        -------
        None
            Side effect: writes JSON to disk.
        """
        json_path = self.get_paths().anomaly_transformations_file
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.syn_anomaly_transformations, f, ensure_ascii=False, indent=2)

    def load_matching_csv(self, csv_path=None):
        """
        Load the matching CSV file produced by the matching step into `self.matching_dict`.

        The expected CSV columns are:
          - control            (string key)
          - anomaly            (string filename/basename)
          - position_factor    (string representation of list, e.g. "[0.1, 0.2, 0.3]")

        This method parses position_factor into a list[float] and then creates:
          self.matching_dict = {
              control_basename: {"anomaly": <str>, "position_factor": <list[float]>},
              ...
          }

        Inputs
        ------
        csv_path:
            Path to the CSV file.
            If None: "<study_folder>/matching_dict.csv"

        Outputs
        -------
        None
            Side effect: updates self.matching_dict.
        """
        if csv_path is None:
            csv_path = self.get_paths().matching_dict_file

        result = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                control = row["control"].strip()
                raw = (row.get("anomaly_list") or "").strip()

                if not raw:
                    result[control] = []
                    continue

                parsed = ast.literal_eval(raw)  # -> list of tuples
                anomalies = []

                for item in parsed:
                    if not (isinstance(item, (tuple, list)) and len(item) == 2):
                        raise ValueError(f"Unexpected element in anomaly_list for {control}: {item!r}")

                    name, coords = item
                    if not (isinstance(name, str) and isinstance(coords, (list, tuple)) and len(coords) in [2, 3]):
                        raise ValueError(f"Unexpected tuple format for {control}: {item!r}")
                    float_coords = [float(c) for c in coords]
                    anomalies.append((name, float_coords))

                result[control] = anomalies
        self.matching_dict = result

# load config file from JSON file
def load_config_file(json_path):
    """
    Load a Configuration object from a jsonpickle JSON file.

    Inputs
    ------
    json_path:
        Path to the config file previously written by save_config_file().

    Outputs
    -------
    Configuration
        The decoded Configuration instance.
    """
    with open(json_path, 'r', encoding='utf-8') as fi:
        config = jsonpickle.decode(fi.read())
    config.model_params = ModelConfiguration.from_value(config.model_params)
    config.fusion_params = FusionConfiguration.from_value(config.fusion_params)
    return config
