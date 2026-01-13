import json
from dataclasses import asdict

import pandas as pd

from models import VAE_ResNet_3D, VAE_ResNet_2D
import os
import jsonpickle

# Allowed model choices (fixed set)
ALLOWED_MODELS = ["VAE_ResNet_3D", "VAE_ResNet_2D"]

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

        # project parameter
        self.study_name = study_name
        if model_name not in ALLOWED_MODELS:
            raise ValueError(f"Model {model_name} is not supported. \nCurrently Supported models: {ALLOWED_MODELS}")
        self.model_name = model_name
        self.config_name = None
        if save_path is None:
            self.study_folder = os.path.join(os.path.join(os.getcwd(),"results"), study_name)
        else:
            self.study_folder = os.path.join(os.path.join(save_path,"results"), study_name)


        # synthesizer parameter
        self.anomaly_size = anomaly_size
        self.min_anomaly_percentage = 0.05
        self.clamp01_output = False
        self.matching_dict= {}
        self.syn_anomaly_transformations = {}
        self.background_threshold = None
        self.fusion_mask_params = {
            "max_alpha": 0.8,
            "sq": 2,
            "steepness_factor": 3,
            "upsampling_factor": 2,
            "sobel_threshold": 0.05,
            "dilation_size": 2,
            "shave_pixels": 1
        }

        # global training parameter, fixed during training
        self.val_ratio = 0.2
        self.batch_size = 32
        self.epochs = 40
        self.lr = 1e-4
        self.log_every = 50
        self.early_stopping = True
        self.early_stopping_params = {
            "patience": 20,
            "delta": 0.0001
        }
        self.lr_scheduler = True
        self.lr_scheduler_params = {
            "patience": 10,
            "factor": 0.1,
            "threshold": 1e-5,
        }

        # model specific hyperparameter for dynamic tuning via optuna
        self.model_params = None
        # VAE3D parameter
        if model_name == "VAE_ResNet_3D":
            _VAE3D_min_params = asdict(VAE_ResNet_3D.Config(
                n_res_blocks=4,
                n_levels=4,
                z_channels=64,
                bottleneck_dim=64,
                use_multires_skips = False,
                recon_weight = 1.0,
                beta_kl = 0.1,
                use_transpose_conv = False))
            _VAE3D_max_params = asdict(VAE_ResNet_3D.Config(
                n_res_blocks=5,
                n_levels=5,
                z_channels=128,
                bottleneck_dim=128,
                use_multires_skips = False,
                recon_weight = 1,
                beta_kl = 0.5,
                use_transpose_conv=False))
            self.model_params = {"min": _VAE3D_min_params, "max": _VAE3D_max_params}

        # VAE2D parameter
        if model_name == "VAE_ResNet_2D":
            _VAE2D_min_params = asdict(VAE_ResNet_2D.Config(
                n_res_blocks=4,
                n_levels=4,
                z_channels=32,
                bottleneck_dim=32,
                use_multires_skips = False,
                recon_weight = 1.0,
                beta_kl = 1.0,
                use_transpose_conv=False))
            _VAE2D_max_params = asdict(VAE_ResNet_2D.Config(
                n_res_blocks=5,
                n_levels=5,
                z_channels=32,
                bottleneck_dim=32,
                use_multires_skips = False,
                recon_weight = 5.0,
                beta_kl = 5.0,
                use_transpose_conv=False))
            self.model_params = {"min": _VAE2D_min_params, "max": _VAE2D_max_params}

    # set hyperparameter space. need min and max config of model.py
    def set_hyperparameter_space(self, min_config, max_config):
        """
        Override the hyperparameter search space used by Optuna.

        Inputs
        ------
        min_config:
            A model Config dataclass instance representing the minimum/baseline parameters.
        max_config:
            A model Config dataclass instance representing the maximum parameters.

        Outputs
        -------
        None
            Side effect: updates self.model_params to {"min": ..., "max": ...}.
        """
        self.model_params = {"min": asdict(min_config), "max": asdict(max_config)}

    # save config as JSON
    def save_config_file(self, json_path=None, overwrite=False):
        """
        Serialize and save the entire Configuration object to a JSON file using jsonpickle.

        Inputs
        ------
        json_path:
            Target path for the JSON file.
            If None: "<study_folder>/configuration.json"
        overwrite:
            If True and file exists, remove it first.

        Outputs
        -------
        None
            Side effect: writes JSON to disk.

        Notes
        -----
        The implementation has a small logic quirk:
        """
        json_string = jsonpickle.encode(self, indent=0)
        if json_path is None:
            json_path = os.path.join(self.study_folder, "configuration.json")
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
            json_path = os.path.join(self.study_folder, "anomaly_transformations.json")
        with open(json_path, "r", encoding="utf-8") as f:
            self.syn_anomaly_transformations = json.load(f)

    def save_anomaly_transformations(self, json_path=None):
        """
        Save `self.syn_anomaly_transformations` to JSON.

        Inputs
        ------
        json_path:
            Target JSON file.
            If None: "<study_folder>/anomaly_transformations.json"

        Outputs
        -------
        None
            Side effect: writes JSON to disk.
        """
        if json_path is None:
            json_path = os.path.join(self.study_folder, "anomaly_transformations.json")
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
            csv_path = os.path.join(self.study_folder, "matching_dict.csv")
        df = pd.read_csv(csv_path)
        if df["position_factor"] is not None:
            df["position_factor"] = (
                df["position_factor"]
                .str.strip("[]")
                .str.split(",")
                .apply(lambda xs: [float(x) for x in xs])
            )
        self.matching_dict = df.set_index("control").to_dict(orient="index")

    def update_fusion_params(self, max_alpha=0.8, sq=2, steepness_factor=3, upsampling_factor=2,
                             sobel_threshold=0.05, dilation_size=2, shave_pixels=1):
        """
        Update parameters controlling the fusion mask creation.

        Inputs
        ------
        max_alpha:
            Maximum blending alpha.
        sq:
            Exponent/shape parameter (implementation dependent in Fusion3D).
        steepness_factor:
            Controls transition steepness for blending mask.
        upsampling_factor:
            Upsampling factor used in mask computation.
        sobel_threshold:
            Threshold for Sobel edge detection.
        dilation_size:
            Dilation size used for mask refinement.
        shave_pixels:
            Number of pixels to shave from boundaries (artifact reduction).

        Outputs
        -------
        None
            Side effect: overwrites self.fusion_mask_params.
        """
        self.fusion_mask_params = {
            "max_alpha": max_alpha,
            "sq": sq,
            "steepness_factor": steepness_factor,
            "upsampling_factor": upsampling_factor,
            "sobel_threshold": sobel_threshold,
            "dilation_size": dilation_size,
            "shave_pixels": shave_pixels
        }

    # print all values of config instance
    def print_config(self):
        """
        Print all configuration fields and values to stdout.

        Inputs
        ------
        None

        Outputs
        -------
        None
            Side effect: prints to stdout.
        """
        print(', \n'.join("%s: %s" % item for item in vars(self).items()))

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
        return jsonpickle.decode(fi.read())