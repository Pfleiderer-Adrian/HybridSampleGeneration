"""Shared configuration values; category scripts contain the actual workflow."""
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.settings import TEXTURE_ROOT, study_folder
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.generation.model_settings import Choice, FloatRange, IntRange


def apply_generator_defaults(config: Configuration) -> None:
    """
    Shared MVTec AD 2 defaults for all categories.
    """
    # extraction settings
    config.extraction.add_background_noise = False
    config.extraction.min_coverage_ratio = 0.01
    config.extraction.roi.fixed_size = None
    config.extraction.roi.min_padding = (20, 20, 20)
    config.extraction.roi.padding_ratio = (0.5, 0.5, 0.5)

    # generation settings
    config.augmentation.random_offset_enabled = True
    config.augmentation.random_offset_max_fraction = 0.8
    config.augmentation.random_offset_foreground_threshold = 0.01
    config.generation.clamp_output = False
    config.extraction.normalization = "z-score"
    config.extraction.normalization_eps = 1e-6
    config.generation.background_threshold = 0.18
    config.evaluation.foreground_threshold = 0.18
    config.generation.sampling_mode = "posterior"
    config.generation.feedback.enabled = False
    config.generation.feedback.similarity_threshold = 0.01
    config.generation.feedback.threshold_relaxation_factor = 0.9
    config.generation.variation_strength = 1.25
    config.generation.variants_per_real_anomaly = 3

    # matching settings
    config.matching.routine = "global"
    config.matching.hybrids_per_original = 3
    config.matching.reuse_synthetic_across_hybrids = True
    config.matching.allow_sibling_variants_in_same_hybrid = False
    config.matching.anomalies_per_hybrid = 2
    config.matching.max_anomalies_per_hybrid_deviation = 1

    # Fusion settings
    config.fusion.set_backend("classical")
    config.fusion.parameters.max_alpha = 1.0
    config.fusion.parameters.sq = 0.1
    config.fusion.parameters.steepness_factor = 5.0
    config.fusion.parameters.upsampling_factor = 2
    config.fusion.parameters.sobel_threshold = 0.01
    config.fusion.parameters.dilation_size = 1
    config.fusion.parameters.shave_pixels = 0
    config.fusion.parameters.fusion_use_sobel_for_alpha_mask = False
    config.fusion.parameters.fusion_variation = True
    config.fusion.parameters.alpha_variation = 0.05
    config.fusion.parameters.sq_variation = 0.1
    config.fusion.parameters.steepness_variation = 1.0
    config.fusion.parameters.selected_confidence = "90%"
    # Training settings
    config.training.num_trials = 10
    config.training.trial_selection = "best"
    config.training.validation_ratio = 0.1
    config.training.batch_size = 8
    config.training.epochs = 1000
    config.training.learning_rate = 1e-4
    config.training.gradient_clip_norm = 1.0
    config.training.log_every = None
    config.training.early_stopping_enabled = True
    config.training.early_stopping = {
        "patience": 400,
        "delta": 0.0001,
    }
    config.training.lr_scheduler_enabled = True
    config.training.lr_scheduler = {
        "patience": 200,
        "factor": 0.1,
        "threshold": 1e-5,
    }

    # Fixed model parameters. Only entries in model.search vary between trials.
    parameters = config.model.parameters
    parameters.recon_weight = 10.0
    parameters.beta_kl_start = 0.0
    parameters.beta_kl_max = 0.08
    parameters.beta_kl_warmup_start = 0
    parameters.beta_kl_warmup_epochs = 900
    parameters.free_bits = 0.001
    parameters.recon_loss = "smoothl1"
    parameters.recon_smoothl1_beta = 0.75
    parameters.use_transpose_conv = False
    parameters.fg_weight = 1.0
    parameters.fg_threshold = 0.0
    parameters.drop_path_rate = 0.04
    parameters.dropout = 0.05
    parameters.skip_dropout_p = 0.85
    parameters.skip_alpha = 0.1

    search = config.model.search
    search.clear()
    search.n_res_blocks = IntRange(2, 4)
    search.n_levels = IntRange(3, 4)
    search.z_channels = Choice((16, 32, 64, 96))
    search.bottleneck_dim = Choice((32, 64, 128, 160))
    search.dropout = FloatRange(0.01, 0.20)

def create_generator_configuration(category: str, anomaly_size: tuple[int, int, int]) -> Configuration:
    config = Configuration(f"mvtecad2_{category}", study_folder=study_folder(category))
    config.extraction.anomaly_size = anomaly_size
    config.model.set_model("cVAE_ConvNeXt_2D")
    apply_generator_defaults(config)
    return config


def create_downstream_configuration() -> DownstreamConfiguration:
    config = DownstreamConfiguration()
    config.seed = 42
    config.data.hybrid_fraction = 0.5
    config.data.normal_fraction = 0.5
    config.data.samples_per_epoch = 1000
    config.data.mode = "patch"
    config.data.patch_size = (512, 512)
    config.data.patch_overlap = 0.5
    config.data.texture_root = TEXTURE_ROOT
    config.data.image_scale = 255.0
    config.training.epochs = 100
    config.training.batch_size = 8
    config.training.learning_rate = 1e-4
    config.training.num_workers = 0
    config.training.device = "auto"
    config.training.reconstruction_width = 128
    config.training.segmentation_width = 64
    return config
