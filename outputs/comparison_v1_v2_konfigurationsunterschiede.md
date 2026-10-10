# Konfigurationsunterschiede: Hybrid V1 → Hybrid V2

Verglichen werden gespeicherte Generator-Konfigurationen und die Konfigurationen der DRAEM-Läufe. Fehlende Felder sind als „nicht gespeichert“ markiert; dies bedeutet nicht automatisch einen anderen wirksamen Default. Studiennamen, Ausgabe- und Texture-Pfade werden als organisatorische Unterschiede separat aufgeführt.

## can

### DRAEM

V1: `/mnt/results/mvtec2/experiments/comparison_v1/can/hybrid/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/can/hybrid_generation/downstream/draem/20261002T083122Z_7f66e13f/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `data.patch_size` | [512, 512] | [256, 256] |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `data.texture_root` | "/mnt/results/mvtec2/experiments/comparison_v1/can/hybrid_generation/downstream/textures/dtd/images" | "/mnt/results/mvtec2/experiments/comparison_v2/can/hybrid_generation/downstream/textures/dtd/images" |

### Hybrid-Erzeugung

V1: `/mnt/results/mvtec2/experiments/comparison_v1/can/hybrid_generation/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/can/hybrid_generation/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `augmentation.mask_transforms.mask_transform_params.elastic.magnitude` | 5 | nicht gespeichert |
| `augmentation.mask_transforms.mask_transform_params.elastic.sigma` | 30 | nicht gespeichert |
| `augmentation.mask_transforms.mask_transform_params.rotate.max_rotation` | 2.0 | nicht gespeichert |
| `augmentation.mask_transforms.mask_transform_params.stretch.max_stretch` | 1.05 | nicht gespeichert |
| `augmentation.mask_transforms.mask_transform_params.stretch.min_stretch` | 0.98 | nicht gespeichert |
| `augmentation.mask_transforms.mask_transform_params.zoom.max_zoom` | 1.0 | nicht gespeichert |
| `augmentation.mask_transforms.mask_transform_params.zoom.min_zoom` | 0.97 | nicht gespeichert |
| `augmentation.mask_transforms.mask_transform_probs.elastic` | 0.2 | 0.5 |
| `augmentation.mask_transforms.mask_transform_probs.stretch` | 0.2 | 0.5 |
| `extraction.roi.min_size` | [1, 256, 256] | [1, 128, 128] |
| `fusion.parameters.image_harmonization` | nicht gespeichert | "LBM" |
| `fusion.parameters.segmentation_min_alpha` | 0.15 | 0.1 |
| `matching.batch_size` | 8 | 16 |
| `matching.hybrids_per_original` | 3 | 1 |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `study.folder` | "/mnt/results/mvtec2/experiments/comparison_v1/can/hybrid_generation" | "/mnt/results/mvtec2/experiments/comparison_v2/can/hybrid_generation" |
| `study.name` | "mvtecad2_can" | "comparison_v2" |

## fabric

### DRAEM

V1: `/mnt/results/mvtec2/experiments/comparison_v1/fabric/hybrid/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/fabric/hybrid_generation/downstream/draem/20261002T160246Z_334b1c1f/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `data.hybrid_fraction` | 0.5 | 0.25 |
| `data.normal_fraction` | 0.5 | 0.75 |
| `data.patch_size` | [512, 512] | [256, 256] |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `data.texture_root` | "/mnt/results/mvtec2/experiments/comparison_v1/downstream/textures/dtd/images" | "/mnt/results/mvtec2/experiments/comparison_v2/fabric/hybrid_generation/downstream/textures/dtd/images" |

### Hybrid-Erzeugung

V1: `/mnt/results/mvtec2/experiments/comparison_v1/fabric/hybrid_generation/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/fabric/hybrid_generation/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `extraction.anomaly_size` | [3, 64, 64] | [3, 128, 128] |
| `extraction.preserve_aspect_ratio` | nicht gespeichert | true |
| `extraction.roi.preserve_aspect_ratio` | nicht gespeichert | false |
| `fusion.parameters.image_harmonization` | nicht gespeichert | null |
| `fusion.parameters.segmentation_min_alpha` | nicht gespeichert | 0.2 |
| `generation.posterior_skip_source` | nicht gespeichert | "transformed" |
| `matching.anomalies_per_hybrid` | 2 | 1 |
| `matching.max_anomalies_per_hybrid_deviation` | 1 | 0 |
| `matching.routine` | "fixed_from_extraction_control_fusion" | "batchwise" |
| `model.parameters.background_weight` | nicht gespeichert | 0.2 |
| `model.parameters.beta_kl_warmup_epochs` | 900 | 200 |
| `model.parameters.drop_path_rate` | 0.04 | 0.001 |
| `model.parameters.dropout` | 0.05 | 0.001 |
| `model.parameters.fg_threshold` | 0.0 | nicht gespeichert |
| `model.parameters.fg_weight` | 1.0 | nicht gespeichert |
| `model.parameters.foreground_weight` | nicht gespeichert | 0.8 |
| `model.parameters.latent_recon_image_noise_std` | nicht gespeichert | 0.03 |
| `model.parameters.latent_recon_noise_scale` | nicht gespeichert | 1.0 |
| `model.parameters.latent_recon_weight` | nicht gespeichert | 0.0 |
| `model.parameters.recon_smoothl1_beta` | 0.75 | 1.0 |
| `model.parameters.skip_alpha` | 0.1 | 0.0 |
| `model.parameters.skip_alphas` | nicht gespeichert | null |
| `model.parameters.skip_dropout_p` | 0.85 | 1.0 |
| `schema_version` | 9 | 10 |
| `training.log_every` | null | nicht gespeichert |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `study.folder` | "/mnt/results/mvtec2/experiments/comparison_v1/fabric/hybrid_generation" | "/mnt/results/mvtec2/experiments/comparison_v2/fabric/hybrid_generation" |
| `study.name` | "mvtecad2_fabric" | "comparison_v2_fabric" |

## fruit_jelly

### DRAEM

V1: `/mnt/results/mvtec2/experiments/comparison_v1/fruit_jelly/hybrid/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/fruit_jelly/hybrid_generation/downstream/draem/20261004T220248Z_4bc23b76/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `data.hybrid_fraction` | 0.5 | 0.25 |
| `data.normal_fraction` | 0.5 | 0.75 |
| `data.patch_size` | [512, 512] | [256, 256] |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `data.texture_root` | "/mnt/results/mvtec2/experiments/comparison_v1/downstream/textures/dtd/images" | "/mnt/results/mvtec2/experiments/comparison_v2/fruit_jelly/hybrid_generation/downstream/textures/dtd/images" |

### Hybrid-Erzeugung

V1: `/mnt/results/mvtec2/experiments/comparison_v1/fruit_jelly/hybrid_generation/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/fruit_jelly/hybrid_generation/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `extraction.preserve_aspect_ratio` | nicht gespeichert | true |
| `extraction.roi.preserve_aspect_ratio` | nicht gespeichert | false |
| `fusion.parameters.image_harmonization` | nicht gespeichert | null |
| `fusion.parameters.segmentation_min_alpha` | nicht gespeichert | 0.0 |
| `generation.posterior_skip_source` | nicht gespeichert | "transformed" |
| `matching.routine` | "fixed_from_extraction_control_fusion" | "batchwise" |
| `model.parameters.background_weight` | nicht gespeichert | 0.2 |
| `model.parameters.beta_kl_warmup_epochs` | 900 | 200 |
| `model.parameters.drop_path_rate` | 0.04 | 0.001 |
| `model.parameters.dropout` | 0.05 | 0.001 |
| `model.parameters.fg_threshold` | 0.0 | nicht gespeichert |
| `model.parameters.fg_weight` | 1.0 | nicht gespeichert |
| `model.parameters.foreground_weight` | nicht gespeichert | 0.8 |
| `model.parameters.latent_recon_image_noise_std` | nicht gespeichert | 0.03 |
| `model.parameters.latent_recon_noise_scale` | nicht gespeichert | 1.0 |
| `model.parameters.latent_recon_weight` | nicht gespeichert | 0.0 |
| `model.parameters.recon_smoothl1_beta` | 0.75 | 1.0 |
| `model.parameters.skip_alpha` | 0.1 | 0.0 |
| `model.parameters.skip_alphas` | nicht gespeichert | null |
| `model.parameters.skip_dropout_p` | 0.85 | 1.0 |
| `schema_version` | 9 | 10 |
| `training.log_every` | null | nicht gespeichert |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `study.folder` | "/mnt/results/mvtec2/experiments/comparison_v1/fruit_jelly/hybrid_generation" | "/mnt/results/mvtec2/experiments/comparison_v2/fruit_jelly/hybrid_generation" |
| `study.name` | "mvtecad2_fruit_jelly" | "comparison_v2_fruit_jelly" |

## rice

### DRAEM

V1: `/mnt/results/mvtec2/experiments/comparison_v1/rice/hybrid/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/rice/hybrid_generation/downstream/draem/20261002T150708Z_7c4bbd96/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `data.hybrid_fraction` | 0.5 | 0.25 |
| `data.normal_fraction` | 0.5 | 0.75 |
| `data.patch_size` | [512, 512] | [256, 256] |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `data.texture_root` | "/mnt/results/mvtec2/experiments/comparison_v1/downstream/textures/dtd/images" | "/mnt/results/mvtec2/experiments/comparison_v2/rice/hybrid_generation/downstream/textures/dtd/images" |

### Hybrid-Erzeugung

V1: `/mnt/results/mvtec2/experiments/comparison_v1/rice/hybrid_generation/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/rice/hybrid_generation/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `augmentation.mask_transforms.mask_transform_params.elastic.magnitude` | nicht gespeichert | 5 |
| `augmentation.mask_transforms.mask_transform_params.elastic.sigma` | nicht gespeichert | 30 |
| `augmentation.mask_transforms.mask_transform_params.rotate.max_rotation` | nicht gespeichert | 2.0 |
| `augmentation.mask_transforms.mask_transform_params.stretch.max_stretch` | nicht gespeichert | 1.05 |
| `augmentation.mask_transforms.mask_transform_params.stretch.min_stretch` | nicht gespeichert | 0.95 |
| `augmentation.mask_transforms.mask_transform_params.zoom.max_zoom` | nicht gespeichert | 1.0 |
| `augmentation.mask_transforms.mask_transform_params.zoom.min_zoom` | nicht gespeichert | 0.97 |
| `augmentation.mask_transforms.mask_transform_probs.elastic` | nicht gespeichert | 0.5 |
| `augmentation.mask_transforms.mask_transform_probs.local_dilate` | nicht gespeichert | 0 |
| `augmentation.mask_transforms.mask_transform_probs.local_elastic` | nicht gespeichert | 0 |
| `augmentation.mask_transforms.mask_transform_probs.local_rotate` | nicht gespeichert | 0 |
| `augmentation.mask_transforms.mask_transform_probs.local_stretch` | nicht gespeichert | 0 |
| `augmentation.mask_transforms.mask_transform_probs.rotate` | nicht gespeichert | 0.5 |
| `augmentation.mask_transforms.mask_transform_probs.stretch` | nicht gespeichert | 0.5 |
| `augmentation.mask_transforms.mask_transform_probs.zoom` | nicht gespeichert | 0.5 |
| `extraction.anomaly_size` | [3, 64, 64] | [3, 128, 128] |
| `extraction.preserve_aspect_ratio` | nicht gespeichert | true |
| `extraction.roi.min_size` | 0 | [1, 128, 128] |
| `extraction.roi.preserve_aspect_ratio` | nicht gespeichert | true |
| `fusion.parameters.alpha_variation` | 0.02 | 0.01 |
| `fusion.parameters.image_harmonization` | nicht gespeichert | "LBM" |
| `fusion.parameters.max_alpha` | 1.0 | 0.8 |
| `fusion.parameters.segmentation_min_alpha` | nicht gespeichert | 0.1 |
| `fusion.parameters.shave_pixels` | 1 | 0 |
| `fusion.parameters.sq` | 0.75 | 1.0 |
| `fusion.parameters.sq_variation` | 0.05 | 0.02 |
| `fusion.parameters.steepness_factor` | 3.5 | 3.0 |
| `fusion.parameters.steepness_variation` | 0.25 | 0.15 |
| `fusion.parameters.upsampling_factor` | 2 | 4 |
| `generation.posterior_skip_source` | "original" | "transformed" |
| `generation.variants_per_real_anomaly` | 3 | 5 |
| `generation.variation_strength` | 1.25 | 1.5 |
| `matching.gradient_weight` | 2.0 | 0.7 |
| `matching.hybrids_per_original` | 3 | 1 |
| `matching.intensity_weight` | 1.0 | 0.3 |
| `matching.reuse_synthetic_across_hybrids` | true | false |
| `matching.routine` | "fixed_from_extraction_control_fusion" | "batchwise" |
| `model.parameters.background_weight` | nicht gespeichert | 0.2 |
| `model.parameters.beta_kl_warmup_epochs` | 900 | 200 |
| `model.parameters.drop_path_rate` | 0.04 | 0.001 |
| `model.parameters.dropout` | 0.05 | 0.001 |
| `model.parameters.fg_threshold` | 0.0 | nicht gespeichert |
| `model.parameters.fg_weight` | 1.0 | nicht gespeichert |
| `model.parameters.foreground_weight` | nicht gespeichert | 0.8 |
| `model.parameters.n_levels` | 4 | 5 |
| `model.parameters.recon_smoothl1_beta` | 0.75 | 1.0 |
| `model.parameters.skip_alpha` | 0.1 | 0.0 |
| `model.parameters.skip_dropout_p` | 0.85 | 1.0 |
| `model.search.n_levels.high` | 4 | 5 |
| `model.search.n_levels.low` | 3 | 4 |
| `schema_version` | 9 | 10 |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `study.folder` | "/mnt/results/mvtec2/experiments/comparison_v1/rice/hybrid_generation" | "/mnt/results/mvtec2/experiments/comparison_v2/rice/hybrid_generation" |
| `study.name` | "mvtecad2_rice" | "comparison_v2_rice" |

## sheet_metal

### DRAEM

V1: `/mnt/results/mvtec2/experiments/comparison_v1/sheet_metal/hybrid/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/sheet_metal/hybrid_generation/downstream/draem/20261005T123519Z_87f2026d/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `data.hybrid_fraction` | 0.5 | 0.25 |
| `data.normal_fraction` | 0.5 | 0.75 |
| `data.patch_size` | [512, 512] | [256, 256] |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `data.texture_root` | "/mnt/results/mvtec2/experiments/comparison_v1/downstream/textures/dtd/images" | "/mnt/results/mvtec2/experiments/comparison_v2/sheet_metal/hybrid_generation/downstream/textures/dtd/images" |

### Hybrid-Erzeugung

V1: `/mnt/results/mvtec2/experiments/comparison_v1/sheet_metal/hybrid_generation/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/sheet_metal/hybrid_generation/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `extraction.roi.preserve_aspect_ratio` | nicht gespeichert | false |
| `fusion.parameters.image_harmonization` | nicht gespeichert | null |
| `fusion.parameters.segmentation_min_alpha` | nicht gespeichert | 0.0 |
| `matching.routine` | "global" | "batchwise" |
| `model.name` | "paired_cVAE_ConvNeXt_2D" | "cVAE_ConvNeXt_2D" |
| `model.parameters.drop_path_rate` | 0.04 | 0.001 |
| `model.parameters.dropout` | 0.05 | 0.001 |
| `model.parameters.identity_pair_probability` | 0.2 | nicht gespeichert |
| `model.parameters.latent_recon_image_noise_std` | nicht gespeichert | 0.03 |
| `model.parameters.latent_recon_noise_scale` | nicht gespeichert | 1.0 |
| `model.parameters.latent_recon_weight` | nicht gespeichert | 0.0 |
| `model.parameters.recon_smoothl1_beta` | 0.75 | 1.0 |
| `model.parameters.skip_alpha` | nicht gespeichert | 0.0 |
| `model.parameters.skip_alphas` | nicht gespeichert | null |
| `model.parameters.skip_dropout_p` | nicht gespeichert | 1.0 |
| `model.parameters.skip_dropout_ps` | nicht gespeichert | null |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `study.folder` | "/mnt/results/mvtec2/experiments/comparison_v1/sheet_metal/hybrid_generation" | "/mnt/results/mvtec2/experiments/comparison_v2/sheet_metal/hybrid_generation" |
| `study.name` | "mvtecad2_sheet_metal" | "comparison_v2_sheet_metal" |

## vial

### DRAEM

V1: `/mnt/results/mvtec2/experiments/comparison_v1/vial/hybrid/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/vial/hybrid_generation/downstream/draem/20261005T232200Z_94be9b23/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `data.hybrid_fraction` | 0.5 | 0.25 |
| `data.normal_fraction` | 0.5 | 0.75 |
| `data.patch_size` | [512, 512] | [256, 256] |
| `training.num_workers` | 0 | 4 |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `data.texture_root` | "/mnt/results/mvtec2/experiments/comparison_v1/downstream/textures/dtd/images" | "/mnt/results/mvtec2/experiments/comparison_v2/vial/hybrid_generation/downstream/textures/dtd/images" |

### Hybrid-Erzeugung

V1: `/mnt/results/mvtec2/experiments/comparison_v1/vial/hybrid_generation/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/vial/hybrid_generation/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `extraction.preserve_aspect_ratio` | nicht gespeichert | true |
| `extraction.roi.preserve_aspect_ratio` | nicht gespeichert | false |
| `fusion.parameters.image_harmonization` | nicht gespeichert | null |
| `fusion.parameters.segmentation_min_alpha` | nicht gespeichert | 0.0 |
| `generation.posterior_skip_source` | nicht gespeichert | "transformed" |
| `matching.routine` | "fixed_from_extraction_control_fusion" | "batchwise" |
| `model.parameters.background_weight` | nicht gespeichert | 0.2 |
| `model.parameters.beta_kl_warmup_epochs` | 900 | 200 |
| `model.parameters.drop_path_rate` | 0.04 | 0.001 |
| `model.parameters.dropout` | 0.05 | 0.001 |
| `model.parameters.fg_threshold` | 0.0 | nicht gespeichert |
| `model.parameters.fg_weight` | 1.0 | nicht gespeichert |
| `model.parameters.foreground_weight` | nicht gespeichert | 0.8 |
| `model.parameters.latent_recon_image_noise_std` | nicht gespeichert | 0.03 |
| `model.parameters.latent_recon_noise_scale` | nicht gespeichert | 1.0 |
| `model.parameters.latent_recon_weight` | nicht gespeichert | 0.0 |
| `model.parameters.recon_smoothl1_beta` | 0.75 | 1.0 |
| `model.parameters.skip_alpha` | 0.1 | 0.0 |
| `model.parameters.skip_alphas` | nicht gespeichert | null |
| `model.parameters.skip_dropout_p` | 0.85 | 1.0 |
| `schema_version` | 9 | 10 |
| `training.log_every` | null | nicht gespeichert |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `study.folder` | "/mnt/results/mvtec2/experiments/comparison_v1/vial/hybrid_generation" | "/mnt/results/mvtec2/experiments/comparison_v2/vial/hybrid_generation" |
| `study.name` | "mvtecad2_vial" | "comparison_v2_vial" |

## wallplugs

### DRAEM

V1: `/mnt/results/mvtec2/experiments/comparison_v1/wallplugs/hybrid/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/wallplugs/hybrid_generation/downstream/draem/20261006T095921Z_b7c0e458/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `data.hybrid_fraction` | 0.5 | 0.25 |
| `data.normal_fraction` | 0.5 | 0.75 |
| `data.patch_size` | [512, 512] | [256, 256] |
| `training.num_workers` | 0 | 4 |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `data.texture_root` | "/mnt/results/mvtec2/experiments/comparison_v1/downstream/textures/dtd/images" | "/mnt/results/mvtec2/experiments/comparison_v2/wallplugs/hybrid_generation/downstream/textures/dtd/images" |

### Hybrid-Erzeugung

V1: `/mnt/results/mvtec2/experiments/comparison_v1/wallplugs/hybrid_generation/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/wallplugs/hybrid_generation/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `fusion.parameters.image_harmonization` | nicht gespeichert | null |
| `fusion.parameters.segmentation_min_alpha` | nicht gespeichert | 0.0 |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `study.folder` | "/mnt/results/mvtec2/experiments/comparison_v1/wallplugs/hybrid_generation" | "/mnt/results/mvtec2/experiments/comparison_v2/wallplugs/hybrid_generation" |
| `study.name` | "mvtecad2_wallplugs" | "comparison_v2_wallplugs" |

## walnuts

### DRAEM

V1: `/mnt/results/mvtec2/experiments/comparison_v1/walnuts/hybrid/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/walnuts/hybrid_generation/downstream/draem/20261007T070005Z_62716436/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `data.hybrid_fraction` | 0.5 | 0.25 |
| `data.normal_fraction` | 0.5 | 0.75 |
| `data.patch_size` | [512, 512] | [256, 256] |
| `training.num_workers` | 0 | 4 |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `data.texture_root` | "/mnt/results/mvtec2/experiments/comparison_v1/downstream/textures/dtd/images" | "/mnt/results/mvtec2/experiments/comparison_v2/walnuts/hybrid_generation/downstream/textures/dtd/images" |

### Hybrid-Erzeugung

V1: `/mnt/results/mvtec2/experiments/comparison_v1/walnuts/hybrid_generation/configuration.json`  
V2: `/mnt/results/mvtec2/experiments/comparison_v2/walnuts/hybrid_generation/configuration.json`

| Parameter | V1 | V2 |
|---|---|---|
| `extraction.preserve_aspect_ratio` | nicht gespeichert | true |
| `extraction.roi.preserve_aspect_ratio` | nicht gespeichert | false |
| `fusion.parameters.image_harmonization` | nicht gespeichert | null |
| `fusion.parameters.segmentation_min_alpha` | nicht gespeichert | 0.0 |
| `generation.posterior_skip_source` | nicht gespeichert | "transformed" |
| `matching.routine` | "fixed_from_extraction_control_fusion" | "batchwise" |
| `model.parameters.background_weight` | nicht gespeichert | 0.2 |
| `model.parameters.beta_kl_warmup_epochs` | 900 | 200 |
| `model.parameters.drop_path_rate` | 0.04 | 0.001 |
| `model.parameters.dropout` | 0.05 | 0.001 |
| `model.parameters.fg_threshold` | 0.0 | nicht gespeichert |
| `model.parameters.fg_weight` | 1.0 | nicht gespeichert |
| `model.parameters.foreground_weight` | nicht gespeichert | 0.8 |
| `model.parameters.latent_recon_image_noise_std` | nicht gespeichert | 0.03 |
| `model.parameters.latent_recon_noise_scale` | nicht gespeichert | 1.0 |
| `model.parameters.latent_recon_weight` | nicht gespeichert | 0.0 |
| `model.parameters.recon_smoothl1_beta` | 0.75 | 1.0 |
| `model.parameters.skip_alpha` | 0.1 | 0.0 |
| `model.parameters.skip_alphas` | nicht gespeichert | null |
| `model.parameters.skip_dropout_p` | 0.85 | 1.0 |
| `schema_version` | 9 | 10 |
| `training.log_every` | null | nicht gespeichert |

Organisatorische Unterschiede:

| Parameter | V1 | V2 |
|---|---|---|
| `study.folder` | "/mnt/results/mvtec2/experiments/comparison_v1/walnuts/hybrid_generation" | "/mnt/results/mvtec2/experiments/comparison_v2/walnuts/hybrid_generation" |
| `study.name` | "mvtecad2_walnuts" | "comparison_v2_walnuts" |

