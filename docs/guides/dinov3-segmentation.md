# DINOv3 segmentation decoder

Select `dinov3_segmentation` for supervised image-to-mask training without a
healthy-partner loss or pair consistency. DRAEM and `dinov3_paired` remain
available independently.

```python
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration

config = DownstreamConfiguration(model="dinov3_segmentation")
config.data.normal_fraction = 0.75
config.data.hybrid_fraction = 0.5
config.dinov3.backbone = "facebook/dinov3-vits16-pretrain-lvd1689m"
config.dinov3.dice_weight = 1.0
config.dinov3.decoder = "dpt"
config.dinov3.decoder_width = 64
config.dinov3.decoder_blocks = 2
```

At 1000 samples per epoch this uses 750 normal, 125 DRAEM-synthesized and
125 hybrid samples. Set `hybrid_fraction = 0.0` for a baseline with 750 normal
and 250 DRAEM samples. Mixture semantics, aligned cropping, flips and stored
splits are unchanged. The shared dataset still supplies healthy targets, but
this model never uses them in its loss.

The frozen DINOv3 backbone, trainable decoder and native-resolution image-detail
branch are shared with [paired DINOv3](dinov3-downstream.md). Install the same
`dinov3` optional extra and provide pretrained model access. Training uses focal
loss plus `dice_weight` times Dice loss on anomalous examples. Healthy examples
use focal loss against an empty mask. `healthy_loss_weight`, `consistency_weight`
and `validation_samples` are only relevant to the paired model.

Checkpoint selection minimizes `segmentation_total_loss` (focal + weighted Dice)
on real validation images and masks. It requires no healthy counterpart or
synthetic validation pairs. The previous focal-only `segmentation_loss` remains
reported separately. Both `best.pt` and `last.pt` include backbone configuration
and weights for offline evaluation. Results live in
`study/downstream/dinov3_segmentation/<run_id>/`.

Inference needs only one image and produces pixel anomaly scores. Existing tiled
evaluation and Image AUROC/AP, Pixel AUROC/AP and AUC-PRO0.05 are supported.
Selecting the model does not launch an experiment automatically.

The optional `dpt` decoder reads four equally spaced transformer layers (3, 6,
9, 12 for ViT-S). Each layer is normalized, projected and reassembled at a
separate spatial scale. Four coarse-to-fine fusion stages each have a residual
skip block and `decoder_blocks` refinement blocks with two 3x3 convolutions per
block. A deep native-resolution detail branch and residual output head refine the
prediction. Width and depth are configurable. This is a DPT-style implementation,
not a pretrained official DPT head. `decoder="cnn"` remains the default to preserve
old configurations and checkpoint compatibility.
