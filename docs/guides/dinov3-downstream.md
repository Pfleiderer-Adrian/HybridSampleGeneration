# Paired DINOv3 downstream segmentation

For image/mask-only training, see [DINOv3 segmentation decoder](dinov3-segmentation.md).

The MVTec AD 2 downstream runner supports `draem` (the default, including old
configuration snapshots) and `dinov3_paired`. DINOv3 uses a frozen pretrained
ViT backbone, a trainable segmentation decoder, and a native-resolution image
branch to retain details smaller than a transformer patch. It does not reconstruct
the healthy image. Inference requires only the image, never its healthy partner
or ground-truth mask.

Install the optional dependency in the environment used for downstream training:

```bash
python -m pip install -e ".[dinov3]"
```

Use a separate environment from libcom harmonization if its pinned Transformers
version conflicts with this extra. The default backbone is
`facebook/dinov3-vits16-pretrain-lvd1689m`. Downloading these pretrained weights
requires access to the model on Hugging Face and authentication where required;
alternatively set `config.dinov3.backbone` to a local Hugging Face DINOv3 ViT
directory. `revision` selects the Hub revision and may be pinned to a commit.

```python
from copy import deepcopy
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream

baseline = DownstreamConfiguration(model="dinov3_paired")
baseline.data.mode = "patch"
baseline.data.patch_size = (512, 512)
baseline.data.normal_fraction = 0.75
baseline.data.hybrid_fraction = 0.0  # All anomalous examples use DRAEM synthesis.
baseline.dinov3.decoder_width = 64
baseline.dinov3.healthy_loss_weight = 1.0
baseline.dinov3.consistency_weight = 1.0
baseline.dinov3.dice_weight = 1.0
baseline.dinov3.validation_samples = 128

hybrid = deepcopy(baseline)
hybrid.data.hybrid_fraction = 0.5  # Half of anomalous examples are hybrids.

# generator_config identifies the existing materialized training study;
# manifest is its unchanged train/validation/test split.
run = train_downstream(generator_config, manifest, hybrid)
metrics = evaluate_downstream(run, manifest)
```

`normal_fraction` specifies the proportion of all samples that are healthy;
`hybrid_fraction` specifies the proportion of the remaining anomalous samples
that are hybrids. With 1000 samples per epoch, the example baseline uses
750 normal / 250 DRAEM / 0 hybrid samples; the hybrid configuration uses
750 normal / 125 DRAEM / 125 hybrid samples. Baseline training does not require
a materialized hybrid study. DRAEM texture preparation and source quotas remain
the same for both model types.

Paired training uses the already aligned image/healthy-target/mask triples:

- Segmentation focal loss supervises the input image against its anomaly mask.
- `dice_weight` weights an additional Dice loss on anomalous examples.
- `healthy_loss_weight` weights focal loss on the healthy partner with an empty mask.
- `consistency_weight` weights squared differences between anomaly probabilities
  of both partners outside the defect mask, normalized by background area.

For DRAEM-synthesized anomalies, the partner is the original healthy image; for
hybrids, it is the persisted healthy original; for normal samples both are the
same image. Crops and flips remain aligned. The frozen backbone stays in evaluation
mode; only the projection, image branch and decoder are optimized. Pair constraints
therefore train the decoder predictions, not the frozen feature extractor.

Results are written to `study/downstream/dinov3_paired/<run_id>/`, independently
of DRAEM runs. Checkpoints include backbone configuration and all backbone weights,
allowing reload without downloading pretrained weights again. Evaluation uses
the existing full-image tiling and reports Image AUROC/AP, Pixel AUROC/AP and
`pixel_aupro_0.05`. DINOv3 checkpoint selection minimizes the full weighted paired
loss (focal + Dice + healthy-partner focal + background consistency) on fixed
DRAEM-synthesized pairs from healthy validation originals. The number of pairs is
`dinov3.validation_samples` (default 128). A separate deterministic seed fixes
source selection, crops, synthesis and flips across epochs; no training hybrids
or test images are used. `paired_validation.json` records the validation sources
and settings. History records both real-image `validation_loss` and the paired
`selection_loss`; checkpoints identify `paired_total_loss` as their selection
metric. Real-defect validation/test metrics remain separate. DRAEM keeps its
existing selection by segmentation loss. Existing checkpoints cannot be selected
retroactively across discarded epochs. No experiment is launched by selecting the model
in a configuration file alone.

Backbone token handling follows the
[Transformers DINOv3 documentation](https://huggingface.co/docs/transformers/model_doc/dinov3).

### Multi-layer residual decoder

Set `config.dinov3.decoder = "residual_fpn"` for a deeper decoder usable with
both paired and ordinary segmentation training. It selects four evenly spaced
DINO depths (layers 3, 6, 9, 12 for ViT-S/16), normalizes and projects each to
`decoder_width` channels, and fuses them with progressive upsampling and
residual refinement. The transformer depths all share the same token grid;
the spatial pyramid is constructed by the decoder. A native-resolution RGB
branch contributes local details before the final segmentation head.

`decoder_blocks` controls residual depth in all four fusion stages, the image
branch and the head. With `decoder_blocks = 2`, there are twelve residual blocks,
each containing two 3×3 convolutions. The image branch uses half the feature
width. The backbone remains frozen; paired training still uses the healthy
partner, segmentation loss and background consistency. Inference needs only
one image. Existing `cnn` and `dpt` checkpoints keep their architectures.

For an isolated decoder comparison, retain the backbone, split, mixtures,
training budget and validation selection criterion from the CNN run.

For controlled decoder experiments, `cnn_residual` retains the original final
DINO layer, RGB branch and interpolation, adding only `decoder_blocks` residual
blocks before the output classifier. `cnn_multilayer` retains the shallow CNN
head and RGB branch, concatenating the four selected token depths before the
1×1 projection. It introduces no spatial pyramid or additional normalization,
and uses the original final-layer output for the last depth. These variants
help separate extra decoder depth from the effect of multiple DINO layers.

### Hard-normal patch sampling

`data.hard_normal_fraction` (default `0.0`) replaces that fraction **of the normal
quota**, not of all samples, with mined healthy training patches. For a normal
fraction of `0.75`, setting `hard_normal_fraction = 1/3` gives 50% random normal,
25% hard normal, and 25% unchanged anomaly samples. Set
`data.hard_normal_manifest` to the pool JSON. Source labels remain `normal`;
`training_history.csv` additionally records the exact `hard_normal` count.

`downstream.hard_normals.mine_pool` uses a fixed trained checkpoint on healthy
training images only. It scores candidate patches by their highest 1% anomaly
probabilities averaged together, limits overlap by IoU, retains at most four
patches per image and caps the global pool at 1024 patches. The pool records its
source checkpoint SHA256, training IDs, crop coordinates and mining settings.
It is copied into each training run. The dataset checks the training-ID set,
patch size and crop bounds; mined patches always have an empty anomaly mask and
an identical healthy partner. They still receive the usual aligned flips.
Validation/test images never supply mining candidates. Sampling is deterministic
per seed and epoch; ordinary normal patches and anomaly quotas are retained.

### ConvNeXt residual FPN

`dinov3.decoder = "convnext_fpn"` uses the same four-depth, progressively
upsampled pyramid as `residual_fpn`, replacing the residual refinement blocks
with ConvNeXt-style blocks: depthwise 7×7 convolution, channel LayerNorm,
4× channel expansion, GELU, projection and a learnable residual scale initialized
to 1e-6. It uses no stochastic depth. `decoder_blocks` controls all four fusion
stages, the RGB detail branch and the head; the default of two gives twelve
blocks. These decoder blocks train from scratch; the DINOv3 backbone stays
frozen. This is a DINO backbone with a ConvNeXt-style decoder, not a pretrained
ConvNeXt backbone. With ViT-L/16 the selected DINO depths are 6, 12, 18, 24.
The original block design is described in the
[official ConvNeXt implementation](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py).
