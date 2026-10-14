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
