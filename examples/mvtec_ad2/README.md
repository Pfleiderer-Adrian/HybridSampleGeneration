# MVTec AD 2 example

This example reproduces the V3 MVTec AD 2 benchmark pipeline: a grouped
train/validation split, synthetic anomalies from several sources, and a
DINOv3 segmentation model trained on a fixed mixture of healthy, real and
synthetic examples. The split, generator and downstream settings correspond
to the submitted V3 runs.

This is a custom anomaly-supervised protocol, not the official unsupervised
MVTec benchmark.

```bash
pip install -e ".[examples,dinov3]"
python -m examples.mvtec_ad2.run --category can
```

The full documentation is in the wiki:

- [MVTec AD 2 benchmark](https://pfleiderer-adrian.github.io/HybridSampleGeneration/mvtec-ad2/): setup, running a category, layout, split and synthetic sources
- [Downstream training](https://pfleiderer-adrian.github.io/HybridSampleGeneration/mvtec-ad2/downstream-training/): preset, real training defects, source quotas and fixed hybrids
- [Downstream evaluation](https://pfleiderer-adrian.github.io/HybridSampleGeneration/mvtec-ad2/downstream-evaluation/): AUC-PRO and checkpoint selection

AnomalyDiffusion generation has its own
[README](synthetic_data_generation/anomalydiffusion/README.md).
