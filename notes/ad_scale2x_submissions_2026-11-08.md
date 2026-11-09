# MVTec AD 2: AD submission export, 2026-11-08

Authorized: prepare submission archives for Real + AD + Hybrid and Real + AD.
Exact checkpoints from the comparison in /mnt/results/mvtec2/reports/ad_mixed_comparison_2026-11-08; no retraining.
AD: 2x, Last50k, AAR. Two physical GPUs, one variant per GPU.

Each bundle contains frozen best checkpoints, configurations, split manifests, source snapshot,
transformers/tokenizers runtime, private-inference status, public threshold calibration,
official format checker, final archive verification and SHA256.
Expected per archive: 4090 float16 TIFF anomaly maps + 4090 uint8 binary PNG masks.
Threshold objective: maximum pooled pixel F1 on public validation, after float16 quantization.

Batch pointer: /mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103/latest_private_submission_batch.json

- real_anomalydiffusion_hybrid: `/mnt/results/mvtec2/submissions/v3_real_anomalydiffusion_hybrid_scale2x50k_20261108T154428Z` (controller PID 2626302).
- real_anomalydiffusion: `/mnt/results/mvtec2/submissions/v3_real_anomalydiffusion_scale2x50k_20261108T154428Z` (controller PID 2626303).

Completion is recorded only when each bundle status.json has stage=complete and official_checker_passed=true.
No upload is performed.

Completed 2026-11-08T16:41:20.298171+00:00. Both archives passed official validation, exact member validation, gzip integrity and SHA256 creation. Details: /mnt/results/mvtec2/reports/ad_submissions_2026-11-08.md.
