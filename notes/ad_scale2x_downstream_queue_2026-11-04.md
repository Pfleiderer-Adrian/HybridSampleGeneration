# V3: Echt + AD, 2x / Last50k / AAR

Continuation of the fresh appearance queue at `/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103`.

## Execution

Updated 2026-11-05 with user authorization to use both physical GPUs. GPU 0 processes can, fabric, fruit_jelly and rice immediately, since their appearance training is complete. GPU 1 retains the running walnuts appearance training, then processes sheet_metal, vial, wallplugs and walnuts after the original appearance queue releases its lock. The waiting continuation controller was replaced; no appearance training was interrupted. Each worker runs generation and Real+AD downstream training serially on its assigned GPU. Categories are disjoint; scientific settings, seeds, splits and output cache paths are unchanged.

The original continuation directory now holds GPU 1's plan/status; `gpu0_worker/` contains GPU 0's plan/status/logs/telemetry. Its inputs link to the original frozen inputs. Root `jobs/` includes symlinks to GPU 0 category states for a combined per-category view. `dual_gpu_workers.json` records assignments and the new immutable source snapshot. Downstream outputs stay in the original shared downstream directory, separated by category. Previous plan/status are archived in `dual_gpu_migration_*`. A category with incomplete appearance training is refused. The GPU 0 worker may bypass the appearance lock only because its GPU is outside the appearance plan. No automatic retry over partial outputs; no submission packaging is included.

## AD production protocol

1,000 paired native 256x256 patches per category. Shuffled cycles through the exact frozen V3 hybrid pool, choosing a component per draw and taking a centered native 256 crop. The mask includes all frozen hybrid foreground inside that crop. Healthy controls and all mask donors are checked against the unchanged V3 training partition. No hybrid RGB image is an AD input or a downstream source in this variant.

AD uses the coherent last appearance checkpoint at 50,000 updates, AAR on, 2x scale, maxpool latent masks, spatial encoder eval, DDIM200/eta1, seed62000; no learned mask generator. Training seed remains42. Positive component-centered native128 windows cover every mask pixel; large components use 64px stride, windows remain within the native256 crop. RGB windows are jointly upsampled bicubic2x with nearest-neighbor masks. Whole decoded RGB windows are downscaled bicubic to native128 and averaged in overlaps; outside all generated windows the native healthy context remains. There is no mask-based copy of healthy pixels inside a generated window. Native mask labels are saved directly, without resampling, so small islands cannot disappear through inverse label scaling. Window coordinates, foreground counts, controls, donor IDs and file hashes are retained per pair.

This is a documented AD2 adaptation using hybrid-derived masks, not the original learned-mask AD protocol. Overlapping generation may create seams or inconsistent appearance across windows; diagnostics should be reviewed before interpreting downstream results. The earlier 4-case previews show single128 windows at256 generation scale, whereas these downstream inputs restore native scale and include256 context. Synthetic caches and downstream outputs are isolated from earlier native100k AD results.

## Frozen V3 downstream

Only the synthetic source/cache changes. 1000 draws/epoch: 500 healthy,100 real anomalous,400 AD. 100epochs,batch8,Adam1e-4,seed42; unchanged DINOv3/CNN architecture and losses, milestones80/90,gamma0.2, native256 patches, same grouped public training/validation split and validation every5epochs. Best downstream checkpoint selected by public-validation pixelAUPRO@0.05. No local test split; no private test labels used. No Real+AD+Hybrid job is included in this continuation.

The downstream implementation is copied from the existing frozen V3 AD runtime snapshot. All eight configurations were loaded with its configuration class and checked against their original V3 reference; only source-mixture fields and cache path may differ. Geometry tests cover tiny pixels, all borders, hollow/large components, randomized islands, overlap averaging and unchanged outer context.

## Outputs and resource analysis

`inputs/<category>/generated/manifest.json` and paired PNGs; `downstream/real_anomalydiffusion/<category>/downstream/dinov3_paired/<run>/` contains checkpoints, history and validation metrics. This queue trains/evaluates models; private-test submission archives are not packaged by it. Each worker starts its own NVML energy/runtime sidecar for its assigned GPU when processing begins; device energy excludes CPU, memory, storage and PUE, and interference remains unattributed. It does not count the waiting period as generation/training cost. Appearance costs remain in the predecessor's telemetry.

Results directory: `/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103/real_ad_continuation_20261104`
