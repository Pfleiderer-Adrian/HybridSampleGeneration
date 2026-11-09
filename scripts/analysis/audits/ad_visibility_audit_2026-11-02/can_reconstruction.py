"""CPU-only frozen first-stage diagnostics; no generator/downstream mutation."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import PROJECT_ROOT, audit_dir

import gc
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

PROJECT = PROJECT_ROOT
TASK = PROJECT / 'experiments/anomalydiffusion'
sys.path.insert(0, str(TASK / 'upstream'))
sys.path.insert(0, str(TASK / '.tools/taming-transformers'))
from ldm.util import instantiate_from_config

OUT = audit_dir('ad_visibility_audit_2026-11-02') / 'can'
ROOT = Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/can')


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = OmegaConf.load(TASK / 'upstream/configs/latent-diffusion/txt2img-1p4B-finetune.yaml')
    vae = instantiate_from_config(config.model.params.first_stage_config).eval()
    checkpoint = torch.load('/mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt', map_location='cpu')
    prefix = 'first_stage_model.'
    weights = {k[len(prefix):]: v for k, v in checkpoint['state_dict'].items() if k.startswith(prefix)}
    vae.load_state_dict(weights, strict=True)
    del checkpoint, weights
    gc.collect()
    rows = [r for r in json.loads((ROOT / 'training_patches.json').read_text())['rows'] if r['label'] != 'good']
    # One regular-light view for each distinct source capture and crop position.
    selected = [r for r in rows if '_regular.png' in r['sample_id']]
    records = []
    with torch.no_grad():
        for i, row in enumerate(selected):
            mask = np.asarray(Image.open(row['mask_path']).convert('L')) > 0
            value = torch.from_numpy(np.repeat(mask[None, :, :], 3, axis=0).astype('float32'))[None] * 2 - 1
            posterior = vae.encode(value)
            for mode, latent in [('mode', posterior.mode()), ('sample', posterior.sample())]:
                decoded = vae.decode(latent)[0].mean(0).numpy()
                recovered = decoded > .8
                union = np.count_nonzero(mask | recovered)
                records.append(dict(sample_id=row['sample_id'], crop_xywh=row['crop_xywh'],
                    posterior=mode, original_pixels=int(mask.sum()), recovered_pixels=int(recovered.sum()),
                    iou=float(np.count_nonzero(mask & recovered) / union) if union else 1.,
                    recall=float(np.count_nonzero(mask & recovered) / mask.sum()),
                    max_decoded=float(decoded.max())))
                Image.fromarray(np.clip((decoded + 1) * 127.5, 0, 255).astype('uint8')).save(OUT / f'mask_reconstruction_{i:02}_{mode}.png')
            print('mask', i, records[-2:], flush=True)
        pairs = json.loads((ROOT / 'generated/manifest.json').read_text())['rows'][:6]
        image_records = []
        for i, row in enumerate(pairs):
            target = np.asarray(Image.open(row['target_path']).convert('RGB'), dtype='float32')
            generated = np.asarray(Image.open(row['image_path']).convert('RGB'), dtype='float32')
            mask = np.asarray(Image.open(row['mask_path']).convert('L')) > 0
            value = torch.from_numpy(target.transpose(2,0,1).copy())[None] / 127.5 - 1
            posterior = vae.encode(value)
            for mode, latent in [('mode', posterior.mode()), ('sample', posterior.sample())]:
                reconstructed = np.clip((vae.decode(latent)[0].permute(1,2,0).numpy()+1)*127.5,0,255)
                error = np.abs(reconstructed - target).mean(2)
                image_records.append(dict(id=i, posterior=mode,
                    reconstruction_inside_mae=float(error[mask].mean()),
                    reconstruction_outside_mae=float(error[~mask].mean()),
                    generation_outside_mae=float(np.abs(generated-target).mean(2)[~mask].mean())))
                Image.fromarray(reconstructed.round().astype('uint8')).save(OUT / f'healthy_reconstruction_{i:02}_{mode}.png')
    summary = dict(device='cpu', seed=42, scope='regular-view can training masks and first six generated healthy partners',
        mask_threshold_in_minus1_plus1_space=.8, masks=records, healthy_reconstructions=image_records)
    (OUT / 'reconstruction.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
