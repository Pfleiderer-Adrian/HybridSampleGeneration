"""Frozen can inference: learned/real masks crossed with reference AAR off/on."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import RESULTS_ROOT

import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from experiments.anomalydiffusion.run import read_inputs, learned_model, seed_all, image_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    results_root = RESULTS_ROOT.resolve()
    if results_root not in output.parents:
        raise ValueError('Diagnostic output must stay in the results folder.')
    output.mkdir(exist_ok=False, parents=True)
    root, patches, protocol = read_inputs('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/can')
    generated = json.loads((root / 'generated/manifest.json').read_text())
    selected, seen = [], set()
    for row in patches['rows']:
        if row['label'] != 'good' and '_regular.png' in row['sample_id'] and row['sample_id'] not in seen:
            selected.append(row); seen.add(row['sample_id'])
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise ValueError('Need six distinct regular-light training images.')
    cases = []
    for i, real in enumerate(selected):
        recipient = generated['rows'][i]
        for key, source in [('healthy', recipient['target_path']), ('learned_mask', recipient['mask_path']),
                            ('real_mask', real['mask_path']), ('real_defect', real['image_path'])]:
            Image.open(source).save(output / f'{i:02}_{key}.png')
        cases.append(dict(index=i, mask_source=real, recipient_source=recipient,
                          healthy=str(output / f'{i:02}_healthy.png')))
    config = SimpleNamespace(checkpoint=Path('/mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt'),
                             latent_mask_policy='maxpool', smoke=False)
    seed_all(42)
    model, metadata = learned_model('appearance', root / 'appearance_training/selection.json',
                                   config, patches, output / 'work')
    record = dict(category='can', smoke_test=True, diagnostic_only=True,
                  selection='first six distinct regular-light anomalous training images; first native patch per image',
                  backgrounds='first six recipients from completed can generated manifest',
                  checkpoint=metadata, seed_base=42000, batch_size=2, ddim_steps=200, ddim_eta=1.,
                  latent_mask_policy='maxpool', caption='a photo of a *', cases=cases, variants=[])
    (output / 'protocol.json').write_text(json.dumps(record, indent=2))
    with torch.no_grad():
        for source in ['learned', 'real']:
            for aar in [False, True]:
                name = f'{source}_aar_{"on" if aar else "off"}'
                for start in range(0, 6, 2):
                    seed = 42000 + start // 2
                    seed_all(seed)
                    indices = [start, start+1]
                    masks = [np.asarray(Image.open(output / f'{i:02}_{source}_mask.png').convert('L')) > 0 for i in indices]
                    batch = dict(image=torch.from_numpy(np.stack([image_array(cases[i]['healthy']) for i in indices])).cuda(),
                                 mask=torch.from_numpy(np.stack(masks).astype('float32')).cuda(),
                                 caption=['a photo of a *']*2, name=['can+bad']*2)
                    started = time.monotonic()
                    logs = model.log_images(batch, N=2, sample=False, inpaint=True, unconditional_only=True,
                                            ddim_steps=200, ddim_eta=1., adaptive_mask=aar)
                    for offset, i in enumerate(indices):
                        value = logs['samples_inpainting'][offset].detach().cpu().permute(1,2,0).numpy()
                        if not np.isfinite(value).all():
                            raise ValueError('Nonfinite generated pixels.')
                        image = np.clip((value+1)*127.5,0,255).round().astype('uint8')
                        path = output / f'{i:02}_{name}.png'
                        Image.fromarray(image).save(path)
                        # Reconstruction uses the same background posterior draw in this call.
                        recon = logs['reconstruction'][offset].detach().cpu().permute(1,2,0).numpy()
                        recon_path = output / f'{i:02}_{name}_reconstruction.png'
                        Image.fromarray(np.clip((recon+1)*127.5,0,255).round().astype('uint8')).save(recon_path)
                        record['variants'].append(dict(index=i, mask_source=source, aar=aar, seed=seed,
                            image=str(path), mask=str(output / f'{i:02}_{source}_mask.png'),
                            healthy=cases[i]['healthy'], reconstruction=str(recon_path)))
                    print(name, indices, 'seconds',time.monotonic()-started,flush=True)
                    (output / 'progress.json').write_text(json.dumps({'completed_images':len(record['variants']),'target_images':24}))
                    del logs, batch
                    gc.collect()
    (output / 'results.json').write_text(json.dumps(record, indent=2))
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
