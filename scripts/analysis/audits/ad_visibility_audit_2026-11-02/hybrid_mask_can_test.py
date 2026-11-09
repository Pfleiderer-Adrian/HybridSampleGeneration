"""AD inference on final frozen hybrid masks and their exact healthy controls."""

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
import sqlite3
from types import SimpleNamespace
import time

import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects
import torch

from experiments.anomalydiffusion.run import read_inputs, learned_model, seed_all, image_array
from experiments.anomalydiffusion.prepare import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if RESULTS_ROOT not in output.parents:
        raise ValueError('Diagnostic results must remain in the results folder.')
    output.mkdir(exist_ok=False, parents=True)
    root, patches, protocol = read_inputs('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/can')
    study = Path(protocol['hybrid_study'])
    frozen = json.loads((root / 'hybrid_provenance.json').read_text())
    split = json.loads((root / 'split_manifest.json').read_text())
    train_ids = {r['sample_id'] for r in split['partitions']['train']}
    con = sqlite3.connect(f'file:{study}/artifacts.sqlite?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    seen, cases = set(), []
    for item in frozen:
        if item['original_id'] in seen:
            continue
        hybrid = dict(con.execute('SELECT * FROM hybrid_samples WHERE id=?', (item['hybrid_id'],)).fetchone())
        original = dict(con.execute('SELECT * FROM original_samples WHERE id=?', (item['original_id'],)).fetchone())
        if original['has_anomaly'] or original['source_name'] not in train_ids:
            raise ValueError('Control must be a healthy training sample.')
        healthy = np.load(study / original['image_path'], allow_pickle=False)
        image = np.load(study / hybrid['image_path'], allow_pickle=False)
        mask = np.any(np.load(study / hybrid['segmentation_path'], allow_pickle=False) > 0, axis=0)
        if healthy.shape != image.shape or mask.shape != image.shape[-2:]:
            raise ValueError('Unaligned source pair.')
        labeled, n = label(mask)
        sizes = np.bincount(labeled.ravel()); sizes[0] = 0
        component = int(np.argmax(sizes))
        region = find_objects(labeled)[component-1]
        y = int(np.clip((region[0].start+region[0].stop)//2-128, 0, mask.shape[0]-256))
        x = int(np.clip((region[1].start+region[1].stop)//2-128, 0, mask.shape[1]-256))
        window = (slice(y,y+256),slice(x,x+256))
        i = len(cases)
        arrays = dict(healthy=healthy[:,y:y+256,x:x+256], hybrid=image[:,y:y+256,x:x+256])
        for key, value in arrays.items():
            Image.fromarray(np.clip(value.transpose(1,2,0),0,255).round().astype('uint8')).save(output/f'{i:02}_{key}.png')
        cropped_mask = mask[window]
        Image.fromarray(cropped_mask.astype('uint8')*255).save(output/f'{i:02}_mask.png')
        cases.append(dict(index=i, frozen_provenance=item, control_sample_id=original['source_name'],
            crop_xywh=[x,y,256,256], largest_component_pixels=int(sizes[component]),
            largest_component_fully_in_crop=bool(region[0].start>=y and region[0].stop<=y+256 and region[1].start>=x and region[1].stop<=x+256),
            cropped_mask_pixels=int(cropped_mask.sum()),
            source_paths=dict(healthy=str(study/original['image_path']), hybrid=str(study/hybrid['image_path']), mask=str(study/hybrid['segmentation_path'])),
            healthy=str(output/f'{i:02}_healthy.png'), hybrid=str(output/f'{i:02}_hybrid.png'), mask=str(output/f'{i:02}_mask.png')))
        seen.add(item['original_id'])
        if len(cases)==4:
            break
    con.close()
    if len(cases)!=4:
        raise ValueError('Four eligible distinct controls required.')
    for c in cases:
        c['input_sha256']={k:sha256(c[k]) for k in ['healthy','hybrid','mask']}
    record=dict(category='can', diagnostic_only=True, selection='first four distinct healthy controls in frozen provenance order; native crop centered on largest final-mask component',
        mask_policy='unchanged final hybrid mask, cropped with same window as healthy/hybrid; no resizing, dilation, cleaning or recomposition',
        cases=cases, variants=[], seed_base=52000, ddim_steps=200, ddim_eta=1., latent_mask_policy='maxpool',
        caption='a photo of a *', study=str(study))
    (output/'protocol.json').write_text(json.dumps(record,indent=2))
    config=SimpleNamespace(checkpoint=Path('/mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt'),latent_mask_policy='maxpool',smoke=False)
    seed_all(42)
    model, metadata=learned_model('appearance',root/'appearance_training/selection.json',config,patches,output/'work')
    record['checkpoint']=metadata
    (output/'protocol.json').write_text(json.dumps(record,indent=2))
    with torch.no_grad():
        for aar in [False,True]:
            flag='on' if aar else 'off'
            for start in range(0,4,2):
                indices=[start,start+1]; seed=52000+start//2; seed_all(seed)
                batch=dict(image=torch.from_numpy(np.stack([image_array(cases[i]['healthy']) for i in indices])).cuda(),
                    mask=torch.from_numpy(np.stack([np.asarray(Image.open(cases[i]['mask']).convert('L'))>0 for i in indices]).astype('float32')).cuda(),
                    caption=['a photo of a *']*2,name=['can+bad']*2)
                started=time.monotonic()
                logs=model.log_images(batch,N=2,sample=False,inpaint=True,unconditional_only=True,ddim_steps=200,ddim_eta=1.,adaptive_mask=aar)
                for offset,i in enumerate(indices):
                    paths={}
                    for key,name in [('samples_inpainting',f'ad_aar_{flag}'),('reconstruction',f'reconstruction_aar_{flag}')]:
                        value=logs[key][offset].detach().cpu().permute(1,2,0).numpy()
                        if not np.isfinite(value).all():raise ValueError('Nonfinite output.')
                        path=output/f'{i:02}_{name}.png';Image.fromarray(np.clip((value+1)*127.5,0,255).round().astype('uint8')).save(path)
                        paths[key]=str(path)
                    record['variants'].append(dict(index=i,aar=aar,seed=seed,**paths))
                (output/'progress.json').write_text(json.dumps(dict(completed_images=len(record['variants']),target_images=8)))
                print(flag,indices,'seconds',time.monotonic()-started,flush=True)
                del logs,batch;gc.collect()
    (output/'results.json').write_text(json.dumps(record,indent=2))
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    main()
