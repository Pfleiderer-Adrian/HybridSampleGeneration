"""Isolated scale, appearance and inference probes; no study mutations."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from experiments.anomalydiffusion.runtime import load_model
from experiments.anomalydiffusion.run import read_inputs, seed_all, image_array

ROOT = Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/can')
INPUT = report_dir('ad_can_hybrid_mask_test_2026-11-02')
OUT = report_dir('ad_can_deep_diagnosis_2026-11-03')


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))


def rows_for(part):
    rows = json.loads((ROOT / f'{part}_patches.json').read_text())['rows']
    selected, seen = [], set()
    for r in rows:
        if r.get('label', 'bad') != 'good' and '_regular.png' in r['sample_id'] and r['sample_id'] not in seen:
            selected.append(r); seen.add(r['sample_id'])
        if len(selected) == 4: break
    return selected


def batch_for(rows):
    return dict(image=torch.from_numpy(np.stack([image_array(r['image_path']) for r in rows])).cuda(),
        mask=torch.from_numpy(np.stack([np.asarray(Image.open(r['mask_path']).convert('L')) > 0 for r in rows]).astype('float32')).cuda(),
        name=['can+bad'] * len(rows), caption=['a photo of a *'] * len(rows))


def probe_loss(model, state_name):
    records = []
    model.eval()
    with torch.no_grad():
        for part in ['training', 'validation']:
            batch = batch_for(rows_for(part))
            seed_all(63000)
            z, c, info = model.get_input(batch, model.first_stage_key, force_c_encode=True)
            seed_all(64000)
            noise = torch.randn_like(z)
            for step in [50, 250, 500, 750]:
                t = torch.full((z.shape[0],), step, dtype=torch.long, device=z.device)
                loss, log = model.p_losses(z, c, t, mask=info['mask'], noise=noise)
                occupancy = torch.nn.functional.adaptive_max_pool2d(info['mask'], (32, 32)).mean().item()
                records.append(dict(state=state_name, partition=part, timestep=step,
                    pixel_scaled_loss=float(loss), mean_latent_mask_fraction=occupancy,
                    occupancy_normalized_loss=float(loss) / occupancy))
            del batch, z, c, info
    return records


def inspect_checkpoint(state, initial):
    sd = state['state_dict']
    groups = {}
    for label, prefix in [('appearance', 'embedding_manager.string_to_param_dict.'),
                          ('spatial', 'embedding_manager.spatial_encoder_model.')]:
        keys = [k for k in initial if k.startswith(prefix) and initial[k].is_floating_point()]
        delta2, initial2, trained2 = 0., 0., 0.
        finite = True
        for k in keys:
            a, b = initial[k].float(), sd[k].float()
            finite = finite and bool(torch.isfinite(b).all())
            delta2 += float(((b-a)**2).sum()); initial2 += float((a*a).sum()); trained2 += float((b*b).sum())
        groups[label] = dict(tensors=len(keys), finite=finite, includes_batchnorm_buffers=(label=='spatial'), delta_l2=delta2**.5,
            initial_l2=initial2**.5, trained_l2=trained2**.5,
            relative_delta_l2=(delta2/max(initial2,1e-30))**.5)
    opt = state['optimizer_states'][0]
    steps = [float(v['step']) for v in opt['state'].values() if 'step' in v]
    return dict(actual_step=state.get('hsg_actual_step'), groups=groups,
        optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt['param_groups']],
        optimizer_parameter_states=len(opt['state']), optimizer_step_min=min(steps), optimizer_step_max=max(steps),
        nonzero_adam_first_moments=sum(bool(v['exp_avg'].abs().sum()>0) for v in opt['state'].values() if 'exp_avg' in v))


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    seed_all(42)
    model, meta = load_model('appearance', Path('/mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt'), OUT/'work', 'can', 'maxpool')
    model.eval()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('embedding_manager.')}
    selection = json.loads((ROOT/'appearance_training/selection.json').read_text())
    best = torch.load(selection['best_checkpoint'], map_location='cpu')
    last = torch.load(ROOT/'appearance_training/checkpoints/last.ckpt', map_location='cpu')
    record = dict(diagnostic_only=True, category='can', model_metadata=meta,
        appearance_checkpoints=dict(best=inspect_checkpoint(best,initial),last=inspect_checkpoint(last,initial)),
        fixed_loss_probe=[], generations=[], seed=62000, ddim_steps=200, ddim_eta=1., aar=False,
        caption='a photo of a *', probe_sample_ids={p:[r['sample_id'] for r in rows_for(p)] for p in ['training','validation']})
    record['fixed_loss_probe'] += probe_loss(model, 'initial')
    model.load_state_dict(best['state_dict'], strict=True)
    record['fixed_loss_probe'] += probe_loss(model, 'best')
    # A real backward pass proves signal reaches both learned components.
    model.zero_grad(set_to_none=True)
    seed_all(65000)
    batch=batch_for(rows_for('training')[:2])
    loss,_=model.shared_step(batch)
    loss.backward()
    gradients={}
    for name,prefix in [('appearance','embedding_manager.string_to_param_dict.'),('spatial','embedding_manager.spatial_encoder_model.')]:
        params=[p for k,p in model.named_parameters() if k.startswith(prefix)]
        gradients[name]=dict(parameters=len(params),with_gradient=sum(p.grad is not None for p in params),
            finite=all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None),
            grad_l2=sum(float(p.grad.float().square().sum()) for p in params if p.grad is not None)**.5)
    record['gradient_probe']=dict(loss=float(loss.detach()),groups=gradients,optimizer_step_performed=False)
    del loss,batch
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    model.load_state_dict(last['state_dict'],strict=True)
    record['fixed_loss_probe'] += probe_loss(model,'last')
    write('training_audit.json',record)
    # Use the previously verified same-control hybrid cases; native and 2x views.
    cases=json.loads((INPUT/'results.json').read_text())['cases']
    for i,c in enumerate(cases):
        for key in ['healthy','hybrid','mask']:
            im=Image.open(c[key]);im.save(OUT/f'{i:02}_{key}_native.png')
            center=im.crop((64,64,192,192))
            if key=='mask' and np.count_nonzero(np.asarray(center)) != c['cropped_mask_pixels']:
                raise ValueError('2x crop must retain the entire final native mask.')
            center.resize((256,256),Image.Resampling.NEAREST if key=='mask' else Image.Resampling.BICUBIC).save(OUT/f'{i:02}_{key}_2x.png')
    record['cases']=cases
    variants=[('best_eval_native',best,False,'native'),('best_reference_bn_native',best,True,'native'),
        ('last_eval_native',last,False,'native'),('best_eval_2x',best,False,'2x'),('best_reference_bn_2x',best,True,'2x')]
    for name,state,bn_train,scale in variants:
        model.load_state_dict(state['state_dict'],strict=True)
        model.eval()
        model.embedding_manager.spatial_encoder_model.train(bn_train)
        seed_all(62000)
        batch=dict(image=torch.from_numpy(np.stack([image_array(OUT/f'{i:02}_healthy_{scale}.png') for i in range(4)])).cuda(),
            mask=torch.from_numpy(np.stack([np.asarray(Image.open(OUT/f'{i:02}_mask_{scale}.png').convert('L'))>0 for i in range(4)]).astype('float32')).cuda(),
            caption=['a photo of a *']*4,name=['can+bad']*4)
        started=time.monotonic()
        with torch.no_grad():
            logs=model.log_images(batch,N=4,sample=False,inpaint=True,unconditional_only=True,ddim_steps=200,ddim_eta=1.,adaptive_mask=False)
        paths=[]
        for i in range(4):
            for key,suffix in [('samples_inpainting','image'),('reconstruction','reconstruction')]:
                value=logs[key][i].detach().cpu().permute(1,2,0).numpy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite output.')
                path=OUT/f'{i:02}_{name}_{suffix}.png'
                Image.fromarray(np.clip((value+1)*127.5,0,255).round().astype('uint8')).save(path)
            paths.append(str(OUT/f'{i:02}_{name}_image.png'))
        record['generations'].append(dict(name=name,spatial_batchnorm_training=bn_train,scale=scale,
            checkpoint_step=state['hsg_actual_step'],batch_size=4,seed=62000,seconds=time.monotonic()-started,images=paths))
        write('progress.json',dict(completed_variants=len(record['generations']),total_variants=len(variants)))
        write('results_partial.json',record)
        print(name,'complete',flush=True)
        del logs,batch;gc.collect();torch.cuda.empty_cache()
    write('results.json',record)
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
