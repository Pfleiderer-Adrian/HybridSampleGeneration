"""Serial per-GPU continuation: native V3 pairs from 2x AD, then Real+AD training.

The existing appearance queue is left untouched. This queue acquires its lock
before any CUDA work, and holds it until its own work has finished.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import traceback

import numpy as np
from PIL import Image

from .prepare import sha256
from .scale_pilot import centered, write
from .scale2x_study import BASE, CATEGORIES, LEGACY, PROJECT, centers

VARIANT = 'real_anomalydiffusion'
COUNT = 1000


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def windows(mask):
    """Cover every native foreground pixel with positive, overlapping 128 windows."""
    from scipy.ndimage import label, find_objects
    labels, _ = label(mask)
    coords = set()
    for component, region in enumerate(find_objects(labels), 1):
        for cy in centers(region[0].start, region[0].stop):
            for cx in centers(region[1].start, region[1].stop):
                x, y = min(128, max(0, cx-64)), min(128, max(0, cy-64))
                if (labels[y:y+128, x:x+128] == component).any():
                    coords.add((x, y))
    cover = np.zeros((256, 256), bool)
    for x, y in coords:
        cover[y:y+128, x:x+128] = True
    if not mask.any() or np.any(mask & ~cover):
        raise ValueError('Empty mask or incomplete native foreground coverage.')
    return sorted(coords)


def stitch(healthy, generated):
    """Inverse-scale whole decoded windows; average overlap, retain outer context.

    No mask-based healthy-pixel replacement: VAE/background changes inside each
    generated window remain part of the AD output.
    """
    total = np.zeros((256, 256, 3), np.float64)
    weight = np.zeros((256, 256), np.uint16)
    for x, y, value in generated:
        native = np.asarray(Image.fromarray(value).resize((128, 128), Image.Resampling.BICUBIC))
        total[y:y+128, x:x+128] += native
        weight[y:y+128, x:x+128] += 1
    out = healthy.copy()
    active = weight > 0
    out[active] = np.clip(np.rint(total[active] / weight[active, None]), 0, 255).astype('uint8')
    return out


def sources(inputs):
    protocol = read(inputs/'protocol.json')
    study = Path(protocol['hybrid_study'])
    split = read(inputs/'split_manifest.json')
    train = {r['sample_id']: r for r in split['partitions']['train']}
    connection = sqlite3.connect(f'file:{study}/artifacts.sqlite?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    originals = {r['id']: dict(r) for r in connection.execute('SELECT * FROM original_samples')}
    hybrids = {r['id']: dict(r) for r in connection.execute('SELECT * FROM hybrid_samples')}
    connection.close()
    frozen = read(inputs/'hybrid_provenance.json')
    if not frozen:
        raise ValueError('Empty frozen hybrid pool.')
    for source in frozen:
        h, o = hybrids[source['hybrid_id']], originals[source['original_id']]
        if h['original_sample_id'] != o['id']:
            raise ValueError('Hybrid/control provenance mismatch.')
        if o['source_name'] not in train or train[o['source_name']]['label'] != 'good' or o['has_anomaly']:
            raise ValueError('Healthy control outside training split.')
        for donor in source['donor_original_ids']:
            if originals[donor]['source_name'] not in train:
                raise ValueError('Hybrid mask donor outside training split.')
    return study, split, frozen, originals, hybrids


def control_and_mask(study, source, originals, hybrids, rng):
    from scipy.ndimage import label, find_objects
    h, o = hybrids[source['hybrid_id']], originals[source['original_id']]
    mask = np.any(np.load(study/h['segmentation_path'], allow_pickle=False)>0, axis=0)
    labels, n = label(mask)
    if not n:
        raise ValueError('Empty frozen hybrid mask.')
    component = int(rng.integers(1, n+1))
    region = find_objects(labels)[component-1]
    cy, cx = (region[0].start+region[0].stop)//2, (region[1].start+region[1].stop)//2
    # Keep a positive center even for hollow/non-convex large components.
    if not (labels[max(0,cy-128):cy+128, max(0,cx-128):cx+128]==component).any():
        ys, xs = np.nonzero(labels==component)
        j = int(np.argmin((ys-cy)**2+(xs-cx)**2)); cy, cx = int(ys[j]), int(xs[j])
    native_mask, crop = centered(mask, cx, cy, 256)
    image = np.load(study/o['image_path'], allow_pickle=False)
    image = np.clip(image.transpose(1,2,0),0,255).round().astype('uint8')
    if image.shape[2]==1:
        image=np.repeat(image,3,axis=2)
    if image.shape[2]!=3:
        raise ValueError('Expected grayscale or RGB healthy control.')
    healthy, _ = centered(image, cx, cy, 256)
    return healthy, native_mask, dict(recipient_sample_id=o['source_name'], crop_xywh=crop,
        hybrid_id=source['hybrid_id'], donor_original_ids=source['donor_original_ids'], selected_component=component)


def prepare(study_root, output):
    output.mkdir(parents=True, exist_ok=False)
    old = read(LEGACY/'launches/20261103T143221Z_gpu1/plan.json')
    plan = dict(version=1, profile='v3_real_ad_scale2x50k_last_aar', gpus=[1],
        appearance_root=str(study_root), source_snapshot=str(output/'source_snapshot'),
        downstream_root=str(output/'downstream'), downstream_python=old['downstream_python'],
        downstream_runtime_libs=old['downstream_runtime_libs'], downstream_hf_cache=old['downstream_hf_cache'],
        base_checkpoint=str(BASE), base_checkpoint_sha256=read(study_root/'plan.json')['base_checkpoint_sha256'],
        variant=VARIANT, count=COUNT, seed=42, generation_seed=62000,
        generation=dict(checkpoint='last_50000', aar=True, scale=2, ddim_steps=200,
            batch_size=4, spatial_encoder_mode='eval',
            mask_source='final frozen V3 hybrid masks with identical healthy controls',
            native_geometry='256 native crop; component-centered 128 windows, 64 stride for large components; joint 2x; inverse RGB bicubic; overlap mean; native mask unchanged',
            outside_windows='healthy context; no mask-based compositing inside generated windows'), jobs=[])
    for cat in CATEGORIES:
        dest=output/'inputs'/cat; dest.mkdir(parents=True)
        for name in ['split_manifest.json','hybrid_provenance.json']:
            shutil.copy2(LEGACY/cat/name, dest/name)
        protocol=read(LEGACY/cat/'protocol.json')
        protocol.update(appearance_steps=50000, appearance_checkpoint_selection='last',
            mask_steps=None, adaptive_attention=True, scale=2,
            generation_protocol=plan['generation'], production_count=COUNT)
        write(dest/'protocol.json',protocol)
        config=read(LEGACY/cat/f'{VARIANT}.json')
        config['data']['anomalydiffusion_manifest']=str(dest/'generated/manifest.json')
        write(dest/f'{VARIANT}.json',config)
        if read(study_root/cat/'centered_2x/inputs/split_manifest.json') != read(dest/'split_manifest.json'):
            raise ValueError('Appearance/downstream split mismatch.')
        study,split,frozen,originals,hybrids=sources(dest)
        rng=np.random.default_rng(42); tested=[]
        for source in frozen[:4]:
            healthy,mask,meta=control_and_mask(study,source,originals,hybrids,rng)
            coords=windows(mask)
            # Round-trip binary mask must be exact, including one-pixel islands.
            for x,y in coords:
                small=Image.fromarray(mask[y:y+128,x:x+128].astype('uint8')*255)
                back=small.resize((256,256),Image.Resampling.NEAREST).resize((128,128),Image.Resampling.NEAREST)
                if not np.array_equal(np.asarray(small),np.asarray(back)):
                    raise ValueError('Mask support changed on inverse scale.')
            tested.append(dict(**meta,windows=len(coords),native_mask_pixels=int(mask.sum())))
        write(dest/'preflight.json',dict(train_only_controls_and_donors=True, frozen_hybrids=len(frozen),
            geometry_examples=tested, split_fingerprint=split['fingerprint']))
        hashes={str(p):sha256(p) for p in dest.glob('*.json')}
        # Check config equality using the same frozen downstream class later used by training.
        plan['jobs'].append(dict(category=cat,inputs=str(dest),input_sha256=hashes))
        print('prepared',cat,'frozen masks',len(frozen),flush=True)
    snapshot=output/'source_snapshot'
    shutil.copytree(old['source_snapshot'],snapshot,symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__'))
    ad=snapshot/'experiments/anomalydiffusion'
    # New generation/queue logic and the exact current appearance runtime.
    for name in ['scale2x_downstream.py','scale2x_study.py','scale_pilot.py','runtime.py','telemetry.py']:
        shutil.copy2(Path(__file__).parent/name,ad/name)
    write(output/'snapshot_sha256.json',{str(p.relative_to(snapshot)):sha256(p)
        for p in snapshot.rglob('*.py') if 'upstream' not in p.parts and '.tools' not in p.parts})
    write(output/'plan.json',plan)
    write(output/'status.json',dict(status='prepared',categories=CATEGORIES,gpus=[1]))


def generate(output, cat):
    import torch
    from .runtime import load_model
    from .run import seed_all
    plan=read(output/'plan.json'); inputs=output/'inputs'/cat
    dest=inputs/'generated'; dest.mkdir(exist_ok=False); (dest/'pairs').mkdir()
    selection=read(Path(plan['appearance_root'])/cat/'centered_2x/training/selection.json')
    if selection['selection']!='last' or selection['training_steps']!=50000 or sha256(selection['checkpoint'])!=selection['checkpoint_sha256']:
        raise ValueError('Expected verified last50k checkpoint.')
    if sha256(BASE)!=plan['base_checkpoint_sha256']:
        raise ValueError('Frozen base changed.')
    seed_all(42); model,_=load_model('appearance',BASE,dest/'work',cat,'maxpool')
    state=torch.load(selection['checkpoint'],map_location='cpu')
    expected={k for k in model.state_dict() if k.startswith('embedding_manager.')}
    if state['step']!=50000 or set(state['learned_state_dict'])!=expected:
        raise ValueError('Incomplete/wrong learned appearance state.')
    model.load_state_dict(state['learned_state_dict'],strict=False); model.eval(); del state
    study,split,frozen,originals,hybrids=sources(inputs)
    seed_all(plan['generation_seed']); rng=np.random.default_rng(plan['seed'])
    # Repeated shuffled cycles, instead of favoring a handful of masks.
    order=[]
    while len(order)<plan['count']:
        order.extend(rng.permutation(len(frozen)).tolist())
    rows=[]; started=time.monotonic()
    for i,index in enumerate(order[:plan['count']]):
        healthy,mask,meta=control_and_mask(study,frozen[index],originals,hybrids,rng)
        coords=windows(mask); generated=[]
        for start in range(0,len(coords),4):
            batchcoords=coords[start:start+4]; real_count=len(batchcoords)
            if real_count==1:batchcoords=batchcoords*2  # upstream squeeze guard
            images=[]; masks=[]
            for x,y in batchcoords:
                images.append(np.asarray(Image.fromarray(healthy[y:y+128,x:x+128]).resize((256,256),Image.Resampling.BICUBIC)).astype('float32')/127.5-1)
                masks.append(np.asarray(Image.fromarray(mask[y:y+128,x:x+128].astype('uint8')*255).resize((256,256),Image.Resampling.NEAREST))>0)
            batch=dict(image=torch.from_numpy(np.stack(images)).cuda(),mask=torch.from_numpy(np.stack(masks).astype('float32')).cuda(),
                caption=['a photo of a *']*len(batchcoords),name=[f'{cat}+bad']*len(batchcoords))
            with torch.no_grad():
                logs=model.log_images(batch,N=len(batchcoords),sample=False,inpaint=True,unconditional_only=True,
                    ddim_steps=200,ddim_eta=1.,adaptive_mask=True)
            for j,(x,y) in enumerate(batchcoords[:real_count]):
                value=logs['samples_inpainting'][j].detach().cpu().permute(1,2,0).numpy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite AD output.')
                generated.append((x,y,np.clip((value+1)*127.5,0,255).round().astype('uint8')))
        image=stitch(healthy,generated); prefix=dest/'pairs'/f'{i:07d}'
        paths={k:str(prefix)+f'_{k}.png' for k in ['image','target','mask']}
        Image.fromarray(image).save(paths['image']); Image.fromarray(healthy).save(paths['target'])
        Image.fromarray(mask.astype('uint8')*255).save(paths['mask'])
        rows.append(dict(image_path=paths['image'],target_path=paths['target'],mask_path=paths['mask'],**meta,
            mask_source='frozen_v3_hybrid', native_windows_xy=coords,native_mask_pixels=int(mask.sum()),
            file_sha256={k:sha256(p) for k,p in paths.items()}))
        write(dest/'progress.json',dict(completed_pairs=len(rows),total_pairs=plan['count'],elapsed_seconds=time.monotonic()-started))
    write(dest/'manifest.json',dict(version=1,source='anomalydiffusion',split_fingerprint=split['fingerprint'],
        rows=rows,training_sample_ids=sorted(r['sample_id'] for r in split['partitions']['train']),
        appearance_training_sample_ids=sorted({r['sample_id'] for r in read(Path(plan['appearance_root'])/cat/'centered_2x/inputs/training_patches.json')['rows']}),
        appearance_checkpoint=selection,generation=plan['generation'],smoke_test=False,
        hybrid_provenance_sha256=sha256(inputs/'hybrid_provenance.json'),seed=plan['generation_seed'],
        elapsed_seconds=time.monotonic()-started))


def validate_config(inputs):
    from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
    protocol=read(inputs/'protocol.json')
    actual=DownstreamConfiguration.load(inputs/f'{VARIANT}.json').to_dict()
    expected=DownstreamConfiguration.load(Path(protocol['reference_run'])/'configuration.json').to_dict()
    if actual['data']['source_quotas']!={'normal':500,'real':100,'anomalydiffusion':400}:
        raise ValueError('Wrong V3 mixture.')
    for key in ['source_quotas','source_quota_rotation','hybrid_fraction','anomalydiffusion_manifest']:
        actual['data'].pop(key,None);expected['data'].pop(key,None)
    if actual!=expected:raise ValueError('Downstream settings differ from frozen V3 reference.')


def recover_preview(output, cat):
    from .scale2x_study import preview, plot_preview
    appearance=Path(read(output/'plan.json')['appearance_root'])
    dest=appearance/cat/'preview_last_aar'
    if (dest/'results.json').exists():return
    if dest.exists():
        dest.rename(dest.with_name('preview_failed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    preview(appearance,cat)
    plot_preview(appearance,cat)
    write(output/'jobs'/cat/'preview_recovery.json',dict(status='complete',finished_utc=now(),
        reason='Repeat grayscale channel into RGB for PNG export; trained weights unchanged',
        results=str(dest/'results.json')))


def queue(output):
    plan=read(output/'plan.json'); appearance=Path(plan['appearance_root'])
    gpu,=plan['gpus']
    categories=[job['category'] for job in plan['jobs']]
    wait_for_appearance=plan.get('wait_for_appearance_queue',True)
    if not wait_for_appearance and gpu in read(appearance/'plan.json')['gpus']:
        raise ValueError('Concurrent continuation must use a GPU outside the appearance queue.')
    state=dict(pid=os.getpid(),status='running',stage='waiting_for_appearance_queue',
        started_utc=now(),gpus=[gpu],categories=categories)
    write(output/'status.json',state)
    # Nonblocking checks allow a heartbeat while the existing queue owns GPU1.
    gate=appearance/'queue.lock' if wait_for_appearance else output/'appearance_gate.lock'
    with gate.open('a') as predecessor:
        while True:
            try:
                fcntl.flock(predecessor,fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError:
                state['updated_utc']=now();write(output/'status.json',state);time.sleep(30)
        parent=read(appearance/'status.json')
        if wait_for_appearance and parent['status'] not in ['complete','failed']:
            raise RuntimeError('Appearance queue ended without terminal status; refusing overlapping/partial jobs.')
        telemetry=output/'telemetry';telemetry.mkdir(exist_ok=True)
        monitor_env=dict(os.environ, PYTHONPATH=plan['source_snapshot'])
        with (telemetry/'monitor.log').open('w') as log:
            monitor=subprocess.Popen([plan['downstream_python'],'-m','experiments.anomalydiffusion.telemetry',
                '--launch',str(output),'--interval','5'],cwd=plan['source_snapshot'],env=monitor_env,
                stdout=log,stderr=subprocess.STDOUT)
        state['telemetry_pid']=monitor.pid
        write(output/'status.json',state)
        # This lock remains held for the whole continuation, preventing a duplicate
        # appearance launch from racing with downstream jobs on the same study.
        failed=[]
        for job in plan['jobs']:
            cat=job['category'];folder=output/'jobs'/cat;folder.mkdir(parents=True,exist_ok=True)
            jobstate=dict(category=cat,gpu=gpu,pid=os.getpid(),status='running',stages=[])
            try:
                state.update(stage='processing',category=cat);write(output/'status.json',state)
                predecessor_job=read(appearance/'jobs'/cat/'status.json')
                training_ok=any(s['stage']=='appearance_training' and s['status']=='complete'
                    for s in predecessor_job.get('stages',[]))
                if not training_ok or read(appearance/cat/'centered_2x/training/completed.json')['steps']!=50000:
                    raise RuntimeError('Appearance training incomplete.')
                for p,digest in job['input_sha256'].items():
                    if sha256(p)!=digest:raise ValueError(f'Frozen input changed: {p}')
                inputs=Path(job['inputs']);snapshot=Path(plan['source_snapshot'])
                commands=[('generation',[str(PROJECT/'experiments/anomalydiffusion/.venv/bin/python'),'-m',
                    'experiments.anomalydiffusion.scale2x_downstream','generate','--output',str(output),'--category',cat]),
                    (VARIANT,[plan['downstream_python'],'-m','experiments.anomalydiffusion.downstream',
                    '--inputs',str(inputs),'--variant',VARIANT,'--output',str(Path(plan['downstream_root'])/VARIANT/cat)])]
                if not (appearance/cat/'preview_last_aar/results.json').is_file():
                    commands.insert(0,('preview_recovery',[str(PROJECT/'experiments/anomalydiffusion/.venv/bin/python'),
                        '-m','experiments.anomalydiffusion.scale2x_downstream','recover-preview',
                        '--output',str(output),'--category',cat]))
                for stage,cmd in commands:
                    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',
                        MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',
                        LD_LIBRARY_PATH=str(PROJECT/'experiments/anomalydiffusion/.tools/cuda')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''))
                    env['PYTHONPATH']=str(snapshot)
                    if stage==VARIANT:
                        env.update(PYTHONPATH=plan['downstream_runtime_libs']+os.pathsep+str(snapshot),
                            HF_HUB_CACHE=plan['downstream_hf_cache'],HF_HUB_OFFLINE='1')
                    entry=dict(stage=stage,status='running',started_utc=now(),command=cmd,log=str(folder/f'{stage}.log'))
                    jobstate['stages'].append(entry);jobstate['stage']=stage
                    with Path(entry['log']).open('w') as log:
                        child=subprocess.Popen(cmd,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT)
                        entry['pid']=child.pid;write(folder/'status.json',jobstate);rc=child.wait()
                    entry.update(status='complete' if rc==0 else 'failed',returncode=rc,finished_utc=now())
                    write(folder/'status.json',jobstate)
                    if rc:raise RuntimeError(f'{stage} exited with {rc}')
                    if stage=='generation':
                        if len(read(inputs/'generated/manifest.json')['rows'])!=COUNT:raise ValueError('Incomplete production pool.')
                    elif stage==VARIANT:
                        runs=list((Path(plan['downstream_root'])/VARIANT/cat/'downstream/dinov3_paired').glob('*'))
                        if len(runs)!=1 or not (runs[0]/'metrics.json').is_file() or not (runs[0]/'checkpoints/best.pt').is_file():
                            raise ValueError('Downstream completion artifacts missing.')
                        entry['downstream_run']=str(runs[0])
                jobstate.update(status='complete',finished_utc=now())
            except Exception as error:
                jobstate.update(status='failed',error=str(error),traceback=traceback.format_exc());failed.append(cat)
            write(folder/'status.json',jobstate)
        state.update(status='failed' if failed else 'complete',stage='finished',failed_categories=failed,finished_utc=now())
        write(output/'status.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','generate','recover-preview','validate','queue'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--appearance-root',type=Path)
    p.add_argument('--category',choices=CATEGORIES)
    args=p.parse_args();output=args.output.resolve()
    if Path('/mnt/results') not in output.parents:raise ValueError('Results must stay under /mnt/results.')
    if args.action=='prepare':prepare(args.appearance_root.resolve(),output)
    elif args.action=='generate':generate(output,args.category)
    elif args.action=='recover-preview':recover_preview(output,args.category)
    elif args.action=='validate':
        for cat in CATEGORIES:validate_config(output/'inputs'/cat)
        print('All eight downstream configurations match the frozen V3 reference.')
    else:
        with (output/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:queue(output)
            except Exception as error:
                state=read(output/'status.json');state.update(status='failed',error=str(error),traceback=traceback.format_exc())
                write(output/'status.json',state);raise


if __name__=='__main__':main()
