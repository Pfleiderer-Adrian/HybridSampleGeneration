"""Isolated paired can scale pilot. All mutations stay below --output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

PROJECT = Path('/workspaces/hybrid-sample-generator')
REFERENCE = Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/can')
BASE = Path('/mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt')
MILESTONES = [1000, 2500, 5000, 10000]


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def centered(array, cx, cy, size):
    """Same fixed center; zero pad out-of-image context, never change the center."""
    x, y = cx-size//2, cy-size//2
    out = np.zeros((size,size)+array.shape[2:],dtype=array.dtype)
    x0,y0=max(0,x),max(0,y);x1,y1=min(array.shape[1],x+size),min(array.shape[0],y+size)
    out[y0-y:y1-y,x0-x:x1-x]=array[y0:y1,x0:x1]
    return out, [x,y,size,size]


def prepare(output):
    from scipy.ndimage import label, find_objects
    from .prepare import sha256
    output.mkdir(parents=True, exist_ok=False)
    split=json.loads((REFERENCE/'split_manifest.json').read_text())
    template=json.loads((REFERENCE/'protocol.json').read_text())
    old=json.loads((REFERENCE/'training_patches.json').read_text())
    by_arm={arm:{part:[] for part in ['train','validation']} for arm in ['centered_native','centered_2x']}
    pairs=[]
    for part in ['train','validation']:
        for source in split['partitions'][part]:
            if source['label']=='good':continue
            image=np.asarray(Image.open(source['image_path']).convert('RGB'))
            mask=np.asarray(Image.open(source['mask_path']).convert('L'))>0
            labeled,n=label(mask)
            for component,region in enumerate(find_objects(labeled),start=1):
                h,w=region[0].stop-region[0].start,region[1].stop-region[1].start
                if max(h,w)>128:raise ValueError('Pilot requires every target component to fit within 128 pixels.')
                cy=(region[0].start+region[0].stop)//2;cx=(region[1].start+region[1].stop)//2
                small_mask,small_crop=centered(mask,cx,cy,128)
                # Both arms condition on exactly the same native support. Any
                # other component outside the shared core is excluded in BOTH.
                large_mask=np.pad(small_mask,64)
                large_image,large_crop=centered(image,cx,cy,256)
                small_image,_=centered(image,cx,cy,128)
                key=f'{len(pairs):05d}'
                pair=dict(key=key,partition=part,sample_id=source['sample_id'],component=component,
                    component_bbox_yxhw=[region[0].start,region[1].start,h,w],center_xy=[cx,cy],
                    core_crop_xywh=small_crop,context_crop_xywh=large_crop,native_positive_pixels=int(small_mask.sum()),
                    source_image_sha256=sha256(source['image_path']),source_mask_sha256=sha256(source['mask_path']))
                for arm,value,condition,crop in [
                    ('centered_native',Image.fromarray(large_image),Image.fromarray(large_mask.astype('uint8')*255),large_crop),
                    ('centered_2x',Image.fromarray(small_image).resize((256,256),Image.Resampling.BICUBIC),
                     Image.fromarray(small_mask.astype('uint8')*255).resize((256,256),Image.Resampling.NEAREST),small_crop)]:
                    folder=output/arm/'inputs/patches';folder.mkdir(parents=True,exist_ok=True)
                    ip,mp=folder/f'{key}_image.png',folder/f'{key}_mask.png';value.save(ip);condition.save(mp)
                    positives=int(np.count_nonzero(np.asarray(condition)))
                    if positives!=int(small_mask.sum())*(4 if arm=='centered_2x' else 1):raise ValueError('Scale support mismatch.')
                    by_arm[arm][part].append(dict(sample_id=source['sample_id'],label='bad',image_path=str(ip),mask_path=str(mp),
                        crop_xywh=crop,pair_key=key,source_component=component,native_positive_pixels=int(small_mask.sum())))
                pairs.append(pair)
    for arm,parts in by_arm.items():
        folder=output/arm/'inputs';write(folder/'split_manifest.json',split)
        for part,name in [('train','training'),('validation','validation')]:
            write(folder/f'{name}_patches.json',dict(version=1,category='can',split_fingerprint=split['fingerprint'],
                upstream_commit=old['upstream_commit'],resolution=arm,rows=parts[part]))
        protocol=dict(template)
        protocol.update(appearance_steps=10000,experiment='isolated_scale_pilot_v1',scale_arm=arm,
            adaptations=template['adaptations']+["Centered matched native128 core, native256 context vs 2x upsampled128 context"],
            validation_patches_sha256=sha256(folder/'validation_patches.json'))
        write(folder/'protocol.json',protocol)
    plan=dict(version=1,category='can',seed=42,arms=['centered_native','centered_2x'],steps=10000,batch_size=4,
        base_learning_rate=.005,effective_learning_rate=.02,latent_mask_policy='maxpool',milestones=MILESTONES,
        split_fingerprint=split['fingerprint'],base_checkpoint=str(BASE),base_checkpoint_sha256=sha256(BASE),
        training_components=len(by_arm['centered_native']['train']),validation_components=len(by_arm['centered_native']['validation']),
        checkpoint_format='learned embedding_manager state, including spatial parameters and BN buffers; frozen base by SHA; latest also optimizer/RNG',
        mask_definition='all GT within shared centered native128 core, padded into native256 control or upsampled jointly with RGB',
        no_mask_generator=True,no_downstream_training=True,pairs=pairs)
    write(output/'plan.json',plan)
    # Freeze implementation; environments and original source are reused by link.
    snapshot=output/'source_snapshot';source=Path(__file__).resolve().parent
    dest=snapshot/'experiments/anomalydiffusion';dest.mkdir(parents=True)
    (snapshot/'experiments/__init__.py').write_text('')
    for f in source.glob('*.py'):shutil.copy2(f,dest/f.name)
    for name in ['upstream','.tools']:(dest/name).symlink_to(source/name,target_is_directory=True)
    write(output/'source_snapshot_sha256.json',{str(f.relative_to(snapshot)):sha256(f) for f in snapshot.rglob('*.py') if not f.is_symlink() and 'upstream' not in f.parts and '.tools' not in f.parts})
    print(json.dumps({k:v for k,v in plan.items() if k!='pairs'},indent=2))


def train(output, arm, *, category='can', steps=10000, milestones=None):
    import random
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from .reference_training import NativePatches,collate_patches,attach_reference_hooks,Progress,MONITOR,validate_partitions
    from .run import seed_all,read_inputs
    from .runtime import load_model
    from .prepare import sha256
    torch.set_num_threads(4)
    plan=json.loads((output/'plan.json').read_text());folder=output/arm;training=folder/'training';training.mkdir(exist_ok=False)
    root,patches,protocol=read_inputs(folder/'inputs');val=json.loads((root/'validation_patches.json').read_text())
    split=json.loads((root/'split_manifest.json').read_text());validate_partitions(patches['rows'],val['rows'],split)
    seed_all(42);pl.seed_everything(42,workers=True)
    model,meta=load_model('appearance',BASE,training/'work',category,'maxpool');model.learning_rate*=4
    selected_milestones=MILESTONES if milestones is None else milestones
    meta.update(experiment=plan.get('experiment','isolated_scale_pilot_v1'),arm=arm,stage='appearance',total_steps=steps,
        base_checkpoint_sha256=plan['base_checkpoint_sha256'],split_fingerprint=split['fingerprint'],seed=42,
        batch_size=4,learning_rate=model.learning_rate,training_patches_sha256=sha256(root/'training_patches.json'),
        validation_patches_sha256=sha256(root/'validation_patches.json'),smoke_test=False,
        checkpoint_format='compact learned state; frozen base external by hash')
    write(training/'metadata.json',meta);attach_reference_hooks(model,meta)
    tr=DataLoader(NativePatches(patches['rows'],'appearance',category,20),batch_size=4,shuffle=True,num_workers=2,collate_fn=collate_patches)
    va=DataLoader(NativePatches(val['rows'],'appearance',category),batch_size=4,shuffle=True,num_workers=2,collate_fn=collate_patches)
    class Compact(Callback):
        best=float('inf')
        def save(self,trainer,model,name,step,optimizer=False,score=None):
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('embedding_manager.')}
            payload=dict(format='hsg_scale_pilot_v1',metadata=meta,step=step,learned_state_dict=state,validation_score=score)
            if optimizer:
                payload.update(optimizer_state=trainer.optimizers[0].state_dict(),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
            path=training/'checkpoints'/name;path.parent.mkdir(exist_ok=True)
            tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
        def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
            step=trainer.global_step+1
            if step in selected_milestones:self.save(trainer,model,f'step_{step:06d}.pt',step)
            if step%1000==0:self.save(trainer,model,'latest_resume.pt',step,optimizer=True)
        def on_validation_end(self,trainer,model):
            if trainer.sanity_checking:return
            metric=trainer.callback_metrics.get(MONITOR)
            if metric is None:return
            score=float(metric)
            if np.isfinite(score) and score<self.best:
                self.best=score;self.save(trainer,model,'best.pt',trainer.global_step,score=score)
                write(training/'best.json',dict(step=trainer.global_step,score=score,monitor=MONITOR,checkpoint=str(training/'checkpoints/best.pt')))
    compact=Compact();progress=Progress(training,'appearance',steps)
    val_interval=min(1000,len(tr))
    while len(tr)%val_interval:val_interval-=1
    trainer=pl.Trainer(gpus=1,max_steps=steps,max_epochs=-1,benchmark=True,enable_checkpointing=False,
        callbacks=[progress,compact],logger=CSVLogger(str(training),name='lightning_logs'),val_check_interval=val_interval,
        num_sanity_val_steps=2,log_every_n_steps=50,enable_progress_bar=False,enable_model_summary=False)
    trainer.fit(model,train_dataloaders=tr,val_dataloaders=va)
    final=trainer.validate(model,dataloaders=va,verbose=False)[0]
    compact.save(trainer,model,'latest_resume.pt',trainer.global_step,optimizer=True)
    write(training/'completed.json',dict(steps=trainer.global_step,elapsed_seconds=time.monotonic()-progress.started,
        best=json.loads((training/'best.json').read_text()),final_validation=final))


def evaluate(output):
    import torch
    from .run import seed_all,image_array
    from .runtime import load_model
    torch.set_num_threads(4);seed_all(42)
    folder=output/'evaluation';folder.mkdir(exist_ok=False)
    # Previously verified matched hybrid pairs, all healthy TRAIN controls.
    source=Path(os.environ.get('HSG_RESULTS_ROOT','/mnt/results/mvtec2'))/'reports/ad_can_hybrid_mask_test_2026-11-02'
    cases=json.loads((source/'results.json').read_text())['cases']
    model,_=load_model('appearance',BASE,folder/'work','can','maxpool');base_keys=set(model.state_dict())
    rows=[]
    for arm in ['centered_native','centered_2x']:
        scale='2x' if arm=='centered_2x' else 'native'
        paths=[]
        for i,c in enumerate(cases):
            for key in ['healthy','hybrid','mask']:
                im=Image.open(c[key])
                if scale=='2x':im=im.crop((64,64,192,192)).resize((256,256),Image.Resampling.NEAREST if key=='mask' else Image.Resampling.BICUBIC)
                im.save(folder/f'{i:02}_{key}_{scale}.png')
        ckpts=[output/arm/'training/checkpoints'/f'step_{n:06d}.pt' for n in MILESTONES]+[output/arm/'training/checkpoints/best.pt']
        for checkpoint in ckpts:
            state=torch.load(checkpoint,map_location='cpu');learned=state['learned_state_dict']
            expected={k for k in base_keys if k.startswith('embedding_manager.')}
            if set(learned)!=expected:raise ValueError('Incomplete compact checkpoint.')
            model.load_state_dict(learned,strict=False);model.eval();seed_all(62000)
            batch=dict(image=torch.from_numpy(np.stack([image_array(folder/f'{i:02}_healthy_{scale}.png') for i in range(4)])).cuda(),
                mask=torch.from_numpy(np.stack([np.asarray(Image.open(folder/f'{i:02}_mask_{scale}.png').convert('L'))>0 for i in range(4)]).astype('float32')).cuda(),caption=['a photo of a *']*4,name=['can+bad']*4)
            with torch.no_grad():logs=model.log_images(batch,N=4,sample=False,inpaint=True,unconditional_only=True,ddim_steps=200,ddim_eta=1.,adaptive_mask=False)
            for i in range(4):
                value=logs['samples_inpainting'][i].detach().cpu().permute(1,2,0).numpy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite inference output.')
                path=folder/f'{arm}_{checkpoint.stem}_{i:02}.png';Image.fromarray(np.clip((value+1)*127.5,0,255).round().astype('uint8')).save(path)
                rows.append(dict(arm=arm,checkpoint=str(checkpoint),step=state['step'],index=i,image=str(path),scale=scale))
            write(folder/'progress.json',dict(completed_images=len(rows),expected_images=40))
            del state,learned,batch,logs
    write(folder/'results.json',dict(cases=cases,rows=rows,seed=62000,aar=False,spatial_encoder_mode='eval',diagnostic_only=True))


def queue(output,gpu):
    snapshot=output/'source_snapshot';env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(snapshot),PYTHONUNBUFFERED='1',
        LD_LIBRARY_PATH=str(PROJECT/'experiments/anomalydiffusion/.tools/cuda')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''))
    python=PROJECT/'experiments/anomalydiffusion/.venv/bin/python'
    state=dict(pid=os.getpid(),status='running',gpu=gpu,started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),stages=[])
    write(output/'status.json',state)
    for action,arm in [('train','centered_native'),('train','centered_2x'),('evaluate',None),('analyse',None)]:
        name=arm or action;cmd=[str(python),'-m','experiments.anomalydiffusion.scale_pilot',action,'--output',str(output)]
        if action=='analyse':cmd[0]=str(PROJECT/'.venv/bin/python')
        if arm:cmd+=['--arm',arm]
        entry=dict(stage=name,status='running',started_unix=time.time(),command=cmd);state['stages'].append(entry);state['stage']=name
        with (output/f'{name}.log').open('w') as log:
            child=subprocess.Popen(cmd,env=env,cwd=snapshot,stdout=log,stderr=subprocess.STDOUT)
            entry['pid']=child.pid;write(output/'status.json',state);rc=child.wait()
        entry.update(status='complete' if rc==0 else 'failed',returncode=rc,finished_unix=time.time());write(output/'status.json',state)
        if rc:
            state['status']='failed';write(output/'status.json',state);return
    state['status']='complete';state['finished_unix']=time.time();write(output/'status.json',state)


def analyse(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads((output/'plan.json').read_text())
    result=json.loads((output/'evaluation/results.json').read_text());folder=output/'evaluation'
    def common(path,scale):
        im=Image.open(path).convert('RGB')
        return np.asarray(im.crop((64,64,192,192)) if scale=='native' else im.resize((128,128),Image.Resampling.BICUBIC),dtype=float)
    metrics=[]
    for r in result['rows']:
        i=r['index'];image=common(r['image'],r['scale']);target=common(folder/f'{i:02}_healthy_native.png','native')
        mask=np.asarray(Image.open(folder/f'{i:02}_mask_native.png').convert('L').crop((64,64,192,192)))>0
        diff=np.abs(image-target).mean(2)
        metrics.append(dict(**r,inside_rgb_mae=float(diff[mask].mean()),outside_rgb_mae=float(diff[~mask].mean())))
    write(output/'evaluation/metrics.json',dict(units='RGB MAE in0..255, not defect quality',rows=metrics))
    labels=['Gesund','Hybrid','Maske']+[f'Nativ {s/1000:g}k' for s in MILESTONES]+[f'2× {s/1000:g}k' for s in MILESTONES]
    for page in range(2):
        fig,axes=plt.subplots(2,len(labels),figsize=(24,5.5),layout='constrained')
        for k,i in enumerate(range(page*2,page*2+2)):
            ims=[common(folder/f'{i:02}_{key}_native.png','native') for key in ['healthy','hybrid','mask']]
            for arm in plan['arms']:
                scale='2x' if arm=='centered_2x' else 'native'
                ims += [common(folder/f'{arm}_step_{s:06d}_{i:02}.png',scale) for s in MILESTONES]
            for j,im in enumerate(ims):
                axes[k,j].imshow(im.astype('uint8'),interpolation='nearest');axes[k,j].set_xticks([]);axes[k,j].set_yticks([])
                if k==0:axes[k,j].set_title(labels[j],fontsize=9)
            axes[k,0].set_ylabel(f'Fall {i}')
        fig.suptitle('Can-Maßstabspilot · gleiche native128×128-Region · gleiche LR0,02 · AAR aus · räumlicher Encoder eval')
        fig.savefig(folder/f'comparison_{page+1}.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    summaries={}
    for arm in plan['arms']:
        inputs=json.loads((output/arm/'inputs/validation_patches.json').read_text())['rows']
        occupancy=np.mean([np.asarray(Image.open(r['mask_path']).convert('L')).reshape(32,8,32,8).max((1,3)).astype(bool).mean() for r in inputs])
        history=[json.loads(line) for line in (output/arm/'training/validation_history.jsonl').read_text().splitlines()]
        ax.plot([r['step'] for r in history],[r['val/loss_simple_ema']/occupancy for r in history],label=arm)
        summaries[arm]=dict(validation_latent_occupancy=float(occupancy),completed=json.loads((output/arm/'training/completed.json').read_text()))
    ax.set(xlabel='Trainingsschritte',ylabel='Validierungs-MSE / mittlere Latentmaskenbelegung',title='Stochastische Generatorvalidierung; kein visueller Qualitätsnachweis')
    ax.legend();fig.savefig(folder/'validation_loss.png',dpi=150);plt.close(fig)
    write(output/'summary.json',dict(arms=summaries,generated_diagnostic_images=len(result['rows']),no_downstream_evaluation=True))
    (output/'RESULTS.md').write_text('# Can: Maßstabspilot abgeschlossen\n\nBeide Läufe erreichten10.000 Schritte. Sie unterscheiden sich im dargestellten Defektmaßstab und im notwendigen Bildkontext. Lernrate, Split, Zielmasken und Schrittbudget sind gepaart.\n\nVergleichsbilder: [Fälle0–1](evaluation/comparison_1.png), [Fälle2–3](evaluation/comparison_2.png). [Validierungskurve](evaluation/validation_loss.png), Einzelwerte in summary.json und evaluation/metrics.json.\n\nPixel-MAE und Denoising-MSE beweisen keinen höheren Realismus oder Downstream-Nutzen. Die Bilder sind im gleichen nativen128×128-Fenster dargestellt. Kein Maskengenerator oder Downstream-Modell wurde neu trainiert. Keine Änderung der Hauptstudie.\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','queue','train','evaluate','analyse']);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['centered_native','centered_2x']);p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    output=a.output.resolve()
    if PROJECT not in output.parents and Path('/mnt/results') not in output.parents:raise ValueError('Pilot output must remain in the project or results folder.')
    if a.action=='prepare':prepare(output)
    elif a.action=='queue':queue(output,a.gpu)
    elif a.action=='train':train(output,a.arm)
    elif a.action=='evaluate':evaluate(output)
    else:analyse(output)


if __name__=='__main__':main()
