"""Regenerate the pilot's best/10k comparison with reference AAR enabled."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

ROOT=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_scale_pilot/can_20261103')
PROJECT=Path('/workspaces/hybrid-sample-generator')


def generate():
    import torch
    from .run import seed_all,image_array
    from .runtime import load_model
    from .prepare import sha256
    torch.set_num_threads(4)
    folder=ROOT/'evaluation_aar_on';folder.mkdir(exist_ok=False)
    original=json.loads((ROOT/'evaluation/results.json').read_text())
    plan=json.loads((ROOT/'plan.json').read_text())
    seed_all(42)
    base=Path(plan['base_checkpoint'])
    if sha256(base)!=plan['base_checkpoint_sha256']:raise ValueError('Wrong frozen base weights.')
    model,_=load_model('appearance',base,folder/'work','can','maxpool')
    expected={k for k in model.state_dict() if k.startswith('embedding_manager.')}
    record=dict(aar=True,diagnostic_only=True,seed=62000,batch_size=4,ddim_steps=200,ddim_eta=1.,
        spatial_encoder_mode='eval',latent_mask_policy='maxpool',caption='a photo of a *',
        base_checkpoint_sha256=plan['base_checkpoint_sha256'],cases=original['cases'],rows=[])
    for arm in ['centered_native','centered_2x']:
        scale='2x' if arm=='centered_2x' else 'native'
        for tag in ['best','step_010000']:
            checkpoint=ROOT/arm/'training/checkpoints'/f'{tag}.pt'
            state=torch.load(checkpoint,map_location='cpu')
            learned=state['learned_state_dict']
            if set(learned)!=expected:raise ValueError('Incomplete learned checkpoint.')
            incompatible=model.load_state_dict(learned,strict=False)
            if incompatible.unexpected_keys:raise ValueError('Unexpected checkpoint weights.')
            model.eval();seed_all(62000)
            paths=[ROOT/'evaluation'/f'{i:02}_healthy_{scale}.png' for i in range(4)]
            masks=[ROOT/'evaluation'/f'{i:02}_mask_{scale}.png' for i in range(4)]
            batch=dict(image=torch.from_numpy(np.stack([image_array(p) for p in paths])).cuda(),
                mask=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('L'))>0 for p in masks]).astype('float32')).cuda(),
                caption=['a photo of a *']*4,name=['can+bad']*4)
            started=time.monotonic()
            with torch.no_grad():
                logs=model.log_images(batch,N=4,sample=False,inpaint=True,unconditional_only=True,ddim_steps=200,ddim_eta=1.,adaptive_mask=True)
            for i in range(4):
                for key,suffix in [('samples_inpainting','image'),('reconstruction','reconstruction')]:
                    value=logs[key][i].detach().cpu().permute(1,2,0).numpy()
                    if not np.isfinite(value).all():raise ValueError('Nonfinite generated output.')
                    path=folder/f'{arm}_{tag}_{i:02}_{suffix}.png'
                    Image.fromarray(np.clip((value+1)*127.5,0,255).round().astype('uint8')).save(path)
                record['rows'].append(dict(arm=arm,tag=tag,step_counter=state['step'],index=i,scale=scale,
                    image=str(folder/f'{arm}_{tag}_{i:02}_image.png'),reconstruction=str(folder/f'{arm}_{tag}_{i:02}_reconstruction.png'),
                    checkpoint=str(checkpoint),healthy=str(paths[i]),mask=str(masks[i]),
                    healthy_sha256=sha256(paths[i]),mask_sha256=sha256(masks[i])))
            (folder/'progress.json').write_text(json.dumps(dict(completed_images=len(record['rows']),expected_images=16)))
            print(arm,tag,'complete','seconds',time.monotonic()-started,flush=True)
            del state,learned,batch,logs
    (folder/'results.json').write_text(json.dumps(record,indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'evaluation_aar_on';record=json.loads((folder/'results.json').read_text())
    target=Path(os.environ.get('HSG_RESULTS_ROOT','/mnt/results/mvtec2'))/'reports/ad_can_scale_pilot_2026-11-03';target.mkdir(exist_ok=True)
    def common(path,scale):
        im=Image.open(path).convert('RGB')
        return np.asarray(im.crop((64,64,192,192)) if scale=='native' else im.resize((128,128),Image.Resampling.BICUBIC))
    labels=['Gesund','Hybrid','Maske','Nativ · Best','Nativ · 10k','2× · Best','2× · 10k']
    metrics=[]
    for row in record['rows']:
        image=common(row['image'],row['scale']).astype(float)
        healthy=common(row['healthy'],row['scale']).astype(float)
        recon=common(row['reconstruction'],row['scale']).astype(float)
        mask=np.asarray(Image.open(ROOT/'evaluation'/f'{row["index"]:02}_mask_native.png').convert('L').crop((64,64,192,192)))>0
        diff=abs(image-healthy).mean(2);res=abs(image-recon).mean(2)
        metrics.append(dict(arm=row['arm'],tag=row['tag'],index=row['index'],inside_mae=float(diff[mask].mean()),
            outside_mae=float(diff[~mask].mean()),residual_inside_mae=float(res[mask].mean()),residual_outside_mae=float(res[~mask].mean())))
    (folder/'metrics.json').write_text(json.dumps(dict(units='RGB MAE0..255, no defect quality score',rows=metrics),indent=2))
    for page in range(2):
        fig,axes=plt.subplots(2,7,figsize=(18,6),layout='constrained')
        for k,i in enumerate(range(page*2,page*2+2)):
            images=[common(ROOT/'evaluation'/f'{i:02}_{key}_native.png','native') for key in ['healthy','hybrid','mask']]
            for arm in ['centered_native','centered_2x']:
                scale='2x' if arm=='centered_2x' else 'native'
                images += [common(folder/f'{arm}_{tag}_{i:02}_image.png',scale) for tag in ['best','step_010000']]
            for j,im in enumerate(images):
                axes[k,j].imshow(im,interpolation='nearest');axes[k,j].set_xticks([]);axes[k,j].set_yticks([])
                if k==0:axes[k,j].set_title(labels[j],fontsize=11)
            axes[k,0].set_ylabel(f'Fall {i}')
        fig.suptitle('can · gleiche Checkpoints, Masken und Seeds · gleiche native Region · AAR AN · räumlicher Encoder eval')
        path=folder/f'best_and_last_aar_on_{page+1}.png';fig.savefig(path,dpi=150);plt.close(fig)
        shutil.copy2(path,target/path.name)
        print(target/path.name)
    (folder/'README.md').write_text('# AAR an: Best-/10k-Bildvergleich\n\n16 neue Diagnosebilder, ohne Neutraining. Gleiche Checkpoints und Eingaben wie AAR aus; Seed62000, Batch4, DDIM200, eta1, räumlicher Encoder eval. Nur adaptive_mask wurde aktiviert. Alle Einzelausgaben im Results-Ordner; beide Vergleichstafeln zusätzlich unter '+str(target)+'.\n\nDie Tabellen zeigen dieselbe native128×128-Region. Positive Maske bleibt die ursprüngliche gemeinsame Maske; AAR kann die tatsächlich bearbeitete Latentregion erweitern. Stärkere Änderungen sind kein automatischer Qualitätsnachweis. Keine Änderung der Hauptstudie. GPU1 kurz mitbenutzt, entsprechende Energieintervalle sind nicht exklusiv der Hauptstudie zuzurechnen.\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','plot']);a=p.parse_args()
    generate() if a.action=='generate' else plot()


if __name__=='__main__':main()
