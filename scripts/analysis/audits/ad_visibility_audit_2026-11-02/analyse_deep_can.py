"""Read-only analysis of completed can scale/training/inference probes."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=report_dir('ad_can_deep_diagnosis_2026-11-03')


def rgb(path):return Image.open(path).convert('RGB')


def common(path,scale):
    im=rgb(path)
    return np.asarray(im.crop((64,64,192,192)) if scale=='native' else im.resize((128,128),Image.Resampling.BICUBIC),dtype=float)


def main():
    a=json.loads((ROOT/'results.json').read_text());rows=[];medians={}
    names=[v['name'] for v in a['generations']]
    for v in a['generations']:
        for i in range(4):
            image=common(ROOT/f'{i:02}_{v["name"]}_image.png',v['scale'])
            recon=common(ROOT/f'{i:02}_{v["name"]}_reconstruction.png',v['scale'])
            target=np.asarray(rgb(ROOT/f'{i:02}_healthy_native.png').crop((64,64,192,192)),dtype=float)
            mask=np.asarray(Image.open(ROOT/f'{i:02}_mask_native.png').convert('L').crop((64,64,192,192)))>0
            diff=np.abs(image-target).mean(2);res=np.abs(image-recon).mean(2)
            rows.append(dict(variant=v['name'],index=i,native_mask_pixels=int(mask.sum()),
                inside_mae=float(diff[mask].mean()),outside_mae=float(diff[~mask].mean()),
                residual_inside_mae=float(res[mask].mean()),residual_outside_mae=float(res[~mask].mean())))
        subset=[r for r in rows if r['variant']==v['name']]
        medians[v['name']]={k:float(np.median([r[k] for r in subset])) for k in ['inside_mae','outside_mae','residual_inside_mae','residual_outside_mae']}
    fixed={}
    for state in ['initial','best','last']:
        fixed[state]={part:float(np.mean([x['occupancy_normalized_loss'] for x in a['fixed_loss_probe'] if x['state']==state and x['partition']==part])) for part in ['training','validation']}
    (ROOT/'analysis.json').write_text(json.dumps(dict(common_coordinates='same central native 128x128; 2x decoded views downsampled to native footprint',
        units='RGB MAE, 0..255; not defect quality',rows=rows,medians=medians,fixed_probe_mean_losses=fixed),indent=2))
    labels=['Kontrollbild','Hybrid','Maske','Best · eval','Best · Referenz-BN','Last · eval','Best · 2×','Best · Ref.-BN · 2×']
    for page in range(2):
        fig,axes=plt.subplots(2,8,figsize=(20,6),layout='constrained')
        for k,i in enumerate(range(page*2,page*2+2)):
            ims=[np.asarray(rgb(ROOT/f'{i:02}_{key}_native.png').crop((64,64,192,192))) for key in ['healthy','hybrid','mask']]
            ims += [common(ROOT/f'{i:02}_{v["name"]}_image.png',v['scale']).astype('uint8') for v in a['generations']]
            for j,im in enumerate(ims):
                axes[k,j].imshow(im,interpolation='nearest');axes[k,j].set_xticks([]);axes[k,j].set_yticks([])
                if k==0:axes[k,j].set_title(labels[j],fontsize=10)
            axes[k,0].set_ylabel(f'Fall {i}')
        fig.suptitle('can · gleiche native 128×128-Region · AAR aus · Best 70k / Last 100k · 2× ohne Neutraining')
        fig.savefig(ROOT/f'comparison_{page+1}.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for part,label in [('training','Trainingsprobe'),('validation','Validierungsprobe')]:
        ax.plot(range(3),[fixed[s][part] for s in ['initial','best','last']],'o-',label=label)
    ax.set_xticks(range(3),['Initialisierung','Best 70.000','Last 100.000'])
    ax.set_ylabel('Auf Maskenbelegung normierter Denoising-MSE')
    ax.set_title('Vier Aufnahmen je Partition × vier feste Rauschzeitpunkte')
    ax.legend();fig.savefig(ROOT/'fixed_loss_probe.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(medians=medians,fixed_probe_mean_losses=fixed),indent=2))


if __name__=='__main__':main()
