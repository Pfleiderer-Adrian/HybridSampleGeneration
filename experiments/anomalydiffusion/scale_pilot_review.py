"""Post-hoc descriptive review of the completed paired can scale pilot."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_scale_pilot/can_20261103')


def common(path,scale):
    im=Image.open(path).convert('RGB')
    return np.asarray(im.crop((64,64,192,192)) if scale=='native' else im.resize((128,128),Image.Resampling.BICUBIC))


def main():
    summary=json.loads((ROOT/'summary.json').read_text())
    metrics=json.loads((ROOT/'evaluation/metrics.json').read_text())['rows']
    review={}
    for arm,data in summary['arms'].items():
        occ=data['validation_latent_occupancy'];completed=data['completed']
        rows=list(csv.DictReader(next((ROOT/arm/'training/lightning_logs').rglob('metrics.csv')).open()))
        train={}
        for label,lo,hi in [('first_1000',0,1000),('1000_to_2500',1000,2500),('5000_to_10000',5000,10001)]:
            vals=[float(x['train/loss_simple_step']) for x in rows if x.get('train/loss_simple_step') and lo<=int(float(x['step']))<hi]
            train[label]=float(np.median(vals))
        snapshots={}
        for name in ['step_001000','step_002500','step_005000','step_010000','best']:
            rs=[r for r in metrics if r['arm']==arm and Path(r['checkpoint']).stem==name]
            snapshots[name]=dict(step_counter=rs[0]['step'],inside_rgb_mae_median=float(np.median([r['inside_rgb_mae'] for r in rs])),
                outside_rgb_mae_median=float(np.median([r['outside_rgb_mae'] for r in rs])))
        review[arm]=dict(best_step_counter=completed['best']['step'],best_validation_raw=completed['best']['score'],
            best_validation_occupancy_normalized=completed['best']['score']/occ,
            final_validation_occupancy_normalized=completed['final_validation']['val/loss_simple_ema']/occ,
            elapsed_hours=completed['elapsed_seconds']/3600,training_loss_medians=train,snapshots=snapshots)
    (ROOT/'review_metrics.json').write_text(json.dumps(dict(arms=review,diagnostic_cases=4,training_seeds=1,
        normalization_warning='Mask occupancy division does not equalize scaled RGB targets, native geometry or context; no downstream metric.'),indent=2))
    folder=ROOT/'evaluation'
    labels=['Gesund','Hybrid','Maske','Nativ · Best','Nativ · 10k','2× · Best','2× · 10k']
    for page in range(2):
        fig,axes=plt.subplots(2,7,figsize=(18,6),layout='constrained')
        for k,i in enumerate(range(page*2,page*2+2)):
            images=[common(folder/f'{i:02}_{key}_native.png','native') for key in ['healthy','hybrid','mask']]
            for arm in ['centered_native','centered_2x']:
                scale='native' if arm=='centered_native' else '2x'
                images += [common(folder/f'{arm}_{tag}_{i:02}.png',scale) for tag in ['best','step_010000']]
            for j,im in enumerate(images):
                axes[k,j].imshow(im,interpolation='nearest');axes[k,j].set_xticks([]);axes[k,j].set_yticks([])
                if k==0:axes[k,j].set_title(labels[j],fontsize=11)
            axes[k,0].set_ylabel(f'Fall {i}')
        fig.suptitle('can · Auswahl nach Generatorvalidierung vs. letzter Stand · gleiche native Region · AAR aus')
        fig.savefig(folder/f'best_and_last_{page+1}.png',dpi=150);plt.close(fig)
    print(json.dumps(review,indent=2))


if __name__=='__main__':main()
