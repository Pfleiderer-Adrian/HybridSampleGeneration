"""Summarize completed diagnostic outputs without changing study results."""

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
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = report_dir('ad_can_controlled_test_2026-11-02')


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=float)


def main():
    result = json.loads((ROOT / 'results.json').read_text())
    rows = []
    for v in result['variants']:
        image, target, recon = rgb(v['image']), rgb(v['healthy']), rgb(v['reconstruction'])
        mask = np.asarray(Image.open(v['mask']).convert('L')) > 0
        diff = np.abs(image-target).mean(2)
        residual = np.abs(image-recon).mean(2)
        outside = ~mask
        far = ~binary_dilation(mask, iterations=16)
        rows.append(dict(index=v['index'], mask_source=v['mask_source'], aar=v['aar'],
            mask_pixels=int(mask.sum()), inside_mae=float(diff[mask].mean()),
            outside_mae=float(diff[outside].mean()),
            reconstruction_inside_mae=float(np.abs(recon-target).mean(2)[mask].mean()),
            residual_inside_mae=float(residual[mask].mean()),
            residual_outside_mae=float(residual[outside].mean()),
            residual_far_mae=float(residual[far].mean()) if far.any() else None,
            residual_energy_outside_fraction=float(residual[outside].sum()/max(residual.sum(),1e-12))))
    summary = {}
    for source in ['learned','real']:
        for aar in [False,True]:
            subset = [r for r in rows if r['mask_source']==source and r['aar']==aar]
            summary[f'{source}_aar_{"on" if aar else "off"}'] = {
                k: float(np.median([r[k] for r in subset])) for k in rows[0] if k not in ['index','mask_source','aar']}
        for page in range(2):
            fig,axes=plt.subplots(3,4,figsize=(12,9.4),layout='constrained')
            for k,i in enumerate(range(page*3,page*3+3)):
                paths=[ROOT/f'{i:02}_healthy.png',ROOT/f'{i:02}_{source}_mask.png',
                       ROOT/f'{i:02}_{source}_aar_off.png',ROOT/f'{i:02}_{source}_aar_on.png']
                for j,path in enumerate(paths):
                    axes[k,j].imshow(Image.open(path),cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    axes[k,j].set_xticks([]);axes[k,j].set_yticks([])
                    if k==0:axes[k,j].set_title(['Gesund','Maske','AAR aus','AAR an'][j])
                axes[k,0].set_ylabel(f'Fall {i}')
            fig.suptitle('can · '+('Gelernte Masken' if source=='learned' else 'Echte Trainingsmasken')+' · gleiche Gewichte, Hintergründe und Seeds')
            fig.savefig(ROOT/f'{source}_comparison_{page+1}.png',dpi=150)
            plt.close(fig)
    # Validate the controlled stochastic inputs via the actual saved reconstruction.
    recon_identical = all(np.array_equal(rgb(ROOT/f'{i:02}_learned_aar_off_reconstruction.png'),
        rgb(ROOT/f'{i:02}_{source}_aar_{flag}_reconstruction.png'))
        for i in range(6) for source in ['learned','real'] for flag in ['off','on'])
    (ROOT/'metrics.json').write_text(json.dumps(dict(units='RGB MAE in 0..255; no perceptual quality labels',
        reconstructions_identical_across_variants=recon_identical, rows=rows, medians=summary),indent=2))
    print(json.dumps(dict(reconstructions_identical=recon_identical,medians=summary),indent=2))


if __name__=='__main__':
    main()
