"""Plot exact-control hybrid/AD diagnostic and record pixel diagnostics."""

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

ROOT=report_dir('ad_can_hybrid_mask_test_2026-11-02')


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'),dtype=float)


def main():
    result=json.loads((ROOT/'results.json').read_text());metrics=[]
    for case in result['cases']:
        i=case['index'];target=rgb(case['healthy']);mask=np.asarray(Image.open(case['mask']).convert('L'))>0
        far=~binary_dilation(mask,iterations=16)
        for name,path,recon in [('hybrid',case['hybrid'],None)]+[
            (f'ad_aar_{"on" if v["aar"] else "off"}',v['samples_inpainting'],v['reconstruction'])
            for v in result['variants'] if v['index']==i]:
            image=rgb(path);diff=np.abs(image-target).mean(2)
            row=dict(index=i,variant=name,mask_pixels=int(mask.sum()),inside_mae=float(diff[mask].mean()),
                outside_mae=float(diff[~mask].mean()),far_mae=float(diff[far].mean()))
            if recon:
                residual=np.abs(image-rgb(recon)).mean(2)
                row.update(residual_inside_mae=float(residual[mask].mean()),residual_outside_mae=float(residual[~mask].mean()))
            metrics.append(row)
    identical=all(np.array_equal(rgb(ROOT/f'{i:02}_reconstruction_aar_off.png'),rgb(ROOT/f'{i:02}_reconstruction_aar_on.png')) for i in range(4))
    (ROOT/'metrics.json').write_text(json.dumps(dict(units='mean absolute RGB channel difference, 0..255; not quality scores',
        reconstructions_identical_across_aar=identical,rows=metrics),indent=2))
    for page in range(2):
        fig,axes=plt.subplots(2,5,figsize=(15,6.5),layout='constrained')
        for k,i in enumerate(range(page*2,page*2+2)):
            c=result['cases'][i]
            paths=[c['healthy'],c['mask'],c['hybrid'],ROOT/f'{i:02}_ad_aar_off.png',ROOT/f'{i:02}_ad_aar_on.png']
            for j,path in enumerate(paths):
                axes[k,j].imshow(Image.open(path),cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[k,j].set_xticks([]);axes[k,j].set_yticks([])
                if k==0:axes[k,j].set_title(['Kontrollbild','Gemeinsame Maske','Vorhandener Hybrid','AD · AAR aus','AD · AAR an'][j])
            axes[k,0].set_ylabel(f'Fall {i} · {c["cropped_mask_pixels"]} Maskenpixel')
        fig.suptitle('can · gleicher gesunder Empfänger, gleiche finale Hybridmaske · native 256×256')
        fig.savefig(ROOT/f'comparison_{page+1}.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(reconstructions_identical=identical,rows=metrics),indent=2))


if __name__=='__main__':main()
