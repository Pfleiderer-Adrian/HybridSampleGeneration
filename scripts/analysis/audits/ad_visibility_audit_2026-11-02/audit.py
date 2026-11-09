"""Read-only pixel diagnostics; differences are not perceptual defect labels."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/fabric/generated')
OUT = audit_dir('ad_visibility_audit_2026-11-02')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    records = []
    for row in manifest['rows']:
        image = np.asarray(Image.open(row['image_path']).convert('RGB'), dtype=np.float32)
        target = np.asarray(Image.open(row['target_path']).convert('RGB'), dtype=np.float32)
        mask = np.asarray(Image.open(row['mask_path']).convert('L')) > 0
        diff = np.abs(image - target).mean(axis=2)
        ring = binary_dilation(mask, iterations=8) & ~mask
        far = ~binary_dilation(mask, iterations=16)
        exact = np.all(image == target, axis=2)
        latent = mask.reshape(32, 8, 32, 8).max(axis=(1, 3))
        inside = float(diff[mask].mean())
        outside = float(diff[~mask].mean()) if (~mask).any() else None
        records.append(dict(
            id=Path(row['image_path']).name.split('_')[0],
            mask_pixels=int(mask.sum()), latent_cells=int(latent.sum()),
            mask_fraction=float(mask.mean()), inside_mae=inside, outside_mae=outside,
            ring_mae=float(diff[ring].mean()) if ring.any() else None,
            far_mae=float(diff[far].mean()) if far.any() else None,
            inside_exact_fraction=float(exact[mask].mean()),
            outside_changed_fraction=float((~exact)[~mask].mean()) if (~mask).any() else None,
            inside_to_outside=inside / outside if outside else None,
        ))
    with (OUT / 'pairs.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary = {'category': ROOT.parent.name, 'pairs': len(records),
               'units': 'RGB channel absolute differences, 0..255; not perceptual quality',
               'adaptive_attention': manifest['adaptive_attention'],
               'selected_appearance_step': manifest['appearance_training']['selected_step'],
               'all_mask_pixels_exactly_unchanged': sum(x['inside_exact_fraction'] == 1 for x in records),
               'inside_mae_le_2': sum(x['inside_mae'] <= 2 for x in records),
               'inside_mae_le_outside': sum(x['inside_mae'] <= x['outside_mae'] for x in records if x['outside_mae'] is not None),
               'empty_masks': sum(x['mask_pixels'] == 0 for x in records),
               'empty_maxpool_latent_masks': sum(x['latent_cells'] == 0 for x in records),
               'quantiles': {}}
    for key in ['mask_pixels', 'latent_cells', 'inside_mae', 'outside_mae', 'ring_mae',
                'far_mae', 'outside_changed_fraction', 'inside_to_outside']:
        values = [x[key] for x in records if x[key] is not None]
        summary['quantiles'][key] = dict(zip(['p05', 'median', 'p95'], np.quantile(values, [.05, .5, .95]).tolist()))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(6, 4, figsize=(10, 14), layout='constrained')
    for i, row in enumerate(manifest['rows'][:6]):
        image = np.asarray(Image.open(row['image_path']).convert('RGB'), dtype=float)
        target = np.asarray(Image.open(row['target_path']).convert('RGB'), dtype=float)
        mask = np.asarray(Image.open(row['mask_path']).convert('L')) > 0
        diff = np.abs(image-target).mean(2)
        for j, value in enumerate([target / 255, image / 255, mask, diff]):
            axes[i,j].imshow(value, **({'cmap':'magma', 'vmin':0, 'vmax':40} if j == 3 else {'cmap':'gray', 'vmin':0, 'vmax':1} if j == 2 else {}))
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            if i == 0: axes[i,j].set_title(['Gesund', 'Generiert', 'Maske', 'RGB-Differenz 0–40'][j])
        axes[i,0].set_ylabel(str(i))
    fig.savefig(OUT / 'first_six_differences.png', dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
