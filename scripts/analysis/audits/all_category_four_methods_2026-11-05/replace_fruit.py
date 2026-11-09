
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

from pathlib import Path
import json,sqlite3,shutil
import numpy as np
from PIL import Image
from experiments.anomalydiffusion.scale_pilot import centered
root=Path('/mnt/results/mvtec2/experiments/all_category_four_methods_20261105');d=root/'fruit_jelly'
assert (d/'ad_2x.png').exists()
archive=d/'previous_selection';archive.mkdir(exist_ok=True)
for p in list(d.glob('*.png'))+list(d.glob('*metadata.json'))+[d/'selection.json']:shutil.copy2(p,archive/p.name)
studyroot=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103');preview=studyroot/'fruit_jelly/preview_last_aar';results=json.loads((preview/'results.json').read_text());case=results['cases'][1];src=case['provenance']
study=Path(json.loads(Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/fruit_jelly/protocol.json').read_text())['hybrid_study'])
c=sqlite3.connect(f'file:{study}/artifacts.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row
tables={t:{r['id']:dict(r) for r in c.execute('SELECT * FROM '+t)} for t in ['original_samples','hybrid_samples','placements','synthetic_anomalies','real_anomalies']};c.close()
h=tables['hybrid_samples'][src['hybrid_id']];o=tables['original_samples'][src['original_id']];x,y,_,_=case['crop_xywh'];cx,cy=x+64,y+64
for key,path in [('healthy',o['image_path']),('hybrid',h['image_path'])]:
 a=np.clip(np.load(study/path).transpose(1,2,0),0,255).round().astype('uint8');patch,crop=centered(a,cx,cy,256);Image.fromarray(patch).save(d/f'{key}_native.png')
m=np.any(np.load(study/h['segmentation_path'])>0,axis=0);patch,_=centered(m,cx,cy,256);Image.fromarray(patch.astype('uint8')*255).save(d/'hybrid_mask_native.png')
p=min((tables['placements'][pid] for pid in src['placement_ids']),key=lambda p:(p['position_y']*m.shape[0]-cy)**2+(p['position_x']*m.shape[1]-cx)**2);real=tables['real_anomalies'][tables['synthetic_anomalies'][p['synthetic_anomaly_id']]['real_anomaly_id']];donor=tables['original_samples'][real['original_sample_id']];meta=json.loads(real['metadata_json']);dy,dx=meta['centroid_voxel']
a=np.clip(np.load(study/donor['image_path']).transpose(1,2,0),0,255).round().astype('uint8');patch,dcrop=centered(a,dx,dy,256);Image.fromarray(patch).save(d/'donor_native.png')
m=np.any(np.load(study/donor['segmentation_path'])>0,axis=0);patch,_=centered(m,dx,dy,256);Image.fromarray(patch.astype('uint8')*255).save(d/'donor_mask_native.png')
shutil.copy2(preview/'01_ad.png',d/'ad_2x.png')
(d/'selection.json').write_text(json.dumps(dict(category='fruit_jelly',source_provenance=src,hybrid_id=h['id'],donor_sample_id=donor['source_name'],control_sample_id=o['source_name'],crop_xywh=crop,donor_crop_xywh=dcrop,selection='Visual choice: existing preview case1, inside jelly rather than outside the object',ad_source=str(preview/'01_ad.png')),indent=2))
(d/'ad_metadata.json').write_text(json.dumps(dict(seed=results['seed'],selection=results['checkpoint'],aar=True,scale=2,ddim_steps=200,spatial_encoder_mode='eval',reused_preview_case=1),indent=2))
print('Replaced fruit jelly with inside-object preview case1.')
