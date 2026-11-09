
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

from pathlib import Path
import json,sqlite3,shutil,sys
import numpy as np
from PIL import Image
from scipy.ndimage import label,find_objects
from experiments.anomalydiffusion.scale_pilot import centered
ROOT=Path('/mnt/results/mvtec2/experiments/all_category_four_methods_20261105');ROOT.mkdir(parents=True,exist_ok=True)
CATS=['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts']
def rgb(value):
 a=np.clip(value.transpose(1,2,0),0,255).round().astype('uint8');return np.repeat(a,3,axis=2) if a.shape[2]==1 else a
for cat in (sys.argv[1:] or CATS):
 d=ROOT/cat;d.mkdir(exist_ok=True)
 if cat=='can':
  old=Path('/mnt/results/mvtec2/experiments/can_four_method_examples_20261105/case_05')
  for p in old.glob('*.png'):shutil.copy2(p,d/p.name)
  oldsel=json.loads((old.parent/'selection.json').read_text())['cases'][4]
  for n in ['ad_metadata.json','baselines_metadata.json']:shutil.copy2(old/n,d/n)
  (d/'selection.json').write_text(json.dumps(oldsel,indent=2));continue
 legacy=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3')/cat
 study=Path(json.loads((legacy/'protocol.json').read_text())['hybrid_study'])
 frozen=json.loads((legacy/'hybrid_provenance.json').read_text())
 train={r['sample_id'] for r in json.loads((legacy/'split_manifest.json').read_text())['partitions']['train']}
 con=sqlite3.connect(f'file:{study}/artifacts.sqlite?mode=ro',uri=True);con.row_factory=sqlite3.Row
 tables={t:{r['id']:dict(r) for r in con.execute('SELECT * FROM '+t)} for t in ['original_samples','hybrid_samples','placements','synthetic_anomalies','real_anomalies']};con.close()
 best=None
 for src in frozen[:80]:
  h=tables['hybrid_samples'][src['hybrid_id']];o=tables['original_samples'][src['original_id']]
  assert o['source_name'] in train and not o['has_anomaly']
  mask=np.any(np.load(study/h['segmentation_path'],mmap_mode='r')>0,axis=0);labs,n=label(mask)
  healthy=np.load(study/o['image_path'],mmap_mode='r');hybrid=np.load(study/h['image_path'],mmap_mode='r')
  regions=find_objects(labs);sizes=np.bincount(labs.ravel());sizes[0]=0
  for component in np.argsort(sizes)[::-1][:6]:
   if component==0:continue
   region=regions[component-1];cy,cx=(region[0].start+region[0].stop)//2,(region[1].start+region[1].stop)//2
   if min(cx,cy)<128 or cy+128>mask.shape[0] or cx+128>mask.shape[1]:continue
   m=mask[cy-64:cy+64,cx-64:cx+64];area=int(m.sum())
   if not 20<=area<=(2400 if cat in ['vial','wallplugs'] else 8500):continue
   hcore=healthy[:,cy-64:cy+64,cx-64:cx+64];vcore=hybrid[:,cy-64:cy+64,cx-64:cx+64]
   
   if cat=='vial' and float((hcore[:,m]>235).mean())>.1:continue
   diff=np.abs(vcore.astype('float32')-hcore.astype('float32')).mean(0);inside=float(diff[m].mean());outside=float(diff[~m].mean())
   score=inside*np.log1p(area)-outside*3
   if best and score<=best['score']:continue
   p=min((tables['placements'][p] for p in src['placement_ids']),key=lambda p:(p['position_y']*mask.shape[0]-cy)**2+(p['position_x']*mask.shape[1]-cx)**2)
   real=tables['real_anomalies'][tables['synthetic_anomalies'][p['synthetic_anomaly_id']]['real_anomaly_id']];donor=tables['original_samples'][real['original_sample_id']]
   assert donor['source_name'] in train
   best=dict(category=cat,score=score,inside_mae=inside,outside_mae=outside,native_mask_area=area,hybrid_id=h['id'],original_id=o['id'],donor_id=donor['id'],real_anomaly_id=real['id'],placement_id=p['id'],control_sample_id=o['source_name'],donor_sample_id=donor['source_name'],center_xy=[cx,cy],source_provenance=src,full_component_inside=bool(region[0].start>=cy-64 and region[0].stop<=cy+64 and region[1].start>=cx-64 and region[1].stop<=cx+64))
 assert best is not None,cat
 h=tables['hybrid_samples'][best['hybrid_id']];o=tables['original_samples'][best['original_id']];cx,cy=best['center_xy']
 for key,path in [('healthy',o['image_path']),('hybrid',h['image_path'])]:
  patch,crop=centered(rgb(np.load(study/path)),cx,cy,256);Image.fromarray(patch).save(d/f'{key}_native.png')
 mask=np.any(np.load(study/h['segmentation_path'])>0,axis=0);patch,_=centered(mask,cx,cy,256);Image.fromarray(patch.astype('uint8')*255).save(d/'hybrid_mask_native.png')
 real=tables['real_anomalies'][best['real_anomaly_id']];meta=json.loads(real['metadata_json']);donor=tables['original_samples'][best['donor_id']];dy,dx=meta['centroid_voxel']
 patch,dcrop=centered(rgb(np.load(study/donor['image_path'])),dx,dy,256);Image.fromarray(patch).save(d/'donor_native.png')
 mask=np.any(np.load(study/donor['segmentation_path'])>0,axis=0);patch,_=centered(mask,dx,dy,256);Image.fromarray(patch.astype('uint8')*255).save(d/'donor_mask_native.png')
 best.update(study=str(study),crop_xywh=crop,donor_crop_xywh=dcrop,selection='Highest hybrid foreground-change score among first80 frozen pool entries, moderate core mask area; selected qualitative example')
 (d/'selection.json').write_text(json.dumps(best,indent=2));print('Selected',cat,best['inside_mae'],flush=True)
