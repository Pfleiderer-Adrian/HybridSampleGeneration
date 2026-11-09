
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

from pathlib import Path
import json,sqlite3
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import label,find_objects
from experiments.anomalydiffusion.scale_pilot import centered
ROOT=Path('/mnt/results/mvtec2/experiments/can_four_method_examples_20261105');ROOT.mkdir(parents=True,exist_ok=False)
legacy=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/can');study=Path(json.loads((legacy/'protocol.json').read_text())['hybrid_study'])
con=sqlite3.connect(f'file:{study}/artifacts.sqlite?mode=ro',uri=True);con.row_factory=sqlite3.Row
tables={t:{r['id']:dict(r) for r in con.execute('SELECT * FROM '+t)} for t in ['original_samples','hybrid_samples','placements','synthetic_anomalies','real_anomalies']};con.close()
provenance=json.loads((legacy/'hybrid_provenance.json').read_text());candidates=[]
train={r['sample_id'] for r in json.loads((legacy/'split_manifest.json').read_text())['partitions']['train']}
for source in provenance:
 h=tables['hybrid_samples'][source['hybrid_id']];o=tables['original_samples'][source['original_id']]
 assert o['source_name'] in train and not o['has_anomaly']
 mask=np.any(np.load(study/h['segmentation_path'],allow_pickle=False,mmap_mode='r')>0,axis=0)
 labs,n=label(mask)
 for ci,region in enumerate(find_objects(labs),1):
  hh,ww=region[0].stop-region[0].start,region[1].stop-region[1].start
  area=int(np.count_nonzero(labs[region]==ci))
  if max(hh,ww)>116 or max(hh,ww)<20 or not 100<=area<=2200:continue
  cy,cx=(region[0].start+region[0].stop)//2,(region[1].start+region[1].stop)//2
  if min(cx,cy)<128 or cx+128>mask.shape[1] or cy+128>mask.shape[0]:continue
  core=mask[cy-64:cy+64,cx-64:cx+64]
  if int(core.sum())!=area:continue
  healthy=np.load(study/o['image_path'],allow_pickle=False,mmap_mode='r')[:,cy-64:cy+64,cx-64:cx+64].transpose(1,2,0)
  hybrid=np.load(study/h['image_path'],allow_pickle=False,mmap_mode='r')[:,cy-64:cy+64,cx-64:cx+64].transpose(1,2,0)
  diff=np.abs(hybrid.astype('float32')-healthy.astype('float32')).mean(-1)
  contrast=float(diff[core].mean());outside=float(diff[~core].mean())
  if contrast<13:continue
  placements=[tables['placements'][p] for p in source['placement_ids']]
  placement=min(placements,key=lambda p:(p['position_y']*mask.shape[0]-cy)**2+(p['position_x']*mask.shape[1]-cx)**2)
  synth=tables['synthetic_anomalies'][placement['synthetic_anomaly_id']];real=tables['real_anomalies'][synth['real_anomaly_id']];donor=tables['original_samples'][real['original_sample_id']]
  assert donor['source_name'] in train
  group=donor['source_name'].rsplit('_',1)[0].split('_regular')[0]
  # Lighting variants share the numeric capture ID, not the final suffix.
  import re
  group=re.search(r'bad_(\d+)',donor['source_name']).group(1)
  candidates.append(dict(hybrid_id=h['id'],control_id=o['id'],control_sample_id=o['source_name'],donor_id=donor['id'],donor_sample_id=donor['source_name'],donor_group=group,
   real_anomaly_id=real['id'],placement_id=placement['id'],native_center_xy=[cx,cy],native_area=area,contrast=contrast,outside_mae=outside,
   score=contrast*np.log1p(area)-outside*5,source_provenance=source))
selected=[];seen_controls=set();seen_groups=set()
for c in sorted(candidates,key=lambda r:r['score'],reverse=True):
 if c['donor_group'] in seen_groups or c['control_id'] in seen_controls:continue
 selected.append(c);seen_groups.add(c['donor_group']);seen_controls.add(c['control_id'])
 if len(selected)==5:break
assert len(selected)==5
for i,c in enumerate(selected):
 d=ROOT/f'case_{i+1:02d}';d.mkdir();cx,cy=c['native_center_xy'];h=tables['hybrid_samples'][c['hybrid_id']];o=tables['original_samples'][c['control_id']]
 for key,path in [('healthy',o['image_path']),('hybrid',h['image_path'])]:
  value=np.clip(np.load(study/path).transpose(1,2,0),0,255).round().astype('uint8');patch,crop=centered(value,cx,cy,256);Image.fromarray(patch).save(d/f'{key}_native.png')
 mask=np.any(np.load(study/h['segmentation_path'])>0,axis=0);patch,_=centered(mask,cx,cy,256);Image.fromarray(patch.astype('uint8')*255).save(d/'hybrid_mask_native.png')
 real=tables['real_anomalies'][c['real_anomaly_id']];meta=json.loads(real['metadata_json']);donor=tables['original_samples'][c['donor_id']]
 dy,dx=meta['centroid_voxel'];value=np.clip(np.load(study/donor['image_path']).transpose(1,2,0),0,255).round().astype('uint8');patch,dcrop=centered(value,dx,dy,256);Image.fromarray(patch).save(d/'donor_native.png')
 dm=np.any(np.load(study/donor['segmentation_path'])>0,axis=0);patch,_=centered(dm,dx,dy,256);Image.fromarray(patch.astype('uint8')*255).save(d/'donor_mask_native.png')
 c.update(index=i+1,crop_xywh=crop,donor_crop_xywh=dcrop)
(ROOT/'selection.json').write_text(json.dumps(dict(cases=selected,study=str(study),selection='Hybrid visibility ranking; five distinct donor capture groups and controls. Qualitative, selected examples; no aggregate-quality claim.'),indent=2))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
im=Image.new('RGB',(5*276,620),'white');dr=ImageDraw.Draw(im)
for i,c in enumerate(selected):
 d=ROOT/f'case_{i+1:02d}';x=i*276;dr.text((x+5,5),f"{i+1}: donor {c['donor_group']}",font=font,fill='black')
 for j,key in enumerate(['healthy','hybrid']):
  v=Image.open(d/f'{key}_native.png').crop((64,64,192,192)).resize((256,256),Image.Resampling.BICUBIC);im.paste(v,(x+5,40+j*280))
im.save(ROOT/'selection_preview.png')
print(json.dumps(selected,indent=2))
