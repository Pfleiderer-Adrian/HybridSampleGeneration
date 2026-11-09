
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

from pathlib import Path
import json,sqlite3,re,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import label,find_objects
from experiments.anomalydiffusion.scale_pilot import centered
ROOT=Path('/mnt/results/mvtec2/experiments/all_category_four_methods_20261105');olds=json.loads((ROOT/'replacement_exclusions.json').read_text())['old']
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
def rgb(a):
 a=np.clip(a.transpose(1,2,0),0,255).round().astype('uint8');return np.repeat(a,3,axis=2) if a.shape[2]==1 else a
for cat in (sys.argv[1:] or ['vial','wallplugs','walnuts']):
 legacy=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3')/cat;study=Path(json.loads((legacy/'protocol.json').read_text())['hybrid_study']);old=olds[cat]
 con=sqlite3.connect(f'file:{study}/artifacts.sqlite?mode=ro',uri=True);con.row_factory=sqlite3.Row
 tables={t:{r['id']:dict(r) for r in con.execute('SELECT * FROM '+t)} for t in ['original_samples','hybrid_samples','placements','synthetic_anomalies','real_anomalies']};con.close()
 train={s['sample_id'] for s in json.loads((legacy/'split_manifest.json').read_text())['partitions']['train']}
 oldgroup=re.search(r'bad_(\d+)',old['donor_sample_id']).group(1);ranked=[]
 for src in json.loads((legacy/'hybrid_provenance.json').read_text()):
  if src['hybrid_id']==old['hybrid_id']:continue
  h=tables['hybrid_samples'][src['hybrid_id']];o=tables['original_samples'][src['original_id']]
  if o['source_name']==old['control_sample_id']:continue
  assert o['source_name'] in train and not o['has_anomaly']
  mask=np.any(np.load(study/h['segmentation_path'],mmap_mode='r')>0,axis=0);labs,n=label(mask)
  healthy=np.load(study/o['image_path'],mmap_mode='r');hybrid=np.load(study/h['image_path'],mmap_mode='r')
  for ci,region in enumerate(find_objects(labs),1):
   hh,ww=region[0].stop-region[0].start,region[1].stop-region[1].start
   if max(hh,ww)>(4096 if cat=='vial' else 112) or max(hh,ww)<10:continue
   cy,cx=(region[0].start+region[0].stop)//2,(region[1].start+region[1].stop)//2
   points=[(cy,cx)]
   if cat=='vial' and max(hh,ww)>112:
    points += [(region[0].start+32,cx),(region[0].stop-32,cx),(cy,region[1].start+32),(cy,region[1].stop-32)]
   for cy,cx in points:
    if min(cx,cy)<128 or cy+128>mask.shape[0] or cx+128>mask.shape[1]:continue
    m=mask[cy-64:cy+64,cx-64:cx+64];area=int(m.sum())
    if not 20<=area<=(8000 if cat=='vial' else 4000):continue
    if cat!='vial' and int(np.count_nonzero(labs[region]==ci))!=area:continue
    hc=healthy[:,cy-64:cy+64,cx-64:cx+64];vc=hybrid[:,cy-64:cy+64,cx-64:cx+64]
    if float((hc[:,m]>235).mean())>(.9 if cat=='vial' else .1):continue
    if cat=='vial' and float(hc.mean())>235:continue
    diff=np.abs(vc.astype('float32')-hc.astype('float32')).mean(0);contrast=float(diff[m].mean());outside=float(diff[~m].mean())
    p=min((tables['placements'][pid] for pid in src['placement_ids']),key=lambda p:(p['position_y']*mask.shape[0]-cy)**2+(p['position_x']*mask.shape[1]-cx)**2)
    real=tables['real_anomalies'][tables['synthetic_anomalies'][p['synthetic_anomaly_id']]['real_anomaly_id']];donor=tables['original_samples'][real['original_sample_id']];group=re.search(r'bad_(\d+)',donor['source_name']).group(1)
    if group==oldgroup:continue
    assert donor['source_name'] in train
    ranked.append(dict(category=cat,score=contrast*np.log1p(area)-outside*3,inside_mae=contrast,outside_mae=outside,native_mask_area=area,component_hw=[hh,ww],hybrid_id=h['id'],original_id=o['id'],donor_id=donor['id'],real_anomaly_id=real['id'],placement_id=p['id'],control_sample_id=o['source_name'],donor_sample_id=donor['source_name'],donor_group=group,center_xy=[cx,cy],source_provenance=src,full_component_inside=bool(hh<=128 and ww<=128),study=str(study)))
 selected=[];seen=set();controls=set()
 for candidate in sorted(ranked,key=lambda c:c['score'],reverse=True):
  if candidate['donor_group'] in seen or candidate['original_id'] in controls:continue
  selected.append(candidate);seen.add(candidate['donor_group']);controls.add(candidate['original_id'])
  if len(selected)==6:break
 assert selected,cat
 dest=ROOT/'replacement_candidates'/cat;dest.mkdir(parents=True,exist_ok=True)
 board=Image.new('RGB',(len(selected)*276,620),'white');draw=ImageDraw.Draw(board)
 for i,c in enumerate(selected,1):
  d=dest/f'{i:02d}';d.mkdir(exist_ok=True);cx,cy=c['center_xy'];h=tables['hybrid_samples'][c['hybrid_id']];o=tables['original_samples'][c['original_id']]
  for key,path in [('healthy',o['image_path']),('hybrid',h['image_path'])]:
   patch,crop=centered(rgb(np.load(study/path)),cx,cy,256);im=Image.fromarray(patch);im.save(d/f'{key}_native.png');board.paste(im.crop((64,64,192,192)).resize((256,256),Image.Resampling.BICUBIC),((i-1)*276+5,60+(key=='hybrid')*280))
  mask=np.any(np.load(study/h['segmentation_path'])>0,axis=0);patch,_=centered(mask,cx,cy,256);Image.fromarray(patch.astype('uint8')*255).save(d/'hybrid_mask_native.png')
  real=tables['real_anomalies'][c['real_anomaly_id']];meta=json.loads(real['metadata_json']);donor=tables['original_samples'][c['donor_id']];dy,dx=meta['centroid_voxel']
  patch,dcrop=centered(rgb(np.load(study/donor['image_path'])),dx,dy,256);Image.fromarray(patch).save(d/'donor_native.png')
  mask=np.any(np.load(study/donor['segmentation_path'])>0,axis=0);patch,_=centered(mask,dx,dy,256);Image.fromarray(patch.astype('uint8')*255).save(d/'donor_mask_native.png')
  c.update(crop_xywh=crop,donor_crop_xywh=dcrop,selection='Replacement candidate: different control and donor capture group; component fully inside128 window; qualitative visibility ranking then visual review')
  (d/'selection.json').write_text(json.dumps(c,indent=2));draw.text(((i-1)*276+5,5),f"{i}: donor {c['donor_group']}\n{c['component_hw']} px",font=font,fill='black')
 board.save(dest/'contact_sheet.png');print(cat,'candidates',len(selected),flush=True)
