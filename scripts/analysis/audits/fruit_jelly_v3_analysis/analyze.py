
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

from pathlib import Path
import sqlite3,json,collections,re
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import label,find_objects
s=Path('/mnt/results/mvtec2/experiments/comparison_v3/fruit_jelly/hybrid_generation');out=audit_dir('fruit_jelly_v3_analysis');db=sqlite3.connect(f'file:{s}/artifacts.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
t={name:{r['id']:dict(r) for r in db.execute('select * from '+name)} for name in ['original_samples','real_anomalies','synthetic_anomalies','hybrid_samples','placements']}
def group(n):return re.sub(r'_(regular|overexposed|underexposed|shift_\d+)\.png$','',n)
def rgb(a):return np.moveaxis(np.repeat(a,3,0) if a.shape[0]==1 else a,0,-1).clip(0,255).astype('uint8')
def crop(m):
 labs,n=label(m);cnt=np.bincount(labs.ravel());cnt[0]=0;yy,xx=find_objects(labs)[cnt.argmax()-1];return max(0,xx.start-55),max(0,yy.start-55),min(m.shape[1],xx.stop+55),min(m.shape[0],yy.stop+55)
used=collections.Counter();variants=collections.Counter();groups=collections.Counter();avail=collections.Counter();per=collections.Counter();dists=[]
for r in t['real_anomalies'].values():avail[group(t['original_samples'][r['original_sample_id']]['source_name'])]+=1
for p in t['placements'].values():
 sy=t['synthetic_anomalies'][p['synthetic_anomaly_id']];r=t['real_anomalies'][sy['real_anomaly_id']];used[r['id']]+=1;variants[sy['id']]+=1;groups[group(t['original_samples'][r['original_sample_id']]['source_name'])]+=1;per[p['hybrid_sample_id']]+=1;dists.append(float(np.hypot(p['position_y']-r['position_y'],p['position_x']-r['position_x'])))
hs=sorted(t['hybrid_samples'].values(),key=lambda h:h['id']);sel=set(np.random.default_rng(42).choice(len(hs),min(16,len(hs)),replace=False).tolist());rows=[];previews=[]
for i,h in enumerate(hs):
 o=t['original_samples'][h['original_sample_id']];a=np.load(s/o['image_path']);b=np.load(s/h['image_path']);m=(np.load(s/h['segmentation_path'])>0).any(0);delta=np.max(np.abs(a-b),axis=0);outside=delta[~m];rows.append({'id':h['id'],'mask_pixels':int(m.sum()),'mask_fraction':float(m.mean()),'median_inside_change':float(np.median(delta[m])),'outside_gt1':int((outside>1).sum()),'outside_gt20':int((outside>20).sum())})
 if i in sel:
  aa=rgb(a);bb=rgb(b);ov=aa.copy();ov[m]=(ov[m]*.3+np.array([255,0,0])*.7).astype('uint8');box=crop(m);previews.append((o['source_name'],[Image.fromarray(x).crop(box) for x in [aa,bb,ov]],[Image.fromarray(x) for x in [aa,bb,ov]]))
for mode in ['focused','overview']:
 for page in range(4):
  canvas=Image.new('RGB',(1200,1200),'white');draw=ImageDraw.Draw(canvas)
  for j,(name,ims,full) in enumerate(previews[page*4:page*4+4]):
   draw.text((5,j*300+5),name+' | original / hybrid / mask',fill='black')
   for k,im in enumerate(ims if mode=='focused' else full):im=im.copy();im.thumbnail((390,265));canvas.paste(im,(k*400,j*300+30))
  canvas.save(out/f'{mode}_{page+1}.jpg',quality=95)
# One true defect example per capture group.
examples={}
for o in t['original_samples'].values():
 if o['has_anomaly'] and o['segmentation_path']:
  key=group(o['source_name'])
  if key not in examples or '_regular.' in o['source_name']:examples[key]=o
canvas=Image.new('RGB',(1200,300*((len(examples)+2)//3)),'white');draw=ImageDraw.Draw(canvas)
for j,(name,o) in enumerate(sorted(examples.items())):
 a=np.load(s/o['image_path']);m=(np.load(s/o['segmentation_path'])>0).any(0);im=Image.fromarray(rgb(a)).crop(crop(m));im.thumbnail((390,265));canvas.paste(im,((j%3)*400,(j//3)*300+30));draw.text(((j%3)*400+5,(j//3)*300+5),name,fill='black')
canvas.save(out/'real_groups.jpg',quality=95)
cfg=json.load(open(s/'configuration.json'));scores={}
for v in ['01_real_only','02_real_draem','03_real_cutmix','04_real_hybrid','05_real_hybrid_draem','06_real_hybrid_draem_cutmix']:
 d=json.load(open(s.parents[2]/'vergleich_benchmark'/v/'fruit_jelly/status.json'));scores[v]={'score':d.get('best_aupro'),'epoch':d.get('selected_epoch')}
summary={'hybrids':len(hs),'placements':len(t['placements']),'real_rois_available':len(avail) and len(t['real_anomalies']),'real_rois_used':len(used),'synthetic_available':len(t['synthetic_anomalies']),'synthetic_used':len(variants),'roi_use_histogram':dict(collections.Counter(used.values())),'group_rois':dict(avail),'group_placements':dict(groups),'placements_per_hybrid':dict(collections.Counter(per.values())),'median_mask_pixels':float(np.median([r['mask_pixels'] for r in rows])),'mask_fraction_percentiles':np.percentile([r['mask_fraction'] for r in rows],[10,50,90,100]).tolist(),'median_inside_change':float(np.median([r['median_inside_change'] for r in rows])),'outside_gt1_hybrids':sum(r['outside_gt1']>0 for r in rows),'outside_gt20_hybrids':sum(r['outside_gt20']>0 for r in rows),'matching':cfg['matching'],'fusion':cfg['fusion'],'scores':scores,'rows':rows}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
