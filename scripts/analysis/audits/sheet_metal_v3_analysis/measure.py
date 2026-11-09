
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

from pathlib import Path
import sqlite3,json,re,collections
import numpy as np
from PIL import Image,ImageDraw
s=Path('/mnt/results/mvtec2/experiments/comparison_v3/sheet_metal/hybrid_generation');db=sqlite3.connect(s/'artifacts.sqlite');db.row_factory=sqlite3.Row;out=audit_dir('sheet_metal_v3_analysis')
rows=db.execute('select h.*,o.image_path original,o.source_name target from hybrid_samples h join original_samples o on o.id=h.original_sample_id order by h.id').fetchall();metrics=[]
for row in rows:
 a=np.load(s/row['original'],mmap_mode='r');b=np.load(s/row['image_path'],mmap_mode='r');m=(np.load(s/row['segmentation_path'],mmap_mode='r')>0).any(0);delta=np.abs(b.astype('float32')-a.astype('float32')).max(0);x=np.where(m)[1]
 metrics.append({'id':row['id'],'target':row['target'],'mask_pixels':int(m.sum()),'median_delta_inside':float(np.median(delta[m])),'fraction_inside_delta_lt1':float((delta[m]<1).mean()),'changed_outside_pixels':int(((delta>1)&~m).sum()),'fraction_mask_in_outer_10pct_width':float(((x<m.shape[1]*.1)|(x>m.shape[1]*.9)).mean())})
print('METRICS', {key:np.percentile([x[key] for x in metrics],[0,10,50,90,100]).tolist() for key in ['mask_pixels','median_delta_inside','fraction_inside_delta_lt1','changed_outside_pixels']})
print('mask mostly low change',sum(x['fraction_inside_delta_lt1']>.5 for x in metrics),'outside changes >0',sum(x['changed_outside_pixels']>0 for x in metrics))
counts=collections.Counter();available=collections.Counter()
for row in db.execute('select o.source_name from real_anomalies r join original_samples o on o.id=r.original_sample_id'):available[re.sub(r'_(regular|overexposed|underexposed|shift_\d+)\.png$','',row[0])]+=1
for row in db.execute('select o.source_name from placements p join synthetic_anomalies s on s.id=p.synthetic_anomaly_id join real_anomalies r on r.id=s.real_anomaly_id join original_samples o on o.id=r.original_sample_id'):counts[re.sub(r'_(regular|overexposed|underexposed|shift_\d+)\.png$','',row[0])]+=1
print('groups available',dict(available),'placements',dict(counts))
# Consistent closeups: original, hybrid, red mask on original.
names=[r['target'] for r in rows];rng=np.random.default_rng(42);chosen=[rows[k] for k in rng.choice(len(rows),6,replace=False)]
for page in range(2):
 canvas=Image.new('RGB',(1200,960),'white');d=ImageDraw.Draw(canvas)
 for j,row in enumerate(chosen[page*3:page*3+3]):
  a=np.load(s/row['original']);b=np.load(s/row['image_path']);m=(np.load(s/row['segmentation_path'])>0).any(0);y,x=np.where(m);cy=int(np.median(y));cx=int(np.median(x));box=(max(0,cx-160),max(0,cy-130),min(m.shape[1],cx+160),min(m.shape[0],cy+130))
  def rgb(v):return np.moveaxis(np.repeat(v,3,0),0,-1).clip(0,255).astype('uint8')
  aa=rgb(a);bb=rgb(b);ov=aa.copy();ov[m]=(ov[m]*.3+np.array([255,0,0])*.7).astype('uint8')
  for k,im in enumerate([aa,bb,ov]):
   img=Image.fromarray(im).crop(box);img.thumbnail((390,280));canvas.paste(img,(400*k,j*320+30))
  d.text((5,j*320+5),row['target']+' | original / hybrid / mask on original',fill='black')
 canvas.save(out/f'closeups_{page+1}.jpg',quality=95)
(out/'measurements.json').write_text(json.dumps({'hybrids':metrics,'source_groups':dict(available),'placement_groups':dict(counts)},indent=2)+'\n')
