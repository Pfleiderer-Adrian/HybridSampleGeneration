
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

from pathlib import Path
import json,sqlite3,re,collections
import numpy as np
from scipy.ndimage import label,find_objects
from PIL import Image,ImageDraw
from hybrid_sample_generator.postprocessing.foreground import object_mask
from hybrid_sample_generator.configuration.postprocessing import ForegroundConfiguration
base=Path('/mnt/results/mvtec2/experiments');out=audit_dir('fruit_jelly_v2_vs_latest_v3')
studies={'v2':base/'comparison_v2/reruns/20261007T184903Z/fruit_jelly/hybrid_generation','v3':Path(json.load(open(base/'vergleich_benchmark/latest_fruit_jelly_full_rerun.json'))['study'])};catalog={};summaries={}
def group(n):return re.sub(r'_(regular|overexposed|underexposed|shift_\d+)\.png$','',n)
def rgb(a):return np.moveaxis(a,0,-1).clip(0,255).astype('uint8')
def focus(m,pad=40):
 lab,n=label(m);cnt=np.bincount(lab.ravel());cnt[0]=0;yy,xx=find_objects(lab)[cnt.argmax()-1];return max(0,xx.start-pad),max(0,yy.start-pad),min(m.shape[1],xx.stop+pad),min(m.shape[0],yy.stop+pad)
for name,s in studies.items():
 c=sqlite3.connect(f'file:{s}/artifacts.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row;t={k:{r['id']:dict(r) for r in c.execute('select * from '+k)} for k in ['original_samples','real_anomalies','synthetic_anomalies','hybrid_samples','placements']};accepted={r[0] for r in c.execute('select hybrid_sample_id from postprocessing_decisions where accepted=1')} if name=='v3' else set(t['hybrid_samples']);hs=[h for h in t['hybrid_samples'].values() if h['status']=='generated' and h['id'] in accepted];by=collections.defaultdict(list);groups=collections.Counter();used=set();syused=set()
 for h in hs:
  donors=[]
  for p in t['placements'].values():
   if p['hybrid_sample_id']!=h['id']:continue
   sy=t['synthetic_anomalies'][p['synthetic_anomaly_id']];r=t['real_anomalies'][sy['real_anomaly_id']];g=group(t['original_samples'][r['original_sample_id']]['source_name']);donors.append(g);groups[g]+=1;used.add(r['id']);syused.add(sy['id'])
  for g in set(donors):by[(t['original_samples'][h['original_sample_id']]['source_name'],'donor groups differ between pools')].append(h)
 rows=[];rng=np.random.default_rng(42)
 for idx in rng.choice(len(hs),min(80,len(hs)),replace=False):
  h=hs[idx];o=t['original_samples'][h['original_sample_id']];a=np.load(s/o['image_path']);b=np.load(s/h['image_path']);m=(np.load(s/h['segmentation_path'])>0).any(0);fg=object_mask(a,ForegroundConfiguration());delta=np.max(np.abs(b-a),axis=0);rows.append({'id':h['id'],'overlap':float(fg[m].mean()),'mask_fraction':float(m.mean()),'inside_change':float(np.median(delta[m])),'outside_gt1':int(((delta>1)&~m).sum())})
 summaries[name]={'hybrids':len(hs),'total_materialized':len(t['hybrid_samples']),'placements':sum(groups.values()),'groups':dict(groups),'real_rois_used':len(used),'synthetic_used':len(syused),'sample_size':len(rows),'sample_below_90_overlap':sum(r['overlap']<.9 for r in rows),'sample_below_50_overlap':sum(r['overlap']<.5 for r in rows),'median_sample_mask_fraction':float(np.median([r['mask_fraction'] for r in rows])),'median_sample_inside_change':float(np.median([r['inside_change'] for r in rows])),'sample_outside_gt1':sum(r['outside_gt1']>0 for r in rows),'rows':rows};catalog[name]=(s,t,by);print(name,{k:v for k,v in summaries[name].items() if k!='rows'},flush=True)
common=sorted(set(catalog['v2'][2])&set(catalog['v3'][2]));rng=np.random.default_rng(42);keys=[common[i] for i in rng.choice(len(common),min(16,len(common)),replace=False)]
for page in range((len(keys)+3)//4):
 canvas=Image.new('RGB',(1600,1200),'white');d=ImageDraw.Draw(canvas)
 for j,key in enumerate(keys[page*4:page*4+4]):
  d.text((5,j*300+5),key[0]+' | '+key[1]+' | V2 original+mask / V2 hybrid / V3 original+mask / V3 hybrid',fill='black')
  for z,name in enumerate(['v2','v3']):
   s,t,by=catalog[name];h=sorted(by[key],key=lambda h:h['id'])[0];o=t['original_samples'][h['original_sample_id']];a=rgb(np.load(s/o['image_path']));b=rgb(np.load(s/h['image_path']));m=(np.load(s/h['segmentation_path'])>0).any(0);ov=a.copy();ov[m]=(ov[m]*.3+np.array([255,0,0])*.7).astype('uint8');box=focus(m)
   for k,im in enumerate([ov,b]):
    im=Image.fromarray(im).crop(box);im.thumbnail((390,265));canvas.paste(im,((z*2+k)*400,j*300+30))
 canvas.save(out/f'paired_{page+1}.jpg',quality=95)
(out/'summary.json').write_text(json.dumps({'studies':{k:str(v) for k,v in studies.items()},'paired_target_and_donor_group_keys':len(common),'summaries':summaries},indent=2))
