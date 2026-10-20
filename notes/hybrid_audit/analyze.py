import json,sqlite3,pathlib,csv,collections,re
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
ROOT=pathlib.Path('/mnt/results/mvtec2/experiments/comparison_v2');B=pathlib.Path('/mnt/results/mvtec2/submissions/dinov3_global_best_wallplugs_updated_20261018T164253Z');OUT=pathlib.Path(__file__).parent

def rgb(a):
 a=np.asarray(a)
 if a.ndim==3 and a.shape[0] in (1,3,4):a=np.moveaxis(a,0,-1)
 if a.ndim==2:a=np.repeat(a[...,None],3,-1)
 if a.shape[-1]==1:a=np.repeat(a,3,-1)
 return np.clip(a[...,:3],0,255).astype('uint8')
def mask2(a):
 a=np.asarray(a).squeeze()>0
 return a.any(0) if a.ndim==3 else a
def box(mask,pad=70):
 y,x=np.where(mask);return (max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(mask.shape[1],int(x.max())+pad+1),min(mask.shape[0],int(y.max())+pad+1))
def tile(a,b=None):
 im=Image.fromarray(a)
 if b:im=im.crop(b)
 im.thumbnail((310,240));panel=Image.new('RGB',(320,250),'#222222');panel.paste(im,((320-im.width)//2,(250-im.height)//2));return panel
summ={}
for cat in ['can','vial','wallplugs']:
 study=ROOT/cat/'hybrid_generation';db=sqlite3.connect('file:'+str(study/'artifacts.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 tables={t:{x['id']:dict(x) for x in db.execute('select * from '+t)} for t in ['original_samples','hybrid_samples','placements','synthetic_anomalies','real_anomalies']}
 manifest=json.loads((B/'models'/cat/'split_manifest.json').read_text());prov=manifest['hybrid_provenance'];ids=[p['hybrid_id'] for p in prov];orig=tables['original_samples'];hyb=tables['hybrid_samples'];pl=tables['placements'];syn=tables['synthetic_anomalies'];real=tables['real_anomalies']
 def load(p):return np.load(study/p,mmap_mode='r')
 donor_counts=collections.Counter(d for p in prov for d in p['donor_original_ids']);captures=collections.Counter()
 for d,n in donor_counts.items():captures[re.sub(r'_(regular|overexposed|underexposed|shift_\d+)\.png$','',orig[d]['source_name'])]+=n
 held={str(pathlib.Path(s['image_path']).resolve()) for k in ['validation','test'] for s in manifest['partitions'][k]}
 leaks=[d for d in donor_counts if str(pathlib.Path(json.loads(orig[d]['metadata_json'])['source_image_path']).resolve()) in held]
 rows=[]
 for p in prov:
  ps=[pl[x] for x in p['placement_ids']];rows.append({'hybrid_id':p['hybrid_id'],'score':min(x['score'] for x in ps),'placements':len(ps),'target':orig[p['original_id']]['source_name'],'donors':'|'.join(orig[d]['source_name'] for d in p['donor_original_ids'])})
 rng=np.random.default_rng(42);idx=set(rng.choice(len(rows),min(32,len(rows)),replace=False).tolist());order=sorted(range(len(rows)),key=lambda i:rows[i]['score']);idx.update(order[:8]+order[-8:]);sample=[rows[i] for i in sorted(idx)]
 for row in sample:
  h=hyb[row['hybrid_id']];o=orig[h['original_sample_id']];a=np.asarray(load(h['image_path']),dtype=np.float32);g=np.asarray(load(o['image_path']),dtype=np.float32);m=mask2(load(h['segmentation_path']))
  if a.shape[0] in [1,3]:delta=np.abs(a-g).mean(0)
  else:delta=np.abs(a-g).mean(-1)
  dil=binary_dilation(m,iterations=8);outside=~dil;changed=delta>2
  row.update(mask_fraction=float(m.mean()),mean_abs_inside=float(delta[m].mean()) if m.any() else 0,mean_abs_far_outside=float(delta[outside].mean()),changed_pixels_outside_share=float(changed[outside].sum()/max(changed.sum(),1)),far_background_changed_fraction=float(changed[outside].mean()),finite=bool(np.isfinite(a).all()),image_min=float(a.min()),image_max=float(a.max()))
  print(cat,row['hybrid_id'],flush=True)
 with (OUT/f'{cat}_sample_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(sample[0]));w.writeheader();w.writerows(sample)
 # Deliberate diagnostic sample: random plus extremes in score / out-of-mask edits / low visibility.
 picks=[]
 for candidate in [min(sample,key=lambda x:x['score']),max(sample,key=lambda x:x['score']),max(sample,key=lambda x:x['mean_abs_far_outside']),max(sample,key=lambda x:x['changed_pixels_outside_share']),min(sample,key=lambda x:x['mean_abs_inside'])]+sample:
  if candidate not in picks:picks.append(candidate)
  if len(picks)==12:break
 for page in range(2):
  sheet=Image.new('RGB',(1600,6*290+35),'white');draw=ImageDraw.Draw(sheet)
  for i,title in enumerate(['Healthy target (crop)','Real donor (defect crop)','Hybrid (same target crop)','Hybrid + mask (red)','Absolute change x4']):draw.text((i*320+5,8),title,fill='black')
  for i,row in enumerate(picks[page*6:(page+1)*6]):
   h=hyb[row['hybrid_id']];o=orig[h['original_sample_id']];a=rgb(load(h['image_path']));g=rgb(load(o['image_path']));m=mask2(load(h['segmentation_path']));b=box(m);ps=[p for p in pl.values() if p['hybrid_sample_id']==h['id']];d=orig[real[syn[ps[0]['synthetic_anomaly_id']]['real_anomaly_id']]['original_sample_id']];di=rgb(load(d['image_path']));dm=mask2(load(d['segmentation_path']));overlay=a.copy();overlay[m]=(overlay[m]*.5+np.array([255,0,0])*.5).astype('uint8');diff=np.clip(np.abs(a.astype(float)-g)*4,0,255).astype('uint8')
   for j,(im,bb) in enumerate([(g,b),(di,box(dm)),(a,b),(overlay,b),(diff,b)]):sheet.paste(tile(im,bb),(j*320,35+i*290))
   draw.text((5,285+i*290),f"{h['id']} score={row['score']:.3f} mask={row['mask_fraction']:.4f} edits outside={row['changed_pixels_outside_share']:.1%}",fill='black')
  sheet.save(OUT/f'{cat}_overview_{page+1}.jpg',quality=90)
 summary={'eligible_training_hybrids':len(prov),'target_images':len(set(p['original_id'] for p in prov)),'donor_images_used':len(donor_counts),'donor_capture_groups_used':len(captures),'donor_heldout_overlap':leaks,'top_donor_images':donor_counts.most_common(10),'capture_usage':captures,'sample_size':len(sample),'metrics_median':{k:float(np.median([r[k] for r in sample])) for k in ['score','mask_fraction','mean_abs_inside','mean_abs_far_outside','changed_pixels_outside_share','far_background_changed_fraction']},'score_all_quantiles':np.quantile([r['score'] for r in rows],[0,.25,.5,.75,1]).tolist(),'study':str(study),'frozen_manifest':str(B/'models'/cat/'split_manifest.json'),'configuration':json.loads((study/'configuration.json').read_text())}
 summ[cat]=summary
 (OUT/'summary.json').write_text(json.dumps(summ,indent=2)+'\n')
