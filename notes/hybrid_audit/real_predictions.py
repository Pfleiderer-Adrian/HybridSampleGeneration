import json,pathlib,csv
import numpy as np
from PIL import Image,ImageDraw
ROOT=pathlib.Path(__file__).parent;B=pathlib.Path('/mnt/results/mvtec2/submissions/dinov3_global_best_wallplugs_updated_20261018T164253Z');jobs=json.loads((B/'manifest.json').read_text())['jobs']
for cat in ['can','vial','wallplugs']:
 j=jobs[cat];run=pathlib.Path(j['source_run']);manifest=json.loads((pathlib.Path(j['run'])/'split_manifest.json').read_text());rows=[]
 for split in ['validation','test']:
  samples={s['sample_id']:s for s in manifest['partitions'][split]}
  if not samples:continue
  for i,r in enumerate(csv.DictReader((run/'predictions'/split/'index.csv').open())):
   s=samples[r['sample_id']]
   if s['label']=='good':continue
   a=np.load(run/'predictions'/split/f'{i:06d}.npy',mmap_mode='r');m=np.asarray(Image.open(s['mask_path']))>0
   if m.ndim==3:m=m.any(-1)
   score=float(a[m].mean()-np.quantile(a[~m],.99));rows.append((score,s,str(run/'predictions'/split/f'{i:06d}.npy')))
 rows.sort(key=lambda x:x[0]);sheet=Image.new('RGB',(1400,4*330),'white');draw=ImageDraw.Draw(sheet)
 for k,(contrast,s,p) in enumerate(rows[:4]):
  a=np.load(p);m=np.asarray(Image.open(s['mask_path']))>0
  if m.ndim==3:m=m.any(-1)
  y,x=np.where(m);box=(max(0,int(x.min())-60),max(0,int(y.min())-60),min(m.shape[1],int(x.max())+61),min(m.shape[0],int(y.max())+61))
  im=Image.open(s['image_path']).convert('RGB');overlay=np.array(im);overlay[m]=(overlay[m]*.5+np.array([255,0,0])*.5).astype('uint8');heat=np.stack([a,np.zeros_like(a),1-a],-1);heat=(heat*255).astype('uint8')
  views=[im,Image.fromarray(overlay),Image.fromarray(overlay).crop(box),Image.fromarray(heat).crop(box)]
  for z,v in enumerate(views):v.thumbnail((345,285));sheet.paste(v,(z*350,30+k*330))
  draw.text((5,k*330+5),s['sample_id']+f'   mean(defect)-q99(background)={contrast:.3f}; heat blue=0 red=1',fill='black')
 sheet.save(ROOT/f'{cat}_real_prediction_examples.jpg',quality=90)
 (ROOT/f'{cat}_real_prediction_ranking.json').write_text(json.dumps([{'contrast':c,'sample':s,'prediction':p} for c,s,p in rows],indent=2))
