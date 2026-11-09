
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

from pathlib import Path
import sqlite3,json
import numpy as np
from PIL import Image,ImageDraw
s=Path('/mnt/results/mvtec2/experiments/comparison_v3/sheet_metal/hybrid_generation');db=sqlite3.connect(s/'artifacts.sqlite');db.row_factory=sqlite3.Row
rows=db.execute('select h.id,h.image_path,h.segmentation_path,o.image_path original,o.source_name from hybrid_samples h join original_samples o on o.id=h.original_sample_id order by h.id').fetchall();rng=np.random.default_rng(42);out=audit_dir('sheet_metal_v3_analysis')
for page in range(3):
 canvas=Image.new('RGB',(1200,1050),'white');draw=ImageDraw.Draw(canvas)
 for j,index in enumerate(rng.choice(len(rows),4,replace=False)):
  r=rows[index];a=np.load(s/r['original'],mmap_mode='r');b=np.load(s/r['image_path'],mmap_mode='r');m=(np.load(s/r['segmentation_path'],mmap_mode='r')>0).any(0)
  def rgb(v):
   if v.shape[0]==1:v=np.repeat(v,3,0)
   return np.moveaxis(v,0,-1).clip(0,255).astype('uint8')
  original=rgb(a);hybrid=rgb(b);overlay=original.copy();overlay[m]=(overlay[m]*.3+np.array([255,0,0])*.7).astype('uint8')
  y,x=np.where(m);box=(max(0,int(x.min())-80),max(0,int(y.min())-80),min(m.shape[1],int(x.max())+80),min(m.shape[0],int(y.max())+80))
  for k,im in enumerate([Image.fromarray(original),Image.fromarray(overlay),Image.fromarray(hybrid).crop(box)]):
   im.thumbnail((395,230));canvas.paste(im,(k*400,j*260+25))
  draw.text((5,j*260+5),r['source_name'],fill='black')
 canvas.save(out/f'random_{page+1}.jpg')
print('shape',a.shape,'pixels',np.percentile(a,[0,5,25,50,75,95,100]))
