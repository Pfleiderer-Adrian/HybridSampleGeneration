
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

from pathlib import Path
import json,hashlib,shutil
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/mnt/results/mvtec2/experiments/v3_actual_training_examples_20261105');OUT=report_dir('v3_actual_training_examples_2026-11-05');OUT.mkdir(exist_ok=True)
choices={'vial':{'hybrid':1,'cutmix':2,'draem':4},'wallplugs':{'hybrid':1,'cutmix':2,'draem':5},'walnuts':{'hybrid':1,'cutmix':6,'draem':5}}
fontdir=Path('/usr/share/fonts/truetype/dejavu')
def font(n,bold=False):return ImageFont.truetype(str(fontdir/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),n)
def pending():
 p=Image.new('RGB',(256,256),'#e8edf2');d=ImageDraw.Draw(p);d.multiline_text((20,85),'No V3 training sample\navailable for this\ncategory yet',font=font(18,True),fill='#627186',spacing=8);return p
manifest=[];figures=[]
for cat,selected in choices.items():
 image=Image.new('RGB',(1280,800),'#f8f9fb');draw=ImageDraw.Draw(image)
 draw.text((20,18),cat.replace('_',' ').title()+' | V3 training examples',font=font(29,True),fill='#172334')
 draw.text((20,61),'Frozen data-loader replay | Epoch 1 | Native 256 x 256 patches | Method-specific healthy partners',font=font(18),fill='#38485d')
 draw.multiline_text((20,225),'Training\nsample',font=font(18,True),fill='#172334',spacing=6)
 draw.multiline_text((20,550),'Healthy\npartner',font=font(18,True),fill='#172334',spacing=6)
 category=dict(category=cat,sources={},illustrative_AD_omitted=True)
 for j,(title,source) in enumerate([('Hybrid','hybrid'),('AnomalyDiffusion',None),('CutMix','cutmix'),('DRAEM-inspired','draem')]):
  x=148+j*280;draw.text((x,104),title,font=font(20,True),fill='#172334')
  if source is None:
   top=bottom=pending();note='Downstream pool pending'
  else:
   d=ROOT/cat/source/f'{selected[source]:02d}';meta=json.loads((d/'provenance.json').read_text());assert meta['replay_bitwise_equal']
   top=Image.open(d/'image.png').convert('RGB');bottom=Image.open(d/'target.png').convert('RGB');assert top.size==bottom.size==(256,256)
   note=f"Index {meta['dataset_index']} | Rank {meta['source_rank']}";category['sources'][source]=meta
   dest=OUT/cat/source;dest.mkdir(parents=True,exist_ok=True)
   for n in ['image.png','target.png','provenance.json']:shutil.copy2(d/n,dest/n)
  image.paste(top,(x,140));image.paste(bottom,(x,460))
  draw.text((x,410),note,font=font(15),fill='#38485d')
  for y in [140,460]:draw.rectangle((x-1,y-1,x+256,y+256),outline='#bdc5d0')
 draw.text((20,755),'Selected from documented V3 runs; no new synthesis seeds, shared-control recomposition, or contrast enhancement.',font=font(17),fill='#38485d')
 p=OUT/f'{cat}_training_comparison.png';image.save(p);figures.append(image);category['figure_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();manifest.append(category)
(OUT/'manifest.json').write_text(json.dumps(dict(examples=manifest,epoch_display=1,epoch_index=0,mask_panels=False),indent=2))
combined=Image.new('RGB',(1280,2400),'white')
for i,image in enumerate(figures):combined.paste(image,(0,i*800))
combined.save(OUT/'three_categories_training_overview.png')
print(OUT)
