
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

from pathlib import Path
import json,hashlib,fcntl
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/mnt/results/mvtec2/experiments/all_category_four_methods_20261105');OUT=report_dir('all_category_four_methods_2026-11-05');OUT.mkdir(exist_ok=True)
_render_lock=(OUT/'render.lock').open('a');fcntl.flock(_render_lock,fcntl.LOCK_EX)
CATS=['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts'];names={'can':'Can','fabric':'Fabric','fruit_jelly':'Fruit jelly','rice':'Rice','sheet_metal':'Sheet metal','vial':'Vial','wallplugs':'Wall plugs','walnuts':'Walnuts'}
fontdir=Path('/usr/share/fonts/truetype/dejavu')
def font(n,bold=False):return ImageFont.truetype(str(fontdir/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),n)
def core(p):return Image.open(p).convert('RGB').crop((64,64,192,192)).resize((256,256),Image.Resampling.BICUBIC)
def placeholder():
 im=Image.new('RGB',(256,256),'#e8edf2');dr=ImageDraw.Draw(im);dr.text((28,99),'AD 50k pending',font=font(22,True),fill='#5c6b7b');dr.text((26,137),'Auto-update queued',font=font(18),fill='#5c6b7b');return im
manifest=[];overview=Image.new('RGB',(1656,8*325+100),'#f8f9fb');od=ImageDraw.Draw(overview)
headers=['Healthy control','Real donor','Hybrid','AnomalyDiffusion','CutMix','DRAEM-inspired']
for j,title in enumerate(headers):od.text((20+j*272,23),title,font=font(20,True),fill='#172334')
for ci,cat in enumerate(CATS):
 d=ROOT/cat
 if not (d/'draem_native.png').exists():continue
 ready=(d/'ad_2x.png').exists();panels=[core(d/'healthy_native.png'),core(d/'donor_native.png'),core(d/'hybrid_native.png'),Image.open(d/'ad_2x.png').convert('RGB') if ready else placeholder(),core(d/'cutmix_native.png'),core(d/'draem_native.png')]
 im=Image.new('RGB',(1656,450),'#f8f9fb');draw=ImageDraw.Draw(im)
 draw.text((20,15),names[cat]+' | Four anomaly synthesis methods',font=font(28,True),fill='#172334')
 draw.text((20,55),'Shared healthy control | Native 128 x 128 detail shown at 2x | Selected qualitative example',font=font(19),fill='#38485d')
 for j,(title,value) in enumerate(zip(headers,panels)):
  x=20+j*272;draw.text((x,94),title,font=font(20,True),fill='#172334');im.paste(value,(x,125));draw.rectangle((x-1,124,x+256,381),outline='#bdc5d0')
  overview.paste(value,(x,100+ci*325));od.rectangle((x-1,99+ci*325,x+256,356+ci*325),outline='#bdc5d0')
 od.text((20,73+ci*325),names[cat],font=font(20,True),fill='#172334')
 draw.text((20,405),'AD: Last50k, AAR on, 2x | Hybrid/AD share conditioning mask; CutMix/DRAEM use their own labels.',font=font(18),fill='#38485d')
 p=OUT/f'{cat}_comparison.png';tmp=p.with_suffix('.tmp.png');im.save(tmp);tmp.replace(p);im.save(ROOT/f'{cat}_comparison.png')
 manifest.append(dict(category=cat,ad_ready=ready,selection=json.loads((d/'selection.json').read_text()),baselines=json.loads((d/'baselines_metadata.json').read_text()),ad=json.loads((d/'ad_metadata.json').read_text()) if ready else None,source_folder=str(d),figure=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 # Keep displayed image panels in the report folder, without exporting masks.
 local=OUT/cat;local.mkdir(exist_ok=True)
 for title,value in zip(headers,panels):value.save(local/(title.lower().replace(' ','_').replace('-','_')+'.png'))
 tmp=overview  # overview saved once after all rows
p=OUT/'all_categories_overview.png';tmp=p.with_suffix('.tmp.png');overview.save(tmp);tmp.replace(p)
(OUT/'manifest.json').write_text(json.dumps(dict(examples=manifest,mask_rows=False),indent=2))
print('Rendered',len(manifest),'categories;',sum(r['ad_ready'] for r in manifest),'AD panels ready',flush=True)
