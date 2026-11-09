
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

from pathlib import Path
import json,shutil,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/mnt/results/mvtec2/experiments/can_four_method_examples_20261105')
OUT=report_dir('can_four_method_examples_2026-11-05');OUT.mkdir(exist_ok=True)
fontdir=Path('/usr/share/fonts/truetype/dejavu')
def font(n,bold=False):return ImageFont.truetype(str(fontdir/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),n)
def core(path,mask=False):
 return Image.open(path).convert('RGB').crop((64,64,192,192)).resize((256,256),Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC)
selection=json.loads((ROOT/'selection.json').read_text());manifest=[]
for c in selection['cases']:
 i=c['index'];d=ROOT/f'case_{i:02d}';local=OUT/f'case_{i:02d}';local.mkdir(exist_ok=True)
 cols=[('Healthy control',core(d/'healthy_native.png'),Image.new('RGB',(256,256),'black')),
       ('Real donor',core(d/'donor_native.png'),core(d/'donor_mask_native.png',True)),
       ('Hybrid',core(d/'hybrid_native.png'),core(d/'hybrid_mask_native.png',True)),
       ('AnomalyDiffusion',Image.open(d/'ad_2x.png').convert('RGB'),core(d/'hybrid_mask_native.png',True)),
       ('CutMix',core(d/'cutmix_native.png'),core(d/'cutmix_mask_native.png',True)),
       ('DRAEM-inspired',core(d/'draem_native.png'),core(d/'draem_mask_native.png',True))]
 im=Image.new('RGB',(1656,750),'#f8f9fb');draw=ImageDraw.Draw(im)
 draw.text((20,15),f"Can | Example {i} | Donor scene {c['donor_group']}",font=font(28,True),fill='#172334')
 draw.text((20,55),'Same healthy control for all synthesis methods | Central native 128 x 128 crop shown at 2x',font=font(19),fill='#38485d')
 draw.text((20,404),'Binary supervision masks (white = anomaly); Hybrid and AD share the same mask.',font=font(18),fill='#38485d')
 for j,(title,image,mask) in enumerate(cols):
  x=20+j*272;draw.text((x,95),title,font=font(20,True),fill='#172334');im.paste(image,(x,128));im.paste(mask,(x,439))
  for y in [128,439]:draw.rectangle((x-1,y-1,x+256,y+256),outline='#bdc5d0')
  name=title.lower().replace(' ','_').replace('-','_');image.save(local/f'{name}.png');mask.save(local/f'{name}_mask.png')
 draw.text((20,715),'AD: Last 50k, AAR on, 2x | CutMix and DRAEM: unchanged V3 synthesis functions | Selected qualitative examples',font=font(18),fill='#38485d')
 p=OUT/f'overview_{i:02d}.png';im.save(p);im.save(ROOT/f'overview_{i:02d}.png')
 meta=dict(case=c,ad=json.loads((d/'ad_metadata.json').read_text()),baselines=json.loads((d/'baselines_metadata.json').read_text()),
           source_folder=str(d),figure=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (local/'metadata.json').write_text(json.dumps(meta,indent=2));manifest.append(meta)
 # Single-sheet native view for checking the visual scale independently of display zoom.
 native=Image.new('RGB',(1656,330),'#f8f9fb');dr=ImageDraw.Draw(native)
 for j,(title,image,mask) in enumerate(cols):
  x=20+j*272;dr.text((x,15),title,font=font(19,True),fill='#172334')
  if j==3:
   v=Image.open(d/'healthy_native.png').convert('RGB');v.paste(image.resize((128,128),Image.Resampling.BICUBIC),(64,64))
  else:v=Image.open(d/(['healthy_native.png','donor_native.png','hybrid_native.png','unused','cutmix_native.png','draem_native.png'][j])).convert('RGB')
  native.paste(v,(x,50))
 native.save(OUT/f'overview_native_{i:02d}.png')
(OUT/'manifest.json').write_text(json.dumps(dict(selection=selection,examples=manifest),indent=2))
text='''# Five Can comparisons of four synthesis methods

Five manually reviewable qualitative figures, selected from the frozen V3 training-derived hybrid pool for visible hybrid changes and five distinct donor capture groups. Same healthy control in each synthesis column. The real donor is shown separately as a reference and is the CutMix donor. The hybrid component is linked to a donor by its nearest recorded placement. These are selected illustrations, not a random sample or aggregate quality evaluation.

Columns: healthy control, real donor, Hybrid, AnomalyDiffusion, defect-aware CutMix, DRAEM-inspired texture synthesis. Main row: central native128 crop enlarged to256 for visibility. Lower row: binary supervision masks, rendered nearest-neighbor. No image contrast, gamma, denoising, masking-based retouch, or artificial visibility enhancement is applied to the panels.

Hybrid: unchanged existing V3 output. AD: same healthy128 window and same final hybrid mask, jointly enlarged2x, coherent last50k checkpoint, AAR on,200DDIM/eta1, spatial encoder eval. No hybrid RGB is passed into AD. CutMix/DRAEM: generated at native256 using unchanged V3 synthesis functions; the central128 crop is displayed. CutMix uses the same real donor scene as the selected hybrid placement;64 seeds retaining >=80% of donor core foreground are ranked for visible labelled defects and less unlabelled donor-background change. DRAEM uses its own sampled DTD texture and Perlin mask; the first seed with a moderate visible defect intersecting the central32 region and <=5% mask on white background is selected. Its defect is not semantically or geometrically identical to the donor anomaly. All selected seeds and source IDs are recorded.

`overview_01.png` through `overview_05.png` are the requested five figures. `overview_native_*.png` provide an optional wider native256 context view; AD's whole decoded128 window is inverse-scaled and inserted in its healthy256 context. Main figures show generated outputs directly, before that inverse scaling. `case_*/` contains all displayed panels and metadata.

Original diagnostic inputs/outputs and logs: `/mnt/results/mvtec2/experiments/can_four_method_examples_20261105`. AD inference ran on physical GPU1 alongside the appearance training; gpu_interval.json records the interval. The predecessor study's `diagnostic_can_four_methods_20261105.json` flags the energy overlap. No study datasets, checkpoints, downstream cache or queue were replaced by this diagnostic.
'''
(OUT/'README.md').write_text(text);(ROOT/'README.md').write_text(text)
# Compact contact sheet for selecting an example without changing the individual figures.
thumbs=[]
for i in range(1,6):
 v=Image.open(OUT/f'overview_{i:02d}.png');v.thumbnail((1104,500));thumbs.append(v)
contact=Image.new('RGB',(1104,500*5),'white')
for i,v in enumerate(thumbs):contact.paste(v,(0,i*500))
contact.save(OUT/'contact_sheet.png')
print(OUT)
