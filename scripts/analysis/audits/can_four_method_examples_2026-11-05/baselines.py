
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from examples.mvtec_ad2.downstream.cutmix import cutmix_defect
from examples.mvtec_ad2.downstream.draem.synthesis import TextureSynthesizer
ROOT=Path('/mnt/results/mvtec2/experiments/can_four_method_examples_20261105');torch.set_num_threads(4)
texture_root=json.loads(Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3/can/real_anomalydiffusion.json').read_text())['data']['texture_root'];synth=TextureSynthesizer(texture_root)
def tensor(p,mask=False):
 a=np.array(Image.open(p).convert('L' if mask else 'RGB'));return torch.from_numpy((a[None] if mask else a.transpose(2,0,1)).copy()).float()/255
for i in range(1,6):
 d=ROOT/f'case_{i:02d}';h=tensor(d/'healthy_native.png');donor=tensor(d/'donor_native.png');dm=tensor(d/'donor_mask_native.png',True)
 core=(slice(None),slice(64,192),slice(64,192));candidates=[]
 for j in range(64):
  seed=91000+i*100+j;image,mask=cutmix_defect(h,donor,dm,np.random.default_rng(seed));m=mask[core][0]>0
  if int(m.sum())<max(30,int(.8*(dm[core][0]>0).sum())):continue
  diff=(image[core]-h[core]).abs().mean(0)*255
  score=float(diff[m].mean())-2*float(diff[~m].mean())
  candidates.append((score,seed,image,mask))
 assert candidates
 _,seed,image,mask=max(candidates,key=lambda r:r[0])
 Image.fromarray((image.permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')).save(d/'cutmix_native.png');Image.fromarray((mask[0].numpy()*255).astype('uint8')).save(d/'cutmix_mask_native.png')
 record=dict(cutmix_seed=seed,cutmix_selection='Best core foreground contrast minus 2x unlabelled-change MAE among 64 seed draws with >=80% donor core mask retained; same training donor; original CutMix function unchanged',draem_texture_root=texture_root)
 for j in range(256):
  seed=101000+i*1000+j;image,mask=synth(h,np.random.default_rng(seed));m=mask[core][0]>0;area=int(m.sum())
  if not 80<=area<=1800 or int(m[48:80,48:80].sum())<10:continue
  diff=(image[core]-h[core]).abs().mean(0)*255
  if float(diff[m].mean())<18:continue
  # Prefer examples on the can rather than its white surrounding background.
  white=(h[core].min(0).values>.94)
  if float(white[m].float().mean())>.05:continue
  break
 else:raise RuntimeError('No suitable visible DRAEM core')
 Image.fromarray((image.permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')).save(d/'draem_native.png');Image.fromarray((mask[0].numpy()*255).astype('uint8')).save(d/'draem_mask_native.png')
 record.update(draem_seed=seed,draem_selection='First seed with visible foreground and moderate nonempty mask intersecting central32 region, <=5% mask on white background; original DRAEM-inspired function unchanged',draem_native_core_area=area)
 (d/'baselines_metadata.json').write_text(json.dumps(record,indent=2));print('Baselines',i,flush=True)
