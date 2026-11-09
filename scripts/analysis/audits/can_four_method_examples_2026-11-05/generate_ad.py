
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

from pathlib import Path
import json,time,datetime,os
import numpy as np
from PIL import Image
import torch
from experiments.anomalydiffusion.runtime import load_model
from experiments.anomalydiffusion.run import seed_all
from experiments.anomalydiffusion.prepare import sha256
ROOT=Path('/mnt/results/mvtec2/experiments/can_four_method_examples_20261105')
STUDY=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103')
record=dict(pid=os.getpid(),physical_gpu=1,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Five selected qualitative can comparisons; concurrent diagnostic inference, not exclusive appearance-training energy')
(ROOT/'gpu_interval.json').write_text(json.dumps(record,indent=2))
(STUDY/'diagnostic_can_four_methods_20261105.json').write_text(json.dumps(dict(record,results=str(ROOT)),indent=2))
selection=json.loads((STUDY/'can/centered_2x/training/selection.json').read_text())
assert selection['selection']=='last' and selection['training_steps']==50000
assert sha256(selection['checkpoint'])==selection['checkpoint_sha256']
seed_all(42);torch.set_num_threads(4)
model,_=load_model('appearance',Path('/mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt'),ROOT/'ad_work','can','maxpool')
state=torch.load(selection['checkpoint'],map_location='cpu')
expected={k for k in model.state_dict() if k.startswith('embedding_manager.')}
assert set(state['learned_state_dict'])==expected
model.load_state_dict(state['learned_state_dict'],strict=False);del state;model.eval()
for start in range(1,6,2):
 ids=list(range(start,min(6,start+2)));actual=len(ids)
 if actual==1:ids*=2
 images=[];masks=[]
 for i in ids:
  d=ROOT/f'case_{i:02d}'
  image=Image.open(d/'healthy_native.png').crop((64,64,192,192)).resize((256,256),Image.Resampling.BICUBIC)
  mask=Image.open(d/'hybrid_mask_native.png').crop((64,64,192,192)).resize((256,256),Image.Resampling.NEAREST)
  images.append(np.asarray(image).astype('float32')/127.5-1);masks.append(np.asarray(mask)>0)
 seed=82000+start;seed_all(seed)
 batch=dict(image=torch.from_numpy(np.stack(images)).cuda(),mask=torch.from_numpy(np.stack(masks).astype('float32')).cuda(),caption=['a photo of a *']*2,name=['can+bad']*2)
 with torch.no_grad():logs=model.log_images(batch,N=2,sample=False,inpaint=True,unconditional_only=True,ddim_steps=200,ddim_eta=1.,adaptive_mask=True)
 for j,i in enumerate(ids[:actual]):
  value=logs['samples_inpainting'][j].detach().cpu().permute(1,2,0).numpy();assert np.isfinite(value).all()
  Image.fromarray(np.clip((value+1)*127.5,0,255).round().astype('uint8')).save(ROOT/f'case_{i:02d}/ad_2x.png')
  (ROOT/f'case_{i:02d}/ad_metadata.json').write_text(json.dumps(dict(seed=seed,selection=selection,aar=True,scale=2,ddim_steps=200,spatial_encoder_mode='eval'),indent=2))
 print('Completed AD examples',ids[:actual],flush=True)
record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(ROOT/'gpu_interval.json').write_text(json.dumps(record,indent=2))
(STUDY/'diagnostic_can_four_methods_20261105.json').write_text(json.dumps(dict(record,results=str(ROOT)),indent=2))
