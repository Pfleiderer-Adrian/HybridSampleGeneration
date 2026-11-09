
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

from pathlib import Path
import json,datetime,os,sys
import numpy as np
from PIL import Image
import torch
from experiments.anomalydiffusion.runtime import load_model
from experiments.anomalydiffusion.run import seed_all
from experiments.anomalydiffusion.prepare import sha256
ROOT=Path('/mnt/results/mvtec2/experiments/all_category_four_methods_20261105');STUDY=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103')
cat=sys.argv[1];d=ROOT/cat
selection=json.loads((STUDY/cat/'centered_2x/training/selection.json').read_text())
assert selection['selection']=='last' and selection['training_steps']==50000 and sha256(selection['checkpoint'])==selection['checkpoint_sha256']
record=dict(pid=os.getpid(),physical_gpu=1,category=cat,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Single qualitative four-method comparison; concurrent inference, energy not exclusive to appearance/production queue')
path=STUDY/f'diagnostic_four_methods_{cat}_20261105.json';path.write_text(json.dumps(record,indent=2))
seed_all(42);torch.set_num_threads(4)
model,_=load_model('appearance',Path('/mnt/results/mvtec2/model_cache/anomalydiffusion/model.ckpt'),d/'ad_work',cat,'maxpool')
state=torch.load(selection['checkpoint'],map_location='cpu');expected={k for k in model.state_dict() if k.startswith('embedding_manager.')};assert set(state['learned_state_dict'])==expected
model.load_state_dict(state['learned_state_dict'],strict=False);del state;model.eval()
image=np.asarray(Image.open(d/'healthy_native.png').convert('RGB').crop((64,64,192,192)).resize((256,256),Image.Resampling.BICUBIC)).astype('float32')/127.5-1
mask=np.asarray(Image.open(d/'hybrid_mask_native.png').convert('L').crop((64,64,192,192)).resize((256,256),Image.Resampling.NEAREST))>0
seed=122000+['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts'].index(cat);seed_all(seed)
batch=dict(image=torch.from_numpy(np.stack([image,image])).cuda(),mask=torch.from_numpy(np.stack([mask,mask]).astype('float32')).cuda(),caption=['a photo of a *']*2,name=[f'{cat}+bad']*2)
with torch.no_grad():logs=model.log_images(batch,N=2,sample=False,inpaint=True,unconditional_only=True,ddim_steps=200,ddim_eta=1.,adaptive_mask=True)
value=logs['samples_inpainting'][0].detach().cpu().permute(1,2,0).numpy();assert np.isfinite(value).all()
Image.fromarray(np.clip((value+1)*127.5,0,255).round().astype('uint8')).save(d/'ad_2x.png')
(d/'ad_metadata.json').write_text(json.dumps(dict(seed=seed,selection=selection,aar=True,scale=2,ddim_steps=200,spatial_encoder_mode='eval',singleton_duplicated=True),indent=2))
record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();path.write_text(json.dumps(record,indent=2));(d/'gpu_interval.json').write_text(json.dumps(record,indent=2))
print('Completed AD',cat,flush=True)
