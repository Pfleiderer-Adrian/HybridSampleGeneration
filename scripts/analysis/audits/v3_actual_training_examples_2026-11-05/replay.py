"""Replay actual epoch-1 draws using each submitted run's frozen data loader."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

from pathlib import Path
import argparse,csv,json,sqlite3,hashlib
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.quota_dataset import FixedQuotaDataset
from examples.mvtec_ad2.downstream.datasets import HybridPairs
from examples.mvtec_ad2.splits import manifest_samples
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source',choices=['hybrid','cutmix','draem'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.bundle/'manifest.json').read_text())
base=Path('/mnt/results/mvtec2/experiments/all_category_four_methods_20261105');exclude={cat:set() for cat in ['vial','wallplugs','walnuts']}
for cat in exclude:
 for f in [base/cat/'selection.json',*base.glob(f'replaced_*/{cat}/selection.json')]:
  if f.exists():exclude[cat].add(json.loads(f.read_text())['hybrid_id'])
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for cat in ['vial','wallplugs','walnuts']:
 job=manifest['jobs'][cat];run=Path(job['source_run']);config=DownstreamConfiguration.load(run/'configuration.json');split=json.loads((run/'split_manifest.json').read_text())
 history=list(csv.DictReader((run/'training_history.csv').open()));epoch1=next(x for x in history if int(x['epoch'])==1)
 assert int(epoch1[a.source])==400
 healthy=manifest_samples(split,'train',healthy_only=True);real=[s for s in manifest_samples(split,'train') if s.label!='good'];hybrids=None;frozen=split.get('hybrid_provenance',[])
 if a.source=='hybrid':
  assert frozen
  proto=json.loads((Path('/mnt/results/mvtec2/experiments/anomalydiffusion_v3')/cat/'protocol.json').read_text());study=Path(proto['hybrid_study'])
  con=sqlite3.connect(f'file:{study}/artifacts.sqlite?mode=ro',uri=True);con.row_factory=sqlite3.Row
  hs={r['id']:dict(r) for r in con.execute('SELECT * FROM hybrid_samples')};os={r['id']:dict(r) for r in con.execute('SELECT * FROM original_samples')};con.close()
  # Use the historical sampler without re-running today's filtering or changing order.
  hybrids=HybridPairs.__new__(HybridPairs);hybrids.store=ArtifactStore(study);hybrids.data=config.data
  hybrids.pairs=[(SimpleNamespace(**hs[s['hybrid_id']]),SimpleNamespace(**os[s['original_id']])) for s in frozen]
 ds=FixedQuotaDataset(healthy,real,hybrids,config);ds.set_epoch(0)
 ranked=[];code=ds.SOURCES.index(a.source)
 for index,(source,rank) in enumerate(ds.entries):
  if source!=a.source or rank>=160:continue
  rng=np.random.default_rng(np.random.SeedSequence([config.seed,0,code,rank]));chosen=int(rng.integers(len(frozen) if a.source=='hybrid' else len(healthy)))
  provenance=frozen[chosen] if a.source=='hybrid' else None
  if provenance and provenance['hybrid_id'] in exclude[cat]:continue
  sample=ds[index];image,target,mask=sample['image'],sample['target'],sample['mask'];m=mask[0].numpy()>0;area=int(m.sum())
  if not 30<=area<=(4000 if a.source=='hybrid' else 25000):continue
  if a.source=='hybrid':
   ys,xs=np.nonzero(m)
   if min(ys.min(),xs.min())<8 or max(ys.max(),xs.max())>247:continue
  if a.source=='hybrid' and cat=='wallplugs' and float((target.mean(0).numpy()[m]*255>40).mean())<.95:continue
  diff=(image-target).abs().mean(0).numpy()*255;inside=float(diff[m].mean());outside=float(diff[~m].mean()) if (~m).any() else 0
  score=inside*np.log1p(area)-outside*2
  metadata=dict(category=cat,source=a.source,source_run=str(run),submission_bundle=str(a.bundle),seed=config.seed,epoch_index=0,epoch_display=1,dataset_index=index,source_rank=rank,
   selected_checkpoint_epoch=job['epoch'],actual_epoch1_source_count=int(epoch1[a.source]),native_shape=list(image.shape),mask_area=area,inside_mae=inside,outside_mae=outside,score=score,
   hybrid_provenance=provenance,healthy_sample_id=(os[provenance['original_id']]['source_name'] if provenance else healthy[chosen].sample_id),
   donor_sample_id=real[ds.donor_order[rank]].sample_id if a.source=='cutmix' else None)
  ranked.append((score,index,sample,metadata))
 ranked.sort(key=lambda x:x[0],reverse=True)
 selected=[];seen=set()
 for candidate in ranked:
  key=candidate[3]['hybrid_provenance']['hybrid_id'] if a.source=='hybrid' else candidate[3]['healthy_sample_id']
  if key in seen:continue
  seen.add(key);selected.append(candidate)
  if len(selected)==6:break
 assert selected,(cat,a.source)
 dest=a.output/cat/a.source;dest.mkdir(parents=True,exist_ok=True)
 board=Image.new('RGB',(len(selected)*272,580),'white');draw=ImageDraw.Draw(board)
 for j,(_,index,sample,meta) in enumerate(selected,1):
  folder=dest/f'{j:02d}';folder.mkdir(exist_ok=True);repeat=ds[index]
  for key in ['image','target','mask']:
   assert torch.equal(sample[key],repeat[key]),(cat,a.source,index,key)
   value=sample[key].numpy();np.save(folder/f'{key}.npy',value,allow_pickle=False)
   array=np.rint(value*255).clip(0,255).astype('uint8');im=Image.fromarray(array[0] if key=='mask' else array.transpose(1,2,0));im.save(folder/f'{key}.png')
   if key in ['image','target']:board.paste(im.convert('RGB'),((j-1)*272+4,52+(key=='target')*264))
  meta['replay_bitwise_equal']=True
  meta['frozen_source_files']={}
  for module in ['examples.mvtec_ad2.downstream.quota_dataset','examples.mvtec_ad2.downstream.datasets','examples.mvtec_ad2.downstream.cutmix','examples.mvtec_ad2.downstream.draem.synthesis']:
   import importlib
   file=Path(importlib.import_module(module).__file__);meta['frozen_source_files'][module]=dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest());assert a.bundle/'source_snapshot' in file.parents
  meta['file_sha256']={key:hashlib.sha256((folder/f'{key}.png').read_bytes()).hexdigest() for key in ['image','target','mask']}
  (folder/'provenance.json').write_text(json.dumps(meta,indent=2));draw.text(((j-1)*272+4,4),f"{j}: idx {index}, rank {meta['source_rank']}\narea {meta['mask_area']}",font=font,fill='black')
 board.save(dest/'contact_sheet.png');print(cat,a.source,'reconstructed actual epoch1 candidates',len(selected),flush=True)
