
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import PROJECT_ROOT

from pathlib import Path
import json,os,subprocess,time,datetime,fcntl
ROOT=Path('/mnt/results/mvtec2/experiments/all_category_four_methods_20261105');STUDY=Path('/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103');PROJECT=PROJECT_ROOT;CATS=['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts']
def write(value):
 p=ROOT/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
with (ROOT/'worker.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(pid=os.getpid(),status='running',started_utc=now(),physical_gpu=1,categories={c:'complete' if (ROOT/c/'ad_2x.png').exists() else 'waiting_for_50k' for c in CATS});failed=set()
 while True:
  pending=[c for c in CATS if not (ROOT/c/'ad_2x.png').exists() and c not in failed]
  if not pending:break
  for cat in pending:
   selection=STUDY/cat/'centered_2x/training/selection.json'
   if not selection.exists():continue
   free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if free<22000:state['stage']='waiting_for_gpu_memory';write(state);continue
   state['stage']='generating';state['categories'][cat]='running';write(state)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LD_LIBRARY_PATH=str(PROJECT/'experiments/anomalydiffusion/.tools/cuda')+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
   with (ROOT/cat/'ad.log').open('w') as log:
    p=subprocess.Popen([str(PROJECT/'experiments/anomalydiffusion/.venv/bin/python'),'-u','-m','scripts.analysis.audits.all_category_four_methods_2026-11-05.generate_ad',cat],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['child_pid']=p.pid;write(state);rc=p.wait()
   state['categories'][cat]='complete' if rc==0 else 'failed'
   if rc:failed.add(cat)
   if all((ROOT/c/'draem_native.png').exists() for c in CATS):
    subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','scripts.analysis.audits.all_category_four_methods_2026-11-05.render'],cwd=PROJECT,check=True)
   write(state)
  state.update(stage='waiting_for_50k',updated_utc=now());write(state)
  if all((ROOT/c/'ad_2x.png').exists() or c in failed for c in CATS):break
  time.sleep(30)
 state.update(status='failed' if failed else 'complete',failed_categories=sorted(failed),finished_utc=now());write(state)
