
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

from pathlib import Path
import json,sqlite3,csv,collections,hashlib,re
base=Path('/mnt/results/mvtec2/experiments');out=report_dir('studieninventar_2026-10-26');paths=[Path(x) for x in Path('/tmp/study_inventory_files_current.txt').read_text().splitlines()];trials={};dbs=[];errors=[]
def root(p):return p.relative_to(base).parts[0]
def js(p):return json.load(open(p))
# Dedupe copies by Optuna start time, trial number and parameter distributions.
for p in [p for p in paths if p.suffix=='.db']:
 try:
  c=sqlite3.connect(f'file:{p}?mode=ro',uri=True);c.row_factory=sqlite3.Row
  if not c.execute("select 1 from sqlite_master where name='trials'").fetchone():continue
  cfg=js(p.parent/'configuration.json') if (p.parent/'configuration.json').exists() else {};keys=[]
  for t in c.execute('select * from trials').fetchall():
   pars=[tuple(r) for r in c.execute('select param_name,param_value,distribution_json from trial_params where trial_id=? order by param_name',(t['trial_id'],))];attrs={r[0]:json.loads(r[1]) for r in c.execute('select key,value_json from trial_user_attributes where trial_id=?',(t['trial_id'],))}
   key=hashlib.sha256(json.dumps([t['datetime_start'],t['number'],pars],sort_keys=True).encode()).hexdigest();keys.append(key);item={'id':key,'number':t['number'],'state':t['state'],'start':t['datetime_start'],'complete':t['datetime_complete'],'model':attrs.get('model_name',cfg.get('model',{}).get('name','unknown')),'best_epoch':attrs.get('best_epoch'),'checkpoint_recorded':bool(attrs.get('model_path')),'databases':[str(p)],'roots':[root(p)],'model_path':attrs.get('model_path')}
   if key not in trials:trials[key]=item
   else:
    prior=trials[key];prior['databases'].append(str(p));prior['roots']=sorted(set(prior['roots']+[root(p)]))
    if item['state']=='COMPLETE' and prior['state']!='COMPLETE':item['databases']=prior['databases'];item['roots']=prior['roots'];trials[key]=item
  dbs.append({'path':str(p),'root':root(p),'trial_ids':keys})
 except Exception as e:errors.append({'path':str(p),'error':str(e)})
# Connected database copies form a single Optuna training lineage.
parents=list(range(len(dbs)))
def find(i):
 while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
 return i
owner={}
for i,d in enumerate(dbs):
 for k in d['trial_ids']:
  if k in owner:parents[find(i)]=find(owner[k])
  else:owner[k]=i
lineages=collections.defaultdict(list)
for i,d in enumerate(dbs):lineages[find(i)].append(d)
generators=[]
for ds in lineages.values():
 ks=set(k for d in ds for k in d['trial_ids']);ts=[trials[k] for k in ks];canonical=min(ds,key=lambda d:(len(Path(d['path']).parts),d['path']))
 # Attribute the lineage to the least nested surviving original database.
 generators.append({'root':canonical['root'],'databases':[d['path'] for d in ds],'trials':len(ks),'states':dict(collections.Counter(t['state'] for t in ts)),'models':sorted({str(t['model']) for t in ts}),'checkpoint_records':sum(t['checkpoint_recorded'] for t in ts),'first_start':min((t['start'] for t in ts if t['start']),default=None)})
# Training runs have their own history; ID/inode removes mirrored copies.
all_histories=[p for p in paths if p.name=='training_history.csv'];backup_parts={'validation_interval_changes','migration_backup','source_snapshot','preprocessing_backups','reprocessing_backups'};histories=[p for p in all_histories if not (set(p.parts)&backup_parts)];excluded_history_backups=[str(p) for p in all_histories if set(p.parts)&backup_parts];seen={};downstream=[];copies=[]
for p in histories:
 try:
  run=p.parent;cfg=js(run/'configuration.json');rows=list(csv.DictReader(open(p)));validrows=[x for x in rows if x.get('epoch')];n=max([int(x['epoch']) for x in validrows],default=0);ck=run/'checkpoints/best.pt';inode=(p.stat().st_dev,p.stat().st_ino);match=re.fullmatch(r'\d{8}T\d{6}Z_[0-9a-f]+',run.name);identity=('run',run.name) if match else ('inode',inode)
  if identity in seen:copies.append({'path':str(run),'original':seen[identity]});continue
  seen[identity]=str(run);model=cfg.get('model','draem');dino=cfg.get('dinov3',{});hasbest=ck.exists();hastlast=(run/'checkpoints/last.pt').exists();metrics=(run/'metrics.json').exists();epochlimit=cfg.get('training',{}).get('epochs',100);trained=n>0
  downstream.append({'run':str(run),'root':root(p),'model':model,'backbone':dino.get('backbone'),'backbone_type':dino.get('backbone_type','vit') if model.startswith('dinov3') else None,'decoder':dino.get('decoder','cnn') if model.startswith('dinov3') else None,'epochs_recorded':n,'epochs_configured':epochlimit,'finished_epochs':n>=epochlimit,'has_best_checkpoint':hasbest,'has_last_checkpoint':hastlast,'has_metrics':metrics,'training_started':trained,'history_rows':len(validrows),'history_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 except Exception as e:errors.append({'path':str(p),'error':str(e)})
# Non-epoch fitted memory-bank runs are counted separately.
for p in [Path(x) for x in Path('/tmp/study_checkpoint_files.txt').read_text().splitlines() if x.endswith('/checkpoints/best.pt')]:
 run=p.parent.parent
 if set(run.parts)&backup_parts:continue
 if any(d['run']==str(run) for d in downstream):continue
 try:
  cfg=js(run/'configuration.json');model=cfg.get('model','draem')
  if model=='superadd_dual_bank':downstream.append({'run':str(run),'root':root(p),'model':model,'backbone':cfg.get('dinov3',{}).get('backbone'),'epochs_recorded':None,'epochs_configured':None,'finished_epochs':None,'has_best_checkpoint':True,'has_metrics':(run/'metrics.json').exists(),'training_started':True,'fit_without_epochs':True})
  else:errors.append({'path':str(run),'error':'checkpoint without recorded training history; not counted as training'})
 except Exception as e:errors.append({'path':str(run),'error':str(e)})
started=[d for d in downstream if d['training_started']];complete=[d for d in started if d['finished_epochs'] or d.get('fit_without_epochs')];byroot={}
for r in sorted({d['root'] for d in dbs}|{d['root'] for d in downstream}):
 gs=[g for g in generators if g['root']==r];ds=[d for d in started if d['root']==r];byroot[r]={'generator_studies':len(gs),'generator_trials_complete':sum(g['states'].get('COMPLETE',0) for g in gs),'generator_trials_all':sum(g['trials'] for g in gs),'downstream_started':len(ds),'downstream_full_epochs_or_bank':sum(bool(d['finished_epochs'] or d.get('fit_without_epochs')) for d in ds),'downstream_models':dict(collections.Counter(d['model'] for d in ds))}
summary={'methodology':'Surviving experiment records only; Optuna copies deduped by start timestamp+trial number+params; history copies by dated run ID or inode. Epoch-trained runs need at least1recorded epoch; fitted memory banks with best checkpoint counted separately. Submission copies outside experiments excluded. Full epochs does not necessarily imply completed final evaluation.','generator_studies_with_completed_trials':sum(g['states'].get('COMPLETE',0)>0 for g in generators),'generator_databases':len(dbs),'unique_generator_training_studies':sum(g['trials']>0 for g in generators),'unique_generator_trials':len(trials),'generator_trial_states':dict(collections.Counter(t['state'] for t in trials.values())),'generator_checkpoint_records':sum(t['checkpoint_recorded'] for t in trials.values()),'downstream_history_files':len(all_histories),'downstream_history_backups_excluded':excluded_history_backups,'downstream_copies_removed':len(copies),'downstream_started':len(started),'downstream_full_epochs_or_bank':len(complete),'downstream_partial':len(started)-len(complete),'downstream_zero_epoch_records':sum(not d['training_started'] for d in downstream),'downstream_models':dict(collections.Counter(d['model'] for d in started)),'dinov3_backbone_types':dict(collections.Counter(d.get('backbone_type') for d in started if d['model'].startswith('dinov3'))),'dinov3_decoders':dict(collections.Counter(d.get('decoder') for d in started if d['model'].startswith('dinov3'))),'by_root':byroot,'errors':errors}
(out/'inventory.json').write_text(json.dumps({'summary':summary,'generator_studies':generators,'generator_trials':list(trials.values()),'downstream':downstream,'duplicate_histories':copies,'excluded_history_backups':excluded_history_backups},indent=2));print(json.dumps(summary,indent=2))
