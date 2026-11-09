"""Adopt existing stages without interrupting them; launch all future work on GPU1."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from .queue import atomic_json, check_inputs, now, stage_commands


def process_identity(pid):
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def same_process(item):
    identity = process_identity(item['pid'])
    return identity is not None and identity == item['start_ticks']


def verify_output(plan,job,name,artifact,environment):
    if artifact is not None:
        if not artifact.is_file():raise RuntimeError(f'Missing completion artifact: {artifact}')
        return {}
    runs=sorted((Path(plan['downstream_root'])/name/job['category']/'downstream').glob('dinov3_paired/*'))
    if len(runs)!=1 or not (runs[0]/'metrics.json').is_file() or not (runs[0]/'checkpoints/best.pt').is_file():
        raise RuntimeError(f'Incomplete downstream output for {job["category"]}/{name}')
    return {'downstream_run':str(runs[0])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    launch=args.plan.resolve().parent;plan=json.loads(args.plan.read_text())
    if plan['gpus'] != [1]:raise ValueError('This continuation is restricted to GPU1.')
    with (launch/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),status='running',started_utc=now(),gpus=[1],categories=[j['category'] for j in plan['jobs']],
            previous_launch=plan['previous_launch'],gpu_policy='All new stages GPU1; previously running GPU0 generation allowed to finish')
        atomic_json(launch/'status.json',state)
        adopted={str(x['pid']):x for x in plan['adopted_processes']}
        for job in plan['jobs']:
            folder=launch/'jobs'/job['category'];folder.mkdir(parents=True,exist_ok=True)
            path=folder/'status.json'
            current=json.loads(path.read_text()) if path.exists() else dict(category=job['category'],gpu=1,status='running',started_utc=now(),stages=[])
            if current.get('status')=='complete':continue
            if current.get('status')=='failed':raise RuntimeError(f'Refuse to retry previously failed job silently: {job["category"]}')
            current['pid']=os.getpid();current['status']='running';atomic_json(path,current)
            try:
                for name,command,artifact,environment in stage_commands(plan,job):
                    check_inputs(job)
                    prior=next((s for s in current['stages'] if s['stage']==name),None)
                    if prior and prior['status']=='complete':
                        verify_output(plan,job,name,artifact,environment);continue
                    if prior and prior['status']=='running':
                        item=adopted.get(str(prior['pid']))
                        if item is None:raise RuntimeError('Unrecognized pre-existing stage; refusing duplicate launch.')
                        print(now(),'adopting',job['category'],name,item,flush=True)
                        while same_process(item):time.sleep(5)
                        prior.update(verify_output(plan,job,name,artifact,environment))
                        prior.update(status='complete',finished_utc=now(),adopted=True,returncode=None,
                            completion_verification='Completion artifact verified after original process exited; exit code unavailable')
                        current['updated_utc']=now();atomic_json(path,current);continue
                    # A different category may still own GPU1 from the old queue.
                    while any(x['gpu']==1 and same_process(x) for x in adopted.values()):
                        current['waiting_for']='existing GPU1 stage to finish';atomic_json(path,current);time.sleep(5)
                    current.pop('waiting_for',None)
                    current.update(gpu=1,stage=name,updated_utc=now())
                    entry=dict(stage=name,status='running',gpu=1,started_utc=now(),command=command,log=str(folder/f'{name}.log'))
                    current['stages'].append(entry);atomic_json(path,current)
                    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
                    paths=[plan['source_snapshot']]
                    if environment=='downstream':
                        paths.insert(0,plan['downstream_runtime_libs']);env.update(HF_HUB_CACHE=plan['downstream_hf_cache'],HF_HUB_OFFLINE='1')
                    env['PYTHONPATH']=os.pathsep.join(paths)
                    with Path(entry['log']).open('w') as log:
                        child=subprocess.Popen(command,cwd=plan['source_snapshot'],env=env,stdout=log,stderr=subprocess.STDOUT)
                        entry['pid']=child.pid;atomic_json(path,current);rc=child.wait()
                    entry.update(returncode=rc,finished_utc=now())
                    if rc:raise RuntimeError(f'{name} failed with exit code{rc}')
                    entry.update(verify_output(plan,job,name,artifact,environment));entry['status']='complete';atomic_json(path,current)
                current.update(status='complete',finished_utc=now());atomic_json(path,current)
            except Exception as error:
                current.update(status='failed',error=str(error),traceback=traceback.format_exc(),finished_utc=now());atomic_json(path,current)
                state.update(status='failed',error=str(error),finished_utc=now());atomic_json(launch/'status.json',state);raise
        state.update(status='complete',finished_utc=now());atomic_json(launch/'status.json',state)


if __name__=='__main__':main()
