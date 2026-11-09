"""Prepare and queue Real+AD+Hybrid using the existing scale2x50k AD pools."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

VARIANT = 'real_anomalydiffusion_hybrid'
ASSIGNMENTS = {0: ['can', 'fabric', 'fruit_jelly', 'rice'],
               1: ['sheet_metal', 'vial', 'wallplugs', 'walnuts']}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, output):
    original = read(source/'plan.json')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/'source_snapshot'
    shutil.copytree(original['source_snapshot'], snapshot, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, snapshot/'experiments/anomalydiffusion/scale2x_mixed.py')
    alljobs = []
    for gpu, categories in ASSIGNMENTS.items():
        worker = output/f'gpu{gpu}_worker'
        worker.mkdir(); (worker/'jobs').mkdir()
        jobs = []
        predecessor = source/'gpu0_worker' if gpu == 0 else source
        for cat in categories:
            inputs = output/'inputs'/cat; inputs.mkdir(parents=True)
            for name in ['protocol.json', 'split_manifest.json', 'hybrid_provenance.json']:
                shutil.copy2(source/'inputs'/cat/name, inputs/name)
            # This symlink also becomes valid when a still-running pool finishes.
            (inputs/'generated').symlink_to(source/'inputs'/cat/'generated', target_is_directory=True)
            config = read(source/'inputs'/cat/'real_anomalydiffusion.json')
            config['data'].update(source_quotas={'normal':500, 'real':100,
                'anomalydiffusion':200, 'hybrid':200}, hybrid_fraction=0.2,
                anomalydiffusion_manifest=str(inputs/'generated/manifest.json'))
            write(inputs/f'{VARIANT}.json', config)
            job = dict(category=cat, gpu=gpu, inputs=str(inputs),
                predecessor_status=str(source/'jobs'/cat/'status.json'),
                input_sha256={str(p):sha(p) for p in inputs.glob('*.json')})
            jobs.append(job); alljobs.append(job)
            folder=worker/'jobs'/cat; folder.mkdir()
            write(folder/'status.json',dict(category=cat,gpu=gpu,status='queued',stages=[]))
        plan = dict(original, gpus=[gpu], jobs=jobs, variant=VARIANT,
                    source_snapshot=str(snapshot), downstream_root=str(output/'downstream'),
                    predecessor=str(predecessor), source_continuation=str(source),
                    scope='Downstream training and public validation only; reuse AD and frozen hybrid pools')
        write(worker/'plan.json', plan)
        write(worker/'status.json', dict(status='prepared',gpus=[gpu],categories=categories))
    write(output/'plan.json',dict(created_utc=now(),variant=VARIANT,source_continuation=str(source),
        source_snapshot=str(snapshot),gpus=[0,1],assignments=ASSIGNMENTS,jobs=alljobs,
        source_quotas={'normal':500,'real':100,'anomalydiffusion':200,'hybrid':200}))
    write(output/'snapshot_sha256.json',{str(p.relative_to(snapshot)):sha(p)
        for p in snapshot.rglob('*.py') if not p.is_symlink()})


def preflight(output):
    """Load configurations with the exact frozen downstream runtime, on CPU."""
    from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
    from .scale2x_downstream import sources
    rows=[]
    for job in read(output/'plan.json')['jobs']:
        inputs=Path(job['inputs']); protocol=read(inputs/'protocol.json')
        actual=DownstreamConfiguration.load(inputs/f'{VARIANT}.json').to_dict()
        expected=DownstreamConfiguration.load(Path(protocol['reference_run'])/'configuration.json').to_dict()
        assert actual['data']['source_quotas']=={'normal':500,'real':100,'anomalydiffusion':200,'hybrid':200}
        for key in ['source_quotas','source_quota_rotation','hybrid_fraction','anomalydiffusion_manifest']:
            actual['data'].pop(key,None); expected['data'].pop(key,None)
        assert actual==expected, job['category']
        study,split,frozen,_,_=sources(inputs)
        assert sha(inputs/'hybrid_provenance.json')==protocol['hybrid_provenance_sha256']
        assert sha(study/'configuration.json')==protocol['hybrid_configuration_sha256']
        rows.append(dict(category=job['category'],split_fingerprint=split['fingerprint'],
            hybrid_count=len(frozen),configuration_matches_frozen_v3=True,train_only_provenance=True))
    write(output/'preflight.json',dict(status='passed',checked_utc=now(),categories=rows))


def queue(worker):
    plan=read(worker/'plan.json'); gpu,=plan['gpus']; predecessor=Path(plan['predecessor'])
    variant=plan.get('variant',VARIANT)
    state=dict(pid=os.getpid(),status='running',stage='waiting_for_real_ad',started_utc=now(),
               gpus=[gpu],categories=[j['category'] for j in plan['jobs']])
    write(worker/'status.json',state)
    # Own worker lock is acquired by main. Hold predecessor lock throughout our
    # work so duplicate predecessor launches cannot race with the same GPU.
    with (predecessor/'queue.lock').open('a') as gate:
        while True:
            try:
                fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError:
                state['updated_utc']=now();write(worker/'status.json',state);time.sleep(30)
        if read(predecessor/'status.json')['status'] not in ['complete','failed']:
            raise RuntimeError('Predecessor ended without terminal status.')
        snapshot=Path(plan['source_snapshot']);telemetry=worker/'telemetry';telemetry.mkdir(exist_ok=True)
        with (telemetry/'monitor.log').open('w') as log:
            monitor=subprocess.Popen([plan['downstream_python'],'-m','experiments.anomalydiffusion.telemetry',
                '--launch',str(worker),'--interval','5'],cwd=snapshot,
                env=dict(os.environ,PYTHONPATH=str(snapshot)),stdout=log,stderr=subprocess.STDOUT)
        state['telemetry_pid']=monitor.pid
        failed=[]
        for job in plan['jobs']:
            cat=job['category'];folder=worker/'jobs'/cat;inputs=Path(job['inputs'])
            js=dict(category=cat,gpu=gpu,pid=os.getpid(),status='running',stages=[])
            try:
                state.update(stage='processing',category=cat);write(worker/'status.json',state)
                if read(job['predecessor_status'])['status']!='complete':
                    raise RuntimeError('Matching Real+AD job not complete.')
                for path,digest in job['input_sha256'].items():
                    if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
                if job.get('ad_manifest_sha256') and sha(inputs/'generated/manifest.json')!=job['ad_manifest_sha256']:
                    raise ValueError('Frozen AD manifest changed.')
                manifest=read(inputs/'generated/manifest.json')
                if len(manifest['rows'])!=1000:raise ValueError('AD pool incomplete.')
                if manifest['split_fingerprint']!=read(inputs/'split_manifest.json')['fingerprint']:
                    raise ValueError('AD split mismatch.')
                selection=manifest['appearance_checkpoint']
                if selection['selection']!='last' or selection['training_steps']!=50000:
                    raise ValueError('Expected Last50k appearance checkpoint.')
                if not manifest['generation']['aar'] or manifest['generation']['scale']!=2:
                    raise ValueError('Expected 2x/AAR generation.')
                dest=Path(plan['downstream_root'])/variant/cat
                if dest.exists():raise FileExistsError('Refusing to overwrite existing run: '+str(dest))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan.get('gpu_uuid',str(gpu)),PYTHONUNBUFFERED='1',
                    PYTHONPATH=plan['downstream_runtime_libs']+os.pathsep+str(snapshot),
                    HF_HUB_CACHE=plan['downstream_hf_cache'],HF_HUB_OFFLINE='1',
                    OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
                cmd=[plan['downstream_python'],'-m','experiments.anomalydiffusion.downstream',
                    '--inputs',str(inputs),'--variant',variant,'--output',str(dest)]
                entry=dict(stage=variant,status='running',started_utc=now(),command=cmd,log=str(folder/'downstream.log'))
                js.update(stage=variant,stages=[entry])
                with (folder/'downstream.log').open('w') as log:
                    child=subprocess.Popen(cmd,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT)
                    entry['pid']=child.pid;write(folder/'status.json',js);rc=child.wait()
                entry.update(status='complete' if rc==0 else 'failed',returncode=rc,finished_utc=now())
                if rc:raise RuntimeError(f'Downstream exited with {rc}')
                runs=list((dest/'downstream/dinov3_paired').glob('*'))
                if len(runs)!=1 or not (runs[0]/'metrics.json').is_file() or not (runs[0]/'checkpoints/best.pt').is_file():
                    raise RuntimeError('Completion artifacts missing.')
                entry['downstream_run']=str(runs[0]);js.update(status='complete',finished_utc=now())
            except Exception as error:
                js.update(status='failed',error=str(error),traceback=traceback.format_exc());failed.append(cat)
            write(folder/'status.json',js)
        state.update(status='failed' if failed else 'complete',stage='finished',
            failed_categories=failed,finished_utc=now());write(worker/'status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','queue'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if Path('/mnt/results') not in output.parents:raise ValueError('Results must remain under /mnt/results.')
    if args.action=='prepare':prepare(args.source.resolve(),output)
    elif args.action=='preflight':preflight(output)
    else:
        with (output/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:queue(output)
            except Exception as error:
                state=read(output/'status.json');state.update(status='failed',error=str(error),traceback=traceback.format_exc())
                write(output/'status.json',state);raise


if __name__=='__main__':main()
