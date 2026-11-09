"""Prepare, validate and launch the AD+DRAEM and AD+DRAEM+Hybrid V3 studies."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess

from .mixtures import profile
from .scale2x_mixed import now, read, write, sha


VARIANTS = ('real_anomalydiffusion_draem', 'real_anomalydiffusion_draem_hybrid')
CATEGORIES = ('can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts')
PROJECT = Path(__file__).resolve().parents[2]
STUDY = Path('/mnt/results/mvtec2/experiments/anomalydiffusion_scale2x_50k/20261103')
SOURCE = STUDY / 'real_ad_continuation_20261104'
FROZEN = STUDY / 'real_ad_hybrid_continuation_20261106'
UPDATED = ('experiments/anomalydiffusion/downstream.py',
           'experiments/anomalydiffusion/mixtures.py',
           'experiments/anomalydiffusion/scale2x_mixed.py',
           'experiments/anomalydiffusion/scale2x_combinations.py',
           'examples/mvtec_ad2/downstream/configuration.py',
           'examples/mvtec_ad2/downstream/training.py')


def prepare(output):
    original = read(FROZEN / 'gpu0_worker/plan.json')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    previous = Path(original['source_snapshot'])
    shutil.copytree(previous, snapshot, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.venv', '.tools', '.git'))
    for name in UPDATED:
        shutil.copy2(PROJECT / name, snapshot / name)
    # Model, loss, sampler, DRAEM and inference implementations stay byte-identical.
    for name in ('models.py', 'dinov3.py', 'quota_dataset.py', 'runner.py',
                 'datasets.py', 'evaluation.py', 'draem/synthesis.py'):
        rel = Path('examples/mvtec_ad2/downstream') / name
        assert sha(snapshot / rel) == sha(previous / rel), name
    all_jobs = []
    for gpu, variant in enumerate(VARIANTS):
        worker = output / f'gpu{gpu}_worker'
        (worker / 'jobs').mkdir(parents=True)
        quotas, rotation = profile(variant)
        jobs = []
        for cat in CATEGORIES:
            inputs = output / 'inputs' / variant / cat
            inputs.mkdir(parents=True)
            for name in ('protocol.json', 'split_manifest.json', 'hybrid_provenance.json'):
                shutil.copy2(SOURCE / 'inputs' / cat / name, inputs / name)
            (inputs / 'generated').symlink_to(SOURCE / 'inputs' / cat / 'generated', target_is_directory=True)
            config = read(SOURCE / 'inputs' / cat / 'real_anomalydiffusion.json')
            config['data'].update(source_quotas=quotas, source_quota_rotation=rotation,
                hybrid_fraction=quotas.get('hybrid', 0) / 1000,
                anomalydiffusion_manifest=str(inputs / 'generated/manifest.json'))
            write(inputs / f'{variant}.json', config)
            job = dict(category=cat, variant=variant, gpu=gpu, inputs=str(inputs),
                predecessor_status=str(SOURCE / 'jobs' / cat / 'status.json'),
                input_sha256={str(p): sha(p) for p in inputs.glob('*.json')},
                ad_manifest_sha256=sha(inputs / 'generated/manifest.json'))
            jobs.append(job)
            all_jobs.append(job)
            folder = worker / 'jobs' / cat
            folder.mkdir()
            write(folder / 'status.json', dict(category=cat, gpu=gpu, status='queued', stages=[]))
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip()
        predecessor = SOURCE / 'gpu0_worker' if gpu == 0 else SOURCE
        plan = dict(original, variant=variant, gpus=[gpu], gpu_uuid=uuid, jobs=jobs,
            source_snapshot=str(snapshot), downstream_root=str(output / 'downstream'),
            predecessor=str(predecessor), source_continuation=str(SOURCE),
            source_quotas=quotas, source_quota_rotation=rotation,
            scope='Fresh downstream training and public validation; reuse frozen AD and hybrid pools')
        write(worker / 'plan.json', plan)
        write(worker / 'status.json', dict(status='prepared', gpus=[gpu], categories=list(CATEGORIES)))
    write(output / 'plan.json', dict(created_utc=now(), variants=list(VARIANTS), gpus=[0, 1],
        source_snapshot=str(snapshot), original_source_snapshot=str(previous), jobs=all_jobs,
        scope='16 downstream jobs; 100 epochs each; identical V3 model, losses, splits, seed42 and selection',
        runtime_changes=['Declare new source mixtures', 'Allow AD in existing quota rotation',
                         'Include AD counts in new training CSVs; preserve historical resume schemas']))
    write(output / 'snapshot_sha256.json', {str(p.relative_to(snapshot)): sha(p)
        for p in snapshot.rglob('*.py') if not p.is_symlink()})


def preflight(output):
    from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
    from examples.mvtec_ad2.downstream.anomalydiffusion_pairs import AnomalyDiffusionPairs
    from examples.mvtec_ad2.downstream.draem.synthesis import TextureSynthesizer
    from .scale2x_downstream import sources

    checked_caches = {}
    rows = []
    for job in read(output / 'plan.json')['jobs']:
        inputs = Path(job['inputs'])
        variant = job['variant']
        config = DownstreamConfiguration.load(inputs / f'{variant}.json')
        quotas, rotation = profile(variant)
        assert config.data.source_quotas == quotas and config.data.source_quota_rotation == rotation
        protocol = read(inputs / 'protocol.json')
        reference = DownstreamConfiguration.load(Path(protocol['reference_run']) / 'configuration.json')
        actual, expected = config.to_dict(), reference.to_dict()
        for key in ('source_quotas', 'source_quota_rotation', 'hybrid_fraction', 'anomalydiffusion_manifest'):
            actual['data'].pop(key, None)
            expected['data'].pop(key, None)
        assert actual == expected, (variant, job['category'])
        study, split, frozen, _, _ = sources(inputs)
        assert sha(inputs / 'hybrid_provenance.json') == protocol['hybrid_provenance_sha256']
        assert sha(study / 'configuration.json') == protocol['hybrid_configuration_sha256']
        assert read(job['predecessor_status'])['status'] == 'complete'
        assert sha(inputs / 'generated/manifest.json') == job['ad_manifest_sha256']
        cache_path = (inputs / 'generated/manifest.json').resolve()
        if cache_path not in checked_caches:
            pairs = AnomalyDiffusionPairs(cache_path, split, config.data)
            assert len(pairs) == 1000
            manifest = pairs.manifest
            assert manifest['appearance_checkpoint']['selection'] == 'last'
            assert manifest['appearance_checkpoint']['training_steps'] == 50000
            assert manifest['generation']['aar'] and manifest['generation']['scale'] == 2
            for row in pairs.rows:
                for kind in ('image', 'target', 'mask'):
                    assert sha(Path(row[kind + '_path'])) == row['file_sha256'][kind]
            checked_caches[cache_path] = len(pairs)
        textures = TextureSynthesizer(config.data.texture_root)
        assert len(textures.paths) == 5640
        # Persist all 100 effective epoch budgets, including the rotating extra draw.
        epoch_quotas = []
        for epoch in range(config.training.epochs):
            effective = dict(quotas)
            if rotation:
                total = sum(effective[source] for source in rotation)
                base, remainder = divmod(total, len(rotation))
                for source in rotation:
                    effective[source] = base
                for offset in range(remainder):
                    effective[rotation[(epoch + offset) % len(rotation)]] += 1
            epoch_quotas.append(dict(epoch=epoch + 1, source_quotas=effective))
        write(inputs / 'epoch_quotas.json', epoch_quotas)
        totals = Counter()
        for row in epoch_quotas:
            totals.update(row['source_quotas'])
        rows.append(dict(category=job['category'], variant=variant, configuration_matches_frozen_v3=True,
            split_fingerprint=split['fingerprint'], hybrid_count=len(frozen), ad_count=1000,
            ad_manifest_sha256=job['ad_manifest_sha256'], ad_files_hash_verified=True,
            train_only_provenance=True, dtd_images=len(textures.paths), source_totals_100_epochs=dict(totals)))
        print('PREFLIGHT', variant, job['category'], 'passed', flush=True)
    write(output / 'preflight.json', dict(status='passed', checked_utc=now(), jobs=rows))


def launch(output):
    assert read(output / 'preflight.json')['status'] == 'passed'
    plan = read(output / 'plan.json')
    snapshot = Path(plan['source_snapshot'])
    for name, digest in read(output / 'snapshot_sha256.json').items():
        assert sha(snapshot / name) == digest, name
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True).strip()
    if active:
        raise RuntimeError('GPUs are in use; refusing to overlap new training: ' + active)
    assert all(read(output / f'gpu{gpu}_worker/status.json')['status'] == 'prepared' for gpu in (0, 1))
    workers = []
    for gpu in (0, 1):
        worker = output / f'gpu{gpu}_worker'
        wp = read(worker / 'plan.json')
        env = dict(os.environ, PYTHONPATH=wp['downstream_runtime_libs'] + os.pathsep + str(snapshot),
                   HF_HUB_CACHE=wp['downstream_hf_cache'], HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
        with (worker / 'queue.log').open('w') as log:
            process = subprocess.Popen([wp['downstream_python'], '-m', 'experiments.anomalydiffusion.scale2x_mixed',
                                        'queue', '--output', str(worker)], cwd=snapshot, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        workers.append(dict(gpu=gpu, variant=wp['variant'], worker=str(worker), pid=process.pid))
    pointer = dict(started_utc=now(), output=str(output), workers=workers, jobs=16,
                   preflight=str(output / 'preflight.json'))
    write(output / 'launch.json', pointer)
    write(STUDY / 'latest_ad_draem_downstream_launch.json', pointer)
    print(json.dumps(pointer, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'launch'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if STUDY not in output.parents:
        raise ValueError('Results must remain within the existing scale2x study.')
    {'prepare': prepare, 'preflight': preflight, 'launch': launch}[args.action](output)


if __name__ == '__main__':
    main()
