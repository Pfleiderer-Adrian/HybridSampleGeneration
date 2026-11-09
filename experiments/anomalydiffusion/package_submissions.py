"""Freeze the reviewed AD runs and build both MVTec AD 2 submission archives."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

import torch

from .mixtures import profile


REPO = Path(__file__).resolve().parents[2]
BASE = Path('/mnt/results/mvtec2')
STUDY = BASE / 'experiments/anomalydiffusion_scale2x_50k/20261103'
TEMPLATE = BASE / 'submissions/v3_variant2_best_20261025T144555Z'
COUNTS = dict(can=321, fabric=314, fruit_jelly=255, rice=277,
              sheet_metal=142, vial=276, wallplugs=232, walnuts=228)
VARIANTS = [
    ('real_anomalydiffusion_hybrid', 'real_ad_hybrid_continuation_20261106', 0),
    ('real_anomalydiffusion', 'real_ad_continuation_20261104', 1),
]
VARIANT_QUEUES = {variant: queue for variant, queue, _ in VARIANTS}
VARIANT_QUEUES.update({
    'real_anomalydiffusion_draem': 'real_ad_draem_combinations_20261108',
    'real_anomalydiffusion_draem_hybrid': 'real_ad_draem_combinations_20261108',
})


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(variant, queue, gpu, timestamp):
    root = STUDY / queue
    matching_plans = [p for p in sorted(root.glob('gpu*_worker/plan.json'))
                      if json.loads(p.read_text()).get('variant') == variant]
    assert matching_plans, ('No matching frozen training plan', variant)
    plan_path = matching_plans[0]
    plan = json.loads(plan_path.read_text())
    snapshot = Path(plan['source_snapshot'])
    bundle = BASE / 'submissions' / f'v3_{variant}_scale2x50k_{timestamp}'
    bundle.mkdir()
    jobs = {}
    expected_quotas, expected_rotation = profile(variant)
    for category, count in COUNTS.items():
        matches = list((root / 'downstream' / variant / category).glob('downstream/dinov3_paired/*/metrics.json'))
        assert len(matches) == 1, (category, matches)
        run = matches[0].parent
        config = json.loads((run / 'configuration.json').read_text())
        score = json.loads(matches[0].read_text())['validation']['pixel_aupro_0.05']
        assert config['data']['source_quotas'] == expected_quotas
        assert config['data'].get('source_quota_rotation') == expected_rotation
        assert config['model'] == 'dinov3_paired'
        assert config['dinov3']['backbone_type'] == 'vit'
        assert config['dinov3']['decoder'] == 'cnn'
        with (run / 'training_history.csv').open() as stream:
            history = list(csv.DictReader(stream))
        assert [int(row['epoch']) for row in history] == list(range(1, 101))
        checkpoint = run / 'checkpoints/best.pt'
        ck = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        assert ck['selection_metric'] == 'pixel_aupro_0.05' and ck['selection_mode'] == 'max'
        assert abs(ck['selection_score'] - score) < 1e-10
        epoch = int(ck['epoch'])
        del ck
        split = json.loads((run / 'split_manifest.json').read_text())
        assert not split['partitions'].get('test')
        assert all('private' not in sample['image_path'].lower()
                   for rows in split['partitions'].values() for sample in rows)
        assert (run / 'predictions/validation/index.csv').is_file()
        for partition, suffix in [('test_private', 'regular'), ('test_private_mixed', 'mixed')]:
            files = {p.name for p in (BASE / 'mvtec_ad_2' / category / partition).glob('*.png')}
            assert files == {f'{i:03d}_{suffix}.png' for i in range(count)}
        frozen = bundle / 'models' / category
        (frozen / 'checkpoints').mkdir(parents=True)
        (bundle / 'jobs' / category).mkdir(parents=True)
        for name in ['configuration.json', 'split_manifest.json', 'metrics.json',
                     'training_history.csv', 'real_training_sources.json',
                     'anomalydiffusion_provenance.json']:
            shutil.copy2(run / name, frozen / name)
        shutil.copy2(checkpoint, frozen / 'checkpoints/best.pt')
        digest = sha(checkpoint)
        assert sha(frozen / 'checkpoints/best.pt') == digest
        inputs = root / 'inputs' / variant / category
        if not inputs.is_dir():
            inputs = root / 'inputs' / category
        shutil.copy2(inputs / 'protocol.json', frozen / 'generation_protocol.json')
        jobs[category] = dict(run=str(frozen), source_run=str(run), checkpoint='best.pt',
                              checkpoint_sha256=digest, epoch=epoch, configuration=config,
                              data_root=str(BASE / 'mvtec_ad_2'), total=2 * count,
                              public_validation_aupro=score, variant=variant)
        save(bundle / 'jobs' / category / 'status.json', dict(stage='queued', completed=0, total=2*count))
    shutil.copytree(snapshot, bundle / 'source_snapshot', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.venv', '.tools', '.git'))
    shutil.copytree(Path(plan['downstream_runtime_libs']), bundle / 'runtime_libs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(TEMPLATE / 'official_utils', bundle / 'official_utils',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # TIFF/PNG members are already compressed. Keep the official validation
    # unchanged and use faster outer gzip compression of the same tar members.
    utility = bundle / 'official_utils/MVTecAD2_public_code_utils/utils.py'
    utility_source = utility.read_text()
    original = 'tarfile.open(output_file, "w:gz")'
    assert utility_source.count(original) == 1
    shutil.copy2(utility, utility.with_name('utils_original.py'))
    utility.write_text(utility_source.replace(original, 'tarfile.open(output_file, "w:gz", compresslevel=1)'))
    builder = (TEMPLATE / 'build_submission.py').read_text()
    # GPU UUID comes from this bundle, keeping the established inference unchanged.
    builder = builder.replace("os.environ['CUDA_VISIBLE_DEVICES']==GPUS[1]",
                              "os.environ['CUDA_VISIBLE_DEVICES']==json.load(open(bundle/'manifest.json'))['gpu_uuid']")
    builder = builder.replace("env['CUDA_VISIBLE_DEVICES']=GPUS[1]",
                              "env['CUDA_VISIBLE_DEVICES']=manifest['gpu_uuid']")
    builder = builder.replace("'gpu_indices':[1]", "'gpu_indices':manifest['gpu_indices']")
    builder = builder.replace("'gpu':1", "'gpu':manifest['gpu_indices'][0]")
    # Offline predictions must import the exact frozen training runtime.
    compile(builder, str(bundle / 'build_submission.py'), 'exec')
    (bundle / 'build_submission.py').write_text(builder)
    shutil.copy2(TEMPLATE / 'submission_binary_masks.py', bundle / 'submission_binary_masks.py')
    shutil.copy2(__file__, bundle / 'prepare_submissions.py')
    shutil.copy2(Path(__file__).with_name('mixtures.py'), bundle / 'mixtures.py')
    gpu_uuid = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    mean = sum(j['public_validation_aupro'] for j in jobs.values()) / len(jobs)
    save(bundle / 'manifest.json', dict(jobs=jobs, created_utc=datetime.now(timezone.utc).isoformat(),
        variant=variant, source_queue=str(root), source_quotas=expected_quotas,
        source_quota_rotation=expected_rotation,
        source_snapshot=str(snapshot), source_plan=str(plan_path), gpu_indices=[gpu], gpu_uuid=gpu_uuid,
        mean_public_validation_aupro=mean, expected_maps=4090, thresholded_maps=True,
        selection='Exact reviewed category runs, best.pt selected on public validation AUPRO@0.05',
        output='Native-resolution float16 TIFF probabilities and uint8 binary PNG masks; no per-image normalization',
        threshold_selection='Maximum pooled pixel F1 on held-out public validation after float16 quantization',
        archive_compression='gzip level 1; official validation unchanged; original utility retained',
        private_data_used_for_training_or_selection=False, upload_performed=False))
    files = [p for p in bundle.rglob('*')
             if p.is_file() and 'jobs' not in p.relative_to(bundle).parts]
    save(bundle / 'frozen_files_sha256.json', {str(p.relative_to(bundle)):sha(p) for p in files})
    lines = [f'# MVTec AD 2: {variant}', '',
             'AD synthesis: 2x, Last50k, AAR. Frozen reviewed seed42 best checkpoints.',
             '4090 native-resolution float16 TIFF anomaly maps + 4090 binary PNG masks.',
             'Thresholds selected only on public validation. No upload performed.', '',
             '| Category | Validation AUPRO@0.05 | Epoch |', '|---|---:|---:|']
    lines += [f"| {c} | {j['public_validation_aupro']*100:.2f}% | {j['epoch']} |" for c,j in jobs.items()]
    lines += ['', f'Mean validation AUPRO: {mean*100:.2f}%.',
              'Final verification: status.json, checker.log, submission.tar.gz.sha256.',
              'Reproduction: build_submission.py uses the frozen source_snapshot and runtime_libs.', '']
    (bundle / 'README.md').write_text('\n'.join(lines))
    save(bundle / 'status.json', dict(stage='prepared', updated_utc=datetime.now(timezone.utc).isoformat()))
    return bundle, gpu_uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=tuple(VARIANT_QUEUES),
                        default=[variant for variant, _, _ in VARIANTS])
    args = parser.parse_args()
    if len(args.variants) > 2 or len(set(args.variants)) != len(args.variants):
        parser.error('Select one or two distinct variants for the two available GPUs.')
    assert shutil.disk_usage(BASE).free > 180 * 1024**3, 'Insufficient free space'
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    assert not active, 'GPUs are currently in use; refusing to overlap the export.'
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prepared = [(variant, *prepare(variant, VARIANT_QUEUES[variant], gpu, timestamp))
                for gpu, variant in enumerate(args.variants)]
    launched = []
    for variant, bundle, gpu_uuid in prepared:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid,
                   PYTHONPATH=str(bundle/'runtime_libs')+os.pathsep+str(bundle/'source_snapshot'),
                   HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        with (bundle / 'build.log').open('w') as log:
            process = subprocess.Popen([str(REPO/'.venv/bin/python'), '-u', str(bundle/'build_submission.py'),
                                        '--bundle', str(bundle)], cwd=bundle/'source_snapshot', env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        launched.append(dict(variant=variant, bundle=str(bundle), pid=process.pid))
    pointer = dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=launched, upload_performed=False)
    save(STUDY/'latest_private_submission_batch.json', pointer)
    print(json.dumps(pointer, indent=2), flush=True)


if __name__ == '__main__':
    main()
