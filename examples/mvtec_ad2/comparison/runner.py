"""Restartable category preparation and paired DRAEM runs."""
import csv
import hashlib
import importlib
import json
import platform
import shutil
import traceback
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from examples.mvtec_ad2.dataset import MVTecAD2Dataloader
from examples.mvtec_ad2.downstream.draem.model import DRAEM
from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream, load_run
from examples.mvtec_ad2.downstream.textures import prepare_textures
from examples.mvtec_ad2.splits import (SplitConfiguration, create_grouped_manifest, validate_grouped_manifest,
                                     manifest_samples, verify_repository)
from examples.mvtec_ad2.downstream.transforms import read_image
from hybrid_sample_generator import HybridDataGenerator
from hybrid_sample_generator.configuration.study import StudyConfiguration
from hybrid_sample_generator.randomness import seeded_random
from .reporting import write_report


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def canonical(value):
    return json.loads(json.dumps(value))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_generator(category, folder, seed):
    config = importlib.import_module(f'examples.mvtec_ad2.categories.{category}').create_configuration()
    config.study = StudyConfiguration(config.study.name, str(folder), seed=seed)
    config.matching.routine = 'fixed_from_extraction_control_fusion'
    config.validate()
    return config


def split_summary(category, manifest):
    rows = []
    from PIL import Image
    for partition, items in manifest['partitions'].items():
        pixels = positive_pixels = 0
        for item in items:
            with Image.open(item['image_path']) as image:
                width, height = image.size
            pixels += width * height
            if item['mask_path']:
                mask = read_image(item['mask_path'])
                if mask.shape[1:] != (height, width):
                    raise ValueError(f"Image/mask shape mismatch: {item['sample_id']}")
                count = int(np.count_nonzero(np.any(mask > 0, axis=0)))
                if item['label'] != 'good' and not count:
                    raise ValueError(f"Positive sample has no usable mask: {item['sample_id']}")
                positive_pixels += count
            elif item['label'] != 'good':
                raise ValueError(f"Positive sample has no usable mask: {item['sample_id']}")
        positive = sum(item['label'] != 'good' for item in items)
        rows.append({'category': category, 'partition': partition, 'positive_images': positive,
                     'normal_images': len(items) - positive, 'positive_fraction': positive / len(items),
                     'acquisition_groups': len({item['acquisition_group'] for item in items}),
                     'positive_groups': len({item['acquisition_group'] for item in items if item['label'] != 'good'}),
                     'positive_pixels': positive_pixels, 'total_pixels': pixels,
                     'pixel_positive_fraction': positive_pixels / pixels})
    return rows


def prepare_hybrids(config, manifest, state, state_path):
    signature = fingerprint({'config': config.to_dict(), 'split': manifest['fingerprint']})
    if state.get('generator_signature') not in (None, signature):
        raise ValueError('Existing generator preparation uses a different configuration or split.')
    state['generator_signature'] = signature
    pipeline = HybridDataGenerator(config)
    steps = (
        ('ingestion', lambda: pipeline.ingest_dataset(MVTecAD2Dataloader(manifest_samples(manifest, 'train')))),
        ('extraction', pipeline.extract_anomalies),
        ('generator_training', pipeline.train_generator),
        ('generation', pipeline.generate_synthetic_anomalies),
        ('planning', pipeline.plan_hybrid_samples),
        ('fusion', pipeline.materialize_hybrid_samples),
    )
    for name, operation in steps:
        if state.get(name) == 'complete':
            if name == 'generator_training' and state.get('generation') != 'complete':
                pipeline.load_generator()
            continue
        state[name] = 'running'
        save_json(state_path, state)
        with seeded_random(config.study.seed):
            operation()
        state[name] = 'complete'
        save_json(state_path, state)
    verify_repository(pipeline.repository, manifest)
    if not pipeline.repository.list_hybrid_samples(status='generated'):
        raise ValueError('Hybrid preparation produced no generated samples.')
    config.save_config_file()


def run_comparison(configuration, dataset_root, output, *, dry_run=False):
    configuration.validate()
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    if dataset_root == output or dataset_root in output.parents:
        raise ValueError('Experiment output must be outside the dataset.')
    output.mkdir(parents=True, exist_ok=True)
    snapshot = canonical({'settings': asdict(configuration), 'dataset_root': str(dataset_root)})
    experiment_path = output / 'experiment.json'
    if experiment_path.exists() and json.loads(experiment_path.read_text()) != snapshot:
        raise ValueError('Existing experiment settings differ. Use a new output directory.')
    save_json(experiment_path, snapshot)
    manifests, summaries = {}, []
    # Validate every category before starting expensive training.
    for category in configuration.categories:
        folder = output / category
        path = folder / 'split_manifest.json'
        expected = {'rule': 'source_split/label/numeric_capture_id', 'test_positive_fraction': configuration.test_positive_fraction}
        split = SplitConfiguration(configuration.test_fraction, configuration.validation_fraction, configuration.seed)
        if path.exists():
            manifest = json.loads(path.read_text())
            validate_grouped_manifest(manifest)
            if manifest['split'] != asdict(split) or manifest['grouping'] != expected or manifest['category_root'] != str(dataset_root / category):
                raise ValueError('Existing split settings differ.')
        else:
            manifest = create_grouped_manifest(dataset_root / category, split, test_positive_fraction=configuration.test_positive_fraction)
            save_json(path, manifest)
        manifests[category] = manifest
        summaries.extend(split_summary(category, manifest))
    with (output / 'split_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        if row['partition'] == 'test':
            print(f"{row['category']}: {row['positive_images']} positive / {row['normal_images']} normal test images ({row['positive_fraction']:.1%}), {row['positive_groups']} positive capture groups")
    if dry_run:
        return summaries
    shared = deepcopy(configuration.downstream)
    shared.seed = configuration.seed
    shared.data.normal_fraction = .5
    shared.data.hybrid_fraction = .5
    prepare_textures(output, shared.data)
    # Freeze one texture inventory for both variants and all categories.
    texture_files = sorted(Path(shared.data.texture_root).rglob('*'))
    textures = {str(path.relative_to(shared.data.texture_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in texture_files if path.is_file()}
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'cuda': torch.version.cuda,
                   'cuda_devices': [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
                   'textures': textures}
    inventory_path = output / 'environment.json'
    if inventory_path.exists() and json.loads(inventory_path.read_text()) != environment:
        raise ValueError('Software or texture inventory changed. Use a new experiment directory.')
    save_json(inventory_path, environment)
    failures = []
    for category, manifest in manifests.items():
        folder = output / category
        state_path = folder / 'status.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        try:
            generator = build_generator(category, folder / 'hybrid_generation', configuration.seed)
            prepare_hybrids(generator, manifest, state, state_path)
            initial = folder / 'draem_initial.pt'
            if not initial.exists():
                with seeded_random(configuration.seed):
                    model = DRAEM(shared.training.reconstruction_width, shared.training.segmentation_width)
                    torch.save(model.state_dict(), initial)
            initial_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
            if state.get('initial_hash') not in (None, initial_hash):
                raise ValueError('Initial DRAEM checkpoint changed.')
            state['initial_hash'] = initial_hash
            for variant, fraction in (('baseline', 0.), ('hybrid', .5)):
                settings = deepcopy(shared)
                settings.data.hybrid_fraction = fraction
                destination = folder / variant
                if state.get(variant) == 'complete':
                    _, saved, _ = load_run(destination, manifest)
                    if canonical(saved.to_dict()) != canonical(settings.to_dict()) or not (destination / 'metrics.json').exists():
                        raise ValueError('Completed run no longer matches its saved configuration/results.')
                    continue
                if destination.exists():
                    # Preserve interrupted runs instead of deleting their checkpoints.
                    archive = folder / 'interrupted'
                    archive.mkdir(exist_ok=True)
                    target = archive / f'{variant}_{len(list(archive.iterdir()))}'
                    shutil.move(str(destination), str(target))
                state[variant] = 'running'
                save_json(state_path, state)
                train_downstream(generator, manifest, settings, output_folder=destination, initial_checkpoint=initial)
                evaluate_downstream(destination, manifest)
                state[variant] = 'complete'
                save_json(state_path, state)
            state.pop('error', None)
            save_json(state_path, state)
        except Exception:
            state['error'] = traceback.format_exc()
            save_json(state_path, state)
            print(f'{category} failed: {state["error"]}')
            failures.append(category)
        write_report(output, configuration.categories)
    if failures:
        raise RuntimeError('Failed categories: ' + ', '.join(failures))
    return summaries
