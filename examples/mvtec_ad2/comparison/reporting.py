"""Paired category metrics and equally weighted category averages."""
import csv
import json
from pathlib import Path

METRICS = ('image_auroc', 'image_ap', 'pixel_auroc', 'pixel_ap', 'segmentation_loss')


def write_report(output, categories):
    output = Path(output)
    rows = []
    for category in categories:
        paths = {variant: output / category / variant / 'metrics.json' for variant in ('baseline', 'hybrid')}
        if not all(path.is_file() for path in paths.values()):
            continue
        results = {variant: json.loads(path.read_text()) for variant, path in paths.items()}
        manifest = json.loads((output / category / 'split_manifest.json').read_text())
        for partition in ('validation', 'test'):
            items = manifest['partitions'][partition]
            positive = sum(item['label'] != 'good' for item in items)
            for metric in METRICS:
                baseline = results['baseline'][partition].get(metric)
                hybrid = results['hybrid'][partition].get(metric)
                rows.append({'category': category, 'partition': partition, 'metric': metric,
                             'positive_images': positive, 'normal_images': len(items) - positive,
                             'baseline': baseline, 'hybrid': hybrid,
                             'delta': hybrid - baseline if baseline is not None and hybrid is not None else None})
    for partition in ('validation', 'test'):
        for metric in METRICS:
            values = [row for row in rows if row['partition'] == partition and row['metric'] == metric and row['delta'] is not None]
            if values:
                rows.append({'category': f'macro_mean ({len(values)} categories)', 'partition': partition,
                             'metric': metric, 'positive_images': '', 'normal_images': '',
                             **{key: sum(row[key] for row in values) / len(values) for key in ('baseline', 'hybrid', 'delta')}})
    fields = ('category', 'partition', 'metric', 'positive_images', 'normal_images', 'baseline', 'hybrid', 'delta')
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    text = ['# DRAEM baseline versus hybrid', '',
            'Delta = hybrid − baseline. Higher is better except segmentation loss.',
            'Macro means include only complete pairs; the number of included categories is shown.', '',
            '| Category | Split | Metric | Baseline | Hybrid | Delta |', '|---|---|---|---:|---:|---:|']
    for row in rows:
        numbers = [f'{row[key]:.6g}' if row[key] is not None else 'N/A' for key in ('baseline', 'hybrid', 'delta')]
        text.append(f"| {row['category']} | {row['partition']} | {row['metric']} | {' | '.join(numbers)} |")
    missing = [category for category in categories if not all((output / category / variant / 'metrics.json').is_file() for variant in ('baseline', 'hybrid'))]
    text.extend(['', 'Incomplete categories: ' + (', '.join(missing) or 'none'), ''])
    (output / 'comparison.md').write_text('\n'.join(text))
    return rows
