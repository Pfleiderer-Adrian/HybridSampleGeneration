"""SuperADD-inspired frozen multi-layer DINOv3 dual memory banks.

Independent implementation: normalized features, bounded random reservoirs,
chunked exact nearest neighbors and additive defect evidence (not upstream parity).
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .dinov3 import DINOv3Segmentation
from .patches import tile_starts


class Reservoir:
    """Uniform priority sampling, with bounded storage across streaming updates."""
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.values = self.priorities = None

    def add(self, values):
        values = values.detach().float().cpu()
        if not len(values):
            return
        priorities = torch.from_numpy(self.rng.random(len(values)))
        if self.values is not None:
            values = torch.cat([self.values, values])
            priorities = torch.cat([self.priorities, priorities])
        indices = torch.topk(priorities, min(self.capacity, len(priorities))).indices
        self.values, self.priorities = values[indices], priorities[indices]


class SuperADDDualBank(nn.Module):
    def __init__(self, config, *, backbone_config=None, backbone=None):
        super().__init__()
        self.settings = config.memory_bank
        self.data = config.data
        # Reuse pretrained loading/preprocessing; no decoder is kept or trained.
        if backbone is None:
            from copy import deepcopy
            settings = deepcopy(config.dinov3)
            settings.decoder = 'cnn'
            backbone = DINOv3Segmentation(settings, backbone_config=backbone_config).backbone
        self.backbone = backbone.requires_grad_(False).eval()
        if max(self.settings.layers) > backbone.config.num_hidden_layers:
            raise ValueError('Memory-bank feature layer exceeds backbone depth.')
        self.patch_size = int(backbone.config.patch_size)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        for layer in self.settings.layers:
            self.register_buffer(f'normal_{layer}', torch.empty(0, backbone.config.hidden_size))
            self.register_buffer(f'defect_{layer}', torch.empty(0, backbone.config.hidden_size))
        self.register_buffer('normal_scales', torch.ones(len(self.settings.layers)))
        self.fitted = False

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def features(self, image):
        h, w = image.shape[-2:]
        p = self.patch_size
        padded = F.pad(image, (0, (-w) % p, 0, (-h) % p), mode='replicate')
        output = self.backbone(pixel_values=(padded - self.mean) / self.std, output_hidden_states=True)
        offset = 1 + self.backbone.config.num_register_tokens
        grid = (padded.shape[-2] // p, padded.shape[-1] // p)
        result = [F.normalize(output.hidden_states[layer][:, offset:].float(), dim=-1)
                  for layer in self.settings.layers]
        if any(x.shape[1] != grid[0] * grid[1] for x in result):
            raise ValueError('Unexpected spatial token layout.')
        return result, grid

    @torch.no_grad()
    def nearest(self, query, bank):
        if not len(bank):
            raise ValueError('Normal reference bank is empty.')
        result = []
        for q in query.split(self.settings.query_chunk_size):
            best = torch.full((len(q),), float('inf'), device=q.device)
            for keys in bank.split(self.settings.reference_chunk_size):
                # Stable normalized squared Euclidean distance without an N x M x C tensor.
                distance = (q.square().sum(1)[:, None] + keys.square().sum(1)[None]
                            - 2 * q @ keys.T).clamp_min(0)
                best = torch.minimum(best, distance.min(1).values)
            result.append(best.sqrt())
        return torch.cat(result)

    def tiles(self, image):
        if self.data.mode == 'image':
            yield image
            return
        ph, pw = self.data.patch_size
        h, w = image.shape[-2:]
        padded = F.pad(image[None], (0, max(0, pw-w), 0, max(0, ph-h)), mode='replicate')[0]
        for y in tile_starts(padded.shape[-2], ph, self.data.patch_overlap):
            for x in tile_starts(padded.shape[-1], pw, self.data.patch_overlap):
                yield padded[:, y:y+ph, x:x+pw]

    @torch.no_grad()
    def fit(self, normal_dataset, mixed_dataset, calibration_dataset, output, seed):
        device = self.mean.device
        settings = self.settings
        normal = [Reservoir(settings.normal_capacity, seed + i) for i in range(len(settings.layers))]
        defect = [Reservoir(settings.defect_capacity, seed + 100 + i) for i in range(len(settings.layers))]
        for i in tqdm(range(len(normal_dataset)), desc='Normal memory bank'):
            for tile in self.tiles(normal_dataset[i]['image']):
                features, _ = self.features(tile[None].to(device))
                for reservoir, values in zip(normal, features):
                    reservoir.add(values[0])
        for layer, reservoir in zip(settings.layers, normal):
            if reservoir.values is None:
                raise ValueError('No normal training features.')
            setattr(self, f'normal_{layer}', reservoir.values.to(device))
        counts = Counter()
        rng = np.random.default_rng(seed + 200)
        if mixed_dataset is not None:
            for i in tqdm(range(len(mixed_dataset)), desc='Defect memory bank'):
                sample = mixed_dataset[i]
                counts[sample['source']] += 1
                if sample['source'] == 'normal':
                    continue
                image, mask = sample['image'], sample['mask']
                features, grid = self.features(image[None].to(device))
                components, number = ndimage.label(mask[0].numpy() > 0, structure=np.ones((3, 3)))
                selected = set()
                for region in range(1, number + 1):
                    region_mask = torch.from_numpy(components == region).float()[None, None]
                    h, w = region_mask.shape[-2:]
                    region_mask = F.pad(region_mask, (0, grid[1]*self.patch_size-w, 0, grid[0]*self.patch_size-h))
                    covered = F.max_pool2d(region_mask, self.patch_size).flatten().nonzero().flatten().numpy()
                    chosen = rng.choice(covered, min(len(covered), settings.tokens_per_region), replace=False)
                    selected.update(chosen.tolist())
                indices = sorted(selected)
                for reservoir, values in zip(defect, features):
                    reservoir.add(values[0, indices])
        for layer, reservoir in zip(settings.layers, defect):
            if reservoir.values is None and mixed_dataset is not None:
                raise ValueError('No masked defect features; check masks and sampling.')
            if reservoir.values is not None:
                setattr(self, f'defect_{layer}', reservoir.values.to(device))
        # Held-out NORMAL validation images only; no test masks, no training-bank queries.
        if not len(calibration_dataset):
            raise ValueError('Normal validation images are required for distance calibration.')
        scales = [Reservoir(10000, seed + 300 + i) for i in range(len(settings.layers))]
        for i in range(min(len(calibration_dataset), settings.calibration_images)):
            for tile in self.tiles(calibration_dataset[i]['image']):
                features, _ = self.features(tile[None].to(device))
                for j, (layer, values) in enumerate(zip(settings.layers, features)):
                    distances = self.nearest(values[0], getattr(self, f'normal_{layer}'))
                    scales[j].add(distances[:, None])
        for i, reservoir in enumerate(scales):
            self.normal_scales[i] = torch.quantile(reservoir.values[:, 0], .95).clamp_min(1e-6).to(device)
        self.fitted = True
        info = {'source_counts': dict(counts), 'normal_images': len(normal_dataset),
                'calibration_images': min(len(calibration_dataset), settings.calibration_images),
                'normal_scales': self.normal_scales.tolist(),
                'bank_sizes': {str(layer): {'normal': len(getattr(self, f'normal_{layer}')),
                                           'defect': len(getattr(self, f'defect_{layer}'))} for layer in settings.layers},
                'implementation': 'superadd_inspired_dual_bank', 'bank_selection': 'uniform_priority_reservoir'}
        output = Path(output)
        (output / 'bank_build.json').write_text(json.dumps(info, indent=2) + '\n')
        checkpoints = output / 'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        checkpoint = {'model': self.state_dict(), 'backbone_config': self.backbone.config.to_dict(),
                      'selection_metric': 'no_gradient_training_bank_construction', 'bank_build': info}
        torch.save(checkpoint, checkpoints / 'best.pt')
        # A bank has one fitted state, so both checkpoint names refer to that state.
        import shutil
        shutil.copy2(checkpoints / 'best.pt', checkpoints / 'last.pt')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for layer in self.settings.layers:
            for prefix in ('normal', 'defect'):
                key = f'{prefix}_{layer}'
                if key in state_dict:
                    setattr(self, key, torch.empty_like(state_dict[key], device=self.mean.device))
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.fitted = True
        return result

    @torch.no_grad()
    def forward(self, image):
        if not self.fitted:
            raise ValueError('Memory banks must be fitted before inference.')
        features, grid = self.features(image)
        scores = []
        for i, (layer, values) in enumerate(zip(self.settings.layers, features)):
            query = values.flatten(0, 1)
            dn = self.nearest(query, getattr(self, f'normal_{layer}'))
            scale = self.normal_scales[i]
            base = dn / (dn + scale)
            defect = getattr(self, f'defect_{layer}')
            weight = self.settings.defect_weight
            if weight and len(defect):
                da = self.nearest(query, defect)
                support = torch.sigmoid((dn - da) / scale)
                # Preserve normal-distance evidence; known defects add bounded support.
                base = base * (1 + weight * support) / (1 + weight)
            scores.append(base.reshape(len(image), 1, *grid))
        probability = torch.stack(scores).mean(0)
        probability = F.interpolate(probability, size=(grid[0]*self.patch_size, grid[1]*self.patch_size), mode='bilinear', align_corners=False)
        probability = probability[..., :image.shape[-2], :image.shape[-1]].clamp(1e-7, 1-1e-7)
        return None, torch.cat([torch.log1p(-probability), probability.log()], dim=1)
