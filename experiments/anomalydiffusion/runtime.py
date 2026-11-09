"""Compatibility and declared resolution adaptation around pinned upstream code."""
import ast
import math
import subprocess
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .prepare import UPSTREAM_COMMIT

UPSTREAM = Path(__file__).parent / "upstream"


def configure_upstream(policy="maxpool"):
    actual = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError(f"Wrong upstream revision: {actual}")
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "--name-only"], text=True).strip():
        raise ValueError("Upstream tracked sources have been modified.")
    sys.path.insert(0, str(UPSTREAM))
    taming = UPSTREAM.parent / ".tools/taming-transformers"
    taming_revision = subprocess.check_output(["git", "-C", str(taming), "rev-parse", "HEAD"], text=True).strip()
    if taming_revision != "3ba01b241669f5ade541ce990f7650a3b8f65318":
        raise ValueError("Wrong taming-transformers revision.")
    sys.path.insert(0, str(taming))
    # pSp uses only EqualLinear, without its optional fused activation. Execute the
    # exact upstream class to avoid importing unrelated StyleGAN CUDA extensions.
    name = "ldm.models.psp_encoder.stylegan2.model"
    source = (UPSTREAM / "ldm/models/psp_encoder/stylegan2/model.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "EqualLinear")
    module = types.ModuleType(name)
    module.__dict__.update(math=math, torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), module.__dict__)
    sys.modules[name] = module
    import ldm.models.diffusion.ddpm as ddpm
    from torchvision import transforms

    class ResolutionTransforms:
        def __getattr__(self, key):
            return getattr(transforms, key)

        def Resize(self, size, *args, **kwargs):
            if policy == "maxpool" and tuple(size) == (32, 32):
                return nn.AdaptiveMaxPool2d((32, 32))
            return transforms.Resize(size, *args, **kwargs)

    if policy not in ("maxpool", "original"):
        raise ValueError("Unknown latent mask policy.")
    ddpm.transforms = ResolutionTransforms()


def load_model(stage, checkpoint, work, category, policy):
    from omegaconf import OmegaConf
    configure_upstream(policy)
    from ldm.util import instantiate_from_config
    suffix = "-encoder+embedding" if stage == "appearance" else ""
    config = OmegaConf.load(UPSTREAM / f"configs/latent-diffusion/txt2img-1p4B-finetune{suffix}.yaml")
    config.model.params.personalization_config.params.initializer_words = ["anomaly" if stage == "appearance" else "crack"]
    # EmbeddingManager2 reads this file in the working directory, without newline.
    import os
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "name-anomaly.txt").write_text(f"{category}+bad")
    os.chdir(work)
    model = instantiate_from_config(config.model)
    weights = torch.load(checkpoint, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(weights, strict=False)
    # Personalized embeddings are new; all pretrained diffusion components must load.
    critical = [key for key in missing if key.startswith(("model.", "first_stage_model.", "cond_stage_model."))]
    if critical:
        raise ValueError(f"Base checkpoint is incomplete: {critical[:20]}")
    del weights
    if stage == "appearance":
        model.prepare_spatial_encoder(optimze_together=True, data_enhance=False, l=4)
    model.learning_rate = .005
    model.cuda()
    return model, {"missing_keys": missing, "unexpected_keys": unexpected,
                   "latent_mask_policy": policy, "stage": stage,
                   "upstream_commit": UPSTREAM_COMMIT}
