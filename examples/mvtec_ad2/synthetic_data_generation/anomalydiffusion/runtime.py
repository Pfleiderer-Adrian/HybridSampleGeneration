"""Pinned upstream AnomalyDiffusion code with the declared latent-mask adaptation.

``setup.sh`` checks out the upstream repository and taming-transformers below
``ANOMALYDIFFUSION_HOME`` (default: this directory).
"""
import ast
import math
import os
import subprocess
import sys
import types
from pathlib import Path

from .prepare import UPSTREAM_COMMIT

TAMING_COMMIT = "3ba01b241669f5ade541ce990f7650a3b8f65318"
HOME = Path(os.environ.get("ANOMALYDIFFUSION_HOME", Path(__file__).parent)).resolve()
UPSTREAM = HOME / "upstream"
TAMING = HOME / ".tools" / "taming-transformers"


def _revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def configure_upstream(policy="maxpool"):
    """Import the verified upstream code; ``maxpool`` keeps every occupied latent cell."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    if policy not in ("maxpool", "original"):
        raise ValueError("Unknown latent mask policy.")
    if _revision(UPSTREAM) != UPSTREAM_COMMIT:
        raise ValueError(f"Wrong upstream revision in {UPSTREAM}.")
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "--name-only"], text=True).strip():
        raise ValueError("Upstream tracked sources have been modified.")
    if _revision(TAMING) != TAMING_COMMIT:
        raise ValueError(f"Wrong taming-transformers revision in {TAMING}.")
    for path in (TAMING, UPSTREAM):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # pSp uses only EqualLinear, without its optional fused activation. Execute the
    # exact upstream class to avoid compiling unrelated StyleGAN CUDA extensions.
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

    ddpm.transforms = ResolutionTransforms()


def load_model(checkpoint, work, category, policy="maxpool"):
    """Appearance model (text embeddings + spatial encoder) on the frozen base LDM."""
    import torch
    from omegaconf import OmegaConf

    configure_upstream(policy)
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(UPSTREAM / "configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml")
    config.model.params.personalization_config.params.initializer_words = ["anomaly"]
    # EmbeddingManager2 reads this file from the working directory, without newline.
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
    model.prepare_spatial_encoder(optimze_together=True, data_enhance=False, l=4)
    model.learning_rate = .005
    model.cuda()
    return model, {"missing_keys": missing, "unexpected_keys": unexpected, "latent_mask_policy": policy,
                   "stage": "appearance", "upstream_commit": UPSTREAM_COMMIT}
