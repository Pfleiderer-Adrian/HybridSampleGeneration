from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from generation_models.interfaces import StepOutput
from synthesizer.mask_manipulation import TransformGenerator


@dataclass
class Config:
    in_channels: int
    pretrained_model_name_or_path: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    revision: Optional[str] = None
    variant: Optional[str] = "fp16"
    torch_dtype: str = "auto"
    use_safetensors: bool = True
    local_files_only: bool = False
    resolution: int = 512
    prompt: str = "a realistic industrial anomaly texture, high detail"
    negative_prompt: str = "blur, low quality, text, watermark"
    num_train_timesteps: int = 1000
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.85
    prior_strength: float = 0.999
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    clamp_input: bool = True
    num_anomaly_classes: Optional[int] = None


class LatentDiffusionLoRA2D(nn.Module):
    """
    SDXL inpainting backend with LoRA fine-tuning and mask-guided generation.

    The pretrained Diffusers pipeline is intentionally loaded in warmup() so
    weights are downloaded before training starts but normal imports stay light.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pipeline = None
        self.noise_scheduler = None
        self._loaded_device = None
        self._loaded_dtype = None

    def warmup(self, shape, device=None, dtype=None, config=None):
        if len(shape) != 3:
            raise ValueError(f"LatentDiffusionLoRA2D expects anomaly_size (C,H,W), got {shape!r}.")
        if int(shape[0]) != int(self.cfg.in_channels):
            raise ValueError(f"Config in_channels={self.cfg.in_channels} does not match shape {shape!r}.")

        device = torch.device("cpu" if device is None else device)
        dtype = self._resolve_dtype(device, dtype)

        if self.pipeline is None:
            self._load_pretrained_pipeline(device=device, dtype=dtype)
        elif self._loaded_device != device:
            self.pipeline.to(device)
            self._loaded_device = device

        return self

    def on_epoch_start(self, epoch: int, config=None) -> None:
        return None

    def configure_optimizers(self, config):
        if self.pipeline is None:
            self.warmup(config.anomaly_size, config=config)

        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError(f"{self.__class__.__name__} has no trainable LoRA parameters.")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )
        return optimizer, None

    def training_step(self, batch, batch_idx: int, config=None) -> StepOutput:
        return self._shared_step(batch)

    def validation_step(self, batch, batch_idx: int, config=None) -> StepOutput:
        with torch.no_grad():
            return self._shared_step(batch)

    def save_checkpoint(self, path: str, **state) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save LatentDiffusionLoRA2D before warmup().")

        checkpoint = {
            "config": asdict(self.cfg),
            "trainable_state_dict": self._trainable_state_dict(),
            "state": self._serializable_checkpoint_state(state),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, **kwargs) -> None:
        if self.pipeline is None:
            self.warmup(kwargs.get("shape", (self.cfg.in_channels, self.cfg.resolution, self.cfg.resolution)))

        # Older checkpoints stored optimizer/scheduler objects in "state".
        # Loading those trusted project checkpoints requires PyTorch's full
        # pickle loader since 2.6 defaults torch.load(weights_only=True).
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("trainable_state_dict", checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        unexpected_lora = [key for key in unexpected if "lora" in key.lower()]
        if unexpected_lora:
            raise RuntimeError(f"Unexpected LoRA checkpoint keys: {unexpected_lora}")
        return missing

    def generate(self, sample, *, mode: str, **kwargs):
        mode = str(mode).lower()
        if mode in ("posterior", "posterior_sampling", "img2img"):
            return self._generate_posterior(sample, **kwargs)
        if mode in ("prior", "prior_sampling"):
            return self._generate_prior(sample, **kwargs)
        raise ValueError(f"Unknown generation mode {mode!r}. Expected 'prior' or 'posterior'.")

    def _shared_step(self, batch) -> StepOutput:
        if self.pipeline is None:
            raise RuntimeError("LatentDiffusionLoRA2D must be warmed up before training.")

        image, ori_mask, _ = self._extract_inputs(batch)
        device = self._model_device()
        image = self._prepare_image_tensor(image, device=device)
        mask = self._prepare_mask_tensor(ori_mask, device=device)
        masked_image = image * (1.0 - mask)

        unet_dtype = self._unet_dtype()
        latents = self._encode_image_latents(image).to(device=device, dtype=unet_dtype)
        masked_image_latents = self._encode_image_latents(masked_image).to(device=device, dtype=unet_dtype)
        mask_latents = F.interpolate(mask, size=latents.shape[-2:], mode="nearest").to(dtype=unet_dtype)

        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            int(self.cfg.num_train_timesteps),
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=unet_dtype)
        latent_model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

        prompt_embeds, pooled_prompt_embeds = self._prompt_embeddings(batch_size, device)
        prompt_embeds = prompt_embeds.to(device=device, dtype=unet_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=unet_dtype)
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": self._time_ids(batch_size, device, unet_dtype),
        }
        model_pred = self.pipeline.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        target = self._diffusion_target(latents, noise, timesteps)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return StepOutput(loss=loss, metrics={"total": loss, "diffusion": loss})

    def _generate_posterior(
        self,
        sample,
        *,
        n: int = 1,
        variation_strength: float = 1.0,
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
        **kwargs,
    ):
        image = self._extract_image(sample)
        ori_mask = self._extract_original_mask(sample)
        target_mask = self._target_mask_from_original(ori_mask, target_mask_generator)
        strength = self._bounded_strength(float(variation_strength) * float(self.cfg.strength))
        return self._run_inpaint_generation(
            image,
            target_mask,
            n=n,
            strength=strength,
            clamp_01=clamp_01,
            return_torch=return_torch,
        )

    def _generate_prior(
        self,
        sample,
        *,
        n: int = 1,
        variation_strength: float = 1.0,
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
        **kwargs,
    ):
        ori_mask = self._extract_original_mask(sample)
        target_mask = self._target_mask_from_original(ori_mask, target_mask_generator)
        canvas = torch.zeros((self.cfg.in_channels, *tuple(target_mask.shape[-2:])), dtype=torch.float32)
        strength = self._bounded_strength(float(variation_strength) * float(self.cfg.prior_strength))
        return self._run_inpaint_generation(
            canvas,
            target_mask,
            n=n,
            strength=strength,
            clamp_01=clamp_01,
            return_torch=return_torch,
        )

    def _run_inpaint_generation(
        self,
        image,
        target_mask,
        *,
        n: int,
        strength: float,
        clamp_01: bool,
        return_torch: bool,
    ):
        if self.pipeline is None:
            self.warmup((self.cfg.in_channels, *tuple(torch.as_tensor(image).shape[-2:])))
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

        image_tensor = self._single_image_tensor(image)
        target_mask_tensor = self._single_mask_tensor(target_mask)
        ref_hw = tuple(image_tensor.shape[-2:])
        image_pil = self._tensor_to_pil(image_tensor, size=(self.cfg.resolution, self.cfg.resolution))
        mask_pil = self._mask_to_pil(target_mask_tensor, size=(self.cfg.resolution, self.cfg.resolution))

        pipe = self.pipeline
        pipe.to(self._model_device())
        with torch.no_grad():
            result = pipe(
                prompt=self.cfg.prompt,
                negative_prompt=self.cfg.negative_prompt,
                image=image_pil,
                mask_image=mask_pil,
                height=self.cfg.resolution,
                width=self.cfg.resolution,
                strength=strength,
                guidance_scale=float(self.cfg.guidance_scale),
                num_inference_steps=int(self.cfg.num_inference_steps),
                num_images_per_prompt=int(n),
            )

        images = [self._pil_to_tensor(pil_image, ref_hw) for pil_image in result.images]
        output = torch.stack(images, dim=0)
        if n == 1:
            output = output[0]
        if clamp_01:
            output = output.clamp(0.0, 1.0)

        mask_out = target_mask_tensor.to(torch.uint8)
        if return_torch:
            return output, mask_out

        return (
            output.detach().cpu().numpy().astype(np.float32, copy=False),
            mask_out.detach().cpu().numpy().astype(np.uint8, copy=False),
        )

    def _load_pretrained_pipeline(self, *, device, dtype):
        try:
            from diffusers import DDPMScheduler, StableDiffusionXLInpaintPipeline
            from diffusers.training_utils import cast_training_params
            from peft import LoraConfig
        except ImportError as exc:
            raise ImportError(
                "LatentDiffusionLoRA2D requires diffusers, transformers, accelerate, peft, "
                "and safetensors. Install the updated requirements before training this model."
            ) from exc

        kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": bool(self.cfg.use_safetensors),
            "local_files_only": bool(self.cfg.local_files_only),
        }
        if self.cfg.revision:
            kwargs["revision"] = self.cfg.revision
        if self.cfg.variant:
            kwargs["variant"] = self.cfg.variant

        self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **kwargs,
        )
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.pipeline.to(device)
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)

        lora_config = LoraConfig(
            r=int(self.cfg.lora_rank),
            lora_alpha=int(self.cfg.lora_alpha),
            lora_dropout=float(self.cfg.lora_dropout),
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(lora_config)
        cast_training_params(self.unet, dtype=torch.float32)
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipeline.scheduler.config,
            num_train_timesteps=int(self.cfg.num_train_timesteps),
        )

        self._loaded_device = device
        self._loaded_dtype = dtype

    def _extract_inputs(self, batch):
        if isinstance(batch, dict):
            image = batch.get("img", batch.get("x"))
            ori_mask = batch.get("ori_mask", batch.get("mask"))
            tgt_mask = batch.get("tgt_mask")
            if image is None or ori_mask is None:
                raise ValueError("LatentDiffusionLoRA2D requires batch keys 'img' and 'ori_mask'/'mask'.")
            return torch.as_tensor(image), torch.as_tensor(ori_mask), tgt_mask
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return torch.as_tensor(batch[0]), torch.as_tensor(batch[1]), batch[2] if len(batch) > 2 else None
        raise TypeError(f"Unknown batch type: {type(batch)}")

    def _extract_image(self, sample):
        if isinstance(sample, dict):
            if "img" not in sample:
                raise KeyError("LatentDiffusionLoRA2D sample dict must contain 'img'.")
            return sample["img"]
        return sample

    def _extract_original_mask(self, sample):
        if isinstance(sample, dict):
            mask = sample.get("ori_mask", sample.get("mask"))
            if mask is None:
                raise KeyError("LatentDiffusionLoRA2D sample dict must contain 'ori_mask' or 'mask'.")
            return mask
        raise ValueError("LatentDiffusionLoRA2D prior/posterior generation requires an original mask.")

    def _target_mask_from_original(self, original_mask, target_mask_generator):
        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()
        return target_mask_generator.create_target_mask(original_mask=original_mask, conditional=True)

    def _prepare_image_tensor(self, image, *, device):
        image = torch.as_tensor(image, device=device).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor (B,C,H,W), got {tuple(image.shape)}")
        if self.cfg.clamp_input:
            image = image.clamp(0.0, 1.0)
        image = self._to_rgb_batch(image)
        image = F.interpolate(
            image,
            size=(int(self.cfg.resolution), int(self.cfg.resolution)),
            mode="bilinear",
            align_corners=False,
        )
        return image * 2.0 - 1.0

    def _prepare_mask_tensor(self, mask, *, device):
        mask = torch.as_tensor(mask, device=device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4 and mask.shape[1] != 1:
            mask = torch.amax(mask.float(), dim=1, keepdim=True)
        elif mask.ndim != 4:
            raise ValueError(f"Expected mask (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")
        mask = (mask.float() > 0).float()
        return F.interpolate(mask, size=(int(self.cfg.resolution), int(self.cfg.resolution)), mode="nearest")

    def _single_image_tensor(self, image):
        image = torch.as_tensor(image).float()
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("Generation expects a single sample, got batched image.")
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected image (C,H,W), got {tuple(image.shape)}")
        if self.cfg.clamp_input:
            image = image.clamp(0.0, 1.0)
        return image

    def _single_mask_tensor(self, mask):
        mask = torch.as_tensor(mask)
        if mask.ndim == 4:
            if mask.shape[0] != 1:
                raise ValueError("Generation expects a single sample, got batched mask.")
            mask = mask[0]
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            else:
                mask = torch.amax(mask.float(), dim=0)
        if mask.ndim != 2:
            raise ValueError(f"Expected mask (H,W), got {tuple(mask.shape)}")
        return (mask > 0).to(torch.uint8)

    def _to_rgb_batch(self, image):
        channels = image.shape[1]
        if channels == 3:
            return image
        if channels == 1:
            return image.repeat(1, 3, 1, 1)
        if channels > 3:
            return image[:, :3]
        repeat = int(np.ceil(3 / channels))
        return image.repeat(1, repeat, 1, 1)[:, :3]

    def _encode_image_latents(self, image):
        vae = self.pipeline.vae
        vae_dtype = self._module_dtype(vae, self._loaded_dtype)
        posterior = vae.encode(image.to(dtype=vae_dtype)).latent_dist
        latents = posterior.sample()
        return latents * vae.config.scaling_factor

    def _prompt_embeddings(self, batch_size, device):
        prompt = [self.cfg.prompt] * batch_size
        with torch.no_grad():
            encoded = self.pipeline.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        prompt_embeds = encoded[0]
        pooled_prompt_embeds = encoded[2] if len(encoded) > 2 else None
        if pooled_prompt_embeds is None:
            raise RuntimeError("SDXL prompt encoding did not return pooled prompt embeddings.")
        return prompt_embeds, pooled_prompt_embeds

    def _time_ids(self, batch_size, device, dtype):
        resolution = int(self.cfg.resolution)
        time_ids = torch.tensor(
            [resolution, resolution, 0, 0, resolution, resolution],
            device=device,
            dtype=dtype,
        )
        return time_ids.unsqueeze(0).repeat(batch_size, 1)

    def _diffusion_target(self, latents, noise, timesteps):
        prediction_type = getattr(self.noise_scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v_prediction":
            return self.noise_scheduler.get_velocity(latents, noise, timesteps)
        raise ValueError(f"Unsupported scheduler prediction_type={prediction_type!r}.")

    def _tensor_to_pil(self, image, *, size):
        rgb = self._to_rgb_batch(image.unsqueeze(0))[0]
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        )[0]
        array = (rgb.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def _mask_to_pil(self, mask, *, size):
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=size, mode="nearest")[0, 0]
        array = (mask.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array, mode="L")

    def _pil_to_tensor(self, image, ref_hw):
        image = image.resize((int(ref_hw[1]), int(ref_hw[0])), Image.BILINEAR)
        array = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.cfg.in_channels == 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        elif self.cfg.in_channels < 3:
            tensor = tensor[: self.cfg.in_channels]
        elif self.cfg.in_channels > 3:
            padding = torch.zeros((self.cfg.in_channels - 3, *tensor.shape[-2:]), dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor

    def _bounded_strength(self, value):
        return max(0.0, min(float(value), 1.0))

    def _resolve_dtype(self, device, dtype):
        if dtype is not None:
            return dtype
        if self.cfg.torch_dtype == "float16":
            return torch.float16
        if self.cfg.torch_dtype == "bfloat16":
            return torch.bfloat16
        if self.cfg.torch_dtype == "float32":
            return torch.float32
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    def _model_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _unet_dtype(self):
        return self._module_dtype(self.pipeline.unet, self._loaded_dtype)

    @staticmethod
    def _module_dtype(module, default):
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return default

    def _trainable_state_dict(self):
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable_names
        }

    @staticmethod
    def _serializable_checkpoint_state(state):
        serializable = {}
        for key, value in state.items():
            if hasattr(value, "state_dict"):
                serializable[key] = value.state_dict()
            else:
                serializable[key] = value
        return serializable
