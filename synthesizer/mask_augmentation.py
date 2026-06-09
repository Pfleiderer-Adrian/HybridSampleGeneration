import numpy as np
import scipy.ndimage as ndi
import torch
from monai.transforms import Rand2DElastic, Rand3DElastic

<<<<<<< HEAD
def to_one_hot_3D(mask: torch.Tensor, num_anomaly_classes: int) -> torch.Tensor:
    """Converts 3D/4D/5D integer masks to 5D one-hot float tensors of shape (B, C, D, H, W)."""
    
    # already 5D and one hot encoded (more than one Channel)
    if mask.ndim == 5 and mask.shape[1] > 1:
        return mask.float()
        
    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask.squeeze(1) # -> (B, D, H, W)

    # missing batch dim
    if mask.ndim == 3:
        mask = mask.unsqueeze(0) # -> (1, D, H, W)
        
    # mask must be (B, D, H, W) here
    if mask.ndim != 4:
        raise ValueError(f"Expected mask shape (B, D, H, W) after cleanup, got: {mask.shape}.")
        
    mask = mask.long()
    
    num_classes = num_anomaly_classes + 1
    # (B, D, H, W) -> (B, D, H, W, num_classes)
    mask_oh = F.one_hot(mask, num_classes=num_classes)
    
    # remove class 0 channel (background channel)
    mask_oh = mask_oh[..., 1:] 
        
    # (B, D, H, W, C) -> (B, C, D, H, W)
    return mask_oh.permute(0, 4, 1, 2, 3).float()

def to_one_hot_2D(mask: torch.Tensor, num_anomaly_classes: int) -> torch.Tensor:
    """Converts 2D/3D/4D integer masks to 4D one-hot float tensors of shape (B, C, H, W)."""

    # already 4D and one hot encoded without background channel
    if mask.ndim == 4 and mask.shape[1] == num_anomaly_classes:
        return mask.float()

    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)  # -> (B, H, W)

    # missing batch dim
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # -> (1, H, W)

    # mask must be (B, H, W) here
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape (B, H, W) after cleanup, got: {mask.shape}.")

    mask = mask.long()

    num_classes = num_anomaly_classes + 1
    # (B, H, W) -> (B, H, W, num_classes)
    mask_oh = F.one_hot(mask, num_classes=num_classes)

    # remove class 0 channel (background channel)
    mask_oh = mask_oh[..., 1:]

    # (B, H, W, C) -> (B, C, H, W)
    return mask_oh.permute(0, 3, 1, 2).float()


def augment_mask(mask_np: np.ndarray, config):
    if config.morph_transform:
        mask_np = random_morphological_transform(
            mask_np,
            config.morph_classes,
            config.morph_priorities,
            config.morph_params,
        )
    if config.elastic_transform:
        mask_np = random_elastic_transform(mask_np, config.sigma, config.magnitude)
    return mask_np


def random_morphological_transform(mask_np: np.ndarray, classes=None, priorities=None, params=None):
    """
    Nimmt eine 2D- oder 3D-Maske im Channel-First-Format:
      2D: (C, H, W)
      3D: (C, D, H, W)

    Wendet optional globales Stretching/Stauchen auf die Labelmaske an und
    danach klassenspezifische Morphologie (standardmaessig Dilation).
    """
    original_dtype = mask_np.dtype
    ndim = mask_np.ndim

    if ndim not in (3, 4):
        raise ValueError(f"Unterstuetzt nur 2D oder 3D Daten. Deine Maske hat {ndim} Dimensionen.")

    params = {} if params is None else params
    transformed_mask = mask_np[0].copy()

    if classes is None:
        classes = [cls for cls in np.unique(transformed_mask) if cls != 0]
    if priorities is None:
        priorities = classes

    if np.random.random() < params.get("global_stretch_prob", 1):
        spatial_ndim = transformed_mask.ndim
        stretch_min = params.get("stretch_min", 0.95)
        stretch_max = params.get("stretch_max", 1.05)
        scales = [np.random.uniform(stretch_min, stretch_max) for _ in range(spatial_ndim)]

        inv_scales = 1.0 / np.array(scales)
        matrix = np.diag(inv_scales)

        center = np.array(transformed_mask.shape) / 2.0
        offset = center - np.dot(matrix, center)

        transformed_mask = ndi.affine_transform(
            transformed_mask,
            matrix=matrix,
            offset=offset,
            output_shape=transformed_mask.shape,
            order=0,
            mode="constant",
            cval=0,
        )

    class_masks = {}
    max_iterations = params.get("max_morph_iterations", 1)
    operations = params.get("operations", ("dilate", "none"))

    for cls in classes:
        binary_mask = transformed_mask == cls

        if not np.any(binary_mask):
            class_masks[cls] = binary_mask
            continue

        operation = np.random.choice(operations)
        iterations = np.random.randint(1, max_iterations + 1)

        if operation == "dilate":
            binary_mask = ndi.binary_dilation(binary_mask, iterations=iterations)
        elif operation != "none":
            raise ValueError(f"Unsupported morph operation: {operation}")

        class_masks[cls] = binary_mask

    final_mask = np.zeros_like(transformed_mask)
    for cls in priorities:
        if cls in class_masks:
            final_mask[class_masks[cls]] = cls

    final_mask = final_mask.astype(original_dtype)
    if mask_np.shape[0] == 1:
        return final_mask[None, ...]
    return np.repeat(final_mask[None, ...], mask_np.shape[0], axis=0)


def random_elastic_transform(mask_np: np.ndarray, sigma=5, magnitude=200):
    """
    Nimmt ein 2D oder 3D NumPy-Array, wählt dynamisch die richtige 
    MONAI-Transformation (2D oder 3D) und gibt das verzerrte Array zurück.
    """
    original_dtype = mask_np.dtype
    ndim = mask_np.ndim

    # MONAI erwartet das Format [Kanal, Raumdimensionen] -> [1, (D), H, W]
    mask_tensor = torch.from_numpy(mask_np).float()

    if ndim == 3:
        # Fuer 2D (PNG) -> Steuerung der Gitterdichte ueber 'spacing'
        re_deform = Rand2DElastic(
            spacing=(sigma, sigma), # abstand der kontrollpunkte
            magnitude_range=(magnitude, magnitude),
            prob=1.0,
            mode="nearest",
            padding_mode="reflection",
        )
    elif ndim == 4:
        # Fuer 3D (NIfTI) -> Steuerung der Gitterglaettung ueber 'sigma_range'
        re_deform = Rand3DElastic(
            sigma_range=(sigma, sigma),
            magnitude_range=(magnitude, magnitude),
            prob=1.0,
            mode="nearest",
            padding_mode="reflection",
        )
    else:
        raise ValueError(f"Unterstützt nur 2D oder 3D Daten. Deine Maske hat {ndim} Dimensionen.")

    transformed_tensor = re_deform(mask_tensor)

    # Kanal entfernen und zurück zu NumPy mit Original-Datentyp
    return transformed_tensor.squeeze(0).numpy().astype(original_dtype)
