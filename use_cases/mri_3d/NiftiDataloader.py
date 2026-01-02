import os
import glob
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from dataclasses import dataclass

import nibabel as nib
import numpy as np


# -------------------------
# Data structures
# -------------------------
@dataclass
class SampleInfo:
    shape: Tuple[int, ...]
    image_dtype: np.dtype
    mask_dtype: np.dtype
    ndim: int
    height: int
    width: int
    depth: int
    channels: int


# -------------------------
# Helpers
# -------------------------
def _load_nifti_array_affine(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI as numpy array + affine.
    Uses get_fdata() to match previous behavior (float output).
    """
    obj = nib.load(path)
    try:
        arr = obj.get_fdata()  # float64 typically
        aff = obj.affine
    finally:
        try:
            obj.uncache()
        except Exception:
            pass
    return arr, aff


def ensure_cdhw(arr: np.ndarray, channels_hint: int = 1) -> np.ndarray:
    """
    Convert common NIfTI array shapes to (C, D, H, W).

    Supports:
      - (H, W) -> (1, 1, H, W)
      - (H, W, D) -> (1, D, H, W)
      - (H, W, D, C) -> (C, D, H, W)
      - (C, D, H, W) -> unchanged
      - (C, H, W, D) -> (C, D, H, W) (heuristic)

    channels_hint is used to detect whether first axis is channel.
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        H, W = arr.shape
        return arr.reshape(1, 1, H, W)

    if arr.ndim == 3:
        # Assume (H, W, D) typical NIfTI -> (1, D, H, W)
        H, W, D = arr.shape
        return arr.transpose(2, 0, 1).reshape(1, D, H, W)

    if arr.ndim == 4:
        # Case A: already (C, D, H, W)
        if arr.shape[0] == channels_hint:
            # could be (C, D, H, W) or (C, H, W, D)
            # Heuristic: if last axis looks like depth (often <= H/W), treat as (C,H,W,D)
            C, A, B, Z = arr.shape
            if Z <= min(A, B) and (A != Z or B != Z):
                return arr.transpose(0, 3, 1, 2)  # (C, D, H, W)
            return arr

        # Case B: assume (H, W, D, C) -> (C, D, H, W)
        return arr.transpose(3, 2, 0, 1)

    raise ValueError(f"Unsupported ndim={arr.ndim}, shape={arr.shape}")


def minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-channel min-max normalization to [0, 1] for (C, D, H, W).
    Constant channels become 0.
    """
    img = x.astype(np.float32, copy=False)
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        xc = img[c]
        mn = np.nanmin(xc)
        mx = np.nanmax(xc)
        rng = mx - mn
        if (not np.isfinite(rng)) or rng < eps:
            out[c] = 0.0
        else:
            out[c] = (xc - mn) / (rng + eps)
    return out


def save_cdhw_as_nifti(
    image_cdhw: np.ndarray,
    affine: np.ndarray,
    out_path: str,
    *,
    squeeze_single_channel: bool = True,
    dtype: Optional[np.dtype] = np.float32,
) -> None:
    """
    Save a (C, D, H, W) numpy array as NIfTI at out_path using the provided affine.
    Transposes to nibabel NIfTI convention: (H, W, D[, C]).
    """
    img = np.asarray(image_cdhw)
    if img.ndim != 4:
        raise ValueError(f"Expected image with ndim=4 (C,D,H,W), got shape={img.shape}")

    aff = np.asarray(affine)
    if aff.shape != (4, 4):
        raise ValueError(f"Expected affine shape (4,4), got {aff.shape}")

    c, d, h, w = img.shape
    data = img.transpose(2, 3, 1, 0)  # (H, W, D, C)

    if squeeze_single_channel and c == 1:
        data = data[..., 0]  # (H, W, D)

    if dtype is not None:
        data = data.astype(dtype, copy=False)

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    nii = nib.Nifti1Image(data, aff)
    nii.header.set_slope_inter(1.0, 0.0)
    nib.save(nii, out_path)


# -------------------------
# Loader
# -------------------------
class NiftiDataloader:
    def __init__(self, img_dir: str, seg_dir: str, modality: str, return_affine: bool = False):
        """
        Create a dataloader-like iterable over NIfTI images and segmentations.

        Matching rule (kept exactly as before):
        - For each image filename `ip_b`, compute:
          expected_seg_name = ip_b[:-20] + "segmentation.nii.gz"
          and match it against files inside `seg_dir`.
        - Union over:
          (seg, img) pairs + seg-only + img-only
        - __iter__ yields only complete pairs (both exist).

        Iterator yields:
          - (img_arr, seg_arr, sample_id)                         if return_affine == False
          - (img_arr, seg_arr, sample_id, img_affine, seg_affine) if return_affine == True

        Shapes:
          - img_arr: (C, D, H, W), float32, min-max normalized to [0, 1]
          - seg_arr: (1, D, H, W), float32, not normalized
        """
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.modality = modality
        self.return_affine = return_affine

        img_paths = list(glob.iglob(os.path.join(img_dir, "*")))
        seg_paths = list(glob.iglob(os.path.join(seg_dir, "*")))
        img_paths.sort()
        seg_paths.sort()

        # Map: erwarteter seg_path -> img_path
        img_by_expected_seg = {}
        for ip in img_paths:
            ip_b = os.path.basename(ip)
            expected_seg_name = ip_b[:-20] + "segmentation.nii.gz"
            expected_seg_path = os.path.join(seg_dir, expected_seg_name)
            img_by_expected_seg[expected_seg_path] = ip

        union: List[Tuple[Optional[str], Optional[str]]] = []
        matched_imgs = set()

        # segs (und zugehörige imgs falls vorhanden)
        for sp in seg_paths:
            ip = img_by_expected_seg.get(sp)
            if ip is not None:
                union.append((sp, ip))
                matched_imgs.add(ip)
            else:
                union.append((sp, None))

        # imgs ohne seg ergänzen
        for ip in img_paths:
            if ip not in matched_imgs:
                union.append((None, ip))

        self.union_paths = union
        self.sample_infos = self.discover_dataset()

    def __iter__(self) -> Iterator:
        img_ch = int(getattr(self.sample_infos, "channels", 1))

        for seg_path, img_path in self.union_paths:
            # yield only complete pairs (same as before)
            if not img_path or not os.path.exists(img_path):
                continue
            if not seg_path or not os.path.exists(seg_path):
                continue

            img_raw, img_aff = _load_nifti_array_affine(img_path)
            seg_raw, seg_aff = _load_nifti_array_affine(seg_path)

            sample_id = os.path.basename(img_path) if img_path else os.path.basename(seg_path)

            img_arr = ensure_cdhw(img_raw, channels_hint=img_ch).astype(np.float32, copy=False)
            seg_arr = ensure_cdhw(seg_raw, channels_hint=1).astype(np.float32, copy=False)

            #img_arr = minmax01(img_arr)

            sid = Path(Path(sample_id).stem).stem  # remove .nii.gz double-stem

            if self.return_affine:
                yield img_arr, seg_arr, sid, img_aff, seg_aff
            else:
                yield img_arr, seg_arr, sid

    def discover_dataset(self) -> SampleInfo:
        """
        Inspect the first available file on disk to infer dataset properties.
        (kept compatible with previous behavior)
        """
        def _load_meta(p: str):
            obj = nib.load(p)
            try:
                shape = tuple(obj.shape)
                dtype = obj.get_data_dtype()
            finally:
                try:
                    obj.uncache()
                except Exception:
                    pass
            return shape, np.dtype(dtype)

        img_shape = seg_shape = None
        img_dtype = seg_dtype = None

        # image-preferred
        for seg_path, img_path in self.union_paths:
            if img_path is not None and os.path.exists(img_path):
                img_shape, img_dtype = _load_meta(img_path)
                if seg_path is not None and os.path.exists(seg_path):
                    seg_shape, seg_dtype = _load_meta(seg_path)
                break

        # seg-only fallback
        if img_shape is None:
            for seg_path, img_path in self.union_paths:
                if seg_path is not None and os.path.exists(seg_path):
                    seg_shape, seg_dtype = _load_meta(seg_path)
                    break

        if img_shape is None and seg_shape is None:
            raise FileNotFoundError("Could not discover dataset: no existing image or segmentation files found.")

        shape = img_shape if img_shape is not None else seg_shape
        ndim = len(shape)

        if ndim == 2:
            height, width = shape
            depth, channels = 1, 1
        elif ndim == 3:
            # (H, W, D)
            height, width, depth = shape
            channels = 1
        else:
            # (H, W, D, C)
            height, width, depth = shape[0], shape[1], shape[2]
            channels = int(shape[3]) if ndim >= 4 else 1

        if img_dtype is None and seg_dtype is not None:
            img_dtype = seg_dtype
        if seg_dtype is None and img_dtype is not None:
            seg_dtype = img_dtype

        return SampleInfo(
            shape=shape,
            image_dtype=img_dtype,
            mask_dtype=seg_dtype,
            ndim=ndim,
            height=int(height),
            width=int(width),
            depth=int(depth),
            channels=int(channels),
        )
