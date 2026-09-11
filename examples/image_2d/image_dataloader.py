"""Load paired channel-first images for the two-dimensional example."""

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from examples.common.image_io import (
    ensure_chw,
    is_image_file,
    load_image_array,
    minmax01_chw,
)
from hybrid_sample_generator.domain.input_sample import InputSample


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
    channels: int


# -------------------------
# Helpers
# -------------------------
def _stem_no_ext(p: str) -> str:
    return Path(p).stem


def _strip_suffix_token(stem: str, tokens: Tuple[str, ...]) -> str:
    """
    Remove common suffix tokens at the end of a stem:
      e.g. "case_001_segmentation" -> "case_001"
           "case_001_mask"         -> "case_001"
    """
    s = stem
    for t in tokens:
        if s.endswith("_" + t):
            s = s[: -(len(t) + 1)]
        elif s.endswith(t):
            # if someone saved as "case001segmentation"
            s = s[: -len(t)]
    return s


def _key_for_image(stem: str, modality: str) -> str:
    """
    Build a matching key for images.
    If stem endswith _{modality}, strip it.
    """
    s = stem
    if modality:
        s = _strip_suffix_token(s, (modality,))
    return s


def _key_for_mask(stem: str) -> str:
    """
    Build a matching key for masks/segmentations.
    Strips common suffixes.
    """
    return _strip_suffix_token(stem, ("segmentation", "seg", "mask", "label", "labels"))


# -------------------------
# Loader
# -------------------------
class ImageDataloader:
    def __init__(
        self,
        img_dir: str,
        seg_dir: str,
        modality: str = "",
        *,
        normalize: bool = False,
        return_paths: bool = False,
        keep_mask_channels: bool = False,
    ):
        """
        Iterable over 2D image + segmentation pairs on disk.

        Yields:
          - (img_arr, seg_arr_or_none, sid)                         if return_paths == False
          - (img_arr, seg_arr_or_none, sid, img_path, seg_path)     if return_paths == True

        Shapes:
          - img_arr: (C, H, W), float32, optionally min-max normalized to [0, 1]
          - seg_arr_or_none: (1, H, W), float32 (by default)
        """
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.modality = modality
        self.normalize = normalize
        self.return_paths = return_paths
        self.keep_mask_channels = keep_mask_channels


        img_paths = [p for p in glob.iglob(os.path.join(img_dir, "*")) if is_image_file(p)]
        seg_paths = [p for p in glob.iglob(os.path.join(seg_dir, "*")) if is_image_file(p)]
        img_paths.sort()
        seg_paths.sort()

        # Build lookup by key
        img_by_key: Dict[str, str] = {}
        for ip in img_paths:
            key = _key_for_image(_stem_no_ext(ip), modality)
            img_by_key[key] = ip

        seg_by_key: Dict[str, str] = {}
        for sp in seg_paths:
            key = _key_for_mask(_stem_no_ext(sp))
            seg_by_key[key] = sp

        union: List[Tuple[Optional[str], Optional[str]]] = []
        matched_imgs = set()

        for key, sp in seg_by_key.items():
            ip = img_by_key.get(key)
            if ip is not None:
                union.append((sp, ip))
                matched_imgs.add(ip)
            else:
                union.append((sp, None))

        for ip in img_paths:
            if ip not in matched_imgs:
                union.append((None, ip))

        self.union_paths = union
        self.sample_infos = self.discover_dataset()

    def __iter__(self) -> Iterator:
        for seg_path, img_path in self.union_paths:
            if not img_path or not os.path.exists(img_path):
                continue

            img_raw = load_image_array(img_path)
            img_arr = ensure_chw(img_raw).astype(np.float32, copy=False)
            seg_arr = None
            if seg_path and os.path.exists(seg_path):
                seg_raw = load_image_array(seg_path)
                seg_arr = ensure_chw(seg_raw).astype(np.float32, copy=False)
                if (not self.keep_mask_channels) and seg_arr.shape[0] > 1:
                    seg_arr = seg_arr[:1]

            if self.normalize:
                img_arr = minmax01_chw(img_arr)

            sid = os.path.basename(img_path)

            if self.return_paths:
                yield img_arr, seg_arr, sid, img_path, seg_path
            else:
                yield img_arr, seg_arr, sid

    def iter_input_samples(self) -> Iterator[InputSample]:
        """Yield the typed pipeline boundary including stable source paths."""
        for seg_path, img_path in self.union_paths:
            if not img_path or not os.path.exists(img_path):
                continue
            img_arr = ensure_chw(load_image_array(img_path)).astype(np.float32, copy=False)
            seg_arr = None
            if seg_path and os.path.exists(seg_path):
                seg_arr = ensure_chw(load_image_array(seg_path)).astype(
                    np.float32, copy=False
                )
                if (not self.keep_mask_channels) and seg_arr.shape[0] > 1:
                    seg_arr = seg_arr[:1]
            if self.normalize:
                img_arr = minmax01_chw(img_arr)
            yield InputSample(
                image=img_arr,
                segmentation=seg_arr,
                source_name=os.path.basename(img_path),
                source_image_path=img_path,
                source_segmentation_path=seg_path,
            )

    def discover_dataset(self) -> SampleInfo:
        """
        Inspect first available pair (image preferred) to infer dataset properties.
        """
        img_shape = seg_shape = None
        img_dtype = seg_dtype = None

        def _meta(p: str):
            arr = load_image_array(p)
            return tuple(arr.shape), np.dtype(arr.dtype)

        # image-preferred
        for seg_path, img_path in self.union_paths:
            if img_path is not None and os.path.exists(img_path):
                img_shape, img_dtype = _meta(img_path)
                if seg_path is not None and os.path.exists(seg_path):
                    seg_shape, seg_dtype = _meta(seg_path)
                break

        # seg-only fallback
        if img_shape is None:
            for seg_path, _ in self.union_paths:
                if seg_path is not None and os.path.exists(seg_path):
                    seg_shape, seg_dtype = _meta(seg_path)
                    break

        if img_shape is None and seg_shape is None:
            raise FileNotFoundError("Could not discover dataset: no existing image or segmentation files found.")

        shape = img_shape if img_shape is not None else seg_shape
        ndim = len(shape)

        if ndim == 2:
            height, width = shape
            channels = 1
        elif ndim == 3:
            height, width, channels = shape
        else:
            raise ValueError(f"Unsupported image ndim={ndim}, shape={shape}")

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
            channels=int(channels),
        )
