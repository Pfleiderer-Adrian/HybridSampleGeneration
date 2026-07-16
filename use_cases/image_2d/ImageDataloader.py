import os
import glob
from typing import Dict, Iterator, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional
import numpy as np
from PIL import Image


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
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS


def _load_image_array(path: str) -> np.ndarray:
    """
    Loads an image file as numpy array.
    Returns:
      - grayscale: (H, W)
      - color:     (H, W, C)
    """
    with Image.open(path) as im:
        # Keep as-is; convert palette/LA/etc. to something sane
        # If you want to force RGB, change to: im = im.convert("RGB")
        if im.mode == "P":
            im = im.convert("L")
        arr = np.asarray(im)
    return arr


def ensure_chw(arr: np.ndarray) -> np.ndarray:
    """
    Convert common 2D image shapes to (C, H, W).

    Supports:
      - (H, W)       -> (1, H, W)
      - (H, W, C)    -> (C, H, W)
      - (C, H, W)    -> unchanged
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        H, W = arr.shape
        return arr.reshape(1, H, W)

    if arr.ndim == 3:
        # if already CHW, keep
        # heuristic: if first axis is small (<=4) and last two look like H,W, treat as CHW
        if arr.shape[0] <= 4 and arr.shape[1] > 4 and arr.shape[2] > 4:
            return arr
        # else assume HWC
        return arr.transpose(2, 0, 1)

    raise ValueError(f"Unsupported ndim={arr.ndim}, shape={arr.shape}")


def minmax01_chw(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-channel min-max normalization to [0, 1] for (C, H, W).
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
        controls_only: bool = False,

    ):
        """
        Iterable over 2D image + segmentation pairs on disk.

        Yields:
          - (img_arr, seg_arr, sid)                                if return_paths == False
          - (img_arr, seg_arr, sid, img_path, seg_path)            if return_paths == True

        Shapes:
          - img_arr: (C, H, W), float32, optionally min-max normalized to [0, 1]
          - seg_arr: (1, H, W), float32 (by default)
        """
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.modality = modality
        self.normalize = normalize
        self.return_paths = return_paths
        self.keep_mask_channels = keep_mask_channels
        self.controls_only = controls_only


        img_paths = [p for p in glob.iglob(os.path.join(img_dir, "*")) if _is_image_file(p)]
        seg_paths = [p for p in glob.iglob(os.path.join(seg_dir, "*")) if _is_image_file(p)]
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
        if self.controls_only:
            self.union_paths = self._filter_controls(self.union_paths)
            print("No. of control samples found: ", len(self.union_paths))
        self.sample_infos = self.discover_dataset()

    def _mask_has_anomaly(self, seg_path: str) -> bool:
        seg_raw = _load_image_array(seg_path)

        # Falls RGB-Masken vorkommen: genauso behandeln wie in __iter__
        seg_arr = ensure_chw(seg_raw)
        if (not self.keep_mask_channels) and seg_arr.shape[0] > 1:
            seg_arr = seg_arr[:1]

        # "Anomalie" = irgendein Pixel > 0
        return np.any(seg_arr > 0)

    def _filter_controls(self, pairs):
        out = []
        for seg_path, img_path in pairs:
            if not img_path or not seg_path:
                continue
            if not (os.path.exists(img_path) and os.path.exists(seg_path)):
                continue

            if not self._mask_has_anomaly(seg_path):
                out.append((seg_path, img_path))
        return out

    def __iter__(self) -> Iterator:
        for seg_path, img_path in self.union_paths:
            # yield only complete pairs (same behavior as before)
            if not img_path or not os.path.exists(img_path):
                continue
            if not seg_path or not os.path.exists(seg_path):
                continue

            img_raw = _load_image_array(img_path)
            seg_raw = _load_image_array(seg_path)

            img_arr = ensure_chw(img_raw).astype(np.float32, copy=False)
            seg_arr = ensure_chw(seg_raw).astype(np.float32, copy=False)

            # optional: if mask is RGB, reduce to 1 channel (common for label PNGs that are saved as RGB)
            if (not self.keep_mask_channels) and seg_arr.shape[0] > 1:
                seg_arr = seg_arr[:1]

            if self.normalize:
                img_arr = minmax01_chw(img_arr)

            sid = os.path.basename(img_path)

            if self.return_paths:
                yield img_arr, seg_arr, sid, img_path, seg_path
            else:
                yield img_arr, seg_arr, sid

    def discover_dataset(self) -> SampleInfo:
        """
        Inspect first available pair (image preferred) to infer dataset properties.
        """
        img_shape = seg_shape = None
        img_dtype = seg_dtype = None

        def _meta(p: str):
            arr = _load_image_array(p)
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


def save_image(arr_chw: np.ndarray, filepath: Union[str, Path], *, clamp: bool = True, quality: int = 95) -> None:
    """
    Save a channel-first image array (C,H,W) to disk.

    Supports:
      - C=1 (grayscale)
      - C=3 (RGB)
      - C=4 (RGBA)  -> PNG recommended (JPEG can't store alpha)

    Dtypes / ranges:
      - float: assumed in [0,1] or [0,255]; will be scaled to uint8
      - int/uint: will be converted/clipped to uint8

    Parameters
    ----------
    arr_chw : np.ndarray
        Image array with shape (C,H,W).
    filepath : str | Path
        Output file path (extension determines format).
    clamp : bool
        If True, clip values before conversion (recommended).
    quality : int
        JPEG quality (only used for .jpg/.jpeg).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(arr_chw)
    if arr.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got shape={arr.shape}")

    C, H, W = arr.shape
    if C not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count C={C}. Expected 1, 3, or 4.")

    # Convert to HWC for PIL when needed
    if C == 1:
        img = arr[0]  # (H,W)
    else:
        img = arr.transpose(1, 2, 0)  # (H,W,C)

    # Convert to uint8
    if np.issubdtype(img.dtype, np.floating):
        m = float(np.nanmax(img)) if img.size else 0.0
        if m <= 1.0:
            img = img * 255.0
        if clamp:
            img = np.clip(img, 0.0, 255.0)
        img_u8 = img.astype(np.uint8)
    else:
        if clamp:
            img = np.clip(img, 0, 255)
        img_u8 = img.astype(np.uint8)

    # Choose PIL mode
    if C == 1:
        pil = Image.fromarray(img_u8, mode="L")
    elif C == 3:
        pil = Image.fromarray(img_u8, mode="RGB")
    else:  # C == 4
        pil = Image.fromarray(img_u8, mode="RGBA")

    ext = filepath.suffix.lower()
    save_kwargs = {}

    # JPEG can't do alpha; drop alpha if user still wants .jpg/.jpeg
    if ext in (".jpg", ".jpeg") and C == 4:
        pil = pil.convert("RGB")
        save_kwargs["quality"] = int(quality)
        save_kwargs["subsampling"] = 0  # nicer JPEGs

    if ext in (".jpg", ".jpeg"):
        save_kwargs.setdefault("quality", int(quality))
        save_kwargs.setdefault("subsampling", 0)

    pil.save(str(filepath), **save_kwargs)
