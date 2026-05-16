from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def save_numpy_as_npy(
    array: np.ndarray,
    out_path: Union[str, os.PathLike],
    *,
    overwrite: bool = False,
    create_dirs: bool = True,
) -> str:
    """
    Save a NumPy array as a `.npy` file.

    Inputs
    ------
    array:
        NumPy array to be saved.
    out_path:
        Target path including filename (ideally ends with `.npy`).
        If the suffix is not `.npy`, it will be replaced with `.npy`.
    overwrite:
        If False and the target file already exists, a FileExistsError is raised.
    create_dirs:
        If True, create parent directories if they do not exist.

    Outputs
    -------
    str:
        Absolute path to the saved file.

    Raises
    ------
    FileExistsError:
        If the output file exists and overwrite=False.
    """
    # Normalize/resolve path (supports "~" and relative paths)
    out_path = Path(out_path).expanduser().resolve()

    # Enforce `.npy` suffix
    if out_path.suffix.lower() != ".npy":
        out_path = out_path.with_suffix(".npy")

    # Optionally create parent folder(s)
    if create_dirs:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Safety check for overwriting
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {out_path}")

    # Save array to .npy (pickle disabled for safety)
    np.save(out_path, array, allow_pickle=False)
    return str(out_path)


class AnomalyDataset(Dataset):
    """
    PyTorch Dataset that loads 2D or 3D samples stored as `.npy` files from a folder.

    Key behavior:
    - The dataset is populated *only* from a folder (no manual add_sample/add_path).
    - File format: `.npy` (not NIfTI).
    - Optional: preload everything into RAM (load_to_ram=True).

    Return format:
    - If return_filename=False -> returns x (or x, org_mask, tgt_mask)
    - If return_filename=True  -> returns (x, fname) (or x, org_mask, tgt_mask, fname)
      where fname is the basename including extension (e.g., "sample_0.npy").
    """

    def __init__(
        self,
        folder: Union[str, os.PathLike],
        org_mask_folder: Optional[Union[str, os.PathLike]] = None,
        tgt_mask_folder: Optional[Union[str, os.PathLike]] = None,
        *,
        return_filename: bool = True,
        dtype: torch.dtype = torch.float32,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        recursive: bool = False,
        extensions: Tuple[str, ...] = (".npy",),
        sort: bool = True,
        load_to_ram: bool = False,
        mmap_mode: Optional[str] = None,
        numpy_mode: bool = False,
    ) -> None:
        """
        Initialize the dataset and discover `.npy` files.

        Inputs
        ------
        folder:
            Path to a directory containing `.npy` files.
        org_mask_folder:
            Optional path to a directory containing the original masks as `.npy` files. (multiclass)
        tgt_mask_folder:
            Optional path to a directory containing the target masks as `.npy` files. (multiclass)
            If org_mask_folder is None, tgt_mask_folder has no effect.
            If org_mask_folder is not None but tgt_mask_folder is None: tgt_mask_folder = org_mask_folder.
        return_filename:
            If True, __getitem__ returns the filename at the end of the tuple.
        dtype:
            Torch dtype used when converting numpy arrays into torch tensors.
            (Ignored if numpy_mode=True.)
        transform:
            Optional transform applied to x after loading.
            NOTE: In numpy_mode=True, transform will receive a NumPy array (not a torch tensor).
        recursive:
            If True, search recursively (rglob). Otherwise only the top-level folder.
        extensions:
            Allowed file extensions (default: (".npy",)).
        sort:
            If True, sort file paths for deterministic ordering.
        load_to_ram:
            If True, preload all `.npy` arrays into memory during initialization.
        mmap_mode:
            Passed to np.load(..., mmap_mode=...) when load_to_ram=False.
            NOTE: If load_to_ram=True, mmap_mode is ignored (set to None).
        numpy_mode:
            If True, __getitem__ returns numpy arrays instead of torch tensors.
        """
        # Resolve folder path
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(str(self.folder))

        self.org_mask_folder = None
        self.tgt_mask_folder = None
        if org_mask_folder is not None:
            self.org_mask_folder = Path(org_mask_folder).expanduser().resolve()
            if not self.org_mask_folder.is_dir():
                raise FileNotFoundError(f"Mask folder not found: {self.org_mask_folder}")
            if tgt_mask_folder is None: 
                tgt_mask_folder = org_mask_folder
            self.tgt_mask_folder = Path(tgt_mask_folder).expanduser().resolve()

        # Store configuration flags
        self.numpy_mode = numpy_mode
        self.return_filename = return_filename
        self.dtype = dtype
        self.transform = transform
        self.recursive = recursive
        self.extensions = tuple(e.lower() for e in extensions)
        self.sort = sort
        self.load_to_ram = load_to_ram

        # mmap_mode is only relevant when we DO NOT preload into RAM
        self.mmap_mode = None if load_to_ram else mmap_mode

        # Collect all .npy files into a list
        self._paths: List[str] = self._collect_paths()

        # Fast lookup tables:
        # - basename -> [indices]
        # - stem (filename without extension) -> [indices]
        # These support load_numpy_by_basename() lookups.
        self._basename_to_indices: dict[str, list[int]] = {}
        self._stem_to_indices: dict[str, list[int]] = {}

        for idx, p in enumerate(self._paths):
            base = os.path.basename(p)      # e.g. "foo.npy"
            stem = Path(base).stem          # e.g. "foo"
            self._basename_to_indices.setdefault(base, []).append(idx)
            self._stem_to_indices.setdefault(stem, []).append(idx)

        # Optional preload into RAM for faster access during training/inference
        self._ram_arrays: Optional[List[np.ndarray]] = None
        self._ram_org_masks: Optional[List[np.ndarray]] = None
        self._ram_tgt_masks: Optional[List[np.ndarray]] = None

        if self.load_to_ram:
            self._ram_arrays = []
            
            if self.org_mask_folder:
                self._ram_org_masks = []
                self._ram_tgt_masks = []
                same_mask_folders = (self.org_mask_folder == self.tgt_mask_folder)
                
            for p in self._paths:
                # allow_pickle=False for safety; loads full array into memory
                self._ram_arrays.append(np.load(p, allow_pickle=False))
                
                # Load masks if folder is provided
                if self.org_mask_folder:
                    org_mask_path = os.path.join(self.org_mask_folder, os.path.basename(p))
                    if not os.path.exists(org_mask_path):
                        raise FileNotFoundError(f"Expected org_mask file missing: {org_mask_path}")
                    
                    loaded_org_mask = np.load(org_mask_path, allow_pickle=False)
                    self._ram_org_masks.append(loaded_org_mask)

                    if same_mask_folders:
                        self._ram_tgt_masks.append(loaded_org_mask)
                    else:
                        tgt_mask_path = os.path.join(self.tgt_mask_folder, os.path.basename(p))
                        if not os.path.exists(tgt_mask_path):
                            raise FileNotFoundError(f"Expected tgt_mask file missing: {tgt_mask_path}")
                        self._ram_tgt_masks.append(np.load(tgt_mask_path, allow_pickle=False))

    def _collect_paths(self) -> List[str]:
        """
        Collect eligible file paths from the dataset folder.

        Inputs
        ------
        None (uses self.folder, self.recursive, self.extensions)

        Outputs
        -------
        List[str]
            List of absolute paths to matching files.

        Raises
        ------
        FileNotFoundError
            If no matching files are found.
        """
        # Choose iterator depending on recursive search
        iterator = self.folder.rglob("*") if self.recursive else self.folder.glob("*")

        # Keep only files matching allowed extensions
        paths = [
            str(p)
            for p in iterator
            if p.is_file() and p.suffix.lower() in self.extensions
        ]

        # Optional sorting for deterministic dataset ordering
        if self.sort:
            paths.sort()

        # Fail early if dataset folder is empty/misconfigured
        if not paths:
            raise FileNotFoundError(
                f"No files with extensions {self.extensions} found in: {self.folder}"
            )

        return paths

    def __len__(self) -> int:
        """
        Number of samples in the dataset.

        Inputs
        ------
        None

        Outputs
        -------
        int
            Number of discovered `.npy` files.
        """
        return len(self._paths)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a torch tensor with a controlled dtype and contiguous layout.

        Inputs
        ------
        x:
            np.ndarray

        Outputs
        -------
        torch.Tensor
            Tensor that shares memory with the contiguous numpy array when possible.

        Notes
        -----
        - For bfloat16: numpy doesn't support bfloat16, so it loads as float32 then casts to bfloat16.
        - This function is ignored when numpy_mode=True.
        """
        # Convert to contiguous array with an explicit dtype, then wrap with torch.from_numpy.
        if self.dtype == torch.float32:
            x_c = np.asarray(x, dtype=np.float32, order="C")
            return torch.from_numpy(x_c)
        if self.dtype == torch.float16:
            x_c = np.asarray(x, dtype=np.float16, order="C")
            return torch.from_numpy(x_c)
        if self.dtype == torch.float64:
            x_c = np.asarray(x, dtype=np.float64, order="C")
            return torch.from_numpy(x_c)
        if self.dtype == torch.int64:
            x_c = np.asarray(x, dtype=np.int64, order="C")
            return torch.from_numpy(x_c)
        if self.dtype == torch.int32:
            x_c = np.asarray(x, dtype=np.int32, order="C")
            return torch.from_numpy(x_c)
        if self.dtype == torch.bfloat16:
            x_c = np.asarray(x, dtype=np.float32, order="C")
            return torch.from_numpy(x_c).to(torch.bfloat16)

        # Fallback: convert via float32 then cast to requested dtype
        x_c = np.asarray(x, dtype=np.float32, order="C")
        return torch.from_numpy(x_c).to(self.dtype)

    def __getitem__(self, idx: int):
        """
        Load one sample by index.

        Inputs
        ------
        idx:
            Index in [0, len(self)-1].

        Outputs
        -------
        If return_filename == True:
            (x, [org_mask, tgt_mask,] fname)
        Else:
            x or (x, org_mask, tgt_mask)

        Notes
        -----
        - If load_to_ram=True, loads from self._ram_arrays (fast).
        - Otherwise loads from disk via np.load(..., mmap_mode=self.mmap_mode).
        - If transform is provided:
            - numpy_mode=False: transform receives torch.Tensor
            - numpy_mode=True : transform receives np.ndarray
        """
        path = self._paths[idx]
        fname = os.path.basename(path)

        # Load from RAM if preloaded, otherwise load from disk (optionally memory-mapped)
        if self._ram_arrays is not None:
            img_np = self._ram_arrays[idx]
        else:
            img_np = np.load(path, allow_pickle=False, mmap_mode=self.mmap_mode)

        # Convert to torch tensor unless numpy_mode is enabled
        if not self.numpy_mode:
            x = self._to_tensor(img_np)
        else:
            x = img_np

        # Optional transform hook (Intensity transforms only, don't apply to masks)
        if self.transform is not None:
            x = self.transform(x)

        if self.org_mask_folder is None:    # no multiclass
            if self.return_filename:
                return x, fname
            return x

        # multiclass -> loads masks
        if self._ram_org_masks is not None:
            org_mask_np = self._ram_org_masks[idx]
            tgt_mask_np = self._ram_tgt_masks[idx]
        else:
            org_mask_path = os.path.join(self.org_mask_folder, fname)
            org_mask_np = np.load(org_mask_path, allow_pickle=False, mmap_mode=self.mmap_mode)
            
            if self.org_mask_folder == self.tgt_mask_folder:
                tgt_mask_np = org_mask_np
            else:
                tgt_mask_path = os.path.join(self.tgt_mask_folder, fname)
                tgt_mask_np = np.load(tgt_mask_path, allow_pickle=False, mmap_mode=self.mmap_mode)

        org_mask_tensor = self._to_tensor(org_mask_np)
        tgt_mask_tensor = self._to_tensor(tgt_mask_np)

        # remove channel dim for masks
        org_mask_tensor = org_mask_tensor.long()
        if org_mask_tensor.shape[0] == 1:
            org_mask_tensor = org_mask_tensor.squeeze(0)
            
        tgt_mask_tensor = tgt_mask_tensor.long()
        if tgt_mask_tensor.shape[0] == 1:
            tgt_mask_tensor = tgt_mask_tensor.squeeze(0)

        if self.numpy_mode:
            org_mask_out = org_mask_tensor.numpy()
            tgt_mask_out = tgt_mask_tensor.numpy()
        else:
            org_mask_out = org_mask_tensor
            tgt_mask_out = tgt_mask_tensor

        if self.return_filename:
            return x, org_mask_out, tgt_mask_out, fname
            
        return x, org_mask_out, tgt_mask_out

    def load_numpy_by_basename(self, basename: str) -> np.ndarray:
        """
        Load a `.npy` sample by its basename (filename) and return the NumPy array.

        The array is retrieved either from:
          - RAM (if load_to_ram=True), or
          - disk (np.load with mmap_mode)

        Inputs
        ------
        basename:
            Basename may be provided with or without extension:
              - "foo.npy"  (exact basename lookup)
              - "foo"      (stem lookup)

        Outputs
        -------
        np.ndarray
            The loaded sample as a NumPy array.

        Raises
        ------
        KeyError
            If no matching basename/stem is found.
        ValueError
            If the basename/stem is ambiguous (multiple matches),
            which can happen in recursive mode if duplicate names exist.
        """
        # Normalize input to just the basename portion
        key = os.path.basename(str(basename))

        # 1) Try exact basename lookup (e.g. "foo.npy")
        candidates = self._basename_to_indices.get(key, [])

        # 2) If not found, try stem lookup (e.g. "foo")
        if not candidates:
            stem = Path(key).stem
            candidates = self._stem_to_indices.get(stem, [])

        # No hits
        if not candidates:
            raise KeyError(f"basename not found: {basename}")

        # Multiple hits => ambiguous
        if len(candidates) > 1:
            names = [os.path.basename(self._paths[i]) for i in candidates]
            raise ValueError(f"basename is ambiguous ({basename}); matches: {names}")

        # Load from RAM or disk based on preload setting
        idx = candidates[0]
        if self._ram_arrays is not None:
            return self._ram_arrays[idx]

        return np.load(self._paths[idx], allow_pickle=False, mmap_mode=self.mmap_mode)
    