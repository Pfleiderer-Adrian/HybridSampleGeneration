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
    PyTorch Dataset that loads 3D samples stored as `.npy` files from a folder.

    Key behavior:
    - The dataset is populated *only* from a folder (no manual add_sample/add_path).
    - File format: `.npy` (not NIfTI).
    - Optional: preload everything into RAM (load_to_ram=True).

    Return format:
    - return_filename=False -> x
    - return_filename=True  -> (x, fname)
      where fname is the basename including extension (e.g., "sample_0.npy").
    """

    def __init__(
        self,
        folder: Union[str, os.PathLike],
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
        return_filename:
            If True, __getitem__ returns (x, filename). Otherwise returns x only.
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

        Outputs
        -------
        None
            Side effects:
              - scans the folder and stores file paths
              - builds fast lookup dicts for basename and stem
              - optionally preloads arrays into RAM
        """
        # Resolve folder path
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(str(self.folder))

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
        if self.load_to_ram:
            self._ram_arrays = []
            for p in self._paths:
                # allow_pickle=False for safety; loads full array into memory
                self._ram_arrays.append(np.load(p, allow_pickle=False))

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
            (x, fname)
              - x: torch.Tensor (default) or np.ndarray (if numpy_mode=True)
              - fname: str basename (e.g. "foo.npy")
        Else:
            x only

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

        # Optional transform hook
        if self.transform is not None:
            x = self.transform(x)

        # Return format
        if self.return_filename:
            return x, fname
        return x

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
