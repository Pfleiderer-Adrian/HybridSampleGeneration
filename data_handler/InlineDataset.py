from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class InlineImageSegDataset(Dataset):
    """
    PyTorch Dataset, das Paare (image, segmentation) aus zwei Ordnern lädt.

    Voraussetzungen:
    - Beide Ordner enthalten `.npy` Dateien
    - Image und Segmentation haben exakt denselben Dateinamen (basename),
      z.B.:
        images/sample_0.npy
        segs/sample_0.npy

    Return format:
    - return_filename=False -> (img, seg)
    - return_filename=True  -> (img, seg, fname)
      wobei fname der basename inkl. Extension ist (z.B. "sample_0.npy")

    Optional:
    - load_to_ram=True lädt alles beim Start in den RAM.
    - mmap_mode unterstützt Memory Mapping (nur wenn load_to_ram=False).
    - numpy_mode=True gibt numpy arrays zurück statt torch tensors.
    """

    def __init__(
        self,
        image_folder: Union[str, os.PathLike],
        seg_folder: Union[str, os.PathLike],
        *,
        return_filename: bool = True,
        dtype: torch.dtype = torch.float32,
        seg_dtype: Optional[torch.dtype] = None,
        transform: Optional[
            Callable[[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]],
                     Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]]
        ] = None,
        recursive: bool = False,
        extensions: Tuple[str, ...] = (".npy",),
        sort: bool = True,
        load_to_ram: bool = False,
        mmap_mode: Optional[str] = None,
        numpy_mode: bool = False,
        strict: bool = True,
    ) -> None:
        """
        Inputs
        ------
        image_folder:
            Ordner mit Image `.npy` Files.
        seg_folder:
            Ordner mit Segmentation `.npy` Files.
        return_filename:
            Wenn True: (img, seg, fname), sonst (img, seg).
        dtype:
            Torch dtype für image.
        seg_dtype:
            Torch dtype für segmentation. Wenn None -> verwendet dtype.
            (Typischerweise seg_dtype=torch.int64 bei Klassenlabels)
        transform:
            Optionaler Transform auf beide:
              img, seg = transform(img, seg)
            In numpy_mode=True bekommen transforms numpy arrays.
        recursive:
            rglob statt glob.
        extensions:
            erlaubte Endungen, default (".npy",)
        sort:
            deterministische Reihenfolge.
        load_to_ram:
            preload beider Ordner in RAM.
        mmap_mode:
            np.load(..., mmap_mode=...) wenn load_to_ram=False.
        numpy_mode:
            Wenn True: gibt numpy arrays zurück statt torch tensors.
        strict:
            Wenn True: wir nehmen nur Paare, die in BEIDEN Ordnern existieren.
            Wenn False: wir erlauben "missing seg" NICHT empfohlen (würde None erfordern).
        """
        self.image_folder = Path(image_folder).expanduser().resolve()
        self.seg_folder = Path(seg_folder).expanduser().resolve()

        if not self.image_folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")
        if not self.seg_folder.is_dir():
            raise FileNotFoundError(f"Seg folder not found: {self.seg_folder}")

        self.return_filename = return_filename
        self.dtype = dtype
        self.seg_dtype = seg_dtype if seg_dtype is not None else dtype
        self.transform = transform
        self.recursive = recursive
        self.extensions = tuple(e.lower() for e in extensions)
        self.sort = sort
        self.load_to_ram = load_to_ram
        self.numpy_mode = numpy_mode

        # mmap_mode nur relevant wenn NICHT preload
        self.mmap_mode = None if load_to_ram else mmap_mode

        # Build basename -> fullpath maps
        img_map = self._collect_map(self.image_folder)
        seg_map = self._collect_map(self.seg_folder)

        # Intersection by basename
        common = sorted(set(img_map.keys()) & set(seg_map.keys())) if sort else list(
            set(img_map.keys()) & set(seg_map.keys())
        )

        if not common:
            raise FileNotFoundError(
                "No matching image/seg pairs found.\n"
                f"image_folder={self.image_folder}\n"
                f"seg_folder={self.seg_folder}\n"
                f"extensions={self.extensions}"
            )

        if strict:
            self._filenames = common
        else:
            # (nicht wirklich sinnvoll ohne None handling – daher strict default True)
            self._filenames = common

        # Paired paths
        self._img_paths: List[str] = [img_map[f] for f in self._filenames]
        self._seg_paths: List[str] = [seg_map[f] for f in self._filenames]

        # Optional preload RAM
        self._img_ram: Optional[List[np.ndarray]] = None
        self._seg_ram: Optional[List[np.ndarray]] = None

        if self.load_to_ram:
            self._img_ram = []
            self._seg_ram = []
            for ip, sp in zip(self._img_paths, self._seg_paths):
                self._img_ram.append(np.load(ip, allow_pickle=False))
                self._seg_ram.append(np.load(sp, allow_pickle=False))

    def _collect_map(self, folder: Path) -> Dict[str, str]:
        iterator = folder.rglob("*") if self.recursive else folder.glob("*")
        files = [
            p for p in iterator
            if p.is_file() and p.suffix.lower() in self.extensions
        ]
        if not files:
            raise FileNotFoundError(
                f"No files with extensions {self.extensions} found in: {folder}"
            )

        # basename->path
        m: Dict[str, str] = {}
        for p in files:
            base = p.name
            # Wenn gleiche basenames in recursive mode vorkommen -> ambiguous
            if base in m:
                raise ValueError(
                    f"Duplicate basename detected in folder {folder} (recursive={self.recursive}): {base}"
                )
            m[base] = str(p.resolve())
        return m

    def __len__(self) -> int:
        return len(self._filenames)

    def _to_tensor(self, x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        # Minimal-copy conversion
        if dtype == torch.bfloat16:
            x_c = np.asarray(x, dtype=np.float32, order="C")
            return torch.from_numpy(x_c).to(torch.bfloat16)

        # numpy dtype mapping
        if dtype == torch.float32:
            x_c = np.asarray(x, dtype=np.float32, order="C")
            return torch.from_numpy(x_c)
        if dtype == torch.float16:
            x_c = np.asarray(x, dtype=np.float16, order="C")
            return torch.from_numpy(x_c)
        if dtype == torch.float64:
            x_c = np.asarray(x, dtype=np.float64, order="C")
            return torch.from_numpy(x_c)
        if dtype == torch.int64:
            x_c = np.asarray(x, dtype=np.int64, order="C")
            return torch.from_numpy(x_c)
        if dtype == torch.int32:
            x_c = np.asarray(x, dtype=np.int32, order="C")
            return torch.from_numpy(x_c)

        x_c = np.asarray(x, dtype=np.float32, order="C")
        return torch.from_numpy(x_c).to(dtype)

    def __getitem__(self, idx: int):
        fname = self._filenames[idx]

        # load image + seg
        if self._img_ram is not None and self._seg_ram is not None:
            img_np = self._img_ram[idx]
            seg_np = self._seg_ram[idx]
        else:
            img_np = np.load(self._img_paths[idx], allow_pickle=False, mmap_mode=self.mmap_mode)
            seg_np = np.load(self._seg_paths[idx], allow_pickle=False, mmap_mode=self.mmap_mode)

        # convert to tensors unless numpy_mode
        if self.numpy_mode:
            img = img_np
            seg = seg_np
        else:
            img = self._to_tensor(img_np, self.dtype)
            seg = self._to_tensor(seg_np, self.seg_dtype)

        # transform hook: expects (img, seg) -> (img, seg)
        if self.transform is not None:
            img, seg = self.transform(img, seg)

        if self.return_filename:
            return img, seg, fname
        return img, seg

    def load_numpy_by_basename(self, basename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lade (img, seg) als numpy arrays über basename, z.B. "foo.npy" oder "foo".
        """
        key = os.path.basename(str(basename))

        # allow "foo" -> "foo.npy"
        if not key.lower().endswith(".npy"):
            key = key + ".npy"

        try:
            idx = self._filenames.index(key)
        except ValueError:
            raise KeyError(f"basename not found in paired dataset: {basename}")

        if self._img_ram is not None and self._seg_ram is not None:
            return self._img_ram[idx], self._seg_ram[idx]

        img_np = np.load(self._img_paths[idx], allow_pickle=False, mmap_mode=self.mmap_mode)
        seg_np = np.load(self._seg_paths[idx], allow_pickle=False, mmap_mode=self.mmap_mode)
        return img_np, seg_np
