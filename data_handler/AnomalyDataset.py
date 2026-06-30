from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from synthesizer.StudyPaths import StudyPaths


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
    PyTorch Dataset that loads selected array and metadata artifacts for one study.

    Key behavior:
    - StudyPaths defines all artifact folders.
    - Only artifacts listed in return_artifacts are loaded.
    - Arrays are loaded from `.npy`; anomaly_meta is loaded from JSON.
    - Optional: preload selected artifacts into RAM (load_to_ram=True).

    Return format:
    - {"img": img, "fname": fname, "ori_mask": ..., ...}

    Supported artifact names:
    - img
    - fname
    - ori_mask
    - tgt_mask
    - anomaly_roi
    - anomaly_roi_mask
    - anomaly_meta
    - synth_anomaly
    - synth_roi
    """

    ALLOWED_ARTIFACTS = (
        "img",
        "fname",
        "ori_mask",
        "tgt_mask",
        "anomaly_roi",
        "anomaly_roi_mask",
        "anomaly_meta",
        "synth_anomaly",
        "synth_roi",
    )
    LOADABLE_ARTIFACTS = tuple(
        artifact
        for artifact in ALLOWED_ARTIFACTS
        if artifact not in ("fname", "anomaly_meta")
    )
    STUDY_PATH_ATTRIBUTES = {
        "img": "anomaly_data",
        "ori_mask": "anomaly_mask_data",
        "tgt_mask": "anomaly_tgt_mask_data",
        "anomaly_roi": "anomaly_roi_data",
        "anomaly_roi_mask": "anomaly_mask_roi_data",
        "synth_anomaly": "synth_anomaly_data",
        "synth_roi": "synth_roi_data",
    }
    ARTIFACT_ALIASES = {
        "x": "img",
        "image": "img",
        "images": "img",
        "anomaly": "img",
        "anomaly_data": "img",
        "org_mask": "ori_mask",
        "orig_mask": "ori_mask",
        "original_mask": "ori_mask",
        "mask": "ori_mask",
        "anomaly_mask": "ori_mask",
        "anomaly_mask_data": "ori_mask",
        "target_mask": "tgt_mask",
        "anomaly_tgt_mask": "tgt_mask",
        "anomaly_tgt_mask_data": "tgt_mask",
        "anomaly_roi_data": "anomaly_roi",
        "anomaly_mask_roi": "anomaly_roi_mask",
        "anomaly_mask_roi_data": "anomaly_roi_mask",
        "meta": "anomaly_meta",
        "metadata": "anomaly_meta",
        "anomaly_metadata": "anomaly_meta",
        "synthetic_anomaly": "synth_anomaly",
        "synthetic_anomaly_data": "synth_anomaly",
        "synth_anomaly_data": "synth_anomaly",
        "synth_roi_data": "synth_roi",
    }

    def __init__(
        self,
        study_paths: StudyPaths,
        *,
        return_artifacts: Optional[Union[str, Sequence[str]]] = None,
        index_artifact: Optional[str] = None,
        anomaly_meta_file: Optional[Union[str, os.PathLike]] = None,
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
        study_paths:
            StudyPaths-like object used for all artifact folders.
        return_artifacts:
            Multi-selection of artifacts to return. If set, __getitem__ returns
            a dict with exactly these keys. If omitted, ("img", "fname") is used.
        index_artifact:
            Artifact used to define dataset length/order and fname. If omitted,
            the first selected loadable artifact is used.
        anomaly_meta_file:
            Optional path to the anomaly transformation JSON. If omitted and
            anomaly_meta is requested, study_paths.anomaly_transformations_file
            is used.
        dtype:
            Torch dtype used when converting numpy arrays into torch tensors.
            (Ignored if numpy_mode=True.)
        transform:
            Optional transform applied to the img artifact after loading.
            NOTE: In numpy_mode=True, transform will receive a NumPy array (not a torch tensor).
        recursive:
            If True, search recursively (rglob). Otherwise only the top-level folder.
            synth_roi is always scanned recursively because the pipeline stores it
            under synth_roi_data/<control>/<anomaly>.npy.
        extensions:
            Allowed file extensions (default: (".npy",)).
        sort:
            If True, sort file paths for deterministic ordering.
        load_to_ram:
            If True, preload selected loadable artifacts into memory.
        mmap_mode:
            Passed to np.load(..., mmap_mode=...) when load_to_ram=False.
            NOTE: If load_to_ram=True, mmap_mode is ignored (set to None).
        numpy_mode:
            If True, __getitem__ returns numpy arrays instead of torch tensors.

        Outputs
        -------
        None
            Side effects:
              - scans selected artifact folders and stores file paths
              - builds fast lookup dicts for basename and stem
              - optionally preloads selected arrays into RAM
        """
        # Store configuration flags
        self.numpy_mode = numpy_mode
        self.return_artifacts = self._normalize_return_artifacts(return_artifacts)
        self.dtype = dtype
        self.transform = transform
        self.recursive = recursive
        self.extensions = tuple(e.lower() for e in extensions)
        self.sort = sort
        self.load_to_ram = load_to_ram

        # mmap_mode is only relevant when we DO NOT preload into RAM
        self.mmap_mode = None if load_to_ram else mmap_mode

        # Resolve configured folders for known artifacts.
        self.study_paths = study_paths
        self._artifact_paths = self._resolve_artifact_paths(study_paths)

        self.index_artifact = self._resolve_index_artifact(index_artifact)
        self._index_folder = self._artifact_paths[self.index_artifact]

        # Collect all index files into a list. This defines dataset length/order.
        self._paths = self._collect_paths(
            self._index_folder,
            artifact=self.index_artifact,
        )
        self._fnames = [os.path.basename(p) for p in self._paths]

        self._artifact_lookup_cache: dict[str, dict[str, dict[str, list[str]]]] = {}

        self.anomaly_metadata: dict[str, dict] = {}
        self._anomaly_meta_by_index: list[dict] = []
        if "anomaly_meta" in self.return_artifacts:
            self.anomaly_metadata = self._load_anomaly_metadata(
                study_paths,
                anomaly_meta_file,
            )
            self._anomaly_meta_by_index = self._align_anomaly_metadata(
                self.anomaly_metadata
            )

        # Paths aligned by dataset index for selected loadable artifacts only.
        self._sample_paths_by_artifact = self._build_sample_paths_by_artifact()

        # Fast lookup tables:
        # - basename -> [indices]
        # - stem (filename without extension) -> [indices]
        # These support load_numpy_by_basename() lookups for the index artifact.
        self._basename_to_indices: dict[str, list[int]] = {}
        self._stem_to_indices: dict[str, list[int]] = {}

        for idx, p in enumerate(self._paths):
            base = os.path.basename(p)      # e.g. "foo.npy"
            stem = Path(base).stem          # e.g. "foo"
            self._basename_to_indices.setdefault(base, []).append(idx)
            self._stem_to_indices.setdefault(stem, []).append(idx)

        # Optional preload into RAM for faster access during training/inference.
        # Only selected loadable artifacts are preloaded.
        self._ram_arrays: dict[str, list[np.ndarray]] = {}
        if self.load_to_ram:
            for artifact, paths in self._sample_paths_by_artifact.items():
                self._ram_arrays[artifact] = [
                    np.load(p, allow_pickle=False) for p in paths
                ]

    def _normalize_artifact_name(self, artifact: str) -> str:
        key = str(artifact).strip().lower()
        normalized = self.ARTIFACT_ALIASES.get(key, key)
        if normalized not in self.ALLOWED_ARTIFACTS:
            raise ValueError(
                f"Unknown artifact {artifact!r}. Supported: {self.ALLOWED_ARTIFACTS}"
            )
        return normalized

    def _normalize_return_artifacts(
        self,
        return_artifacts: Optional[Union[str, Sequence[str]]],
    ) -> tuple[str, ...]:
        if return_artifacts is None:
            return ("img", "fname")

        if isinstance(return_artifacts, str):
            raw_artifacts = (return_artifacts,)
        else:
            raw_artifacts = tuple(return_artifacts)

        artifacts = []
        for artifact in raw_artifacts:
            normalized = self._normalize_artifact_name(artifact)
            if normalized not in artifacts:
                artifacts.append(normalized)

        if not artifacts:
            raise ValueError("return_artifacts must contain at least one artifact.")

        return tuple(artifacts)

    def _resolve_artifact_paths(self, study_paths: StudyPaths) -> dict[str, Path]:
        resolved: dict[str, Path] = {}

        if study_paths is None:
            raise ValueError("study_paths must be provided.")

        for artifact, attr_name in self.STUDY_PATH_ATTRIBUTES.items():
            if not hasattr(study_paths, attr_name):
                continue
            value = getattr(study_paths, attr_name)
            if value is not None:
                resolved[artifact] = Path(value).expanduser().resolve()

        return resolved

    def _resolve_index_artifact(self, index_artifact: Optional[str]) -> str:
        candidates: list[str] = []
        selected_loadable = [
            artifact
            for artifact in self.return_artifacts
            if artifact in self.LOADABLE_ARTIFACTS
        ]

        if index_artifact is not None:
            normalized_index = self._normalize_artifact_name(index_artifact)
            if normalized_index not in self.LOADABLE_ARTIFACTS:
                raise ValueError(
                    f"index_artifact must be a NumPy artifact, got {normalized_index!r}."
                )
            candidates.append(normalized_index)

        candidates.extend(
            artifact
            for artifact in selected_loadable
            if artifact not in candidates
        )
        candidates.extend(
            artifact
            for artifact in self.LOADABLE_ARTIFACTS
            if artifact in self._artifact_paths and artifact not in candidates
        )

        for artifact in candidates:
            folder = self._artifact_paths.get(artifact)
            if folder is not None:
                if folder.is_dir():
                    return artifact
                if index_artifact is not None or artifact in selected_loadable:
                    raise FileNotFoundError(
                        f"Configured folder for artifact {artifact!r} does not exist: {folder}"
                    )

        raise ValueError(
            "No loadable artifact folder configured. Pass a StudyPaths instance "
            "with the requested artifact folder."
        )

    def _load_anomaly_metadata(
        self,
        study_paths: StudyPaths,
        anomaly_meta_file: Optional[Union[str, os.PathLike]],
    ) -> dict[str, dict]:
        if anomaly_meta_file is None:
            anomaly_meta_file = getattr(
                study_paths,
                "anomaly_transformations_file",
                None,
            )
        if anomaly_meta_file is None:
            raise ValueError(
                "anomaly_meta was requested, but no anomaly metadata file was provided."
            )

        path = Path(anomaly_meta_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Anomaly metadata file does not exist: {path}")

        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Anomaly metadata must be a JSON object keyed by basename: {path}"
            )

        invalid_keys = [
            key for key, value in metadata.items()
            if not isinstance(key, str) or not isinstance(value, dict)
        ]
        if invalid_keys:
            raise ValueError(
                "Every anomaly metadata entry must map a string basename to a dict. "
                f"Invalid entries: {invalid_keys}"
            )

        return metadata

    def _align_anomaly_metadata(self, metadata: dict[str, dict]) -> list[dict]:
        by_basename: dict[str, list[dict]] = {}
        by_stem: dict[str, list[dict]] = {}

        for name, value in metadata.items():
            basename = os.path.basename(name)
            by_basename.setdefault(basename, []).append(value)
            by_stem.setdefault(Path(basename).stem, []).append(value)

        aligned = []
        for fname in self._fnames:
            candidates = by_basename.get(fname, [])
            if not candidates:
                candidates = by_stem.get(Path(fname).stem, [])
            if not candidates:
                raise KeyError(f"anomaly_meta not found for dataset sample: {fname}")
            if len(candidates) > 1:
                raise ValueError(f"anomaly_meta is ambiguous for dataset sample: {fname}")
            aligned.append(candidates[0])

        return aligned

    def _collect_paths(self, folder: Path, *, artifact: str) -> list[str]:
        """
        Collect eligible file paths from the dataset folder.

        Inputs
        ------
        folder:
            Folder to scan.
        artifact:
            Artifact name, used for error messages and synth_roi recursion.

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
        recursive = self.recursive or artifact == "synth_roi"
        iterator = folder.rglob("*") if recursive else folder.glob("*")

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
                f"No files with extensions {self.extensions} found for artifact "
                f"{artifact!r} in: {folder}"
            )

        return paths

    def _build_sample_paths_by_artifact(self) -> dict[str, list[str]]:
        sample_paths: dict[str, list[str]] = {}
        selected_loadable = [
            artifact
            for artifact in self.return_artifacts
            if artifact in self.LOADABLE_ARTIFACTS
        ]

        for artifact in selected_loadable:
            folder = self._artifact_paths.get(artifact)
            if folder is None:
                raise ValueError(
                    f"No folder configured for requested artifact {artifact!r}."
                )
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Configured folder for artifact {artifact!r} does not exist: {folder}"
                )

            if artifact == self.index_artifact:
                sample_paths[artifact] = list(self._paths)
            else:
                sample_paths[artifact] = [
                    self._matching_artifact_path(artifact, index_path)
                    for index_path in self._paths
                ]

        return sample_paths

    def _matching_artifact_path(self, artifact: str, index_path: str) -> str:
        artifact_folder = self._artifact_paths[artifact]
        index_folder = self._artifact_paths[self.index_artifact]
        index_path_obj = Path(index_path)

        try:
            rel_path = index_path_obj.relative_to(index_folder)
        except ValueError:
            rel_path = Path(index_path_obj.name)

        exact_path = artifact_folder / rel_path
        if exact_path.is_file() and exact_path.suffix.lower() in self.extensions:
            return str(exact_path)

        lookup = self._artifact_lookup(artifact)
        return self._single_lookup_match(
            lookup,
            key=os.path.basename(index_path),
            artifact=artifact,
        )

    def _artifact_lookup(self, artifact: str) -> dict[str, dict[str, list[str]]]:
        if artifact in self._artifact_lookup_cache:
            return self._artifact_lookup_cache[artifact]

        folder = self._artifact_paths.get(artifact)
        if folder is None:
            raise ValueError(f"No folder configured for artifact {artifact!r}.")
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Configured folder for artifact {artifact!r} does not exist: {folder}"
            )

        paths = self._collect_paths(folder, artifact=artifact)
        by_basename: dict[str, list[str]] = {}
        by_stem: dict[str, list[str]] = {}

        for path in paths:
            base = os.path.basename(path)
            stem = Path(base).stem
            by_basename.setdefault(base, []).append(path)
            by_stem.setdefault(stem, []).append(path)

        lookup = {"basename": by_basename, "stem": by_stem}
        self._artifact_lookup_cache[artifact] = lookup
        return lookup

    def _single_lookup_match(
        self,
        lookup: dict[str, dict[str, list[str]]],
        *,
        key: str,
        artifact: str,
    ) -> str:
        candidates = lookup["basename"].get(key, [])

        if not candidates:
            stem = Path(key).stem
            candidates = lookup["stem"].get(stem, [])

        if not candidates:
            raise KeyError(f"basename not found for artifact {artifact!r}: {key}")

        if len(candidates) > 1:
            names = [os.path.basename(p) for p in candidates]
            raise ValueError(
                f"basename is ambiguous for artifact {artifact!r} ({key}); matches: {names}"
            )

        return candidates[0]

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
        dict
            Sample dictionary with the selected artifact keys.
            Loadable artifacts are torch.Tensor by default or np.ndarray if
            numpy_mode=True. fname is a basename string and anomaly_meta is a
            dictionary.

        Notes
        -----
        - If load_to_ram=True, loads selected arrays from self._ram_arrays (fast).
        - Otherwise loads from disk via np.load(..., mmap_mode=self.mmap_mode).
        - If transform is provided for img:
            - numpy_mode=False: transform receives torch.Tensor
            - numpy_mode=True : transform receives np.ndarray
        """
        fname = self._fnames[idx]

        sample = {}
        for artifact in self.return_artifacts:
            if artifact == "fname":
                sample["fname"] = fname
                continue
            if artifact == "anomaly_meta":
                sample["anomaly_meta"] = dict(self._anomaly_meta_by_index[idx])
                continue

            value = self._load_artifact_value(artifact, idx)
            if artifact == "img" and self.transform is not None:
                value = self.transform(value)
            sample[artifact] = value

        return sample

    def _load_artifact_value(self, artifact: str, idx: int):
        if artifact in self._ram_arrays:
            array = self._ram_arrays[artifact][idx]
        else:
            path = self._sample_paths_by_artifact[artifact][idx]
            array = np.load(path, allow_pickle=False, mmap_mode=self.mmap_mode)

        if self.numpy_mode:
            return array
        return self._to_tensor(array)

    def load_numpy_by_basename(self, basename: str, artifact: Optional[str] = None) -> np.ndarray:
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
        artifact:
            Which loadable artifact to read. If omitted, the dataset's
            index_artifact is used.

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
        if artifact is None:
            artifact = self.index_artifact
        artifact = self._normalize_artifact_name(artifact)
        if artifact not in self.LOADABLE_ARTIFACTS:
            raise ValueError(
                f"{artifact} is metadata, not a loadable NumPy artifact."
            )

        # Normalize input to just the basename portion
        key = os.path.basename(str(basename))

        # Fast path for the index artifact keeps the old behavior.
        if artifact == self.index_artifact:
            candidates = self._basename_to_indices.get(key, [])

            if not candidates:
                stem = Path(key).stem
                candidates = self._stem_to_indices.get(stem, [])

            if not candidates:
                raise KeyError(f"basename not found: {basename}")

            if len(candidates) > 1:
                names = [os.path.basename(self._paths[i]) for i in candidates]
                raise ValueError(f"basename is ambiguous ({basename}); matches: {names}")

            idx = candidates[0]
            if artifact in self._ram_arrays:
                return self._ram_arrays[artifact][idx]

            return np.load(self._paths[idx], allow_pickle=False, mmap_mode=self.mmap_mode)

        selected_indices = self._selected_artifact_indices_by_basename(artifact, key)
        if selected_indices:
            if len(selected_indices) > 1:
                names = [
                    os.path.basename(self._sample_paths_by_artifact[artifact][i])
                    for i in selected_indices
                ]
                raise ValueError(f"basename is ambiguous ({basename}); matches: {names}")

            idx = selected_indices[0]
            if artifact in self._ram_arrays:
                return self._ram_arrays[artifact][idx]

            path = self._sample_paths_by_artifact[artifact][idx]
            return np.load(path, allow_pickle=False, mmap_mode=self.mmap_mode)

        lookup = self._artifact_lookup(artifact)
        path = self._single_lookup_match(lookup, key=key, artifact=artifact)
        return np.load(path, allow_pickle=False, mmap_mode=self.mmap_mode)

    def load_sample_by_basename(self, basename: str) -> dict:
        """
        Load all selected artifacts for one index sample by basename.

        The returned dictionary has the same structure and value types as
        ``__getitem__``.
        """
        key = os.path.basename(str(basename))
        candidates = self._basename_to_indices.get(key, [])

        if not candidates:
            candidates = self._stem_to_indices.get(Path(key).stem, [])

        if not candidates:
            raise KeyError(f"basename not found: {basename}")

        if len(candidates) > 1:
            names = [os.path.basename(self._paths[i]) for i in candidates]
            raise ValueError(f"basename is ambiguous ({basename}); matches: {names}")

        return self[candidates[0]]

    def _selected_artifact_indices_by_basename(
        self,
        artifact: str,
        key: str,
    ) -> list[int]:
        paths = self._sample_paths_by_artifact.get(artifact)
        if paths is None:
            return []

        candidates = [
            idx
            for idx, path in enumerate(paths)
            if os.path.basename(path) == key
        ]
        if candidates:
            return candidates

        stem = Path(key).stem
        return [
            idx
            for idx, path in enumerate(paths)
            if Path(os.path.basename(path)).stem == stem
        ]
