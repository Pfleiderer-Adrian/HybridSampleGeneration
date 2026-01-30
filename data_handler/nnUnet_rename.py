import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".nii", ".gz")

def _is_image_file(p: Path) -> bool:
    if p.name.lower().endswith(".nii.gz"):
        return True
    return p.suffix.lower() in IMG_EXTS

def _stem_no_ext(p: Path) -> str:
    name = p.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return p.stem

def _full_ext(p: Path) -> str:
    name = p.name
    if name.lower().endswith(".nii.gz"):
        return ".nii.gz"
    return p.suffix

def _strip_suffix_token(stem: str, tokens: Tuple[str, ...]) -> str:
    s = stem
    for t in tokens:
        if s.endswith("_" + t):
            s = s[: -(len(t) + 1)]
        elif s.endswith(t):
            s = s[: -len(t)]
    return s

def _key_for_image(stem: str, modality: str) -> str:
    s = stem
    if modality:
        s = _strip_suffix_token(s, (modality,))
    return s

def _key_for_mask(stem: str) -> str:
    return _strip_suffix_token(stem, ("segmentation", "seg", "mask", "label", "labels"))

def _safe_plan_rename(src: Path, dst: Path, planned_dsts: set) -> None:
    if dst in planned_dsts:
        raise FileExistsError(f"Planned destination collision: {dst}")
    planned_dsts.add(dst)

def rename_to_nnunet_compatible(
    dataset_root: str,
    modality: str = "",
    modality_index: int = 0,
    *,
    imagesTr: str = "imagesTr",
    imagesTs: str = "imagesTs",
    labelsTr: str = "labelsTr",
    labelsTs: str = "labelsTs",          # <-- NEU
    dry_run: bool = True,
    allow_unpaired_train_images: bool = False,
    allow_unpaired_train_labels: bool = False,
    allow_unpaired_test_images: bool = True,   # <-- NEU (default: test labels oft nicht komplett)
    allow_unpaired_test_labels: bool = False,  # <-- NEU
    allow_test_train_id_overlap: bool = True,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Rename dataset to nnU-Net-style naming:
      imagesTr: CASEID_0000.ext
      labelsTr: CASEID.ext
      imagesTs: CASEID_0000.ext
      labelsTs: CASEID.ext   (optional)

    Matching keys:
      - image key: strip optional _{modality}
      - label key: strip suffix tokens segmentation/seg/mask/label/labels

    Returns a dict with lists of (src, dst).
    """
    root = Path(dataset_root)
    imgtr_dir = root / imagesTr
    imgts_dir = root / imagesTs
    labtr_dir = root / labelsTr
    labts_dir = root / labelsTs

    # labelsTs ist optional: nur prüfen, wenn vorhanden
    required_dirs = [imgtr_dir, imgts_dir, labtr_dir]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Folder not found: {d}")

    labelsTs_exists = labts_dir.exists()

    def list_imgs(d: Path) -> List[Path]:
        return sorted([p for p in d.iterdir() if p.is_file() and _is_image_file(p)])

    imgtr_paths = list_imgs(imgtr_dir)
    imgts_paths = list_imgs(imgts_dir)
    labtr_paths = list_imgs(labtr_dir)
    labts_paths: List[Path] = list_imgs(labts_dir) if labelsTs_exists else []

    # Build lookups by key
    imgtr_by_key: Dict[str, Path] = {}
    for p in imgtr_paths:
        key = _key_for_image(_stem_no_ext(p), modality)
        imgtr_by_key[key] = p

    imgts_by_key: Dict[str, Path] = {}
    for p in imgts_paths:
        key = _key_for_image(_stem_no_ext(p), modality)
        imgts_by_key[key] = p

    labtr_by_key: Dict[str, Path] = {}
    for p in labtr_paths:
        key = _key_for_mask(_stem_no_ext(p))
        labtr_by_key[key] = p

    labts_by_key: Dict[str, Path] = {}
    for p in labts_paths:
        key = _key_for_mask(_stem_no_ext(p))
        labts_by_key[key] = p

    planned: Dict[str, List[Tuple[str, str]]] = {
        "imagesTr": [], "labelsTr": [],
        "imagesTs": [], "labelsTs": []  # <-- labelsTs neu
    }
    planned_dsts = set()

    def nnunet_img_name(case_id: str, ext: str) -> str:
        return f"{case_id}_{modality_index:04d}{ext}"

    def nnunet_lbl_name(case_id: str, ext: str) -> str:
        return f"{case_id}{ext}"

    # -----------------------
    # Train: label -> image
    # -----------------------
    matched_train_imgs = set()
    for case_id, lab_path in labtr_by_key.items():
        img_path = imgtr_by_key.get(case_id)

        if img_path is None:
            if allow_unpaired_train_labels:
                continue
            raise FileNotFoundError(f"No matching training image for label key='{case_id}': {lab_path.name}")

        matched_train_imgs.add(img_path)

        img_ext = _full_ext(img_path)
        lab_ext = _full_ext(lab_path)

        new_img = imgtr_dir / nnunet_img_name(case_id, img_ext)
        new_lab = labtr_dir / nnunet_lbl_name(case_id, lab_ext)

        _safe_plan_rename(img_path, new_img, planned_dsts)
        _safe_plan_rename(lab_path, new_lab, planned_dsts)

        planned["imagesTr"].append((str(img_path), str(new_img)))
        planned["labelsTr"].append((str(lab_path), str(new_lab)))

    if not allow_unpaired_train_images:
        leftovers = [p for p in imgtr_paths if p not in matched_train_imgs]
        if leftovers:
            names = ", ".join([p.name for p in leftovers[:10]])
            raise FileNotFoundError(
                f"Found {len(leftovers)} training images without matching label (examples: {names}). "
                f"Set allow_unpaired_train_images=True to ignore."
            )
    """
    # -----------------------
    # Test images: immer umbenennen
    # -----------------------
    train_case_ids = set(labtr_by_key.keys())

    matched_test_imgs = set()
    test_case_ids = set()
    for p in imgts_paths:
        case_id = _key_for_image(_stem_no_ext(p), modality)
        test_case_ids.add(case_id)
        matched_test_imgs.add(p)

        if (not allow_test_train_id_overlap) and (case_id in train_case_ids):
            raise ValueError(
                f"Test case_id '{case_id}' overlaps with training IDs. "
                f"Set allow_test_train_id_overlap=True or rename upstream."
            )

        ext = _full_ext(p)
        new_p = imgts_dir / nnunet_img_name(case_id, ext)
        _safe_plan_rename(p, new_p, planned_dsts)
        planned["imagesTs"].append((str(p), str(new_p)))

    # -----------------------
    # labelsTs: label <-> imagesTs matchen und umbenennen
    # -----------------------
    if labelsTs_exists and labts_by_key:
        matched_test_imgs_for_labels = set()

        for case_id, lab_path in labts_by_key.items():
            img_path = imgts_by_key.get(case_id)

            if img_path is None:
                if allow_unpaired_test_labels:
                    continue
                raise FileNotFoundError(
                    f"No matching test image in imagesTs for labelsTs key='{case_id}': {lab_path.name}"
                )

            matched_test_imgs_for_labels.add(img_path)

            # imageTs rename ist bereits geplant (oben) — kann aber identisch sein.
            # labelTs planen:
            lab_ext = _full_ext(lab_path)
            new_lab = labts_dir / nnunet_lbl_name(case_id, lab_ext)
            _safe_plan_rename(lab_path, new_lab, planned_dsts)
            planned["labelsTs"].append((str(lab_path), str(new_lab)))

        if not allow_unpaired_test_images:
            # welche imagesTs haben kein labelTs?
            labeled_img_paths = matched_test_imgs_for_labels
            leftovers = [p for p in imgts_paths if p not in labeled_img_paths]
            if leftovers:
                names = ", ".join([p.name for p in leftovers[:10]])
                raise FileNotFoundError(
                    f"Found {len(leftovers)} test images without matching labelsTs (examples: {names}). "
                    f"Set allow_unpaired_test_images=True to ignore."
                )
                """
    # -----------------------
    # Execute (two-phase)
    # -----------------------
    if not dry_run:
        temp_map: List[Tuple[Path, Path, Path]] = []  # (src, tmp, dst)

        for group in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            for src_s, dst_s in planned[group]:
                src = Path(src_s)
                dst = Path(dst_s)
                if src == dst:
                    continue
                tmp = src.with_name(src.name + ".tmp_renaming")
                if tmp.exists():
                    tmp.unlink()
                os.replace(src, tmp)
                temp_map.append((src, tmp, dst))

        for _, tmp, dst in temp_map:
            if dst.exists():
                raise FileExistsError(f"Destination already exists: {dst}")
            os.replace(tmp, dst)

    return planned


if __name__ == "__main__":
    plan = rename_to_nnunet_compatible(
        dataset_root="/mnt/results/results/nnUNet/nnUNet_raw/Dataset002_BerlinFusionAnomalies5050",
        modality="",
        modality_index=0,
        dry_run=False,
        labelsTs="labelsTs",
        allow_unpaired_train_labels=True,
          allow_unpaired_train_images=True,
            allow_unpaired_test_images=True,
              allow_unpaired_test_labels=True  # falls der Ordner so heißt
    )

    for k, ops in plan.items():
        if not ops:
            continue
        print(f"\n[{k}] {len(ops)} renames")
        for src, dst in ops[:20]:
            print(f"  {Path(src).name}  ->  {Path(dst).name}")
        if len(ops) > 20:
            print("  ...")



