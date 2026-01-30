from pathlib import Path

def rename_mask_files(folder: str, suffix: str = "_mask", dry_run: bool = True) -> None:
    p = Path(folder)
    if not p.is_dir():
        raise ValueError(f"Ordner existiert nicht: {folder}")

    for f in p.iterdir():
        if not f.is_file():
            continue

        # f.stem = Dateiname ohne Endung, z.B. "dateinamexy_mask"
        # f.suffix = ".png"
        if f.stem.endswith(suffix):
            new_name = f.stem[: -len(suffix)] + f.suffix  # remove "_mask"
            target = f.with_name(new_name)

            if target.exists():
                print(f"SKIP (Ziel existiert): {f.name} -> {target.name}")
                continue

            if dry_run:
                print(f"DRY-RUN: {f.name} -> {target.name}")
            else:
                f.rename(target)
                print(f"RENAMED: {f.name} -> {target.name}")

if __name__ == "__main__":
    # Beispiel:
    rename_mask_files("/mnt/results/results/nnUNet/nnUNet_raw/Dataset002_BerlinFusionAnomalies9010/labelsTr", dry_run=True)  # nur anzeigen
    #rename_mask_files("/pfad/zu/deinem/ordner", dry_run=False) # wirklich umbenennen

    #rename_mask_files(".", dry_run=True)