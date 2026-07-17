#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")

CACHE_MARKER = "# --- Scenario adapter cache creation patch ---"
CACHE_BLOCK = r'''    # --- Scenario adapter cache creation patch ---
    # Create Windows-stable cache files for adapter readers before HTC training.
    # This avoids pandas/pyarrow feather crashes in DataPathAdapter and DatasetMedianPixelAdapter.
    meta_cache_path = tables_dir / f"{dataset_name}@meta_cache.csv"
    if meta_path.exists():
        pd.read_feather(meta_path).reset_index(drop=True)[["image_name", "path", "data_dir"]].to_csv(meta_cache_path, index=False)
        print(f"[INFO] Wrote adapter meta cache: {meta_cache_path}")

    median_feather_path = tables_dir / f"{dataset_name}@median_spectra@{annotation_name}.feather"
    median_pickle_path = tables_dir / f"{dataset_name}@median_spectra@{annotation_name}.pkl"
    if median_feather_path.exists():
        pd.read_feather(median_feather_path).reset_index(drop=True).to_pickle(median_pickle_path)
        print(f"[INFO] Wrote adapter median spectra pickle cache: {median_pickle_path}")
    # --- End scenario adapter cache creation patch ---

'''


def patch_launcher() -> None:
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)
    text = LAUNCHER.read_text(encoding="utf-8")
    backup = LAUNCHER.with_suffix(LAUNCHER.suffix + ".bak_force_cache_creation")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    if CACHE_MARKER in text:
        print("Launcher already has forced scenario cache creation patch.")
    else:
        marker = "    return adapter_root, manifest\n"
        if marker not in text:
            raise RuntimeError("Could not find 'return adapter_root, manifest' in launcher")
        text = text.replace(marker, CACHE_BLOCK + marker, 1)
        LAUNCHER.write_text(text, encoding="utf-8")
        print(f"Patched launcher cache creation: {LAUNCHER}")


def main() -> None:
    patch_launcher()
    print("Done. Re-run the scenario training batch. You should see two new [INFO] Wrote adapter cache lines before HTC training starts.")


if __name__ == "__main__":
    main()
