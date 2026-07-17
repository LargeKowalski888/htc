#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")


def main() -> None:
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)

    text = LAUNCHER.read_text(encoding="utf-8")
    backup = LAUNCHER.with_suffix(LAUNCHER.suffix + ".bak_create_adapter_cache")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    if "@meta_cache.csv" in text and "to_pickle" in text and "median_spectra@" in text:
        print("Launcher already appears to create adapter cache files.")
        return

    marker = "    return adapter_root, manifest\n"
    if marker not in text:
        raise RuntimeError("Could not find 'return adapter_root, manifest' in launcher")

    cache_code = '''    # Create Windows-stable cache files for the adapter readers.\n    # These avoid pandas/pyarrow feather crashes seen in HTC DataPath/Dataset startup.\n    meta_cache_path = tables_dir / f"{dataset_name}@meta_cache.csv"\n    if meta_path.exists():\n        pd.read_feather(meta_path).reset_index(drop=True)[["image_name", "path", "data_dir"]].to_csv(meta_cache_path, index=False)\n\n    median_feather_path = tables_dir / f"{dataset_name}@median_spectra@{annotation_name}.feather"\n    median_pickle_path = tables_dir / f"{dataset_name}@median_spectra@{annotation_name}.pkl"\n    if median_feather_path.exists():\n        pd.read_feather(median_feather_path).reset_index(drop=True).to_pickle(median_pickle_path)\n\n'''

    text = text.replace(marker, cache_code + marker, 1)
    LAUNCHER.write_text(text, encoding="utf-8")
    print(f"Patched launcher to create adapter cache files: {LAUNCHER}")


if __name__ == "__main__":
    main()
