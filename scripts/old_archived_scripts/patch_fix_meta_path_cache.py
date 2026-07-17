#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")


def main() -> None:
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)
    text = LAUNCHER.read_text(encoding="utf-8")
    backup = LAUNCHER.with_suffix(LAUNCHER.suffix + ".bak_fix_meta_path_cache")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    target = '    meta_cache_path = tables_dir / f"{dataset_name}@meta_cache.csv"\n    if meta_path.exists():\n'
    replacement = '    meta_cache_path = tables_dir / f"{dataset_name}@meta_cache.csv"\n    meta_path = tables_dir / f"{dataset_name}@meta.feather"\n    if meta_path.exists():\n'

    if replacement in text:
        print("meta_path cache block already fixed.")
    elif target in text:
        text = text.replace(target, replacement, 1)
        LAUNCHER.write_text(text, encoding="utf-8")
        print(f"Fixed missing meta_path definition in: {LAUNCHER}")
    else:
        raise RuntimeError("Could not find cache block. Please inspect htc_median_pixel_from_yaml_official.py near meta_cache_path.")


if __name__ == "__main__":
    main()
