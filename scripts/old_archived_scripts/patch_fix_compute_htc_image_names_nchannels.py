#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")


def replace_function_body(text: str, func_name: str, replacements: list[tuple[str, str]]) -> str:
    start = text.find(f"def {func_name}(")
    if start < 0:
        raise RuntimeError(f"Could not find function {func_name}")
    next_def = text.find("\ndef ", start + 1)
    end = len(text) if next_def < 0 else next_def + 1
    body = text[start:end]
    new_body = body
    for old, new in replacements:
        if old in new_body:
            new_body = new_body.replace(old, new)
        else:
            print(f"[INFO] Text not present in {func_name}, maybe already fixed: {old}")
    return text[:start] + new_body + text[end:]


def main() -> None:
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)

    text = LAUNCHER.read_text(encoding="utf-8")
    backup = LAUNCHER.with_suffix(LAUNCHER.suffix + ".bak_fix_nchannels_scope")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    # compute_htc_image_names has n_channels in scope, not args.
    text = replace_function_body(text, "compute_htc_image_names", [
        ("int(args.n_channels)", "int(n_channels)"),
        ('"shape": [480, 640, int(args.n_channels)]', '"shape": [480, 640, int(n_channels)]'),
    ])

    # Ensure compute_htc_image_names accepts n_channels.
    old_sig = "def compute_htc_image_names(manifest: pd.DataFrame, adapter_root: Path, annotation_name: str, dataset_name: str, label_mapping: dict[str, int]) -> pd.DataFrame:"
    new_sig = "def compute_htc_image_names(manifest: pd.DataFrame, adapter_root: Path, annotation_name: str, dataset_name: str, label_mapping: dict[str, int], n_channels: int = 100) -> pd.DataFrame:"
    if old_sig in text:
        text = text.replace(old_sig, new_sig)
        print("Patched function signature")

    # Ensure build_htc_adapter passes args.n_channels into compute_htc_image_names.
    old_call = "manifest = compute_htc_image_names(manifest, adapter_root, annotation_name, dataset_name, label_mapping)"
    new_call = "manifest = compute_htc_image_names(manifest, adapter_root, annotation_name, dataset_name, label_mapping, int(args.n_channels))"
    if old_call in text:
        text = text.replace(old_call, new_call)
        print("Patched compute_htc_image_names call")
    elif new_call in text:
        print("compute_htc_image_names call already patched")
    else:
        print("[WARNING] Could not find compute_htc_image_names call. Please inspect manually.")

    LAUNCHER.write_text(text, encoding="utf-8")
    print(f"Fixed n_channels scope issue in: {LAUNCHER}")
    print("Now run: python -m py_compile D:\\HTC_github\\htc\\scripts\\newer_scripts\\htc_median_pixel_from_yaml_official.py")


if __name__ == "__main__":
    main()
