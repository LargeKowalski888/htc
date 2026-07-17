#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")

HELPER_MARKER = "# --- YAML run settings patch ---"
CALL_MARKER = "    args = apply_yaml_run_settings(args)\n"

HELPER_CODE = r'''
# --- YAML run settings patch ---
def _yaml_get_nested(cfg: dict, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _yaml_set_arg(args, attr: str, value):
    if value is not None and hasattr(args, attr):
        setattr(args, attr, value)


def apply_yaml_run_settings(args):
    """Apply optional scenario run settings from the YAML file.

    Supported YAML sections:
      run_settings: training_profile, accelerator, devices, precision,
                    batch_size, num_workers, epoch_size, max_epochs
      wavelength:   min, max
      paths:        path_prefix_from, path_prefix_to
      htc:          adapter_dataset_name, annotation_name, label_mode,
                    label_mapping, require_standardized_or_001
    """
    try:
        with open(args.settings_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARNING] Could not read YAML run settings from {args.settings_yaml}: {e}")
        return args

    run = cfg.get("run_settings", {}) or {}
    wave = cfg.get("wavelength", {}) or {}
    paths = cfg.get("paths", {}) or {}
    htc_cfg = cfg.get("htc", {}) or {}

    mapping = {
        "training_profile": run.get("training_profile"),
        "accelerator": run.get("accelerator"),
        "devices": run.get("devices"),
        "precision": run.get("precision"),
        "batch_size": run.get("batch_size"),
        "num_workers": run.get("num_workers"),
        "epoch_size": run.get("epoch_size"),
        "max_epochs": run.get("max_epochs"),
        "wavelength_min": wave.get("min", wave.get("wavelength_min")),
        "wavelength_max": wave.get("max", wave.get("wavelength_max")),
        "path_prefix_from": paths.get("path_prefix_from"),
        "path_prefix_to": paths.get("path_prefix_to"),
        "adapter_dataset_name": htc_cfg.get("adapter_dataset_name"),
        "annotation_name": htc_cfg.get("annotation_name"),
        "label_mode": htc_cfg.get("label_mode"),
        "label_mapping": htc_cfg.get("label_mapping"),
    }

    for attr, value in mapping.items():
        _yaml_set_arg(args, attr, value)

    # Optional YAML override for the old CLI flag.
    if htc_cfg.get("require_standardized_or_001") is not None and hasattr(args, "require_standardized_or_001"):
        args.require_standardized_or_001 = bool(htc_cfg.get("require_standardized_or_001"))

    # Keep n_channels synchronized with selected wavelength range.
    if hasattr(args, "wavelength_min") and hasattr(args, "wavelength_max"):
        if args.wavelength_min > args.wavelength_max:
            raise ValueError("YAML/CLI wavelength min must be <= wavelength max")
        args.n_channels = int(round((float(args.wavelength_max) - float(args.wavelength_min)) / 5.0)) + 1

    print("[INFO] Applied YAML run settings:")
    for attr in [
        "training_profile", "accelerator", "devices", "precision", "batch_size",
        "num_workers", "epoch_size", "max_epochs", "wavelength_min", "wavelength_max",
        "path_prefix_from", "path_prefix_to", "adapter_dataset_name", "annotation_name",
        "label_mode", "label_mapping",
    ]:
        if hasattr(args, attr):
            print(f"[INFO]   {attr}: {getattr(args, attr)}")

    return args
# --- End YAML run settings patch ---
'''


def main() -> None:
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)

    text = LAUNCHER.read_text(encoding="utf-8")
    backup = LAUNCHER.with_suffix(LAUNCHER.suffix + ".bak_yaml_run_settings")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    if HELPER_MARKER not in text:
        # Insert helper after imports. Prefer before first major function.
        insert_at = text.find("def ")
        if insert_at < 0:
            raise RuntimeError("Could not find first function definition in launcher")
        text = text[:insert_at] + HELPER_CODE + "\n" + text[insert_at:]
        print("Inserted YAML run settings helper")
    else:
        print("YAML run settings helper already present")

    if CALL_MARKER not in text:
        target = "    args = parser.parse_args()\n"
        if target not in text:
            raise RuntimeError("Could not find args = parser.parse_args() line")
        text = text.replace(target, target + CALL_MARKER, 1)
        print("Inserted apply_yaml_run_settings(args) call")
    else:
        print("apply_yaml_run_settings(args) call already present")

    LAUNCHER.write_text(text, encoding="utf-8")
    print(f"Patched launcher: {LAUNCHER}")
    print("Run: python -m py_compile D:\\HTC_github\\htc\\scripts\\newer_scripts\\htc_median_pixel_from_yaml_official.py")


if __name__ == "__main__":
    main()
