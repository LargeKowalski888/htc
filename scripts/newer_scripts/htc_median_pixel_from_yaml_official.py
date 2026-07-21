#!/usr/bin/env python
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

OFFICIAL_ATLAS_LABEL_ORDER = [
    "stomach", "small_bowel", "colon", "liver", "gallbladder", "pancreas", "kidney", "lung", "heart", "cartilage",
    "bile_fluid", "kidney_with_Gerotas_fascia", "major_vein", "peritoneum", "muscle", "skin", "bone", "omentum", "bladder", "spleen",
]
FOLDER_KEYWORD_TO_OFFICIAL_LABEL = [
    ("kidney_with_ger", "kidney_with_Gerotas_fascia"), ("ger_fascia", "kidney_with_Gerotas_fascia"), ("gerotas", "kidney_with_Gerotas_fascia"),
    ("bile_fluid", "bile_fluid"), ("bile", "bile_fluid"), ("major_vein", "major_vein"), ("vena_cava", "major_vein"), ("cava", "major_vein"),
    ("small_bowel", "small_bowel"), ("jejunum", "small_bowel"), ("stomach", "stomach"), ("colon", "colon"), ("liver", "liver"),
    ("gallbladder", "gallbladder"), ("gall_bladder", "gallbladder"), ("pancreas", "pancreas"), ("kidney", "kidney"), ("spleen", "spleen"),
    ("bladder", "bladder"), ("omentum", "omentum"), ("lung", "lung"), ("heart", "heart"), ("cartilage", "cartilage"), ("bone", "bone"),
    ("skin", "skin"), ("muscle", "muscle"), ("peritoneum", "peritoneum"),
]
TIMESTAMP_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")

@dataclass
class ScanRow:
    organ_key: str
    label_name: str
    class_folder_name: str
    class_folder_path: str
    subject_name: str
    timestamp: str
    image_name: str
    sample_dir: str
    hypergui_dir: str
    labelling_file_path: str
    spectrum_xlsx_path: str


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

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data

def norm_path_string(s: str) -> str:
    return str(s).replace("\\", "/")

def resolve_experiment_path(raw_path: str, parent_root: str | None, path_prefix_from: str | None, path_prefix_to: str | None) -> Path:
    value = str(raw_path)
    if path_prefix_from:
        a = norm_path_string(value)
        b = norm_path_string(path_prefix_from)
        if a.lower().startswith(b.lower()):
            value = str(path_prefix_to) + a[len(b):]
    p = Path(value)
    if not p.is_absolute() and parent_root:
        p = Path(parent_root) / p
    return p


def normalize_yaml_label(key: str) -> str:
    """
    Convert YAML class keys such as:

        01_kidney
        02_ureter
        10_duct_def

    into:

        kidney
        ureter
        duct_def
    """
    key = str(key).strip()
    key = re.sub(r"^\d+_", "", key)
    return key

def infer_label(organ_key: str, folder_path: str) -> str:
    text = f" {organ_key} {Path(folder_path).name}".lower()
    for token, label in FOLDER_KEYWORD_TO_OFFICIAL_LABEL:
        if token.lower() in text:
            return label
    return Path(folder_path).name


def get_label_for_key(
    key: str,
    cfg: dict[str, Any],
    folder_path: str,
    label_mode: str,
) -> str:
    """
    Always derive labels from YAML keys.

    Examples:

        01_kidney  -> kidney
        02_ureter  -> ureter
        04_prostate -> prostate
        17_epididymis -> epididymis
    """

    labels = cfg.get("labels") or cfg.get("label_names") or {}

    if isinstance(labels, dict) and key in labels:
        return str(labels[key])

    return normalize_yaml_label(key)


def value_for_key(value: Any, key: str, default: str) -> str:
    """Return a YAML value for one class key and clean path-fragment values.

    This is used for per-class labelling_file and hyperguis entries. We strip
    leading slashes/backslashes so YAML values like "\\_hypergui_6" behave like
    folder names instead of rooted paths on Windows.
    """
    if isinstance(value, dict):
        raw = value.get(key, default)
    else:
        raw = value if value is not None else default

    cleaned = str(raw).strip().strip('"').strip("'")
    cleaned = cleaned.replace("/", "\\")
    cleaned = cleaned.lstrip("\\/")
    return cleaned


def label_file_matches(
    sample_dir: Path,
    pattern: str,
    require_standardized_or_001: bool,
) -> Path | None:
    """Find the class-specific labelling file for one sample directory.

    If the YAML provides labelling_file for a class, use that exact file or glob
    pattern first. Only fall back to the old _labelling_001/standardized logic
    when no class-specific match is found.
    """
    pattern = str(pattern or "").strip().strip('"').strip("'").replace("/", "\\").lstrip("\\/")

    candidates: list[Path] = []

    if pattern:
        # Exact file name, e.g. _labelling_paper_tissue_atlas_uro_patient.txt
        exact = sample_dir / pattern
        if exact.exists():
            candidates.append(exact)

        # Glob pattern, e.g. _labelling_paper_*.txt
        candidates.extend(sorted(sample_dir.glob(pattern)))

    if candidates:
        return candidates[0]

    if require_standardized_or_001:
        strict_patterns = ["_labelling_001.txt", "*_labelling_standardized.txt", "_labelling_standardized.txt"]
        for p in strict_patterns:
            matches = sorted(sample_dir.glob(p))
            if matches:
                return matches[0]
        return None

    # Flexible fallback for legacy YAML files.
    matches = sorted(sample_dir.glob("_labelling*"))
    if matches:
        return matches[0]

    return None


def scan_yaml_dataset(cfg: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    experiment_folders = cfg.get("experiment_folders") or {}
    if isinstance(experiment_folders, list):
        experiment_folders = {str(i): p for i, p in enumerate(experiment_folders)}
    if not isinstance(experiment_folders, dict) or not experiment_folders:
        raise ValueError("YAML must contain non-empty experiment_folders mapping/list")
    labelling_cfg = cfg.get("labelling_file", args.default_labelling_file) or args.default_labelling_file
    hypergui_cfg = cfg.get("hyperguis", args.default_hypergui) or args.default_hypergui
    rows = []
    audit_rows = []
    for organ_key, raw_folder in experiment_folders.items():
        class_folder = resolve_experiment_path(str(raw_folder), args.parent_root or None, args.path_prefix_from or None, args.path_prefix_to or None)
        label_name = get_label_for_key(str(organ_key), cfg, str(class_folder), args.label_mode)
        data_dir = class_folder / args.data_subdir
        label_pattern = value_for_key(labelling_cfg, str(organ_key), args.default_labelling_file)
        hypergui_name = value_for_key(hypergui_cfg, str(organ_key), args.default_hypergui)
        count = 0
        status = "missing_data_dir"
        if data_dir.is_dir():
            status = "no_valid_samples"
            for subject_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
                for sample_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and TIMESTAMP_RE.match(p.name)):
                    labelling_path = label_file_matches(sample_dir, label_pattern, args.require_standardized_or_001)
                    if labelling_path is None:
                        continue
                    hypergui_dir = sample_dir / hypergui_name
                    spectra = sorted(hypergui_dir.glob(args.csv_pattern)) if hypergui_dir.is_dir() else []
                    if not spectra:
                        continue
                    timestamp = sample_dir.name
                    subject_name = subject_dir.name
                    rows.append(ScanRow(str(organ_key), label_name, class_folder.name, str(class_folder), subject_name, timestamp, f"{subject_name}#{timestamp}", str(sample_dir), str(hypergui_dir), str(labelling_path), str(spectra[0])))
                    count += 1
            if count:
                status = "ok"
        audit_rows.append({"organ_key": organ_key, "class_folder_path": str(class_folder), "label_name": label_name, "labelling_file": label_pattern, "hypergui": hypergui_name, "status": status, "count": count})
    df = pd.DataFrame([asdict(r) for r in rows])
    if df.empty:
        raise ValueError("No valid samples found")
    args._audit_df = pd.DataFrame(audit_rows)
    return df

def ordered_labels(df: pd.DataFrame, cfg: dict[str, Any], label_mode: str) -> list[str]:
    present = set(df["label_name"].astype(str))
    labels_cfg = cfg.get("label_order")
    if labels_cfg:
        return [str(x) for x in labels_cfg if str(x) in present]
    if label_mode == "official_atlas":
        return [x for x in OFFICIAL_ATLAS_LABEL_ORDER if x in present]
    seen = []
    for x in df["label_name"].astype(str).tolist():
        if x not in seen:
            seen.append(x)
    return seen

def make_label_mapping(labels: list[str]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}

def split_by_subject(df: pd.DataFrame, image_col: str) -> list[dict[str, Any]]:
    """Create train/val/test splits with YAML control.

    This version supports manual splits and a randomized balanced subject search.
    The automatic mode searches many subject-separated candidate splits and
    chooses the one with the best class support across train/val/test.

    YAML example:
      splits:
        mode: auto
        train_fraction: 0.60
        val_fraction: 0.20
        test_fraction: 0.20
        min_train_per_class: 25
        min_val_per_class: 5
        min_test_per_class: 10
        split_seed: 1337
        split_trials: 5000
    """
    split_cfg = globals().get("YAML_SPLITS_CONFIG", {}) or {}
    if not isinstance(split_cfg, dict):
        split_cfg = {}

    subjects_all = sorted(df["subject_name"].dropna().astype(str).unique().tolist())
    labels_all = sorted(df["label_name"].dropna().astype(str).unique().tolist())
    if not subjects_all:
        raise ValueError("No subjects available for train/val/test split")

    def as_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(x) for x in value]

    def names_for(subs: list[str]) -> list[str]:
        return sorted(df[df["subject_name"].astype(str).isin(subs)][image_col].drop_duplicates().tolist())

    def report(split_name: str, subs: list[str], min_required: int | None = None) -> None:
        d = df[df["subject_name"].astype(str).isin(subs)]
        counts = d["label_name"].astype(str).value_counts().sort_index()
        print(f"[INFO] {split_name}: n_subjects={len(subs)}, n_images={len(names_for(subs))}, n_labels={counts.shape[0]}")
        print(counts.to_string() if len(counts) else "<empty>")
        missing = sorted(set(labels_all) - set(counts.index.astype(str)))
        if missing:
            print(f"[WARNING] {split_name} split is missing labels: {missing}")
        if min_required is not None:
            low = {label: int(counts.get(label, 0)) for label in labels_all if int(counts.get(label, 0)) < min_required}
            if low:
                print(f"[WARNING] {split_name} split has labels below minimum {min_required}: {low}")

    manual_keys = {"train_subjects", "val_subjects", "test_subjects"}
    if str(split_cfg.get("mode", "")).lower() == "manual" or any(k in split_cfg for k in manual_keys):
        train_subjects = as_list(split_cfg.get("train_subjects", split_cfg.get("train", [])))
        val_subjects = as_list(split_cfg.get("val_subjects", split_cfg.get("val", [])))
        test_subjects = as_list(split_cfg.get("test_subjects", split_cfg.get("test", [])))

        if not train_subjects or not val_subjects or not test_subjects:
            raise ValueError("Manual YAML split requires train_subjects, val_subjects, and test_subjects")

        known = set(subjects_all)
        unknown = sorted(set(train_subjects + val_subjects + test_subjects) - known)
        if unknown:
            raise ValueError(f"Manual YAML split contains unknown subjects: {unknown}")

        overlap = (set(train_subjects) & set(val_subjects)) | (set(train_subjects) & set(test_subjects)) | (set(val_subjects) & set(test_subjects))
        if overlap:
            raise ValueError(f"Manual YAML split contains subjects in multiple splits: {sorted(overlap)}")

        print("[INFO] Using manual train/val/test subject split from YAML")
        report("train", train_subjects, int(split_cfg.get("min_train_per_class", 0) or 0) or None)
        report("val", val_subjects, int(split_cfg.get("min_val_per_class", 0) or 0) or None)
        report("test", test_subjects, int(split_cfg.get("min_test_per_class", 0) or 0) or None)
        return [{
            "fold_name": "fold_yaml",
            "train": {"image_names": names_for(train_subjects)},
            "val": {"image_names": names_for(val_subjects)},
            "test": {"image_names": names_for(test_subjects)},
        }]

    import random

    min_train = int(split_cfg.get("min_train_per_class", 25) or 25)
    min_val = int(split_cfg.get("min_val_per_class", 5) or 5)
    min_test = int(split_cfg.get("min_test_per_class", 10) or 10)
    train_fraction = float(split_cfg.get("train_fraction", 0.60))
    val_fraction = float(split_cfg.get("val_fraction", 0.20))
    test_fraction = float(split_cfg.get("test_fraction", 0.20))
    seed = int(split_cfg.get("split_seed", 1337) or 1337)
    trials = int(split_cfg.get("split_trials", 5000) or 5000)

    n_subjects = len(subjects_all)
    n_test = max(1, round(n_subjects * test_fraction))
    n_val = max(1, round(n_subjects * val_fraction))
    if n_test + n_val >= n_subjects:
        raise ValueError("val_fraction + test_fraction leaves no training subjects")

    subject_counts = {
        s: df[df["subject_name"].astype(str).eq(s)]["label_name"].astype(str).value_counts().to_dict()
        for s in subjects_all
    }
    total_counts = df["label_name"].astype(str).value_counts().to_dict()

    def counts_for(subs: list[str]) -> dict[str, int]:
        counts = {label: 0 for label in labels_all}
        for s in subs:
            for label, n in subject_counts.get(s, {}).items():
                counts[label] = counts.get(label, 0) + int(n)
        return counts

    def split_score(train_subs: list[str], val_subs: list[str], test_subs: list[str]) -> tuple:
        train_counts = counts_for(train_subs)
        val_counts = counts_for(val_subs)
        test_counts = counts_for(test_subs)

        # Heavy penalties for missing classes, then below-minimum support.
        missing_train = sum(1 for l in labels_all if train_counts.get(l, 0) == 0)
        missing_val = sum(1 for l in labels_all if val_counts.get(l, 0) == 0)
        missing_test = sum(1 for l in labels_all if test_counts.get(l, 0) == 0)

        train_def = sum(max(0, min(min_train, int(total_counts.get(l, 0))) - int(train_counts.get(l, 0))) for l in labels_all)
        val_def = sum(max(0, min(min_val, int(total_counts.get(l, 0))) - int(val_counts.get(l, 0))) for l in labels_all)
        test_def = sum(max(0, min(min_test, int(total_counts.get(l, 0))) - int(test_counts.get(l, 0))) for l in labels_all)

        # Prefer approximate subject fractions as secondary criterion.
        desired_train = n_subjects - n_val - n_test
        subject_imbalance = abs(len(train_subs) - desired_train) + abs(len(val_subs) - n_val) + abs(len(test_subs) - n_test)

        # Prefer more samples in training after hard constraints.
        train_rows = sum(sum(subject_counts.get(s, {}).values()) for s in train_subs)

        # Python tuple is minimized. Negative train_rows means more training rows is better.
        return (
            missing_test * 100000 + missing_val * 50000 + missing_train * 50000,
            test_def * 1000 + val_def * 500 + train_def * 500,
            subject_imbalance,
            -train_rows,
        )

    rng = random.Random(seed)
    best = None
    best_score = None

    # Deterministic-ish candidate: shuffled candidates plus a sorted baseline.
    candidates: list[list[str]] = [subjects_all[:]]
    for _ in range(trials):
        shuffled = subjects_all[:]
        rng.shuffle(shuffled)
        candidates.append(shuffled)

    for shuffled in candidates:
        test_subs = sorted(shuffled[:n_test])
        val_subs = sorted(shuffled[n_test:n_test + n_val])
        train_subs = sorted(shuffled[n_test + n_val:])
        score = split_score(train_subs, val_subs, test_subs)
        if best_score is None or score < best_score:
            best_score = score
            best = (train_subs, val_subs, test_subs)

    if best is None:
        raise ValueError("Could not find any candidate split")

    train_subjects, val_subjects, test_subjects = best

    print("[INFO] Using randomized balanced subject split search")
    print(f"[INFO] Fractions by subject: train~{train_fraction}, val={val_fraction}, test={test_fraction}")
    print(f"[INFO] Subject counts: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
    print(f"[INFO] Minimum support targets: train={min_train}, val={min_val}, test={min_test}")
    print(f"[INFO] Split search seed={seed}, trials={trials}, score={best_score}")
    report("train", train_subjects, min_train)
    report("val", val_subjects, min_val)
    report("test", test_subjects, min_test)

    return [{
        "fold_name": "fold_yaml",
        "train": {"image_names": names_for(train_subjects)},
        "val": {"image_names": names_for(val_subjects)},
        "test": {"image_names": names_for(test_subjects)},
    }]


def read_reflectance_spectrum(xlsx_path: str, wavelength_min: float = 500.0, wavelength_max: float = 995.0) -> np.ndarray:
    """Read CSV1 spectrum and keep only the requested inclusive wavelength range."""
    if wavelength_min > wavelength_max:
        raise ValueError(f"wavelength_min must be <= wavelength_max, got {wavelength_min} > {wavelength_max}")

    data = pd.read_excel(xlsx_path, sheet_name="0_derivative", header=None)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {xlsx_path}, got {data.shape}")

    wavelengths = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(data.iloc[:, 1], errors="coerce")
    mask = wavelengths.notna() & values.notna() & (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
    spectrum = values.loc[mask].astype("float32").to_numpy()

    if len(spectrum) == 0:
        raise ValueError(f"No spectral channels selected in {xlsx_path} for wavelength range {wavelength_min}-{wavelength_max} nm")

    return spectrum

def l1_normalize(x: np.ndarray) -> np.ndarray:
    denom = float(np.sum(np.abs(x)))
    return x.astype("float32") if denom == 0 else (x / denom).astype("float32")

def infer_htc_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def patch_persistent_workers(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".bak_persistent_workers")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    new = re.sub(r"persistent_workers\s*=\s*True", 'persistent_workers=self.config["dataloader_kwargs"].get("num_workers", 0) > 0', text)
    if new != text:
        path.write_text(new, encoding="utf-8")
        print(f"[INFO] Patched persistent_workers in {path}")

def patch_windows_rename(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".bak_windows_rename")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    helper = '''
def safe_rename_path(src, dst):
    try:
        src.rename(dst)
        return dst
    except PermissionError as e:
        print(f"[WARNING][htc] Could not rename {src} to {dst}: {e}")
        print(f"[WARNING][htc] Keeping existing folder: {src}")
        return src
'''
    if "def safe_rename_path(src, dst):" not in text:
        text = text.replace("class FoldTrainer:", helper + "\nclass FoldTrainer:")
    text = text.replace("model_dir_tmp.rename(run_path / fold_name)", "safe_rename_path(model_dir_tmp, run_path / fold_name)")
    text = text.replace("run_path_tmp.rename(run_path)", "safe_rename_path(run_path_tmp, run_path)")
    path.write_text(text, encoding="utf-8")
    print(f"[INFO] Patched Windows-safe rename in {path}")

def install_htc_adapter_code(htc_project_root: Path, patch_htc: bool) -> None:
    tivita_dir = htc_project_root / "htc" / "tivita"
    median_dir = htc_project_root / "htc" / "models" / "median_pixel"
    models_dir = htc_project_root / "htc" / "models"
    tivita_dir.mkdir(parents=True, exist_ok=True)
    median_dir.mkdir(parents=True, exist_ok=True)
    (tivita_dir / "DataPathAdapter.py").write_text(DATA_PATH_ADAPTER_CODE, encoding="utf-8")
    (median_dir / "DatasetMedianPixelAdapter.py").write_text(DATASET_MEDIAN_PIXEL_ADAPTER_CODE, encoding="utf-8")
    (median_dir / "LightningMedianPixelAdapter.py").write_text(LIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE, encoding="utf-8")
    print(f"[INFO] Installed adapter classes in {htc_project_root}")
    if patch_htc:
        patch_persistent_workers(median_dir / "LightningMedianPixel.py")
        patch_windows_rename(models_dir / "run_training.py")
        datapath_file = htc_project_root / "htc" / "tivita" / "DataPath.py"
        dp_text = datapath_file.read_text(encoding="utf-8")
        if "# --- Cat HTC Adapter from_image_name patch ---" not in dp_text:
            datapath_file.write_text(dp_text + DATAPATH_FORCE_PATCH, encoding="utf-8")
            print(f"[INFO] Patched adapter-aware DataPath.from_image_name in {datapath_file}")

DATA_PATH_ADAPTER_CODE = r'''from pathlib import Path
import json
import os

import pandas as pd

from htc.tivita.DataPathTivita import DataPathTivita
from htc.tivita.DatasetSettings import DatasetSettings


class DataPathAdapter(DataPathTivita):
    _image_name_to_row_cache = None
    _path_to_image_name_cache = None

    @staticmethod
    def _norm_path(path) -> str:
        return str(Path(path)).replace("/", "\\").rstrip("\\").lower()

    @classmethod
    def _adapter_root(cls) -> Path:
        value = os.environ.get("PATH_Tivita_Cat_HTC_Adapter")
        if value is None:
            raise RuntimeError("PATH_Tivita_Cat_HTC_Adapter is not set")
        return Path(value)

    @classmethod
    def _load_image_name_cache(cls):
        if cls._image_name_to_row_cache is not None:
            return cls._image_name_to_row_cache

        adapter_root = cls._adapter_root()
        tables_dir = adapter_root / "intermediates" / "tables"
        csv_path = tables_dir / "Cat_HTC_Adapter@meta_cache.csv"

        if not csv_path.exists():
            feather_files = sorted(tables_dir.glob("*@meta.feather"))
            if not feather_files:
                raise FileNotFoundError(f"No *@meta.feather found in {tables_dir}")
            df = pd.read_feather(feather_files[0]).reset_index(drop=True)
            df[["image_name", "path", "data_dir"]].to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)
        image_cache = {}
        path_cache = {}

        for _, row in df.iterrows():
            image_name = str(row["image_name"])
            row_data = {"path": str(row["path"]), "data_dir": str(row["data_dir"])}
            image_cache[image_name] = row_data
            path_cache[cls._norm_path(row["path"])] = image_name

        cls._image_name_to_row_cache = image_cache
        cls._path_to_image_name_cache = path_cache
        return image_cache

    @classmethod
    def from_image_name(cls, image_name: str):
        base_image_name = str(image_name).split("@")[0]
        cache = cls._load_image_name_cache()

        if base_image_name not in cache:
            raise AssertionError(f"Could not find {base_image_name} in adapter meta cache (n={len(cache)})")

        adapter_root = cls._adapter_root()
        row = cache[base_image_name]
        settings_path = adapter_root / "data" / "dataset_settings.json"
        settings = DatasetSettings(json.loads(settings_path.read_text(encoding="utf-8")))

        obj = cls(Path(row["path"]), Path(row["data_dir"]), adapter_root / "intermediates", settings)
        obj._adapter_image_name = base_image_name
        return obj

    def image_name(self) -> str:
        if hasattr(self, "_adapter_image_name"):
            return self._adapter_image_name

        self._load_image_name_cache()
        key = self._norm_path(self.image_dir)
        if self._path_to_image_name_cache is not None and key in self._path_to_image_name_cache:
            self._adapter_image_name = self._path_to_image_name_cache[key]
            return self._adapter_image_name

        return super().image_name()
'''

DATASET_MEDIAN_PIXEL_ADAPTER_CODE = r'''import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from htc.models.common.HTCDataset import HTCDataset
from htc.tivita.DataPath import DataPath
from htc.utils.Task import Task


class DatasetMedianPixelAdapter(HTCDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dataset_name = self.config.get("input/dataset_name", "Cat_HTC_Adapter")
        annotation_name = self.config.get("input/annotation_name", "semantic#primary")

        requested_image_names = [str(p.image_name()) for p in self.paths]
        requested_set = set(requested_image_names)
        requested_order = {name: i for i, name in enumerate(requested_image_names)}

        root = Path(os.environ["PATH_Tivita_Cat_HTC_Adapter"])
        table_path = root / "intermediates" / "tables" / f"{dataset_name}@median_spectra@{annotation_name}.pkl"

        if not table_path.exists():
            feather_path = root / "intermediates" / "tables" / f"{dataset_name}@median_spectra@{annotation_name}.feather"
            df_tmp = pd.read_feather(feather_path).reset_index(drop=True)
            df_tmp.to_pickle(table_path)

        df = pd.read_pickle(table_path)

        # Avoid pandas/pyarrow boolean indexing on Windows; use pure Python records.
        records = df.to_dict("records")
        records = [r for r in records if str(r["image_name"]) in requested_set]
        records.sort(key=lambda r: (requested_order.get(str(r["image_name"]), 10**12), str(r.get("annotation_name", "")), str(r.get("label_name", ""))))

        label_mapping = self.config["label_mapping"]
        self.labels = torch.tensor([int(label_mapping[str(r["label_name"])]) for r in records], dtype=torch.long) if label_mapping else None
        self.image_labels = None

        self.paths = [DataPath.from_image_name(f"{str(r['image_name'])}@{annotation_name}") for r in records]

        feature_columns = self.config.get("input/feature_columns", None)
        if feature_columns is None:
            feature_columns = ["median_normalized_spectrum"] if self.config["input/normalization"] == "L1" or "L1" in self.config["input/preprocessing"] else ["median_spectrum"]

        feature_arrays = []
        for r in records:
            parts = []
            for c in feature_columns:
                arr = np.asarray(r[c], dtype=np.float32)
                if arr.ndim == 0:
                    arr = np.expand_dims(arr, axis=0)
                parts.append(arr)
            feature_arrays.append(np.concatenate(parts, axis=0))

        self.features = torch.from_numpy(np.stack(feature_arrays).astype(np.float32).copy())
        self.features = self.apply_transforms(self.features)
        self.meta = torch.stack([self.read_meta(path) for path in self.paths]) if self.config["input/meta"] else None

        assert len(self.features) == len(self.paths)
        if self.labels is not None:
            assert len(self.labels) == len(self.features)

    def label_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        task = Task.from_config(self.config)
        return getattr(self, task.labels_name()).unique(return_counts=True)

    def __len__(self) -> int:
        task = Task.from_config(self.config)
        return len(getattr(self, task.labels_name()))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {"features": self.features[index, :]}
        if self.labels is not None:
            sample["labels"] = self.labels[index]
        if self.image_labels is not None:
            sample["image_labels"] = self.image_labels[index]
        if self.meta is not None:
            sample["meta"] = self.meta[index, :]
        if not self.train:
            path = self.paths[index]
            sample["image_name"] = path.image_name()
            sample["image_name_annotations"] = path.image_name_annotations()
            sample["image_index"] = index
        return sample
'''

DATAPATH_FORCE_PATCH = r'''

# --- Cat HTC Adapter from_image_name patch ---
try:
    import os

    if not hasattr(DataPath, "_adapter_original_from_image_name"):
        DataPath._adapter_original_from_image_name = DataPath.from_image_name

    def _adapter_aware_from_image_name(image_name, *args, **kwargs):
        if os.environ.get("PATH_Tivita_Cat_HTC_Adapter"):
            from htc.tivita.DataPathAdapter import DataPathAdapter
            return DataPathAdapter.from_image_name(image_name)
        return DataPath._adapter_original_from_image_name(image_name, *args, **kwargs)

    DataPath.from_image_name = staticmethod(_adapter_aware_from_image_name)

except Exception as e:
    print(f"[WARNING][htc] Could not install adapter-aware DataPath.from_image_name patch: {e}")
# --- End Cat HTC Adapter from_image_name patch ---
'''

LIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE = r'''from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from htc.models.common.HTCDataset import HTCDataset
from htc.models.median_pixel.DatasetMedianPixelAdapter import DatasetMedianPixelAdapter
from htc.models.median_pixel.LightningMedianPixel import LightningMedianPixel


class LightningMedianPixelAdapter(LightningMedianPixel):
    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        return DatasetMedianPixelAdapter(**kwargs)

    def on_test_start(self) -> None:
        self._adapter_test_prediction_rows = []
        parent = getattr(super(), "on_test_start", None)
        if callable(parent):
            parent()

    def _prediction_output_path(self) -> Path:
        root = getattr(getattr(self, "trainer", None), "log_dir", None)
        if root is None:
            root = getattr(getattr(self, "trainer", None), "default_root_dir", None)
        if root is None:
            root = "."
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        return root / "test_predictions.csv"

    @staticmethod
    def _as_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _predict_from_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        out = self.model(batch["features"])
        if isinstance(out, dict):
            out = out.get("class", out.get("logits", next(iter(out.values()))))
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim == 1:
            return out.detach().long()
        return torch.argmax(out.detach(), dim=1)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        result = super().test_step(batch, batch_idx)
        preds = self._predict_from_batch(batch).detach().cpu().tolist()
        labels = self._as_list(batch.get("labels"))
        image_names = self._as_list(batch.get("image_name"))
        image_name_annotations = self._as_list(batch.get("image_name_annotations"))
        image_indices = self._as_list(batch.get("image_index"))
        for i, pred in enumerate(preds):
            self._adapter_test_prediction_rows.append({
                "batch_idx": batch_idx,
                "row_in_batch": i,
                "image_index": image_indices[i] if i < len(image_indices) else "",
                "image_name": image_names[i] if i < len(image_names) else "",
                "image_name_annotations": image_name_annotations[i] if i < len(image_name_annotations) else "",
                "label_index_mapped": labels[i] if i < len(labels) else "",
                "prediction": pred,
            })
        return result

    def on_test_epoch_end(self) -> None:
        parent = getattr(super(), "on_test_epoch_end", None)
        if callable(parent):
            parent()
        rows = getattr(self, "_adapter_test_prediction_rows", [])
        out = self._prediction_output_path()
        if rows:
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[INFO][htc] Wrote adapter test predictions to {out}")
        else:
            print("[WARNING][htc] No adapter test prediction rows were collected")
'''

def compute_htc_image_names(manifest: pd.DataFrame, adapter_root: Path, annotation_name: str, dataset_name: str, label_mapping: dict[str, int], n_channels: int = 100) -> pd.DataFrame:
    from htc.tivita.DataPathTivita import DataPathTivita
    from htc.tivita.DatasetSettings import DatasetSettings
    dsettings = DatasetSettings({"dataset_name": dataset_name, "data_path_class": "htc.tivita.DataPathAdapter>DataPathAdapter", "annotation_name_default": annotation_name, "label_mapping": label_mapping, "last_valid_label_index": max(label_mapping.values()), "shape": [480, 640, int(n_channels)], "shape_names": ["height", "width", "channels"]})
    inter_dir = adapter_root / "intermediates"
    out = manifest.copy()
    out["htc_image_name"] = out.apply(lambda row: DataPathTivita(Path(row["sample_dir"]), Path(row["class_folder_path"]) / "data", inter_dir, dsettings).image_name(), axis=1)
    out["htc_image_name_annotation"] = out["htc_image_name"].astype(str) + "@" + annotation_name
    return out

def build_htc_adapter(manifest: pd.DataFrame, label_mapping: dict[str, int], args: argparse.Namespace, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    dataset_name = args.adapter_dataset_name
    annotation_name = args.annotation_name
    adapter_root = out_dir / "htc_adapter_dataset"
    tables_dir = adapter_root / "intermediates" / "tables"
    data_dir = adapter_root / "data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = compute_htc_image_names(manifest, adapter_root, annotation_name, dataset_name, label_mapping, int(args.n_channels))
    manifest.to_csv(out_dir / "median_pixel_yaml_manifest_htc_adapter.csv", index=False)
    settings = {"dataset_name": dataset_name, "data_path_class": "htc.tivita.DataPathAdapter>DataPathAdapter", "annotation_name_default": annotation_name, "label_mapping": label_mapping, "last_valid_label_index": max(label_mapping.values()), "shape": [480, 640, int(args.n_channels)], "shape_names": ["height", "width", "channels"]}
    (data_dir / "dataset_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    meta = manifest.sort_values(["htc_image_name", "label_index"]).drop_duplicates("htc_image_name")
    pd.DataFrame({"image_name": meta["htc_image_name"], "subject_name": meta["subject_name"], "timestamp": meta["timestamp"], "path": meta["sample_dir"], "data_dir": meta["class_folder_path"].apply(lambda p: str(Path(p) / "data"))}).to_feather(tables_dir / f"{dataset_name}@meta.feather")
    rows = []
    print(f"[INFO] Reading {len(manifest)} Excel spectra and writing median spectra table...")
    for i, row in manifest.iterrows():
        if i % 250 == 0:
            print(f"[INFO] spectra {i}/{len(manifest)}")
        spec = read_reflectance_spectrum(row["spectrum_xlsx_path"])
        norm = l1_normalize(spec)
        zeros = np.zeros_like(spec, dtype="float32")
        rows.append({"image_name": row["htc_image_name"], "subject_name": row["subject_name"], "timestamp": row["timestamp"], "label_index": int(row["label_index"]), "label_name": row["label_name"], "median_spectrum": spec.tolist(), "std_spectrum": zeros.tolist(), "median_normalized_spectrum": norm.tolist(), "std_normalized_spectrum": zeros.tolist(), "n_pixels": 1, "path": row["sample_dir"], "data_dir": str(Path(row["class_folder_path"]) / "data"), "annotation_name": annotation_name})
    pd.DataFrame(rows).reset_index(drop=True).to_feather(tables_dir / f"{dataset_name}@median_spectra@{annotation_name}.feather")
    # --- Scenario adapter cache creation patch ---
    # Create Windows-stable cache files for adapter readers before HTC training.
    # This avoids pandas/pyarrow feather crashes in DataPathAdapter and DatasetMedianPixelAdapter.
    meta_cache_path = tables_dir / f"{dataset_name}@meta_cache.csv"
    meta_path = tables_dir / f"{dataset_name}@meta.feather"
    if meta_path.exists():
        pd.read_feather(meta_path).reset_index(drop=True)[["image_name", "path", "data_dir"]].to_csv(meta_cache_path, index=False)
        print(f"[INFO] Wrote adapter meta cache: {meta_cache_path}")

    median_feather_path = tables_dir / f"{dataset_name}@median_spectra@{annotation_name}.feather"
    median_pickle_path = tables_dir / f"{dataset_name}@median_spectra@{annotation_name}.pkl"
    if median_feather_path.exists():
        pd.read_feather(median_feather_path).reset_index(drop=True).to_pickle(median_pickle_path)
        print(f"[INFO] Wrote adapter median spectra pickle cache: {median_pickle_path}")
    # --- End scenario adapter cache creation patch ---

    return adapter_root, manifest

def profile_defaults(profile: str, n_rows: int) -> dict[str, Any]:
    profiles = {
        "cpu_smoke": {"accelerator": "cpu", "devices": 1, "precision": None, "batch_size": 512, "num_workers": 0, "epoch_size": 2000, "max_epochs": 1},
        "cpu_practical": {"accelerator": "cpu", "devices": 1, "precision": None, "batch_size": 512, "num_workers": 0, "epoch_size": n_rows, "max_epochs": 10},
        "gpu_smoke": {"accelerator": "gpu", "devices": 1, "precision": "16-mixed", "batch_size": 512, "num_workers": 4, "epoch_size": 2000, "max_epochs": 1},
        "gpu_practical": {"accelerator": "gpu", "devices": 1, "precision": "16-mixed", "batch_size": 20000, "num_workers": 8, "epoch_size": n_rows, "max_epochs": 10},
        "gpu_paper": {"accelerator": "gpu", "devices": 1, "precision": "16-mixed", "batch_size": 20000, "num_workers": 12, "epoch_size": 10000000, "max_epochs": 10},
    }
    return profiles[profile].copy()

def apply_overrides(d: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for value, key in [(args.accelerator, "accelerator"), (args.devices, "devices"), (args.precision, "precision"), (args.batch_size, "batch_size"), (args.num_workers, "num_workers"), (args.max_epochs, "max_epochs")]:
        if value is not None:
            d[key] = value
    if args.epoch_size is not None:
        d["epoch_size"] = args.epoch_size
    return d

def write_data_spec(manifest: pd.DataFrame, out_dir: Path, use_adapter_names: bool) -> Path:
    image_col = "htc_image_name_annotation" if use_adapter_names else "image_name"
    path = out_dir / ("yaml_generated_median_pixel_data_spec_htc_adapter_annotated.json" if use_adapter_names else "yaml_generated_median_pixel_data_spec.json")
    path.write_text(json.dumps(split_by_subject(manifest, image_col=image_col), indent=2), encoding="utf-8")
    return path

def write_config(args: argparse.Namespace, out_dir: Path, label_mapping: dict[str, int], data_spec_name: str, n_rows: int) -> Path:
    d = apply_overrides(profile_defaults(args.training_profile, n_rows), args)
    config = {"lightning_class": "htc.models.median_pixel.LightningMedianPixelAdapter>LightningMedianPixelAdapter" if args.build_htc_adapter else "htc.models.median_pixel.LightningMedianPixel>LightningMedianPixel", "label_mapping": label_mapping, "input": {"data_spec": data_spec_name, "normalization": "L1", "n_channels": int(args.n_channels), "oversampling": False, "epoch_size": int(d["epoch_size"])}, "optimization": {"optimizer": {"name": "Adam", "lr": 0.0001, "weight_decay": 0}, "lr_scheduler": {"name": "ExponentialLR", "gamma": 0.9}}, "model": {"activation_function": "elu", "class_weight_method": "softmin", "softmin_scaling": -2.0, "normalization": "BatchNorm", "dropout": 0.2}, "dataloader_kwargs": {"batch_size": int(d["batch_size"]), "num_workers": int(d["num_workers"])}, "trainer_kwargs": {"max_epochs": int(d["max_epochs"]), "devices": int(d["devices"]), "accelerator": d["accelerator"]}, "swa_kwargs": {"annealing_epochs": 0}, "validation": {"checkpoint_metric": "accuracy"}}
    if d.get("precision"):
        config["trainer_kwargs"]["precision"] = d["precision"]
    if args.build_htc_adapter:
        config["input"]["dataset_name"] = args.adapter_dataset_name
        config["input"]["annotation_name"] = args.annotation_name
    path = out_dir / f"generated_median_pixel_config_htc_adapter_{args.training_profile}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path

def run_htc_training(config_path: Path, out_dir: Path, args: argparse.Namespace, adapter_root: Path | None) -> None:
    env = os.environ.copy()
    if adapter_root is not None:
        env["PATH_Tivita_" + re.sub(r"[^A-Za-z0-9_]", "_", args.adapter_dataset_name)] = str(adapter_root)
    if args.results_dir:
        env["PATH_HTC_RESULTS"] = str(Path(args.results_dir))
    env["HTC_ADD_NETWORK_ALTERNATIVES"] = "false"
    cmd = ["htc", "training", "--model", "median_pixel", "--config", config_path.name]
    if args.test:
        cmd.append("--test")
    print("[INFO] Running official HTC command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(out_dir), env=env, check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and run HTC median_pixel from manifold YAML with adapter support")
    parser.add_argument("--settings-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-root", default="")
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")
    parser.add_argument("--data-subdir", default="data")
    parser.add_argument("--default-labelling-file", default="_labelling_001.txt")
    parser.add_argument("--default-hypergui", default="_hypergui_1")
    parser.add_argument("--require-standardized-or-001", action="store_true")
    parser.add_argument("--csv-pattern", default="spectrum_fromCSV1*_masked_data.xlsx")
    parser.add_argument("--wavelength-min", type=float, default=500.0)
    parser.add_argument("--wavelength-max", type=float, default=995.0)
    parser.add_argument("--label-mode", default="official_atlas", choices=["official_atlas", "yaml_or_infer"])
    parser.add_argument("--label-mapping", default="generated_dict", choices=["generated_dict", "official_atlas_ref"])
    parser.add_argument("--build-htc-adapter", action="store_true")
    parser.add_argument("--adapter-dataset-name", default="Cat_HTC_Adapter")
    parser.add_argument("--annotation-name", default="semantic#primary")
    parser.add_argument("--htc-project-root", default="")
    parser.add_argument("--no-install-htc-adapter-code", action="store_true")
    parser.add_argument("--no-patch-htc", action="store_true")
    parser.add_argument("--training-profile", default="cpu_smoke", choices=["cpu_smoke", "cpu_practical", "gpu_smoke", "gpu_practical", "gpu_paper"])
    parser.add_argument("--accelerator", default=None, choices=["cpu", "gpu", "cuda", "auto"])
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epoch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    args = apply_yaml_run_settings(args)

    if args.wavelength_min > args.wavelength_max:
        raise ValueError("--wavelength-min must be <= --wavelength-max")
    args.n_channels = int(round((args.wavelength_max - args.wavelength_min) / 5.0)) + 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(Path(args.settings_yaml))
    globals()["YAML_SPLITS_CONFIG"] = cfg.get("splits", cfg.get("split_subjects", {})) or {}
    df = scan_yaml_dataset(cfg, args)
    labels = ordered_labels(df, cfg, args.label_mode)
    label_mapping = make_label_mapping(labels)
    # Normalize YAML class keys like "01_kidney" -> "kidney" so flexible YAML files work.
    # This prevents NaN label indices when label_mapping keys do not include numeric prefixes.
    df["label_name_raw"] = df["label_name"].astype(str)
    df["label_name"] = df["label_name_raw"].str.replace(r"^\d+_", "", regex=True)

    missing_labels = sorted(set(df["label_name"].dropna().astype(str)) - set(label_mapping.keys()))
    if missing_labels:
        if str(getattr(args, "label_mapping", "")).lower() == "generated_dict":
            labels_in_order = list(dict.fromkeys(df["label_name"].astype(str).tolist()))
            label_mapping = {label: i for i, label in enumerate(labels_in_order)}
            print(f"[INFO] Rebuilt generated label_mapping after YAML label normalization: {label_mapping}")
        else:
            raise ValueError(
                "Labels from YAML were not found in label_mapping after normalization. "
                f"Missing labels: {missing_labels}. Available mapping keys: {sorted(label_mapping.keys())}"
            )

    df["label_index"] = df["label_name"].map(label_mapping).astype(int)
    manifest_path = out_dir / "median_pixel_yaml_manifest.csv"
    df.to_csv(manifest_path, index=False)
    args._audit_df.to_csv(out_dir / "yaml_scan_audit.csv", index=False)
    (out_dir / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")
    adapter_root = None
    work_manifest = df
    if args.build_htc_adapter:
        htc_root = Path(args.htc_project_root) if args.htc_project_root else infer_htc_project_root()
        if not args.no_install_htc_adapter_code:
            install_htc_adapter_code(htc_root, patch_htc=not args.no_patch_htc)
        adapter_root, work_manifest = build_htc_adapter(df, label_mapping, args, out_dir)
    data_spec_path = write_data_spec(work_manifest, out_dir, use_adapter_names=args.build_htc_adapter)
    config_path = write_config(args, out_dir, label_mapping, data_spec_path.name, n_rows=len(work_manifest))
    summary = {"manifest": str(manifest_path), "n_manifest_rows": int(len(df)), "n_unique_images": int(df["image_name"].nunique()), "labels": labels, "label_mapping": label_mapping, "data_spec": str(data_spec_path), "config": str(config_path), "build_htc_adapter": bool(args.build_htc_adapter), "adapter_root": str(adapter_root) if adapter_root else None, "training_profile": args.training_profile}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.run_training:
        run_htc_training(config_path, out_dir, args, adapter_root)

if __name__ == "__main__":
    main()
