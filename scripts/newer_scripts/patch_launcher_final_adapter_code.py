#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")

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


def replace_block(text: str, name: str, code: str) -> str:
    pat = rf"{name}\s*=\s*r'''(.*?)'''"
    repl = f"{name} = r'''{code}'''"
    new, n = re.subn(pat, repl, text, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError(f"Could not replace block {name}; replacements={n}")
    return new


def main() -> None:
    path = LAUNCHER
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".bak_final_adapter_code")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    text = replace_block(text, "DATA_PATH_ADAPTER_CODE", DATA_PATH_ADAPTER_CODE)
    text = replace_block(text, "DATASET_MEDIAN_PIXEL_ADAPTER_CODE", DATASET_MEDIAN_PIXEL_ADAPTER_CODE)
    text = replace_block(text, "LIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE", LIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE)

    # Ensure generated feather tables are written with clean indexes.
    text = text.replace("meta_df.reset_index(drop=True).to_feather(meta_path)", "meta_df.reset_index(drop=True).to_feather(meta_path)")
    text = text.replace("pd.DataFrame(rows).to_feather(tables_dir / f\"{dataset_name}@median_spectra@{annotation_name}.feather\")", "pd.DataFrame(rows).reset_index(drop=True).to_feather(tables_dir / f\"{dataset_name}@median_spectra@{annotation_name}.feather\")")

    # Add robust DataPath patching after patch_windows_rename in install_htc_adapter_code.
    if "DATAPATH_FORCE_PATCH" not in text:
        insert_after = "patch_windows_rename(models_dir / \"run_training.py\")"
        replacement = insert_after + "\n        datapath_file = htc_project_root / \"htc\" / \"tivita\" / \"DataPath.py\"\n        dp_text = datapath_file.read_text(encoding=\"utf-8\")\n        if \"# --- Cat HTC Adapter from_image_name patch ---\" not in dp_text:\n            datapath_file.write_text(dp_text + DATAPATH_FORCE_PATCH, encoding=\"utf-8\")\n            print(f\"[INFO] Patched adapter-aware DataPath.from_image_name in {datapath_file}\")"
        text = text.replace(insert_after, replacement)
        text = text.replace("LIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE = r'''", "DATAPATH_FORCE_PATCH = r'''" + DATAPATH_FORCE_PATCH + "'''\n\nLIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE = r'''")

    path.write_text(text, encoding="utf-8")
    print(f"Patched launcher final adapter code: {path}")


if __name__ == "__main__":
    main()
