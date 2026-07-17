#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re

HTC_ROOT = Path(r"C:\Programme\HTC_github\htc")
ADAPTER_PATH = HTC_ROOT / "htc" / "models" / "median_pixel" / "LightningMedianPixelAdapter.py"
LAUNCHER_PATH = HTC_ROOT / "scripts" / "newer_scripts" / "htc_median_pixel_from_yaml_official.py"

ADAPTER_CODE = r'''from __future__ import annotations

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
        self._adapter_prediction_export_errors = []
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
        # Match HTC's own median-pixel forward path as closely as possible.
        # For ModelPixel, self.model(features) usually returns a dict with key "class".
        out = self.model(batch["features"])
        if isinstance(out, dict):
            if "class" in out:
                out = out["class"]
            elif "logits" in out:
                out = out["logits"]
            elif "prediction" in out:
                out = out["prediction"]
            else:
                out = next(iter(out.values()))
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim == 1:
            return out.detach().long()
        return torch.argmax(out.detach(), dim=1)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        result = super().test_step(batch, batch_idx)
        try:
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
        except Exception as e:
            self._adapter_prediction_export_errors.append(f"batch_idx={batch_idx}: {type(e).__name__}: {e}")
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
            errors = getattr(self, "_adapter_prediction_export_errors", [])
            for msg in errors[:10]:
                print(f"[WARNING][htc] Prediction export error: {msg}")
'''


def write_adapter() -> None:
    ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    backup = ADAPTER_PATH.with_suffix(ADAPTER_PATH.suffix + ".bak_prediction_export_v2")
    if ADAPTER_PATH.exists() and not backup.exists():
        backup.write_text(ADAPTER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup written: {backup}")
    ADAPTER_PATH.write_text(ADAPTER_CODE, encoding="utf-8")
    print(f"Patched adapter: {ADAPTER_PATH}")


def patch_launcher_embedded_code() -> None:
    if not LAUNCHER_PATH.exists():
        print(f"Launcher not found, skipped: {LAUNCHER_PATH}")
        return
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    backup = LAUNCHER_PATH.with_suffix(LAUNCHER_PATH.suffix + ".bak_prediction_export_v2")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")
    pattern = r"LIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE\s*=\s*r'''(.*?)'''"
    replacement = "LIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE = r'''" + ADAPTER_CODE + "'''"
    new_text, n = re.subn(pattern, replacement, text, flags=re.DOTALL)
    if n == 0:
        print("Could not find embedded LIGHTNING_MEDIAN_PIXEL_ADAPTER_CODE block in launcher. Skipped launcher update.")
    else:
        LAUNCHER_PATH.write_text(new_text, encoding="utf-8")
        print(f"Patched launcher embedded adapter code: {LAUNCHER_PATH}")


def main() -> None:
    write_adapter()
    patch_launcher_embedded_code()
    print("Done. Rerun cpu_smoke or cpu_practical, then run the confusion matrix script again.")


if __name__ == "__main__":
    main()
