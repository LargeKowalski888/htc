from __future__ import annotations

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
