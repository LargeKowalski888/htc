import os
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
