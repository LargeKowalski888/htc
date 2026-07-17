from pathlib import Path

dataset_path = Path(r"D:\HTC_github\htc\htc\models\median_pixel\DatasetMedianPixelAdapter.py")

code = r'''import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    pd.set_option("mode.string_storage", "python")
except Exception:
    pass

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

        root = os.environ["PATH_Tivita_Cat_HTC_Adapter"]
        table_path = Path(root) / "intermediates" / "tables" / f"{dataset_name}@median_spectra@{annotation_name}.pkl"

        if not table_path.exists():
            feather_path = Path(root) / "intermediates" / "tables" / f"{dataset_name}@median_spectra@{annotation_name}.feather"
            df_tmp = pd.read_feather(feather_path).reset_index(drop=True)
            df_tmp.to_pickle(table_path)

        df = pd.read_pickle(table_path).reset_index(drop=True)

        # Avoid pandas Arrow string dtype operations, which are causing Windows access violations.
        df["image_name"] = df["image_name"].astype(str).astype(object)
        df["label_name"] = df["label_name"].astype(str).astype(object)
        if "annotation_name" in df.columns:
            df["annotation_name"] = df["annotation_name"].astype(str).astype(object)

        # Avoid df["image_name"].isin(...) because pandas/pyarrow crashes there in this environment.
        mask = [name in requested_set for name in df["image_name"].tolist()]
        df = df.loc[mask].copy().reset_index(drop=True)

        df["__order"] = df["image_name"].map(requested_order)
        df.sort_values(["__order", "annotation_name", "label_name"], inplace=True, ignore_index=True)
        df.drop(columns=["__order"], inplace=True)

        label_mapping = self.config["label_mapping"]
        if label_mapping:
            df["label_index_mapped"] = df["label_name"].map(label_mapping).astype(int)
            self.labels = torch.from_numpy(df["label_index_mapped"].values.copy())
        else:
            self.labels = None

        image_labels_enabled = bool(self.config.get("input/image_labels", False))
        if image_labels_enabled and "image_labels" in df.columns:
            self.image_labels = torch.from_numpy(np.stack(df["image_labels"]))
        else:
            self.image_labels = None

        self.paths = [
            DataPath.from_image_name(f"{image_name}@{annotation_name}")
            for image_name in df["image_name"].tolist()
        ]

        feature_columns = self.config.get("input/feature_columns", None)
        if feature_columns is None:
            if self.config["input/normalization"] == "L1" or "L1" in self.config["input/preprocessing"]:
                feature_columns = ["median_normalized_spectrum"]
            else:
                feature_columns = ["median_spectrum"]

        features = []
        for c in feature_columns:
            arr = np.stack(df[c].values).copy()
            if arr.ndim == 1:
                arr = np.expand_dims(arr, axis=1)
            features.append(arr)

        self.features = torch.from_numpy(np.concatenate(features, axis=1).copy())
        self.features = self.apply_transforms(self.features)

        if self.config["input/meta"]:
            self.meta = torch.stack([self.read_meta(path) for path in self.paths])
        else:
            self.meta = None

        assert len(self.features) == len(self.paths)

        if self.labels is not None:
            assert len(self.labels) == len(self.features)

        if self.image_labels is not None:
            assert len(self.image_labels) == len(self.features)

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

backup = dataset_path.with_suffix(dataset_path.suffix + ".bak_no_arrow_isin")
if dataset_path.exists() and not backup.exists():
    backup.write_text(dataset_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup written: {backup}")

dataset_path.write_text(code, encoding="utf-8")
print(f"Patched {dataset_path}")