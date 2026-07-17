from pathlib import Path
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
