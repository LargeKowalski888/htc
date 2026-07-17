from pathlib import Path

adapter_root = Path(r"D:\HTC_github\outputs\median_pixel_yaml_gpu_smoke\htc_adapter_dataset")
meta_feather = adapter_root / "intermediates" / "tables" / "Cat_HTC_Adapter@meta.feather"
meta_csv = adapter_root / "intermediates" / "tables" / "Cat_HTC_Adapter@meta_cache.csv"

# First create a simple CSV cache from the feather file.
import pandas as pd

df = pd.read_feather(meta_feather).reset_index(drop=True)
df[["image_name", "path", "data_dir"]].to_csv(meta_csv, index=False)
print(f"Wrote CSV cache: {meta_csv}")

# Now patch DataPathAdapter.py to read the CSV cache instead of feather.
path = Path(r"D:\HTC_github\htc\htc\tivita\DataPathAdapter.py")

code = r'''from pathlib import Path
import json
import os

import pandas as pd

from htc.tivita.DataPathTivita import DataPathTivita
from htc.tivita.DatasetSettings import DatasetSettings


class DataPathAdapter(DataPathTivita):
    _path_to_image_name_cache = {}
    _image_name_to_row_cache = None

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
            raise FileNotFoundError(f"Adapter meta CSV cache does not exist: {csv_path}")

        df = pd.read_csv(csv_path)

        cache = {}
        for _, row in df.iterrows():
            cache[str(row["image_name"])] = {
                "path": str(row["path"]),
                "data_dir": str(row["data_dir"]),
            }

        cls._image_name_to_row_cache = cache
        return cache

    @classmethod
    def from_image_name(cls, image_name: str):
        base_image_name = str(image_name).split("@")[0]

        cache = cls._load_image_name_cache()
        if base_image_name not in cache:
            raise AssertionError(
                f"Could not find {base_image_name} in adapter meta CSV cache "
                f"(n={len(cache)})"
            )

        adapter_root = cls._adapter_root()
        row = cache[base_image_name]

        settings_path = adapter_root / "data" / "dataset_settings.json"
        settings = DatasetSettings(json.loads(settings_path.read_text(encoding="utf-8")))

        return cls(
            Path(row["path"]),
            Path(row["data_dir"]),
            adapter_root / "intermediates",
            settings,
        )

    @classmethod
    def _load_meta_cache(cls, intermediates_dir: Path) -> None:
        cache = cls._load_image_name_cache()
        for image_name, row in cache.items():
            cls._path_to_image_name_cache[cls._norm_path(row["path"])] = image_name

    def image_name(self) -> str:
        self._load_meta_cache(self.intermediates_dir)
        key = self._norm_path(self.image_dir)
        if key in self._path_to_image_name_cache:
            return self._path_to_image_name_cache[key]
        return super().image_name()
'''

backup = path.with_suffix(path.suffix + ".bak_csv_cache")
if path.exists() and not backup.exists():
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup written: {backup}")

path.write_text(code, encoding="utf-8")
print(f"Patched {path}")