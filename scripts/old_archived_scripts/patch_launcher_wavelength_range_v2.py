#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")

NEW_READ_FUNCTION = '''def read_reflectance_spectrum(xlsx_path: str, wavelength_min: float = 500.0, wavelength_max: float = 995.0) -> np.ndarray:
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
'''


def replace_function_by_name(text: str, func: str, new_func: str) -> str:
    start = text.find(f"def {func}(")
    if start < 0:
        raise RuntimeError(f"Could not find function {func}")
    next_def = text.find("\ndef ", start + 1)
    if next_def < 0:
        end = len(text)
    else:
        end = next_def + 1
    return text[:start] + new_func + "\n" + text[end:]


def main() -> None:
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)

    text = LAUNCHER.read_text(encoding="utf-8")
    backup = LAUNCHER.with_suffix(LAUNCHER.suffix + ".bak_wavelength_range_v2")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    # Add args if the first failed patch has not already added them.
    if "--wavelength-min" not in text:
        anchor = '    parser.add_argument("--csv-pattern", default="spectrum_fromCSV1*_masked_data.xlsx")\n'
        text = text.replace(anchor, anchor + '    parser.add_argument("--wavelength-min", type=float, default=500.0)\n    parser.add_argument("--wavelength-max", type=float, default=995.0)\n')
        print("Added wavelength args")
    else:
        print("Wavelength args already present")

    text = replace_function_by_name(text, "read_reflectance_spectrum", NEW_READ_FUNCTION)
    print("Replaced read_reflectance_spectrum")

    if "args.n_channels = int(round((args.wavelength_max - args.wavelength_min) / 5.0)) + 1" not in text:
        target = "    args = parser.parse_args()\n"
        text = text.replace(target, target + "\n    if args.wavelength_min > args.wavelength_max:\n        raise ValueError(\"--wavelength-min must be <= --wavelength-max\")\n    args.n_channels = int(round((args.wavelength_max - args.wavelength_min) / 5.0)) + 1\n", 1)
        print("Added n_channels calculation")
    else:
        print("n_channels calculation already present")

    text = text.replace('spectrum = read_reflectance_spectrum(row["spectrum_xlsx_path"])', 'spectrum = read_reflectance_spectrum(row["spectrum_xlsx_path"], args.wavelength_min, args.wavelength_max)')

    text = text.replace('"n_channels": 100', '"n_channels": int(args.n_channels)')
    text = text.replace('"shape": [480, 640, 100]', '"shape": [480, 640, int(args.n_channels)]')

    if '"wavelength_min": float(args.wavelength_min)' not in text:
        anchor = '        "training_profile": args.training_profile,\n'
        if anchor in text:
            text = text.replace(anchor, anchor + '        "wavelength_min": float(args.wavelength_min),\n        "wavelength_max": float(args.wavelength_max),\n        "n_channels": int(args.n_channels),\n')

    LAUNCHER.write_text(text, encoding="utf-8")
    print(f"Patched wavelength range support in {LAUNCHER}")


if __name__ == "__main__":
    main()
