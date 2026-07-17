#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")

NEW_READ_FUNCTION = r'''def read_reflectance_spectrum(xlsx_path: str, wavelength_min: float = 500.0, wavelength_max: float = 995.0) -> np.ndarray:
    """Read CSV1 spectrum and keep only the requested wavelength range.

    HyperGui CSV1 Excel files are expected to have wavelength in column 0 and
    reflectance/spectrum value in column 1 on sheet "0_derivative".
    The range is inclusive, e.g. 500-600 keeps 500, 505, ..., 600.
    """
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
        raise ValueError(
            f"No spectral channels selected in {xlsx_path} for wavelength range "
            f"{wavelength_min}-{wavelength_max} nm"
        )

    return spectrum
'''


def replace_function(text: str, func_name: str, new_func: str) -> tuple[str, int]:
    pattern = rf"def {func_name}\([^\n]*\):\n(?:(?!\ndef ).)*"
    return re.subn(pattern, new_func, text, count=1, flags=re.DOTALL)


def patch_function_body(text: str, func_name: str, replacements: list[tuple[str, str]]) -> str:
    pattern = rf"(def {func_name}\([^\n]*\):\n(?:(?!\ndef ).)*)"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not find function {func_name}")
    body = m.group(1)
    new_body = body
    for old, new in replacements:
        if old not in new_body:
            raise RuntimeError(f"Could not find expected text in {func_name}: {old}")
        new_body = new_body.replace(old, new)
    return text[:m.start(1)] + new_body + text[m.end(1):]


def main() -> None:
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)

    text = LAUNCHER.read_text(encoding="utf-8")
    backup = LAUNCHER.with_suffix(LAUNCHER.suffix + ".bak_wavelength_range")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    # 1) Add CLI arguments.
    if "--wavelength-min" not in text:
        anchor = '    parser.add_argument("--csv-pattern", default="spectrum_fromCSV1*_masked_data.xlsx")\n'
        if anchor not in text:
            raise RuntimeError("Could not find --csv-pattern parser line")
        insert = anchor + '    parser.add_argument("--wavelength-min", type=float, default=500.0, help="Minimum wavelength in nm, inclusive")\n' + '    parser.add_argument("--wavelength-max", type=float, default=995.0, help="Maximum wavelength in nm, inclusive")\n'
        text = text.replace(anchor, insert)
        print("Added --wavelength-min/--wavelength-max arguments")
    else:
        print("Wavelength CLI arguments already present")

    # 2) Replace spectrum reader.
    text, n = replace_function(text, "read_reflectance_spectrum", NEW_READ_FUNCTION)
    if n != 1:
        raise RuntimeError(f"Could not replace read_reflectance_spectrum, replacements={n}")
    print("Patched read_reflectance_spectrum()")

    # 3) Compute args.n_channels after parse_args.
    if "args.n_channels = int(round((args.wavelength_max - args.wavelength_min) / 5.0)) + 1" not in text:
        old = "    args = parser.parse_args()\n"
        new = old + "\n" + "    if args.wavelength_min > args.wavelength_max:\n" + "        raise ValueError(\"--wavelength-min must be <= --wavelength-max\")\n" + "    args.n_channels = int(round((args.wavelength_max - args.wavelength_min) / 5.0)) + 1\n"
        if old not in text:
            raise RuntimeError("Could not find args = parser.parse_args()")
        text = text.replace(old, new, 1)
        print("Added args.n_channels calculation")
    else:
        print("args.n_channels calculation already present")

    # 4) Read selected wavelength range in build_htc_adapter.
    old_call = 'spectrum = read_reflectance_spectrum(row["spectrum_xlsx_path"])'
    new_call = 'spectrum = read_reflectance_spectrum(row["spectrum_xlsx_path"], args.wavelength_min, args.wavelength_max)'
    if old_call in text:
        text = text.replace(old_call, new_call)
        print("Patched spectrum read call in build_htc_adapter()")
    elif new_call in text:
        print("Spectrum read call already patched")
    else:
        raise RuntimeError("Could not find spectrum read call")

    # 5) Patch compute_htc_image_names signature and its shape.
    old_sig = "def compute_htc_image_names(manifest: pd.DataFrame, adapter_root: Path, annotation_name: str, dataset_name: str, label_mapping: dict[str, int]) -> pd.DataFrame:"
    new_sig = "def compute_htc_image_names(manifest: pd.DataFrame, adapter_root: Path, annotation_name: str, dataset_name: str, label_mapping: dict[str, int], n_channels: int = 100) -> pd.DataFrame:"
    if old_sig in text:
        text = text.replace(old_sig, new_sig)
        print("Patched compute_htc_image_names() signature")
    elif new_sig in text:
        print("compute_htc_image_names() signature already patched")
    else:
        print("Could not find exact compute_htc_image_names signature; skipping signature patch")

    # Replace the shape inside compute_htc_image_names if present.
    try:
        text = patch_function_body(text, "compute_htc_image_names", [
            ('"shape": [480, 640, 100]', '"shape": [480, 640, int(n_channels)]')
        ])
        print("Patched compute_htc_image_names() dataset shape")
    except RuntimeError as e:
        if 'int(n_channels)' in text:
            print("compute_htc_image_names() shape already patched")
        else:
            print(f"WARNING: {e}")

    old_compute_call = "manifest = compute_htc_image_names(manifest, adapter_root, annotation_name, dataset_name, label_mapping)"
    new_compute_call = "manifest = compute_htc_image_names(manifest, adapter_root, annotation_name, dataset_name, label_mapping, int(args.n_channels))"
    if old_compute_call in text:
        text = text.replace(old_compute_call, new_compute_call)
        print("Patched compute_htc_image_names() call")
    elif new_compute_call in text:
        print("compute_htc_image_names() call already patched")
    else:
        print("WARNING: Could not find compute_htc_image_names() call")

    # 6) Patch build_htc_adapter dataset_settings shape.
    try:
        text = patch_function_body(text, "build_htc_adapter", [
            ('"shape": [480, 640, 100]', '"shape": [480, 640, int(args.n_channels)]')
        ])
        print("Patched build_htc_adapter() dataset shape")
    except RuntimeError as e:
        if 'int(args.n_channels)' in text:
            print("build_htc_adapter() shape already patched")
        else:
            print(f"WARNING: {e}")

    # 7) Patch generated config input/n_channels.
    if '"n_channels": 100' in text:
        text = text.replace('"n_channels": 100', '"n_channels": int(args.n_channels)')
        print("Patched generated config n_channels")
    elif '"n_channels": int(args.n_channels)' in text:
        print("Generated config n_channels already patched")
    else:
        print("WARNING: Could not find config n_channels line")

    # 8) Add wavelength info to summary if possible.
    if '"wavelength_min": float(args.wavelength_min)' not in text:
        anchor = '        "training_profile": args.training_profile,\n'
        insert = anchor + '        "wavelength_min": float(args.wavelength_min),\n' + '        "wavelength_max": float(args.wavelength_max),\n' + '        "n_channels": int(args.n_channels),\n'
        if anchor in text:
            text = text.replace(anchor, insert)
            print("Added wavelength info to run_summary.json")
        else:
            print("WARNING: Could not add wavelength info to summary")

    LAUNCHER.write_text(text, encoding="utf-8")
    print(f"Patched launcher: {LAUNCHER}")
    print("\nAdd these to your .bat file if desired:")
    print("set WAVELENGTH_MIN=500")
    print("set WAVELENGTH_MAX=600")
    print("and pass: --wavelength-min %WAVELENGTH_MIN% --wavelength-max %WAVELENGTH_MAX%")


if __name__ == "__main__":
    main()
