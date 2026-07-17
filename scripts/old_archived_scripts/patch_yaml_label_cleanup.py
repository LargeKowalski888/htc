from pathlib import Path
import re

TARGET = Path(
    r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py"
)

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

text = TARGET.read_text(encoding="utf-8")

if "ROOT_CAUSE_YAML_LABEL_FIX_V1" in text:
    print("[INFO] YAML label fix already installed")
    raise SystemExit(0)

# =====================================================
# Add normalize_yaml_label()
# =====================================================

helper = r'''
# ROOT_CAUSE_YAML_LABEL_FIX_V1
def normalize_yaml_label(key: str) -> str:
    """
    Convert:

        01_kidney
        02_ureter
        06_artery
        15_tun_vag

    to:

        kidney
        ureter
        artery
        tun_vag
    """
    key = str(key).strip()
    key = re.sub(r"^\d+_", "", key)
    return key

'''

if "def normalize_yaml_label(" not in text:

    anchor = "def infer_label("

    idx = text.find(anchor)

    if idx == -1:
        raise RuntimeError(
            "Could not locate infer_label()"
        )

    text = text[:idx] + helper + "\n" + text[idx:]

    print("[INFO] Added normalize_yaml_label()")

# =====================================================
# Replace get_label_for_key()
# =====================================================

start = text.find(
    "def get_label_for_key("
)

if start == -1:
    raise RuntimeError(
        "Could not locate get_label_for_key()"
    )

end = text.find(
    "\ndef value_for_key(",
    start,
)

if end == -1:
    raise RuntimeError(
        "Could not locate end of get_label_for_key()"
    )

replacement = '''
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

'''

text = text[:start] + replacement + text[end:]

# =====================================================
# Write backup
# =====================================================

backup = TARGET.with_suffix(
    TARGET.suffix + ".bak_yaml_labels"
)

if not backup.exists():
    backup.write_text(
        TARGET.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

TARGET.write_text(text, encoding="utf-8")

print()
print("[SUCCESS] Installed ROOT_CAUSE_YAML_LABEL_FIX_V1")
print("[SUCCESS] Future manifests will use YAML labels")
print()
print("Examples:")
print("  01_kidney      -> kidney")
print("  02_ureter      -> ureter")
print("  03_bladder     -> bladder")
print("  04_prostate    -> prostate")
print("  06_artery      -> artery")
print("  17_epididymis  -> epididymis")
print()
print("Next steps:")
print("  1. Rebuild HTC adapter dataset")
print("  2. Retrain")
print("  3. Verify label_mapping.json")
