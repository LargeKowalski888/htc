from pathlib import Path

path = Path(r"D:\HTC_github\htc\htc\tivita\DataPath.py")
text = path.read_text(encoding="utf-8")

marker = "# --- Cat HTC Adapter from_image_name patch ---"

if marker in text:
    print("Already patched")
else:
    backup = path.with_suffix(path.suffix + ".bak_force_adapter")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    patch = r'''

# --- Cat HTC Adapter from_image_name patch ---
# If the exported HyperGui/CSV1 adapter dataset is registered, bypass the
# default DataPath cache builder and use DataPathAdapter directly. This avoids
# crashes in pandas/pyarrow while HTC scans the adapter meta feather table.
try:
    import os

    _original_from_image_name = DataPath.from_image_name.__func__

    def _adapter_aware_from_image_name(cls, image_name, *args, **kwargs):
        if os.environ.get("PATH_Tivita_Cat_HTC_Adapter"):
            try:
                from htc.tivita.DataPathAdapter import DataPathAdapter
                return DataPathAdapter.from_image_name(image_name)
            except Exception:
                # If adapter lookup fails, fall back to original HTC behavior
                # so non-adapter datasets still work.
                pass

        return _original_from_image_name(cls, image_name, *args, **kwargs)

    DataPath.from_image_name = classmethod(_adapter_aware_from_image_name)

except Exception as e:
    print(f"[WARNING][htc] Could not install adapter-aware DataPath.from_image_name patch: {e}")
# --- End Cat HTC Adapter from_image_name patch ---
'''

    path.write_text(text + patch, encoding="utf-8")
    print(f"Patched {path}")