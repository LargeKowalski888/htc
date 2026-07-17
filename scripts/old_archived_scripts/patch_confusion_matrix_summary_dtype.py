#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

SCRIPT = Path(r"D:\HTC_github\htc\scripts\newer_scripts\show_confusion_matrix_from_htc_run.py")


def main() -> None:
    if not SCRIPT.exists():
        raise FileNotFoundError(SCRIPT)

    text = SCRIPT.read_text(encoding="utf-8")
    backup = SCRIPT.with_suffix(SCRIPT.suffix + ".bak_summary_dtype")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    old = '''        summary_df = out_df.copy()
        summary_df.loc["__accuracy__", :] = ""
        summary_df.loc["__accuracy__", labels[0]] = acc
        summary_df.loc["__n_test_rows__", :] = ""
        summary_df.loc["__n_test_rows__", labels[0]] = n_rows
        summary_df.to_csv(summary_csv)
'''

    new = '''        # Convert to object dtype so pandas can safely store text summary rows
        # even when the matrix contains float values from normalization.
        summary_df = out_df.astype(object).copy()
        summary_df.loc["__accuracy__"] = [""] * len(summary_df.columns)
        summary_df.loc["__accuracy__", labels[0]] = acc
        summary_df.loc["__n_test_rows__"] = [""] * len(summary_df.columns)
        summary_df.loc["__n_test_rows__", labels[0]] = n_rows
        summary_df.to_csv(summary_csv)
'''

    if old in text:
        text = text.replace(old, new)
        SCRIPT.write_text(text, encoding="utf-8")
        print(f"Patched summary CSV dtype issue in: {SCRIPT}")
    elif "summary_df = out_df.astype(object).copy()" in text:
        print("Script already patched.")
    else:
        raise RuntimeError("Could not find expected summary_df block. Please inspect show_confusion_matrix_from_htc_run.py manually.")


if __name__ == "__main__":
    main()
