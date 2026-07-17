#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")


def main() -> None:
    if not LAUNCHER.exists():
        raise FileNotFoundError(LAUNCHER)

    text = LAUNCHER.read_text(encoding="utf-8")
    backup = LAUNCHER.with_suffix(LAUNCHER.suffix + ".bak_remove_split_subjects_metadata")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    old_lines = [
        '        "split_subjects": split_subjects,\n',
        "        'split_subjects': split_subjects,\n",
    ]

    changed = False
    for old in old_lines:
        if old in text:
            text = text.replace(old, "")
            changed = True

    if not changed:
        print("No split_subjects metadata line found. Launcher may already be fixed.")
    else:
        LAUNCHER.write_text(text, encoding="utf-8")
        print(f"Fixed HTC data spec output in: {LAUNCHER}")
        print("The generated JSON will now contain only fold_name/train/val/test, which HTC expects.")


if __name__ == "__main__":
    main()
