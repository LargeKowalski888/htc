from pathlib import Path

path = Path(r"D:\HTC_github\htc\htc\tivita\DataPath.py")
text = path.read_text(encoding="utf-8")

snippet = 'import pandas as pd\npd.set_option("mode.string_storage", "python")\n'

if 'pd.set_option("mode.string_storage", "python")' in text:
    print("Already patched")
else:
    backup = path.with_suffix(path.suffix + ".bak_pandas_string_storage")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    lines = text.splitlines()

    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("from __future__ import"):
            insert_at = i + 1
            break

    lines.insert(insert_at, snippet.rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Patched {path}")