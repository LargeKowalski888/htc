from pathlib import Path

files = [
    Path(r"D:\HTC_github\htc\htc\tivita\DataPathAdapter.py"),
    Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py"),
]

good_line = '        return str(Path(path)).replace("/", "\\\\").rstrip("\\\\").lower()'

for path in files:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False

    new_lines = []
    for line in lines:
        if "return str(Path(path)).replace(" in line and ".rstrip(" in line and ".lower()" in line:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(indent + 'return str(Path(path)).replace("/", "\\\\").rstrip("\\\\").lower()')
            changed = True
        else:
            new_lines.append(line)

    if changed:
        backup = path.with_suffix(path.suffix + ".bak_backslash_escape")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
            print(f"Backup written: {backup}")

        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"Fixed: {path}")
    else:
        print(f"No matching broken line found in: {path}")