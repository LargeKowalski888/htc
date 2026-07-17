from pathlib import Path

SCRIPT = Path(
    r"D:\HTC_github\htc\scripts\newer_scripts\show_confusion_matrix_from_htc_run.py"
)

text = SCRIPT.read_text(encoding="utf-8")

if 'fig.savefig(pdf_path, format="pdf"' in text:
    print("[INFO] PDF export already installed.")
    raise SystemExit(0)

old = """    fig.savefig(output_png, dpi=200)
    plt.close(fig)
"""

new = """    fig.savefig(output_png, dpi=200)

    pdf_path = output_png.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf")

    print(f"[OK] wrote: {pdf_path}")

    plt.close(fig)
"""

if old not in text:
    raise RuntimeError(
        "Could not locate save_plot() save block in "
        "show_confusion_matrix_from_htc_run.py"
    )

text = text.replace(old, new, 1)

SCRIPT.write_text(text, encoding="utf-8")

print("[INFO] Installed PDF confusion-matrix export.")
print(f"[INFO] Patched: {SCRIPT}")