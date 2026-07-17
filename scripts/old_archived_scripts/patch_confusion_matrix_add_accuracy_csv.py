#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

SCRIPT = Path(r"D:\HTC_github\htc\scripts\newer_scripts\show_confusion_matrix_from_htc_run.py")


def main() -> None:
    if not SCRIPT.exists():
        raise FileNotFoundError(SCRIPT)

    text = SCRIPT.read_text(encoding="utf-8")
    backup = SCRIPT.with_suffix(SCRIPT.suffix + ".bak_accuracy_csv")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    # Patch imports only if needed.
    if "from pathlib import Path" in text and "import json" not in text:
        text = text.replace("from pathlib import Path\n", "from pathlib import Path\nimport json\n", 1)

    old = '''        out_df = pd.DataFrame(mat, index=labels, columns=labels)
        out_df.index.name = "true_label"
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_csv)
        save_plot(mat, labels, args.output_png, args.normalize)
        acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
        print(f"[OK] prediction table: {p}")
        print(f"[OK] true column: {true_col}")
        print(f"[OK] pred column: {pred_col}")
        print(f"[OK] rows: {len(y_true)}")
        print(f"[OK] accuracy: {acc:.4f}")
        print(f"[OK] wrote: {args.output_csv}")
        print(f"[OK] wrote: {args.output_png}")
        return
'''

    new = '''        out_df = pd.DataFrame(mat, index=labels, columns=labels)
        out_df.index.name = "true_label"
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)

        acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
        n_rows = int(len(y_true))

        # Write the normal confusion matrix CSV.
        out_df.to_csv(args.output_csv)

        # Also write an easy-to-read metrics CSV next to the matrix CSV.
        metrics_csv = args.output_csv.with_name(args.output_csv.stem + "_metrics.csv")
        metrics_df = pd.DataFrame([
            {"metric": "accuracy", "value": acc},
            {"metric": "n_test_rows", "value": n_rows},
            {"metric": "prediction_table", "value": str(p)},
            {"metric": "true_column", "value": str(true_col)},
            {"metric": "prediction_column", "value": str(pred_col)},
        ])
        metrics_df.to_csv(metrics_csv, index=False)

        # Add a summary row to a second copy of the confusion matrix for supervisors
        # who prefer one CSV file containing both matrix and accuracy.
        summary_csv = args.output_csv.with_name(args.output_csv.stem + "_with_accuracy.csv")
        summary_df = out_df.copy()
        summary_df.loc["__accuracy__", :] = ""
        summary_df.loc["__accuracy__", labels[0]] = acc
        summary_df.loc["__n_test_rows__", :] = ""
        summary_df.loc["__n_test_rows__", labels[0]] = n_rows
        summary_df.to_csv(summary_csv)

        save_plot(mat, labels, args.output_png, args.normalize)
        print(f"[OK] prediction table: {p}")
        print(f"[OK] true column: {true_col}")
        print(f"[OK] pred column: {pred_col}")
        print(f"[OK] rows: {n_rows}")
        print(f"[OK] accuracy: {acc:.4f}")
        print(f"[OK] wrote: {args.output_csv}")
        print(f"[OK] wrote: {metrics_csv}")
        print(f"[OK] wrote: {summary_csv}")
        print(f"[OK] wrote: {args.output_png}")
        return
'''

    if old not in text:
        if "_metrics.csv" in text and "_with_accuracy.csv" in text:
            print("Script already appears patched.")
        else:
            raise RuntimeError("Could not find expected CSV-writing block. The script may have changed.")
    else:
        text = text.replace(old, new)
        SCRIPT.write_text(text, encoding="utf-8")
        print(f"Patched accuracy CSV output in: {SCRIPT}")
        print("New outputs will include *_metrics.csv and *_with_accuracy.csv next to the confusion matrix CSV.")


if __name__ == "__main__":
    main()
