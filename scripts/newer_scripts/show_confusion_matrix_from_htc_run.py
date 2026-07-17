#!/usr/bin/env python
"""
Create a confusion matrix from an HTC run if a predictions/results table was saved.

Usage example:
python show_confusion_matrix_from_htc_run.py ^
  --run-dir "C:\\Cuyler2026\\htc_results\\training\\median_pixel\\running_2026-07-06_11-49-16_generated_median_pixel_config_htc_adapter_cpu_smoke" ^
  --label-mapping "C:\\Cuyler2026\\outputs\\final_cpu_smoke_test\\label_mapping.json" ^
  --output-png "C:\\Cuyler2026\\outputs\\final_cpu_smoke_test\\confusion_matrix.png" ^
  --output-csv "C:\\Cuyler2026\\outputs\\final_cpu_smoke_test\\confusion_matrix.csv" ^
  --list-candidates
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRUE_COLUMNS = [
    "label_index_mapped", "label_index", "labels", "label", "target", "targets",
    "y_true", "true_label", "true", "gt", "ground_truth",
]
PRED_COLUMNS = [
    "prediction", "predictions", "pred", "y_pred", "label_pred", "pred_label",
    "pred_index", "predicted_label", "class_prediction", "class_pred",
]
SCORE_COLUMNS = ["logits", "probabilities", "probs", "scores", "pred_scores"]
TABLE_EXTENSIONS = [".csv", ".tsv", ".feather", ".parquet", ".json", ".jsonl", ".pkl", ".pickle"]


def read_table(path: Path) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".feather":
            return pd.read_feather(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in [".json", ".jsonl"]:
            try:
                return pd.read_json(path, lines=True)
            except ValueError:
                return pd.read_json(path)
        if suffix in [".pkl", ".pickle"]:
            obj = pd.read_pickle(path)
            return obj if isinstance(obj, pd.DataFrame) else None
    except Exception:
        return None
    return None


def load_label_mapping(path: Path) -> tuple[list[int], list[str]]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    idx_to_name = {int(v): str(k) for k, v in mapping.items()}
    indices = sorted(idx_to_name)
    return indices, [idx_to_name[i] for i in indices]


def find_tables(run_dir: Path) -> list[Path]:
    files = []
    for ext in TABLE_EXTENSIONS:
        files.extend(run_dir.rglob(f"*{ext}"))
    def score(p: Path) -> tuple[int, float]:
        name = p.name.lower()
        s = 0
        if "test" in name:
            s += 20
        if "pred" in name:
            s += 20
        if "result" in name or "output" in name:
            s += 10
        if "metric" in name:
            s += 5
        if "config" in name:
            s -= 20
        return s, p.stat().st_mtime
    return sorted(files, key=score, reverse=True)


def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower:
            return lower[opt.lower()]
    return None


def parse_vector(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            try:
                return json.loads(x)
            except Exception:
                return x
    return x


def to_int_array(values) -> np.ndarray:
    out = []
    for x in values:
        x = parse_vector(x)
        if isinstance(x, (list, tuple, np.ndarray)):
            out.append(int(np.asarray(x).argmax()))
        else:
            out.append(int(x))
    return np.asarray(out, dtype=int)


def extract_predictions(df: pd.DataFrame):
    true_col = find_col(df, TRUE_COLUMNS)
    pred_col = find_col(df, PRED_COLUMNS)
    if true_col and pred_col:
        return to_int_array(df[true_col].values), to_int_array(df[pred_col].values), true_col, pred_col
    if true_col:
        score_col = find_col(df, SCORE_COLUMNS)
        if score_col:
            return to_int_array(df[true_col].values), to_int_array(df[score_col].values), true_col, score_col
    return None


def make_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_indices: list[int]) -> np.ndarray:
    pos = {idx: i for i, idx in enumerate(class_indices)}
    mat = np.zeros((len(class_indices), len(class_indices)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if int(t) in pos and int(p) in pos:
            mat[pos[int(t)], pos[int(p)]] += 1
    return mat


def save_plot(mat: np.ndarray, labels: list[str], output_png: Path, normalize: bool):
    display = mat.astype(float)
    if normalize:
        denom = display.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        display = display / denom
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.7), max(7, len(labels) * 0.6)))
    im = ax.imshow(display)
    ax.set_title("Confusion matrix" + (" normalized" if normalize else ""))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{display[i, j]:.2f}" if normalize else str(int(mat[i, j])), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)

    pdf_path = output_png.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf")

    print(f"[OK] wrote: {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--predictions-file", type=Path, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--list-candidates", action="store_true")
    args = parser.parse_args()

    class_indices, labels = load_label_mapping(args.label_mapping)
    candidates = [args.predictions_file] if args.predictions_file else find_tables(args.run_dir)
    inspected = []

    for p in candidates:
        if p is None or not p.exists():
            continue
        df = read_table(p)
        if df is None or df.empty:
            continue
        inspected.append((p, list(df.columns)))
        if args.list_candidates:
            print(f"[CANDIDATE] {p}")
            print(f"  columns={list(df.columns)}")
        extracted = extract_predictions(df)
        if extracted is None:
            continue
        y_true, y_pred, true_col, pred_col = extracted
        mat = make_matrix(y_true, y_pred, class_indices)
        out_df = pd.DataFrame(mat, index=labels, columns=labels)
        out_df.index.name = "true_label"
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)

        acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
        n_rows = int(len(y_true))

        # Write the normal confusion matrix CSV.
        out_df.to_csv(args.output_csv)

        # Write normalized confusion matrix values (0.0000-1.0000)
        normalized_df = out_df.astype(float)

        row_sums = normalized_df.sum(axis=1)
        row_sums[row_sums == 0] = 1

        normalized_df = normalized_df.div(row_sums, axis=0)

        normalized_csv = args.output_csv.with_name(
            args.output_csv.stem + "_normalized.csv"
        )

        normalized_df.round(4).to_csv(
            normalized_csv,
            float_format="%.4f"
        )

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
        # Convert to object dtype so pandas can safely store text summary rows
        # even when the matrix contains float values from normalization.
        summary_df = out_df.astype(object).copy()
        summary_df.loc["__accuracy__"] = [""] * len(summary_df.columns)
        summary_df.loc["__accuracy__", labels[0]] = acc
        summary_df.loc["__n_test_rows__"] = [""] * len(summary_df.columns)
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
        print(f"[OK] wrote: {normalized_csv}")
        print(f"[OK] wrote: {args.output_png}")
        return

    print("[ERROR] Could not find a prediction table with true and predicted labels.")
    print("[INFO] Inspected tables:")
    for p, cols in inspected[:50]:
        print(f"  {p}")
        print(f"    columns={cols}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
