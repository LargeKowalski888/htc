#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re

LAUNCHER = Path(r"D:\HTC_github\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py")

NEW_FUNCTION = '''def split_by_subject(df: pd.DataFrame, image_col: str) -> list[dict[str, Any]]:
    """Create a subject-separated but organ-aware train/val/test split.

    The old split used sorted subjects and could leave some organs absent from
    the test set. This split greedily selects validation and test subjects so
    that every label present in the full manifest is represented in test and,
    when possible, validation too.
    """
    subjects = sorted(df["subject_name"].dropna().astype(str).unique().tolist())
    all_labels = set(df["label_name"].dropna().astype(str).unique().tolist())

    if not subjects:
        raise ValueError("No subjects available for train/val/test split")

    n_subjects = len(subjects)
    n_test_target = max(1, round(n_subjects * 0.2))
    n_val_target = max(1, round(n_subjects * 0.2))

    subject_to_labels = {
        s: set(df.loc[df["subject_name"].astype(str).eq(s), "label_name"].dropna().astype(str).unique().tolist())
        for s in subjects
    }
    subject_to_rows = {
        s: int(df.loc[df["subject_name"].astype(str).eq(s)].shape[0])
        for s in subjects
    }

    def greedy_subjects(available: list[str], target_n: int, required_labels: set[str]) -> list[str]:
        selected: list[str] = []
        uncovered = set(required_labels)
        remaining = list(available)

        while remaining and (uncovered or len(selected) < target_n):
            def score(s: str):
                labels = subject_to_labels.get(s, set())
                return (
                    len(labels & uncovered),
                    len(labels),
                    subject_to_rows.get(s, 0),
                    s,
                )

            best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)
            uncovered -= subject_to_labels.get(best, set())

            if len(remaining) <= 1 and len(selected) >= target_n:
                break

        return selected

    # Choose test first because confusion-matrix support is the key evaluation requirement.
    test_subjects = greedy_subjects(subjects, n_test_target, all_labels)
    remaining_after_test = [s for s in subjects if s not in set(test_subjects)]

    # Choose validation from remaining subjects.
    val_subjects = greedy_subjects(remaining_after_test, n_val_target, all_labels)
    train_subjects = [s for s in subjects if s not in set(test_subjects) and s not in set(val_subjects)]

    if not train_subjects:
        raise ValueError("Balanced split left no training subjects. Add more subjects or define split manually.")

    def names_for(subs: list[str]) -> list[str]:
        return sorted(df[df["subject_name"].astype(str).isin(subs)][image_col].drop_duplicates().tolist())

    def labels_for(subs: list[str]) -> set[str]:
        return set(df[df["subject_name"].astype(str).isin(subs)]["label_name"].dropna().astype(str).unique().tolist())

    split_subjects = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }

    print("[INFO] Organ-aware subject split:")
    for split_name, subs in split_subjects.items():
        split_labels = labels_for(subs)
        missing = sorted(all_labels - split_labels)
        label_counts = df[df["subject_name"].astype(str).isin(subs)]["label_name"].value_counts().sort_index()
        print(f"[INFO] {split_name}: n_subjects={len(subs)}, n_images={len(names_for(subs))}, n_labels={len(split_labels)}")
        print(label_counts.to_string())
        if missing:
            print(f"[WARNING] {split_name} split is missing labels: {missing}")

    test_missing = sorted(all_labels - labels_for(test_subjects))
    if test_missing:
        raise ValueError(
            "Test split is missing labels and confusion matrix would be incomplete: "
            + ", ".join(test_missing)
        )

    return [{
        "fold_name": "fold_yaml",
        "train": {"image_names": names_for(train_subjects)},
        "val": {"image_names": names_for(val_subjects)},
        "test": {"image_names": names_for(test_subjects)},
        "split_subjects": split_subjects,
    }]
'''


def main() -> None:
    path = LAUNCHER
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".bak_balanced_subject_split")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup written: {backup}")

    pattern = r"def split_by_subject\(df: pd\.DataFrame, image_col: str\) -> list\[dict\[str, Any\]\]:\n(?:(?!\ndef ).)*"
    new_text, n = re.subn(pattern, NEW_FUNCTION, text, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError(f"Could not replace split_by_subject function. replacements={n}")

    path.write_text(new_text, encoding="utf-8")
    print(f"Patched balanced subject split in: {path}")
    print("Future launcher runs will create test splits that include all labels when possible.")


if __name__ == "__main__":
    main()
