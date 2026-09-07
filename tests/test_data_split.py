"""
Unit tests for src/data_split.py. Uses a small synthetic dataset at the
SAME class ratios as the real Davidson data, so this runs anywhere
without needing the actual (gitignored) dataset file.
"""

import pandas as pd

from src.data_split import make_splits

LABEL_NAMES = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}


def _make_synthetic_csv(tmp_path, n_hate=143, n_offensive=1919, n_neither=416):
    labels = [0] * n_hate + [1] * n_offensive + [2] * n_neither
    df = pd.DataFrame(
        {
            "text": [f"sample {i}" for i in range(len(labels))],
            "clean_text": [f"sample {i}" for i in range(len(labels))],
            "label": labels,
            "label_name": [LABEL_NAMES[label] for label in labels],
        }
    )
    csv_path = tmp_path / "cleaned_labeled_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path), df


def test_splits_sum_to_original_row_count(tmp_path):
    csv_path, df = _make_synthetic_csv(tmp_path)
    train, val, test = make_splits(cleaned_csv_path=csv_path, out_dir=str(tmp_path))
    assert len(train) + len(val) + len(test) == len(df)


def test_no_overlap_between_splits(tmp_path):
    csv_path, _ = _make_synthetic_csv(tmp_path)
    train, val, test = make_splits(cleaned_csv_path=csv_path, out_dir=str(tmp_path))
    assert len(set(train.index) & set(val.index)) == 0
    assert len(set(train.index) & set(test.index)) == 0
    assert len(set(val.index) & set(test.index)) == 0


def test_stratification_preserves_class_ratios(tmp_path):
    csv_path, df = _make_synthetic_csv(tmp_path)
    train, val, test = make_splits(cleaned_csv_path=csv_path, out_dir=str(tmp_path))

    original_ratios = df["label"].value_counts(normalize=True)
    for split in (train, val, test):
        split_ratios = split["label"].value_counts(normalize=True)
        for label in original_ratios.index:
            # within 3 percentage points is plenty tight for a stratified split
            assert abs(split_ratios[label] - original_ratios[label]) < 0.03


def test_split_files_written_to_disk(tmp_path):
    csv_path, _ = _make_synthetic_csv(tmp_path)
    make_splits(cleaned_csv_path=csv_path, out_dir=str(tmp_path))
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()
