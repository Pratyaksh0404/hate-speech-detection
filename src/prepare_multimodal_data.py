"""
One-time data preparation for the MultiOFF multimodal dataset
(master_plan.md section 4A). Run ONCE (on Kaggle, where the raw dataset
and images are attached) to produce clean train/val/test CSVs with:

- normalized binary labels (see src/multimodal_labels.py)
- an added `combined_text` column: OCR-extracted meme text + the
  dataset's own caption, combined via src.ocr.combine_ocr_and_caption
- the one row with a missing image file dropped (found during EDA --
  val split, image_name 'KbTk7Rq.png' referenced in the CSV but not
  present in the image folder)

This is a data-PREPARATION step, not something that runs inside the
Dataset class at training time. OCR is CPU-bound and deterministic per
image, so running it once here and caching the result avoids re-running
it every epoch for zero benefit.

Run with (from repo root, wherever the raw dataset + images are mounted --
Kaggle, or locally if you've downloaded them):

    python -m src.prepare_multimodal_data \
        --split_dir "/kaggle/input/datasets/aditya1220/hate-speech-detection-dataset/Dataset/Split Dataset" \
        --image_dir "/kaggle/input/datasets/aditya1220/hate-speech-detection-dataset/Dataset/Labelled Images" \
        --out_dir data/multimodal
"""

import argparse
import os

import pandas as pd

from src.multimodal_labels import normalize_label
from src.ocr import combine_ocr_and_caption, extract_text_from_image

SPLIT_FILES = {
    "train": "Training_meme_dataset.csv",
    "val": "Validation_meme_dataset.csv",
    "test": "Testing_meme_dataset.csv",
}


def prepare_split(name: str, split_dir: str, image_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(split_dir, SPLIT_FILES[name]))

    # drop rows whose image file doesn't actually exist on disk -- found
    # during EDA: exactly 1 row in val references a missing file
    images_on_disk = set(os.listdir(image_dir))
    before = len(df)
    df = df[df["image_name"].isin(images_on_disk)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[{name}] dropped {dropped} row(s) with missing image file")

    df["label"] = df["label"].apply(normalize_label)

    ocr_texts = []
    for img_name in df["image_name"]:
        ocr_texts.append(extract_text_from_image(os.path.join(image_dir, img_name)))
    df["ocr_text"] = ocr_texts

    df["combined_text"] = [
        combine_ocr_and_caption(ocr, caption)
        for ocr, caption in zip(df["ocr_text"], df["sentence"])
    ]

    print(f"[{name}] n={len(df)}, label distribution: {df['label'].value_counts().to_dict()}")
    n_empty_ocr = (df["ocr_text"].str.strip() == "").sum()
    print(f"[{name}] {n_empty_ocr}/{len(df)} images produced no OCR text (falls back to caption alone)")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="data/multimodal")
    args, _unknown_args = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for name in ("train", "val", "test"):
        df = prepare_split(name, args.split_dir, args.image_dir)
        out_path = os.path.join(args.out_dir, f"{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[{name}] saved: {out_path}\n")


if __name__ == "__main__":
    main()
