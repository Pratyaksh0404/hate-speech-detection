"""
Merges the counterfactual augmentation examples (src/counterfactual_augmentation.py)
into the original training set ONLY -- val.csv and test.csv stay
completely untouched, so evaluation remains on pristine, uncontaminated
data. Augmentation leaking into val/test would make any "it got better"
claim meaningless.

Produces data/text/train_augmented.csv and a recomputed
data/text/class_weights_augmented.json (augmentation shifts the class
balance, so the original class_weights.json no longer applies).

Oversamples the severity-augmentation examples 3x specifically -- held-out
evaluation showed the first version (16 unique examples, no oversampling)
had no measurable effect on severity-monotonicity behavior, most likely
because 16-24 examples out of 17,453+ training rows (roughly 0.1-0.14%) is
too diluted a signal to move the model. The identity-term augmentation
worked without oversampling (84 examples across 12 terms was apparently
enough), so only the severity group is oversampled here, not the whole
augmentation set -- a targeted response to what the held-out evidence
actually showed, not a blanket "add more of everything."

Run with (from repo root, wherever data/text/train.csv exists):
    python -m src.prepare_augmented_training_data --oversample_factor 3 --suffix ""
    python -m src.prepare_augmented_training_data --oversample_factor 2 --suffix "_2x"
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.counterfactual_augmentation import (
    HATE_SPEECH_AUGMENTATION,
    SEVERITY_AUGMENTATION,
    build_identity_augmentation,
)

RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oversample_factor", type=int, default=3)
    parser.add_argument("--suffix", default="", help="appended to output filenames, e.g. '_2x'")
    args, _unknown_args = parser.parse_known_args()

    train = pd.read_csv("data/text/train.csv")
    train["text"] = train["text"].fillna("")

    print(f"original train size: {len(train)}")
    print("original label distribution:", train["label"].value_counts().to_dict())

    severity_rows = list(SEVERITY_AUGMENTATION) * args.oversample_factor
    hate_rows = list(HATE_SPEECH_AUGMENTATION)
    identity_rows = build_identity_augmentation()
    aug_rows = severity_rows + hate_rows + identity_rows

    aug_df = pd.DataFrame(aug_rows, columns=["text", "label"])
    print(f"\naugmentation rows added: {len(aug_df)} "
          f"(severity x{args.oversample_factor}: {len(severity_rows)}, "
          f"hate: {len(hate_rows)}, identity: {len(identity_rows)})")
    print("augmentation label distribution:", aug_df["label"].value_counts().to_dict())

    combined = pd.concat([train[["text", "label"]], aug_df], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle

    print(f"\ncombined train size: {len(combined)}")
    print("combined label distribution:", combined["label"].value_counts().to_dict())

    os.makedirs("data/text", exist_ok=True)
    train_path = f"data/text/train_augmented{args.suffix}.csv"
    weights_path = f"data/text/class_weights_augmented{args.suffix}.json"

    combined.to_csv(train_path, index=False)
    print(f"\nsaved: {train_path}")

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1, 2]), y=combined["label"]
    )
    weight_dict = {int(c): float(w) for c, w in zip([0, 1, 2], class_weights_arr)}
    with open(weights_path, "w") as f:
        json.dump(weight_dict, f, indent=2)
    print(f"saved: {weights_path}")
    print("weights:", weight_dict)


if __name__ == "__main__":
    main()
