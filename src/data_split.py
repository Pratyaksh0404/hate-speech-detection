import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def make_splits(
    cleaned_csv_path: str = "data/text/cleaned_labeled_data.csv",
    out_dir: str = "data/text",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "fractions must sum to 1"

    df = pd.read_csv(cleaned_csv_path)

    # Step 1: carve off the test set first
    train_val, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    # Step 2: split what's left into train/val (val_frac is relative to
    # the remaining train_val pool, not the original total)
    val_relative_frac = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_relative_frac,
        stratify=train_val["label"],
        random_state=RANDOM_STATE,
    )

    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv", index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)

    print(f"total rows: {len(df)}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = (split["label_name"].value_counts(normalize=True) * 100).round(2)
        print(f"\n{name}: n={len(split)}")
        print(dist.to_dict())

    return train, val, test


if __name__ == "__main__":
    make_splits()
