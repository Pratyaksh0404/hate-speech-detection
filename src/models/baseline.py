import os
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import compute_metrics, measure_latency_ms, save_results

RANDOM_STATE = 42


def load_splits(data_dir: str = "data/text"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    val = pd.read_csv(f"{data_dir}/val.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    # clean_text can end up empty (not NaN) for e.g. a URL-only tweet per
    # clean_text()'s own design, but fillna guards against any stray NaN
    # from CSV round-tripping too — TfidfVectorizer would otherwise crash.
    for df in (train, val, test):
        df["clean_text"] = df["clean_text"].fillna("")
    return train, val, test


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", classifier),
        ]
    )


def train_and_evaluate(name: str, classifier, train, val, test) -> dict:
    pipe = build_pipeline(classifier)

    t0 = time.perf_counter()
    pipe.fit(train["clean_text"], train["label"])
    train_time_s = time.perf_counter() - t0

    val_preds = pipe.predict(val["clean_text"])
    test_preds = pipe.predict(test["clean_text"])

    val_metrics = compute_metrics(val["label"], val_preds)
    test_metrics = compute_metrics(test["label"], test_preds)

    # Single-sample latency, mimicking how the real /predict/text endpoint
    # will actually be called (one request in, one prediction out).
    test_texts = test["clean_text"].tolist()
    latency_ms = measure_latency_ms(
        lambda x: pipe.predict([x]), test_texts, n_repeats=200
    )

    results = {
        "train_time_seconds": train_time_s,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latency_ms_per_sample": latency_ms,
    }

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{name}.joblib"
    joblib.dump(pipe, model_path)

    results_path = save_results(results, name)

    print(f"\n=== {name} ===")
    print(f"train time: {train_time_s:.2f}s")
    print(f"val macro-F1: {val_metrics['macro_f1']:.4f}")
    print(
        f"test macro-F1: {test_metrics['macro_f1']:.4f}  |  "
        f"test accuracy: {test_metrics['accuracy']:.4f}"
    )
    print(f"latency: {latency_ms:.3f} ms/sample")
    print(f"model saved: {model_path}")
    print(f"results saved: {results_path}")

    return results


def main():
    train, val, test = load_splits()

    train_and_evaluate(
        "baseline_logreg",
        LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        train,
        val,
        test,
    )
    train_and_evaluate(
        "baseline_linearsvm",
        LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        train,
        val,
        test,
    )


if __name__ == "__main__":
    main()
