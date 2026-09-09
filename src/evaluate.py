"""
Shared evaluation utilities for every model in the text ladder (baseline
through DistilBERT), and later the multimodal fusion model. Writing this
once here means every rung of the ladder is scored identically, so the
final comparison table (master_plan.md section 4.6) is an honest,
apples-to-apples comparison rather than five different ad-hoc scripts
each computing metrics slightly differently.
"""

import json
import os
import time

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neither"]


def compute_metrics(y_true, y_pred, target_names: list = None) -> dict:
    """
    Macro-F1 is the headline metric (master_plan.md section 6 — accuracy
    alone is misleading on this imbalanced a dataset), but we compute and
    store everything so later analysis/README tables have full context.

    target_names defaults to the 3-class text-ladder label names
    (LABEL_NAMES) for backward compatibility with every existing caller.
    Pass e.g. ["Non-offensive", "Offensive"] for the binary multimodal
    task — using the wrong label count/names here would crash or silently
    mislabel the classification report.
    """
    target_names = target_names or LABEL_NAMES
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def measure_latency_ms(predict_one_fn, samples, n_repeats: int = 200) -> float:
    """
    Average SINGLE-SAMPLE inference latency in milliseconds.

    predict_one_fn(x) must predict on exactly one example, not a batch —
    this deliberately mimics how the real FastAPI /predict/text endpoint
    will be called in production (one request in, one prediction out), so
    the number in the comparison table means something for deployment,
    not just an artificially-fast batched throughput number.
    """
    if len(samples) == 0:
        raise ValueError("measure_latency_ms needs at least one sample")

    predict_one_fn(samples[0])  # warm-up call; some models pay a one-time lazy-init cost

    start = time.perf_counter()
    for i in range(n_repeats):
        predict_one_fn(samples[i % len(samples)])
    elapsed = time.perf_counter() - start

    return (elapsed / n_repeats) * 1000.0


def save_results(results: dict, model_name: str, out_dir: str = "results") -> str:
    """
    Saves one model's results as results/<model_name>.json. Small, human-
    readable, and meant to be committed to git (unlike model weights) —
    this is exactly what feeds the final README comparison table and
    notebooks/results_comparison.ipynb later.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}.json")
    payload = {"model_name": model_name, **results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
