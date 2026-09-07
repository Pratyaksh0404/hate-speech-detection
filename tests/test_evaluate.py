"""
Unit tests for src/evaluate.py. Pure pandas/scikit-learn — no data file
or network needed, runs anywhere.
"""

import json
import os
import time

from src.evaluate import compute_metrics, measure_latency_ms, save_results


def test_compute_metrics_known_values():
    # hand-checkable example: 5/7 correct predictions
    y_true = [0, 0, 1, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 2, 2]
    m = compute_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - (5 / 7)) < 1e-9
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert len(m["confusion_matrix"]) == 3
    assert "classification_report" in m


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_measure_latency_roughly_matches_sleep_time():
    def fake_predict(x):
        time.sleep(0.002)
        return x

    latency = measure_latency_ms(fake_predict, [1, 2, 3], n_repeats=20)
    # allow generous tolerance since sleep() timing isn't exact
    assert 1.0 < latency < 20.0


def test_measure_latency_requires_at_least_one_sample():
    try:
        measure_latency_ms(lambda x: x, [], n_repeats=5)
        assert False, "expected ValueError for empty samples"
    except ValueError:
        pass


def test_save_results_round_trips(tmp_path):
    out_dir = str(tmp_path)
    path = save_results({"foo": "bar", "macro_f1": 0.5}, "unit_test_model", out_dir=out_dir)
    assert os.path.exists(path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["model_name"] == "unit_test_model"
    assert loaded["foo"] == "bar"
    assert loaded["macro_f1"] == 0.5
