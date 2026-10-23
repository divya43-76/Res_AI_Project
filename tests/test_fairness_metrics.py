import pandas as pd
import pytest

from src.fairness_metrics import compute_confusion_counts, compute_group_metrics, safe_divide


def test_confusion_matrix_calculations():
    truth = pd.Series([1, 1, 0, 0])
    pred = pd.Series([1, 0, 1, 0])
    counts = compute_confusion_counts(truth, pred)
    assert counts == {"true_positive": 1, "false_positive": 1, "true_negative": 1, "false_negative": 1}


def test_false_positive_and_false_negative_rates():
    df = pd.DataFrame(
        {
            "decile_score": [6, 7, 1, 2],
            "two_year_recid": [1, 0, 0, 1],
        }
    )
    metrics = compute_group_metrics(df, threshold=5)
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["fnr"] == pytest.approx(0.5)


def test_zero_denominator_cases():
    df = pd.DataFrame({"decile_score": [1, 1, 1, 1], "two_year_recid": [0, 0, 0, 0]})
    metrics = compute_group_metrics(df, threshold=5)
    assert metrics["fpr"] == pytest.approx(0.0)
    assert metrics["tpr"] is None
    assert metrics["precision"] is None


def test_safe_divide_handles_zero_denominator():
    assert safe_divide(5, 0) is None
    assert safe_divide(3, 6) == pytest.approx(0.5)
