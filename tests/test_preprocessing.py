import pandas as pd
import pytest

from src.preprocessing import apply_propublica_filters


def _make_clean_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race": ["African-American", "African-American", "Caucasian", "Caucasian"],
            "age": [25, 30, 40, 55],
            "sex": ["Male", "Female", "Male", "Male"],
            "decile_score": [3, 8, 6, 1],
            "score_text": ["Low", "High", "Medium", "Low"],
            "two_year_recid": [1, 0, 1, 0],
            "days_b_screening_arrest": [-5, 10, 2, 0],
            "is_recid": [1, 0, 1, 0],
            "c_charge_degree": ["F", "F", "F", "M"],
            "priors_count": [0, 1, 2, 0],
        }
    )


def test_filtering_rules():
    df = _make_clean_dataset()
    filtered, summary = apply_propublica_filters(df, threshold=5)
    assert len(filtered) == 4
    assert set(filtered["race"]) == {"African-American", "Caucasian"}
    assert summary["remaining_row_count"] == 4
    assert summary["group_sample_sizes"]["African-American"] == 2
    assert summary["group_sample_sizes"]["Caucasian"] == 2


def test_binary_outcome_validation():
    df = _make_clean_dataset().copy()
    df.loc[0, "two_year_recid"] = 2
    with pytest.raises(ValueError):
        apply_propublica_filters(df, threshold=5)


def test_empty_filtered_datasets():
    df = _make_clean_dataset().copy()
    df["race"] = "Other"
    with pytest.raises(ValueError):
        apply_propublica_filters(df, threshold=5)


def test_group_sample_size_and_threshold_flags():
    df = _make_clean_dataset().copy()
    cleaned, summary = apply_propublica_filters(df, threshold=5)
    assert cleaned["predicted_high_risk"].isin([0, 1]).all()
    assert summary["group_sample_sizes"]["African-American"] == 2
