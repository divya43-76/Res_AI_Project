import pandas as pd
import pytest

from src.data_loader import load_dataset, validate_numeric_columns, validate_required_columns


def _make_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race": ["African-American", "Caucasian", "African-American"],
            "age": [25, 40, 30],
            "sex": ["Male", "Female", "Male"],
            "decile_score": [3, 6, 9],
            "score_text": ["Low", "Medium", "High"],
            "two_year_recid": [1, 0, 1],
            "days_b_screening_arrest": [-5, 25, 10],
            "is_recid": [1, 0, 1],
            "c_charge_degree": ["F", "M", "F"],
            "priors_count": [0, 2, 1],
        }
    )


def test_valid_dataset_loading(tmp_path):
    path = tmp_path / "valid.csv"
    _make_valid_dataframe().to_csv(path, index=False)
    df = load_dataset(path)
    assert list(df.columns)[:10] == [
        "race",
        "age",
        "sex",
        "decile_score",
        "score_text",
        "two_year_recid",
        "days_b_screening_arrest",
        "is_recid",
        "c_charge_degree",
        "priors_count",
    ]
    assert len(df) == 3


def test_missing_file_handling():
    with pytest.raises(FileNotFoundError):
        load_dataset("d:/does/not/exist.csv")


def test_missing_required_columns():
    df = _make_valid_dataframe().drop(columns=["decile_score"])
    with pytest.raises(ValueError):
        validate_required_columns(df)


def test_invalid_numeric_values():
    df = _make_valid_dataframe().copy()
    df.loc[0, "age"] = "not-a-number"
    issues = validate_numeric_columns(df)
    assert "age" in issues
