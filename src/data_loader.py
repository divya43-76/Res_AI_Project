from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "compas-scores-two-years.csv"

REQUIRED_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "age",
    "decile_score",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "days_b_screening_arrest",
    "is_recid",
    "two_year_recid",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required dataset columns: {missing_text}")


def validate_binary_column(df: pd.DataFrame, column_name: str, *, allowed_values: Iterable[int] = (0, 1)) -> None:
    if column_name not in df.columns:
        return
    series = pd.to_numeric(df[column_name], errors="coerce")
    invalid = series[~series.isin(list(allowed_values))]
    if not invalid.empty:
        invalid_values = sorted({int(value) for value in invalid.dropna().astype(int).unique() if pd.notna(value)})
        allowed_text = ", ".join(str(value) for value in allowed_values)
        raise ValueError(
            f"Column '{column_name}' contains invalid values for a binary outcome: {invalid_values}. "
            f"Expected only {allowed_text}."
        )


def validate_numeric_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    for column_name in NUMERIC_COLUMNS:
        if column_name not in df.columns:
            continue
        numeric_series = pd.to_numeric(df[column_name], errors="coerce")
        invalid_mask = numeric_series.isna() & df[column_name].notna()
        invalid_values = df.loc[invalid_mask, column_name].astype(str).unique().tolist()
        if invalid_values:
            issues[column_name] = invalid_values[:10]
        if column_name == "decile_score":
            out_of_range = numeric_series[(numeric_series < 1) | (numeric_series > 10)]
            if not out_of_range.empty:
                issues.setdefault(column_name, []).extend(
                    [f"out-of-range values detected: {out_of_range.dropna().astype(int).unique().tolist()}" ]
                )
    return issues


def load_dataset(data_path: str | Path | None = None, uploaded_file: Any | None = None) -> pd.DataFrame:
    """Load a CSV dataset from the default path or an uploaded file object."""
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        except Exception as exc:  # pragma: no cover - exercised by tests via ValueError for malformed CSV
            raise ValueError(f"Failed to read uploaded CSV file: {exc}") from exc
        dataset_name = getattr(uploaded_file, "name", "uploaded_file.csv")
    else:
        resolved_path = Path(data_path) if data_path is not None else DEFAULT_DATASET_PATH
        if not resolved_path.exists():
            raise FileNotFoundError(
                "Missing dataset file: "
                f"{resolved_path}. Download the official CSV from the ProPublica repository and place it in the "
                "project's data folder."
            )
        try:
            df = pd.read_csv(resolved_path)
        except Exception as exc:  # pragma: no cover - exercised by tests via ValueError for malformed CSV
            raise ValueError(f"Failed to read dataset at {resolved_path}: {exc}") from exc
        dataset_name = resolved_path.name

    if df.empty:
        raise ValueError(f"Dataset '{dataset_name}' is empty.")

    validate_required_columns(df)
    numeric_issues = validate_numeric_columns(df)
    if numeric_issues:
        details = "; ".join(f"{column}: {values}" for column, values in numeric_issues.items())
        raise ValueError(f"Dataset contains invalid numerical values: {details}")

    validate_binary_column(df, "two_year_recid")
    return df
