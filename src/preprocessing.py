from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import validate_binary_column


def _as_string(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.strip()


def apply_propublica_filters(
    df: pd.DataFrame,
    *,
    threshold: int = 5,
    methodology: str = "propublica_original",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the original ProPublica-style filters and return the cleaned dataset plus a summary."""
    original_rows = len(df)
    working = df.copy()

    validate_binary_column(working, "two_year_recid")

    summary: dict[str, Any] = {
        "dataset_name": getattr(df, "attrs", {}).get("source_name", "unknown_dataset"),
        "methodology": methodology,
        "threshold": threshold,
        "original_row_count": original_rows,
        "removed_rows": 0,
        "remaining_row_count": original_rows,
        "filter_steps": [],
        "group_sample_sizes": {},
    }

    def add_step(step_name: str, mask: pd.Series) -> None:
        before_count = len(working)
        working_filtered = working.loc[mask].copy()
        removed = before_count - len(working_filtered)
        summary["filter_steps"].append({
            "step": step_name,
            "removed_rows": removed,
            "remaining_rows": len(working_filtered),
        })
        summary["removed_rows"] += removed
        return working_filtered

    working["decile_score"] = pd.to_numeric(working["decile_score"], errors="coerce")
    working["two_year_recid"] = pd.to_numeric(working["two_year_recid"], errors="coerce")
    working["is_recid"] = pd.to_numeric(working["is_recid"], errors="coerce")

    working = add_step(
        "days_b_screening_arrest within [-30, 30]",
        working["days_b_screening_arrest"].between(-30, 30, inclusive="both"),
    )
    working = add_step("is_recid != -1", working["is_recid"] != -1)
    working = add_step("c_charge_degree != 'O'", working["c_charge_degree"].astype(str).str.upper().str.strip() != "O")
    score_text = _as_string(working["score_text"])
    working = add_step("score_text not missing and not N/A", score_text.notna() & ~score_text.str.upper().eq("N/A"))
    working = add_step(
        "restrict to African-American and Caucasian defendants",
        working["race"].astype(str).str.strip().isin(["African-American", "Caucasian"]),
    )
    working = add_step("drop rows missing decile_score", working["decile_score"].notna())

    if working.empty:
        raise ValueError("Filtered dataset is empty after applying the required ProPublica methodology filters.")

    working["race"] = working["race"].astype(str).str.strip()
    working["predicted_high_risk"] = (working["decile_score"] >= threshold).astype(int)
    working["actual_recidivism"] = (working["two_year_recid"] == 1).astype(int)
    working["high_risk_label"] = working["predicted_high_risk"].map({0: "Low", 1: "High"})

    group_sample_sizes = working["race"].value_counts().sort_index().to_dict()
    summary["group_sample_sizes"] = group_sample_sizes
    summary["remaining_row_count"] = len(working)
    summary["removed_rows"] = original_rows - len(working)
    summary["filters_applied"] = [entry["step"] for entry in summary["filter_steps"]]
    summary["sample_size_by_race"] = group_sample_sizes

    return working, summary
