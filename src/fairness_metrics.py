from __future__ import annotations

import pandas as pd


def safe_divide(numerator: float | int, denominator: float | int) -> float | None:
    if denominator in (None, 0, 0.0):
        return None
    return numerator / denominator


def compute_confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    true_series = pd.Series(y_true).astype(int)
    pred_series = pd.Series(y_pred).astype(int)
    if true_series.empty or pred_series.empty:
        raise ValueError("Confusion-matrix inputs must not be empty.")
    if not (true_series.isin([0, 1]).all() and pred_series.isin([0, 1]).all()):
        raise ValueError("Confusion-matrix inputs must be binary 0/1 values.")

    tp = int(((true_series == 1) & (pred_series == 1)).sum())
    fp = int(((true_series == 0) & (pred_series == 1)).sum())
    tn = int(((true_series == 0) & (pred_series == 0)).sum())
    fn = int(((true_series == 1) & (pred_series == 0)).sum())
    return {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn}


def compute_group_metrics(group_df: pd.DataFrame, threshold: int = 5) -> dict[str, float | int | None]:
    if group_df.empty:
        raise ValueError("Cannot compute metrics for an empty group.")

    if "decile_score" not in group_df.columns or "two_year_recid" not in group_df.columns:
        raise ValueError("Group data must include 'decile_score' and 'two_year_recid'.")

    data = group_df[["decile_score", "two_year_recid"]].copy()
    data["decile_score"] = pd.to_numeric(data["decile_score"], errors="coerce")
    data["two_year_recid"] = pd.to_numeric(data["two_year_recid"], errors="coerce")
    valid_mask = data["decile_score"].notna() & data["two_year_recid"].isin([0, 1])
    filtered = data.loc[valid_mask].copy()
    if filtered.empty:
        raise ValueError("No valid binary outcome values remain for this group.")

    y_true = filtered["two_year_recid"].astype(int)
    y_pred = (filtered["decile_score"] >= threshold).astype(int)
    counts = compute_confusion_counts(y_true, y_pred)

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]

    fpr = safe_divide(fp, fp + tn)
    fnr = safe_divide(fn, fn + tp)
    tpr = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    predicted_high_risk_rate = safe_divide((y_pred == 1).sum(), len(filtered))
    observed_recidivism_rate = safe_divide((y_true == 1).sum(), len(filtered))

    return {
        "group_size": int(len(filtered)),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "specificity": specificity,
        "accuracy": accuracy,
        "precision": precision,
        "predicted_high_risk_rate": predicted_high_risk_rate,
        "observed_recidivism_rate": observed_recidivism_rate,
    }


def compute_race_metrics(cleaned_df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    if cleaned_df.empty:
        raise ValueError("Cannot compute fairness metrics for an empty cleaned dataset.")
    if "race" not in cleaned_df.columns:
        raise ValueError("The cleaned dataset must include a 'race' column.")

    rows = []
    for race, group in cleaned_df.groupby("race", sort=True):
        metrics = compute_group_metrics(group, threshold=threshold)
        rows.append({"race": race, **metrics})
    return pd.DataFrame(rows)


def compute_confusion_matrix_table(cleaned_df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    if cleaned_df.empty:
        raise ValueError("Cannot compute confusion matrices for an empty cleaned dataset.")

    rows = []
    for race, group in cleaned_df.groupby("race", sort=True):
        counts = compute_group_metrics(group, threshold=threshold)
        rows.append({
            "race": race,
            "true_positive": counts["true_positive"],
            "false_positive": counts["false_positive"],
            "true_negative": counts["true_negative"],
            "false_negative": counts["false_negative"],
            "group_size": counts["group_size"],
        })
    return pd.DataFrame(rows)
