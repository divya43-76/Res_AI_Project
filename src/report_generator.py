from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .fairness_metrics import compute_confusion_matrix_table, compute_race_metrics
from .visualizations import (
    save_error_rate_plot,
    save_risk_category_plot,
    save_risk_comparison_plot,
    save_score_distribution_plot,
)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _percent_change(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f} percentage points"


def export_outputs(
    cleaned_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary: dict,
    output_dir: str | Path,
    *,
    dataset_name: str,
    threshold: int,
    methodology: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fairness_path = output_dir / "fairness_metrics.csv"
    confusion_path = output_dir / "confusion_matrices.csv"
    summary_path = output_dir / "preprocessing_summary.csv"
    cleaned_path = output_dir / "cleaned_compas_data.csv"
    error_plot_path = output_dir / "error_rates.png"
    risk_plot_path = output_dir / "risk_comparison.png"
    score_plot_path = output_dir / "score_distribution.png"
    category_plot_path = output_dir / "risk_category.png"
    report_path = output_dir / "analysis_report.md"

    metrics_df.to_csv(fairness_path, index=False)
    confusion_df = compute_confusion_matrix_table(cleaned_df, threshold=threshold)
    confusion_df.to_csv(confusion_path, index=False)

    summary_rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                summary_rows.append({"summary_key": key, "summary_subkey": nested_key, "value": nested_value})
        else:
            summary_rows.append({"summary_key": key, "summary_subkey": "", "value": value})
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    cleaned_df.to_csv(cleaned_path, index=False)

    save_error_rate_plot(metrics_df, error_plot_path)
    save_risk_comparison_plot(metrics_df, risk_plot_path)
    save_score_distribution_plot(cleaned_df, score_plot_path)
    save_risk_category_plot(cleaned_df, category_plot_path)

    report = build_analysis_report(
        cleaned_df=cleaned_df,
        metrics_df=metrics_df,
        summary=summary,
        dataset_name=dataset_name,
        threshold=threshold,
        methodology=methodology,
        report_path=report_path,
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "fairness_metrics": fairness_path,
        "confusion_matrices": confusion_path,
        "preprocessing_summary": summary_path,
        "cleaned_compas_data": cleaned_path,
        "error_rates": error_plot_path,
        "risk_comparison": risk_plot_path,
        "score_distribution": score_plot_path,
        "risk_category": category_plot_path,
        "analysis_report": report_path,
    }


def build_analysis_report(
    cleaned_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary: dict,
    *,
    dataset_name: str,
    threshold: int,
    methodology: str,
    report_path: str | Path,
) -> str:
    if metrics_df.empty:
        raise ValueError("Metrics table must not be empty before writing the analysis report.")

    metrics_by_race = metrics_df.set_index("race")
    black = metrics_by_race.loc["African-American"] if "African-American" in metrics_by_race.index else None
    white = metrics_by_race.loc["Caucasian"] if "Caucasian" in metrics_by_race.index else None

    if black is not None and white is not None:
        fpr_diff = (black["fpr"] if black["fpr"] is not None else 0.0) - (white["fpr"] if white["fpr"] is not None else 0.0)
        fnr_diff = (black["fnr"] if black["fnr"] is not None else 0.0) - (white["fnr"] if white["fnr"] is not None else 0.0)
        higher_fpr_group = "African-American" if (black["fpr"] or 0.0) >= (white["fpr"] or 0.0) else "Caucasian"
        higher_fnr_group = "African-American" if (black["fnr"] or 0.0) >= (white["fnr"] or 0.0) else "Caucasian"
    else:
        fpr_diff = None
        fnr_diff = None
        higher_fpr_group = "N/A"
        higher_fnr_group = "N/A"

    lines = [
        "# COMPAS Fairness Audit Report",
        "",
        f"- Dataset: **{dataset_name}**",
        f"- Generated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**",
        f"- Methodology: **{methodology}**",
        f"- COMPAS high-risk threshold: **{threshold}** (decile_score >= {threshold})",
        f"- Final analysis sample size: **{len(cleaned_df)}**",
        f"- Group sample sizes: **{summary.get('group_sample_sizes', {})}**",
        "",
        "## Methodology",
        "",
        "This audit uses the ProPublica-style COMPAS filtering and thresholding rules. The risk assignment is defined as decile_score >= threshold, and the observed outcome is the recorded two-year recidivism indicator. This is a descriptive audit of existing COMPAS scores, not a model for making individual justice decisions.",
        "",
        "## Descriptive metrics",
        "",
        "| Race | Sample size | Predicted high risk | Observed recidivism | FPR | FNR | TPR | Accuracy | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['race']} | {row['group_size']} | {_format_pct(row['predicted_high_risk_rate'])} | {_format_pct(row['observed_recidivism_rate'])} | {_format_pct(row['fpr'])} | {_format_pct(row['fnr'])} | {_format_pct(row['tpr'])} | {_format_pct(row['accuracy'])} | {_format_pct(row['precision'])} |"
        )

    lines.extend([
        "",
        "## Key fairness findings",
        "",
        f"- Higher false-positive rate: **{higher_fpr_group}**",
        f"- Higher false-negative rate: **{higher_fnr_group}**",
        f"- Absolute percentage-point difference in FPR: **{_percent_change(fpr_diff)}**",
        f"- Absolute percentage-point difference in FNR: **{_percent_change(fnr_diff)}**",
        "",
        "These differences describe unequal error rates across groups under the same threshold. They do not prove a causal relationship between race and the score, and fairness metrics often conflict with one another. Equalizing one metric does not automatically equalize all others.",
        "",
        "## Interpretation",
        "",
        "- False-positive rate = FP / (FP + TN): the share of defendants who did not recidivate but were classified as high risk.",
        "- False-negative rate = FN / (FN + TP): the share of defendants who did recidivate but were classified as low risk.",
        "- Accuracy = (TP + TN) / total: the share of all cases correctly classified.",
        "- Precision = TP / (TP + FP): among those predicted high risk, the share who actually recidivated.",
        "- Calibration describes whether predicted risk matches observed outcomes; demographic parity concerns group-level prediction rates; equalized odds compares error rates across groups.",
        "",
        "## Data quality and limitations",
        "",
        "The implementation validates required columns, checks binary outcome values, and removes records using the original ProPublica-style filters before computing metrics. This is an educational analysis of historical records, not a validated decision-support system for criminal justice outcomes.",
    ])

    return "\n".join(lines) + "\n"
