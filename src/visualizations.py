from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_error_rate_plot(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    races = metrics_df["race"].tolist()
    fpr_values = metrics_df["fpr"].fillna(0) * 100
    fnr_values = metrics_df["fnr"].fillna(0) * 100
    x = range(len(races))
    ax.bar(x, fpr_values, width=0.35, label="False-positive rate (%)", color="#d62728")
    ax.bar([i + 0.35 for i in x], fnr_values, width=0.35, label="False-negative rate (%)", color="#1f77b4")
    ax.set_xticks([i + 0.175 for i in x])
    ax.set_xticklabels(races)
    ax.set_ylabel("Percent")
    ax.set_title("False-positive vs false-negative rates by racial group")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_risk_comparison_plot(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    plot_df = metrics_df[["race", "predicted_high_risk_rate", "observed_recidivism_rate"]].copy()
    plot_df["predicted_high_risk_rate"] = plot_df["predicted_high_risk_rate"].fillna(0) * 100
    plot_df["observed_recidivism_rate"] = plot_df["observed_recidivism_rate"].fillna(0) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(plot_df))
    ax.bar(x, plot_df["predicted_high_risk_rate"], width=0.35, label="Predicted high risk (%)", color="#ff7f0e")
    ax.bar([i + 0.35 for i in x], plot_df["observed_recidivism_rate"], width=0.35, label="Observed recidivism (%)", color="#2ca02c")
    ax.set_xticks([i + 0.175 for i in x])
    ax.set_xticklabels(plot_df["race"].tolist())
    ax.set_ylabel("Percent")
    ax.set_title("Predicted high-risk vs observed recidivism by racial group")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_score_distribution_plot(cleaned_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    for race, group in cleaned_df.groupby("race", sort=True):
        ax.hist(
            group["decile_score"].dropna(),
            bins=10,
            alpha=0.6,
            label=str(race),
            range=(1, 10),
        )
    ax.set_xlabel("COMPAS decile score")
    ax.set_ylabel("Count")
    ax.set_title("COMPAS score distribution by racial group")
    ax.legend(title="Race")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_risk_category_plot(cleaned_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    summary = (
        cleaned_df.assign(score_text=cleaned_df["score_text"].astype(str).str.strip())
        .groupby(["race", "score_text"], dropna=False)["two_year_recid"]
        .mean()
        .reset_index(name="observed_recidivism_rate")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for race, group in summary.groupby("race", sort=True):
        values = group.set_index("score_text")["observed_recidivism_rate"] * 100
        ax.plot(values.index, values.values, marker="o", linewidth=2, label=str(race))
    ax.set_xlabel("COMPAS risk category")
    ax.set_ylabel("Observed recidivism rate (%)")
    ax.set_title("Observed recidivism rate by COMPAS risk category")
    ax.legend(title="Race")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
