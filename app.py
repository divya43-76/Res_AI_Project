from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import DEFAULT_DATASET_PATH, load_dataset
from src.fairness_metrics import compute_race_metrics
from src.preprocessing import apply_propublica_filters
from src.report_generator import export_outputs

st.set_page_config(page_title="COMPAS Fairness Audit", page_icon="⚖️", layout="wide")


def show_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    st.dataframe(df.head(10), use_container_width=True)
    st.write("### Missing-value summary")
    st.dataframe(df.isna().sum().rename("missing_values").reset_index().rename(columns={"index": "column"}), use_container_width=True)
    st.write("### Race distribution")
    st.dataframe(df["race"].value_counts().rename_axis("race").reset_index(name="count"), use_container_width=True)
    st.write("### Age distribution")
    st.bar_chart(df["age"].value_counts().sort_index())
    st.write("### COMPAS score distribution")
    st.bar_chart(df["decile_score"].value_counts().sort_index())


def run_analysis(dataframe: pd.DataFrame, threshold: int) -> tuple[pd.DataFrame, dict, dict[str, Path]]:
    cleaned_df, summary = apply_propublica_filters(dataframe, threshold=threshold, methodology="propublica_original")
    metrics_df = compute_race_metrics(cleaned_df, threshold=threshold)
    output_dir = Path("outputs")
    report_files = export_outputs(
        cleaned_df=cleaned_df,
        metrics_df=metrics_df,
        summary=summary,
        output_dir=output_dir,
        dataset_name="uploaded_dataset.csv" if "uploaded" in str(dataframe.attrs.get("source_name", "")) else DEFAULT_DATASET_PATH.name,
        threshold=threshold,
        methodology="propublica_original",
    )
    return metrics_df, summary, report_files


st.title("COMPAS Fairness Audit")
st.caption(
    "Educational analysis of ProPublica's COMPAS recidivism dataset. This project evaluates documented risk-score error rates between African-American and Caucasian defendants."
)

st.markdown(
    """
    - Dataset source: ProPublica COMPAS recidivism dataset
    - Methodology: Original ProPublica filtering rules plus a configurable decile score threshold
    - Important: This is an audit of existing historical scores and recorded outcomes, not a model that can determine whether an individual should be imprisoned.
    """
)

local_path = DEFAULT_DATASET_PATH
if local_path.exists():
    st.success(f"Found local dataset at {local_path}.")
else:
    st.warning(
        f"Local dataset not found at {local_path}. Download the official CSV and place it in the project data folder or upload one here."
    )

uploaded_file = st.file_uploader("Upload a COMPAS CSV file", type=["csv"])
threshold = st.slider("Select the COMPAS high-risk threshold", min_value=1, max_value=10, value=5)

analysis_state = st.session_state.get("analysis_result")
if uploaded_file is not None:
    st.session_state["uploaded_dataset_name"] = uploaded_file.name
else:
    st.session_state.setdefault("uploaded_dataset_name", local_path.name)

if st.button("Run Analysis", type="primary"):
    try:
        if uploaded_file is not None:
            df = load_dataset(uploaded_file=uploaded_file)
            df.attrs["source_name"] = uploaded_file.name
        else:
            df = load_dataset(local_path)
            df.attrs["source_name"] = local_path.name
        metrics_df, summary, files = run_analysis(df, threshold)
        st.session_state.analysis_result = {"metrics": metrics_df, "summary": summary, "files": files, "dataset": df}
        st.success("Analysis succeeded.")
    except Exception as exc:
        st.session_state.analysis_result = None
        st.error(f"Analysis failed: {exc}")

if "analysis_result" in st.session_state and st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result
    metrics_df = result["metrics"]
    summary = result["summary"]
    files = result["files"]

    st.subheader("Fairness analysis")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Confusion-matrix counts")
    confusion_df = pd.read_csv(files["confusion_matrices"])
    st.dataframe(confusion_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(files["error_rates"]))
    with col2:
        st.image(str(files["risk_comparison"]))

    col3, col4 = st.columns(2)
    with col3:
        st.image(str(files["score_distribution"]))
    with col4:
        st.image(str(files["risk_category"]))

    st.subheader("Findings")
    black = metrics_df[metrics_df["race"] == "African-American"].iloc[0]
    white = metrics_df[metrics_df["race"] == "Caucasian"].iloc[0]
    fpr_diff = abs((black["fpr"] or 0.0) - (white["fpr"] or 0.0))
    fnr_diff = abs((black["fnr"] or 0.0) - (white["fnr"] or 0.0))
    st.write(f"The higher false-positive rate is in **{('African-American' if (black['fpr'] or 0.0) >= (white['fpr'] or 0.0) else 'Caucasian')}**.")
    st.write(f"The higher false-negative rate is in **{('African-American' if (black['fnr'] or 0.0) >= (white['fnr'] or 0.0) else 'Caucasian')}**.")
    st.write(f"Absolute FPR difference: {fpr_diff * 100:.2f} percentage points.")
    st.write(f"Absolute FNR difference: {fnr_diff * 100:.2f} percentage points.")
    st.write("These differences signal unequal error rates under the same threshold and methodology. Fairness metrics can conflict, and a similar value on one metric does not guarantee fairness across all metrics.")

    st.subheader("Downloads")
    for label, path in files.items():
        with open(path, "rb") as fh:
            st.download_button(label=label.replace("_", " ").title(), data=fh.read(), file_name=Path(path).name, mime="application/octet-stream")

    st.subheader("Dataset explorer")
    show_dataset_overview(result["dataset"])

    st.subheader("Preprocessing summary")
    summary_df = pd.DataFrame(summary["filter_steps"])
    st.dataframe(summary_df, use_container_width=True)
