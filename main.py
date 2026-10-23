from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_loader import DEFAULT_DATASET_PATH, load_dataset
from src.fairness_metrics import compute_race_metrics
from src.preprocessing import apply_propublica_filters
from src.report_generator import export_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ProPublica COMPAS fairness analysis.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to the COMPAS CSV file")
    parser.add_argument("--threshold", type=int, default=5, help="COMPAS threshold for high risk (default: 5)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for generated output files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    try:
        df = load_dataset(args.data)
        cleaned_df, summary = apply_propublica_filters(df, threshold=args.threshold, methodology="propublica_original")
        metrics_df = compute_race_metrics(cleaned_df, threshold=args.threshold)
        export_results = export_outputs(
            cleaned_df=cleaned_df,
            metrics_df=metrics_df,
            summary=summary,
            output_dir=output_dir,
            dataset_name=Path(args.data).name,
            threshold=args.threshold,
            methodology="propublica_original",
        )
        print("COMPAS analysis completed successfully.")
        print(f"Dataset: {Path(args.data).name}")
        print(f"Final sample size: {len(cleaned_df)}")
        print(metrics_df.to_string(index=False))
        print(f"Outputs saved to: {output_dir}")
        for output_name, output_path in export_results.items():
            print(f"- {output_name}: {output_path}")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Download the official COMPAS data from https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv and save it in the data folder.", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - last-resort safety
        print(f"Unhandled analysis error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
