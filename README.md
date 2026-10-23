# COMPAS Fairness Analysis

This project audits the historical ProPublica COMPAS recidivism dataset to compare the risk classifications and prediction errors between African-American and Caucasian defendants. It is an educational fairness-auditing tool that documents how the original COMPAS scores compare with observed two-year recidivism outcomes under a transparent high-risk threshold.

## Dataset source

Official dataset:
- https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv
- https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv

Expected local file path:
- D:/COMPAS-Fairness-Analysis/data/compas-scores-two-years.csv

The CSV must be downloaded manually and placed inside the project's data folder. The application will also accept an uploaded CSV through the dashboard if the local file is missing.

## Project structure

COMPAS-Fairness-Analysis/
- app.py
- main.py
- requirements.txt
- README.md
- .gitignore
- data/
  - .gitkeep
  - compas-scores-two-years.csv
- outputs/
  - .gitkeep
  - fairness_metrics.csv
  - confusion_matrices.csv
  - preprocessing_summary.csv
  - cleaned_compas_data.csv
  - error_rates.png
  - risk_comparison.png
  - score_distribution.png
  - risk_category.png
  - analysis_report.md
- src/
  - __init__.py
  - data_loader.py
  - preprocessing.py
  - fairness_metrics.py
  - visualizations.py
  - report_generator.py
- tests/
  - test_data_loader.py
  - test_preprocessing.py
  - test_fairness_metrics.py

## Python installation (Windows Command Prompt)

1. Open Command Prompt.
2. Change to the project directory:
   ```cmd
   cd /d D:\COMPAS-Fairness-Analysis
   ```
3. Create a virtual environment (optional but recommended):
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
4. Install dependencies:
   ```cmd
   python -m pip install -r requirements.txt
   ```

## Command-line execution

Run the CLI analysis using the default local dataset:
```cmd
cd /d D:\COMPAS-Fairness-Analysis
python main.py
```

Run with a custom dataset path:
```cmd
python main.py --data "data/compas-scores-two-years.csv"
```

Run with a different threshold (default = 5):
```cmd
python main.py --threshold 6
```

## Streamlit dashboard

Launch the interactive dashboard:
```cmd
cd /d D:\COMPAS-Fairness-Analysis
streamlit run app.py
```

The dashboard contains:
- Overview and local dataset status
- Dataset explorer with preview, missing values, and distributions
- Fairness metrics comparison by race
- Confusion matrices and error-rate charts
- Download buttons for generated CSV/PNG/Markdown files
- Upload workflow for alternative CSV files

## Methodology

This project follows the original ProPublica-style filtering approach:
- days_b_screening_arrest within [-30, 30]
- is_recid != -1
- c_charge_degree != "O"
- score_text not missing and not "N/A"
- restrict analysis to African-American and Caucasian defendants
- pair the observed outcome with the COMPAS decile score under a configurable threshold

The prediction logic is intentionally transparent:
- decile_score >= 5 => predicted high risk
- decile_score < 5 => predicted low risk

The threshold is configurable in the dashboard and CLI, but 5 is the default choice used for the comparison.

## Metrics and formulas

This project calculates:
- True positive (TP)
- False positive (FP)
- True negative (TN)
- False negative (FN)
- False-positive rate = FP / (FP + TN)
- False-negative rate = FN / (FN + TP)
- True-positive rate = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- Accuracy = (TP + TN) / total
- Precision = TP / (TP + FP)
- Predicted high-risk rate = predicted high risk / total
- Observed recidivism rate = recidivism cases / total

## Output files

After a successful analysis, the project saves these files in the outputs folder:
- fairness_metrics.csv
- confusion_matrices.csv
- preprocessing_summary.csv
- cleaned_compas_data.csv
- error_rates.png
- risk_comparison.png
- score_distribution.png
- risk_category.png
- analysis_report.md

## Troubleshooting

If the local dataset is missing:
1. Download the official CSV from the ProPublica repository.
2. Save it to: D:\COMPAS-Fairness-Analysis\data\compas-scores-two-years.csv
3. Re-run the analysis.

If you upload a file through the Streamlit interface, the app processes it directly without requiring manual movement into the data folder.

If a CSV is malformed or missing required columns, the application raises clear validation errors instead of silently using incorrect data.

## Limitations and interpretation

- This is a historical audit of existing COMPAS scores and recorded outcomes, not a decision tool for individual detention or release recommendations.
- Fairness metrics can conflict: improving one metric does not automatically improve all others.
- Calibration, demographic parity, and equalized odds are different fairness concepts and should not be confused.
- Results are specific to the selected methodology, threshold, and filtered sample. They should be interpreted as educational analysis rather than a claim that scores or policy are definitively fair or unfair.
