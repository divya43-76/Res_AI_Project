# COMPAS Fairness Audit Report

- Dataset: **compas-scores-two-years.csv**
- Generated: **2026-10-24 01:13:25**
- Methodology: **propublica_original**
- COMPAS high-risk threshold: **2** (decile_score >= 2)
- Final analysis sample size: **5278**
- Group sample sizes: **{'African-American': 3175, 'Caucasian': 2103}**

## Methodology

This audit uses the ProPublica-style COMPAS filtering and thresholding rules. The risk assignment is defined as decile_score >= threshold, and the observed outcome is the recorded two-year recidivism indicator. This is a descriptive audit of existing COMPAS scores, not a model for making individual justice decisions.

## Descriptive metrics

| Race | Sample size | Predicted high risk | Observed recidivism | FPR | FNR | TPR | Accuracy | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| African-American | 3175 | 88.50% | 52.31% | 81.51% | 5.12% | 94.88% | 58.46% | 56.09% |
| Caucasian | 2103 | 71.23% | 39.09% | 62.76% | 15.57% | 84.43% | 55.68% | 46.33% |

## Key fairness findings

- Higher false-positive rate: **African-American**
- Higher false-negative rate: **Caucasian**
- Absolute percentage-point difference in FPR: **18.74 percentage points**
- Absolute percentage-point difference in FNR: **-10.45 percentage points**

These differences describe unequal error rates across groups under the same threshold. They do not prove a causal relationship between race and the score, and fairness metrics often conflict with one another. Equalizing one metric does not automatically equalize all others.

## Interpretation

- False-positive rate = FP / (FP + TN): the share of defendants who did not recidivate but were classified as high risk.
- False-negative rate = FN / (FN + TP): the share of defendants who did recidivate but were classified as low risk.
- Accuracy = (TP + TN) / total: the share of all cases correctly classified.
- Precision = TP / (TP + FP): among those predicted high risk, the share who actually recidivated.
- Calibration describes whether predicted risk matches observed outcomes; demographic parity concerns group-level prediction rates; equalized odds compares error rates across groups.

## Data quality and limitations

The implementation validates required columns, checks binary outcome values, and removes records using the original ProPublica-style filters before computing metrics. This is an educational analysis of historical records, not a validated decision-support system for criminal justice outcomes.
