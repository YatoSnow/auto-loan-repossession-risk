# Model evaluation report

Written by `repo_risk evaluate`. All numbers come from the run that produced this file.
The data is synthetic. These numbers do not describe real loans.

## 1. Complete-window test (2023-01 to 2023-12)

79,713 landmarks from 23,578 loans. Every landmark has its full window observed, so each one has a known yes/no outcome.

What happened within 12 months: repossessed 4.30%, paid off 14.23%, charged off 1.49%, still open 79.98%.

### Repossession within 3 months

| Model | ROC-AUC | PR-AUC | Brier | ECE | Mean predicted | Observed |
|---|---:|---:|---:|---:|---:|---:|
| Markov chain | 0.984 | 0.334 | 0.0069 | 0.0011 | 0.89% | 0.96% |
| Logistic hazard | 0.986 | 0.472 | 0.0066 | 0.0043 | 0.86% | 0.96% |
| XGBoost hazard | 0.988 | 0.534 | 0.0060 | 0.0014 | 0.95% | 0.96% |

### Repossession within 6 months

| Model | ROC-AUC | PR-AUC | Brier | ECE | Mean predicted | Observed |
|---|---:|---:|---:|---:|---:|---:|
| Markov chain | 0.938 | 0.351 | 0.0145 | 0.0025 | 1.83% | 2.03% |
| Logistic hazard | 0.948 | 0.441 | 0.0146 | 0.0079 | 1.74% | 2.03% |
| XGBoost hazard | 0.950 | 0.468 | 0.0138 | 0.0025 | 1.87% | 2.03% |

### Repossession within 12 months

| Model | ROC-AUC | PR-AUC | Brier | ECE | Mean predicted | Observed |
|---|---:|---:|---:|---:|---:|---:|
| Markov chain | 0.770 (0.761-0.778) | 0.244 (0.227-0.260) | 0.0348 | 0.0059 | 3.71% | 4.30% |
| Logistic hazard | 0.824 (0.815-0.833) | 0.324 (0.303-0.343) | 0.0344 | 0.0106 | 3.24% | 4.30% |
| XGBoost hazard | 0.824 (0.816-0.833) | 0.330 (0.311-0.350) | 0.0340 | 0.0078 | 3.51% | 4.30% |
| XGBoost direct | 0.823 (0.814-0.831) | 0.325 (0.307-0.345) | 0.0342 | 0.0083 | 3.47% | 4.30% |

### Loans never used for fitting (12 months)

| Model | n | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| Markov chain | 12,116 | 0.774 | 0.250 | 0.0336 |
| Logistic hazard | 12,116 | 0.821 | 0.322 | 0.0332 |
| XGBoost hazard | 12,116 | 0.824 | 0.336 | 0.0328 |
| XGBoost direct | 12,116 | 0.825 | 0.326 | 0.0330 |

### Predicted vs observed by origination year (12 months)

| Group | n | Observed | Markov chain | Logistic hazard | XGBoost hazard | XGBoost direct |
|---|---:|---:|---:|---:|---:|---:|
| 2019 | 10,753 | 3.64% | 3.72% | 2.63% | 3.58% | 3.22% |
| 2020 | 14,867 | 3.96% | 3.71% | 2.95% | 3.33% | 3.27% |
| 2021 | 19,130 | 4.55% | 3.94% | 3.37% | 3.57% | 3.58% |
| 2022 | 23,049 | 4.89% | 3.95% | 3.77% | 3.79% | 3.82% |
| 2023 | 11,914 | 3.76% | 2.84% | 2.90% | 3.07% | 3.07% |

### Predicted vs observed by servicing group (12 months)

| Group | n | Observed | Markov chain | Logistic hazard | XGBoost hazard | XGBoost direct |
|---|---:|---:|---:|---:|---:|---:|
| A | 39,799 | 4.58% | 3.61% | 3.61% | 3.76% | 3.66% |
| B | 24,279 | 4.54% | 3.79% | 3.10% | 3.44% | 3.47% |
| C | 15,635 | 3.21% | 3.84% | 2.50% | 3.01% | 2.97% |

### Predicted vs observed by status at the landmark (12 months)

| Group | n | Observed | Markov chain | Logistic hazard | XGBoost hazard | XGBoost direct |
|---|---:|---:|---:|---:|---:|---:|
| current | 67,018 | 1.86% | 1.34% | 1.26% | 1.37% | 1.40% |
| dpd_1_29 | 7,196 | 7.57% | 6.78% | 3.35% | 5.67% | 5.74% |
| dpd_30_59 | 2,886 | 17.84% | 17.68% | 14.13% | 14.39% | 12.79% |
| dpd_60_89 | 1,362 | 34.73% | 33.38% | 29.76% | 31.82% | 34.48% |
| dpd_90_plus | 1,251 | 51.64% | 48.54% | 54.52% | 50.31% | 45.76% |

## 2. Censored check on the most recent landmarks (9 months)

77,038 landmarks, 73% of them with a window that runs past the end of the data.

| Estimate of the repossession rate | Value |
|---|---:|
| Aalen-Johansen (handles censoring and competing exits) | 3.42% |
| Naive: count censored landmarks as 'not repossessed' | 2.13% |
| Complete windows only (n = 21,061) | 3.46% |
| Mean prediction, Markov chain | 3.07% |
| Mean prediction, Logistic hazard | 2.75% |
| Mean prediction, XGBoost hazard | 3.06% |

By tercile of predicted risk (XGBoost hazard):

| Tercile | n | Aalen-Johansen | Logistic hazard | XGBoost hazard |
|---|---:|---:|---:|---:|
| low | 25,679 | 0.36% | 0.56% | 0.39% |
| middle | 25,679 | 0.76% | 0.89% | 0.73% |
| high | 25,680 | 9.22% | 6.80% | 8.05% |
