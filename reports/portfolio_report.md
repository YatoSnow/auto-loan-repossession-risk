# Portfolio analytics report

Written by `repo_risk report`. All numbers come from the run that produced this file.
The portfolio is synthetic. Nothing here describes a real lender.

## 1. Monthly transition matrix

Estimated from 915,951 month-to-month moves. Rows are the state this month, columns the state next month. Closed states never change, so their rows are left out.

| From \ To | Current | 1-29 DPD | 30-59 DPD | 60-89 DPD | 90+ DPD | Paid off | Repossessed | Charged off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current | 93.66% | 5.21% | 0.00% | 0.00% | 0.00% | 1.13% | 0.00% | 0.00% |
| 1-29 DPD | 32.96% | 45.36% | 21.42% | 0.00% | 0.00% | 0.26% | 0.00% | 0.00% |
| 30-59 DPD | 16.78% | 11.87% | 43.95% | 26.62% | 0.00% | 0.21% | 0.56% | 0.00% |
| 60-89 DPD | 13.07% | 1.87% | 8.68% | 40.15% | 33.23% | 0.24% | 2.76% | 0.00% |
| 90+ DPD | 6.23% | 0.12% | 0.00% | 3.08% | 68.39% | 0.13% | 15.19% | 6.85% |

## 2. Roll rates

The same matrix summarized: share of accounts in each state that improved, stayed, got worse, or closed.

| State | cure_to_current | improve | stay | roll_forward | paid_off | repossessed | charged_off |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current |  | 0.00% | 93.66% | 5.21% | 1.13% | 0.00% | 0.00% |
| 1-29 DPD | 32.96% | 32.96% | 45.36% | 21.42% | 0.26% | 0.00% | 0.00% |
| 30-59 DPD | 16.78% | 28.66% | 43.95% | 26.62% | 0.21% | 0.56% | 0.00% |
| 60-89 DPD | 13.07% | 23.62% | 40.15% | 33.23% | 0.24% | 2.76% | 0.00% |
| 90+ DPD | 6.23% | 9.43% | 68.39% | 0.00% | 0.13% | 15.19% | 6.85% |

## 3. What the Markov chain implies

If the matrix held forever, this is where an account in each state would eventually end up, and its chance of repossession within 12 months.

| Starting state | Paid off | Repossessed | Charged off | Repossessed within 12 months |
|---|---:|---:|---:|---:|
| Current | 67.93% | 23.56% | 8.52% | 1.44% |
| 1-29 DPD | 60.96% | 28.67% | 10.36% | 7.26% |
| 30-59 DPD | 49.77% | 36.90% | 13.34% | 18.63% |
| 60-89 DPD | 33.97% | 47.93% | 18.09% | 35.20% |
| 90+ DPD | 17.34% | 57.50% | 25.17% | 50.77% |

## 4. Markov projection against what happened

I took the 19,893 loans on the books at 2023-06-30, estimated the matrix from earlier months only, and projected 12 months ahead.

| State after 12 months | actual | markov_all_history | markov_last_12_months | error_all_history | error_last_12_months |
|---|---:|---:|---:|---:|---:|
| Current | 66.75% | 72.38% | 71.13% | +5.64% | +4.38% |
| 1-29 DPD | 7.53% | 7.73% | 7.55% | +0.20% | +0.02% |
| 30-59 DPD | 3.00% | 3.20% | 3.15% | +0.21% | +0.15% |
| 60-89 DPD | 1.46% | 1.53% | 1.49% | +0.07% | +0.03% |
| 90+ DPD | 1.57% | 1.65% | 1.49% | +0.08% | -0.08% |
| Paid off | 13.91% | 8.75% | 10.65% | -5.16% | -3.26% |
| Repossessed | 4.32% | 3.45% | 3.40% | -0.87% | -0.92% |
| Charged off | 1.47% | 1.30% | 1.15% | -0.17% | -0.32% |

## 5. Checking the Markov assumptions

### Same probabilities in every period?

Chi-square test of whether each state's row is the same across periods. With this many rows nearly
everything is significant, so Cramer's V (0 = identical, 1 = completely different) is the number to read.

| From state | Transitions | Chi-square | p-value | Cramer's V |
|---|---:|---:|---:|---:|
| Current | 774,024 | 736 | 1.3e-155 | 0.022 |
| 1-29 DPD | 81,283 | 229 | 3.2e-44 | 0.031 |
| 30-59 DPD | 32,060 | 93 | 2.3e-13 | 0.031 |
| 60-89 DPD | 14,447 | 54 | 1.6e-05 | 0.035 |
| 90+ DPD | 14,137 | 89 | 2.1e-11 | 0.046 |

Where accounts at 90+ DPD go next, by period:

| Period | Current | 1-29 DPD | 60-89 DPD | 90+ DPD | Paid off | Repossessed | Charged off |
|---|---:|---:|---:|---:|---:|---:|---:|
| after_2020 | 6.50% | 0.15% | 3.31% | 68.10% | 0.02% | 15.02% | 6.91% |
| before_pause | 6.39% | 0.68% | 2.74% | 71.46% | 0.00% | 14.16% | 4.57% |
| recent | 6.07% | 0.08% | 2.97% | 67.89% | 0.21% | 15.98% | 6.79% |
| servicing_pause | 5.67% | 0.00% | 2.32% | 79.38% | 0.00% | 2.58% | 10.05% |

### Does history matter?

For accounts in the same state today, split by where they were last month.

| State now | Last month | Improve next month | Roll forward next month |
|---|---|---:|---:|
| 1-29 DPD | came back from worse | 35.40% | 17.74% |
| 1-29 DPD | rolled in from better | 34.23% | 23.09% |
| 1-29 DPD | stayed | 31.14% | 19.96% |
| 30-59 DPD | came back from worse | 34.23% | 14.55% |
| 30-59 DPD | rolled in from better | 26.54% | 31.04% |
| 30-59 DPD | stayed | 30.79% | 22.20% |
| 60-89 DPD | came back from worse | 28.44% | 15.62% |
| 60-89 DPD | rolled in from better | 20.43% | 39.78% |
| 60-89 DPD | stayed | 27.99% | 24.84% |

| State now | Chi-square | Cramer's V |
|---|---:|---:|
| 1-29 DPD | 339 | 0.046 |
| 30-59 DPD | 674 | 0.103 |
| 60-89 DPD | 864 | 0.173 |

## 6. Vintage analysis

Cumulative repossession rate by origination half-year at fixed ages. A blank means the cohort is too young for every loan to have reached that age.

| Cohort | loans | observed_through_mob | repo_by_mob_12 | repo_by_mob_24 | repo_by_mob_36 | ever_60dpd_by_mob_12 |
|---|---:|---:|---:|---:|---:|---:|
| 2019-H1 | 2,734 | 60 | 2.30% | 5.38% | 7.97% | 10.28% |
| 2019-H2 | 2,738 | 60 | 1.61% | 5.51% | 8.04% | 10.96% |
| 2020-H1 | 2,946 | 54 | 2.85% | 6.21% | 8.28% | 10.45% |
| 2020-H2 | 2,787 | 48 | 2.19% | 5.31% | 7.97% | 8.43% |
| 2021-H1 | 2,996 | 42 | 1.70% | 4.24% | 7.58% | 8.78% |
| 2021-H2 | 2,954 | 36 | 1.79% | 5.11% | 9.31% | 8.29% |
| 2022-H1 | 3,197 | 30 | 2.06% | 6.47% |  | 9.73% |
| 2022-H2 | 3,115 | 24 | 1.83% | 6.65% |  | 9.12% |
| 2023-H1 | 3,393 | 18 | 2.30% |  |  | 9.78% |
| 2023-H2 | 3,140 | 12 | 2.10% |  |  | 9.39% |

## 7. Cures and delinquency duration

17,408 episodes of 30+ DPD, of which 1,348 were still open when the data ends. How resolved episodes ended, by the worst bucket reached:

| Worst bucket | episodes | charged_off | cured | paid_off | repossessed | median_months |
|---|---:|---:|---:|---:|---:|---:|
| 30-59 | 8,588 | 0.00% | 97.90% | 0.73% | 1.36% | 1 |
| 60-89 | 3,152 | 0.00% | 87.21% | 1.17% | 11.61% | 4 |
| 90+ | 4,320 | 22.43% | 25.14% | 0.46% | 51.97% | 6 |

Kaplan-Meier median episode length in months (unresolved episodes treated as censored): 30-59: 1, 60-89: 4, 90+: 6, all: 3.

Where delinquent accounts are three months later:

| State now | accounts | 1-29 DPD | charged off | current | paid off | repossessed | still 30+ DPD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-29 DPD | 78,020 | 20.31% | 0.00% | 54.08% | 1.44% | 0.22% | 23.95% |
| 30-59 DPD | 30,656 | 12.59% | 0.00% | 39.90% | 1.15% | 1.97% | 44.39% |
| 60-89 DPD | 13,735 | 5.15% | 0.00% | 30.05% | 1.16% | 12.92% | 50.72% |
| 90+ DPD | 13,444 | 1.46% | 21.04% | 12.14% | 0.44% | 39.77% | 25.15% |

## 8. Repossession as a servicing outcome

Loans that reached 60+ DPD, by servicing group. The groups share the same borrowers on average and differ only in how quickly the servicer assigns accounts for repossession.

| Servicing group | loans_reaching_60dpd | share_repossessed | share_charged_off | median_months_60dpd_to_repo |
|---|---:|---:|---:|---:|
| A | 3,731 | 39.10% | 11.36% | 4 |
| B | 2,246 | 34.51% | 13.58% | 4 |
| C | 1,439 | 26.96% | 16.68% | 5 |

Repossessions per account 60+ DPD at the prior month end: 3.44% during the simulated servicing pause, 9.19% in other months.

## 9. Monthly trend summary

| Year | avg_open_loans | avg_30_plus | avg_60_plus | repossessions | charge_offs | payoffs |
|---|---:|---:|---:|---:|---:|---:|
| 2019 | 2,674 | 3.30% | 1.24% | 31 | 5 | 136 |
| 2020 | 7,606 | 6.78% | 3.21% | 197 | 103 | 489 |
| 2021 | 12,178 | 6.06% | 2.84% | 398 | 143 | 1,003 |
| 2022 | 16,325 | 5.67% | 2.53% | 450 | 163 | 1,691 |
| 2023 | 19,925 | 6.82% | 3.19% | 697 | 233 | 2,489 |
| 2024 | 19,266 | 7.79% | 3.87% | 955 | 322 | 3,436 |
