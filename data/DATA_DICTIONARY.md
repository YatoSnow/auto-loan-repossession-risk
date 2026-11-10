# Data dictionary

All tables are synthetic. They are produced by `autoloan_sim` and linked by `loan_id`.

## `loans`

| Column | Description |
|---|---|
| `loan_id` | Synthetic loan identifier. |
| `borrower_id` | Synthetic borrower identifier. About 6% of loans share a borrower with another loan. |
| `origination_date` | Date the loan was funded. |
| `channel` | Dealer channel: franchise or independent. |
| `servicing_group` | Synthetic servicing policy group (A, B, C). Groups differ in how fast accounts are assigned for repossession. Not a geography. |
| `vehicle_type` | new or used. |
| `vehicle_age_years` | Vehicle age at origination. |
| `vehicle_mileage` | Odometer reading at origination. |
| `vehicle_value` | Vehicle value at origination, in dollars. |
| `loan_amount` | Amount financed: price plus taxes and fees, minus down payment. |
| `down_payment` | Cash down payment. |
| `ltv` | Loan-to-value ratio at origination (loan_amount / vehicle_value). |
| `apr` | Annual percentage rate as a decimal. |
| `term_months` | Contract term in months. |
| `scheduled_payment` | Level monthly payment from the amortization formula. |
| `payment_due_day` | Day of the month the payment is due (1-25). |
| `credit_score` | Synthetic credit score at origination (450-820). Not a real bureau score. |
| `monthly_income` | Synthetic stated monthly income. |
| `pti` | Payment-to-income ratio at origination. |

## `payments`

| Column | Description |
|---|---|
| `loan_id` | Loan identifier. |
| `due_date` | Due date of the installment. |
| `scheduled_amount` | Amount due for the month. |
| `amount_paid` | Total amount received during the billing month (0 if nothing was paid). |
| `payment_date` | Date the payment was received. Missing if no payment was made. |
| `days_late` | Days between the due date and the payment date (0 if on time). Missing if no payment. |

## `monthly_performance`

| Column | Description |
|---|---|
| `loan_id` | Loan identifier. |
| `as_of_date` | Month-end snapshot date. |
| `months_on_book` | Number of billing months since origination (first due month = 1). |
| `beginning_balance` | Principal balance at the start of the month. |
| `ending_balance` | Principal balance at month end. |
| `scheduled_amount` | Amount due for the month. |
| `amount_paid` | Amount received during the month. |
| `payments_behind` | Past-due amount in units of the scheduled payment. |
| `amount_past_due` | Past-due amount in dollars. |
| `days_past_due` | Days since the oldest unpaid due date, using 30-day months. A payment of at least 90% of the amount due counts as made. |
| `status` | current, dpd_1_29, dpd_30_59, dpd_60_89, dpd_90_plus, paid_off, repossessed, charged_off. |
| `extension_granted` | True if the servicer granted a payment extension this month. |
| `repo_assigned` | True if the account is assigned for repossession at month end. |
| `est_vehicle_value` | Depreciated vehicle value estimate. |

## `loan_outcomes`

| Column | Description |
|---|---|
| `loan_id` | Loan identifier. |
| `final_status` | active (still open at the data cutoff), paid_off, repossessed, or charged_off. |
| `censored` | True if the loan was still open at the data cutoff, so its final outcome is unknown. |
| `close_date` | Date the account closed. Missing for open loans. |
| `default_date` | First date the loan reached 120+ days past due, or was repossessed or charged off. |
| `repo_assigned_date` | Month end of the first repossession assignment. |
| `repossession_date` | Date the vehicle was repossessed or surrendered. |
| `voluntary_surrender` | True if the borrower surrendered the vehicle. |
| `balance_at_close` | Principal balance when the account closed. |
| `recovery_amount` | Net sale proceeds after a repossession. |
| `extensions_granted` | Number of extensions over the life of the loan. |

## Simulator assumptions

- Loans are simple-interest, fully amortizing, with one payment due per month.
- Originations run monthly from the first to the last origination month. Performance is
  observed through the data cutoff, so loans still open at that point are right censored.
- Each borrower has a hidden "stability" trait. The credit score is a noisy view of it.
  The trait itself is never written to the output tables.
- Each month a borrower is in a hidden normal or stressed state. Entering stress is more
  likely with low stability, high payment-to-income, high LTV, in the first six months,
  and when the simulated macro factor is high. Stress lasts about four to five months on average.
- Payment behavior depends on the hidden state and on how far behind the account is:
  pay in full, pay part, miss, or catch up. Catch-up payments are more common from
  February to April (a tax-refund effect).
- About 12% of borrowers habitually pay late in the month without missing, and about 10%
  have irregular payment amounts. These traits are hidden too.
- Days past due use 30-day months. A payment of 90% or more of the amount due counts as made.
- Voluntary prepayment (trade-in, refinance) is possible for accounts that are not behind.
- Default means the first time a loan reaches 120+ days past due, or is repossessed or charged off.
- Repossession is a servicer action. Accounts 60+ days past due can be assigned for
  repossession with a probability that depends on the delinquency bucket, the servicing
  group, and a monthly capacity factor. Assigned vehicles are recovered with a fixed
  monthly probability, and about 8% are never located. There is a simulated four-month
  servicing pause in 2020 when almost no accounts are assigned.
- Borrowers in stress can also surrender the vehicle voluntarily.
- Accounts 150+ days past due that were not repossessed are charged off and leave the portfolio.
  Real lenders often keep working charged-off accounts; this simulator does not.
- Vintages originated from mid-2021 to mid-2022 have looser underwriting.
- No protected characteristics, no geography, and no real personal data are simulated.
