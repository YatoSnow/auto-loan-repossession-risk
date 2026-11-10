"""Consistency checks for a simulated portfolio.

``validate_portfolio`` returns a list of problems. An empty list means the
tables agree with each other and the event dates are in a sensible order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generator import ACTIVE_STATUSES, TERMINAL_STATUSES, Portfolio, amortized_payment


def validate_portfolio(p: Portfolio) -> list[str]:
    problems: list[str] = []
    loans, monthly, payments, outcomes = p.loans, p.monthly, p.payments, p.outcomes

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    # --- keys -----------------------------------------------------------
    check(loans["loan_id"].is_unique, "loan_id is not unique in loans")
    check(outcomes["loan_id"].is_unique, "loan_id is not unique in loan_outcomes")
    check(set(monthly["loan_id"]) <= set(loans["loan_id"]), "monthly rows reference unknown loans")
    check(set(payments["loan_id"]) <= set(loans["loan_id"]), "payment rows reference unknown loans")
    check(not monthly.duplicated(["loan_id", "as_of_date"]).any(), "duplicate loan-month rows")

    # --- origination ----------------------------------------------------
    expected = amortized_payment(loans["loan_amount"], loans["apr"], loans["term_months"])
    check(np.allclose(expected, loans["scheduled_payment"], atol=0.01), "scheduled payment does not amortize the loan")
    check(loans["ltv"].between(0.3, 2.0).all(), "LTV outside a plausible range")
    check(loans["apr"].between(0.0, 0.30).all(), "APR outside a plausible range")
    check((loans["pti"] <= p.config.max_pti + 1e-6).all(), "PTI above the underwriting cap")
    check((loans["down_payment"] >= 0).all(), "negative down payment")

    # --- monthly performance --------------------------------------------
    m = monthly.merge(loans[["loan_id", "origination_date"]], on="loan_id")
    check((m["as_of_date"] > m["origination_date"]).all(), "performance row dated before origination")
    check((monthly["ending_balance"] >= 0).all(), "negative balance")
    check((monthly["amount_paid"] >= 0).all(), "negative payment amount")
    g = monthly.groupby("loan_id", sort=False)
    check((g["months_on_book"].diff().dropna() == 1).all(), "months on book does not increase by one")
    check((g["months_on_book"].first() == 1).all(), "first performance row is not month 1")

    status = monthly["status"].astype(str)
    dpd = monthly["days_past_due"]
    active = status.isin(ACTIVE_STATUSES)
    bucket = pd.cut(dpd, [-1, 0, 29, 59, 89, 10_000], labels=ACTIVE_STATUSES).astype(str)
    check((bucket[active] == status[active]).all(), "status does not match days past due")

    terminal = status.isin(TERMINAL_STATUSES)
    check(terminal.groupby(monthly["loan_id"]).sum().max() <= 1, "more than one terminal row for a loan")
    is_last = ~monthly.duplicated("loan_id", keep="last")
    check(is_last[terminal].all(), "performance rows exist after a terminal status")

    # balance roll-forward: ending = beginning + interest - paid for active rows
    mm = monthly.merge(loans[["loan_id", "apr"]], on="loan_id")
    rolled = mm["beginning_balance"] * (1 + mm["apr"] / 12) - mm["amount_paid"]
    ok = np.isclose(rolled.clip(lower=0), mm["ending_balance"], atol=1.01) | terminal.to_numpy() & (
        mm["status"].astype(str) != "paid_off"
    ).to_numpy()
    check(bool(ok.all()), "balance does not roll forward from payments and interest")

    # --- outcomes and chronology ----------------------------------------
    last = monthly[is_last].set_index("loan_id")["status"].astype(str)
    out = outcomes.set_index("loan_id")
    joined = out.join(last.rename("last_status"), how="left")
    closed = ~joined["censored"]
    check((joined.loc[closed, "final_status"] == joined.loc[closed, "last_status"]).all(),
          "final status disagrees with the last performance row")
    check(joined.loc[~closed, "last_status"].dropna().isin(ACTIVE_STATUSES).all(),
          "censored loan has a terminal performance row")
    check(joined.loc[closed, "close_date"].notna().all(), "closed loan without a close date")
    check(joined.loc[~closed, "close_date"].isna().all(), "open loan with a close date")

    repo = joined["final_status"] == "repossessed"
    check(joined.loc[repo, "repossession_date"].notna().all(), "repossessed loan without a repossession date")
    check(joined.loc[~repo, "repossession_date"].isna().all(), "repossession date on a loan that was not repossessed")
    both = joined["default_date"].notna() & joined["close_date"].notna()
    check((joined.loc[both, "default_date"] <= joined.loc[both, "close_date"]).all(), "default dated after closure")
    check(joined.loc[repo | (joined["final_status"] == "charged_off"), "default_date"].notna().all(),
          "repossessed or charged-off loan without a default date")

    # a repossession or charge-off must follow at least 30 days of delinquency
    prev_dpd = g["days_past_due"].shift(1)
    bad = monthly.loc[status.isin(["repossessed", "charged_off"]) & ~(prev_dpd >= 30)]
    check(bad.empty, "repossession or charge-off without prior delinquency")

    # --- payments -------------------------------------------------------
    paid = payments.dropna(subset=["payment_date"])
    check((payments.loc[payments["payment_date"].isna(), "amount_paid"] == 0).all(), "amount paid without a payment date")
    check((paid["payment_date"].dt.to_period("M") == paid["due_date"].dt.to_period("M")).all(),
          "payment recorded outside its billing month")
    cutoff = pd.Period(p.config.data_cutoff, freq="M").to_timestamp(how="end")
    check(monthly["as_of_date"].max() <= cutoff, "performance data after the cutoff date")
    return problems


def portfolio_summary(p: Portfolio) -> dict[str, float]:
    """A few headline rates, handy for calibration and for the README."""
    monthly, outcomes = p.monthly, p.outcomes
    status = monthly["status"].astype(str)
    active = monthly[status.isin(ACTIVE_STATUSES)]
    ever60 = monthly.loc[monthly["days_past_due"] >= 60, "loan_id"].nunique()
    ever30 = monthly.loc[monthly["days_past_due"] >= 30, "loan_id"].nunique()
    n = len(outcomes)
    fs = outcomes["final_status"].value_counts(normalize=True)
    return {
        "n_loans": n,
        "n_loan_months": len(monthly),
        "share_active_at_cutoff": float(fs.get("active", 0.0)),
        "share_paid_off": float(fs.get("paid_off", 0.0)),
        "share_repossessed": float(fs.get("repossessed", 0.0)),
        "share_charged_off": float(fs.get("charged_off", 0.0)),
        "share_ever_30dpd": ever30 / n,
        "share_ever_60dpd": ever60 / n,
        "avg_active_30plus_rate": float((active["days_past_due"] >= 30).mean()),
        "avg_active_60plus_rate": float((active["days_past_due"] >= 60).mean()),
        "mean_apr": float(p.loans["apr"].mean()),
        "mean_ltv": float(p.loans["ltv"].mean()),
        "mean_loan_amount": float(p.loans["loan_amount"].mean()),
        "mean_credit_score": float(p.loans["credit_score"].mean()),
    }
