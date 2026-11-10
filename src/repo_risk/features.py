"""Landmark table: one row per open loan per observation month.

A landmark is the date we stand on when making a prediction. Features use
data up to the landmark. What happens after the landmark is stored as
"months until the loan closed" plus the way it closed, or as censored if
the loan was still open when the data ends.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from autoloan_sim import Portfolio
from autoloan_sim import panel as pn

logger = logging.getLogger(__name__)

# exit codes used throughout the project
NO_EVENT, REPO, PAYOFF, CHARGEOFF = 0, 1, 2, 3
EXIT_CODES = {"repossessed": REPO, "paid_off": PAYOFF, "charged_off": CHARGEOFF}
EXIT_NAMES = {NO_EVENT: "open", REPO: "repossessed", PAYOFF: "paid_off", CHARGEOFF: "charged_off"}

FEATURES = [
    "credit_score",
    "pti",
    "ltv_orig",
    "apr",
    "term_months",
    "is_used",
    "months_on_book",
    "remaining_term",
    "balance_ratio",
    "current_ltv",
    "days_past_due",
    "payments_behind",
    "missed_3m",
    "missed_6m",
    "max_dpd_6m",
    "max_dpd_ever",
    "months_30plus_12m",
    "delinq_episodes_ever",
    "months_since_30plus",
    "months_in_episode",
    "pay_ratio_3m",
    "pay_ratio_6m",
    "extensions_ever",
    "repo_assigned",
    "servicing_group_B",
    "servicing_group_C",
]

FEATURE_DESCRIPTIONS = {
    "credit_score": "Synthetic credit score at origination",
    "pti": "Payment-to-income ratio at origination",
    "ltv_orig": "Loan-to-value ratio at origination",
    "apr": "Annual percentage rate",
    "term_months": "Contract term",
    "is_used": "1 if the vehicle was used",
    "months_on_book": "Billing months since origination",
    "remaining_term": "Contract term minus months on book",
    "balance_ratio": "Current balance divided by original amount",
    "current_ltv": "Current balance divided by depreciated vehicle value",
    "days_past_due": "Days past due at the landmark",
    "payments_behind": "Past-due amount in units of the monthly payment",
    "missed_3m": "Months with no payment, last 3 months",
    "missed_6m": "Months with no payment, last 6 months",
    "max_dpd_6m": "Worst month-end days past due, last 6 months",
    "max_dpd_ever": "Worst month-end days past due since origination",
    "months_30plus_12m": "Month ends at 30+ days past due, last 12 months",
    "delinq_episodes_ever": "Number of separate 30+ delinquency episodes so far",
    "months_since_30plus": "Months since last 30+ days past due (99 = never)",
    "months_in_episode": "Consecutive months at 30+ days past due, ending now",
    "pay_ratio_3m": "Amount paid divided by amount due, last 3 months",
    "pay_ratio_6m": "Amount paid divided by amount due, last 6 months",
    "extensions_ever": "Payment extensions granted so far",
    "repo_assigned": "1 if the account is already assigned for repossession",
    "servicing_group_B": "1 if the loan is in servicing group B",
    "servicing_group_C": "1 if the loan is in servicing group C",
}


def _running_max(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    big = x.max() + 1.0
    group_no = np.cumsum(np.concatenate([[0], (starts[1:] != starts[:-1]).astype(int)]))
    return np.maximum.accumulate(x + group_no * big) - group_no * big


def _streak(flag: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Length of the current run of True values inside each group."""
    i = np.arange(len(flag))
    last_false = np.maximum.accumulate(np.where(~flag, i, -1))
    return np.where(flag, i - np.maximum(last_false, starts - 1), 0).astype(float)


def add_features(m: pd.DataFrame) -> pd.DataFrame:
    """Add backward-looking features to a panel from ``prepare_panel``."""
    s = pn.group_starts(m["loan_id"].to_numpy())
    dpd = m["days_past_due"].to_numpy(dtype=float)
    bal, amt = m["ending_balance"].to_numpy(), m["loan_amount"].to_numpy()
    paid, sched = m["amount_paid"].to_numpy(), m["scheduled_amount"].to_numpy()
    missed = m["missed"].to_numpy(dtype=float)
    is30 = dpd >= 30

    f = pd.DataFrame(index=m.index)
    f["ltv_orig"] = m["ltv"]
    f["is_used"] = (m["vehicle_type"] == "used").astype(int)
    f["remaining_term"] = m["term_months"] - m["months_on_book"]
    f["balance_ratio"] = bal / amt
    f["current_ltv"] = bal / m["est_vehicle_value"].to_numpy()
    f["missed_3m"] = pn.trailing_sum(missed, s, 3)
    f["missed_6m"] = pn.trailing_sum(missed, s, 6)
    f["max_dpd_6m"] = pn.trailing_max(dpd, s, 6)
    f["max_dpd_ever"] = _running_max(dpd, s)
    f["months_30plus_12m"] = pn.trailing_sum(is30.astype(float), s, 12)
    entered = is30 & ~(pn.lag(is30.astype(float), s, 1, fill=0.0) > 0)
    f["delinq_episodes_ever"] = pn.trailing_sum(entered.astype(float), s, None)
    f["months_since_30plus"] = pn.months_since_last(is30, s)
    f["months_in_episode"] = _streak(is30, s)
    for w in (3, 6):
        due = pn.trailing_sum(sched, s, w)
        f[f"pay_ratio_{w}m"] = np.divide(pn.trailing_sum(paid, s, w), due, out=np.ones(len(m)), where=due > 0)
    f["extensions_ever"] = pn.trailing_sum(m["extension_granted"].to_numpy(dtype=float), s, None)
    f["servicing_group_B"] = (m["servicing_group"] == "B").astype(int)
    f["servicing_group_C"] = (m["servicing_group"] == "C").astype(int)
    out = pd.concat([m, f], axis=1)
    out["repo_assigned"] = out["repo_assigned"].astype(int)
    return out


def loan_exits(p: Portfolio) -> pd.DataFrame:
    """Month and type of each loan's exit. Open loans have exit_code 0 and exit_ord = NaN."""
    o = p.outcomes
    code = o["final_status"].map(EXIT_CODES).fillna(NO_EVENT).astype(int)
    exit_ord = o["close_date"].dt.to_period("M").apply(lambda x: x.ordinal if pd.notna(x) else np.nan)
    return pd.DataFrame({"loan_id": o["loan_id"], "exit_code": code, "exit_ord": exit_ord.astype(float)})


def build_landmarks(p: Portfolio, sample_every: int = 3, min_months_on_book: int = 1) -> pd.DataFrame:
    """Open loan-months with features and time-to-exit information."""
    m = add_features(pn.prepare_panel(p))
    loan_no = m["loan_id"].str[1:].astype(int)
    keep = m["is_active"] & (m["months_on_book"] >= min_months_on_book) & (
        (m["months_on_book"] + loan_no) % sample_every == 0
    )
    cols = ["loan_id", "borrower_id", "origination_date", "servicing_group", "as_of_date", "status", *FEATURES]
    lm = m.loc[keep, cols].rename(columns={"as_of_date": "landmark_date"}).reset_index(drop=True)
    lm["status"] = lm["status"].astype(str)
    lm = lm.merge(loan_exits(p), on="loan_id", how="left")

    cutoff_ord = pd.Period(p.config.data_cutoff, freq="M").ordinal
    lm["t_ord"] = lm["landmark_date"].dt.to_period("M").apply(lambda x: x.ordinal).astype(int)
    lm["months_to_exit"] = lm["exit_ord"] - lm["t_ord"]  # NaN while the loan is still open
    lm["months_observed"] = cutoff_ord - lm["t_ord"]  # follow-up available after the landmark
    logger.info("Built %d landmarks from %d loans", len(lm), lm["loan_id"].nunique())
    return lm


def outcome_within(lm: pd.DataFrame, horizon: int, last_ord: int | None = None) -> pd.DataFrame:
    """Outcome of each landmark inside ``horizon`` months.

    Returns ``event`` (exit code inside the window, 0 if none seen) and
    ``complete`` (True if the data covers the whole window).

    ``complete`` depends only on the calendar, never on what happened to
    the loan. It is tempting to also keep landmarks from the last few
    months whose loan happened to close early, because their outcome is
    known. Doing that keeps the early exits and drops the survivors, which
    inflates the event rate. So an incomplete window stays incomplete.
    """
    observed = lm["months_observed"] if last_ord is None else np.minimum(lm["months_observed"], last_ord - lm["t_ord"])
    exited = lm["months_to_exit"].notna() & (lm["months_to_exit"] <= np.minimum(horizon, observed))
    event = np.where(exited, lm["exit_code"], NO_EVENT)
    return pd.DataFrame({"event": event, "complete": observed >= horizon}, index=lm.index)
