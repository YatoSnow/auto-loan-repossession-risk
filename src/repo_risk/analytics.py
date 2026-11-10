"""Portfolio analytics: vintages, months on book, cures, delinquency duration, trends.

Three terms that are easy to mix up, as used in this project:

* **Delinquency** is a payment status: the account is some number of days past due.
* **Default** is a definition: the first time a loan reaches 120+ days past
  due, or is repossessed or charged off.
* **Repossession** is an action by the servicer: the vehicle is recovered.
  It depends on the borrower falling behind, and also on servicing policy,
  capacity, and whether the vehicle can be found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoloan_sim import ACTIVE_STATUSES, Portfolio
from autoloan_sim import panel as pn


def _ord(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").apply(lambda x: x.ordinal)


# ----------------------------------------------------------------------
# Vintage analysis
# ----------------------------------------------------------------------
def loan_event_ages(p: Portfolio) -> pd.DataFrame:
    """Per loan: cohort labels and the month on book of key events (NaN if it never happened)."""
    m = p.monthly
    status = m["status"].astype(str)
    first = lambda mask: m[mask].groupby("loan_id")["months_on_book"].min()  # noqa: E731
    out = p.loans[["loan_id", "origination_date", "servicing_group"]].set_index("loan_id")
    out["mob_repossessed"] = first(status == "repossessed")
    out["mob_charged_off"] = first(status == "charged_off")
    out["mob_paid_off"] = first(status == "paid_off")
    out["mob_first_30"] = first((m["days_past_due"] >= 30) & status.isin(ACTIVE_STATUSES))
    out["mob_first_60"] = first((m["days_past_due"] >= 60) & status.isin(ACTIVE_STATUSES))
    out["orig_ord"] = _ord(out["origination_date"])
    out["orig_year"] = out["origination_date"].dt.year
    half = np.where(out["origination_date"].dt.month <= 6, "H1", "H2")
    out["cohort"] = out["orig_year"].astype(str) + "-" + half
    return out.reset_index()


def vintage_curves(ages: pd.DataFrame, event_col: str, cutoff: str, cohort_col: str = "cohort",
                   max_mob: int = 60) -> pd.DataFrame:
    """Cumulative share of each cohort that had the event by each month on book.

    A cohort's curve stops at the last month on book that *every* loan in
    the cohort has had the chance to reach. Going further would mix loans
    with different amounts of follow-up and bend the curve down for no
    real reason. Loans that paid off stay in the denominator, so this is a
    cumulative incidence with payoff as a competing exit.
    """
    cutoff_ord = pd.Period(cutoff, freq="M").ordinal
    curves = {}
    for cohort, g in ages.groupby(cohort_col):
        observed_through = int(cutoff_ord - g["orig_ord"].max())  # youngest loan's months on book at cutoff
        mobs = np.arange(1, min(max_mob, observed_through) + 1)
        event_age = g[event_col].to_numpy()
        curves[cohort] = pd.Series([(event_age <= k).mean() for k in mobs], index=mobs)
    return pd.DataFrame(curves).rename_axis("months_on_book")


def cohort_summary(ages: pd.DataFrame, cutoff: str, at_mobs: tuple[int, ...] = (12, 24, 36)) -> pd.DataFrame:
    """Loans per cohort and cumulative repossession rate at fixed ages (NaN if not yet observable)."""
    repo = vintage_curves(ages, "mob_repossessed", cutoff)
    d60 = vintage_curves(ages, "mob_first_60", cutoff)
    rows = []
    for cohort, g in ages.groupby("cohort"):
        row = {"cohort": cohort, "loans": len(g), "observed_through_mob": int(repo[cohort].last_valid_index())}
        for k in at_mobs:
            row[f"repo_by_mob_{k}"] = repo[cohort].get(k, np.nan)
        row["ever_60dpd_by_mob_12"] = d60[cohort].get(12, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("cohort")


# ----------------------------------------------------------------------
# Months-on-book hazards
# ----------------------------------------------------------------------
def mob_hazards(p: Portfolio, max_mob: int = 60) -> pd.DataFrame:
    """Monthly rate of repossession, charge-off, payoff and first 60+ DPD by months on book."""
    m = p.monthly
    status = m["status"].astype(str)
    mob = m["months_on_book"]
    # every row is a loan that entered that month open, so the row count is the number at risk
    at_risk = mob.value_counts().sort_index()
    out = pd.DataFrame({"at_risk": at_risk})
    for name in ("repossessed", "charged_off", "paid_off"):
        out[f"{name}_hazard"] = mob[status == name].value_counts().reindex(out.index).fillna(0) / at_risk
    ages = loan_event_ages(p)
    first60 = ages["mob_first_60"].value_counts().reindex(out.index).fillna(0)
    out["first_60dpd_rate"] = first60 / at_risk
    return out.loc[:max_mob].rename_axis("months_on_book")


# ----------------------------------------------------------------------
# Delinquency episodes, cures and duration
# ----------------------------------------------------------------------
def delinquency_episodes(p: Portfolio, threshold: int = 30) -> pd.DataFrame:
    """One row per stretch of consecutive months at ``threshold``+ days past due.

    ``end_type`` is how the stretch ended: cured (back under the
    threshold), repossessed, charged_off, paid_off, or censored (still
    delinquent when the data ends).
    """
    m = p.monthly.sort_values(["loan_id", "as_of_date"], kind="stable").reset_index(drop=True)
    status = m["status"].astype(str)
    active = status.isin(ACTIVE_STATUSES).to_numpy()
    delinquent = active & (m["days_past_due"].to_numpy() >= threshold)
    starts = pn.group_starts(m["loan_id"].to_numpy())
    prev = pn.lag(delinquent.astype(float), starts, 1, fill=0.0) > 0
    episode_no = np.cumsum(delinquent & ~prev)
    d = m[delinquent].assign(episode=episode_no[delinquent])
    ep = d.groupby("episode").agg(
        loan_id=("loan_id", "first"), start=("as_of_date", "first"), last=("as_of_date", "last"),
        months=("as_of_date", "size"), max_dpd=("days_past_due", "max"), start_mob=("months_on_book", "first"),
    )
    # the row right after the last delinquent month tells us how the episode ended
    last_idx = d.groupby("episode").apply(lambda g: g.index[-1], include_groups=False).to_numpy()
    nxt = last_idx + 1
    same_loan = (nxt < len(m)) & (m["loan_id"].to_numpy()[np.minimum(nxt, len(m) - 1)] == ep["loan_id"].to_numpy())
    next_status = np.where(same_loan, status.to_numpy()[np.minimum(nxt, len(m) - 1)], "censored")
    ep["end_type"] = np.where(np.isin(next_status, ACTIVE_STATUSES), "cured", next_status)
    ep["worst_bucket"] = pd.cut(ep["max_dpd"], [29, 59, 89, 10_000], labels=["30-59", "60-89", "90+"])
    return ep.reset_index(drop=True)


def cure_rates(episodes: pd.DataFrame) -> pd.DataFrame:
    """How episodes ended, by the worst bucket they reached. Unresolved episodes are left out."""
    done = episodes[episodes["end_type"] != "censored"]
    table = pd.crosstab(done["worst_bucket"], done["end_type"], normalize="index")
    table.insert(0, "episodes", done["worst_bucket"].value_counts().reindex(table.index))
    table["median_months"] = done.groupby("worst_bucket", observed=True)["months"].median()
    return table


def forward_cure_table(p: Portfolio, months: int = 3) -> pd.DataFrame:
    """Where accounts in each delinquent state are ``months`` later.

    Only uses snapshots that are at least ``months`` before the data
    cutoff, so every row has a known answer.
    """
    m = p.monthly.sort_values(["loan_id", "as_of_date"], kind="stable").reset_index(drop=True)
    status = m["status"].astype(str)
    wide = status.to_numpy()
    starts = pn.group_starts(m["loan_id"].to_numpy())
    code = pd.Categorical(wide).codes.astype(float)
    cats = pd.Categorical(wide).categories
    # status `months` ahead; if the loan closed earlier, carry its closing status forward
    ahead = pn.lag(code, starts, -months)
    final_code = pd.Series(code).groupby(m["loan_id"].to_numpy()).transform("last").to_numpy()
    closed = ~np.isin(cats[final_code.astype(int)], ACTIVE_STATUSES)
    ahead = np.where(np.isnan(ahead) & closed, final_code, ahead)
    cutoff = pd.Period(p.config.data_cutoff, freq="M").to_timestamp(how="end").normalize()
    known = ~np.isnan(ahead) & (m["as_of_date"] <= cutoff - pd.offsets.MonthEnd(months)).to_numpy()
    df = pd.DataFrame({"from": wide[known], "to": cats[ahead[known].astype(int)]})
    df = df[df["from"].isin(ACTIVE_STATUSES[1:])]
    group = df["to"].map({"current": "current", "dpd_1_29": "1-29 DPD", "paid_off": "paid off",
                          "repossessed": "repossessed", "charged_off": "charged off"}).fillna("still 30+ DPD")
    table = pd.crosstab(df["from"], group, normalize="index")
    table.insert(0, "accounts", df["from"].value_counts().reindex(table.index))
    return table


def time_to_repossession(p: Portfolio) -> pd.DataFrame:
    """Months from first reaching 60+ DPD to repossession, by servicing group."""
    ages = loan_event_ages(p)
    reached = ages[ages["mob_first_60"].notna()].copy()
    reached["repossessed"] = reached["mob_repossessed"].notna()
    reached["months_60_to_repo"] = reached["mob_repossessed"] - reached["mob_first_60"]
    g = reached.groupby("servicing_group")
    return pd.DataFrame({
        "loans_reaching_60dpd": g.size(),
        "share_repossessed": g["repossessed"].mean(),
        "share_charged_off": g["mob_charged_off"].apply(lambda s: s.notna().mean()),
        "median_months_60dpd_to_repo": g["months_60_to_repo"].median(),
    })


# ----------------------------------------------------------------------
# Monthly trends
# ----------------------------------------------------------------------
def monthly_trends(p: Portfolio) -> pd.DataFrame:
    """Calendar-month view: open loans, delinquency rates, repossessions and how fast 60+ accounts are repossessed."""
    m = p.monthly
    status = m["status"].astype(str)
    is_open = status.isin(ACTIVE_STATUSES)
    g = m.groupby("as_of_date")
    out = pd.DataFrame({
        "open_loans": is_open.groupby(m["as_of_date"]).sum(),
        "dpd_30_plus_rate": ((m["days_past_due"] >= 30) & is_open).groupby(m["as_of_date"]).sum(),
        "dpd_60_plus_rate": ((m["days_past_due"] >= 60) & is_open).groupby(m["as_of_date"]).sum(),
        "repossessions": (status == "repossessed").groupby(m["as_of_date"]).sum(),
        "charge_offs": (status == "charged_off").groupby(m["as_of_date"]).sum(),
        "payoffs": (status == "paid_off").groupby(m["as_of_date"]).sum(),
        "rows": g.size(),
    })
    open_60 = out["dpd_60_plus_rate"].copy()
    out["dpd_30_plus_rate"] /= out["open_loans"]
    out["dpd_60_plus_rate"] /= out["open_loans"]
    # repossessions this month per account that was 60+ DPD at the end of last month
    out["repo_per_60plus_account"] = out["repossessions"] / open_60.shift(1)
    return out.drop(columns="rows")
