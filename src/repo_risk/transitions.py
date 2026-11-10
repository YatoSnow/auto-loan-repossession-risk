"""Delinquency transition matrices and Markov projections.

States, in order:

    current, dpd_1_29, dpd_30_59, dpd_60_89, dpd_90_plus   (open)
    paid_off, repossessed, charged_off                      (closed, absorbing)

A first-order, time-homogeneous Markov chain assumes two things:

1. Next month's state depends only on this month's state, not on how the
   account got here or how long it has been here.
2. The transition probabilities are the same in every month.

Both are checked against the data in ``markov_checks``. Neither holds
exactly, which is why the Markov model is used here as a baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from autoloan_sim import STATUS_NAMES
from autoloan_sim import panel as pn

STATES = list(STATUS_NAMES)
OPEN_STATES = STATES[:5]
ABSORBING = STATES[5:]
STATE_LABELS = {
    "current": "Current",
    "dpd_1_29": "1-29 DPD",
    "dpd_30_59": "30-59 DPD",
    "dpd_60_89": "60-89 DPD",
    "dpd_90_plus": "90+ DPD",
    "paid_off": "Paid off",
    "repossessed": "Repossessed",
    "charged_off": "Charged off",
}
_CODE = {s: i for i, s in enumerate(STATES)}


def transition_pairs(monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per observed month-to-month move: state now, next state, previous state."""
    m = monthly.sort_values(["loan_id", "as_of_date"], kind="stable").reset_index(drop=True)
    code = m["status"].astype(str).map(_CODE).to_numpy()
    starts = pn.group_starts(m["loan_id"].to_numpy())
    nxt = pn.lag(code, starts, -1)
    prev = pn.lag(code, starts, 1)
    out = pd.DataFrame(
        {
            "loan_id": m["loan_id"],
            "as_of_date": m["as_of_date"],
            "months_on_book": m["months_on_book"],
            "from_state": code,
            "to_state": nxt,
            "prev_state": prev,
        }
    )
    out = out[~np.isnan(nxt)].copy()  # the last row of each loan has no next month
    out["to_state"] = out["to_state"].astype(int)
    return out


def transition_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    n = len(STATES)
    counts = np.zeros((n, n), dtype=np.int64)
    np.add.at(counts, (pairs["from_state"].to_numpy(), pairs["to_state"].to_numpy()), 1)
    return pd.DataFrame(counts, index=STATES, columns=STATES)


def transition_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """Row-normalized monthly transition probabilities. Closed states stay closed."""
    counts = transition_counts(pairs).astype(float)
    for s in ABSORBING:
        counts.loc[s] = 0.0
        counts.loc[s, s] = 1.0
    totals = counts.sum(axis=1)
    empty = totals == 0
    for s in counts.index[empty]:  # a state never seen: assume it stays put
        counts.loc[s, s] = 1.0
    return counts.div(counts.sum(axis=1), axis=0)


def roll_rates(matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarize each open state's row: got better, stayed, got worse, or closed."""
    rows = []
    for i, s in enumerate(OPEN_STATES):
        row = matrix.loc[s]
        rows.append(
            {
                "state": s,
                "cure_to_current": row["current"] if s != "current" else np.nan,
                "improve": row.iloc[:i].sum(),
                "stay": row.iloc[i],
                "roll_forward": row.iloc[i + 1 : 5].sum(),
                "paid_off": row["paid_off"],
                "repossessed": row["repossessed"],
                "charged_off": row["charged_off"],
            }
        )
    return pd.DataFrame(rows).set_index("state")


def project(matrix: pd.DataFrame, start: pd.Series, months: int) -> pd.DataFrame:
    """Project a state distribution forward. Row ``k`` is the distribution after ``k`` months."""
    P = matrix.to_numpy()
    v = start.reindex(STATES).fillna(0.0).to_numpy(dtype=float)
    rows = [v]
    for _ in range(months):
        v = v @ P
        rows.append(v)
    return pd.DataFrame(rows, columns=STATES).rename_axis("months_ahead")


def horizon_probability(matrix: pd.DataFrame, target: str, months: int) -> pd.Series:
    """Probability of being in absorbing state ``target`` within ``months``, by starting state."""
    Pn = np.linalg.matrix_power(matrix.to_numpy(), months)
    return pd.Series(Pn[:, STATES.index(target)], index=STATES, name=f"p_{target}_{months}m")


def eventual_absorption(matrix: pd.DataFrame) -> pd.DataFrame:
    """Probability of ending in each closed state, by starting open state (B = (I - Q)^-1 R)."""
    k = len(OPEN_STATES)
    P = matrix.to_numpy()
    Q, R = P[:k, :k], P[:k, k:]
    B = np.linalg.solve(np.eye(k) - Q, R)
    return pd.DataFrame(B, index=OPEN_STATES, columns=ABSORBING)


def actual_distribution(monthly: pd.DataFrame, snapshot: pd.Timestamp, months: int) -> pd.DataFrame:
    """Follow the loans open at ``snapshot`` and record where they are each month after.

    A loan that closes stays in its closing state, so every row sums to one.
    """
    m = monthly[["loan_id", "as_of_date", "status"]].copy()
    m["status"] = m["status"].astype(str)
    cohort = m.loc[(m["as_of_date"] == snapshot) & m["status"].isin(OPEN_STATES), "loan_id"]
    sub = m[m["loan_id"].isin(cohort) & (m["as_of_date"] >= snapshot)]
    wide = sub.pivot(index="loan_id", columns="as_of_date", values="status").sort_index(axis=1).ffill(axis=1)
    wide = wide.iloc[:, : months + 1]
    dist = pd.DataFrame([wide[c].value_counts(normalize=True).reindex(STATES).fillna(0.0) for c in wide.columns])
    dist.index = pd.RangeIndex(len(dist), name="months_ahead")
    return dist


def markov_checks(pairs: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> dict:
    """Test the two Markov assumptions with chi-square tests of homogeneity.

    ``periods`` maps a label to a (first month, last month) pair. With this
    many rows almost any difference is statistically significant, so the
    effect sizes in the returned tables matter more than the p-values.
    """
    out: dict = {}
    month = pairs["as_of_date"].dt.to_period("M")
    label = pd.Series("other", index=pairs.index)
    for name, (lo, hi) in periods.items():
        label[(month >= pd.Period(lo, freq="M")) & (month <= pd.Period(hi, freq="M"))] = name

    # 1. Time homogeneity: does the row for a state look the same in each period?
    homog = {}
    for s in OPEN_STATES:
        sub = pairs[pairs["from_state"] == _CODE[s]]
        table = pd.crosstab(label[sub.index], sub["to_state"])
        table = table.loc[:, table.sum() > 0]
        chi2, p, dof, _ = chi2_contingency(table)
        probs = table.div(table.sum(axis=1), axis=0)
        probs.columns = [STATES[c] for c in probs.columns]
        homog[s] = {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "n": int(table.to_numpy().sum()),
                    "cramers_v": float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))),
                    "probs_by_period": probs.round(4).to_dict(orient="index")}
    out["time_homogeneity"] = homog

    # 2. Memory: for accounts in a state now, does last month's state change what happens next?
    memory = {}
    for s in ("dpd_1_29", "dpd_30_59", "dpd_60_89"):
        sub = pairs[(pairs["from_state"] == _CODE[s]) & pairs["prev_state"].notna()]
        prev = np.sign(sub["prev_state"].to_numpy() - _CODE[s])
        came_from = pd.Series(prev, index=sub.index).map({-1.0: "rolled in from better", 0.0: "stayed", 1.0: "came back from worse"})
        table = pd.crosstab(came_from, sub["to_state"])
        table = table.loc[:, table.sum() > 0]
        chi2, p, dof, _ = chi2_contingency(table)
        probs = table.div(table.sum(axis=1), axis=0)
        cure = probs.loc[:, [c for c in probs.columns if c < _CODE[s]]].sum(axis=1)
        worse = probs.loc[:, [c for c in probs.columns if _CODE[s] < c < 5]].sum(axis=1)
        memory[s] = {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "n": int(table.to_numpy().sum()),
                     "cramers_v": float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))),
                     "improve_rate_by_history": cure.round(4).to_dict(),
                     "roll_forward_rate_by_history": worse.round(4).to_dict()}
    out["memory"] = memory
    return out
