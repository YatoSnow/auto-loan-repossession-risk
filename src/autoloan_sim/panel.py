"""Helpers for working with the loan-month panel.

The monthly table has one row per loan per month, sorted by loan and date,
with no gaps. That lets us compute trailing-window statistics with plain
numpy instead of slow group-by rolling calls.

Every function here only looks backward: the value on a row uses that row
and earlier rows of the same loan. This is what keeps features free of
information from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generator import ACTIVE_STATUSES, Portfolio


def group_starts(keys: np.ndarray) -> np.ndarray:
    """For each row, the position of the first row of its group (keys must be sorted)."""
    keys = np.asarray(keys)
    is_start = np.ones(len(keys), dtype=bool)
    is_start[1:] = keys[1:] != keys[:-1]
    return np.maximum.accumulate(np.where(is_start, np.arange(len(keys)), 0))


def trailing_sum(x: np.ndarray, starts: np.ndarray, window: int | None) -> np.ndarray:
    """Sum of the last ``window`` rows of the group, including the current row.

    ``window=None`` gives the running total since the first row of the group.
    """
    x = np.asarray(x, dtype=float)
    i = np.arange(len(x))
    if window is None:
        cs = np.concatenate([[0.0], np.cumsum(x)])
        return cs[i + 1] - cs[starts]
    # Add the lags one by one. This is exact for a short window, while a
    # difference of two long cumulative sums picks up rounding noise.
    out = x.copy()
    for k in range(1, window):
        j = i - k
        ok = j >= starts
        out[ok] += x[j[ok]]
    return out


def trailing_count(starts: np.ndarray, window: int | None) -> np.ndarray:
    i = np.arange(len(starts))
    lo = starts if window is None else np.maximum(i - window + 1, starts)
    return (i - lo + 1).astype(float)


def trailing_mean(x: np.ndarray, starts: np.ndarray, window: int | None) -> np.ndarray:
    return trailing_sum(x, starts, window) / trailing_count(starts, window)


def trailing_std(x: np.ndarray, starts: np.ndarray, window: int | None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = trailing_count(starts, window)
    mean = trailing_sum(x, starts, window) / n
    var = trailing_sum(x * x, starts, window) / n - mean**2
    return np.sqrt(np.clip(var, 0, None))


def trailing_max(x: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    i = np.arange(len(x))
    out = x.copy()
    for k in range(1, window):
        j = i - k
        ok = j >= starts
        out[ok] = np.maximum(out[ok], x[j[ok]])
    return out


def lag(x: np.ndarray, starts: np.ndarray, k: int, fill: float = np.nan) -> np.ndarray:
    """Value ``k`` rows earlier in the same group (``k`` may be negative to look ahead)."""
    x = np.asarray(x, dtype=float)
    i = np.arange(len(x))
    j = i - k
    out = np.full(len(x), fill, dtype=float)
    if k >= 0:
        ok = j >= starts
    else:
        ends = _group_ends(starts)
        ok = j <= ends
    out[ok] = x[j[ok]]
    return out


def _group_ends(starts: np.ndarray) -> np.ndarray:
    n = len(starts)
    is_end = np.ones(n, dtype=bool)
    is_end[:-1] = starts[1:] != starts[:-1]
    idx = np.where(is_end, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def months_since_last(flag: np.ndarray, starts: np.ndarray, never: float = 99.0) -> np.ndarray:
    """Months since ``flag`` was last true in the group (0 if true now)."""
    flag = np.asarray(flag, dtype=bool)
    i = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, i, -1))
    valid = last >= starts
    return np.where(valid, i - last, never).astype(float)


def prepare_panel(p: Portfolio) -> pd.DataFrame:
    """Monthly panel joined with loan terms and payment timing, plus basic payment flags.

    Returns one row per loan-month, sorted by loan and date. Terminal rows
    (paid off, repossessed, charged off) are kept so that outcomes can be
    looked up, and are marked with ``is_active = False``.
    """
    m = p.monthly.copy()
    pay = p.payments[["loan_id", "due_date", "days_late"]].copy()
    pay["as_of_date"] = pay["due_date"] + pd.offsets.MonthEnd(0)
    m = m.merge(pay[["loan_id", "as_of_date", "days_late"]], on=["loan_id", "as_of_date"], how="left")
    m["days_late"] = m["days_late"].astype("float64")
    m = m.merge(p.loans, on="loan_id", how="left")
    m = m.sort_values(["loan_id", "as_of_date"], kind="stable").reset_index(drop=True)

    status = m["status"].astype(str)
    m["is_active"] = status.isin(ACTIVE_STATUSES)
    sched = m["scheduled_amount"].to_numpy()
    paid = m["amount_paid"].to_numpy()
    billed = m["is_active"].to_numpy() & (sched > 0)
    ratio = np.divide(paid, sched, out=np.zeros(len(m)), where=sched > 0)
    m["pay_ratio"] = np.where(billed, np.minimum(ratio, 3.0), np.nan)
    m["missed"] = billed & (paid == 0)
    m["partial"] = billed & (paid > 0) & (ratio < 0.9)
    m["paid_late"] = billed & (m["days_late"].fillna(0).to_numpy() > 0)
    m["days_late_filled"] = np.where(m["missed"], 30.0, m["days_late"].fillna(0).to_numpy())
    return m
