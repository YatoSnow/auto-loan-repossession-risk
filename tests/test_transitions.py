"""Transition matrix and Markov projection tests."""

import numpy as np
import pandas as pd
import pytest

from repo_risk import transitions as tr


def _toy_monthly():
    dates = pd.date_range("2022-01-31", periods=4, freq="ME")
    rows = [
        ("L1", ["current", "dpd_1_29", "current", "paid_off"]),
        ("L2", ["current", "current", "dpd_1_29", "dpd_30_59"]),
        ("L3", ["dpd_60_89", "dpd_90_plus", "repossessed"]),
    ]
    frames = [pd.DataFrame({"loan_id": lid, "as_of_date": dates[: len(s)], "status": s,
                            "months_on_book": range(1, len(s) + 1)}) for lid, s in rows]
    return pd.concat(frames, ignore_index=True)


def test_counts_on_hand_built_panel():
    pairs = tr.transition_pairs(_toy_monthly())
    counts = tr.transition_counts(pairs)
    assert len(pairs) == 8  # 3 + 3 + 2 moves; the last row of each loan has no next month
    assert counts.loc["current", "dpd_1_29"] == 2
    assert counts.loc["current", "current"] == 1
    assert counts.loc["current", "paid_off"] == 1
    assert counts.loc["dpd_90_plus", "repossessed"] == 1
    assert counts.to_numpy().sum() == 8


def test_matrix_rows_sum_to_one_and_closed_states_absorb(portfolio):
    matrix = tr.transition_matrix(tr.transition_pairs(portfolio.monthly))
    assert np.allclose(matrix.sum(axis=1), 1.0)
    for s in tr.ABSORBING:
        assert matrix.loc[s, s] == 1.0
    # an account cannot skip a bucket in one month (30-day months)
    assert matrix.loc["current", ["dpd_30_59", "dpd_60_89", "dpd_90_plus"]].sum() == 0
    assert matrix.loc["dpd_1_29", ["dpd_60_89", "dpd_90_plus"]].sum() == 0


def test_no_transitions_out_of_closed_states(portfolio):
    pairs = tr.transition_pairs(portfolio.monthly)
    assert (pairs["from_state"] < 5).all()


def test_projection_matches_matrix_power(portfolio):
    matrix = tr.transition_matrix(tr.transition_pairs(portfolio.monthly))
    start = pd.Series({"current": 0.9, "dpd_30_59": 0.1})
    proj = tr.project(matrix, start, 12)
    assert np.allclose(proj.sum(axis=1), 1.0)
    expected = start.reindex(tr.STATES).fillna(0).to_numpy() @ np.linalg.matrix_power(matrix.to_numpy(), 12)
    assert np.allclose(proj.iloc[-1], expected)
    # closed states can only grow
    assert (proj[tr.ABSORBING].diff().dropna() >= -1e-12).all().all()


def test_horizon_probability_grows_and_approaches_absorption(portfolio):
    matrix = tr.transition_matrix(tr.transition_pairs(portfolio.monthly))
    p6 = tr.horizon_probability(matrix, "repossessed", 6)
    p12 = tr.horizon_probability(matrix, "repossessed", 12)
    assert (p12[tr.OPEN_STATES] >= p6[tr.OPEN_STATES]).all()
    assert p12["dpd_90_plus"] > p12["dpd_30_59"] > p12["current"]
    absorb = tr.eventual_absorption(matrix)
    assert np.allclose(absorb.sum(axis=1), 1.0)
    far = tr.horizon_probability(matrix, "repossessed", 2000)
    assert np.allclose(far[tr.OPEN_STATES], absorb["repossessed"], atol=1e-6)


def test_roll_rates_partition_each_row(portfolio):
    matrix = tr.transition_matrix(tr.transition_pairs(portfolio.monthly))
    rolls = tr.roll_rates(matrix)
    total = rolls[["improve", "stay", "roll_forward", "paid_off", "repossessed", "charged_off"]].sum(axis=1)
    assert np.allclose(total, 1.0)


def test_actual_distribution_follows_the_snapshot_cohort(portfolio):
    snap = pd.Timestamp("2023-06-30")
    actual = tr.actual_distribution(portfolio.monthly, snap, 12)
    assert len(actual) == 13
    assert np.allclose(actual.sum(axis=1), 1.0)
    assert actual.loc[0, tr.ABSORBING].sum() == 0  # everyone is open at the snapshot
    assert (actual[tr.ABSORBING].diff().dropna() >= -1e-12).all().all()
    # cross-check the repossessed share at month 12 with a direct count
    m = portfolio.monthly
    cohort = m.loc[(m["as_of_date"] == snap) & m["status"].astype(str).isin(tr.OPEN_STATES), "loan_id"]
    repo = m[(m["loan_id"].isin(cohort)) & (m["status"].astype(str) == "repossessed")
             & (m["as_of_date"] <= snap + pd.offsets.MonthEnd(12))]
    assert actual.loc[12, "repossessed"] == pytest.approx(len(repo) / len(cohort))


def test_markov_checks_return_both_tests(portfolio, cfg):
    pairs = tr.transition_pairs(portfolio.monthly)
    checks = tr.markov_checks(pairs, {k: tuple(v) for k, v in cfg["analytics"]["periods"].items()})
    assert set(checks["time_homogeneity"]) == set(tr.OPEN_STATES)
    for c in checks["memory"].values():
        assert 0 <= c["cramers_v"] <= 1
        assert set(c["improve_rate_by_history"]) <= {"rolled in from better", "stayed", "came back from worse"}
