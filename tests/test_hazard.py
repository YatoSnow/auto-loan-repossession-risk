"""Person-period expansion, cumulative incidence and censoring tests."""

import numpy as np
import pandas as pd
import pytest

from autoloan_sim import Portfolio
from autoloan_sim import panel as pn
from repo_risk.features import (
    CHARGEOFF,
    FEATURES,
    NO_EVENT,
    PAYOFF,
    REPO,
    add_features,
    build_landmarks,
    outcome_within,
)
from repo_risk.hazard import HazardModel, aalen_johansen, cumulative_incidence, expand_person_period


def _toy_landmarks():
    # months_to_exit NaN = still open; months_observed = follow-up available
    return pd.DataFrame({
        "t_ord": [100, 100, 100, 100],
        "months_to_exit": [3.0, np.nan, 2.0, 20.0],
        "months_observed": [30, 5, 30, 30],
        "exit_code": [REPO, NO_EVENT, PAYOFF, CHARGEOFF],
    })


def test_expand_person_period_on_hand_built_landmarks():
    pp = expand_person_period(_toy_landmarks(), horizon=12)
    by_row = pp.groupby("row")
    assert by_row.size().tolist() == [3, 5, 2, 12]  # exit at 3, censored at 5, exit at 2, horizon reached
    assert pp[pp["row"] == 0]["y"].tolist() == [0, 0, REPO]
    assert pp[pp["row"] == 1]["y"].tolist() == [0] * 5  # censored: only "survived" months
    assert pp[pp["row"] == 2]["y"].tolist() == [0, PAYOFF]
    assert pp[pp["row"] == 3]["y"].sum() == 0  # the exit at month 20 is outside the horizon
    assert pp.groupby("row")["k"].apply(list).iloc[0] == [1, 2, 3]


def test_training_truncation_hides_later_events():
    # cut follow-up 2 months after the landmark: the exit at month 3 must not be visible
    pp = expand_person_period(_toy_landmarks(), horizon=12, last_ord=102)
    assert pp.groupby("row").size().tolist() == [2, 2, 2, 2]
    assert pp[pp["row"] == 0]["y"].tolist() == [0, 0]
    assert pp[pp["row"] == 2]["y"].tolist() == [0, PAYOFF]


def test_outcome_window_completeness_depends_only_on_the_calendar():
    lm = _toy_landmarks()
    lm.loc[0, "months_observed"] = 4  # this loan was repossessed at month 3 with only 4 months of data
    o = outcome_within(lm, horizon=12)
    assert o["event"].tolist() == [REPO, NO_EVENT, PAYOFF, NO_EVENT]
    # the early repossession is known, but the window is still incomplete and must not be cherry-picked
    assert o["complete"].tolist() == [False, False, True, True]


def test_cumulative_incidence_closed_form():
    # constant monthly hazards: 2% repossession, 3% payoff
    H = 12
    hz = np.zeros((1, H, 4))
    hz[:, :, REPO], hz[:, :, PAYOFF] = 0.02, 0.03
    hz[:, :, NO_EVENT] = 0.95
    cif = cumulative_incidence(hz)
    expected = 0.02 * (1 - 0.95**H) / 0.05
    assert cif["repo"][0, -1] == pytest.approx(expected)
    assert cif["any_exit"][0, -1] == pytest.approx(1 - 0.95**H)
    # ignoring the competing exit would overstate the probability
    assert cif["repo"][0, -1] < 1 - 0.98**H
    assert (np.diff(cif["repo"][0]) > 0).all()


def test_aalen_johansen_equals_simple_share_without_censoring():
    rng = np.random.default_rng(0)
    time = rng.integers(1, 20, 5000)
    event = rng.choice([REPO, PAYOFF], 5000, p=[0.3, 0.7])
    aj = aalen_johansen(time, event, horizon=12)
    assert aj["cif"].iloc[-1] == pytest.approx(((time <= 12) & (event == REPO)).mean())


def test_aalen_johansen_matches_lifelines_with_censoring():
    from lifelines import AalenJohansenFitter

    rng = np.random.default_rng(1)
    n = 4000
    t_event = rng.geometric(0.06, n)
    cause = rng.choice([REPO, PAYOFF], n, p=[0.4, 0.6])
    t_cens = rng.integers(1, 15, n)
    time = np.minimum(t_event, t_cens)
    event = np.where(t_event <= t_cens, cause, NO_EVENT)
    mine = aalen_johansen(time, event, horizon=12)["cif"].iloc[-1]
    # lifelines assumes continuous time; put censoring just after events in the same month to match
    ajf = AalenJohansenFitter(calculate_variance=False, jitter_level=1e-6, seed=0)
    ajf.fit(time + np.where(event == NO_EVENT, 0.5, 0.0), event, event_of_interest=REPO)
    theirs = float(ajf.cumulative_density_.loc[:12.01].iloc[-1, 0])
    assert mine == pytest.approx(theirs, abs=0.003)
    # treating censored subjects as event-free understates the incidence
    naive = ((time <= 12) & (event == REPO)).mean()
    assert naive < mine


def test_hazard_models_recover_a_known_hazard():
    """Simulate exits from a known hazard that depends on one feature and check the fitted CIF."""
    rng = np.random.default_rng(2)
    n, H = 6000, 6
    x = rng.integers(0, 2, n)
    h_repo = np.where(x == 1, 0.06, 0.01)
    h_pay = 0.03
    to_exit, code = np.full(n, np.nan), np.zeros(n, dtype=int)
    for k in range(1, 40):
        u = rng.random(n)
        open_ = np.isnan(to_exit)
        repo = open_ & (u < h_repo)
        pay = open_ & ~repo & (u < h_repo + h_pay)
        to_exit[repo | pay] = k
        code[repo], code[pay] = REPO, PAYOFF
    lm = pd.DataFrame({f: 0.0 for f in FEATURES}, index=range(n))
    lm["months_on_book"] = x
    lm["t_ord"], lm["months_to_exit"], lm["exit_code"] = 100, to_exit, code
    lm["months_observed"] = rng.integers(1, 10, n)  # heavy censoring
    pp = expand_person_period(lm, H)
    truth = {v: h * (1 - (1 - h - h_pay) ** H) / (h + h_pay) for v, h in ((0, 0.01), (1, 0.06))}
    for kind, params in (("logit", {"C": 10.0}), ("xgb", {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.1})):
        model = HazardModel(kind, H, params).fit(lm, pp)
        cif = model.predict_cif(lm)[:, -1]
        for v in (0, 1):
            assert cif[x == v].mean() == pytest.approx(truth[v], abs=0.02), kind


def test_features_do_not_use_future_data(portfolio):
    cut = pd.Timestamp("2022-06-30")
    full = add_features(pn.prepare_panel(portfolio))
    truncated = Portfolio(portfolio.loans, portfolio.payments[portfolio.payments["due_date"] <= cut],
                          portfolio.monthly[portfolio.monthly["as_of_date"] <= cut], portfolio.outcomes, portfolio.config)
    trunc = add_features(pn.prepare_panel(truncated))
    a = full[(full["as_of_date"] == cut) & full["is_active"]].set_index("loan_id")[FEATURES]
    b = trunc[(trunc["as_of_date"] == cut) & trunc["is_active"]].set_index("loan_id")[FEATURES]
    assert len(a) > 100
    pd.testing.assert_frame_equal(a, b.loc[a.index], check_exact=False, rtol=1e-9, atol=1e-9)


def test_landmarks_agree_with_outcomes(portfolio):
    lm = build_landmarks(portfolio, sample_every=3)
    assert not lm[FEATURES].isna().any().any()
    exited = lm["months_to_exit"].notna()
    assert (lm.loc[exited, "months_to_exit"] >= 1).all()  # landmarks are open, so the exit is in the future
    assert (lm.loc[exited, "months_to_exit"] <= lm.loc[exited, "months_observed"]).all()
    assert (lm.loc[~exited, "exit_code"] == NO_EVENT).all()
    final = portfolio.outcomes.set_index("loan_id")["final_status"]
    repo_loans = set(lm.loc[lm["exit_code"] == REPO, "loan_id"])
    assert repo_loans <= set(final[final == "repossessed"].index)
