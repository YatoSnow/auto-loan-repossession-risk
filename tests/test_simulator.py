"""Tests for the shared synthetic data generator."""

import numpy as np
import pandas as pd
import pytest

from autoloan_sim import (
    SimConfig,
    amortized_payment,
    load_portfolio,
    portfolio_summary,
    save_portfolio,
    simulate_portfolio,
    validate_portfolio,
)
from autoloan_sim.schema import DATA_DICTIONARY


@pytest.fixture(scope="module")
def sim():
    return simulate_portfolio(SimConfig(n_loans=1500, seed=7))


def test_amortized_payment_matches_known_value():
    # $20,000 at 6% for 60 months is a textbook $386.66
    assert amortized_payment(20_000, 0.06, 60) == pytest.approx(386.66, abs=0.01)
    assert amortized_payment(12_000, 0.0, 48) == pytest.approx(250.0)


def test_portfolio_passes_validation(sim):
    assert validate_portfolio(sim) == []


def test_same_seed_gives_same_data():
    a = simulate_portfolio(SimConfig(n_loans=400, seed=3))
    b = simulate_portfolio(SimConfig(n_loans=400, seed=3))
    c = simulate_portfolio(SimConfig(n_loans=400, seed=4))
    pd.testing.assert_frame_equal(a.monthly, b.monthly)
    pd.testing.assert_frame_equal(a.loans, b.loans)
    assert not a.loans["loan_amount"].equals(c.loans["loan_amount"])


def test_sample_size_is_configurable(sim):
    assert len(sim.loans) == 1500
    assert len(sim.outcomes) == 1500


def test_tables_match_data_dictionary(sim):
    for name, df in sim.tables().items():
        assert list(df.columns) == list(DATA_DICTIONARY[name])


def test_event_chronology(sim):
    o = sim.outcomes.merge(sim.loans[["loan_id", "origination_date"]], on="loan_id")
    closed = o.dropna(subset=["close_date"])
    assert (closed["close_date"] > closed["origination_date"]).all()
    repo = o.dropna(subset=["repossession_date"])
    assert (repo["default_date"] <= repo["repossession_date"]).all()
    assigned_then_repo = repo.dropna(subset=["repo_assigned_date"])
    assert (assigned_then_repo["repo_assigned_date"] < assigned_then_repo["repossession_date"]).all()


def test_no_activity_after_cutoff_and_open_loans_are_censored(sim):
    cutoff = pd.Timestamp("2024-12-31")
    assert sim.monthly["as_of_date"].max() <= cutoff
    assert sim.outcomes["censored"].sum() > 0
    assert (sim.outcomes.loc[sim.outcomes["censored"], "final_status"] == "active").all()


def test_headline_rates_are_plausible(sim):
    s = portfolio_summary(sim)
    assert 0.02 < s["share_repossessed"] < 0.25
    assert 0.05 < s["share_ever_60dpd"] < 0.50
    assert s["share_paid_off"] > 0.10


def test_no_single_feature_separates_outcomes(sim):
    """Guard against a generator where one column gives the answer away."""
    from sklearn.metrics import roc_auc_score

    o = sim.outcomes.set_index("loan_id")
    y = (o["final_status"] == "repossessed").astype(int)
    loans = sim.loans.set_index("loan_id").loc[y.index]
    for col in ["credit_score", "ltv", "pti", "apr", "loan_amount", "term_months"]:
        auc = roc_auc_score(y, loans[col])
        assert 0.25 < auc < 0.75, f"{col} alone has AUC {auc:.2f}"


def test_hidden_traits_are_not_exported(sim):
    for df in sim.tables().values():
        assert not {"z", "stressed", "skip", "habitual_late", "irregular"} & set(df.columns)


def test_save_and_load_roundtrip(sim, tmp_path):
    save_portfolio(sim, tmp_path)
    back = load_portfolio(tmp_path)
    assert len(back.monthly) == len(sim.monthly)
    assert back.config.seed == sim.config.seed
    assert np.allclose(back.loans["apr"], sim.loans["apr"])


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError):
        SimConfig.from_dict({"n_loan": 10})
