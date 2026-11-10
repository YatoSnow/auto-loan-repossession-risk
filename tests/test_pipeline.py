"""Analytics, training and evaluation tests on a small portfolio."""

import json

import numpy as np
import pandas as pd
import pytest

from repo_risk import analytics as an
from repo_risk.evaluate import evaluate_models
from repo_risk.features import REPO, build_landmarks, outcome_within
from repo_risk.report import build_report
from repo_risk.train import MODEL_NAMES, assign_blocks, load_bundle, month_ord, train_models
from repo_risk.transitions import transition_pairs


@pytest.fixture(scope="module")
def landmarks(portfolio, cfg):
    return assign_blocks(build_landmarks(portfolio, cfg["model"]["sample_every"]), cfg["split"])


@pytest.fixture(scope="module")
def bundle(portfolio, landmarks, cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train_models(landmarks, transition_pairs(portfolio.monthly), cfg, out)
    return load_bundle(out)


def test_blocks_respect_time_and_holdout(landmarks, cfg):
    end = month_ord(cfg["split"]["train_end"])
    assert (landmarks.loc[landmarks["block"].isin(["train", "val"]), "t_ord"] < end).all()
    assert (landmarks.loc[landmarks["block"] == "test", "t_ord"] >= month_ord(cfg["split"]["test_start"])).all()
    train_loans = set(landmarks.loc[landmarks["block"] == "train", "loan_id"])
    val_loans = set(landmarks.loc[landmarks["block"] == "val", "loan_id"])
    assert not train_loans & val_loans
    unseen = set(landmarks.loc[(landmarks["block"] == "test") & landmarks["holdout_loan"], "loan_id"])
    assert unseen and not unseen & train_loans


def test_test_windows_are_fully_observed(landmarks, cfg):
    test = landmarks[landmarks["block"] == "test"]
    assert outcome_within(test, cfg["model"]["horizon_months"])["complete"].all()
    recent = landmarks[landmarks["block"] == "recent"]
    assert not outcome_within(recent, cfg["model"]["horizon_months"])["complete"].any()


def test_vintage_curves_are_valid(portfolio):
    ages = an.loan_event_ages(portfolio)
    cutoff = portfolio.config.data_cutoff
    curves = an.vintage_curves(ages, "mob_repossessed", cutoff)
    for cohort in curves:
        c = curves[cohort].dropna()
        assert (c.diff().dropna() >= 0).all()  # cumulative curves never fall
        assert 0 <= c.iloc[-1] < 0.5
    # the youngest cohort only has 12 fully observed months, the oldest has the 60-month cap
    assert curves["2023-H2"].last_valid_index() == 12
    assert curves["2019-H1"].last_valid_index() == 60
    # direct recomputation for one cohort and age
    g = ages[ages["cohort"] == "2021-H1"]
    assert curves.loc[24, "2021-H1"] == pytest.approx((g["mob_repossessed"] <= 24).mean())


def test_episodes_and_cure_rates(portfolio):
    ep = an.delinquency_episodes(portfolio)
    assert (ep["months"] >= 1).all() and (ep["max_dpd"] >= 30).all()
    assert set(ep["end_type"]) <= {"cured", "repossessed", "charged_off", "paid_off", "censored"}
    # every repossessed loan's last episode must end in repossession
    repo_loans = portfolio.outcomes.loc[portfolio.outcomes["final_status"] == "repossessed", "loan_id"]
    last = ep.groupby("loan_id")["end_type"].last()
    assert (last.reindex(repo_loans) == "repossessed").all()
    cures = an.cure_rates(ep)
    shares = cures.drop(columns=["episodes", "median_months"]).sum(axis=1)
    assert np.allclose(shares, 1.0)
    assert cures.loc["30-59", "cured"] > cures.loc["90+", "cured"]


def test_forward_cure_table_rows_sum_to_one(portfolio):
    table = an.forward_cure_table(portfolio, 3)
    assert np.allclose(table.drop(columns="accounts").sum(axis=1), 1.0)


def test_monthly_trends_show_the_servicing_pause(portfolio):
    trends = an.monthly_trends(portfolio)
    pause = trends.loc["2020-04-30":"2020-07-31", "repossessions"].sum()
    after = trends.loc["2020-08-31":"2020-11-30", "repossessions"].sum()
    assert pause < after
    assert trends["dpd_30_plus_rate"].between(0, 1).all()


def test_models_predict_valid_probabilities(landmarks, bundle, cfg):
    test = landmarks[landmarks["block"] == "test"].reset_index(drop=True)
    pred = bundle.predict(test, [3, 12])
    assert pred.notna().all().all()
    assert ((pred >= 0) & (pred <= 1)).all().all()
    for name in ("markov", "logit_hazard", "xgb_hazard"):
        assert (pred[f"{name}_12"] >= pred[f"{name}_3"] - 1e-9).all()  # longer horizon, higher probability
    assert "xgb_direct_12" in pred and "xgb_direct_3" not in pred


def test_evaluation_report(landmarks, bundle, cfg, tmp_path):
    results = evaluate_models(landmarks, bundle, cfg, tmp_path / "reports", tmp_path / "figures")
    block = results["complete_window_test"]["12"]
    assert set(block) == set(MODEL_NAMES)
    for name in ("logit_hazard", "xgb_hazard", "xgb_direct"):
        assert block[name]["roc_auc"] > 0.65
    test = landmarks[landmarks["block"] == "test"]
    assert block["markov"]["event_rate"] == pytest.approx((outcome_within(test, 12)["event"] == REPO).mean())
    c = results["censored_check"]
    assert c["naive_censored_as_zero"] <= c["aalen_johansen"]
    saved = json.loads((tmp_path / "reports" / "model_metrics.json").read_text())
    assert saved["horizon_months"] == 12
    assert (tmp_path / "reports" / "model_evaluation.md").exists()
    for fig in ("calibration_12m", "cohort_predicted_vs_observed", "risk_distribution", "censored_check",
                "metrics_by_horizon"):
        assert (tmp_path / "figures" / f"{fig}.png").stat().st_size > 5000


def test_portfolio_report(portfolio, cfg, tmp_path):
    out = build_report(portfolio, cfg, tmp_path / "reports", tmp_path / "figures")
    assert out["n_transitions"] > 10_000
    text = (tmp_path / "reports" / "portfolio_report.md").read_text(encoding="utf-8")
    assert "## 4. Markov projection against what happened" in text
    comparison = pd.DataFrame(out["projection"]["comparison"]).T
    assert np.allclose(comparison.sum(), 1.0, atol=1e-3)
    for fig in ("transition_matrix", "markov_projection_vs_actual", "transition_stability", "vintage_curves",
                "months_on_book_hazards", "delinquency_duration", "monthly_trends"):
        assert (tmp_path / "figures" / f"{fig}.png").stat().st_size > 5000
