"""Streamlit dashboard: delinquency transitions and repossession analytics on synthetic loans.

Run from the repository root after the pipeline has been run:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoloan_sim import load_portfolio
from repo_risk import analytics as an
from repo_risk import transitions as tr
from repo_risk.cli import load_config
from repo_risk.train import MODEL_LABELS, load_bundle

st.set_page_config(page_title="Repossession risk (synthetic)", layout="wide")
cfg = load_config(ROOT / "configs" / "default.yaml")
paths = {k: ROOT / v for k, v in cfg["paths"].items()}
LABELS = [tr.STATE_LABELS[s] for s in tr.STATES]


@st.cache_resource
def get_portfolio():
    return load_portfolio(paths["raw"])


@st.cache_data
def get_pairs():
    return tr.transition_pairs(get_portfolio().monthly)


@st.cache_data
def get_vintage(event_col: str):
    p = get_portfolio()
    return an.vintage_curves(an.loan_event_ages(p), event_col, p.config.data_cutoff)


@st.cache_data
def get_episode_tables():
    p = get_portfolio()
    return an.cure_rates(an.delinquency_episodes(p)), an.forward_cure_table(p, 3), an.time_to_repossession(p)


@st.cache_data
def get_latest_scores():
    lm = pd.read_parquet(paths["processed"] / "landmarks.parquet")
    latest = lm[lm["landmark_date"] >= lm["landmark_date"].max() - pd.offsets.MonthEnd(2)].reset_index(drop=True)
    pred = load_bundle(paths["models"]).predict(latest, [12])
    return pd.concat([latest[["status", "servicing_group", "months_on_book"]], pred], axis=1)


st.title("Delinquency transitions and repossession risk")
st.warning(
    "Demo on **synthetic** data. Repossession here follows simulated servicing rules. "
    "Nothing on this page is a validated model or a basis for decisions about real accounts."
)
if not (paths["raw"] / "loans.parquet").exists() or not (paths["models"] / "models.joblib").exists():
    st.error("No data or models found. Run `python -m repo_risk.cli run-all` first.")
    st.stop()

tabs = st.tabs(["Transition matrix", "Markov projection", "Vintage curves", "Cures and servicing",
                "Portfolio risk", "Model evaluation"])

with tabs[0]:
    pairs = get_pairs()
    years = sorted(pairs["as_of_date"].dt.year.unique())
    choice = st.selectbox("Months used to estimate the matrix", ["All months", *[str(y) for y in years]])
    sub = pairs if choice == "All months" else pairs[pairs["as_of_date"].dt.year == int(choice)]
    matrix = tr.transition_matrix(sub)
    shown = matrix.loc[tr.OPEN_STATES]
    fig = go.Figure(go.Heatmap(z=np.sqrt(shown.to_numpy()), x=LABELS, y=LABELS[:5], colorscale="Blues", showscale=False,
                               text=[[f"{v:.1%}" for v in row] for row in shown.to_numpy()], texttemplate="%{text}"))
    fig.update_layout(yaxis_autorange="reversed", xaxis_title="State next month", yaxis_title="State this month",
                      title=f"Monthly transition probabilities ({len(sub):,} transitions)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Color uses a square-root scale so small probabilities stay visible.")
    st.subheader("Roll rates")
    st.dataframe(tr.roll_rates(matrix).rename(index=tr.STATE_LABELS).style.format("{:.2%}", na_rep=""))

with tabs[1]:
    st.markdown(
        "A first-order Markov chain assumes next month depends only on this month's state and that the "
        "probabilities never change. Both are approximations. See `reports/portfolio_report.md` for the checks."
    )
    matrix = tr.transition_matrix(get_pairs())
    c1, c2 = st.columns([1, 3])
    months = c1.slider("Months to project", 3, 36, 12)
    shares = {s: c1.number_input(f"Start share: {tr.STATE_LABELS[s]}", 0.0, 1.0, v, 0.01)
              for s, v in zip(tr.OPEN_STATES, (0.88, 0.07, 0.03, 0.01, 0.01), strict=True)}
    start = pd.Series(shares)
    if start.sum() <= 0:
        st.error("Enter at least one positive share.")
        st.stop()
    proj = tr.project(matrix, start / start.sum(), months).rename(columns=tr.STATE_LABELS)
    c2.plotly_chart(px.area(proj, labels={"value": "Share of accounts", "months_ahead": "Months ahead"},
                            title="Projected state distribution"), width="stretch")
    h = tr.horizon_probability(matrix, "repossessed", months).loc[tr.OPEN_STATES].rename(index=tr.STATE_LABELS)
    c2.dataframe(h.to_frame(f"Chance of repossession within {months} months").style.format("{:.2%}"))

with tabs[2]:
    what = st.radio("Event", ["Repossession", "First time 60+ DPD"], horizontal=True)
    curves = get_vintage("mob_repossessed" if what == "Repossession" else "mob_first_60")
    fig = px.line(curves, labels={"value": "Cumulative share of cohort", "months_on_book": "Months on book",
                                  "variable": "Origination cohort"}, title=f"{what}: cumulative rate by cohort")
    st.plotly_chart(fig, width="stretch")
    st.caption("Each curve stops at the last age that every loan in the cohort has had time to reach. "
               "Loans that paid off stay in the denominator.")

with tabs[3]:
    cures, forward, t2r = get_episode_tables()
    st.subheader("How 30+ DPD episodes ended, by worst bucket reached")
    st.dataframe(cures.style.format({c: "{:.1%}" for c in cures.columns if c not in ("episodes", "median_months")}
                                    | {"episodes": "{:,.0f}", "median_months": "{:.0f}"}))
    st.subheader("Where delinquent accounts are three months later")
    st.dataframe(forward.rename(index=tr.STATE_LABELS).style.format(
        {c: "{:.1%}" for c in forward.columns if c != "accounts"} | {"accounts": "{:,.0f}"}))
    st.subheader("Repossession depends on the servicer too")
    st.dataframe(t2r.style.format({"loans_reaching_60dpd": "{:,.0f}", "share_repossessed": "{:.1%}",
                                   "share_charged_off": "{:.1%}", "median_months_60dpd_to_repo": "{:.0f}"}))
    st.caption("Servicing groups A, B and C have the same kinds of borrowers. They differ only in how fast "
               "accounts are assigned for repossession, and their repossession rates differ as a result.")

with tabs[4]:
    scores = get_latest_scores()
    name = st.selectbox("Model", ["xgb_hazard", "logit_hazard", "markov", "xgb_direct"], format_func=MODEL_LABELS.get)
    col = f"{name}_12"
    st.metric("Open accounts in the latest landmarks", f"{len(scores):,}")
    st.metric("Expected share repossessed within 12 months", f"{scores[col].mean():.2%}")
    by_status = scores.groupby("status")[col].agg(["size", "mean", "sum"]).reindex(tr.OPEN_STATES)
    by_status.columns = ["accounts", "mean predicted probability", "expected repossessions"]
    st.dataframe(by_status.rename(index=tr.STATE_LABELS).style.format(
        {"accounts": "{:,.0f}", "mean predicted probability": "{:.2%}", "expected repossessions": "{:,.0f}"}))
    fig = px.histogram(scores.assign(status=scores["status"].map(tr.STATE_LABELS)), x=col, color="status", nbins=60,
                       log_y=True, labels={col: "Predicted probability of repossession within 12 months"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Aggregate view only. These are the most recent landmarks, so their outcomes are not known yet.")

with tabs[5]:
    metrics = json.loads((paths["reports"] / "model_metrics.json").read_text(encoding="utf-8"))
    h = st.selectbox("Horizon (months)", metrics["horizons"], index=len(metrics["horizons"]) - 1)
    block = metrics["complete_window_test"][str(h)]
    table = pd.DataFrame({MODEL_LABELS[m]: {"ROC-AUC": b["roc_auc"], "PR-AUC": b["pr_auc"], "Brier": b["brier"],
                                            "Mean predicted": b["mean_predicted"], "Observed": b["event_rate"]}
                          for m, b in block.items()}).T
    st.dataframe(table.style.format("{:.4f}"))
    c1, c2 = st.columns(2)
    c1.image(str(paths["figures"] / "calibration_12m.png"))
    c2.image(str(paths["figures"] / "cohort_predicted_vs_observed.png"))
    c1.image(str(paths["figures"] / "metrics_by_horizon.png"))
    c2.image(str(paths["figures"] / "censored_check.png"))
