"""Portfolio analytics report: transition matrices, Markov checks, vintages, cures, trends."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from autoloan_sim import Portfolio

from . import analytics as an
from . import transitions as tr

logger = logging.getLogger(__name__)
STATE_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2", "#64B5CD", "#000000", "#8C8C8C"]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _md_table(df: pd.DataFrame, fmt: dict[str, str] | str = "{:.2%}", index_name: str | None = None) -> list[str]:
    df = df.copy()
    head = [index_name or df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * len(df.columns)]
    for idx, row in df.iterrows():
        cells = []
        for col, v in row.items():
            f = fmt.get(col, "{:.2%}") if isinstance(fmt, dict) else fmt
            cells.append("" if pd.isna(v) else f.format(v))
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return lines


def build_report(p: Portfolio, cfg: dict, report_dir: Path, figure_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    a_cfg = cfg["analytics"]
    cutoff = p.config.data_cutoff
    labels = [tr.STATE_LABELS[s] for s in tr.STATES]
    out: dict = {}

    # ------------------------------------------------------------------ transitions
    pairs = tr.transition_pairs(p.monthly)
    matrix = tr.transition_matrix(pairs)
    rolls = tr.roll_rates(matrix)
    absorb = tr.eventual_absorption(matrix)
    h12 = tr.horizon_probability(matrix, "repossessed", 12)
    out["transition_matrix"] = matrix.round(5).to_dict(orient="index")
    out["n_transitions"] = int(len(pairs))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    shown = matrix.loc[tr.OPEN_STATES]
    ax.imshow(np.sqrt(shown.to_numpy()), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(shown.to_numpy()):
        ax.text(j, i, f"{v:.1%}" if v >= 0.0005 else "-", ha="center", va="center", fontsize=8,
                color="white" if v > 0.45 else "black")
    ax.set_xticks(range(8), labels, rotation=35, ha="right")
    ax.set_yticks(range(5), labels[:5])
    ax.set(xlabel="State next month", ylabel="State this month",
           title=f"Monthly transition matrix, all months ({len(pairs):,} transitions)\ncolor on a square-root scale")
    _save(fig, figure_dir / "transition_matrix.png")

    # ------------------------------------------------------------------ Markov projection vs actual
    snap = pd.Period(a_cfg["projection_snapshot"], freq="M").to_timestamp(how="end").normalize()
    n_ahead = int(a_cfg["projection_months"])
    before = pairs[pairs["as_of_date"] < snap]  # transitions that finished on or before the snapshot
    recent = before[before["as_of_date"] >= snap - pd.offsets.MonthEnd(12)]
    actual = tr.actual_distribution(p.monthly, snap, n_ahead)
    proj_all = tr.project(tr.transition_matrix(before), actual.iloc[0], n_ahead)
    proj_recent = tr.project(tr.transition_matrix(recent), actual.iloc[0], n_ahead)
    comparison = pd.DataFrame({"actual": actual.iloc[-1], "markov_all_history": proj_all.iloc[-1],
                               "markov_last_12_months": proj_recent.iloc[-1]})
    comparison.index = labels
    out["projection"] = {"snapshot": str(snap.date()), "months": n_ahead,
                         "open_loans_at_snapshot": int(((p.monthly["as_of_date"] == snap)
                                                        & p.monthly["status"].astype(str).isin(tr.OPEN_STATES)).sum()),
                         "comparison": comparison.round(5).to_dict(orient="index")}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, states, title in ((axes[0], ["dpd_30_59", "dpd_60_89", "dpd_90_plus"], "Delinquent states"),
                              (axes[1], ["paid_off", "repossessed", "charged_off"], "Closed states (cumulative)")):
        for s in states:
            c = STATE_COLORS[tr.STATES.index(s)]
            ax.plot(actual.index, actual[s], "-", color=c, lw=2, label=f"{tr.STATE_LABELS[s]} actual")
            ax.plot(proj_all.index, proj_all[s], "--", color=c, label=f"{tr.STATE_LABELS[s]} Markov")
        ax.set(xlabel=f"Months after {snap.date()}", ylabel="Share of the snapshot cohort", title=title, ylim=(0, None))
        ax.legend(fontsize=7)
    fig.suptitle("Markov projection (dashed) vs what actually happened (solid)", y=1.02)
    _save(fig, figure_dir / "markov_projection_vs_actual.png")

    # ------------------------------------------------------------------ Markov assumption checks
    checks = tr.markov_checks(pairs, {k: tuple(v) for k, v in a_cfg["periods"].items()})
    out["markov_checks"] = checks
    monthly_rows = []
    for date, g in pairs.groupby("as_of_date"):
        r30 = g[g["from_state"] == 2]
        r90 = g[g["from_state"] == 4]
        monthly_rows.append({"date": date,
                             "roll_30_to_60": (r30["to_state"] == 3).mean() if len(r30) >= 30 else np.nan,
                             "repo_from_90": (r90["to_state"] == 6).mean() if len(r90) >= 30 else np.nan})
    by_month = pd.DataFrame(monthly_rows).set_index("date")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(by_month.index, by_month["roll_30_to_60"], label="30-59 DPD rolls to 60-89 DPD", color="#DD8452")
    ax.plot(by_month.index, by_month["repo_from_90"], label="90+ DPD is repossessed", color="black")
    ax.axhline(matrix.loc["dpd_30_59", "dpd_60_89"], color="#DD8452", ls=":", lw=1)
    ax.axhline(matrix.loc["dpd_90_plus", "repossessed"], color="black", ls=":", lw=1)
    ax.set(ylabel="Monthly transition probability", ylim=(0, None),
           title="Two transition probabilities month by month (dotted = pooled matrix value)")
    ax.legend(fontsize=8)
    _save(fig, figure_dir / "transition_stability.png")

    # ------------------------------------------------------------------ vintages
    ages = an.loan_event_ages(p)
    repo_curves = an.vintage_curves(ages, "mob_repossessed", cutoff)
    d60_curves = an.vintage_curves(ages, "mob_first_60", cutoff)
    cohorts = an.cohort_summary(ages, cutoff)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    for ax, curves, title in ((axes[0], repo_curves, "Cumulative repossession rate"),
                              (axes[1], d60_curves, "Cumulative share ever 60+ DPD")):
        for i, c in enumerate(curves.columns):
            ax.plot(curves.index, curves[c], color=cmap(i / max(len(curves.columns) - 1, 1)), label=c)
        ax.set(xlabel="Months on book", ylabel="Share of cohort", title=f"{title} by origination half-year")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Vintage curves. Each curve stops where its youngest loans run out of history.", y=1.02)
    _save(fig, figure_dir / "vintage_curves.png")

    # ------------------------------------------------------------------ months on book
    haz = an.mob_hazards(p)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for col, label, c in (("first_60dpd_rate", "First time 60+ DPD", "#C44E52"), ("repossessed_hazard", "Repossession", "black"),
                          ("paid_off_hazard", "Payoff", "#64B5CD"), ("charged_off_hazard", "Charge-off", "#8C8C8C")):
        ax.plot(haz.index, haz[col].rolling(3, min_periods=1, center=True).mean(), label=label, color=c)
    ax.set(xlabel="Months on book", ylabel="Monthly rate among open loans (3-month average)",
           title="Monthly event rates by months on book")
    ax.legend(fontsize=8)
    _save(fig, figure_dir / "months_on_book_hazards.png")

    # ------------------------------------------------------------------ episodes, cures, duration
    episodes = an.delinquency_episodes(p)
    cures = an.cure_rates(episodes)
    forward = an.forward_cure_table(p, 3)
    t2r = an.time_to_repossession(p)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    km_medians = {}
    for bucket, c in (("30-59", "#DD8452"), ("60-89", "#C44E52"), ("90+", "#8172B2")):
        e = episodes[episodes["worst_bucket"] == bucket]
        km = KaplanMeierFitter().fit(e["months"], event_observed=e["end_type"] != "censored", label=f"Worst bucket {bucket}")
        km.plot_survival_function(ax=ax, color=c, ci_show=False)
        km_medians[bucket] = float(km.median_survival_time_)
    km_all = KaplanMeierFitter().fit(episodes["months"], event_observed=episodes["end_type"] != "censored")
    km_medians["all"] = float(km_all.median_survival_time_)
    ax.set(xlabel="Months since the account went 30+ DPD", ylabel="Share of episodes still unresolved",
           title="How long delinquency episodes last (Kaplan-Meier)", xlim=(0, 12))
    _save(fig, figure_dir / "delinquency_duration.png")
    out["episodes"] = {"count": int(len(episodes)), "censored": int((episodes["end_type"] == "censored").sum()),
                       "km_median_months": km_medians,
                       "end_types": episodes["end_type"].value_counts(normalize=True).round(4).to_dict()}

    # ------------------------------------------------------------------ monthly trends
    trends = an.monthly_trends(p)
    pause = [pd.Period(x, freq="M").to_timestamp(how="end") for x in a_cfg["periods"]["servicing_pause"]]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    axes[0].plot(trends.index, trends["dpd_30_plus_rate"], label="30+ DPD", color="#DD8452")
    axes[0].plot(trends.index, trends["dpd_60_plus_rate"], label="60+ DPD", color="#C44E52")
    axes[0].set(ylabel="Share of open loans", title="Delinquency rates by calendar month", ylim=(0, None))
    axes[1].plot(trends.index, trends["repo_per_60plus_account"], color="black")
    axes[1].set(ylabel="Repossessions per account 60+ DPD\nat the prior month end", ylim=(0, None),
                title="Repossession intensity: how quickly seriously delinquent accounts are repossessed")
    for ax in axes:
        ax.axvspan(pause[0] - pd.offsets.MonthBegin(1), pause[1], color="grey", alpha=0.2, label="Simulated servicing pause")
        ax.legend(fontsize=8)
    _save(fig, figure_dir / "monthly_trends.png")

    # ------------------------------------------------------------------ write-up
    out["cohorts"] = cohorts.round(5).reset_index().to_dict(orient="records")
    out["cure_by_worst_bucket"] = cures.round(4).reset_index().astype({"worst_bucket": str}).to_dict(orient="records")
    out["forward_3m"] = forward.round(4).reset_index().to_dict(orient="records")
    out["time_to_repossession"] = t2r.round(4).reset_index().to_dict(orient="records")
    out["roll_rates"] = rolls.round(5).reset_index().to_dict(orient="records")
    out["absorption"] = absorb.round(5).reset_index().to_dict(orient="records")
    pause_rows = trends.loc[(trends.index >= pause[0].normalize()) & (trends.index <= pause[1].normalize())]
    out["repo_intensity"] = {"during_pause": float(pause_rows["repo_per_60plus_account"].mean()),
                             "outside_pause": float(trends.drop(pause_rows.index)["repo_per_60plus_account"].mean())}

    (report_dir / "portfolio_metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (report_dir / "portfolio_report.md").write_text(
        _render(out, matrix, rolls, absorb, h12, comparison, checks, cohorts, cures, forward, t2r, trends), encoding="utf-8")
    logger.info("Wrote portfolio report with %d transitions and %d delinquency episodes", len(pairs), len(episodes))
    return out


def _render(out, matrix, rolls, absorb, h12, comparison, checks, cohorts, cures, forward, t2r, trends) -> str:
    lab = tr.STATE_LABELS
    m = matrix.loc[tr.OPEN_STATES].rename(index=lab, columns=lab)
    L = ["# Portfolio analytics report", "",
         "Written by `repo_risk report`. All numbers come from the run that produced this file.",
         "The portfolio is synthetic. Nothing here describes a real lender.", "",
         "## 1. Monthly transition matrix", "",
         f"Estimated from {out['n_transitions']:,} month-to-month moves. Rows are the state this month, "
         "columns the state next month. Closed states never change, so their rows are left out.", ""]
    L += _md_table(m, "{:.2%}", "From \\ To")
    L += ["", "## 2. Roll rates", "",
          "The same matrix summarized: share of accounts in each state that improved, stayed, got worse, or closed.", ""]
    L += _md_table(rolls.rename(index=lab), "{:.2%}", "State")
    L += ["", "## 3. What the Markov chain implies", "",
          "If the matrix held forever, this is where an account in each state would eventually end up, "
          "and its chance of repossession within 12 months.", ""]
    implied = absorb.rename(index=lab, columns=lab)
    implied["Repossessed within 12 months"] = h12.loc[tr.OPEN_STATES].to_numpy()
    L += _md_table(implied, "{:.2%}", "Starting state")

    pj = out["projection"]
    L += ["", "## 4. Markov projection against what happened", "",
          f"I took the {pj['open_loans_at_snapshot']:,} loans on the books at {pj['snapshot']}, estimated the matrix "
          f"from earlier months only, and projected {pj['months']} months ahead.", ""]
    comp = comparison.copy()
    comp["error_all_history"] = comp["markov_all_history"] - comp["actual"]
    comp["error_last_12_months"] = comp["markov_last_12_months"] - comp["actual"]
    L += _md_table(comp, {c: "{:+.2%}" if c.startswith("error") else "{:.2%}" for c in comp.columns},
                   f"State after {pj['months']} months")

    L += ["", "## 5. Checking the Markov assumptions", "", "### Same probabilities in every period?", "",
          "Chi-square test of whether each state's row is the same across periods. With this many rows nearly",
          "everything is significant, so Cramer's V (0 = identical, 1 = completely different) is the number to read.", "",
          "| From state | Transitions | Chi-square | p-value | Cramer's V |", "|---|---:|---:|---:|---:|"]
    for s, c in checks["time_homogeneity"].items():
        L.append(f"| {lab[s]} | {c['n']:,} | {c['chi2']:,.0f} | {c['p_value']:.1e} | {c['cramers_v']:.3f} |")
    probs = pd.DataFrame(checks["time_homogeneity"]["dpd_90_plus"]["probs_by_period"]).T.fillna(0.0)
    probs.columns = [lab.get(c, c) for c in probs.columns]
    L += ["", "Where accounts at 90+ DPD go next, by period:", ""] + _md_table(probs, "{:.2%}", "Period")
    L += ["", "### Does history matter?", "",
          "For accounts in the same state today, split by where they were last month.", "",
          "| State now | Last month | Improve next month | Roll forward next month |", "|---|---|---:|---:|"]
    for s, c in checks["memory"].items():
        for hist in c["improve_rate_by_history"]:
            L.append(f"| {lab[s]} | {hist} | {c['improve_rate_by_history'][hist]:.2%} | "
                     f"{c['roll_forward_rate_by_history'].get(hist, 0):.2%} |")
    L += ["", "| State now | Chi-square | Cramer's V |", "|---|---:|---:|"]
    L += [f"| {lab[s]} | {c['chi2']:,.0f} | {c['cramers_v']:.3f} |" for s, c in checks["memory"].items()]

    L += ["", "## 6. Vintage analysis", "",
          "Cumulative repossession rate by origination half-year at fixed ages. A blank means the cohort is too "
          "young for every loan to have reached that age.", ""]
    L += _md_table(cohorts, {"loans": "{:,.0f}", "observed_through_mob": "{:.0f}"}, "Cohort")

    L += ["", "## 7. Cures and delinquency duration", "",
          f"{out['episodes']['count']:,} episodes of 30+ DPD, of which {out['episodes']['censored']:,} were still "
          "open when the data ends. How resolved episodes ended, by the worst bucket reached:", ""]
    L += _md_table(cures, {"episodes": "{:,.0f}", "median_months": "{:.0f}"}, "Worst bucket")
    km = out["episodes"]["km_median_months"]
    L += ["", "Kaplan-Meier median episode length in months (unresolved episodes treated as censored): "
          + ", ".join(f"{k}: {v:.0f}" for k, v in km.items()) + ".", "",
          "Where delinquent accounts are three months later:", ""]
    L += _md_table(forward.rename(index=lab), {"accounts": "{:,.0f}"}, "State now")

    L += ["", "## 8. Repossession as a servicing outcome", "",
          "Loans that reached 60+ DPD, by servicing group. The groups share the same borrowers on average and "
          "differ only in how quickly the servicer assigns accounts for repossession.", ""]
    L += _md_table(t2r, {"loans_reaching_60dpd": "{:,.0f}", "median_months_60dpd_to_repo": "{:.0f}"}, "Servicing group")
    ri = out["repo_intensity"]
    L += ["", f"Repossessions per account 60+ DPD at the prior month end: {ri['during_pause']:.2%} during the simulated "
          f"servicing pause, {ri['outside_pause']:.2%} in other months.", "",
          "## 9. Monthly trend summary", ""]
    yearly = trends.groupby(trends.index.year).agg(
        avg_open_loans=("open_loans", "mean"), avg_30_plus=("dpd_30_plus_rate", "mean"),
        avg_60_plus=("dpd_60_plus_rate", "mean"), repossessions=("repossessions", "sum"),
        charge_offs=("charge_offs", "sum"), payoffs=("payoffs", "sum"))
    L += _md_table(yearly, {"avg_open_loans": "{:,.0f}", "repossessions": "{:,.0f}", "charge_offs": "{:,.0f}",
                            "payoffs": "{:,.0f}"}, "Year")
    return "\n".join(L) + "\n"
