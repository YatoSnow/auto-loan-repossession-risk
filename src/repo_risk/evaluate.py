"""Evaluate the repossession-risk models.

Two evaluations, kept apart on purpose:

1. **Complete-window test.** Landmarks from the test period whose full
   horizon was observed before the data ends. Every one of these has a
   known yes/no answer, so ROC-AUC, PR-AUC and Brier score are valid.
   Loans that paid off inside the window are known non-repossessions.

2. **Censored check.** Landmarks from the most recent months, where the
   window runs past the end of the data. Ordinary classification metrics
   would be wrong here: counting "not repossessed yet" as "not
   repossessed" pulls the observed rate down. Instead the mean predicted
   probability is compared with the Aalen-Johansen estimate, which
   handles censoring and competing exits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .features import NO_EVENT, REPO, outcome_within
from .hazard import aalen_johansen
from .train import MODEL_LABELS, MODEL_NAMES, ModelBundle

logger = logging.getLogger(__name__)
COLORS = {"markov": "#8C8C8C", "logit_hazard": "#4C72B0", "xgb_hazard": "#C44E52", "xgb_direct": "#55A868"}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p, kind="stable")
    return float(sum(abs(y[i].mean() - p[i].mean()) * len(i) for i in np.array_split(order, bins) if len(i)) / len(y))


def summary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    out = {"n": int(len(y)), "events": int(y.sum()), "event_rate": float(y.mean()), "mean_predicted": float(p.mean()),
           "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p)}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    return out


def bootstrap_ci(y, scores: dict[str, np.ndarray], groups, n_boot: int, seed: int = 0) -> dict:
    """Bootstrap by loan, since a loan contributes several landmarks."""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(np.asarray(groups))
    members = pd.Series(np.arange(len(y))).groupby(codes).apply(np.array).to_numpy()
    draws = {m: {"roc_auc": [], "pr_auc": [], "brier": []} for m in scores}
    for _ in range(n_boot):
        idx = np.concatenate(members[rng.integers(0, len(members), len(members))])
        if y[idx].min() == y[idx].max():
            continue
        for m, s in scores.items():
            draws[m]["roc_auc"].append(roc_auc_score(y[idx], s[idx]))
            draws[m]["pr_auc"].append(average_precision_score(y[idx], s[idx]))
            draws[m]["brier"].append(brier_score_loss(y[idx], s[idx]))
    return {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
            for m, d in draws.items()}


def group_table(df: pd.DataFrame, by: str, y: str, pred_cols: dict[str, str]) -> list[dict]:
    """Observed rate next to each model's mean prediction, for each group."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        row = {by: str(key), "n": int(len(g)), "observed": float(g[y].mean())}
        row.update({m: float(g[c].mean()) for m, c in pred_cols.items()})
        rows.append(row)
    return rows


def evaluate_models(lm: pd.DataFrame, bundle: ModelBundle, cfg: dict, report_dir: Path, figure_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    H = bundle.horizon
    horizons = [int(h) for h in cfg["evaluation"]["horizons"]]
    n_boot = int(cfg["evaluation"]["n_boot"])
    results: dict = {"horizon_months": H, "horizons": horizons, "complete_window_test": {}, "censored_check": {}}

    # ------------------------------------------------------------------
    # 1. Complete-window test
    # ------------------------------------------------------------------
    test = lm[lm["block"] == "test"].reset_index(drop=True)
    pred = bundle.predict(test, horizons)
    for h in horizons:
        o = outcome_within(test, h)
        if not o["complete"].all():
            raise ValueError(f"Test landmarks do not all have a complete {h}-month window. Check the split dates.")
        y = (o["event"] == REPO).to_numpy().astype(int)
        names = [m for m in MODEL_NAMES if f"{m}_{h}" in pred]
        scores = {m: pred[f"{m}_{h}"].to_numpy() for m in names}
        block = {m: summary_metrics(y, s) for m, s in scores.items()}
        if h == H:
            ci = bootstrap_ci(y, scores, test["loan_id"].to_numpy(), n_boot)
            for m in names:
                block[m]["ci95"] = ci[m]
                unseen = test["holdout_loan"].to_numpy()
                block[m]["unseen_loans"] = summary_metrics(y[unseen], scores[m][unseen])
            results["exit_mix_in_window"] = {
                name: float((o["event"] == code).mean())
                for code, name in ((REPO, "repossessed"), (2, "paid_off"), (3, "charged_off"), (NO_EVENT, "still_open"))
            }
            frame = test.assign(y=y, **{f"p_{m}": s for m, s in scores.items()})
            frame["origination_year"] = frame["origination_date"].dt.year
            frame["status_label"] = frame["status"]
            cols = {m: f"p_{m}" for m in names}
            results["by_origination_year"] = group_table(frame, "origination_year", "y", cols)
            results["by_servicing_group"] = group_table(frame, "servicing_group", "y", cols)
            results["by_status"] = group_table(frame, "status_label", "y", cols)
            _plot_calibration(y, scores, H, figure_dir / "calibration_12m.png")
            _plot_cohorts(results["by_origination_year"], names, figure_dir / "cohort_predicted_vs_observed.png")
            _plot_risk_distribution(frame, figure_dir / "risk_distribution.png")
        results["complete_window_test"][str(h)] = block
    results["test_period"] = {"first": str(test["landmark_date"].min().date()),
                              "last": str(test["landmark_date"].max().date()), "landmarks": int(len(test)),
                              "loans": int(test["loan_id"].nunique())}

    # ------------------------------------------------------------------
    # 2. Censored check on the most recent landmarks
    # ------------------------------------------------------------------
    recent = lm[lm["block"] == "recent"].reset_index(drop=True)
    h_c = int(cfg["evaluation"]["censored_horizon"])
    if len(recent):
        pred_r = bundle.predict(recent, [h_c])
        time = np.where(recent["months_to_exit"].notna() & (recent["months_to_exit"] <= recent["months_observed"]),
                        recent["months_to_exit"], recent["months_observed"]).astype(float)
        event = np.where(recent["months_to_exit"].notna() & (recent["months_to_exit"] <= recent["months_observed"]),
                         recent["exit_code"], NO_EVENT)
        o = outcome_within(recent, h_c)
        naive = float((o["event"] == REPO).mean())  # wrong on purpose: treats censored as "no event"
        complete_case = float((o.loc[o["complete"], "event"] == REPO).mean())
        aj = aalen_johansen(time, event, h_c)
        check = {
            "horizon_months": h_c,
            "landmarks": int(len(recent)),
            "share_with_incomplete_window": float(1 - o["complete"].mean()),
            "aalen_johansen": float(aj["cif"].iloc[-1]),
            "naive_censored_as_zero": naive,
            "complete_windows_only": complete_case,
            "complete_windows_n": int(o["complete"].sum()),
            "mean_predicted": {m: float(pred_r[f"{m}_{h_c}"].mean()) for m in ("markov", "logit_hazard", "xgb_hazard")},
            "by_risk_tercile": [],
        }
        tercile = pd.qcut(pred_r[f"xgb_hazard_{h_c}"].rank(method="first"), 3, labels=["low", "middle", "high"])
        for name in ("low", "middle", "high"):
            mask = (tercile == name).to_numpy()
            aj_g = aalen_johansen(time[mask], event[mask], h_c)
            check["by_risk_tercile"].append({
                "tercile": name, "n": int(mask.sum()), "aalen_johansen": float(aj_g["cif"].iloc[-1]),
                "logit_hazard": float(pred_r.loc[mask, f"logit_hazard_{h_c}"].mean()),
                "xgb_hazard": float(pred_r.loc[mask, f"xgb_hazard_{h_c}"].mean()),
            })
        results["censored_check"] = check
        _plot_censored(aj, bundle, recent, h_c, figure_dir / "censored_check.png")

    _plot_horizon_metrics(results, figure_dir / "metrics_by_horizon.png")
    (report_dir / "model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (report_dir / "model_evaluation.md").write_text(render_report(results), encoding="utf-8")
    return results


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(y, scores, h, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for m, p in scores.items():
        if m == "markov":  # only five distinct values, so plot one point per status
            d = pd.DataFrame({"p": p, "y": y}).groupby("p")["y"].mean()
            ax.plot(d.index, d.values, "s", color=COLORS[m], label=MODEL_LABELS[m], ms=7)
            continue
        order = np.argsort(p, kind="stable")
        parts = np.array_split(order, 10)
        ax.plot([p[i].mean() for i in parts], [y[i].mean() for i in parts], "o-", color=COLORS[m],
                label=MODEL_LABELS[m], ms=4)
    ax.plot([1e-4, 1], [1e-4, 1], "k--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="Mean predicted probability", ylabel="Observed repossession rate",
           title=f"Calibration, repossession within {h} months (test landmarks)")
    ax.legend(fontsize=8)
    _save(fig, path)


def _plot_cohorts(rows, names, path):
    d = pd.DataFrame(rows).set_index("origination_year")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(d))
    width = 0.8 / (len(names) + 1)
    ax.bar(x, d["observed"], width, label="Observed", color="black")
    for i, m in enumerate(names, start=1):
        ax.bar(x + i * width, d[m], width, label=MODEL_LABELS[m], color=COLORS[m])
    ax.set_xticks(x + width * len(names) / 2, d.index)
    ax.set(xlabel="Origination year", ylabel="Repossessed within 12 months",
           title="Predicted vs observed repossession rate by origination year (test landmarks)")
    ax.legend(fontsize=8)
    _save(fig, path)


def _plot_risk_distribution(frame, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.logspace(-4, 0, 45)
    for status, color in (("current", "#4C72B0"), ("dpd_1_29", "#55A868"), ("dpd_30_59", "#DD8452"),
                          ("dpd_60_89", "#C44E52"), ("dpd_90_plus", "#8172B2")):
        p = frame.loc[frame["status"] == status, "p_xgb_hazard"].clip(lower=1e-4)
        ax.hist(p, bins=bins, alpha=0.6, label=f"{status} (n={len(p):,})", color=color)
    ax.set(xscale="log", yscale="log", xlabel="Predicted probability of repossession within 12 months",
           ylabel="Landmarks", title="Distribution of predicted risk by current status (XGBoost hazard)")
    ax.legend(fontsize=8)
    _save(fig, path)


def _plot_censored(aj, bundle, recent, h, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.step(aj["month"], aj["cif"], where="post", color="black", label="Aalen-Johansen (observed, censoring handled)")
    for name, model in (("logit_hazard", bundle.logit_hazard), ("xgb_hazard", bundle.xgb_hazard)):
        curve = model.predict_cif(recent)[:, :h].mean(axis=0)
        ax.plot(np.arange(1, h + 1), curve, "o-", ms=3, color=COLORS[name], label=f"{MODEL_LABELS[name]} (mean prediction)")
    ax.set(xlabel="Months after landmark", ylabel="Cumulative repossession probability", ylim=(0, None),
           title="Most recent landmarks: predicted vs observed cumulative incidence")
    ax.legend(fontsize=8)
    _save(fig, path)


def _plot_horizon_metrics(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric, title in zip(axes, ("roc_auc", "pr_auc", "brier"), ("ROC-AUC", "PR-AUC", "Brier score"), strict=True):
        for m in MODEL_NAMES:
            pts = [(int(h), b[m][metric]) for h, b in results["complete_window_test"].items() if m in b and metric in b[m]]
            if pts:
                ax.plot(*zip(*pts, strict=True), "o-" if len(pts) > 1 else "D", color=COLORS[m], label=MODEL_LABELS[m])
        ax.set(xlabel="Horizon (months)", title=title)
        ax.legend(fontsize=7)
    fig.suptitle("Complete-window test metrics by horizon", y=1.03)
    _save(fig, path)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def render_report(r: dict) -> str:
    H = r["horizon_months"]
    tp = r["test_period"]
    L = ["# Model evaluation report", "",
         "Written by `repo_risk evaluate`. All numbers come from the run that produced this file.",
         "The data is synthetic. These numbers do not describe real loans.", "",
         f"## 1. Complete-window test ({tp['first'][:7]} to {tp['last'][:7]})", "",
         f"{tp['landmarks']:,} landmarks from {tp['loans']:,} loans. Every landmark has its full window observed, "
         "so each one has a known yes/no outcome.", ""]
    mix = r["exit_mix_in_window"]
    L += [f"What happened within {H} months: repossessed {mix['repossessed']:.2%}, paid off {mix['paid_off']:.2%}, "
          f"charged off {mix['charged_off']:.2%}, still open {mix['still_open']:.2%}.", ""]
    for h in r["horizons"]:
        block = r["complete_window_test"][str(h)]
        L += [f"### Repossession within {h} months", "",
              "| Model | ROC-AUC | PR-AUC | Brier | ECE | Mean predicted | Observed |", "|---|---:|---:|---:|---:|---:|---:|"]
        for m in MODEL_NAMES:
            if m not in block:
                continue
            b = block[m]
            ci = b.get("ci95")
            auc = f"{b['roc_auc']:.3f}" + (f" ({ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f})" if ci else "")
            pr = f"{b['pr_auc']:.3f}" + (f" ({ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f})" if ci else "")
            L.append(f"| {MODEL_LABELS[m]} | {auc} | {pr} | {b['brier']:.4f} | {b['ece']:.4f} | "
                     f"{b['mean_predicted']:.2%} | {b['event_rate']:.2%} |")
        L.append("")
    L += [f"### Loans never used for fitting ({H} months)", "", "| Model | n | ROC-AUC | PR-AUC | Brier |", "|---|---:|---:|---:|---:|"]
    for m in MODEL_NAMES:
        u = r["complete_window_test"][str(H)][m]["unseen_loans"]
        L.append(f"| {MODEL_LABELS[m]} | {u['n']:,} | {u['roc_auc']:.3f} | {u['pr_auc']:.3f} | {u['brier']:.4f} |")
    for key, title, col in (("by_origination_year", "origination year", "origination_year"),
                            ("by_servicing_group", "servicing group", "servicing_group"),
                            ("by_status", "status at the landmark", "status_label")):
        L += ["", f"### Predicted vs observed by {title} ({H} months)", "",
              "| Group | n | Observed | " + " | ".join(MODEL_LABELS[m] for m in MODEL_NAMES) + " |",
              "|---|---:|---:|" + "---:|" * len(MODEL_NAMES)]
        for row in r[key]:
            L.append(f"| {row[col]} | {row['n']:,} | {row['observed']:.2%} | "
                     + " | ".join(f"{row[m]:.2%}" for m in MODEL_NAMES) + " |")
    c = r.get("censored_check")
    if c:
        L += ["", f"## 2. Censored check on the most recent landmarks ({c['horizon_months']} months)", "",
              f"{c['landmarks']:,} landmarks, {c['share_with_incomplete_window']:.0%} of them with a window that runs "
              "past the end of the data.", "",
              "| Estimate of the repossession rate | Value |", "|---|---:|",
              f"| Aalen-Johansen (handles censoring and competing exits) | {c['aalen_johansen']:.2%} |",
              f"| Naive: count censored landmarks as 'not repossessed' | {c['naive_censored_as_zero']:.2%} |",
              f"| Complete windows only (n = {c['complete_windows_n']:,}) | {c['complete_windows_only']:.2%} |"]
        L += [f"| Mean prediction, {MODEL_LABELS[m]} | {v:.2%} |" for m, v in c["mean_predicted"].items()]
        L += ["", "By tercile of predicted risk (XGBoost hazard):", "",
              "| Tercile | n | Aalen-Johansen | Logistic hazard | XGBoost hazard |", "|---|---:|---:|---:|---:|"]
        L += [f"| {t['tercile']} | {t['n']:,} | {t['aalen_johansen']:.2%} | {t['logit_hazard']:.2%} | {t['xgb_hazard']:.2%} |"
              for t in c["by_risk_tercile"]]
    return "\n".join(L) + "\n"
