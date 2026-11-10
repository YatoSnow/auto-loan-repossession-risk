"""Discrete-time competing-risk hazard models for repossession.

Idea
----
Stand at a landmark month with an open loan. In each following month
``k = 1, 2, ...`` exactly one of four things happens: the loan stays open,
is repossessed, pays off, or is charged off. A hazard model estimates the
probability of each, given that the loan was still open at the start of
month ``k``.

To fit it, every landmark is expanded into one row per month it was
followed ("person-period" data). A loan that is still open when the data
ends simply stops contributing rows. That is how right censoring is
handled: the loan counts as "survived" for the months we saw and says
nothing about the months we did not see.

The probability of repossession within ``H`` months is then

    CIF(H) = sum over k <= H of  S(k-1) * h_repo(k)

where ``S(k-1)`` is the probability of still being open after ``k-1``
months. Payoff and charge-off lower ``S`` and so lower the repossession
probability. This is the competing-risks cumulative incidence, not
``1 - product(1 - h_repo)``, which would overstate it.

Covariates are frozen at the landmark. The model does not try to forecast
how the features will move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURES, NO_EVENT, REPO

N_CLASSES = 4


def expand_person_period(lm: pd.DataFrame, horizon: int, last_ord: int | None = None) -> pd.DataFrame:
    """One row per landmark per follow-up month.

    ``last_ord`` truncates follow-up at a calendar month (used to keep the
    training data strictly before the test period). Columns: ``row`` (index
    into ``lm``), ``k`` (months after the landmark) and ``y`` (exit code in
    that month, 0 if the loan stayed open).
    """
    observed = lm["months_observed"].to_numpy(dtype=float)
    if last_ord is not None:
        observed = np.minimum(observed, last_ord - lm["t_ord"].to_numpy())
    to_exit = lm["months_to_exit"].to_numpy(dtype=float)
    follow = np.minimum(np.minimum(horizon, observed), np.where(np.isnan(to_exit), np.inf, to_exit))
    follow = np.clip(follow, 0, None).astype(int)

    row = np.repeat(np.arange(len(lm)), follow)
    first = np.cumsum(follow) - follow
    k = np.arange(follow.sum()) - np.repeat(first, follow) + 1
    exit_code = lm["exit_code"].to_numpy()
    y = np.where(k == to_exit[row], exit_code[row], NO_EVENT).astype(int)
    return pd.DataFrame({"row": row, "k": k, "y": y})


def cumulative_incidence(hazards: np.ndarray) -> dict[str, np.ndarray]:
    """Turn monthly hazards of shape (n, H, 4) into cumulative incidence curves.

    Returns ``repo`` with shape (n, H): probability of repossession within
    1..H months, and ``any_exit`` with the same shape.
    """
    stay = hazards[:, :, NO_EVENT]
    surv_before = np.concatenate([np.ones((len(stay), 1)), np.cumprod(stay, axis=1)[:, :-1]], axis=1)
    repo = np.cumsum(surv_before * hazards[:, :, REPO], axis=1)
    return {"repo": repo, "any_exit": 1.0 - np.cumprod(stay, axis=1)}


@dataclass
class HazardModel:
    """Common interface for the two hazard models."""

    kind: str  # "logit" or "xgb"
    horizon: int
    params: dict[str, Any] = field(default_factory=dict)
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    model: Any = None
    scaler: StandardScaler | None = None
    classes_: np.ndarray | None = None

    # ---- design matrix ---------------------------------------------------
    def _design(self, X: np.ndarray, k: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.kind == "xgb":
            return np.column_stack([X, k])
        if fit:
            self.scaler = StandardScaler().fit(X)
        # one indicator per follow-up month gives a free-form baseline hazard
        month = (k[:, None] == np.arange(2, self.horizon + 1)[None, :]).astype(float)
        return np.column_stack([self.scaler.transform(X), month])

    # ---- fit -------------------------------------------------------------
    def fit(self, lm: pd.DataFrame, pp: pd.DataFrame, lm_val: pd.DataFrame | None = None,
            pp_val: pd.DataFrame | None = None) -> "HazardModel":
        X = lm[self.features].to_numpy(dtype=float)
        D = self._design(X[pp["row"].to_numpy()], pp["k"].to_numpy(), fit=True)
        self.classes_, y = np.unique(pp["y"].to_numpy(), return_inverse=True)
        if self.kind == "logit":
            self.model = LogisticRegression(C=self.params.get("C", 1.0), max_iter=300)
            self.model.fit(D, y)
            return self
        fixed = "n_estimators" in self.params
        self.model = XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            n_estimators=self.params["n_estimators"] if fixed else 600,
            early_stopping_rounds=None if fixed or pp_val is None else 30,
            n_jobs=-1,
            random_state=0,
            **{k: v for k, v in self.params.items() if k != "n_estimators"},
        )
        if not fixed and pp_val is not None:
            Xv = lm_val[self.features].to_numpy(dtype=float)
            Dv = self._design(Xv[pp_val["row"].to_numpy()], pp_val["k"].to_numpy())
            known = np.isin(pp_val["y"], self.classes_)
            yv = np.searchsorted(self.classes_, pp_val["y"].to_numpy()[known])
            self.model.fit(D, y, eval_set=[(Dv[known], yv)], verbose=False)
            self.params = {**self.params, "n_estimators": int(self.model.best_iteration) + 1}
            self.best_score_ = float(self.model.best_score)
        else:
            self.model.fit(D, y)
        return self

    # ---- predict ---------------------------------------------------------
    def hazards(self, lm: pd.DataFrame, batch: int = 50_000) -> np.ndarray:
        """Monthly hazards with shape (n, horizon, 4) for landmarks in ``lm``."""
        X = lm[self.features].to_numpy(dtype=float)
        out = np.zeros((len(X), self.horizon, N_CLASSES))
        ks = np.arange(1, self.horizon + 1)
        for lo in range(0, len(X), batch):
            xb = X[lo : lo + batch]
            D = self._design(np.repeat(xb, self.horizon, axis=0), np.tile(ks, len(xb)))
            prob = self.model.predict_proba(D).reshape(len(xb), self.horizon, -1)
            out[lo : lo + batch][:, :, self.classes_] = prob
        return out

    def predict_cif(self, lm: pd.DataFrame) -> np.ndarray:
        """Probability of repossession within 1..horizon months, shape (n, horizon)."""
        return cumulative_incidence(self.hazards(lm))["repo"]

    def coefficients(self) -> pd.DataFrame:
        """Standardized coefficients of the logistic hazard, one column per exit type."""
        if self.kind != "logit":
            raise ValueError("Only the logistic hazard has coefficients.")
        names = self.features + [f"month_{k}" for k in range(2, self.horizon + 1)]
        coef = self.model.coef_
        if coef.shape[0] == 1:  # binary case in very small samples
            coef = np.vstack([-coef, coef])
        # multinomial coefficients are identified up to a shift; report them relative to "stay open"
        rel = coef - coef[list(self.classes_).index(NO_EVENT)]
        cols = {int(c): rel[i] for i, c in enumerate(self.classes_) if c != NO_EVENT}
        return pd.DataFrame(cols, index=names)


def aalen_johansen(time: np.ndarray, event: np.ndarray, horizon: int, cause: int = REPO) -> pd.DataFrame:
    """Non-parametric cumulative incidence for discrete time with competing risks.

    ``time`` is months until exit or censoring, ``event`` is the exit code
    (0 = censored). Subjects censored at month ``t`` are counted as at risk
    through month ``t``.
    """
    time, event = np.asarray(time), np.asarray(event)
    surv, cif, rows = 1.0, 0.0, []
    for k in range(1, horizon + 1):
        at_risk = int((time >= k).sum())
        if at_risk == 0:
            break
        d_cause = int(((time == k) & (event == cause)).sum())
        d_any = int(((time == k) & (event != NO_EVENT)).sum())
        cif += surv * d_cause / at_risk
        surv *= 1.0 - d_any / at_risk
        rows.append({"month": k, "at_risk": at_risk, "events": d_cause, "cif": cif, "survival": surv})
    return pd.DataFrame(rows)
