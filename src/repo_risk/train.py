"""Fit the four repossession-risk models on data before the test period.

Models, from simplest to most flexible:

* ``markov``      transition matrix raised to the horizon; uses only the current status
* ``logit_hazard`` multinomial logistic discrete-time hazard
* ``xgb_hazard``  gradient-boosted discrete-time hazard
* ``xgb_direct``  gradient-boosted yes/no classifier for "repossessed within H months",
                  fitted only on landmarks whose full window was observed

Everything is fitted on months up to ``train_end``. For the hazard models
that means follow-up is cut at ``train_end`` even if we know what happened
later. For the direct classifier it means dropping landmarks from the last
H months of the training period, because their label is not known yet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from .features import FEATURES, REPO, outcome_within
from .hazard import HazardModel, expand_person_period
from .transitions import STATES, horizon_probability, transition_matrix

logger = logging.getLogger(__name__)
MODEL_NAMES = ("markov", "logit_hazard", "xgb_hazard", "xgb_direct")
MODEL_LABELS = {
    "markov": "Markov chain",
    "logit_hazard": "Logistic hazard",
    "xgb_hazard": "XGBoost hazard",
    "xgb_direct": "XGBoost direct",
}


def month_ord(month: str) -> int:
    return pd.Period(month, freq="M").ordinal


def loan_bucket(loan_id: pd.Series) -> pd.Series:
    """Stable pseudo-random bucket 0-99 per loan, used for the holdout group."""
    return (loan_id.str[1:].astype("int64") * 2654435761 % 4294967296) % 100


def assign_blocks(lm: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    """Label each landmark: train, val (held-out loans, training period), test, recent, or None."""
    out = lm.copy()
    out["holdout_loan"] = loan_bucket(out["loan_id"]) < int(split_cfg["holdout_loan_pct"])
    t = out["t_ord"]
    block = pd.Series(None, index=out.index, dtype="object")
    in_train_period = t < month_ord(split_cfg["train_end"])
    block[in_train_period & ~out["holdout_loan"]] = "train"
    block[in_train_period & out["holdout_loan"]] = "val"
    block[(t >= month_ord(split_cfg["test_start"])) & (t <= month_ord(split_cfg["test_end"]))] = "test"
    block[t > month_ord(split_cfg["test_end"])] = "recent"
    out["block"] = block
    return out


@dataclass
class MarkovModel:
    matrix: pd.DataFrame

    def predict(self, lm: pd.DataFrame, horizon: int) -> np.ndarray:
        return lm["status"].map(horizon_probability(self.matrix, "repossessed", horizon)).to_numpy()


@dataclass
class DirectModel:
    model: XGBClassifier
    horizon: int
    params: dict[str, Any]

    def predict(self, lm: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(lm[FEATURES])[:, 1]


@dataclass
class ModelBundle:
    horizon: int
    markov: MarkovModel
    logit_hazard: HazardModel
    xgb_hazard: HazardModel
    xgb_direct: DirectModel

    def predict(self, lm: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
        """Predicted probability of repossession within each horizon, one column per model and horizon.

        The direct classifier only exists for the horizon it was trained on.
        """
        lm = lm.reset_index(drop=True)
        out = pd.DataFrame(index=lm.index)
        cif = {"logit_hazard": self.logit_hazard.predict_cif(lm), "xgb_hazard": self.xgb_hazard.predict_cif(lm)}
        for h in horizons:
            out[f"markov_{h}"] = self.markov.predict(lm, h)
            for name, curve in cif.items():
                out[f"{name}_{h}"] = curve[:, h - 1]
            if h == self.horizon:
                out[f"xgb_direct_{h}"] = self.xgb_direct.predict(lm)
        return out


def _sample_xgb(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "max_depth": int(rng.integers(2, 6)),
        "learning_rate": float(10 ** rng.uniform(-1.5, -0.7)),
        "min_child_weight": float(rng.choice([5, 20, 50])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(10 ** rng.uniform(-0.5, 1.5)),
    }


def train_models(lm: pd.DataFrame, pairs: pd.DataFrame, cfg: dict, out_dir: Path) -> ModelBundle:
    """Fit all four models. ``lm`` must have blocks assigned, ``pairs`` are monthly transitions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H = int(cfg["model"]["horizon_months"])
    end_ord = month_ord(cfg["split"]["train_end"])
    rng = np.random.default_rng(int(cfg["model"]["seed"]))
    tr = lm[lm["block"] == "train"].reset_index(drop=True)
    va = lm[lm["block"] == "val"].reset_index(drop=True)
    log: dict[str, Any] = {}

    # --- Markov chain: transitions that finished on or before train_end --------
    to_month = pairs["as_of_date"].dt.to_period("M").apply(lambda x: x.ordinal) + 1
    markov = MarkovModel(transition_matrix(pairs[to_month <= end_ord]))

    # --- hazard models on person-period data, follow-up cut at train_end -------
    pp_tr = expand_person_period(tr, H, last_ord=end_ord)
    pp_va = expand_person_period(va, H, last_ord=end_ord)
    log["person_period_rows"] = {"train": int(len(pp_tr)), "val": int(len(pp_va))}
    log["train_event_months"] = {str(k): int(v) for k, v in pp_tr["y"].value_counts().sort_index().items()}
    logger.info("Person-period rows: train %d, val %d", len(pp_tr), len(pp_va))

    logit = HazardModel("logit", H, {"C": float(cfg["model"]["logit_C"])}).fit(tr, pp_tr)
    logger.info("Fitted logistic hazard")

    best, trials = None, []
    for i in range(int(cfg["model"]["n_iter"]["xgb_hazard"])):
        params = _sample_xgb(rng)
        m = HazardModel("xgb", H, params).fit(tr, pp_tr, va, pp_va)
        trials.append({"model": "xgb_hazard", "trial": i, "val_mlogloss": m.best_score_, **m.params})
        logger.info("xgb_hazard trial %d: val mlogloss %.5f", i + 1, m.best_score_)
        if best is None or m.best_score_ < best.best_score_:
            best = m
    xgb_hazard = best

    # --- direct classifier: only landmarks with a fully observed window --------
    def complete(block: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        o = outcome_within(block, H, last_ord=end_ord)
        keep = o["complete"].to_numpy()
        return block[keep], (o.loc[keep, "event"] == REPO).astype(int).to_numpy()

    (tr_c, y_tr), (va_c, y_va) = complete(tr), complete(va)
    log["direct_rows"] = {"train": int(len(tr_c)), "train_dropped_incomplete": int(len(tr) - len(tr_c)),
                          "train_event_rate": float(y_tr.mean())}
    best_d = None
    for i in range(int(cfg["model"]["n_iter"]["xgb_direct"])):
        params = _sample_xgb(rng)
        clf = XGBClassifier(n_estimators=800, early_stopping_rounds=40, tree_method="hist", eval_metric="logloss",
                            n_jobs=-1, random_state=0, **params)
        clf.fit(tr_c[FEATURES], y_tr, eval_set=[(va_c[FEATURES], y_va)], verbose=False)
        score = average_precision_score(y_va, clf.predict_proba(va_c[FEATURES])[:, 1])
        params = {**params, "n_estimators": int(clf.best_iteration) + 1}
        trials.append({"model": "xgb_direct", "trial": i, "val_pr_auc": score, **params})
        logger.info("xgb_direct trial %d: val PR-AUC %.4f", i + 1, score)
        if best_d is None or score > best_d[0]:
            best_d = (score, DirectModel(clf, H, params))

    bundle = ModelBundle(H, markov, logit, xgb_hazard, best_d[1])
    joblib.dump(bundle, out_dir / "models.joblib")
    pd.DataFrame(trials).to_csv(out_dir / "search_trials.csv", index=False)
    log["params"] = {"logit_hazard": logit.params, "xgb_hazard": xgb_hazard.params, "xgb_direct": best_d[1].params}
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    assert list(markov.matrix.index) == STATES
    return bundle


def load_bundle(model_dir: Path) -> ModelBundle:
    path = model_dir / "models.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the train step first.")
    return joblib.load(path)
