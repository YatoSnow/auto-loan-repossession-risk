import matplotlib

matplotlib.use("Agg")

import pytest

from autoloan_sim import SimConfig, simulate_portfolio


@pytest.fixture(scope="session")
def portfolio():
    """A small simulated portfolio shared by all tests."""
    return simulate_portfolio(SimConfig(n_loans=2500, seed=11))


@pytest.fixture(scope="session")
def cfg():
    return {
        "model": {"horizon_months": 12, "sample_every": 3, "seed": 0, "logit_C": 1.0,
                  "n_iter": {"xgb_hazard": 1, "xgb_direct": 1}},
        "split": {"train_end": "2022-12", "test_start": "2023-01", "test_end": "2023-12", "holdout_loan_pct": 15},
        "evaluation": {"horizons": [3, 12], "censored_horizon": 9, "n_boot": 10},
        "analytics": {
            "projection_snapshot": "2023-06",
            "projection_months": 12,
            "periods": {"before_pause": ["2019-02", "2020-03"], "servicing_pause": ["2020-04", "2020-07"],
                        "after_2020": ["2020-08", "2022-12"], "recent": ["2023-01", "2024-12"]},
        },
    }
