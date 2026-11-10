"""Configuration for the synthetic auto loan portfolio simulator.

Every number that shapes the simulated portfolio lives here, so the
assumptions are in one place and can be changed from a YAML file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SimConfig:
    # ---- size, seed and calendar -------------------------------------
    n_loans: int = 30_000
    seed: int = 42
    first_origination: str = "2019-01"
    last_origination: str = "2023-12"
    data_cutoff: str = "2024-12"  # last month with observed performance

    # ---- borrowers ----------------------------------------------------
    repeat_borrower_share: float = 0.06  # share of loans that belong to a borrower with another loan
    score_mean: float = 610.0
    score_sd: float = 58.0
    score_latent_corr: float = 0.60  # how much the score knows about the hidden stability trait
    income_log_mean: float = 8.25  # exp(8.25) ~ $3,800 / month
    income_log_sd: float = 0.38
    habitual_late_share: float = 0.12  # pays late in the month but rarely misses
    irregular_income_share: float = 0.10  # payment amounts move around month to month

    # ---- underwriting -------------------------------------------------
    used_share: float = 0.75
    max_pti: float = 0.22
    loose_vintage_start: str = "2021-07"  # a stretch of looser underwriting, for vintage analysis
    loose_vintage_end: str = "2022-06"
    loose_vintage_shift: float = 0.30  # shift (in SD units) of the hidden stability trait

    # ---- hidden financial-stress regime --------------------------------
    stress_base_logit: float = -4.70
    stress_z_coef: float = 0.75
    stress_pti_coef: float = 0.30
    stress_ltv_coef: float = 0.15
    stress_early_life_bump: float = 0.35  # months on book <= 6
    stress_exit_prob: float = 0.22
    macro_ar: float = 0.85
    macro_sd: float = 0.12
    macro_shock_start: str = "2020-04"
    macro_shock_end: str = "2020-06"
    macro_shock_size: float = 0.50

    # ---- voluntary prepayment ------------------------------------------
    prepay_base: float = 0.004
    prepay_age_slope: float = 0.00025  # per month on book, capped at 36 months

    # ---- servicing: extensions, repossession, charge-off ---------------
    extension_prob: float = 0.04
    max_extensions: int = 2
    repo_assign_probs: dict[str, list[float]] = field(
        default_factory=lambda: {
            # monthly probability of assigning the account for repossession,
            # by servicing group, for 60-89 / 90-119 / 120+ days past due
            "A": [0.10, 0.30, 0.40],
            "B": [0.05, 0.25, 0.35],
            "C": [0.02, 0.15, 0.30],
        }
    )
    servicing_group_shares: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    repo_recovery_prob: float = 0.60  # chance per month that an assigned vehicle is recovered
    skip_share: float = 0.08  # vehicles that are never located
    repo_capacity_sd: float = 0.20  # month-to-month noise in repossession capacity
    repo_pause_start: str = "2020-04"  # simulated servicing pause
    repo_pause_end: str = "2020-07"
    repo_pause_factor: float = 0.05
    voluntary_surrender_prob: float = 0.012
    chargeoff_prob: float = 0.45  # per month once 150+ days past due

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SimConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown simulator settings: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        # project configs keep simulator settings under a "simulation" key
        return cls.from_dict(raw.get("simulation", raw))
