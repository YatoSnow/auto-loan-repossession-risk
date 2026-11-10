"""Save and load simulated portfolios as parquet files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import SimConfig
from .generator import Portfolio

TABLES = ("loans", "payments", "monthly_performance", "loan_outcomes")


def save_portfolio(p: Portfolio, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in p.tables().items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    (out / "sim_config.json").write_text(json.dumps(p.config.to_dict(), indent=2), encoding="utf-8")
    return out


def load_portfolio(data_dir: str | Path) -> Portfolio:
    d = Path(data_dir)
    missing = [t for t in TABLES if not (d / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing tables {missing} in {d}. Run the data generation step first."
        )
    cfg = SimConfig.from_dict(json.loads((d / "sim_config.json").read_text(encoding="utf-8")))
    t = {name: pd.read_parquet(d / f"{name}.parquet") for name in TABLES}
    return Portfolio(t["loans"], t["payments"], t["monthly_performance"], t["loan_outcomes"], cfg)
