"""Command line entry point: ``python -m autoloan_sim.cli --out data/raw``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import SimConfig
from .generator import simulate_portfolio
from .io import save_portfolio
from .schema import render_data_dictionary
from .validation import portfolio_summary, validate_portfolio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic auto loan portfolio.")
    parser.add_argument("--config", type=Path, help="YAML file with simulator settings")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="output folder")
    parser.add_argument("--n-loans", type=int, help="override the number of loans")
    parser.add_argument("--seed", type=int, help="override the random seed")
    parser.add_argument("--write-dictionary", type=Path, help="also write the data dictionary to this file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = SimConfig.from_yaml(args.config) if args.config else SimConfig()
    if args.n_loans:
        cfg.n_loans = args.n_loans
    if args.seed is not None:
        cfg.seed = args.seed

    portfolio = simulate_portfolio(cfg)
    problems = validate_portfolio(portfolio)
    if problems:
        for msg in problems:
            logging.error("Validation failed: %s", msg)
        return 1
    save_portfolio(portfolio, args.out)
    logging.info("Saved tables to %s", args.out)
    print(json.dumps(portfolio_summary(portfolio), indent=2))
    if args.write_dictionary:
        args.write_dictionary.write_text(render_data_dictionary(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
