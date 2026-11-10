"""Command line interface.

    python -m repo_risk.cli run-all     # everything, in order
    python -m repo_risk.cli generate    # simulate the portfolio
    python -m repo_risk.cli report      # transition matrices and portfolio analytics
    python -m repo_risk.cli landmarks   # build the modeling table
    python -m repo_risk.cli train       # fit the four models
    python -m repo_risk.cli evaluate    # test metrics and censored check
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

from autoloan_sim import SimConfig, load_portfolio, save_portfolio, simulate_portfolio, validate_portfolio

logger = logging.getLogger("repo_risk")
DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: Path(v) for k, v in cfg["paths"].items()}
    return cfg


def _landmark_path(cfg) -> Path:
    return cfg["paths"]["processed"] / "landmarks.parquet"


def cmd_generate(cfg) -> None:
    portfolio = simulate_portfolio(SimConfig.from_dict(cfg.get("simulation", {})))
    problems = validate_portfolio(portfolio)
    if problems:
        raise SystemExit("Simulated data failed validation: " + "; ".join(problems))
    save_portfolio(portfolio, cfg["paths"]["raw"])
    logger.info("Saved %d loans to %s", len(portfolio.loans), cfg["paths"]["raw"])


def cmd_report(cfg) -> None:
    from .report import build_report

    build_report(load_portfolio(cfg["paths"]["raw"]), cfg, cfg["paths"]["reports"], cfg["paths"]["figures"])


def cmd_landmarks(cfg) -> None:
    from .features import build_landmarks
    from .train import assign_blocks

    lm = build_landmarks(load_portfolio(cfg["paths"]["raw"]), int(cfg["model"]["sample_every"]))
    lm = assign_blocks(lm, cfg["split"])
    cfg["paths"]["processed"].mkdir(parents=True, exist_ok=True)
    lm.to_parquet(_landmark_path(cfg), index=False)
    logger.info("Blocks: %s", lm["block"].value_counts(dropna=False).to_dict())


def _load_landmarks(cfg) -> pd.DataFrame:
    path = _landmark_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the 'landmarks' step first.")
    return pd.read_parquet(path)


def cmd_train(cfg) -> None:
    from .train import train_models
    from .transitions import transition_pairs

    pairs = transition_pairs(load_portfolio(cfg["paths"]["raw"]).monthly)
    train_models(_load_landmarks(cfg), pairs, cfg, cfg["paths"]["models"])


def cmd_evaluate(cfg) -> None:
    from .evaluate import evaluate_models
    from .train import MODEL_LABELS, load_bundle

    bundle = load_bundle(cfg["paths"]["models"])
    results = evaluate_models(_load_landmarks(cfg), bundle, cfg, cfg["paths"]["reports"], cfg["paths"]["figures"])
    for name, r in results["complete_window_test"][str(bundle.horizon)].items():
        logger.info("%-16s ROC-AUC %.3f  PR-AUC %.3f  Brier %.4f  predicted %.2f%% vs observed %.2f%%",
                    MODEL_LABELS[name], r["roc_auc"], r["pr_auc"], r["brier"], 100 * r["mean_predicted"],
                    100 * r["event_rate"])


COMMANDS = {"generate": cmd_generate, "report": cmd_report, "landmarks": cmd_landmarks, "train": cmd_train,
            "evaluate": cmd_evaluate}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repo_risk", description="Repossession risk and delinquency transitions.")
    parser.add_argument("command", choices=[*COMMANDS, "run-all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    matplotlib.use("Agg")  # write figures to files, never open a window
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    steps = list(COMMANDS) if args.command == "run-all" else [args.command]
    for step in steps:
        logger.info("=== %s ===", step)
        COMMANDS[step](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
