"""Synthetic auto loan portfolio simulator shared by the three portfolio projects."""

from .config import SimConfig
from .generator import (
    ACTIVE_STATUSES,
    STATUS_NAMES,
    TERMINAL_STATUSES,
    Portfolio,
    amortized_payment,
    simulate_portfolio,
)
from .io import load_portfolio, save_portfolio
from .validation import portfolio_summary, validate_portfolio

__version__ = "0.1.0"

__all__ = [
    "ACTIVE_STATUSES",
    "STATUS_NAMES",
    "TERMINAL_STATUSES",
    "Portfolio",
    "SimConfig",
    "amortized_payment",
    "load_portfolio",
    "portfolio_summary",
    "save_portfolio",
    "simulate_portfolio",
    "validate_portfolio",
    "__version__",
]
