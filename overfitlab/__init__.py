"""overfitlab: small, tested tools for judging whether a backtest result is real.

Everything works on per-period returns (for example daily). Annualise only for display.
"""

from .stats import sharpe, annualize, skew_kurtosis
from .dsr import psr, expected_max_sharpe, deflated_sharpe, DSRResult
from .pbo import cscv, PBOResult
from .placebo import circular_shift_test, PlaceboResult
from .lookahead import simulate_bars, breakout_biased, breakout_honest

__all__ = [
    "sharpe", "annualize", "skew_kurtosis",
    "psr", "expected_max_sharpe", "deflated_sharpe", "DSRResult",
    "cscv", "PBOResult",
    "circular_shift_test", "PlaceboResult",
    "simulate_bars", "breakout_biased", "breakout_honest",
]
