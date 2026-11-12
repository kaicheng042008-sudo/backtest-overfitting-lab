"""Basic return statistics shared by the other modules."""

from __future__ import annotations

import numpy as np
from scipy import stats as st


def sharpe(returns) -> float:
    """Per-period Sharpe ratio (mean / sample standard deviation, risk-free rate = 0)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def annualize(sr_per_period: float, periods_per_year: int = 252) -> float:
    """Scale a per-period Sharpe ratio to annual units. For display only."""
    return float(sr_per_period * np.sqrt(periods_per_year))


def skew_kurtosis(returns) -> tuple[float, float]:
    """Sample skewness and (non-excess) kurtosis. A normal distribution has kurtosis 3."""
    r = np.asarray(returns, dtype=float)
    return float(st.skew(r)), float(st.kurtosis(r, fisher=False))
