"""Placebo test for timing signals using circular shifts.

If a signal really predicts returns, shifting its positions in time (keeping their number, size and
clustering) should destroy the result. Rotating the position series circularly keeps its
autocorrelation and exposure intact, which a plain shuffle would not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import sharpe


@dataclass(frozen=True)
class PlaceboResult:
    observed: float        # Sharpe of the real positions
    null: np.ndarray       # Sharpe under each circularly shifted placebo
    p_value: float         # share of placebos at least as good as the real result (one-sided)
    n_perm: int


def circular_shift_test(returns, position, n_perm: int = 2000, seed: int = 0) -> PlaceboResult:
    """returns:  asset returns per period.
    position: exposure held DURING each period (already lagged: decided before the return is known).
    """
    r = np.asarray(returns, dtype=float)
    pos = np.asarray(position, dtype=float)
    if r.shape != pos.shape:
        raise ValueError("returns and position must have the same length")
    if r.size < 20:
        raise ValueError("need at least 20 observations")

    observed = sharpe(pos * r)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, r.size, size=n_perm)
    null = np.array([sharpe(np.roll(pos, int(k)) * r) for k in shifts])
    p = (1 + int(np.sum(null >= observed))) / (n_perm + 1)
    return PlaceboResult(observed=float(observed), null=null, p_value=float(p), n_perm=n_perm)
