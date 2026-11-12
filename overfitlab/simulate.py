"""Synthetic strategy universes used by the tests and the demo."""

from __future__ import annotations

import numpy as np

from .stats import sharpe


def null_strategies(n_obs: int, n_strats: int, seed: int = 0, daily_vol: float = 0.01) -> np.ndarray:
    """N strategies with NO skill: independent zero-mean Gaussian returns, shape (T, N)."""
    return np.random.default_rng(seed).normal(0.0, daily_vol, size=(n_obs, n_strats))


def add_edge(matrix: np.ndarray, column: int, annual_sharpe: float, daily_vol: float = 0.01,
             periods_per_year: int = 252) -> np.ndarray:
    """Return a copy where one strategy has a real, constant edge of the given annual Sharpe ratio."""
    out = matrix.copy()
    out[:, column] += (annual_sharpe / np.sqrt(periods_per_year)) * daily_vol
    return out


def trial_sharpes(matrix: np.ndarray) -> np.ndarray:
    return np.array([sharpe(matrix[:, j]) for j in range(matrix.shape[1])])
