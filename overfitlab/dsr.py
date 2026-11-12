"""Probabilistic and Deflated Sharpe Ratio.

References
----------
Bailey, D. H. and Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
    Journal of Risk 15(2).  (Probabilistic Sharpe Ratio)
Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality." Journal of Portfolio Management 40(5).

All Sharpe ratios here are PER PERIOD (not annualised).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .stats import sharpe, skew_kurtosis

EULER_MASCHERONI = 0.5772156649015329


def psr(sr_hat: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probability that the true Sharpe ratio exceeds `sr_benchmark`.

    Accounts for sample length and for non-normal returns (skewness, kurtosis).
    """
    denom_sq = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if denom_sq <= 0 or n_obs < 2:
        return float("nan")
    z = (sr_hat - sr_benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe ratio among `n_trials` independent trials with zero true skill.

    `sr_variance` is the variance of the Sharpe estimates across trials.
    """
    if n_trials <= 1:
        return 0.0
    g = EULER_MASCHERONI
    return float(
        np.sqrt(sr_variance)
        * ((1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials) + g * norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    )


@dataclass(frozen=True)
class DSRResult:
    sharpe: float            # per-period Sharpe of the selected strategy
    benchmark_sharpe: float  # expected best Sharpe from luck alone, given n_trials
    psr_vs_zero: float       # naive significance: ignores how many strategies were tried
    dsr: float               # significance after accounting for the search
    n_trials: int
    n_obs: int


def deflated_sharpe(selected_returns, trial_sharpes, n_trials: int | None = None) -> DSRResult:
    """Deflate the best strategy's Sharpe ratio for the number of strategies that were tried.

    selected_returns: per-period returns of the strategy you picked.
    trial_sharpes:    per-period Sharpe ratio of EVERY strategy tried (including the winner).
    n_trials:         override the count if some trials were not recorded.
    """
    r = np.asarray(selected_returns, dtype=float)
    trials = np.asarray(trial_sharpes, dtype=float)
    n = int(n_trials) if n_trials else int(trials.size)
    sr = sharpe(r)
    skew, kurt = skew_kurtosis(r)
    sr0 = expected_max_sharpe(n, float(np.nanvar(trials, ddof=1)))
    return DSRResult(
        sharpe=sr,
        benchmark_sharpe=sr0,
        psr_vs_zero=psr(sr, 0.0, r.size, skew, kurt),
        dsr=psr(sr, sr0, r.size, skew, kurt),
        n_trials=n,
        n_obs=int(r.size),
    )
