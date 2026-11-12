"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation (CSCV).

Reference
---------
Bailey, D. H., Borwein, J., Lopez de Prado, M., Zhu, Q. J. (2015). "The Probability of Backtest
Overfitting." Journal of Computational Finance 20(4).

Idea: split the history into S blocks. For every way of choosing S/2 blocks as "in-sample", pick the
best strategy in-sample, then see where it ranks out-of-sample (on the other S/2 blocks). If the
in-sample winner usually lands in the bottom half out-of-sample, the selection process is overfit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class PBOResult:
    pbo: float                 # fraction of splits where the in-sample winner ranks at or below the OOS median
    logits: np.ndarray         # logit of the winner's relative OOS rank, one per split
    is_sharpe: np.ndarray      # in-sample Sharpe of the winner, one per split
    oos_sharpe: np.ndarray     # out-of-sample Sharpe of that same strategy, one per split
    prob_loss: float           # fraction of splits where the winner loses money out-of-sample
    n_splits: int


def _sharpe_from_sums(s: np.ndarray, q: np.ndarray, n: int) -> np.ndarray:
    mean = s / n
    var = (q - n * mean**2) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 0, mean / np.sqrt(var), np.nan)


def cscv(returns_matrix, n_blocks: int = 16) -> PBOResult:
    """returns_matrix: array of shape (T, N): T periods of returns for N strategy configurations."""
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("returns_matrix must be 2-D with at least two strategies")
    if n_blocks < 4 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 4")
    t, n_strat = m.shape
    per_block = t // n_blocks
    if per_block < 2:
        raise ValueError("not enough observations for this many blocks")
    if t % n_blocks:
        warnings.warn(f"dropping the last {t % n_blocks} of {t} observations so blocks are equal length",
                      stacklevel=2)

    blocks = m[: per_block * n_blocks].reshape(n_blocks, per_block, n_strat)
    s = blocks.sum(axis=1)          # (S, N) sum of returns per block
    q = (blocks**2).sum(axis=1)     # (S, N) sum of squared returns per block

    half = n_blocks // 2
    n_obs_half = per_block * half
    everything = set(range(n_blocks))

    logits, is_sel, oos_sel = [], [], []
    for is_idx in combinations(range(n_blocks), half):
        is_list = list(is_idx)
        oos_list = sorted(everything - set(is_idx))
        sr_is = _sharpe_from_sums(s[is_list].sum(axis=0), q[is_list].sum(axis=0), n_obs_half)
        sr_oos = _sharpe_from_sums(s[oos_list].sum(axis=0), q[oos_list].sum(axis=0), n_obs_half)
        if np.all(np.isnan(sr_is)):
            raise ValueError("every strategy has zero variance in some in-sample window")
        winner = int(np.nanargmax(sr_is))
        rank = 1 + int(np.sum(sr_oos < sr_oos[winner]))        # 1 = worst, N = best
        omega = rank / (n_strat + 1)
        logits.append(np.log(omega / (1.0 - omega)))
        is_sel.append(sr_is[winner])
        oos_sel.append(sr_oos[winner])

    logits_a = np.array(logits)
    is_a, oos_a = np.array(is_sel), np.array(oos_sel)
    return PBOResult(
        pbo=float(np.mean(logits_a <= 0.0)),
        logits=logits_a,
        is_sharpe=is_a,
        oos_sharpe=oos_a,
        prob_loss=float(np.mean(oos_a < 0.0)),
        n_splits=int(logits_a.size),
    )
