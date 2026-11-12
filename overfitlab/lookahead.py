"""A worked look-ahead bug: choosing trades by the close but filling at an intrabar level.

Rule being tested: "buy when price breaks above the prior bar's high, sell at the close".

* breakout_biased  picks bars where the CLOSE finished above the level, then assumes a fill AT the level.
  That uses information (where the bar closed) that did not exist when the order would have filled.
  It keeps only winning bars, so every trade is profitable by construction. Its size and t-statistic
  are mechanical, not evidence of anything.
* breakout_honest  trades every bar whose price reached the level, including failed breakouts that
  close back below it, and fills at the first price at or beyond the level (the open if the bar
  gapped through it), like a stop order that can slip.

On a driftless random walk the honest version has no edge. The biased version looks excellent.
(A first draft of the honest version filled exactly at the level even when the simulated price had
jumped past it; that handed it a free, statistically significant edge. The test suite caught it.
See README, Limitations.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bars:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    path: np.ndarray        # (n_bars, steps) prices inside each bar, after the open


def simulate_bars(n_bars: int, seed: int = 0, start: float = 100.0, bar_vol: float = 0.01,
                  gap_vol: float = 0.003, steps: int = 24) -> Bars:
    """Driftless random-walk OHLC bars with an overnight gap and `steps` sub-steps inside each bar."""
    rng = np.random.default_rng(seed)
    gaps = rng.normal(0.0, gap_vol * start, size=n_bars)
    incr = rng.normal(0.0, bar_vol * start / np.sqrt(steps), size=(n_bars, steps))
    close = np.empty(n_bars)
    open_ = np.empty(n_bars)
    high = np.empty(n_bars)
    low = np.empty(n_bars)
    paths = np.empty((n_bars, steps))
    prev = start
    for i in range(n_bars):
        o = prev + gaps[i]
        path = o + np.cumsum(incr[i])
        open_[i] = o
        high[i] = max(o, path.max())
        low[i] = min(o, path.min())
        close[i] = path[-1]
        paths[i] = path
        prev = close[i]
    return Bars(open_, high, low, close, paths)


def _summarise(pnl: np.ndarray, level: np.ndarray) -> dict:
    ret_bps = 1e4 * pnl / level
    n = ret_bps.size
    mean = float(ret_bps.mean()) if n else float("nan")
    se = float(ret_bps.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n_trades": int(n), "mean_bps": mean, "se_bps": se, "t_stat": mean / se if se else float("nan")}


def breakout_biased(bars: Bars) -> dict:
    level = bars.high[:-1]
    c = bars.close[1:]
    take = c > level                      # keeps only winners, chosen after the fact: profit is guaranteed
    return _summarise((c - level)[take], level[take])


def breakout_honest(bars: Bars) -> dict:
    level = bars.high[:-1]
    o, c, path = bars.open[1:], bars.close[1:], bars.path[1:]
    crossed = path >= level[:, None]
    touched = (o >= level) | crossed.any(axis=1)            # the order would have filled
    first_px = path[np.arange(level.size), crossed.argmax(axis=1)]
    fill = np.where(o >= level, o, first_px)                # gap -> open; else first price beyond level
    return _summarise((c - fill)[touched], level[touched])
