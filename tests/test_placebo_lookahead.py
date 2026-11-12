import numpy as np
import pytest

from overfitlab import circular_shift_test, simulate_bars, breakout_biased, breakout_honest


def _market(n=1500, beta=0.2, seed=5):
    """Returns whose next value really depends on a persistent signal."""
    rng = np.random.default_rng(seed)
    sig = np.zeros(n)
    for i in range(1, n):                      # slow-moving signal so shifts matter
        sig[i] = 0.97 * sig[i - 1] + rng.normal()
    sig /= sig.std()
    r = np.empty(n)
    r[0] = rng.normal(0, 0.01)
    r[1:] = beta * 0.01 * sig[:-1] + rng.normal(0, 0.01, n - 1)
    pos = np.sign(sig)
    pos_for_period = np.concatenate([[0.0], pos[:-1]])   # decided one period earlier
    return r, pos_for_period


def test_placebo_rejects_a_real_signal():
    r, pos = _market()
    res = circular_shift_test(r, pos, n_perm=1000, seed=1)
    assert res.p_value < 0.05


def test_placebo_false_positive_rate_is_near_nominal():
    """Unrelated persistent signals should be flagged at p < 0.05 only about 5% of the time."""
    r = np.random.default_rng(9).normal(0, 0.01, 1500)
    flagged = 0
    reps = 60
    for k in range(reps):
        g = np.random.default_rng(100 + k)
        sig = np.zeros(1500)
        for i in range(1, 1500):
            sig[i] = 0.97 * sig[i - 1] + g.normal()
        pos = np.concatenate([[0.0], np.sign(sig)[:-1]])
        flagged += circular_shift_test(r, pos, n_perm=200, seed=k).p_value < 0.05
    assert flagged / reps < 0.15


def test_placebo_validates_inputs():
    with pytest.raises(ValueError):
        circular_shift_test(np.zeros(50), np.zeros(49))
    with pytest.raises(ValueError):
        circular_shift_test(np.zeros(10), np.zeros(10))


def test_lookahead_bug_creates_a_fake_edge_and_honest_fill_removes_it():
    honest_t = []
    for seed in range(3, 9):
        bars = simulate_bars(20000, seed=seed)
        biased = breakout_biased(bars)
        honest = breakout_honest(bars)
        assert biased["t_stat"] > 20                      # spectacular, and wrong
        assert honest["n_trades"] > biased["n_trades"]    # failed breakouts are trades too
        honest_t.append(honest["t_stat"])
    assert max(abs(t) for t in honest_t) < 3.5            # no edge on a driftless random walk
    assert abs(np.mean(honest_t)) < 1.5
