import math

import numpy as np

from overfitlab import psr, expected_max_sharpe, deflated_sharpe, sharpe, skew_kurtosis
from overfitlab.simulate import null_strategies, trial_sharpes


def test_psr_is_half_when_estimate_equals_benchmark():
    assert abs(psr(0.1, 0.1, 500, 0.0, 3.0) - 0.5) < 1e-12


def test_psr_matches_a_hand_computed_value():
    """Sharpe 0.08 per period over 400 periods, normal returns: z = 0.08*sqrt(399)/sqrt(1.0032) = 1.5953.

    The normal CDF is evaluated with math.erf, independent of scipy and of the implementation.
    """
    z = 0.08 * math.sqrt(399) / math.sqrt(1 + 0.08**2 / 2)
    assert abs(z - 1.5953) < 5e-4
    assert abs(psr(0.08, 0.0, 400, 0.0, 3.0) - 0.9447) < 5e-4


def test_psr_penalises_negative_skew_and_fat_tails():
    normal = psr(0.08, 0.0, 400, 0.0, 3.0)
    assert psr(0.08, 0.0, 400, -1.0, 3.0) < normal      # negative skew lowers confidence
    assert psr(0.08, 0.0, 400, 0.0, 9.0) < normal       # fat tails lower confidence


def test_psr_grows_with_sample_length():
    assert psr(0.05, 0.0, 1000) > psr(0.05, 0.0, 100)


def test_psr_is_calibrated_under_the_null():
    """With zero true skill, PSR(0) should exceed 0.95 about 5% of the time."""
    rng = np.random.default_rng(1)
    hits = 0
    reps = 2000
    for _ in range(reps):
        r = rng.normal(0.0, 0.01, 250)
        sk, ku = skew_kurtosis(r)
        hits += psr(sharpe(r), 0.0, r.size, sk, ku) > 0.95
    assert abs(hits / reps - 0.05) < 0.02


def test_expected_max_sharpe_properties():
    assert expected_max_sharpe(1, 0.01) == 0.0
    values = [expected_max_sharpe(n, 0.001) for n in (2, 10, 100, 1000)]
    assert values == sorted(values) and values[0] > 0
    assert expected_max_sharpe(100, 0.004) > expected_max_sharpe(100, 0.001)


def test_expected_max_sharpe_matches_known_expected_maxima_of_standard_normals():
    """Exact expected maxima of N standard normals (order-statistic tables): 1.5388, 2.5076, 3.2414.

    The formula is an approximation, so allow 0.05.
    """
    for n, exact in ((10, 1.5388), (100, 2.5076), (1000, 3.2414)):
        assert abs(expected_max_sharpe(n, 1.0) - exact) < 0.05


def test_expected_max_sharpe_tracks_simulation():
    """The formula should be close to the average best Sharpe among skill-free strategies."""
    reps, n_strats, t = 200, 100, 500
    best, predicted = [], []
    for k in range(reps):
        trials = trial_sharpes(null_strategies(t, n_strats, seed=1000 + k))
        best.append(np.nanmax(trials))
        predicted.append(expected_max_sharpe(n_strats, np.nanvar(trials, ddof=1)))
    assert abs(np.mean(best) / np.mean(predicted) - 1.0) < 0.10


def test_deflation_removes_false_significance_in_one_universe():
    m = null_strategies(1000, 270, seed=7)
    trials = trial_sharpes(m)
    best = int(np.nanargmax(trials))
    res = deflated_sharpe(m[:, best], trials)
    assert res.psr_vs_zero > 0.99
    assert res.dsr < 0.95
    assert res.n_trials == 270


def test_deflation_works_across_many_universes():
    """Skill-free universes: the naive test flags nearly all of them, the deflated test almost none."""
    naive = deflated = 0
    reps = 60
    for k in range(reps):
        m = null_strategies(500, 100, seed=3000 + k)
        trials = trial_sharpes(m)
        res = deflated_sharpe(m[:, int(np.nanargmax(trials))], trials)
        naive += res.psr_vs_zero > 0.95
        deflated += res.dsr > 0.95
    assert naive / reps > 0.9
    assert deflated / reps <= 0.10
