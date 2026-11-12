import numpy as np
import pytest

from overfitlab import cscv
from overfitlab.simulate import null_strategies, add_edge


def test_pbo_near_one_half_without_skill():
    values = [cscv(null_strategies(800, 40, seed=s), n_blocks=10).pbo for s in range(12)]
    assert 0.35 < float(np.mean(values)) < 0.65


def test_pbo_low_with_a_real_edge():
    m = add_edge(null_strategies(800, 40, seed=3), column=5, annual_sharpe=2.0)
    res = cscv(m, n_blocks=10)
    assert res.pbo < 0.15
    assert res.oos_sharpe.mean() > 0


def test_in_sample_winner_degrades_out_of_sample_without_skill():
    res = cscv(null_strategies(800, 60, seed=11), n_blocks=10)
    assert res.is_sharpe.mean() > 0.05          # looks good in-sample
    assert res.oos_sharpe.mean() < res.is_sharpe.mean() / 2


def test_warns_when_trailing_observations_are_dropped():
    with pytest.warns(UserWarning, match="dropping the last"):
        cscv(null_strategies(805, 10, seed=2), n_blocks=10)


def test_constant_strategies_raise_a_clear_error():
    with pytest.raises(ValueError, match="zero variance"):
        cscv(np.ones((100, 4)), n_blocks=10)


def test_input_validation():
    with pytest.raises(ValueError):
        cscv(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        cscv(np.zeros((100, 5)), n_blocks=7)
    with pytest.raises(ValueError):
        cscv(np.zeros((10, 5)), n_blocks=16)
