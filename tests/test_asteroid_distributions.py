"""Truncated distributions for asteroid fields (asteroids.distributions)."""

import math
import pickle

import numpy as np
import pytest

from world_builder.asteroids.distributions import (
    TABLE_SIZE,
    Distribution,
    interpolate,
)

KINDS = {
    "uniform": Distribution.uniform(2.0, 7.0),
    "normal": Distribution.normal(5.0, 2.0, 0.0, 8.0),
    "lognormal": Distribution.lognormal(150.0, 0.6, 50.0, 1000.0),
    "power_law": Distribution.power_law(3.5, 50.0, 2000.0),
    "power_law_1": Distribution.power_law(1.0, 1.0, 100.0),
    "u_shaped_log": Distribution.u_shaped(0.4, 50.0, 2000.0),
    "u_shaped_linear": Distribution.u_shaped(0.3, 0.0, 1.0, log=False),
    "table": Distribution.table([0.0, 1.0, 3.0], [0.0, 2.0, 0.0]),
}


def _ks(samples: np.ndarray, dist: Distribution) -> float:
    """The Kolmogorov-Smirnov distance between samples and dist's CDF."""
    x = np.sort(samples)
    n = len(x)
    cdf = dist.cdf(x)
    return float(
        max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))
    )


@pytest.mark.parametrize("name", list(KINDS))
def test_cdf_and_ppf_invert_each_other(name):
    dist = KINDS[name]
    u = np.linspace(0.001, 0.999, 999)
    np.testing.assert_allclose(dist.cdf(dist.ppf(u)), u, atol=1e-9)
    assert dist.cdf(dist.lo) == pytest.approx(0.0, abs=1e-12)
    assert dist.cdf(dist.hi) == pytest.approx(1.0, abs=1e-12)
    assert dist.cdf(dist.lo - 1.0) == 0.0 and dist.cdf(dist.hi + 1.0) == 1.0


@pytest.mark.parametrize("name", list(KINDS))
def test_draws_follow_the_distribution(name):
    dist = KINDS[name]
    u = np.random.default_rng(7).random(100_000)
    samples = dist.sample(u)
    assert samples.min() >= dist.lo and samples.max() <= dist.hi
    # 1.63 / sqrt(n): the 1 % critical value
    assert _ks(samples, dist) < 1.63 / math.sqrt(len(u))


def test_the_table_spans_the_interval_and_never_decreases():
    for dist in KINDS.values():
        table = dist.table_values()
        assert len(table) == TABLE_SIZE
        assert table[0] == dist.lo and table[-1] == dist.hi
        assert np.all(np.diff(table) >= 0.0)


def test_interpolate_hits_the_table_points():
    table = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    np.testing.assert_array_equal(interpolate(table, [0.0, 0.25, 0.5, 0.75]), table[:4])
    assert interpolate(table, 0.125) == 0.5
    assert interpolate(table, np.nextafter(1.0, 0.0)) == pytest.approx(16.0)


def test_medians_and_means():
    assert KINDS["uniform"].median() == pytest.approx(4.5)
    assert KINDS["uniform"].mean() == pytest.approx(4.5)
    # untruncated around the median: the median holds
    wide = Distribution.lognormal(150.0, 0.6, 1.0, 1e6)
    assert wide.median() == pytest.approx(150.0, rel=1e-6)
    # power law: the closed-form median
    q, lo, hi = 3.5, 50.0, 2000.0
    a, b = lo ** (1 - q), hi ** (1 - q)
    expected = (0.5 * (a + b)) ** (1 / (1 - q))
    assert KINDS["power_law"].median() == pytest.approx(expected, rel=1e-9)
    # triangle on [0, 3] peaking at 1: mean (0 + 1 + 3) / 3
    assert KINDS["table"].mean() == pytest.approx(4.0 / 3.0, rel=1e-4)
    assert Distribution.uniform(0.0, 1.0).moment(2.0) == pytest.approx(1 / 3, rel=1e-4)


def test_u_shaped_favours_both_ends():
    dist = KINDS["u_shaped_linear"]
    ends = dist.pdf([0.02, 0.98])
    middle = dist.pdf([0.5])
    assert np.all(ends > 3.0 * middle)
    assert dist.median() == pytest.approx(0.5, abs=1e-9)
    # a = 1 is flat across log x
    flat = Distribution.u_shaped(1.0, 10.0, 1000.0)
    assert flat.median() == pytest.approx(100.0, rel=1e-9)


def test_constant():
    dist = Distribution.constant(3.0)
    assert np.all(dist.sample(np.linspace(0.0, 0.99, 10)) == 3.0)
    assert dist.median() == 3.0


def test_invalid_distributions_are_refused():
    with pytest.raises(ValueError):
        Distribution.uniform(2.0, 1.0)
    with pytest.raises(ValueError):
        Distribution.power_law(3.5, 0.0, 10.0)
    with pytest.raises(ValueError):
        Distribution.u_shaped(0.5, 0.0, 10.0)
    with pytest.raises(ValueError):
        Distribution.table([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        Distribution("cauchy", 0.0, 1.0)


def test_distributions_pickle_and_hash():
    for dist in KINDS.values():
        dist.table_values()  # caches
        again = pickle.loads(pickle.dumps(dist))
        assert again == dist and hash(again) == hash(dist)
        np.testing.assert_array_equal(again.table_values(), dist.table_values())
