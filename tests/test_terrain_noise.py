"""
Unit tests for the terrain prototype's noise and chart
(world_builder.terrain.noise / chart). Pure NumPy.
"""

import math

import numpy as np
import pytest

from world_builder.terrain.chart import (
    AzimuthalEquidistantChart,
    dirs_from_latlon,
    latlon,
)
from world_builder.terrain.noise import (
    fbm,
    octave_weight,
    ridged,
    simplex3,
    smoothstep,
    warp,
)


@pytest.fixture(scope="module")
def points():
    return np.random.default_rng(0).uniform(-1e4, 1e4, size=(20000, 3))


def test_simplex_is_deterministic(points):
    np.testing.assert_array_equal(simplex3(points, 5), simplex3(points.copy(), 5))


def test_simplex_seeds_are_independent(points):
    a, b = simplex3(points, 1), simplex3(points, 2)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.05


def test_simplex_range_and_mean(points):
    n = simplex3(points, 3)
    assert np.abs(n).max() <= 1.0
    assert np.abs(n).max() > 0.8
    assert abs(n.mean()) < 0.02


def test_simplex_is_continuous():
    """Tiny steps give tiny changes everywhere: no lattice discontinuities."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        start, direction = rng.uniform(-50, 50, 3), rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        t = np.linspace(0.0, 10.0, 100001)[:, None]
        values = simplex3(start + t * direction, 4)
        assert np.abs(np.diff(values)).max() < 1e-3


def test_simplex_does_not_repeat_at_256_cells():
    """A permutation-table noise would repeat every 256 lattice cells."""
    p = np.random.default_rng(2).uniform(0, 100, size=(2000, 3))
    a = simplex3(p, 0)
    b = simplex3(p + [256.0, 0.0, 0.0], 0)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.1


def test_fbm_band_limit_only_removes_detail(points):
    full = fbm(points, 5000.0, 8, seed=9)
    limited = fbm(points, 5000.0, 8, seed=9, min_wavelength=1500.0)
    # Same normalisation: the coarse octaves are identical, so the two stay close.
    assert np.corrcoef(full, limited)[0, 1] > 0.9
    assert np.abs(full - limited).max() < 0.3


def test_octave_weight_fades_smoothly():
    assert octave_weight(0.1, None) == 1.0
    assert octave_weight(0.4, 1.0) == 1.0
    assert octave_weight(1.0, 1.0) == 0.0
    assert 0.0 < octave_weight(0.75, 1.0) < 1.0


def test_ridged_range(points):
    r = ridged(points, 5000.0, 6, seed=3)
    assert r.min() >= 0.0
    assert r.max() <= 1.0


def test_warp_moves_points_by_about_the_amplitude(points):
    moved = warp(points, 20000.0, 100.0, 3, seed=1)
    shift = np.linalg.norm(moved - points, axis=1)
    assert 10.0 < np.median(shift) < 200.0


def test_smoothstep():
    np.testing.assert_allclose(
        smoothstep(0.0, 1.0, np.array([-1, 0, 0.5, 1, 2])), [0, 0, 0.5, 1, 1]
    )
    np.testing.assert_allclose(smoothstep(1.0, 0.0, np.array([0.0, 1.0])), [1, 0])


# ── chart ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def chart():
    return AzimuthalEquidistantChart(42.0, 10.0, 6371000.0)


def test_chart_origin_is_the_reference_point(chart):
    lat, lon = latlon(chart.to_dirs(np.array([0.0]), np.array([0.0])))
    assert lat[0] == pytest.approx(42.0)
    assert lon[0] == pytest.approx(10.0)


def test_chart_round_trip(chart):
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-2e6, 2e6, 1000), rng.uniform(-2e6, 2e6, 1000)
    dirs = chart.to_dirs(x, y)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)
    bx, by = chart.from_dirs(dirs)
    np.testing.assert_allclose(bx, x, atol=1e-6)
    np.testing.assert_allclose(by, y, atol=1e-6)


def test_chart_distances_from_origin_are_exact(chart):
    """Great-circle distance from the origin equals the flat radius."""
    x, y = np.array([300e3, -50e3, 0.0]), np.array([400e3, 20e3, -1234.5])
    dirs = chart.to_dirs(x, y)
    angle = np.arccos(np.clip(dirs @ chart.frame.up, -1.0, 1.0))
    np.testing.assert_allclose(
        angle * chart.radius, np.hypot(x, y), rtol=1e-9, atol=1e-6
    )


def test_chart_axes_point_east_and_north(chart):
    lat_n, lon_n = latlon(chart.to_dirs(np.array([0.0]), np.array([10e3])))
    lat_e, lon_e = latlon(chart.to_dirs(np.array([10e3]), np.array([0.0])))
    assert lat_n[0] > 42.0 and lon_n[0] == pytest.approx(10.0)
    assert lon_e[0] > 10.0


def test_dirs_from_latlon_inverts_latlon():
    dirs = dirs_from_latlon([10.0, -45.0], [20.0, 170.0])
    lat, lon = latlon(dirs)
    np.testing.assert_allclose(lat, [10.0, -45.0])
    np.testing.assert_allclose(lon, [20.0, 170.0])
    assert math.isclose(float(np.linalg.norm(dirs[0])), 1.0)
