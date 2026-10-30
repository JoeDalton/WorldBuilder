"""
Unit tests for the tectonic plates (world_builder.terrain.plates) and the
desert landforms (world_builder.terrain.landforms). Pure NumPy.
"""

import math

import numpy as np
import pytest
from scipy import ndimage

from world_builder.terrain import biomes, landforms, strata
from world_builder.terrain.chart import AzimuthalEquidistantChart
from world_builder.terrain.plates import Plates, make_plates, plate_fields

R = 600e3
WIDTH = 80e3


def _random_dirs(n, seed=0):
    dirs = np.random.default_rng(seed).normal(size=(n, 3))
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


# ── plates ─────────────────────────────────────────────────────────────────────


def test_plates_are_deterministic_and_spread():
    a, b = make_plates(42, 14), make_plates(42, 14)
    np.testing.assert_array_equal(a.centres, b.centres)
    np.testing.assert_allclose(np.linalg.norm(a.centres, axis=1), 1.0)
    assert not np.array_equal(a.centres, make_plates(43, 14).centres)
    # Jittered, but no two plates crowd each other.
    cos = a.centres @ a.centres.T
    np.fill_diagonal(cos, -1.0)
    spacing = math.sqrt(4.0 * math.pi / 14)
    assert math.acos(cos.max()) > 0.3 * spacing


def _two_plates(spin_a, spin_b):
    """Plate a around +x, b around -x: their edge is the great circle x = 0."""
    centres = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    return Plates.of(centres, np.array([spin_a, spin_b], float), np.zeros(2))


@pytest.mark.parametrize(
    "spin_a, spin_b, kind",
    [
        # At (0, 1, 0), a moves toward -x (into b) and b toward +x: closing.
        ((0, 0, 1), (0, 0, -1), "convergent"),
        ((0, 0, -1), (0, 0, 1), "divergent"),
        # Both move along the edge, in opposite directions.
        ((1, 0, 0), (-1, 0, 0), "transform"),
    ],
)
def test_boundaries_are_classified_by_relative_motion(spin_a, spin_b, kind):
    plates = _two_plates(spin_a, spin_b)
    edge = np.array([[0.0, 1.0, 0.0]])
    fields = plate_fields(plates, edge, R, WIDTH)
    for name in ("convergent", "divergent", "transform"):
        value = float(getattr(fields, name)[0])
        assert value == pytest.approx(1.0 if name == kind else 0.0, abs=1e-6), name
    # Belts fade with distance from the edge.
    inside = np.array([[math.sin(0.3), math.cos(0.3), 0.0]])  # 180 km into a
    assert float(getattr(plate_fields(plates, inside, R, WIDTH), kind)[0]) == 0.0


def test_crust_is_the_plates_own_away_from_edges():
    centres = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    plates = Plates.of(centres, np.zeros((2, 3)), np.array([0.8, -0.6]))
    fields = plate_fields(plates, centres, R, WIDTH)
    np.testing.assert_allclose(fields.crust, [0.8, -0.6])
    edge = plate_fields(plates, np.array([[0.0, 0.0, 1.0]]), R, WIDTH)
    assert float(edge.crust[0]) == pytest.approx(0.1)


def test_fields_are_continuous_through_triple_junctions():
    """No jump anywhere, triple junctions included: neighbouring points 1 km
    apart differ by far less than a jump would (a belt rises over 80 km)."""
    plates = make_plates(19107, 14)
    dirs = _random_dirs(300000, seed=3)
    step = np.cross(dirs, _random_dirs(len(dirs), seed=4))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    moved = dirs + step * (1000.0 / R)
    moved /= np.linalg.norm(moved, axis=1, keepdims=True)
    a = plate_fields(plates, dirs, R, WIDTH)
    b = plate_fields(plates, moved, R, WIDTH)
    for name in ("crust", "convergent", "divergent", "transform"):
        jump = np.abs(getattr(a, name) - getattr(b, name)).max()
        assert jump < 0.08, name


# ── landforms ──────────────────────────────────────────────────────────────────


def _chart_grid(n, spacing, lat=20.0, lon=30.0):
    chart = AzimuthalEquidistantChart(lat, lon, R)
    coords = spacing * (np.arange(n) - n / 2)
    gx, gy = np.meshgrid(coords, coords)
    dirs = chart.to_dirs(gx, gy)
    return dirs, dirs * R


def test_dunes_match_sandastray():
    """The layer stack gives Sandastray's heights: its PNMImage version, 1 m a
    pixel, spans 0.10 / 0.35 / 0.63 at the 1st / 50th / 99th percentiles."""
    points = np.random.default_rng(1).uniform(-2e5, 2e5, size=(60000, 3))
    full = (1.0,) * len(landforms.DUNE_LAYERS)
    h = landforms._dune_stack(points, 1.0, 5, full)
    np.testing.assert_allclose(
        np.percentile(h, [1, 50, 99]), [0.10, 0.35, 0.63], atol=0.03
    )


def test_dunes_have_sharp_crests_at_several_scales():
    dirs, points = _chart_grid(512, 4.0)
    z = landforms.dunes(points, 1.0, 200.0, 7, None).reshape(512, 512)
    assert 100.0 < np.ptp(z) < 170.0
    # Big dunes (a few hundred metres) and small ones on them (tens).
    large = ndimage.gaussian_filter(z, 300.0 / 8.0)
    small = z - ndimage.gaussian_filter(z, 50.0 / 8.0)
    assert large.std() > 6.0 and small.std() > 1.5
    # Crests sharp enough that the steepest are rock.
    slope = biomes.slope_degrees(z, 4.0, 1)[2:-2, 2:-2]
    assert np.median(slope) < 25.0
    assert np.percentile(slope, 99.5) > biomes.ROCK_SLOPE[0]


def test_dune_offsets_are_measured():
    measured = landforms.dune_layer_offsets(samples=60000, seed=8)
    np.testing.assert_allclose(measured, landforms.DUNE_OFFSETS, atol=0.004)


def test_dunes_keep_their_mean_when_coarse():
    """Coarser, the layers fade finest first, the mean height kept: flat
    once all have faded."""
    dirs, points = _chart_grid(200, 40.0)
    fine = landforms.dunes(points, 1.0, 200.0, 7, None).mean()
    for min_wavelength in (100.0, 400.0, 1600.0):
        coarse = landforms.dunes(points, 1.0, 200.0, 7, min_wavelength)
        assert coarse.mean() == pytest.approx(fine, abs=5.0)
    flat = landforms.dunes(points, 1.0, 200.0, 7, 1e5)
    assert np.ptp(flat) < 1e-9


# ── tablelands ────────────────────────────────────────────────────────────────


def test_cells_are_a_voronoi_partition():
    points = np.random.default_rng(2).uniform(-2e4, 2e4, size=(20000, 3))
    f1, edge, ident = landforms.cells(points, 1000.0, 5)
    again = landforms.cells(points, 1000.0, 5)
    for a, b in zip((f1, edge, ident), again):
        np.testing.assert_array_equal(a, b)
    assert np.all(edge >= 0.0) and np.all(f1 >= 0.0)
    assert np.all((ident >= 0.0) & (ident < 1.0))
    assert abs(ident.mean() - 0.5) < 0.05  # one id per cell, uniform
    # The edge distance is continuous: a 5 m step changes it by at most 5 m.
    step = np.random.default_rng(3).normal(size=points.shape)
    step *= 5.0 / np.linalg.norm(step, axis=1, keepdims=True)
    _, moved, _ = landforms.cells(points + step, 1000.0, 5)
    assert np.abs(moved - edge).max() <= 5.0 + 1e-6


def test_strata_shape_walls():
    layers = strata.make_strata(11)
    assert np.all(np.diff(layers.runs) > 0.0)
    assert np.all(np.diff(layers.heights) >= 0.0)
    assert 0.2 < layers.hard.mean() < 0.5
    # A wall climbs from its foot to the ceiling, never past either.
    foot = np.full(4000, 300.0)
    s = np.linspace(-50.0, 3000.0, 4000)
    wall = layers.profile(foot, s, foot + 600.0)
    assert np.all(np.diff(wall) >= -1e-9)
    assert wall[0] == 300.0 and wall[-1] == pytest.approx(900.0)
    # Hard layers stand steeper than soft ones; their tops are ledges.
    slope = np.degrees(np.arctan(np.gradient(wall, s)))
    layer = np.searchsorted(layers.tops, wall)
    rising = (wall > 300.5) & (wall < 899.5)
    hard = rising & layers.hard[layer] & (slope > 1.0)
    soft = rising & ~layers.hard[layer] & (slope > 1.0)
    assert np.median(slope[hard]) == pytest.approx(strata.HARD_SLOPE, abs=1.0)
    assert np.median(slope[soft]) == pytest.approx(strata.SOFT_SLOPE, abs=1.0)
    # A talus at the foot: the gentler debris slope first.
    talus = layers.profile(foot, s, foot + 600.0, 100.0)
    debris = (talus > 300.5) & (talus < 399.5)
    talus_slope = np.degrees(np.arctan(np.gradient(talus, s)))[debris]
    assert np.median(talus_slope) == pytest.approx(strata.TALUS_SLOPE, abs=1.0)
    assert talus[-1] == pytest.approx(900.0)
    ledges = np.isin(np.round(wall, 6), np.round(layers.tops[layers.hard], 6))
    assert np.any(ledges & (slope < 1.0))
    lut = strata.strata_lut(layers)
    assert lut.shape == (1024, 3) and lut.min() >= 0.5


def _tableland(points, stage, min_wavelength=None):
    n = len(points)
    return landforms.tablelands(
        points,
        np.full(n, 400.0),
        np.ones(n),
        np.full(n, stage),
        np.full(n, 500.0),
        strata.make_strata(11),
        4000.0,
        1200.0,
        25.0,
        7,
        min_wavelength,
    )


@pytest.mark.parametrize("stage", [0.1, 0.9])
def test_tablelands_are_continuous(stage):
    """No jump anywhere: cell edges, dropped tables and spires included."""
    dirs, points = _chart_grid(400, 50.0)
    step = np.array([1.0, 0.0, 0.0]) * 2.0
    h = _tableland(points, stage)
    moved = _tableland(points + step, stage)
    # 2 m apart: a steep wall's rise over 2 m (an 80 degree cliff: 11 m,
    # steeper where the walls are crenellated), never a table's height.
    assert np.abs(moved - h).max() < 30.0
    assert h.min() >= 0.0 and h.max() <= 525.0 + 1e-6


def test_tablelands_have_tables_canyons_and_hilly_tops():
    dirs, points = _chart_grid(300, 40.0)
    young = _tableland(points, 0.05).reshape(300, 300)
    old = _tableland(points, 0.95).reshape(300, 300)
    top = young >= 500.0
    assert 0.4 < top.mean() < 0.95  # broad tables, narrow canyons
    assert (young < 5.0).mean() > 0.01  # canyon floors
    assert (old < 5.0).mean() > 0.6  # mostly plain once old
    assert (old > 300.0).mean() > 0.005  # with mesas and buttes
    # The tops: gently hilly, within the relief above the stack.
    tops = young[top]
    assert 5.0 < np.ptp(tops) <= 25.0 + 1e-6
    slope = biomes.slope_degrees(young, 40.0)[2:-2, 2:-2]
    assert np.percentile(slope[top[2:-2, 2:-2]], 95) < 15.0


def test_tablelands_blur_when_coarse():
    """Coarse, the walls are averaged over the sampling: the mean height
    stays, the cliffs soften."""
    dirs, points = _chart_grid(300, 40.0)
    fine = _tableland(points, 0.3)
    coarse = _tableland(points, 0.3, 400.0)
    assert coarse.mean() == pytest.approx(fine.mean(), rel=0.05)
    assert np.abs(coarse - fine).max() < 500.0
