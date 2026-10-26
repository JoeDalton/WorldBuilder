"""A field's rocks from its spec (asteroids.scatter): determinism, density,
sizes, the hard core, keep-outs, clumping, spins, and the native scatter
against the numpy one."""

import math
import os
import pickle

import numpy as np
import pytest
from scipy.spatial import cKDTree

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.motion import rotate
from world_builder.asteroids.scatter import ROCK_DTYPE, Scatter, fbm_quantile
from world_builder.asteroids.spec import (
    PRESETS,
    Annulus,
    AroundAxis,
    Box,
    Clumping,
    Density,
    FieldSpec,
    KeepOut,
    Sphere,
    SpinSpec,
    Tumble,
)
from world_builder.terrain import noise


@pytest.fixture
def native():
    """Skip where the native library is not built (fail when it must be)."""
    if noise.native_library() is None:
        if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
            pytest.fail("the native library is required but not built")
        pytest.skip("native library not built")


def _uniform_field(count_per_km3=20.0, sizes=None, **kwargs) -> FieldSpec:
    """A box of uniform density, rocks far apart for their size: the hard
    core removes few."""
    return FieldSpec(
        shape=Box((4000.0, 4000.0, 4000.0)),
        density=Density(count_per_km3),
        sizes=sizes or Distribution.power_law(3.0, 40.0, 160.0),
        **kwargs,
    )


def _dense_field() -> FieldSpec:
    """Rocks about as far apart as they are big: the hard core removes about
    a third of them. (Near packing, it would remove almost all: a candidate
    goes wherever any larger one overlaps it, kept or not.)"""
    return FieldSpec(
        shape=Box((1500.0, 1500.0, 1500.0)),
        density=Density(60.0),
        sizes=Distribution.power_law(3.0, 40.0, 200.0),
        seed=5,
    )


def _all(scatter: Scatter) -> np.ndarray:
    lo, hi = scatter.spec.shape.bounds()
    return scatter.rocks_in_box(lo - 1.0, hi + 1.0)


def test_the_record_matches_the_native_struct():
    assert ROCK_DTYPE.itemsize == 128


def test_the_same_spec_gives_the_same_rocks_whatever_is_asked():
    scatter = Scatter(PRESETS["cube"])
    whole = scatter.rocks_in_box([-3000.0] * 3, [3000.0] * 3)
    halves = np.concatenate(
        [
            scatter.rocks_in_box([-3000.0, -3000.0, -3000.0], [0.0, 3000.0, 3000.0]),
            scatter.rocks_in_box(
                [np.nextafter(0.0, 1.0), -3000.0, -3000.0], [3000.0, 3000.0, 3000.0]
            ),
        ]
    )
    assert len(whole) > 50
    assert sorted(whole["id"].tolist()) == sorted(halves["id"].tolist())
    # cells in another order: the same rocks per cell
    cells = scatter.cells_in_box(0, [-2000.0] * 3, [2000.0] * 3)
    rng = np.random.default_rng(1)
    shuffled = cells[rng.permutation(len(cells))]
    a = scatter.rocks_in_cells(0, cells)
    b = scatter.rocks_in_cells(0, shuffled)
    assert sorted(a["id"].tolist()) == sorted(b["id"].tolist())
    # another seed, another field
    other = Scatter(FieldSpec(**{**PRESETS["cube"].__dict__, "seed": 1}))
    again = other.rocks_in_box([-3000.0] * 3, [3000.0] * 3)
    assert not set(again["id"].tolist()) & set(whole["id"].tolist())


def test_every_rock_is_one_rock():
    """No two cells draw the same candidates (the noise's lattice hash
    would repeat cells: the scatter hashes its own way)."""
    rocks = _all(Scatter(_dense_field()))
    assert len(np.unique(rocks["id"])) == len(rocks)
    centres = np.round(rocks["centre"], 6)
    assert len(np.unique(centres, axis=0)) == len(rocks)


def test_the_count_follows_the_density():
    spec = _uniform_field(sizes=Distribution.constant(40.0))
    rocks = _all(Scatter(spec))
    expected = spec.summary()["expected_count"]
    assert expected == pytest.approx(20.0 * 8.0**3)
    # Poisson: within 4 sigma, less the few the hard core removes
    assert abs(len(rocks) - expected) < 4.0 * math.sqrt(expected)


def test_the_sizes_follow_the_distribution():
    sizes = Distribution.power_law(3.5, 40.0, 640.0)
    spec = _uniform_field(count_per_km3=20.0, sizes=sizes, seed=3)
    radius = np.sort(_all(Scatter(spec))["radius"])
    n = len(radius)
    cdf = sizes.cdf(radius)
    ks = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))
    assert ks < 1.63 / math.sqrt(n)
    # every rock's class holds its radius
    rocks = _all(Scatter(spec))
    classes = Scatter(spec).classes
    lo = np.array([classes[c].lo for c in rocks["cls"]])
    hi = np.array([classes[c].hi for c in rocks["cls"]])
    # (to the size table's interpolation)
    assert np.all(
        (rocks["radius"] >= lo * (1 - 1e-3)) & (rocks["radius"] <= hi * (1 + 1e-3))
    )


def test_no_two_rocks_overlap():
    scatter = Scatter(_dense_field())
    rocks = _all(scatter)
    assert len(rocks) > 1000
    tree = cKDTree(rocks["centre"])
    pairs = np.array(sorted(tree.query_pairs(2.0 * rocks["radius"].max())))
    d = np.linalg.norm(
        rocks["centre"][pairs[:, 0]] - rocks["centre"][pairs[:, 1]], axis=1
    )
    reach = rocks["radius"][pairs[:, 0]] + rocks["radius"][pairs[:, 1]]
    assert np.all(d >= reach)
    # and the hard core did remove some: more candidates than rocks
    spec = _dense_field()
    assert len(rocks) < 0.9 * spec.summary()["expected_count"]


def test_keep_outs_stay_empty():
    ball = KeepOut((0.0, 0.0, 0.0), 1500.0)
    capsule = KeepOut((-4000.0, 2000.0, 0.0), 300.0, end=(4000.0, 2000.0, 500.0))
    spec = _uniform_field(count_per_km3=60.0, keep_outs=(ball,))
    scatter = Scatter(spec, keep_outs=(capsule,))
    assert scatter.is_clear((0.0, 0.0, 0.0), 1500.0)
    for t in np.linspace(0.0, 1.0, 9):
        point = np.array([-4000.0, 2000.0, 0.0]) + t * np.array([8000.0, 0.0, 500.0])
        assert scatter.is_clear(point, 300.0)
    # outside them, rocks
    assert not scatter.is_clear((0.0, 0.0, 0.0), 2500.0)
    # added at runtime, the same as in the spec
    late = Scatter(_uniform_field(count_per_km3=60.0)).with_keep_outs((ball, capsule))
    assert sorted(_all(late)["id"].tolist()) == sorted(_all(scatter)["id"].tolist())


def test_clumping_populates_its_coverage():
    clumping = Clumping(scale=2000.0, coverage=0.3, softness=0.01)
    spec = FieldSpec(
        shape=Box((1e6, 1e6, 1e6)),
        density=Density(1.0, clumping),
        sizes=Distribution.constant(50.0),
    )
    scatter = Scatter(spec)
    points = np.random.default_rng(2).uniform(-200_000.0, 200_000.0, size=(40_000, 3))
    density = scatter.density(points)
    assert np.mean(density > 0.5) == pytest.approx(0.3, abs=0.03)
    # a soft edge in the noise's units: mostly 0 or 1 here
    assert np.mean((density > 0.05) & (density < 0.95)) < 0.1
    assert fbm_quantile(4, 0.5) == pytest.approx(0.0, abs=0.05)


def test_the_profile_fades_over_the_edge():
    spec = FieldSpec(
        shape=Box((1000.0, 2000.0, 3000.0), edge=200.0),
        density=Density(1.0),
        sizes=Distribution.constant(50.0),
    )
    scatter = Scatter(spec)
    density = scatter.density(
        [
            [0.0, 0.0, 0.0],
            [900.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
            [0.0, 1900.0, 0.0],
            [0.0, 0.0, 3100.0],
        ]
    )
    np.testing.assert_allclose(density, [1.0, 0.5, 0.0, 0.5, 0.0])
    sphere = Scatter(
        FieldSpec(
            shape=Sphere(1000.0, edge=100.0),
            density=Density(1.0),
            sizes=Distribution.constant(50.0),
        )
    )
    np.testing.assert_allclose(
        sphere.density([[0, 0, 0], [0, 950.0, 0], [0, 0, 1000.0]]), [1.0, 0.5, 0.0]
    )


def test_spin_axes_and_rates():
    kappa = 10.0
    rate = Distribution.lognormal(0.05, 0.5, 0.005, 0.5)
    spec = _uniform_field(
        count_per_km3=40.0,
        sizes=Distribution.power_law(2.5, 40.0, 320.0),
        spin=SpinSpec(rate, axis=AroundAxis((0.0, 0.0, 2.0), kappa), size_exponent=1.5),
    )
    rocks = _all(Scatter(spec))
    np.testing.assert_allclose(np.linalg.norm(rocks["axis"], axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(rocks["q0"], axis=1), 1.0, atol=1e-12)
    # von Mises-Fisher: E[cos] = coth(kappa) - 1 / kappa
    mean_cos = np.mean(rocks["axis"][:, 2])
    assert mean_cos == pytest.approx(1.0 / math.tanh(kappa) - 1.0 / kappa, abs=0.01)
    # the rate of the smallest rocks, scaled back: the rate distribution
    base = rocks["precession"] * (rocks["radius"] / 40.0) ** 1.5
    x = np.sort(base)
    n = len(x)
    cdf = rate.cdf(x)
    ks = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))
    assert ks < 1.63 / math.sqrt(n)
    assert np.all(rocks["spin"] == 0.0)
    # isotropic axes
    iso = _all(Scatter(_uniform_field(count_per_km3=40.0)))
    assert np.abs(np.mean(iso["axis"], axis=0)).max() < 0.05


def test_tumbling_rocks_sit_on_their_cone():
    tumble = Tumble(0.4, Distribution.uniform(0.2, 1.0), Distribution.uniform(0.5, 1.5))
    spec = _uniform_field(
        count_per_km3=40.0,
        spin=SpinSpec(Distribution.constant(0.1), tumble=tumble, size_exponent=0.0),
    )
    rocks = _all(Scatter(spec))
    tumbling = rocks["spin"] != 0.0
    assert np.mean(tumbling) == pytest.approx(0.4, abs=0.04)
    t = rocks[tumbling]
    body_z = rotate(t["q0"], np.array([0.0, 0.0, 1.0]))
    cone = np.arccos(np.clip(np.sum(body_z * t["axis"], axis=1), -1.0, 1.0))
    assert np.all((cone > 0.2 - 1e-9) & (cone < 1.0 + 1e-9))
    ratio = t["spin"] / (t["precession"] * np.cos(cone))
    assert np.all((ratio > 0.5 - 1e-6) & (ratio < 1.5 + 1e-6))


def test_prototypes_and_ice():
    from world_builder.asteroids.spec import MaterialSpec, ShapeSpec

    spec = _uniform_field(
        count_per_km3=40.0,
        shapes=ShapeSpec(prototypes=7),
        material=MaterialSpec(Distribution.uniform(0.2, 0.6)),
    )
    rocks = _all(Scatter(spec))
    assert set(rocks["proto"].tolist()) == set(range(7))
    assert rocks["ice"].min() >= 0.2 and rocks["ice"].max() <= 0.6


def test_rocks_near_and_find_clear():
    scatter = Scatter(_dense_field())
    rocks = _all(scatter)
    point = np.array([100.0, -200.0, 50.0])
    near = scatter.rocks_near(point, 300.0)
    d = np.linalg.norm(rocks["centre"] - point, axis=1)
    expected = rocks[d < 300.0 + rocks["radius"]]
    assert sorted(near["id"].tolist()) == sorted(expected["id"].tolist())
    assert len(near) > 0
    clear = scatter.find_clear(point, 60.0, search=2000.0)
    assert clear is not None and scatter.is_clear(clear, 60.0)
    assert np.linalg.norm(clear - point) <= 2000.0
    # an empty spot is its own answer
    empty = np.array([1e5, 0.0, 0.0])
    np.testing.assert_array_equal(scatter.find_clear(empty, 60.0, 100.0), empty)


def test_too_many_cells_are_refused():
    scatter = Scatter(PRESETS["cube"])
    with pytest.raises(ValueError):
        scatter.cells_in_box(0, [-7500.0] * 3, [7500.0] * 3, limit=1000)


def test_specs_pickle():
    for spec in PRESETS.values():
        assert pickle.loads(pickle.dumps(spec)) == spec


def test_invalid_specs_are_refused():
    with pytest.raises(ValueError):
        FieldSpec(Box((1.0, 1.0, 1.0)), Density(1.0), Distribution.uniform(0.0, 1.0))
    with pytest.raises(ValueError):
        FieldSpec(
            Box((1.0, 1.0, 1.0)),
            Density(1.0),
            Distribution.constant(1.0),
            cell_factor=2.0,
        )


# ── native against numpy ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["cube", "icy_ball", "dense", "keep_outs"])
def test_the_native_scatter_gives_numpys_rocks(native, name):
    spec = {
        "dense": _dense_field(),
        "keep_outs": _uniform_field(
            count_per_km3=20.0,
            keep_outs=(KeepOut((0.0, 0.0, 0.0), 900.0, end=(2000.0, 0.0, 0.0)),),
        ),
    }.get(name) or PRESETS[name]
    fast = Scatter(spec, native=True)
    slow = Scatter(spec, native=False)
    # small rocks, small cells: a smaller region
    reach = 1500.0 if spec.sizes.lo < 50.0 else 3000.0
    lo, hi = np.array([-1.0, -0.8, -0.7]) * reach, np.array([0.9, 1.0, 0.7]) * reach
    for cls in range(len(fast.classes)):
        cells = fast.cells_in_box(cls, lo, hi)
        a = fast.rocks_in_cells(cls, cells)
        b = slow.rocks_in_cells(cls, cells)
        np.testing.assert_array_equal(a["id"], b["id"])
        for field in ("cls", "proto", "centre", "radius", "ice"):
            np.testing.assert_array_equal(a[field], b[field])
        # through sin, cos, log, pow: the last bits may differ
        for field in ("q0", "axis", "precession", "spin"):
            np.testing.assert_allclose(a[field], b[field], rtol=1e-12, atol=1e-14)


def test_numpy_noise_gives_nearly_the_same_field(native, monkeypatch):
    """Without the native noise, the clumping's last bits differ: a rock on
    the very edge of existing may differ, no more."""
    spec = PRESETS["cube"]
    lo, hi = [-4000.0] * 3, [4000.0] * 3
    with_native = Scatter(spec, native=False).rocks_in_box(lo, hi)
    monkeypatch.setattr(noise, "_native", lambda: None)
    without = Scatter(spec, native=False).rocks_in_box(lo, hi)
    a, b = set(with_native["id"].tolist()), set(without["id"].tolist())
    assert len(a ^ b) <= max(2, len(a) // 1000)


# ── rings ───────────────────────────────────────────────────────────────────


def test_an_annulus_profile():
    ring = Annulus(
        1000.0,
        2000.0,
        100.0,
        edge=20.0,
        radial_edge=50.0,
        radial_profile=(1.0, 0.0, 1.0),
    )
    rho = np.array([900.0, 1025.0, 1100.0, 1500.0, 1750.0, 1990.0, 2100.0])
    radial = ring.radial(rho)
    assert radial[0] == 0.0 and radial[-1] == 0.0  # the hole, outside
    assert radial[3] == pytest.approx(0.0)  # the profile's gap
    assert radial[4] == pytest.approx(0.5)  # half way back up
    assert 0.0 < radial[1] < radial[2]  # the inner rim's fade
    assert 0.0 < radial[5] < 1.0
    np.testing.assert_allclose(
        ring.vertical([0.0, 30.0, 45.0, 50.0, 60.0]), [1.0, 1.0, 0.15625, 0.0, 0.0]
    )
    points = np.stack([rho, np.zeros_like(rho), np.full_like(rho, 45.0)], axis=1)
    np.testing.assert_allclose(ring.profile(points), radial * 0.15625)
    # a plain annulus's volume
    plain = Annulus(1000.0, 2000.0, 100.0)
    assert plain.volume() == pytest.approx(
        math.pi * (2000.0**2 - 1000.0**2) * 100.0, rel=1e-6
    )
    with pytest.raises(ValueError):
        Annulus(2000.0, 1000.0, 100.0)


def test_a_ring_holds_its_density():
    """Rocks in a patch of a ring, where its profile is even: as many as
    its density says (the hard core removes few at this spacing)."""
    spec = FieldSpec(
        shape=Annulus(1.0e6, 2.0e6, 2000.0, radial_profile=(1.0, 0.5)),
        density=Density(30.0),
        sizes=Distribution.uniform(40.0, 60.0),
    )
    scatter = Scatter(spec)
    centre = np.array([0.0, 1.25e6, 0.0])
    half = np.array([3000.0, 3000.0, 1000.0])
    rocks = scatter.rocks_in_box(centre - half, centre + half)
    expected = 30.0 * 6.0 * 6.0 * 2.0 * float(spec.shape.radial([1.25e6])[0])
    assert abs(len(rocks) - expected) < 4.0 * math.sqrt(expected)
    assert np.all(np.abs(rocks["centre"][:, 2]) <= 1000.0)
    # none in the hole
    assert len(scatter.rocks_in_box([-3000.0] * 3, [3000.0] * 3)) == 0


def test_the_native_scatter_gives_numpys_ring(native):
    spec = PRESETS["ring"]
    fast = Scatter(spec, native=True)
    slow = Scatter(spec, native=False)
    # within the ring, across its dense band's edge
    centre = np.array([0.0, 9.0e7 + 0.5 * 2.0e7, 0.0])
    reach = np.array([2500.0, 2500.0, 1000.0])
    for cls in range(len(fast.classes)):
        cells = fast.cells_in_box(cls, centre - reach, centre + reach)
        a = fast.rocks_in_cells(cls, cells)
        b = slow.rocks_in_cells(cls, cells)
        np.testing.assert_array_equal(a["id"], b["id"])
        np.testing.assert_array_equal(a["centre"], b["centre"])
        np.testing.assert_array_equal(a["radius"], b["radius"])
    assert len(fast.rocks_in_box(centre - reach, centre + reach)) > 0
