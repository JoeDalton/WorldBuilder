"""
The gravity field (world_builder.gravity): point masses, in numpy and in the
native kernel. The native tests are skipped when the library is not built
(invoke native), failed instead when WORLD_BUILDER_REQUIRE_NATIVE is set.
"""

import os

import numpy as np
import pytest

from world_builder.gravity import Gravity
from world_builder.terrain import noise

POSITIONS = np.array([[0.0, 0.0, 0.0], [4e6, -1e6, 2e6], [-3e6, 5e6, 1e6]])
MASSES = np.array([6e24, 7e22, 2e23])
RADII = np.array([6.4e6, 1.7e6, 2.5e6])


@pytest.fixture(params=["numpy", "native"])
def backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setenv("WORLD_BUILDER_NO_NATIVE_NOISE", "1")
        noise._native.cache_clear()
    else:
        noise._native.cache_clear()
        if noise.native_library() is None:
            if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
                pytest.fail("the native library is not built")
            pytest.skip("the native library is not built")
    yield request.param
    noise._native.cache_clear()


def _reference(points, g=1.0):
    out = np.zeros_like(points)
    for k, p in enumerate(points):
        for c, m, r in zip(POSITIONS, MASSES, RADII):
            d = c - p
            out[k] += g * m * d / max(np.linalg.norm(d), r) ** 3
    return out


def test_single_body_is_radial_and_inverse_square(backend):
    gravity = Gravity([[10.0, 0.0, 0.0]], [5e10], [1.0], g_constant=2.0)
    g = gravity.acceleration([[10.0, 0.0, 100.0], [10.0, 0.0, 200.0]])
    assert np.allclose(g[:, :2], 0.0)
    assert g[0, 2] == pytest.approx(-2.0 * 5e10 / 100.0**2)
    assert g[1, 2] == pytest.approx(g[0, 2] / 4.0)


def test_matches_a_scalar_loop(backend):
    rng = np.random.default_rng(1)
    points = rng.uniform(-1e7, 1e7, (500, 3))
    gravity = Gravity(POSITIONS, MASSES, RADII, g_constant=6.674e-11)
    expected = _reference(points, 6.674e-11)
    assert np.allclose(gravity.acceleration(points), expected, rtol=1e-12, atol=0)


def test_equal_masses_cancel_at_the_midpoint(backend):
    gravity = Gravity([[-5.0, 0, 0], [5.0, 0, 0]], [1e3, 1e3], [1.0, 1.0])
    assert np.allclose(gravity.acceleration([0.0, 0.0, 0.0]), 0.0)


def test_finite_at_and_capped_inside_a_body(backend):
    gravity = Gravity([[1e9, 1e9, 1e9]], [1e6], [100.0])
    centre = gravity.acceleration([1e9, 1e9, 1e9])
    assert np.isfinite(centre).all()
    surface = np.linalg.norm(gravity.acceleration([1e9 + 100.0, 1e9, 1e9]))
    inside = np.linalg.norm(gravity.acceleration([1e9 + 10.0, 1e9, 1e9]))
    assert surface == pytest.approx(1e6 / 100.0**2)
    assert inside == pytest.approx(10.0 / 100.0 * surface)


def test_shapes_and_out(backend):
    gravity = Gravity(POSITIONS, MASSES, RADII)
    assert gravity.acceleration([1e7, 0.0, 0.0]).shape == (3,)
    assert gravity.acceleration(np.zeros((0, 3)) + 1e7).shape == (0, 3)
    points = np.full((4, 3), 1e7)
    out = np.empty((4, 3))
    assert gravity.acceleration(points, out=out) is out
    with pytest.raises(ValueError):
        gravity.acceleration(points, out=np.empty((3, 3)))


def test_native_matches_numpy(monkeypatch):
    monkeypatch.setenv("WORLD_BUILDER_NO_NATIVE_NOISE", "1")
    reference = Gravity(POSITIONS, MASSES, RADII)
    monkeypatch.delenv("WORLD_BUILDER_NO_NATIVE_NOISE")
    noise._native.cache_clear()
    if noise.native_library() is None:
        pytest.skip("the native library is not built")
    native = Gravity(POSITIONS, MASSES, RADII)
    points = np.random.default_rng(2).uniform(-1e7, 1e7, (1000, 3))
    assert np.allclose(
        native.acceleration(points), reference.acceleration(points), rtol=1e-12
    )


def test_validation():
    with pytest.raises(ValueError):
        Gravity([[0, 0, 0]], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        Gravity([[0, 0, 0]], [1.0], [0.0])
