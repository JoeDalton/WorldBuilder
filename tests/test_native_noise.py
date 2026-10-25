"""
The native noise library (world_builder.terrain.native/noise.c, issue #20)
against the numpy noise it replaces.

Skipped when the library is not built (invoke native), failed instead when
WORLD_BUILDER_REQUIRE_NATIVE is set (CI).
"""

import os
from dataclasses import replace

import numpy as np
import pytest

from world_builder.terrain import noise
from world_builder.terrain.chunkgen import generate_chunk, planet_for
from world_builder.terrain.planet import PRESETS


@pytest.fixture(scope="module")
def native():
    lib = noise._native()
    if lib is None:
        if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
            pytest.fail("native noise library required")
        pytest.skip(f"{noise.NATIVE_LIBRARY.name} is not built")
    return lib


@pytest.fixture(scope="module")
def points():
    # Planet-sized coordinates, as the terrain samples them.
    return np.random.default_rng(0).uniform(-6e5, 6e5, size=(20000, 3))


def _numpy(monkeypatch, fn, *args, **kwargs):
    with monkeypatch.context() as m:
        m.setattr(noise, "_native", lambda: None)
        return fn(*args, **kwargs)


def test_simplex_is_bit_identical(native, points):
    for seed in (0, 7, 1234):
        np.testing.assert_array_equal(
            native.simplex3(points, noise._mix_seed(seed)),
            noise.simplex3(points, seed),
        )


@pytest.mark.parametrize("min_wavelength", [None, 1500.0, 40.0])
def test_fbm_matches_numpy(native, points, monkeypatch, min_wavelength):
    args = (points, 5000.0, 8, 9)
    expected = _numpy(monkeypatch, noise.fbm, *args, min_wavelength=min_wavelength)
    result = noise.fbm(*args, min_wavelength=min_wavelength)
    # numpy rotates the points with BLAS: last bits only.
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("min_wavelength", [None, 300.0, 3000.0])
def test_ridged_matches_numpy(native, points, monkeypatch, min_wavelength):
    """Band limiting leaves octaves of zero weight out: same result."""
    args = (points, 40e3, 9, 37)
    expected = _numpy(monkeypatch, noise.ridged, *args, min_wavelength=min_wavelength)
    result = noise.ridged(*args, min_wavelength=min_wavelength)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-9)


def test_warp_matches_numpy(native, points, monkeypatch):
    expected = _numpy(monkeypatch, noise.warp, points, 70e3, 4.2e3, 3, 11)
    np.testing.assert_allclose(
        noise.warp(points, 70e3, 4.2e3, 3, 11), expected, rtol=0, atol=1e-6
    )


def test_native_is_deterministic(native, points):
    a = noise.fbm(points, 900.0, 8, 67, min_wavelength=20.0)
    b = noise.fbm(points.copy(), 900.0, 8, 67, min_wavelength=20.0)
    np.testing.assert_array_equal(a, b)


def test_no_points(native):
    empty = np.zeros((0, 3))
    assert noise.fbm(empty, 900.0, 4, 1).shape == (0,)
    assert noise.ridged(empty, 900.0, 4, 1).shape == (0,)


def test_chunks_match_numpy(native, monkeypatch):
    spec = replace(PRESETS["earthlike"], seed=19).resolved()
    grid = planet_for(spec).grid
    for level in (0, 5, 10):
        middle = grid.side(level) // 2
        key = (level, 2, middle, middle)
        expected = _numpy(monkeypatch, generate_chunk, spec, key)
        chunk = generate_chunk(spec, key)
        np.testing.assert_allclose(chunk.elevation, expected.elevation, atol=1e-3)
        np.testing.assert_allclose(
            chunk.vertices()["vertex"], expected.vertices()["vertex"], atol=1e-3
        )
        np.testing.assert_array_equal(chunk.weights, expected.weights)


def test_numpy_can_be_forced(monkeypatch):
    monkeypatch.setenv("WORLD_BUILDER_NO_NATIVE_NOISE", "1")
    noise._native.cache_clear()
    try:
        assert not noise.native_noise()
    finally:
        monkeypatch.delenv("WORLD_BUILDER_NO_NATIVE_NOISE")
        noise._native.cache_clear()
