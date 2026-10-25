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


# ── the clouds' volume noise ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def volume():
    from world_builder.clouds.noise import quantise, value_noise_volume

    return quantise(value_noise_volume(seed=7))


def test_cloud_volume_fbm_is_bit_identical(native, volume, monkeypatch):
    """Billboards are placed where the CPU fbm says there is cloud: native and
    numpy must agree exactly, or a test of the placement could disagree."""
    from world_builder.clouds import noise as clouds_noise

    coords = np.random.default_rng(1).uniform(-300.0, 300.0, size=(20000, 3))
    field = clouds_noise.DensityField()
    expected = {}
    with monkeypatch.context() as m:
        m.setattr(clouds_noise, "native_library", lambda: None)
        expected["sample"] = clouds_noise.sample_volume(volume, coords)
        for octaves in (2, 3, 4):
            expected[octaves] = clouds_noise.fbm(volume, coords, field, octaves)
        expected["mean0"] = clouds_noise.fbm(volume, coords, field, 2, 0.0)
    np.testing.assert_array_equal(
        clouds_noise.sample_volume(volume, coords), expected["sample"]
    )
    for octaves in (2, 3, 4):
        np.testing.assert_array_equal(
            clouds_noise.fbm(volume, coords, field, octaves), expected[octaves]
        )
    np.testing.assert_array_equal(
        clouds_noise.fbm(volume, coords, field, 2, 0.0), expected["mean0"]
    )


def test_the_swell_matches_numpy(native, monkeypatch):
    """The water collisions stand on (swell.h) is the numpy fallback's."""
    from world_builder.ocean import swell as swell_module
    from world_builder.ocean.ocean import swell_rotations
    from world_builder.terrain.chart import TangentFrame

    frame = TangentFrame.at(10.0, 20.0)
    a, b = swell_rotations(
        frame.up, frame.east, frame.north, (0.6, 0.8), 0.004, 0.1, 600e3, 1234.5
    )
    swell = swell_module.Swell(20.0, 0.004, a, b)
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    depths = rng.uniform(0.0, 120.0, len(dirs))
    max_freqs = rng.choice([0.0, 0.25 / 1.8, 0.25 / 29.0, 0.25 / 230.0], len(dirs))
    native_heights = swell.heights(dirs * 600e3, depths, max_freqs)
    with monkeypatch.context() as m:
        m.setattr(swell_module, "native_library", lambda: None)
        numpy_heights = swell.heights(dirs * 600e3, depths, max_freqs)
    np.testing.assert_allclose(native_heights, numpy_heights, rtol=0, atol=1e-9)
    # About sea level, within the amplitude.
    assert abs(float(np.mean(numpy_heights))) < 1.0
    assert np.all(np.abs(numpy_heights) <= 0.75 * 20.0)
