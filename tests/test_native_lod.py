"""
The native LOD selection (world_builder.terrain.native/lod.c, issue #20)
against the Python LodSelector it mirrors.

Both run in lockstep through flights in which chunks become ready, are
evicted and change bounds; every pass must select the same chunks in the same
order. Skipped when the library is not built (invoke native), failed instead
when WORLD_BUILDER_REQUIRE_NATIVE is set (CI).
"""

import math
import os

import numpy as np
import pytest

from world_builder.terrain import noise
from world_builder.terrain.cubesphere import grid_for
from world_builder.terrain.quadtree import (
    LodSelector,
    NativeLodSelector,
    make_selector,
)

R = 600e3
GRID = grid_for(R)


@pytest.fixture(scope="module")
def native():
    lib = noise._native()
    if lib is None:
        if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
            pytest.fail("native noise library required")
        pytest.skip(f"{noise.NATIVE_LIBRARY.name} is not built")
    return lib


class Pair:
    """A Python and a native selector fed the same chunk state."""

    def __init__(self, **kwargs):
        self.python = LodSelector(GRID, **kwargs)
        self.native = NativeLodSelector(GRID, **kwargs)
        self.ready: set = set()
        self.bounds: dict = {}

    def set(self, name, value):
        setattr(self.python, name, value)
        setattr(self.native, name, value)

    def store(self, key, bounds):
        self.bounds[key] = bounds
        self.native.set_bounds(key, bounds)

    def forget(self, key):
        self.bounds.pop(key, None)
        self.native.set_bounds(key, None)

    def make_ready(self, key, ready=True):
        if ready:
            self.ready.add(key)
        else:
            self.ready.discard(key)
        self.native.set_ready(key, ready)

    def select(self, camera, forward=None):
        expected = self.python.select(
            camera,
            ready=self.ready.__contains__,
            bounds=self.bounds.get,
            forward=forward,
        )
        got = self.native.select(camera, forward=forward)
        assert got.draw == expected.draw
        assert got.wanted == expected.wanted
        assert got.prefetch == expected.prefetch
        assert got.keep == expected.keep
        return expected


def _bounds(key, rng):
    """Plausible height bounds: coarse chunks span more."""
    spread = 30.0 * (1 << key[0]) ** 0.5
    low = rng.uniform(-3000.0, 2000.0)
    return (low, low + rng.uniform(0.0, spread * 20.0))


def _camera(lat, lon, altitude):
    lat, lon = math.radians(lat), math.radians(lon)
    up = np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )
    return up * (R + altitude), up


def _fly(pair, path, rng, uploads=40, evict=0.05):
    """Select along a path; after each pass some wanted chunks arrive (bounds
    known, ready) and a few drawn ones are evicted."""
    for camera, forward in path:
        selection = pair.select(camera, forward)
        for key in selection.wanted[:uploads] + selection.prefetch[: uploads // 4]:
            pair.store(key, _bounds(key, rng))
            pair.make_ready(key)
        for key in list(pair.ready):
            if key not in selection.keep and rng.random() < evict:
                pair.make_ready(key, False)
                if rng.random() < 0.5:
                    pair.forget(key)


def test_the_same_selection_from_orbit_to_the_ground(native):
    rng = np.random.default_rng(1)
    pair = Pair()
    path = []
    for altitude in np.geomspace(3.0 * R, 20.0, 80):
        camera, up = _camera(38.9, -125.8, altitude)
        forward = -up + 0.3 * np.cross(up, [0.0, 0.0, 1.0])
        path.append((camera, forward))
    _fly(pair, path, rng)


def test_the_same_selection_in_low_flight(native):
    """Fast and low: hysteresis, horizon culling and the behind-the-view
    penalty all in play."""
    rng = np.random.default_rng(2)
    pair = Pair()
    path = []
    for step in range(120):
        camera, up = _camera(10.0 + 0.01 * step, 20.0 + 0.03 * step, 300.0 + 5.0 * step)
        east = np.cross([0.0, 0.0, 1.0], up)
        path.append((camera, east / np.linalg.norm(east)))
    _fly(pair, path, rng)
    # Back and forth over the same ground: merges past the hysteresis only.
    _fly(pair, path[::-1], rng)


def test_the_same_selection_with_its_parameters_changed(native):
    rng = np.random.default_rng(3)
    pair = Pair(max_distance=50_000.0, horizon_floor=-7000.0, min_level=2)
    path = [(_camera(45.0, 45.0 + 0.02 * k, 2000.0)[0], None) for k in range(30)]
    _fly(pair, path, rng)
    pair.set("max_distance", None)
    pair.set("horizon_floor", 0.0)
    pair.set("min_level", 0)
    pair.set("split_factor", 1.6)
    _fly(pair, path, rng)
    pair.python.reset()
    pair.native.reset()
    _fly(pair, path[:5], rng)


def test_the_same_selection_at_a_cube_corner(native):
    """Three faces meet: neighbours from different faces, coarse and fine."""
    rng = np.random.default_rng(4)
    pair = Pair()
    corner = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    path = [(corner * (R + h), -corner) for h in np.geomspace(1e5, 50.0, 40)]
    _fly(pair, path, rng)


def test_make_selector_picks_the_native_one(native, monkeypatch):
    assert isinstance(make_selector(GRID), NativeLodSelector)
    monkeypatch.setattr("world_builder.terrain.quadtree.native_library", lambda: None)
    assert isinstance(make_selector(GRID), LodSelector)
