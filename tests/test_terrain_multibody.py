"""
Several planets in one world (world_builder.terrain): the shared worker pool,
per-planet positions, and memory released by planets seen from afar.
"""

import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from world_builder.terrain import biomes, workers
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.chunkgen import generate_chunk
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.workers import ChunkExecutor


@pytest.fixture(scope="module")
def planet_spec():
    return replace(PRESETS["earthlike"], seed=19).resolved()


@pytest.fixture(scope="module")
def moon_spec():
    return replace(PRESETS["moon"], seed=7).resolved()


# ── shared executor ────────────────────────────────────────────────────────────


@pytest.fixture
def counted(monkeypatch):
    """Replace chunk generation with a slow stub counting its calls."""
    calls = []
    lock = threading.Lock()

    def fake(spec, key, version):
        with lock:
            calls.append((spec.name, key))
        time.sleep(0.01)
        return _stub(key, version)

    monkeypatch.setattr(workers, "generate_chunk", fake)
    return calls


def _stub(key, version):
    from world_builder.terrain.chunkgen import ChunkData

    zero = np.zeros(3)
    return ChunkData(key, version, zero, b"", zero, zero, zero, zero, 0, 0, 0, 0, 0)


def _drain(executor, channels, count, timeout=10.0):
    done = {c: [] for c in channels}
    start = time.perf_counter()
    while sum(len(v) for v in done.values()) < count:
        assert time.perf_counter() - start < timeout, "chunks never arrived"
        for channel in channels:
            done[channel] += executor.collect(channel)
        time.sleep(0.005)
    return done


def test_channels_share_the_pool_without_duplicates(counted, planet_spec, moon_spec):
    executor = ChunkExecutor(backend="thread", workers=2, max_inflight=4)
    v_planet = executor.set_spec(planet_spec, "planet")
    v_moon = executor.set_spec(moon_spec, "moon")
    assert v_planet != v_moon
    planet_keys = [(9, 0, i, 0) for i in range(8)]
    moon_keys = [(9, 1, i, 0) for i in range(3)]
    executor.request(planet_keys, "planet")
    executor.request(moon_keys, "moon")
    # Re-requesting what is already queued must not submit it again.
    executor.request(planet_keys, "planet")
    done = _drain(executor, ("planet", "moon"), len(planet_keys) + len(moon_keys))
    executor.shutdown()
    assert sorted(c.key for c in done["planet"]) == planet_keys
    assert sorted(c.key for c in done["moon"]) == moon_keys
    assert len(counted) == len(set(counted)) == len(planet_keys) + len(moon_keys)


def test_round_robin_serves_every_channel(counted, planet_spec, moon_spec):
    executor = ChunkExecutor(backend="thread", workers=1, max_inflight=2)
    executor.set_spec(planet_spec, "planet")
    executor.set_spec(moon_spec, "moon")
    executor.request([(9, 0, i, 0) for i in range(6)], "planet")
    executor.request([(9, 1, 0, 0)], "moon")
    # The planet filled the queue first; the moon gets the next free slot,
    # not the end of the planet's list.
    done = _drain(executor, ("planet", "moon"), 3)
    executor.shutdown()
    assert done["moon"], "the moon waited behind the whole planet"


def test_a_spec_change_on_one_channel_keeps_the_other(planet_spec, moon_spec):
    executor = ChunkExecutor(backend="inline", inline_budget=4)
    executor.set_spec(planet_spec, "planet")
    executor.set_spec(moon_spec, "moon")
    executor.request([(10, 0, 0, 0)], "planet")
    executor.request([(10, 1, 0, 0)], "moon")
    executor.set_spec(replace(planet_spec, seed=3, sea_bias=None), "planet")
    assert executor.collect("planet") == []
    assert [c.key for c in executor.collect("moon")] == [(10, 1, 0, 0)]


# ── barren planets ─────────────────────────────────────────────────────────────


def test_barren_planets_grow_nothing():
    rng = np.random.default_rng(0)
    w = biomes.weights(
        rng.uniform(-30, 40, 500),
        rng.uniform(0, 1, 500),
        rng.uniform(-100, 5000, 500),
        rng.uniform(0, 70, 500),
        False,
        vegetation=False,
    )
    assert np.all(w[:, biomes.GRASS] == 0.0) and np.all(w[:, biomes.FOREST] == 0.0)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)


def test_the_moon_is_sand(moon_spec):
    chunk = generate_chunk(moon_spec, (6, 2, 10, 20))
    w = chunk.weights.astype(float) / 255.0
    sand_or_rock = w[:, biomes.DESERT] + w[:, biomes.ROCK]
    assert sand_or_rock.min() > 0.99
    assert w[:, biomes.DESERT].mean() > 0.8


# ── terrains placed in one world ───────────────────────────────────────────────


@pytest.fixture
def two_bodies(showbase, planet_spec, moon_spec):
    from world_builder.terrain.terrain import Terrain

    # Coarse (no level finer than 7) to keep the inline generation short.
    executor = ChunkExecutor(backend="inline", inline_budget=64)
    planet = Terrain(showbase.render, planet_spec, executor=executor, min_level=7)
    moon_position = np.array([3.0e6, 9.0e6, 1.0e6])
    moon = Terrain(
        showbase.render,
        moon_spec,
        executor=executor,
        position=moon_position,
        haze_distance=None,
        materials=planet.materials,
        min_level=7,
    )
    yield planet, moon
    planet.destroy()
    moon.destroy()


def _settle(terrains, camera, frames=60):
    for _ in range(frames):
        for terrain in terrains:
            terrain.update(camera)


def test_bodies_are_placed_in_one_world(showbase, two_bodies):
    planet, moon = two_bodies
    up = TangentFrame.at(30.0, 20.0).up
    camera = up * (planet.radius + 2000.0)
    _settle((planet, moon), camera, frames=12)
    assert planet.shown and moon.shown
    # Every drawn moon chunk sits where the moon is, relative to the camera
    # at the render origin.
    for key in moon.shown:
        node = moon.nodes[key]
        expected = moon.store[key].centre + moon.position - camera
        np.testing.assert_allclose(
            node.get_pos(showbase.render), expected, rtol=1e-6, atol=1.0
        )
    # Seen from the planet, the moon is a handful of coarse chunks.
    assert max(k[0] for k in moon.shown) >= moon.grid.max_level - 2
    assert len(moon.shown) <= 30


def test_a_planet_seen_from_afar_releases_its_ground(showbase, two_bodies, monkeypatch):
    from world_builder.terrain import terrain as terrain_module
    from world_builder.terrain.terrain import RESIDENT_FACTOR

    monkeypatch.setattr(terrain_module, "MIN_RESIDENT", 16)
    planet, moon = two_bodies
    camera = TangentFrame.at(30.0, 20.0).up * (planet.radius + 5.0)
    _settle((planet, moon), camera, frames=40)
    near_ground = len(planet.store)
    assert near_ground > 64
    # Off to the moon: the planet shrinks to what its distant view needs.
    camera = moon.position + TangentFrame.at(0.0, 0.0).up * (moon.radius + 5.0)
    _settle((planet, moon), camera, frames=30)
    limit = max(16, RESIDENT_FACTOR * len(planet._keep))
    assert len(planet.store) <= limit < near_ground
    assert len(planet.nodes) <= limit
