"""
Several planets in one world (world_builder.terrain): the shared worker pool,
per-planet positions, and memory released by planets seen from afar.
"""

import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from timing import TIME_SCALE, unbudgeted, wait_until

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

    def all_done():
        for channel in channels:
            done[channel] += executor.collect(channel)
        return sum(len(v) for v in done.values()) >= count

    wait_until(all_done, timeout, "chunks never arrived", interval=0.005)
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
    # The planet filled the queue first; the moon gets a slot in the first
    # refill, not the end of the planet's list. One worker runs tasks in
    # submission order, so `counted` is that order whatever the timing: the
    # moon runs third (one slot freed) or fourth (both freed by one collect).
    _drain(executor, ("planet", "moon"), 7)
    executor.shutdown()
    order = [name for name, _ in counted]
    assert order.index("moon") <= 3, "the moon waited behind the whole planet"


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
    from world_builder.terrain.terrain import Terrain, TerrainSettings

    # Coarse (no level finer than 7) to keep the inline generation short.
    executor = ChunkExecutor(backend="inline", inline_budget=64)
    coarse = unbudgeted(TerrainSettings(min_level=7))
    planet = Terrain(showbase.render, planet_spec, executor=executor, settings=coarse)
    moon_position = np.array([3.0e6, 9.0e6, 1.0e6])
    moon = Terrain(
        showbase.render,
        moon_spec,
        executor=executor,
        position=moon_position,
        materials=planet.materials,
        settings=coarse,
    )
    yield planet, moon
    planet.destroy()
    moon.destroy()


def _settle(terrains, camera, frames=60):
    """Step frames (a pass every few frames while uploads arrive, and six
    uploads a frame: refining takes a few frames per level).

    ``Terrain._upload`` also stops at ``upload_ms_per_frame`` of wall-clock
    time, so a fixed count only does the same work on every machine for a
    terrain built with :func:`timing.unbudgeted` settings (issue #28)."""
    for _ in range(frames):
        for terrain in terrains:
            terrain.update(camera)


def _settle_until_loaded(terrain, camera, timeout=60.0, stable_frames=5):
    """Update until ``terrain.loading`` has stayed False for ``stable_frames``
    frames. Waits on the state, not a frame count (the uploads' wall-clock budget
    makes the frames needed depend on the machine). Fails after ``timeout`` s."""
    deadline = time.perf_counter() + timeout * TIME_SCALE
    stable = 0
    while stable < stable_frames:
        assert time.perf_counter() < deadline, "terrain still loading"
        terrain.update(camera)
        stable = 0 if terrain.loading else stable + 1


def test_bodies_are_placed_in_one_world(showbase, two_bodies):
    planet, moon = two_bodies
    up = TangentFrame.at(30.0, 20.0).up
    camera = up * (planet.radius + 2000.0)
    _settle((planet, moon), camera, frames=40)
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
    _settle((planet, moon), camera, frames=120)
    near_ground = len(planet.store)
    assert near_ground > 64
    # Off to the moon: the planet shrinks to what its distant view needs.
    camera = moon.position + TangentFrame.at(0.0, 0.0).up * (moon.radius + 5.0)
    _settle((planet, moon), camera, frames=60)
    limit = max(16, RESIDENT_FACTOR * len(planet._keep))
    assert len(planet.store) <= limit < near_ground
    assert len(planet.nodes) <= limit


def test_chunks_a_query_generates_join_the_terrain_store(
    showbase, moon_spec, monkeypatch
):
    """A chunk generated on the query path (generate=True) is stored like one
    from the workers: with its bounds, so the selector knows its heights and
    a later eviction finds them."""
    from world_builder.terrain import terrain as terrain_module
    from world_builder.terrain.terrain import Terrain, TerrainSettings

    # Keep no more than the selection needs, so flying on evicts.
    monkeypatch.setattr(terrain_module, "MIN_RESIDENT", 16)
    monkeypatch.setattr(terrain_module, "RESIDENT_FACTOR", 1)
    executor = ChunkExecutor(backend="inline", inline_budget=64)
    terrain = Terrain(
        showbase.render,
        moon_spec,
        executor=executor,
        settings=unbudgeted(TerrainSettings(min_level=7)),
    )
    try:
        camera = TangentFrame.at(30.0, 20.0).up * (terrain.radius + 2000.0)
        _settle((terrain,), camera, frames=40)
        # The far side: nothing there is generated yet.
        far = -camera / np.linalg.norm(camera)
        key = terrain.query._key_of(far[None, :], terrain.query.level)
        assert key not in terrain.store
        terrain.query.sphere_contact(far * terrain.radius, 5.0)  # generate=True
        chunk = terrain.store[key]
        assert terrain._bounds[key] == (chunk.h_min, chunk.h_max)
        # Fly around the moon, away from it, until the store evicts it.
        for lon in range(20, -110, -15):
            ground = TangentFrame.at(-30.0, float(lon)).up
            _settle((terrain,), ground * (terrain.radius + 2000.0), frames=15)
        assert key not in terrain.store
        assert set(terrain.store) == set(terrain._bounds)
    finally:
        terrain.destroy()
        executor.shutdown()


def test_terrain_reports_loading(showbase, moon_spec):
    """loading holds while the view still wants ground, for a game's loading
    phase: undrawn ground is not solid either."""
    from world_builder.terrain.terrain import Terrain, TerrainSettings

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    terrain = Terrain(
        showbase.render,
        moon_spec,
        executor=executor,
        settings=TerrainSettings(min_level=7),
    )
    try:
        assert terrain.loading
        camera = TangentFrame.at(30.0, 20.0).up * (terrain.radius + 2000.0)
        _settle_until_loaded(terrain, camera)
        assert not terrain.loading
        # Unseen ground: loading again until it is drawn.
        elsewhere = TangentFrame.at(-30.0, -100.0).up * (terrain.radius + 2000.0)
        terrain.update(elsewhere)
        assert terrain.loading
        _settle_until_loaded(terrain, elsewhere)
        assert not terrain.loading
    finally:
        terrain.destroy()
        executor.shutdown()


def test_uploads_are_shown_a_few_frames_later_whenever_they_land(
    showbase, moon_spec, monkeypatch
):
    """Regression (#20): an upload landing one frame after a pass waited for
    the camera to move or the 15-frame fallback, not the 3 frames meant."""
    from world_builder.terrain.terrain import Terrain, TerrainSettings

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    terrain = Terrain(
        showbase.render,
        moon_spec,
        executor=executor,
        settings=TerrainSettings(min_level=7),
    )
    try:
        camera = TangentFrame.at(30.0, 20.0).up * (terrain.radius + 2000.0)
        terrain.update(camera)  # the first pass
        passes = []
        select = terrain._select
        monkeypatch.setattr(
            terrain, "_select", lambda cam: (passes.append(terrain._frame), select(cam))
        )
        first = terrain._frame
        uploads = iter([1])  # one upload, in the frame after the pass
        monkeypatch.setattr(terrain, "_upload", lambda: next(uploads, 0))
        for _ in range(6):
            terrain.update(camera)
        every = TerrainSettings().select_every_on_upload
        assert passes == [first + every]
    finally:
        terrain.destroy()
        executor.shutdown()


# ── pool size ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cores, expected", [(1, 1), (2, 1), (4, 3), (8, 7)])
def test_default_workers_leave_a_core_and_never_drop_below_one(
    monkeypatch, cores, expected
):
    monkeypatch.setattr(workers, "physical_cores", lambda: cores)
    assert workers.default_workers() == expected


@pytest.mark.parametrize("cpus, expected", [(1, 1), (2, 1), (3, 1), (8, 6)])
def test_default_workers_without_a_core_count(monkeypatch, cpus, expected):
    monkeypatch.setattr(workers, "physical_cores", lambda: None)
    monkeypatch.setattr(workers, "usable_cpus", lambda: cpus)
    assert workers.default_workers() == expected


@pytest.mark.parametrize("asked", [0, -3])
def test_a_pool_always_has_a_worker(asked):
    executor = ChunkExecutor(backend="thread", workers=asked)
    try:
        assert executor.workers == 1 and executor.max_inflight >= 1
    finally:
        executor.shutdown()


def _square(spec, key, version):
    """A stand-in generator: any top-level function(spec, key, version)."""
    from types import SimpleNamespace

    return SimpleNamespace(key=key, version=version, value=spec * key[0])


def test_a_channel_runs_its_own_generator(planet_spec):
    executor = ChunkExecutor(backend="thread", workers=2)
    executor.set_spec(planet_spec, "terrain")
    executor.set_spec(3, "custom", generator=_square)
    executor.request([(4, 0, 0, 0)], "custom")
    done = []

    def arrived():
        done.extend(executor.collect("custom"))
        return bool(done)

    try:
        wait_until(arrived, timeout=10.0, msg="chunk never arrived")
    finally:
        executor.shutdown()
    assert done[0].value == 12


def test_chunks_keep_their_place_across_a_rebase(showbase, two_bodies):
    """A rebase moves the cells, not the chunks: every drawn chunk stays
    where it is relative to the camera, to float32 precision (millimetres
    under a cell, centimetres for the coarse chunks under the planet's)."""
    planet, _ = two_bodies
    frame = TangentFrame.at(-12.0, 40.0)
    cameras = [
        frame.up * (planet.radius + 1500.0) + frame.east * east
        for east in (0.0, 3000.0, 9000.0)
    ]
    origins = []
    for camera in cameras:
        _settle((planet,), camera, frames=30)
        origins.append(planet._origin.copy())
        fine = 0
        for key in planet.shown:
            node = planet.nodes[key]
            expected = planet.store[key].centre - camera
            under_cell = key[0] < planet._cell_level
            fine += under_cell
            np.testing.assert_allclose(
                node.get_pos(showbase.render),
                expected,
                rtol=0,
                atol=0.02 if under_cell else 0.2,
            )
        assert fine > 0
    assert any(np.any(a != b) for a, b in zip(origins, origins[1:]))
