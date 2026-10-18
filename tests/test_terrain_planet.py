"""
Unit tests for the terrain prototype's planet, biomes, chunks and LOD
(world_builder.terrain). Pure NumPy; no window.
"""

import pickle
from dataclasses import replace

import numpy as np
import pytest

from world_builder.terrain import biomes
from world_builder.terrain.chunkgen import (
    LEAF_SIZE,
    QUADS,
    VERTEX_DTYPE,
    VERTS,
    chunk_size,
    generate_chunk,
    parent_interpolation,
    skirt_ring,
    triangle_indices,
)
from world_builder.terrain.planet import PRESETS, Planet
from world_builder.terrain.quadtree import LodSelector, parent
from world_builder.terrain.workers import ChunkExecutor


@pytest.fixture(scope="module")
def spec():
    return replace(PRESETS["earthlike"], seed=19).resolved()


@pytest.fixture(scope="module")
def planet(spec):
    return Planet(spec)


# ── planet ─────────────────────────────────────────────────────────────────────


def test_spec_is_picklable_and_hashable(spec):
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert hash(spec) == hash(replace(spec))


def test_sea_bias_sets_the_ocean_fraction(spec, planet):
    rng = np.random.default_rng(99)
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    continent = planet.low_band(dirs)[:, 3]
    assert np.mean(continent < 0.0) == pytest.approx(
        spec.elevation.ocean_fraction, abs=0.03
    )


def test_interpolated_low_band_matches_direct(planet):
    """The lattice interpolation is invisible: centimetres of elevation."""
    xs = np.arange(-1500.0, 1500.0, 16.0) + 300.0
    ys = np.arange(-800.0, 800.0, 16.0)
    interpolated = planet._interpolated_low_band(xs, ys)
    gx, gy = np.meshgrid(xs, ys)
    dirs = planet.chart.to_dirs(gx, gy)
    direct = planet.low_band(dirs)
    e_i, _, _ = planet.evaluate(dirs, interpolated, 64.0)
    e_d, _, _ = planet.evaluate(dirs, direct, 64.0)
    assert np.abs(e_i - e_d).max() < 0.05


def test_sample_grid_is_deterministic(planet):
    a = planet.sample_grid(1000.0, -2000.0, 4.0, 33)
    b = planet.sample_grid(1000.0, -2000.0, 4.0, 33)
    np.testing.assert_array_equal(a.elevation, b.elevation)


def test_temperature_falls_with_altitude(planet):
    t = np.array([15.0, 15.0])
    cooled = planet.lapse(t, np.array([0.0, 2000.0]))
    assert cooled[1] == pytest.approx(15.0 - 13.0)


@pytest.mark.parametrize(
    "preset, expected",
    [("desert", biomes.DESERT), ("forest", biomes.FOREST), ("ice", biomes.SNOW)],
)
def test_presets_bias_the_biomes(preset, expected):
    planet = Planet(replace(PRESETS[preset], seed=2))
    s = planet.sample_grid(-200e3, -200e3, 4000.0, 101)
    slope = biomes.slope_degrees(s.elevation, 4000.0)
    w = biomes.weights(
        s.temperature, s.humidity, s.elevation, slope, planet.spec.has_ocean
    )
    land = s.elevation.ravel() > 0.0
    counts = np.bincount(biomes.dominant(w[land]), minlength=len(biomes.BIOMES))
    assert counts.argmax() == expected


# ── biomes ─────────────────────────────────────────────────────────────────────


def _weights(t, h, z, slope, ocean=True):
    return biomes.weights(
        np.array([t]), np.array([h]), np.array([z]), np.array([slope]), ocean
    )[0]


def test_weights_are_normalised():
    rng = np.random.default_rng(0)
    w = biomes.weights(
        rng.uniform(-30, 40, 1000),
        rng.uniform(0, 1, 1000),
        rng.uniform(-100, 5000, 1000),
        rng.uniform(0, 70, 1000),
        True,
    )
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert w.min() >= 0.0


def test_biome_rules():
    assert _weights(-20, 0.5, 3000, 5).argmax() == biomes.SNOW
    assert _weights(15, 0.5, 300, 60).argmax() == biomes.ROCK
    assert _weights(-20, 0.5, 3000, 60).argmax() == biomes.ROCK  # cliffs shed snow
    assert _weights(30, 0.05, 300, 5).argmax() == biomes.DESERT
    assert _weights(15, 0.8, 300, 5).argmax() == biomes.FOREST
    assert _weights(15, 0.35, 300, 5).argmax() == biomes.GRASS
    assert _weights(15, 0.5, 1.0, 3).argmax() == biomes.BEACH
    assert _weights(15, 0.5, 1.0, 3, ocean=False).argmax() != biomes.BEACH


def test_slope_of_a_plane():
    x = np.arange(20) * 10.0
    plane = np.tile(x, (20, 1))  # rises 1 m per metre: 45 degrees
    slope = biomes.slope_degrees(plane, 10.0, step=2)
    np.testing.assert_allclose(slope[5:-5, 5:-5], 45.0)


# ── chunks ─────────────────────────────────────────────────────────────────────


def test_chunk_layout(spec):
    chunk = generate_chunk(spec, (0, 3, -4))
    vertices = np.frombuffer(chunk.vertex_bytes, VERTEX_DTYPE)
    assert len(vertices) == VERTS * VERTS + 4 * QUADS
    assert chunk.elevation.shape == (VERTS, VERTS)
    np.testing.assert_allclose(
        vertices["vertex"][: VERTS * VERTS, 2], chunk.elevation.ravel(), atol=1e-3
    )
    assert vertices["vertex"][:, 0].min() == -LEAF_SIZE / 2
    assert vertices["vertex"][:, 0].max() == LEAF_SIZE / 2
    np.testing.assert_allclose(
        np.linalg.norm(vertices["normal"], axis=1), 1.0, atol=1e-5
    )
    skirt = vertices[VERTS * VERTS :]
    assert np.all(skirt["biome_b"][:, 3] == 255)
    assert np.all(skirt["vertex"][:, 2] < vertices["vertex"][skirt_ring(), 2])


def test_indices_reference_valid_vertices():
    indices = triangle_indices()
    assert indices.max() == VERTS * VERTS + 4 * QUADS - 1
    assert len(indices) % 3 == 0


def test_neighbour_chunks_share_edges(spec):
    """Same-level neighbours compute the same heights on their shared edge."""
    for level in (0, 3):
        west = generate_chunk(spec, (level, 5, 2))
        east = generate_chunk(spec, (level, 6, 2))
        np.testing.assert_allclose(
            west.elevation[:, -1], east.elevation[:, 0], atol=1e-6
        )
        south = generate_chunk(spec, (level, 5, 1))
        np.testing.assert_allclose(
            south.elevation[-1, :], west.elevation[0, :], atol=1e-6
        )


def test_parent_interpolation_keeps_even_samples():
    e = np.random.default_rng(0).normal(size=(VERTS, VERTS))
    m = parent_interpolation(e)
    np.testing.assert_array_equal(m[::2, ::2], e[::2, ::2])
    plane = np.add.outer(np.arange(VERTS) * 2.0, np.arange(VERTS) * 3.0)
    np.testing.assert_allclose(parent_interpolation(plane), plane)


# ── LOD selection ──────────────────────────────────────────────────────────────


def _covered_area(keys):
    return sum(chunk_size(k[0]) ** 2 for k in keys)


def test_selection_tiles_without_overlap():
    selector = LodSelector(6371000.0, max_distance=60000.0)
    selection = selector.select(
        (1000.0, -3000.0, 200.0), ready=lambda k: True, bounds=lambda k: (0.0, 300.0)
    )
    draw = set(selection.draw)
    assert not selection.wanted
    # No drawn chunk is an ancestor of another drawn chunk.
    for key in draw:
        node = key
        while node[0] < 12:
            node = parent(node)
            assert node not in draw
    # Fine near the camera, coarse far away.
    assert min(k[0] for k in draw) == 0
    assert max(k[0] for k in draw) >= 7


def test_selection_never_drops_a_parent_before_all_children_are_ready():
    selector = LodSelector(6371000.0, max_distance=60000.0)
    ready = {(12, ix, iy) for ix in range(-1, 2) for iy in range(-1, 2)}
    selection = selector.select(
        (10.0, 10.0, 100.0), ready=ready.__contains__, bounds=lambda k: None
    )
    assert set(selection.draw) <= ready
    assert any(k[0] == 11 for k in selection.wanted)
    assert all(k[0] >= 11 for k in selection.wanted)


def test_selection_wants_coarse_first():
    selector = LodSelector(6371000.0, max_distance=60000.0)
    selection = selector.select(
        (0.0, 0.0, 100.0), ready=lambda k: False, bounds=lambda k: None
    )
    assert selection.draw == []
    levels = [k[0] for k in selection.wanted]
    assert levels[0] == 12
    assert levels == sorted(levels, reverse=True)


def test_inline_executor_generates_and_versions(spec):
    executor = ChunkExecutor(backend="inline", inline_budget=2)
    executor.set_spec(spec)
    executor.request([(9, 0, 0), (9, 1, 0), (9, 2, 0)])
    done = executor.collect()
    assert [c.key for c in done] == [(9, 0, 0), (9, 1, 0)]
    executor.request([(9, 2, 0)])
    executor.set_spec(replace(spec, seed=spec.seed + 1, sea_bias=None))
    assert executor.collect() == []  # the old version's chunk is dropped


def test_thread_executor_completes(spec):
    executor = ChunkExecutor(backend="thread", workers=2)
    executor.set_spec(spec)
    keys = [(10, ix, 0) for ix in range(3)]
    executor.request(keys)
    done = []
    for _ in range(400):
        done += executor.collect()
        if len(done) == len(keys):
            break
        import time

        time.sleep(0.01)
    executor.shutdown()
    assert sorted(c.key for c in done) == keys


def test_selection_keeps_and_prefetches_every_child_of_split_nodes():
    """Regression: a culled child drifting into view must already be ready.

    Otherwise its parent cannot stay split, and the whole subtree (the fine
    detail under the camera included) collapses to one coarse chunk.
    """
    from world_builder.terrain.quadtree import _children

    selector = LodSelector(6371000.0, max_distance=60000.0)
    camera = (1000.0, -3000.0, 200.0)
    everything = selector.select(camera, ready=lambda k: True, bounds=lambda k: None)
    for key in everything.draw:
        node = key
        while node[0] < 12:
            node = parent(node)
            assert set(_children(node)) <= everything.keep
    split = {parent(k) for k in everything.keep if k[0] < 12}
    children = {k for k in everything.keep if k[0] < 12}  # roots are not children
    culled_children = children - set(everything.draw) - split
    assert culled_children  # some kept children are neither drawn nor split

    drawable = set(everything.draw) | split
    selector.reset()
    partial = selector.select(
        camera, ready=drawable.__contains__, bounds=lambda k: None
    )
    assert set(partial.prefetch) == culled_children - drawable
    assert partial.draw == everything.draw


def test_min_level_drops_the_finest_levels():
    full = LodSelector(6371000.0, max_distance=60000.0)
    coarse = LodSelector(6371000.0, max_distance=60000.0, min_level=3)
    camera, kwargs = (
        (500.0, 500.0, 50.0),
        dict(ready=lambda k: True, bounds=lambda k: None),
    )
    draw_full = full.select(camera, **kwargs).draw
    draw_coarse = coarse.select(camera, **kwargs).draw
    assert min(k[0] for k in draw_full) == 0
    assert min(k[0] for k in draw_coarse) == 3
    assert len(draw_coarse) < len(draw_full)
    # Only the finest levels change: coarse chunks far away are the same.
    assert {k for k in draw_full if k[0] >= 5} == {k for k in draw_coarse if k[0] >= 5}
