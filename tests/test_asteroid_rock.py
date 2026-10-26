"""A rock as a terrain body (asteroids.rock) and the terrain's body protocol:
planets unchanged, rocks generated, bounded, textured and culled as bodies of
their size."""

import math
import os
import pickle
import time

import numpy as np
import pytest

from world_builder.asteroids.rock import ROCK_UV_PERIOD, RockSpec
from world_builder.asteroids.shape import face_grid_dirs, prototype_shape
from world_builder.asteroids.spec import ShapeSpec
from world_builder.terrain import noise
from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chunkgen import generate_chunk, planet_for
from world_builder.terrain.cubesphere import UV_PERIOD, grid_for
from world_builder.terrain.planet import PLANET_HEIGHT_BOUNDS, PRESETS, Planet
from world_builder.terrain.quadtree import LodSelector, NativeLodSelector
from world_builder.terrain.workers import ChunkExecutor

SHAPES = ShapeSpec(prototypes=4)


def _rock(radius: float = 200.0, **kwargs) -> RockSpec:
    return RockSpec(SHAPES, seed=2, index=1, radius=radius, **kwargs)


def test_a_planet_is_its_own_body():
    spec = PRESETS["moon"]
    planet = spec.evaluator()
    assert isinstance(planet, Planet)
    assert planet.height_bounds == PLANET_HEIGHT_BOUNDS
    assert planet_for(spec.resolved()).spec == spec.resolved()


def test_the_texture_period_follows_the_body():
    # planets: whole UV_PERIODs a face, unchanged
    planet = grid_for(600e3)
    assert planet.uv_unit * planet.n0 % UV_PERIOD == pytest.approx(0.0)
    # a small rock: whole rock periods, about its arc (not 4 km of texture)
    rock = grid_for(200.0, ROCK_UV_PERIOD)
    span = rock.uv_unit * rock.n0
    assert span % ROCK_UV_PERIOD == pytest.approx(0.0)
    assert 0.5 < span / (0.5 * math.pi * 200.0) < 1.5
    with pytest.raises(ValueError):
        grid_for(200.0, 100.0)  # must divide UV_PERIOD


def test_the_rock_is_its_prototype_scaled_with_its_own_detail():
    spec = _rock(detail_m=0.0)
    body = spec.evaluator()
    shape = prototype_shape(SHAPES, 2, 1)
    dirs = face_grid_dirs(16).reshape(-1, 3)
    np.testing.assert_allclose(
        body.elevation(dirs), 200.0 * shape.radius(dirs) - 200.0, rtol=1e-12
    )
    detailed = _rock(detail_m=1.5, detail_seed=7).evaluator().elevation(dirs)
    assert 0.0 < np.max(np.abs(detailed - body.elevation(dirs))) <= 1.5
    # another rock's detail differs
    other = _rock(detail_m=1.5, detail_seed=8).evaluator().elevation(dirs)
    assert not np.array_equal(detailed, other)


def test_the_rock_sets_its_own_biomes():
    body = _rock(ice=0.3).evaluator()
    grid = body.grid
    ka = kb = np.arange(0, grid.n0 + 1, grid.n0 // 8)
    sample = body.sample_face_grid(2, ka, kb, grid.n0 // 8)
    assert sample.weights.shape == (len(kb), len(ka), len(BIOMES))
    np.testing.assert_allclose(sample.weights[..., BIOMES.index("snow")], 0.3)
    np.testing.assert_allclose(sample.weights[..., BIOMES.index("rock")], 0.7)
    np.testing.assert_allclose(sample.weights.sum(axis=-1), 1.0)


@pytest.mark.parametrize("radius", [200.0, 2000.0])
def test_its_chunks_stay_within_its_bounds(radius):
    spec = _rock(radius, ice=0.4, detail_seed=3)
    body = spec.evaluator()
    lo, hi = body.height_bounds
    assert lo < 0.0 < hi
    for key in body.grid.roots():
        chunk = generate_chunk(spec, key)
        assert lo <= chunk.h_min <= chunk.h_max <= hi
        # snow and rock in the rock's share (as bytes)
        w = chunk.weights.reshape(-1, len(BIOMES)).astype(int)
        assert abs(int(w[0, BIOMES.index("snow")]) - round(0.4 * 255)) <= 1
        assert np.all(w[:, 2:] == 0)


@pytest.mark.parametrize("backend", ["inline", "process"])
def test_the_workers_generate_rock_chunks(backend):
    spec = _rock(ice=0.1)
    assert pickle.loads(pickle.dumps(spec)) == spec
    executor = ChunkExecutor(backend=backend, workers=1, inline_budget=8)
    try:
        version = executor.set_spec(spec, "rock")
        roots = spec.evaluator().grid.roots()
        executor.request(roots, "rock")
        done = []
        for _ in range(400):
            done += executor.collect("rock")
            if len(done) == len(roots):
                break
            time.sleep(0.02)
        assert sorted(c.key for c in done) == sorted(roots)
        assert all(c.version == version for c in done)
    finally:
        executor.shutdown()


def _drawn(selector, radius):
    camera = np.array([0.0, 0.0, 2.5 * radius])
    return selector.select(camera, ready=lambda key: True, bounds=lambda key: None).draw


def _faces(selector, radius):
    return {key[1] for key in _drawn(selector, radius)}


@pytest.mark.parametrize("radius", [200.0, 2000.0])
def test_a_small_body_is_culled_by_its_own_bounds(radius):
    """The planets' bounds (-7 km of ground) culled most of a small body,
    or refined a bigger one far too much. Seen from 2.5 radii over +z, the
    +z face and the four around it have ground in view (the back face may
    hide behind the rock's smallest radius)."""
    body = _rock(radius).evaluator()
    planet_bounds = LodSelector(body.grid, horizon_floor=PLANET_HEIGHT_BOUNDS[0])
    own = LodSelector(
        body.grid,
        horizon_floor=body.height_bounds[0],
        unknown_bounds=body.height_bounds,
    )
    assert set(range(5)) <= _faces(own, radius)
    if radius < 1000.0:
        assert not set(range(5)) <= _faces(planet_bounds, radius)  # culled
    else:
        assert len(_drawn(planet_bounds, radius)) > 2 * len(_drawn(own, radius))


def test_the_native_lod_pass_takes_the_bodys_bounds():
    if noise.native_library() is None:
        if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
            pytest.fail("native library required")
        pytest.skip("native library not built")
    body = _rock(800.0).evaluator()
    kwargs = dict(
        horizon_floor=body.height_bounds[0], unknown_bounds=body.height_bounds
    )
    python = LodSelector(body.grid, **kwargs)
    native = NativeLodSelector(body.grid, **kwargs)
    roots = body.grid.roots()
    for key in roots:
        native.set_ready(key, True)
    camera = np.array([0.0, 0.0, 2.5 * 800.0])
    expected = python.select(
        camera, ready=set(roots).__contains__, bounds=lambda key: None
    )
    got = native.select(camera)
    assert got.draw == expected.draw and got.wanted == expected.wanted
    assert set(range(5)) <= {key[1] for key in got.draw}
