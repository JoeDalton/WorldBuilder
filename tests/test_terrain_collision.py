"""
Native terrain collision (issue #16): the CollisionTerrain solid against
TerrainQuery, and Terrain.enable_collision with Panda's traverser.

Skipped when the native library is not built for the running Panda3D
(invoke native), failed instead when WORLD_BUILDER_REQUIRE_NATIVE is set (CI).
"""

import os
from dataclasses import replace

import numpy as np
import pytest
from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionSegment,
    CollisionSphere,
    CollisionTraverser,
    GeomNode,
)

from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chart import AzimuthalEquidistantChart, TangentFrame
from world_builder.terrain.chunkgen import QUADS, generate_chunk
from world_builder.terrain.native import NativeCollisionError, NativeTerrain, _binding
from world_builder.terrain.planet import PRESETS, Planet
from world_builder.terrain.query import TerrainQuery

#: The demo level's reference point: a coast, with a range to the south.
REFERENCE = (38.85, -125.72)
MASK = BitMask32.bit(5)


def _skip_without_native(error):
    if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
        pytest.fail(f"native library required: {error}")
    pytest.skip(str(error))


def _native_or_skip(planet, ocean):
    grid = planet.grid
    try:
        return NativeTerrain(planet.spec.radius, grid.max_level, grid.n0, ocean)
    except NativeCollisionError as error:
        _skip_without_native(error)


@pytest.fixture(scope="module")
def planet():
    spec = replace(
        PRESETS["earthlike"],
        seed=19,
        reference_lat=REFERENCE[0],
        reference_lon=REFERENCE[1],
    )
    return Planet(spec)


@pytest.fixture(scope="module")
def chart(planet):
    spec = planet.spec
    return AzimuthalEquidistantChart(
        spec.reference_lat, spec.reference_lon, spec.radius
    )


def _key_at(planet, direction, level):
    face, i, j, _, _ = planet.grid.locate(direction[None, :], level)
    return level, int(face[0]), int(i[0]), int(j[0])


def _inside(planet, chunk, rng, count):
    """Unit directions inside a chunk, away from its edges."""
    level, face, i, j = chunk.key
    grid = planet.grid
    per = QUADS << level
    ka = (i + rng.uniform(0.1, 0.9, count)) * per
    kb = (j + rng.uniform(0.1, 0.9, count)) * per
    from world_builder.terrain.cubesphere import face_dirs

    return face_dirs(face, grid.param(ka), grid.param(kb))


@pytest.fixture(scope="module")
def land(planet, chart):
    """One level-2 chunk inland (7 m spacing), as both the query and the
    native solid see it."""
    direction = chart.to_dirs(np.array([0.0]), np.array([-8000.0]))[0]
    key = _key_at(planet, direction, 2)
    chunk = generate_chunk(planet.spec, key, 0)
    assert chunk.h_min > 0.0, "the test chunk should be dry land"
    query = TerrainQuery(planet, {key: chunk}, include_ocean=False, level=2)
    native = _native_or_skip(planet, ocean=False)
    native.add_chunk(chunk)
    return chunk, query, native


@pytest.fixture(scope="module")
def sea(planet, chart):
    """One level-4 chunk wholly under the sea, near the reference coast."""
    xs, ys = np.meshgrid(np.linspace(-30e3, 30e3, 13), np.linspace(-30e3, 30e3, 13))
    dirs = chart.to_dirs(xs.ravel(), ys.ravel())
    heights = planet.sample_dirs(dirs, apply_stamps=True)[0]
    for index in np.argsort(heights)[:5]:
        key = _key_at(planet, dirs[index], 4)
        chunk = generate_chunk(planet.spec, key, 0)
        if chunk.h_max < 0.0:
            native = _native_or_skip(planet, ocean=True)
            native.add_chunk(chunk)
            query = TerrainQuery(planet, {key: chunk}, include_ocean=True, level=4)
            return chunk, query, native
    pytest.fail("no sea chunk near the reference point")


def _angle(a, b) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ── the solid against TerrainQuery ─────────────────────────────────────────────


def test_radial_segments_hit_the_drawn_triangles(planet, land):
    chunk, query, native = land
    radius = planet.spec.radius
    for direction in _inside(planet, chunk, np.random.default_rng(1), 20):
        hit = native.segment(
            direction * (radius + 9000.0), direction * (radius - 7000.0)
        )
        assert hit is not None
        _, point, normal = hit
        expected = query.surface_point(direction * radius, generate=False)
        np.testing.assert_allclose(point, expected, atol=1e-3)
        assert (
            _angle(normal, query.normal_at(direction * radius, generate=False)) < 1e-3
        )


def test_slanted_segments_match_segment_intersect(planet, land):
    chunk, query, native = land
    rng = np.random.default_rng(2)
    radius = planet.spec.radius
    ground = query.height_at(_inside(planet, chunk, rng, 30) * radius, generate=False)
    dirs = _inside(planet, chunk, rng, 30)
    for direction, height in zip(dirs, ground):
        up = direction * (radius + height + 40.0)
        across = np.cross(direction, [0.0, 0.0, 1.0])
        p0 = up + 60.0 * across / np.linalg.norm(across)
        p1 = up - 60.0 * across / np.linalg.norm(across) - 120.0 * direction
        expected = query.segment_intersect(p0, p1, generate=False)
        got = native.segment(p0, p1)
        assert (expected is None) == (got is None)
        if got is not None:
            np.testing.assert_allclose(got[1], expected[1], atol=0.01)
            assert _angle(got[2], expected[2]) < 0.01


def test_segment_above_the_ground_misses(planet, land):
    chunk, _, native = land
    direction = _inside(planet, chunk, np.random.default_rng(3), 1)[0]
    top = direction * (planet.spec.radius + chunk.h_max + 50.0)
    assert native.segment(top, top + 30.0 * direction) is None


def test_sphere_contacts_lie_on_the_nearest_triangle(planet, land):
    """Where sphere_contact keeps the triangle under the centre, the solid
    slides to the nearest one on steep ground: its contact is on the surface,
    with that triangle's normal, and no farther than sphere_contact's."""
    chunk, query, native = land
    radius = planet.spec.radius
    dirs = _inside(planet, chunk, np.random.default_rng(4), 20)
    heights = query.height_at(dirs * radius, generate=False)
    for direction, height in zip(dirs, heights):
        touching = direction * (radius + height + 3.0)
        expected = query.sphere_contact(touching, 5.0, generate=False)
        got = native.sphere(touching, 5.0)
        assert expected is not None and got is not None
        point, normal = got
        assert abs(float(query.altitude(point, generate=False))) < 1e-3
        assert _angle(normal, query.normal_at(point, generate=False)) < 0.01
        nearest = np.linalg.norm(touching - expected[0])
        assert np.linalg.norm(touching - point) <= nearest + 0.05
        clear = direction * (radius + height + 30.0)
        assert native.sphere(clear, 5.0) is None


def test_material_matches_the_query(planet, land):
    chunk, query, native = land
    radius = planet.spec.radius
    dirs = _inside(planet, chunk, np.random.default_rng(5), 20)
    points = query.surface_point(dirs * radius, generate=False)
    for point in points:
        assert BIOMES[native.material(point)] == query.material_at(
            point, generate=False
        )


def test_nothing_registered_there(planet, land):
    _, _, native = land
    far = -planet.spec.radius * np.array([0.0, 0.0, 1.0])
    assert native.segment(far * 1.01, far * 0.99) is None
    assert native.sphere(far, 10.0) is None
    assert native.material(far) == -2


def test_the_sea_surface_is_solid(planet, sea):
    chunk, query, native = sea
    radius = planet.spec.radius
    for direction in _inside(planet, chunk, np.random.default_rng(6), 10):
        hit = native.segment(direction * (radius + 100.0), direction * (radius - 100.0))
        assert hit is not None
        _, point, normal = hit
        assert np.linalg.norm(point) == pytest.approx(radius, abs=1e-3)
        assert _angle(normal, direction) < 1e-3
        assert native.material(point) == -1
        assert query.material_at(point, generate=False) == "water"
        contact = native.sphere(direction * (radius + 2.0), 5.0)
        assert contact is not None
        assert _angle(contact[1], direction) < 1e-3


def test_a_panda_series_without_a_build_is_refused(monkeypatch):
    monkeypatch.setattr(_binding, "panda_series", lambda: (1, 99))
    _binding._library.cache_clear()
    try:
        with pytest.raises(NativeCollisionError, match="not built for Panda3D"):
            NativeTerrain(1000.0, 1, QUADS * 2, False)
    finally:
        monkeypatch.undo()
        _binding._library.cache_clear()


# ── Terrain.enable_collision with the traverser ────────────────────────────────


@pytest.fixture(scope="module")
def moon(showbase):
    from world_builder.terrain.terrain import Terrain, TerrainSettings
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=16)
    terrain = Terrain(
        showbase.render,
        PRESETS["moon"],
        executor=executor,
        settings=TerrainSettings(min_level=7),
    )
    camera = TangentFrame.at(10.0, 20.0).up * (terrain.radius + 2000.0)
    _settle(terrain, camera)
    try:
        terrain.enable_collision(MASK)
    except NativeCollisionError as error:
        terrain.destroy()
        executor.shutdown()
        _skip_without_native(error)
    terrain.disable_collision()
    yield terrain, camera
    terrain.destroy()
    executor.shutdown()


def _settle(terrain, camera, frames=60):
    for _ in range(frames):
        terrain.update(camera)


@pytest.fixture
def collision(moon):
    terrain, _ = moon
    made = terrain.enable_collision(MASK, name="ground", owner_tag="who")
    yield made
    terrain.disable_collision()


def _traverse(render, *colliders):
    traverser = CollisionTraverser()
    queue = CollisionHandlerQueue()
    holder = render.attach_new_node("colliders")
    for name, solid, position in colliders:
        node = CollisionNode(name)
        node.add_solid(solid)
        node.set_from_collide_mask(MASK)
        node.set_into_collide_mask(BitMask32.all_off())
        path = holder.attach_new_node(node)
        path.set_pos(*position)
        traverser.add_collider(path, queue)
    traverser.traverse(render)
    entries = list(queue.entries)
    holder.remove_node()
    return entries


def test_the_collision_node(showbase, moon, collision):
    terrain, _ = moon
    node = collision.node
    assert node.name == "ground"
    assert node.node().get_into_collide_mask() == MASK
    assert node.node().get_from_collide_mask().is_zero()
    assert node.get_python_tag("who") is collision.surface
    assert collision.surface.terrain is terrain
    # Beside the terrain root, not under it, and placed like it.
    assert node.get_parent().get_parent() == showbase.render
    assert not terrain.root.is_ancestor_of(node)
    assert node.get_parent().get_transform() == terrain.root.get_transform()
    assert collision.registered and collision.registered <= terrain.shown


def test_a_laser_and_a_ship_through_the_traverser(showbase, moon, collision):
    terrain, camera = moon
    up = camera / np.linalg.norm(camera)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    ground = terrain.query.surface_point(camera, generate=False)
    p0 = ground + 30.0 * up - 20.0 * east
    p1 = ground - 30.0 * up + 20.0 * east
    start = terrain.to_render(p0)
    ship = terrain.to_render(ground + 6.0 * up + 200.0 * east)
    entries = _traverse(
        showbase.render,
        ("laser", CollisionSegment(0, 0, 0, *(terrain.to_render(p1) - start)), start),
        ("ship", CollisionSphere(0, 0, 0, 10.0), ship),
    )
    # One entry per collider: one event each for the game.
    assert sorted(e.from_node.name for e in entries) == ["laser", "ship"]
    laser = next(e for e in entries if e.from_node.name == "laser")
    expected = terrain.query.segment_intersect(p0, p1, generate=False)
    np.testing.assert_allclose(
        laser.get_surface_point(showbase.render),
        terrain.to_render(expected[1]),
        atol=0.05,
    )
    assert (
        _angle(np.array(laser.get_surface_normal(showbase.render)), expected[2]) < 0.01
    )
    material = collision.surface.material_at(laser.get_surface_point(showbase.render))
    assert material == terrain.query.material_at(expected[1], generate=False)


def test_material_at_takes_a_frame(showbase, moon, collision):
    terrain, camera = moon
    ground = terrain.query.surface_point(camera, generate=False)
    frame = showbase.render.attach_new_node("game-root")
    frame.set_pos(100.0, -50.0, 7.0)
    local = frame.get_relative_point(showbase.render, tuple(terrain.to_render(ground)))
    assert collision.surface.material_at(local, frame) == terrain.query.material_at(
        ground, generate=False
    )
    frame.remove_node()


def test_only_chunks_within_reach_are_solid(moon):
    terrain, _ = moon
    collision = terrain.enable_collision(MASK, max_distance=0.0)
    try:
        assert 0 < len(collision.registered) < len(terrain.shown)
    finally:
        terrain.disable_collision()


def test_drawn_chunks_are_not_collision_targets(showbase, moon):
    """A collider with GeomNode's default bit would otherwise hit every drawn
    triangle (slowly); only the collision node is solid."""
    terrain, camera = moon
    assert terrain.collision is None
    up = camera / np.linalg.norm(camera)
    ground = terrain.query.surface_point(camera, generate=False)
    start = terrain.to_render(ground + 30.0 * up)
    end = terrain.to_render(ground - 30.0 * up)
    traverser = CollisionTraverser()
    queue = CollisionHandlerQueue()
    node = CollisionNode("laser")
    node.add_solid(CollisionSegment(0, 0, 0, *(end - start)))
    node.set_from_collide_mask(GeomNode.get_default_collide_mask() | MASK)
    node.set_into_collide_mask(BitMask32.all_off())
    path = showbase.render.attach_new_node(node)
    path.set_pos(*start)
    traverser.add_collider(path, queue)
    traverser.traverse(showbase.render)
    path.remove_node()
    assert queue.get_num_entries() == 0


def test_disable_removes_the_node(showbase, moon):
    terrain, _ = moon
    terrain.enable_collision(MASK)
    terrain.disable_collision()
    assert terrain.collision is None
    assert showbase.render.find("**/terrain-collision").is_empty()


def test_collision_follows_a_rebase(showbase, moon, collision):
    terrain, camera = moon
    up = camera / np.linalg.norm(camera)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    moved = camera + 9000.0 * east
    origin = terrain._origin.copy()
    _settle(terrain, moved, frames=30)
    try:
        assert np.any(terrain._origin != origin)
        assert (
            collision.node.get_parent().get_transform() == terrain.root.get_transform()
        )
        ground = terrain.query.surface_point(moved, generate=False)
        dirn = moved / np.linalg.norm(moved)
        p0, p1 = ground + 20.0 * dirn, ground - 20.0 * dirn
        start = terrain.to_render(p0)
        entries = _traverse(
            showbase.render,
            (
                "laser",
                CollisionSegment(0, 0, 0, *(terrain.to_render(p1) - start)),
                start,
            ),
        )
        assert len(entries) == 1
        np.testing.assert_allclose(
            entries[0].get_surface_point(showbase.render),
            terrain.to_render(ground),
            atol=0.05,
        )
    finally:
        _settle(terrain, camera, frames=30)


def test_a_new_planet_resets_the_chunks(moon):
    terrain, camera = moon
    spec = terrain.spec
    collision = terrain.enable_collision(MASK)
    try:
        terrain.set_spec(replace(spec, seed=spec.seed + 1))
        assert collision.registered == frozenset()
        assert len(collision.native) == 0
        _settle(terrain, camera)
        assert collision.registered and collision.registered <= terrain.shown
    finally:
        terrain.disable_collision()
        terrain.set_spec(spec)
        _settle(terrain, camera)


# ── the sea, through Terrain ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ocean_terrain(showbase):
    """An earthlike terrain settled over deep sea."""
    from world_builder.terrain.terrain import Terrain, TerrainSettings
    from world_builder.terrain.workers import ChunkExecutor

    spec = replace(PRESETS["earthlike"], seed=19).resolved()
    planet = Planet(spec)
    dirs = np.random.default_rng(8).normal(size=(4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    heights = planet.sample_dirs(dirs, apply_stamps=True)[0]
    deep = dirs[int(np.argmin(heights))]
    assert heights.min() < -1500.0
    executor = ChunkExecutor(backend="inline", inline_budget=16)
    terrain = Terrain(
        showbase.render, spec, executor=executor, settings=TerrainSettings(min_level=7)
    )
    camera = deep * (terrain.radius + 2000.0)
    _settle(terrain, camera)
    try:
        terrain.enable_collision(MASK)
    except NativeCollisionError as error:
        terrain.destroy()
        executor.shutdown()
        _skip_without_native(error)
    terrain.disable_collision()
    yield terrain, camera
    terrain.destroy()
    executor.shutdown()


def _radial_laser(showbase, terrain, camera):
    up = camera / np.linalg.norm(camera)
    p0 = up * (terrain.radius + 30.0)
    p1 = up * (terrain.radius - 30.0 + float(terrain.query.ground_height_at(camera)))
    start = terrain.to_render(p0)
    entries = _traverse(
        showbase.render,
        ("laser", CollisionSegment(0, 0, 0, *(terrain.to_render(p1) - start)), start),
    )
    assert len(entries) == 1
    return entries[0], up


def test_the_sea_surface_is_solid_through_the_terrain(showbase, ocean_terrain):
    terrain, camera = ocean_terrain
    collision = terrain.enable_collision(MASK)  # include_ocean follows the planet
    try:
        entry, up = _radial_laser(showbase, terrain, camera)
        point = np.array(entry.get_surface_point(showbase.render))
        np.testing.assert_allclose(
            point, terrain.to_render(up * terrain.radius), atol=0.05
        )
        assert _angle(np.array(entry.get_surface_normal(showbase.render)), up) < 0.01
        assert collision.surface.material_at(point) == "water"
    finally:
        terrain.disable_collision()


def test_without_the_sea_the_seabed_is_hit(showbase, ocean_terrain):
    terrain, camera = ocean_terrain
    collision = terrain.enable_collision(MASK, include_ocean=False)
    try:
        entry, up = _radial_laser(showbase, terrain, camera)
        point = np.array(entry.get_surface_point(showbase.render))
        seabed = terrain.query.ground_height_at(camera, generate=False)
        assert float(seabed) < -1000.0
        expected = terrain.to_render(up * (terrain.radius + float(seabed)))
        np.testing.assert_allclose(point, expected, atol=0.05)
        assert collision.surface.material_at(point) != "water"
    finally:
        terrain.disable_collision()
