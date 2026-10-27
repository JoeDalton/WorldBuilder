"""Big rocks drawn as terrain near the camera (asteroids.promotion), and the
terrain's orientation that lets a body turn (terrain.Terrain)."""

import math
import os

import numpy as np
import pytest
from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionSegment,
    CollisionTraverser,
)

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.motion import axis_angle, orientation, quat_multiply
from world_builder.asteroids.promotion import show_distance
from world_builder.asteroids.rock import RockSpec
from world_builder.asteroids.settings import AsteroidSettings
from world_builder.asteroids.shape import prototype_shape
from world_builder.asteroids.spec import Box, Density, FieldSpec, ShapeSpec, SpinSpec
from world_builder.terrain.terrain import Terrain, TerrainSettings
from world_builder.terrain.workers import ChunkExecutor

SHAPES = ShapeSpec(prototypes=2)


def _native_or_skip():
    from world_builder.terrain.native import NativeCollisionError, NativeTerrain

    try:
        NativeTerrain(100.0, 0, 64, False).close()
    except NativeCollisionError as error:
        if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
            pytest.fail(str(error))
        pytest.skip(str(error))


# ---------------------------------------------------------------------------
# A turned terrain
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rock_terrain(showbase):
    executor = ChunkExecutor(backend="inline", inline_budget=16)
    spec = RockSpec(SHAPES, seed=1, index=0, radius=300.0, ice=0.2)
    turn = axis_angle(np.array([0.36, 0.48, 0.8]), 1.1)
    made = Terrain(
        showbase.render,
        spec,
        executor=executor,
        position=(1.0e5, -2.0e4, 3.0e3),
        orientation=turn,
        settings=TerrainSettings(min_level=1),
    )
    yield made
    made.destroy()
    executor.shutdown()


def test_a_turned_terrains_frames(rock_terrain):
    t = rock_terrain
    rotation = t.rotation
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    p = np.array([10.0, -250.0, 40.0])
    world = t.position + rotation @ p
    np.testing.assert_allclose(t.to_planet(world), p, atol=1e-9)
    camera = world + np.array([0.0, 0.0, 500.0])
    t.update(camera, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(t._camera, t.to_planet(camera), atol=1e-9)
    # a planet point in render: turned, relative to the camera
    np.testing.assert_allclose(
        t.to_render(p), rotation @ (p - t._camera) + [1.0, 2.0, 3.0], atol=1e-9
    )
    # the root is turned with the body
    quat = t.root.get_quat()
    np.testing.assert_allclose(
        abs(np.dot([quat[0], quat[1], quat[2], quat[3]], t.orientation)), 1.0, atol=1e-6
    )


def test_a_turned_terrain_is_solid_where_it_is_drawn(showbase, rock_terrain):
    """A segment fired at a turned rock in render hits where the body's own
    query hits, turned: collision follows the body's orientation."""
    _native_or_skip()
    t = rock_terrain
    t.enable_collision(BitMask32.bit(7), name="rock")
    try:
        camera = t.position + np.array([0.0, 0.0, 900.0])
        for _ in range(60):
            t.update(camera)
            if not t.loading:
                break
        assert not t.loading
        render = showbase.render
        start = t.position + np.array([40.0, 25.0, 700.0]) - camera
        end = t.position - camera
        node = CollisionNode("probe")
        node.add_solid(CollisionSegment(*start, *end))
        node.set_from_collide_mask(BitMask32.bit(7))
        node.set_into_collide_mask(BitMask32.all_off())
        probe = render.attach_new_node(node)
        try:
            queue = CollisionHandlerQueue()
            traverser = CollisionTraverser()
            traverser.add_collider(probe, queue)
            traverser.traverse(render)
            assert queue.get_num_entries() == 1
            hit = np.array(queue.get_entry(0).get_surface_point(render))
        finally:
            probe.remove_node()
        # the native solid's own answer, planet frame
        a = t.to_planet(start + camera)
        b = t.to_planet(end + camera)
        _, point, _ = t.collision.native.segment(a, b)
        expected = t.rotation @ point + t.position - camera
        np.testing.assert_allclose(hit, expected, atol=0.05)
    finally:
        t.disable_collision()


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_the_terrain_is_finer_within_the_show_distance():
    # 16 x split 2 x pi R / 128 texels: about 0.8 R
    assert show_distance(1000.0, 128, 2.0) == pytest.approx(32 * math.pi * 1000 / 128)
    assert show_distance(1000.0, 256, 2.0) == pytest.approx(
        show_distance(500.0, 128, 2.0)
    )


SPEC = FieldSpec(
    shape=Box((2500.0, 2500.0, 2500.0)),
    density=Density(3.0),
    sizes=Distribution.uniform(250.0, 450.0),
    shapes=SHAPES,
    spin=SpinSpec(Distribution.constant(0.05), size_exponent=0.0),
)


@pytest.fixture(scope="module")
def field(showbase):
    from world_builder.asteroids.field import AsteroidField

    executor = ChunkExecutor(backend="inline", inline_budget=16)
    clock = {"t": 2.0}
    made = AsteroidField(
        showbase,
        SPEC,
        clock=lambda: clock["t"],
        settings=AsteroidSettings(
            max_lod=16, normal_size=64, texture_size=256, max_terrain_rocks=2
        ),
        executor=executor,
        position=(5.0e4, 1.0e4, -2.0e4),
        orientation=axis_angle(np.array([0.0, 0.6, 0.8]), 0.7),
    )
    made.enable_collision(BitMask32.bit(6), name="terrain")
    for _ in range(10):
        made.update(made.position, cameras=[])
        if not made.loading:
            break
    assert not made.loading
    yield made, clock
    made.destroy()
    executor.shutdown()


def _fly(field, camera_field, frames=80):
    """Put the camera at a field-frame point and run frames until the rocks'
    terrains have settled."""
    world = field.position + field.rotation @ np.asarray(camera_field, dtype=float)
    for i in range(frames):
        field.update(world, cameras=[])
        # (the rocks near come from the last frame's visible pass)
        if i > 1 and all(not t.loading for t in field.terrain_rocks.terrains):
            break
    field.update(world, cameras=[])


def _near_rock(field, index=0):
    pop = field.population
    slots = np.flatnonzero(pop.live)
    order = np.argsort(np.linalg.norm(pop.rocks["centre"][slots], axis=1))
    slot = int(slots[order[index]])
    return slot, pop.rocks[slot]


def test_far_rocks_stay_instances(field):
    made, _ = field
    _fly(made, [0.0, 0.0, 2.0e5])
    assert not made.terrain_rocks.promoted.any()
    assert made.terrain_rocks.terrains == []


def test_a_near_rock_becomes_terrain(field):
    made, clock = field
    slot, rock = _near_rock(made)
    show = show_distance(rock["radius"], 64, 2.0)
    direction = np.array([0.0, -1.0, 0.2]) / np.linalg.norm([0.0, -1.0, 0.2])
    _fly(made, rock["centre"] + direction * (rock["radius"] + 0.5 * show))
    assert made.terrain_rocks.promoted[slot]
    (terrain,) = [
        p.terrain for p in made.terrain_rocks._rocks.values() if p.slot == slot
    ]
    # at the rock's place, turned as the rock is at the field's time
    np.testing.assert_allclose(
        terrain.position, made.position + made.rotation @ rock["centre"], atol=1e-6
    )
    q = orientation(rock["q0"], rock["axis"], rock["precession"], rock["spin"], 2.0)
    expected = quat_multiply(made.orientation / np.linalg.norm(made.orientation), q)
    assert abs(np.dot(terrain.orientation, expected)) == pytest.approx(1.0, abs=1e-9)
    # solid as terrain, not as an instance
    assert terrain.collision is not None
    assert not made.collision._registered[slot]
    # its spec is the rock's
    assert terrain.spec.radius == pytest.approx(rock["radius"])
    assert terrain.spec.index == rock["proto"]


def test_leaving_gives_the_rock_back_to_its_instance(field):
    made, _ = field
    slot, rock = _near_rock(made)
    show = show_distance(rock["radius"], 64, 2.0)
    direction = np.array([0.0, -1.0, 0.2]) / np.linalg.norm([0.0, -1.0, 0.2])
    _fly(made, rock["centre"] + direction * (rock["radius"] + 0.5 * show))
    assert made.terrain_rocks.promoted[slot]
    _fly(made, rock["centre"] + direction * (rock["radius"] + 1.5 * show), frames=2)
    assert not made.terrain_rocks.promoted[slot]  # hidden past the show distance
    assert made.collision._registered[slot]  # solid as an instance again
    pool = len(made.terrain_rocks._pool)
    _fly(made, [0.0, 0.0, 2.0e5], frames=2)
    assert made.terrain_rocks.terrains == []
    # back near: the pooled terrains are reused
    _fly(made, rock["centre"] + direction * (rock["radius"] + 0.5 * show))
    assert len(made.terrain_rocks._pool) == pool


def test_no_more_rocks_than_allowed_turn_into_terrain(field):
    made, _ = field
    made.set_settings(
        AsteroidSettings(
            max_lod=16,
            normal_size=64,
            texture_size=256,
            max_terrain_rocks=1,
            terrain_min_radius=50.0,
        )
    )
    try:
        _fly(made, [0.0, 0.0, 0.0])  # among the rocks
        assert len(made.terrain_rocks.terrains) <= 1
    finally:
        made.set_settings(
            AsteroidSettings(
                max_lod=16, normal_size=64, texture_size=256, max_terrain_rocks=2
            )
        )


def test_the_terrain_continues_the_instance():
    """The rock's terrain at its coarsest level is the instanced rock: the
    finest instanced mesh lies within the prototype's error of it."""
    from world_builder.asteroids.prototype import PrototypeJob, generate_prototype

    prototype = generate_prototype(
        PrototypeJob(SHAPES, 1, lods=(8, 16, 32), normal_size=64), 0
    )
    mesh = prototype.meshes[-1]
    radius = 400.0
    body = RockSpec(SHAPES, seed=1, index=0, radius=radius, detail_m=1.5).evaluator()
    dirs = mesh.vertices.astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    grid = body.grid
    root_band = 4.0 * grid.spacing0 * (1 << grid.max_level)
    terrain_radius = radius + body.elevation(dirs, root_band)
    instance_radius = radius * np.linalg.norm(mesh.vertices, axis=1)
    gap = np.abs(terrain_radius - instance_radius)
    assert np.mean(gap) < radius * mesh.error + 1.5
    assert prototype_shape(SHAPES, 1, 0) is prototype_shape(SHAPES, 1, 0)
