"""Asteroid field collision (asteroids.collision, native/asteroid_collision.cpp):
the native solid against brute force over the drawn triangles, through
Panda's traverser, and the surface's material and velocity."""

import math
import os

import numpy as np
import pytest

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.motion import orientation, rotate
from world_builder.asteroids.prototype import (
    PrototypeJob,
    generate_prototype,
    mesh_indices,
)
from world_builder.asteroids.scatter import Scatter
from world_builder.asteroids.spec import (
    Box,
    Density,
    FieldSpec,
    MaterialSpec,
    ShapeSpec,
    SpinSpec,
    Tumble,
)


def _native_or_skip():
    from world_builder.terrain.native import NativeAsteroids, NativeCollisionError

    try:
        NativeAsteroids([1.0], [1.0]).close()
    except NativeCollisionError as error:
        if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
            pytest.fail(str(error))
        pytest.skip(str(error))
    return NativeAsteroids


SPEC = FieldSpec(
    shape=Box((2000.0, 2000.0, 2000.0)),
    density=Density(30.0),
    sizes=Distribution.uniform(60.0, 200.0),
    shapes=ShapeSpec(prototypes=3),
    spin=SpinSpec(
        Distribution.uniform(0.05, 0.3),
        size_exponent=0.0,
        tumble=Tumble(
            0.5, Distribution.uniform(0.2, 0.8), Distribution.uniform(0.5, 1.5)
        ),
    ),
    material=MaterialSpec(Distribution.uniform(0.0, 1.0)),
)
JOB = PrototypeJob(SPEC.shapes, SPEC.seed, lods=(4, 8, 16), normal_size=16)


@pytest.fixture(scope="module")
def world():
    """The native rocks of a small field, and its prototypes and rocks."""
    native_type = _native_or_skip()
    scatter = Scatter(SPEC)
    prototypes = [generate_prototype(JOB, k) for k in range(3)]
    native = native_type(
        [c.cell for c in scatter.classes], [c.hi for c in scatter.classes]
    )
    for p in prototypes:
        native.set_prototype(p.key, p.meshes[-1].quads, p.meshes[-1].vertices)
    lo, hi = SPEC.shape.bounds()
    rocks = scatter.rocks_in_box(lo, hi)
    from world_builder.asteroids.collision import _rows

    native.add(np.arange(len(rocks)), _rows(rocks))
    yield native, prototypes, rocks
    native.close()


def _triangles(rock, prototypes, t):
    """(n, 3, 3) the rock's finest-mesh triangles as drawn at time t."""
    mesh = prototypes[rock["proto"]].meshes[-1]
    q = orientation(rock["q0"], rock["axis"], rock["precession"], rock["spin"], t)
    points = (
        rotate(q, mesh.vertices.astype(np.float64) * rock["radius"]) + rock["centre"]
    )
    return points[mesh_indices(mesh.quads).astype(np.int64)].reshape(-1, 3, 3)


def _segment_hit(tris, a, b, every: bool = False):
    """Möller-Trumbore against every triangle: the first t, or None (or
    with *every*, all the crossings' t, sorted)."""
    d = b - a
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(d, e2)
    det = np.sum(e1 * p, axis=1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = a - tris[:, 0]
    u = np.sum(s * p, axis=1) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.sum(e2 * q, axis=1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
    if every:
        return np.sort(t[hit])
    return float(t[hit].min()) if np.any(hit) else None


def _nearest_on_mesh(tris, c):
    """The distance from c to the triangles (sampled finely)."""
    w = np.linspace(0.0, 1.0, 9)
    u, v = np.meshgrid(w, w)
    keep = u + v <= 1.0
    u, v = u[keep], v[keep]
    pts = (
        tris[:, None, 0] * (1 - u - v)[None, :, None]
        + tris[:, None, 1] * u[None, :, None]
        + tris[:, None, 2] * v[None, :, None]
    ).reshape(-1, 3)
    return float(np.min(np.linalg.norm(pts - c, axis=1)))


@pytest.mark.parametrize("t", [0.0, 7.5, 123.0])
def test_segments_hit_the_drawn_triangles(world, t):
    native, prototypes, rocks = world
    native.set_time(t)
    rng = np.random.default_rng(int(t * 10) + 1)
    hits = 0
    for rock in rocks[:12]:
        tris = _triangles(rock, prototypes, t)
        for _ in range(4):
            # from outside the rock toward a point near its centre
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            a = rock["centre"] + direction * rock["radius"] * 1.5
            b = rock["centre"] + rng.normal(size=3) * 0.2 * rock["radius"]
            expected = _segment_hit(tris, a, b)
            got = native.segment(a, b)
            if expected is None:
                continue
            assert got is not None
            t_hit, point, normal, slot = got
            # another rock may stand in the way: then it is nearer
            if rocks[slot]["id"] != rock["id"]:
                assert t_hit <= expected + 1e-6
                continue
            hits += 1
            # The native march samples every half mesh spacing: a sliver of
            # rock thinner than that (a bulge just grazed) may be stepped
            # over, the next entry found instead
            crossings = _segment_hit(tris, a, b, every=True)
            step = 0.5 * (np.pi / 2.0 / JOB.lods[-1]) * rock["radius"]
            step_t = step / np.linalg.norm(b - a)
            k = 0
            while (
                k + 2 < len(crossings)
                and crossings[k + 1] - crossings[k] < step_t
                and abs(t_hit - crossings[k]) > 1e-4
            ):
                k += 2
            expected = crossings[k]
            # (a point on a quad's diagonal may take the neighbouring triangle)
            assert t_hit == pytest.approx(expected, abs=1e-4)
            np.testing.assert_allclose(
                point, a + expected * (b - a), atol=1e-3 * rock["radius"]
            )
            assert np.dot(normal, b - a) < 0.0  # it faces the incoming segment
    assert hits > 20


def test_segments_miss_where_nothing_is(world):
    native, _, rocks = world
    native.set_time(0.0)
    far = np.array([1e5, 0.0, 0.0])
    assert native.segment(far, far + [0.0, 50.0, 0.0]) is None


@pytest.mark.parametrize("t", [0.0, 42.0])
def test_spheres_touch_where_the_triangles_are_near(world, t):
    native, prototypes, rocks = world
    native.set_time(t)
    rng = np.random.default_rng(int(t) + 7)
    checked = 0
    for rock in rocks[:10]:
        tris = _triangles(rock, prototypes, t)
        for _ in range(8):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            # about the surface that way (a crooked rock is far thinner
            # than its bounding radius in places)
            far = rock["centre"] + direction * rock["radius"] * 1.5
            entry = _segment_hit(tris, far, rock["centre"])
            surface = (
                1.5 * rock["radius"] * (1.0 - (entry if entry is not None else 0.0))
            )
            c = rock["centre"] + direction * surface * rng.uniform(0.8, 1.2)
            r = rock["radius"] * 0.15
            gap = _nearest_on_mesh(tris, c)
            inside = (
                _segment_hit(tris, rock["centre"], c) is None
            )  # no crossing: within
            got = native.sphere(c, r, samples=7)
            if inside or gap < 0.7 * r:
                assert got is not None
                checked += 1
                point, normal, depth, _ = got
                assert depth > 0.0
                assert np.linalg.norm(normal) == pytest.approx(1.0)
            elif gap > 1.3 * r and native.sphere(c, r) is not None:
                # only another rock may touch it
                _, _, _, slot = native.sphere(c, r)
                assert rocks[slot]["id"] != rock["id"]
    assert checked > 5


def test_rocks_can_be_removed(world):
    native, _, rocks = world
    native.set_time(0.0)
    rock = rocks[0]
    a = rock["centre"] + np.array([0.0, 0.0, 3.0 * rock["radius"]])
    b = rock["centre"]
    assert native.segment(a, b) is not None
    count = len(native)
    native.remove(np.array([0]))
    assert len(native) == count - 1
    got = native.segment(a, b)
    assert got is None or got[3] != 0
    from world_builder.asteroids.collision import _rows

    native.add(np.array([0]), _rows(rocks[:1]))
    assert len(native) == count


# ── a field, through Panda's traverser ─────────────────────────────────────


@pytest.fixture(scope="module")
def field(showbase):
    _native_or_skip()
    from panda3d.core import BitMask32, Camera, NodePath, PerspectiveLens

    from world_builder.asteroids.field import AsteroidField
    from world_builder.asteroids.settings import AsteroidSettings
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=4)
    clock = {"t": 5.0}
    made = AsteroidField(
        showbase,
        SPEC,
        clock=lambda: clock["t"],
        settings=AsteroidSettings(max_lod=16, normal_size=64, texture_size=256),
        executor=executor,
        position=(1.0e6, -2.0e6, 3.0e5),
        orientation=(math.cos(0.3), 0.0, math.sin(0.3) * 0.6, math.sin(0.3) * 0.8),
    )
    camera = NodePath(Camera("test_cam", PerspectiveLens()))
    camera.reparent_to(showbase.render)
    made.enable_collision(BitMask32.bit(5), max_distance=1500.0)
    state = {"camera": camera, "clock": clock}

    def update(camera_field):
        world = made.position + made.rotation @ camera_field
        made.update(world, (0.0, 0.0, 0.0), cameras=[camera])
        state["camera_field"] = np.asarray(camera_field, dtype=np.float64)

    state["update"] = update
    for _ in range(10):
        update(np.zeros(3))
        if not made.loading:
            break
    assert not made.loading
    yield made, state
    made.destroy()
    executor.shutdown()
    camera.remove_node()


def _render_point(field, state, p_field):
    """A field point in render: the camera at render's origin."""
    return field.rotation @ (np.asarray(p_field) - state["camera_field"])


def _traverse(render, *colliders):
    from panda3d.core import CollisionHandlerQueue, CollisionTraverser

    traverser = CollisionTraverser()
    queue = CollisionHandlerQueue()
    for path in colliders:
        traverser.add_collider(path, queue)
    traverser.traverse(render)
    queue.sort_entries()
    return [queue.get_entry(i) for i in range(queue.get_num_entries())]


def _collider(render, solid, mask_bit=5):
    from panda3d.core import BitMask32, CollisionNode

    node = CollisionNode("probe")
    node.add_solid(solid)
    node.set_from_collide_mask(BitMask32.bit(mask_bit))
    node.set_into_collide_mask(BitMask32.all_off())
    return render.attach_new_node(node)


def test_the_traverser_sees_the_rocks(showbase, field):
    from panda3d.core import CollisionSegment, CollisionSphere

    made, state = field
    pop = made.population
    slots = np.flatnonzero(made.collision._registered)
    rock = pop.rocks[slots[0]]
    state["update"](rock["centre"] + np.array([0.0, -4.0 * rock["radius"], 0.0]))
    render = showbase.render
    centre = _render_point(made, state, rock["centre"])
    start = _render_point(
        made, state, rock["centre"] + [0.0, 0.0, 3.0 * rock["radius"]]
    )
    segment = _collider(render, CollisionSegment(*start, *centre))
    sphere = _collider(render, CollisionSphere(*centre, 0.5 * rock["radius"]))
    try:
        entries = _traverse(render, segment, sphere)
        assert len(entries) == 2  # one per collider
        assert all(e.get_into_node().get_name() == "asteroids" for e in entries)
        owner = entries[0].get_into_node_path().get_python_tag("owner")
        assert owner is made.collision.surface
        by_from = {e.get_from_node_path(): e for e in entries}
        hit = by_from[segment]
        point = np.array(hit.get_surface_point(render))
        # the native answer, in render
        native = made.collision.native.segment(
            rock["centre"] + [0.0, 0.0, 3.0 * rock["radius"]], rock["centre"]
        )
        np.testing.assert_allclose(
            point, _render_point(made, state, native[1]), atol=0.05
        )
        normal = np.array(hit.get_surface_normal(render))
        np.testing.assert_allclose(normal, made.rotation @ native[2], atol=1e-4)
        # material and the surface's speed
        # snow on the rock's ice patches: the hit's direction, body frame
        from world_builder.asteroids.motion import orientation, rotate
        from world_builder.asteroids.shape import ice_patches, ice_weight

        q = orientation(
            rock["q0"],
            rock["axis"],
            rock["precession"],
            rock["spin"],
            made.collision._time,
        )
        body = rotate(
            q * [1.0, -1.0, -1.0, -1.0], np.asarray(native[1]) - rock["centre"]
        )
        patches = ice_patches(
            made.spec.seed, int(rock["proto"]), made.spec.material.patch_scale
        )
        snow = ice_weight(patches.rank(body / np.linalg.norm(body)), rock["ice"])
        expected = "snow" if float(snow[0]) > 0.5 else "rock"
        assert owner.material_at(point, render) == expected
        assert owner.material_at(point + [1e5, 0, 0], render) is None
    finally:
        segment.remove_node()
        sphere.remove_node()


def test_the_surface_velocity_is_the_rocks_turn(field):
    made, state = field
    pop = made.population
    slots = np.flatnonzero(made.collision._registered)
    rock = pop.rocks[slots[1]]
    state["update"](rock["centre"] + np.array([0.0, -4.0 * rock["radius"], 0.0]))
    t = state["clock"]["t"]
    surface = made.collision.surface
    body = np.array([0.3, -0.5, 0.7]) * rock["radius"]

    def at(time):
        q = orientation(
            rock["q0"], rock["axis"], rock["precession"], rock["spin"], time
        )
        return rock["centre"] + rotate(q, body)

    p = at(t)
    dt = 1e-4
    finite = (at(t + dt) - at(t - dt)) / (2 * dt)  # field frame
    got = surface.velocity_at(_render_point(made, state, p))  # render frame
    np.testing.assert_allclose(
        got, made.rotation @ finite, atol=1e-3 * np.linalg.norm(finite)
    )
    assert np.linalg.norm(finite) > 0.0


def test_far_rocks_are_not_solid(field):
    made, state = field
    pop = made.population
    state["update"](np.zeros(3))
    registered = made.collision._registered
    d = np.linalg.norm(pop.rocks["centre"], axis=1) - pop.rocks["radius"]
    live = pop.live
    assert np.all(registered[live & (d > 1500.0 + 1.0)] == False)  # noqa: E712
    assert np.all(registered[live & (d < 1500.0 - 1.0)])
    assert len(made.collision.native) == int(np.count_nonzero(registered))
