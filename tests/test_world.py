"""
The world facade (world_builder.world): its looks against the parts'
arguments, its checks, its per-frame wiring and maths on stand-ins, and one
whole world built on an offscreen window.
"""

import inspect
import math
from dataclasses import fields, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from panda3d.core import LPoint3f, NodePath

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.field import AsteroidField
from world_builder.asteroids.spec import Annulus, Density, FieldSpec, ShapeSpec
from world_builder.atmosphere import AtmosphereParams
from world_builder.clouds.sphere import SphericalClouds
from world_builder.depth import (
    FAR_CAMERA_MASK,
    NEAR_CAMERA_MASK,
    SplitDepthCameras,
)
from world_builder.lights import TrackedLights
from world_builder.ocean import Sea
from world_builder.settings import WorldSettings
from world_builder.shadows import ECLIPSE_SKY_FLOOR, SUN_ANGULAR_RADIUS, Shadows
from world_builder.sky import Sky
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PLANET_HEIGHT_BOUNDS, PRESETS
from world_builder.terrain.quadtree import UNKNOWN_BOUNDS
from world_builder.world import (
    Body,
    BodyLook,
    CloudLook,
    FieldLook,
    OceanLook,
    SkyLook,
    SunLook,
    World,
)

R = 600e3
EARTHLIKE = PRESETS["earthlike"]
#: A small icy ring around the world's origin.
RING = FieldSpec(
    shape=Annulus(2.0e6, 3.0e6, 2000.0, edge=300.0),
    density=Density(5.0),
    sizes=Distribution.power_law(3.0, 50.0, 2000.0),
    shapes=ShapeSpec(prototypes=2),
)
MOON = replace(PRESETS["moon"], seed=7)


# ── looks ──────────────────────────────────────────────────────────────────────


def _defaults(function) -> dict:
    return {
        name: p.default
        for name, p in inspect.signature(function).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def _same(a, b) -> bool:
    try:
        return tuple(a) == pytest.approx(tuple(b))
    except TypeError:
        return a == b


@pytest.mark.parametrize(
    "look,part,own",
    [
        (CloudLook, SphericalClouds.__init__, ()),
        (OceanLook, Sea.__init__, OceanLook.WORLD_FIELDS),
        (SkyLook, Sky.__init__, ("stars",)),
        (FieldLook, AsteroidField.__init__, ("name", "spec", "ambient")),
    ],
)
def test_looks_are_their_parts_arguments(look, part, own):
    """Each look field is the part's argument of that name, with its default:
    the looks invent nothing, and follow the parts."""
    defaults = _defaults(part)
    for f in fields(look):
        if f.name in own:
            continue
        assert f.name in defaults, f"{look.__name__}.{f.name} is no argument"
        assert _same(f.default, defaults[f.name]), f"{look.__name__}.{f.name}"


# ── checks ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bodies,message",
    [
        (
            [BodyLook("a", EARTHLIKE, mass=1.0), BodyLook("a", MOON, mass=1.0)],
            "must differ",
        ),
        (
            [BodyLook("a", EARTHLIKE, clouds=CloudLook(), mass=1.0)],
            "need an atmosphere",
        ),
        ([BodyLook("a", PRESETS["desert"], ocean=OceanLook(), mass=1.0)], "has_ocean"),
    ],
)
def test_a_world_refuses_what_it_cannot_build(bodies, message):
    with pytest.raises(ValueError, match=message):
        World(None, bodies)


def test_a_world_refuses_two_fields_of_one_name():
    with pytest.raises(ValueError, match="must differ"):
        World(None, [], fields=[FieldLook("a", RING), FieldLook("a", RING)])


def test_fields_follow_the_camera_the_sun_and_the_loading():
    """Each field: updated around the camera, lit by the bare sun and its
    own ambient (or the space's), part of the loading, found by name."""
    world = _world(camera_render=(1.0, 2.0, 3.0))
    ring = _field(world)
    belt = _field(world, "belt", ambient=(0.5, 0.5, 0.5), loading=True)
    origin = np.array([2.5e6, 0.0, 0.0])
    world.update(origin)
    (camera, render), _ = ring.update.call_args
    np.testing.assert_allclose(camera, origin + [1.0, 2.0, 3.0])
    np.testing.assert_allclose(render, [1.0, 2.0, 3.0])
    assert world.loading  # the belt
    belt.loading = False
    assert not world.loading
    world.set_sun((0.0, 1.0, 0.0))
    _, color, ambient = ring.set_sun.call_args[0]
    np.testing.assert_allclose(color, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(ambient, [0.1, 0.2, 0.3])  # the space's
    np.testing.assert_allclose(belt.set_sun.call_args[0][2], [0.5, 0.5, 0.5])
    assert world.field("belt") is belt
    with pytest.raises(KeyError):
        world.field("rings")


# ── stand-ins ──────────────────────────────────────────────────────────────────


def _body(name="planet", spec=EARTHLIKE, position=(0.0, 0.0, 0.0), mass=1.0, **look):
    terrain = MagicMock(position=np.array(position, float), radius=spec.radius)
    terrain.loading = False
    terrain.planet.height_bounds = PLANET_HEIGHT_BOUNDS
    return Body(look=BodyLook(name, spec, position, mass=mass, **look), terrain=terrain)


def _world(*bodies, camera_render=(0.0, 0.0, 0.0), sky=True):
    """A World over stand-in parts, without building any."""
    world = object.__new__(World)
    camera = MagicMock()
    camera.get_pos.return_value = LPoint3f(*camera_render)
    world.app = SimpleNamespace(camera=camera, render=MagicMock(), camLens=None)
    world.bodies = list(bodies)
    world.shadows = Shadows()
    for body in bodies:
        world.shadows.add(body.radius, body.position)
    world.shadows.set_sun([0.0, 0.0, 1.0])  # as world._sun below
    world.tracked = TrackedLights(NodePath("render"))
    world.light_scale = 1.0
    world.atmospheres = MagicMock()
    world.atmospheres.light_at.return_value = (np.ones(3), np.full(3, 0.5))
    world.sky = MagicMock() if sky else None
    world.sun_light = world.ambient_light = None
    world._sun = np.array([0.0, 0.0, 1.0])
    world._sun_color = np.array([2.0, 2.0, 2.0])
    world._space_ambient = np.array([0.1, 0.2, 0.3])
    world.fields = []
    world._field_looks = []
    return world


def _field(world, name="ring", ambient=None, loading=False):
    """A stand-in field added to a stand-in world."""
    field = MagicMock(loading=loading)
    world.fields.append(field)
    world._field_looks.append(FieldLook(name, RING, ambient=ambient))
    return field


# ── frames, ranges, light ──────────────────────────────────────────────────────


def test_update_works_around_the_camera():
    """The camera is render's origin plus its own render position: every part
    gets that, the terrain both."""
    body = _body()
    world = _world(body, camera_render=(1.0, 2.0, 3.0))
    origin = np.array([10.0, 20.0, R + 1000.0])

    world.update(origin, far=5e6)

    camera = origin + [1.0, 2.0, 3.0]
    (got, render), _ = body.terrain.update.call_args
    np.testing.assert_allclose(got, camera)
    np.testing.assert_allclose(render, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(world.atmospheres.update.call_args[0][0], camera)
    world.sky.update.assert_called_once()
    assert world.sky.update.call_args[0][1] == 5e6
    np.testing.assert_allclose(world.camera_world, camera)


def test_update_without_far_takes_the_depth_ranges():
    world = _world(_body())
    origin = np.array([0.0, 0.0, R + 1000.0])
    world.update(origin)
    assert world.sky.update.call_args[0][1] == world.depth_ranges(origin)[1]


def test_open_space_is_a_world_without_bodies():
    """No planet: the sky, the sun and the lights alone, a fixed depth split,
    the bare sun and the space ambient light, nothing loading."""
    from world_builder.world import SPACE_FAR_M, SPACE_SPLIT_M

    world = _world()
    origin = np.array([1e6, -2e6, 3e5])
    assert world.depth_ranges(origin) == (SPACE_SPLIT_M, SPACE_FAR_M)
    direct, ambient = world.light_at(origin)
    np.testing.assert_allclose(direct, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(ambient, [0.1, 0.2, 0.3])
    assert not world.loading
    world.update(origin)
    world.sky.update.assert_called_once()
    assert world.sky.update.call_args[0][1] == SPACE_FAR_M


def test_depth_ranges_cover_every_body():
    planet = _body()
    moon = _body("moon", MOON, position=(0.0, 1e7, 0.0))
    world = _world(planet, moon)
    origin = np.array([0.0, 0.0, R + 800.0])
    expected = SplitDepthCameras.system_ranges(
        [
            (R, R + 800.0, UNKNOWN_BOUNDS[1], 0.0),
            (
                MOON.radius,
                float(np.linalg.norm(origin - [0.0, 1e7, 0.0])),
                UNKNOWN_BOUNDS[1],
                UNKNOWN_BOUNDS[0],
            ),
        ]
    )
    assert world.depth_ranges(origin) == pytest.approx(expected)


def test_nearest_body_is_by_sea_level():
    planet = _body()
    moon = _body("moon", MOON, position=(0.0, 1e7, 0.0))
    world = _world(planet, moon)
    assert world.nearest([0.0, 0.0, R + 1e5]) is planet
    assert world.nearest([0.0, 1e7 - MOON.radius - 10.0, 0.0]) is moon


def test_gravity_field_pulls_toward_the_bodies():
    planet = _body(mass=6e24)
    moon = _body("moon", MOON, position=(0.0, 1e7, 0.0), mass=7e22)
    field = _world(planet, moon).gravity_field(g_constant=6.674e-11)
    g = field.acceleration([0.0, 0.0, R + 1e5])
    assert g[2] == pytest.approx(-6.674e-11 * 6e24 / (R + 1e5) ** 2, rel=1e-2)
    assert field.acceleration([0.0, 1e7 - MOON.radius, 0.0])[1] > 0


def test_light_through_the_air():
    planet = _body(atmosphere=AtmosphereParams())
    planet.atmosphere = 3
    world = _world(planet)
    direct, ambient = world.light_at([0.0, 0.0, R + 10.0])
    assert world.atmospheres.light_at.call_args[0][0] == 3
    np.testing.assert_allclose(direct, np.ones(3))


def test_panda_lights_are_the_light_at_the_camera_scaled():
    """Panda3D's lights take the world's light at the camera, times the
    world's light scale (for models with no tone map)."""
    planet = _body(atmosphere=AtmosphereParams())
    planet.atmosphere = 0
    world = _world(planet)
    world.light_scale = 0.5
    world.sun_light, world.ambient_light = MagicMock(), MagicMock()

    camera = np.array([0.0, 0.0, R + 10.0])
    world._light(camera, camera)

    world.sun_light.node().set_color.assert_called_once_with((0.5, 0.5, 0.5, 1.0))
    world.ambient_light.node().set_color.assert_called_once_with(
        (0.25, 0.25, 0.25, 1.0)
    )


def test_airless_light_is_the_sun_or_the_shadow():
    moon = _body("moon", MOON, ambient=(0.01, 0.01, 0.01))
    world = _world(moon)
    lit, ambient = world.light_at([0.0, 0.0, MOON.radius + 10.0])  # sun overhead
    np.testing.assert_allclose(lit, world._sun_color)
    np.testing.assert_allclose(ambient, 0.01)
    dark, _ = world.light_at([0.0, 0.0, -MOON.radius - 10.0])  # night side
    np.testing.assert_allclose(dark, 0.0)


def test_a_moon_eclipses_the_light_on_the_planet():
    """Under a moon as wide as the sun, straight toward it: no sunlight,
    and the sky's light down to its floor; well off its shadow, both
    whole."""
    planet = _body(atmosphere=AtmosphereParams())
    planet.atmosphere = 0
    distance = MOON.radius / math.sin(SUN_ANGULAR_RADIUS) + R
    moon = _body("moon", MOON, position=(0.0, 0.0, distance))
    world = _world(planet, moon)
    direct, ambient = world.light_at([0.0, 0.0, R + 10.0])
    np.testing.assert_allclose(direct, 0.0, atol=1e-6)
    np.testing.assert_allclose(ambient, 0.5 * ECLIPSE_SKY_FLOOR, rtol=1e-4)
    side = [R * math.sin(1.2), 0.0, R * math.cos(1.2)]  # past the penumbra
    direct, ambient = world.light_at(side)
    np.testing.assert_allclose(direct, 1.0)
    np.testing.assert_allclose(ambient, 0.5)


def test_a_planet_eclipses_an_airless_moon():
    """A lunar eclipse: the moon's day side, behind the planet from the
    sun, is dark (its own shadow alone left it lit)."""
    planet = _body()
    moon = _body("moon", MOON, position=(0.0, 0.0, -1e7), ambient=(0.01,) * 3)
    world = _world(planet, moon)
    # The sun along +z: the planet stands between it and the moon.
    direct, _ = world.light_at([0.0, 0.0, -1e7 + MOON.radius + 10.0])
    np.testing.assert_allclose(direct, 0.0)


def test_lights_at_many_points_are_each_points_light():
    """The vectorised light is the single point's, point by point, each by
    its nearest body; sun_visibility is its eclipse share."""
    planet = _body(ambient=(0.2, 0.2, 0.2))
    distance = MOON.radius / math.sin(SUN_ANGULAR_RADIUS) + R
    moon = _body("moon", MOON, position=(0.0, 0.0, distance), ambient=(0.01,) * 3)
    world = _world(planet, moon)
    rng = np.random.default_rng(2)
    points = rng.normal(size=(50, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = np.concatenate([points * (R + 100.0), points[:10] * 3e5 + moon.position])
    direct, ambient = world.lights_at(points)
    for point, d, a in zip(points, direct, ambient):
        one_d, one_a = world.light_at(point)
        np.testing.assert_allclose(d, one_d)
        np.testing.assert_allclose(a, one_a)
    visible = world.sun_visibility(points)
    assert visible[0] == pytest.approx(
        world.shadows.eclipse(points[0], world._sun, 0)[()]
    )
    assert visible.shape == (60,) and visible.min() >= 0.0


def test_one_light_call_lights_the_camera_the_nodes_and_the_sea():
    """The frame's lights come from one lights_at call: the camera's, each
    tracked node's and the near sea's, each the light where it is."""
    from world_builder.atmosphere import Atmospheres

    planet = _body(atmosphere=AtmosphereParams(), ocean=OceanLook())
    planet.atmosphere = 0
    planet.ocean = MagicMock(near=1.0)
    world = _world(planet)
    world.atmospheres = Atmospheres()
    world.atmospheres.add(R, AtmosphereParams())
    world._sun = np.array([0.6, 0.0, 0.8])
    world.sun_light, world.ambient_light = MagicMock(), MagicMock()
    camera = np.array([0.0, 0.0, R + 300.0])
    ball = world.tracked.render.attach_new_node("ball")
    ball.set_pos(0.0, 2e5, 0.0)  # 200 km away, still near the planet
    world.track_light(ball)
    calls = []
    lights_at = world.lights_at
    world.lights_at = lambda points: calls.append(len(points)) or lights_at(points)

    world._light(camera, camera)

    assert calls == [3]
    direct, ambient = world.light_at(camera)
    assert world.sun_light.node().set_color.call_args[0][0][:3] == pytest.approx(
        tuple(direct)
    )
    ball_sun = world.tracked.render.find("**/ball_sun").node().get_color()
    expected, _ = world.light_at(camera + [0.0, 2e5, 0.0])
    np.testing.assert_allclose(tuple(ball_sun)[:3], expected, rtol=1e-6)
    sea = np.array([0.0, 0.0, R])
    direct, ambient = world.light_at(sea)
    light = direct * 0.8 + ambient
    np.testing.assert_allclose(
        planet.ocean.set_light.call_args[0][0],
        np.clip(light / planet.ocean_reference, 0.0, 1.5),
    )


def test_a_tracked_node_is_lit_where_it_is():
    """A ball in the moon's umbra goes dark while the camera, outside the
    shadow, keeps the sun; untracked, it takes the camera's light again."""
    planet = _body()
    distance = MOON.radius / math.sin(SUN_ANGULAR_RADIUS) + R
    moon = _body("moon", MOON, position=(0.0, 0.0, distance))
    world = _world(planet, moon)
    world.light_scale = 0.5
    world.sun_light, world.ambient_light = MagicMock(), MagicMock()
    origin = np.array([R * math.sin(1.2), 0.0, R * math.cos(1.2) + 10.0])
    ball = world.tracked.render.attach_new_node("ball")
    ball.set_pos(*(np.array([0.0, 0.0, R + 10.0]) - origin))
    world.track_light(ball)

    world.update(origin)

    camera_sun = world.sun_light.node().set_color.call_args[0][0]
    assert camera_sun[:3] == pytest.approx((1.0, 1.0, 1.0))  # 2.0 * 0.5
    ball_sun = world.tracked.render.find("**/ball_sun").node()
    assert tuple(ball_sun.get_color())[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    ball_sky = world.tracked.render.find("**/ball_ambient").node().get_color()
    expected = 0.5 * np.asarray(planet.look.ambient) * ECLIPSE_SKY_FLOOR
    np.testing.assert_allclose(tuple(ball_sky)[:3], expected, rtol=1e-4)
    world.untrack_light(ball)
    assert world.tracked.render.find("**/ball_sun").is_empty()


# ── loading, collision ─────────────────────────────────────────────────────────


def test_loading_waits_for_every_terrain_and_cloud_layer():
    planet, moon = _body(), _body("moon", MOON)
    planet.clouds = SimpleNamespace(stats={"pending": 0})
    world = _world(planet, moon)
    assert not world.loading
    moon.terrain.loading = True
    assert world.loading
    moon.terrain.loading = False
    planet.clouds.stats["pending"] = 2
    assert world.loading


def test_collision_for_the_chosen_bodies():
    planet, moon = _body(), _body("moon", MOON)
    world = _world(planet, moon)
    world.enable_collision("mask", bodies=["planet"], max_distance=5.0)
    planet.terrain.enable_collision.assert_called_once_with("mask", max_distance=5.0)
    moon.terrain.enable_collision.assert_not_called()
    world.enable_collision("mask")
    moon.terrain.enable_collision.assert_called_once()


def test_body_by_name():
    planet = _body()
    world = _world(planet)
    assert world.body("planet") is planet
    with pytest.raises(KeyError):
        world.body("nope")


# ── ocean ──────────────────────────────────────────────────────────────────────


def _sea(look=OceanLook()):
    """A Sea over stand-ins: what its altitude band and its chart do."""
    sea = object.__new__(Sea)
    sea.near_band = look.near_band
    sea.anchor_range = look.anchor_range
    sea.terrain = MagicMock(radius=R, position=np.zeros(3))
    sea.terrain.to_render.side_effect = lambda p: np.asarray(p)
    sea.terrain.spec.has_ocean = True
    sea.app = MagicMock()
    sea.frame = MagicMock()
    sea.refl_buffer = MagicMock()
    sea._showing = None
    sea._frame_inputs = MagicMock()
    sea._phase_view = np.zeros(64, np.float32)
    sea._anchor = None
    return sea


def test_the_near_look_fades_through_its_band():
    sea = _sea()
    sea._place = MagicMock()
    sea._mirror = MagicMock()
    sea._drift_swell = MagicMock()
    sea._clock = lambda: 1.0
    sea._directions = np.zeros((64, 2))
    sea._chart_xy = (0.0, 0.0)
    low, high = OceanLook().near_band
    up = TangentFrame.at(10.0, 20.0).up
    sea.update(up * (R + low - 1.0))
    assert sea.near == 1.0
    sea.terrain.set_sea_near.assert_called_with(True)
    sea.update(up * (R + 0.5 * (low + high)))
    assert sea.near == pytest.approx(0.5)
    sea.update(up * (R + high + 1.0))
    assert sea.near == 0.0
    sea.refl_buffer.set_active.assert_called_with(False)
    sea.terrain.set_sea_near.assert_called_with(False)
    sea.terrain.spec.has_ocean = False  # a dry planet: never the near look
    sea.update(up * (R + 10.0))
    assert sea.near == 0.0


def test_the_waves_stay_anchored_then_re_anchor():
    sea = _sea()
    start = TangentFrame.at(10.0, 20.0).up
    sea._place(start)
    assert sea._chart_xy == (0.0, 0.0)
    near = TangentFrame.at(10.0 + math.degrees(10e3 / R), 20.0).up  # 10 km north
    sea._place(near)
    gx, gy = sea._chart_xy
    assert gx == pytest.approx(0.0, abs=1.0) and gy == pytest.approx(10e3, rel=1e-3)
    far = TangentFrame.at(10.0 + math.degrees(60e3 / R), 20.0).up  # past 50 km
    sea._place(far)
    assert sea._chart_xy == (0.0, 0.0)


# ── a whole world ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def window(showbase):
    """The session's ShowBase with an offscreen window; skipped where none
    can be opened."""
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


def test_a_whole_world(window, monkeypatch):
    """A planet with air, clouds and ocean, and an airless moon, under a sky:
    one pool, one air, the parts wired, the sea's frame on the nadir's tangent
    plane, Panda3D's lights on, and all of it gone after destroy."""
    window.camLens.set_near_far(1.0, 600e3)
    frame = TangentFrame.at(30.0, 40.0)
    world = World(
        window,
        bodies=[
            BodyLook(
                "planet",
                EARTHLIKE,
                atmosphere=AtmosphereParams(),
                clouds=CloudLook(altitude=2200.0),
                ocean=OceanLook(),
                mass=5.97e24,
            ),
            BodyLook("moon", MOON, position=tuple(frame.up * 1e7), mass=7.35e22),
        ],
        sun=SunLook(direction=tuple(frame.up + frame.north)),
        sky=SkyLook(),
        settings=WorldSettings.preset("low"),
        backend="inline",
        clock=lambda: 2.0,
    )
    shutdowns = []
    monkeypatch.setattr(world.executor, "shutdown", lambda: shutdowns.append(1))
    depth = SplitDepthCameras(window)
    try:
        planet, moon = world.body("planet"), world.body("moon")
        assert planet.terrain.executor is world.executor
        assert moon.terrain.executor is world.executor
        assert planet.clouds.executor is world.executor
        assert planet.atmosphere == 0 and moon.atmosphere is None
        # Drawn by the far camera only
        assert world.sky.node.is_hidden(NEAR_CAMERA_MASK)
        assert not world.sky.node.is_hidden(FAR_CAMERA_MASK)

        origin = frame.up * (R + 300.0)
        for _ in range(2):
            depth.update(*world.depth_ranges(origin))
            world.update(origin, far=depth.far)
            window.graphicsEngine.render_frame()

        assert planet.ocean.near == 1.0
        # The sea's plane is the tangent plane at the camera's nadir (the
        # camera is wherever earlier tests left it, near render's origin)
        nadir = world.camera_world / np.linalg.norm(world.camera_world)
        up = planet.ocean.frame.get_quat(window.render).get_up()
        np.testing.assert_allclose(np.array(up), nadir, atol=1e-5)
        assert window.render.has_light(world.sun_light)
        assert world.sun_light.node().get_color()[0] > 0.0
    finally:
        depth.destroy()
        world.destroy()
    assert shutdowns == [1]
    assert world.bodies == [] and world.sky is None
    assert not window.render.find("**/planet_sea_frame")


def test_open_space_builds_and_draws(window):
    """A world of the sky, the sun and the lights alone (no body), as a space
    level has: built, drawn, its lights the bare sun, gone after destroy."""
    world = World(
        window,
        bodies=[],
        sun=SunLook(direction=(0.3, -0.8, 0.5)),
        sky=SkyLook(),
        settings=WorldSettings.preset("low"),
        backend="inline",
        light_scale=0.5,
        space_ambient=(0.1, 0.1, 0.2),
    )
    try:
        assert world.bodies == [] and not world.loading
        for _ in range(2):
            world.update(np.array([5e6, 0.0, 0.0]), far=1e6)
            window.graphicsEngine.render_frame()
        sun = np.array(world.sun_light.node().get_color())[:3]
        np.testing.assert_allclose(sun, 0.5 * np.array(SunLook().color), rtol=1e-6)
        ambient = np.array(world.ambient_light.node().get_color())[:3]
        np.testing.assert_allclose(ambient, [0.05, 0.05, 0.1], rtol=1e-6)
    finally:
        world.destroy()
    assert world.sky is None or world.sky.node.is_empty()


def test_a_ring_in_open_space(window):
    """A world of a ring alone: built on the world's pool and settings,
    loaded, drawn (rocks and haze), set, and gone after destroy."""
    world = World(
        window,
        bodies=[],
        sun=SunLook(direction=(0.3, -0.8, 0.5)),
        settings=WorldSettings.preset("low"),
        backend="inline",
        clock=lambda: 4.0,
        fields=[FieldLook("ring", RING, far=200_000.0)],
    )
    try:
        ring = world.field("ring")
        assert ring.executor is world.executor
        assert ring.settings == world.settings.asteroids
        assert ring.far == 200_000.0
        origin = np.array([2.5e6, 0.0, 5000.0])
        for _ in range(300):
            world.update(origin, far=1e6)
            window.graphicsEngine.render_frame()
            if not world.loading:
                break
        assert not world.loading
        assert ring.population.count > 0
        high = WorldSettings.preset("high")
        world.set_settings(high)
        assert ring.settings == high.asteroids
    finally:
        world.destroy()
    assert world.fields == [] and ring.root.is_empty()
