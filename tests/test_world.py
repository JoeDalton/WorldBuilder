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
from panda3d.core import LPoint3f

from world_builder.atmosphere import AtmosphereParams
from world_builder.clouds.sphere import SphericalClouds
from world_builder.depth import (
    FAR_CAMERA_MASK,
    NEAR_CAMERA_MASK,
    SplitDepthCameras,
)
from world_builder.ocean import Ocean
from world_builder.settings import WorldSettings
from world_builder.sky import Sky
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.quadtree import UNKNOWN_BOUNDS
from world_builder.world import (
    Body,
    BodyLook,
    BodyOcean,
    CloudLook,
    OceanLook,
    SkyLook,
    SunLook,
    World,
)

R = 600e3
EARTHLIKE = PRESETS["earthlike"]
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
        (OceanLook, Ocean.__init__, OceanLook.WORLD_FIELDS),
        (SkyLook, Sky.__init__, ("stars",)),
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
        ([], "needs a body"),
        ([BodyLook("a", EARTHLIKE), BodyLook("a", MOON)], "must differ"),
        ([BodyLook("a", EARTHLIKE, clouds=CloudLook())], "need an atmosphere"),
        ([BodyLook("a", PRESETS["desert"], ocean=OceanLook())], "has_ocean"),
    ],
)
def test_a_world_refuses_what_it_cannot_build(bodies, message):
    with pytest.raises(ValueError, match=message):
        World(None, bodies)


# ── stand-ins ──────────────────────────────────────────────────────────────────


def _body(name="planet", spec=EARTHLIKE, position=(0.0, 0.0, 0.0), **look):
    terrain = MagicMock(position=np.array(position, float), radius=spec.radius)
    terrain.loading = False
    return Body(look=BodyLook(name, spec, position, **look), terrain=terrain)


def _world(*bodies, camera_render=(0.0, 0.0, 0.0), sky=True):
    """A World over stand-in parts, without building any."""
    world = object.__new__(World)
    camera = MagicMock()
    camera.get_pos.return_value = LPoint3f(*camera_render)
    world.app = SimpleNamespace(camera=camera, render=MagicMock(), camLens=None)
    world.bodies = list(bodies)
    world.atmospheres = MagicMock()
    world.atmospheres.light_at.return_value = (np.ones(3), np.full(3, 0.5))
    world.sky = MagicMock() if sky else None
    world.sun_light = world.ambient_light = None
    world._sun = np.array([0.0, 0.0, 1.0])
    world._sun_color = np.array([2.0, 2.0, 2.0])
    return world


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

    world._light(np.array([0.0, 0.0, R + 10.0]))

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


def _ocean(look=OceanLook()):
    """A BodyOcean over stand-ins: what its handover switches."""
    ocean = object.__new__(BodyOcean)
    ocean.look = look
    ocean.terrain = MagicMock(radius=R)
    ocean.ocean = MagicMock()
    ocean._anchor = None
    ocean.swell = True
    return ocean


def test_ocean_hands_the_sea_over_with_hysteresis():
    ocean = _ocean()
    low, high = OceanLook().handover
    ocean.handover(high + 1.0)
    assert not ocean.swell
    ocean.terrain.set_sea_surface.assert_called_with(True)
    ocean.ocean.base_node.hide.assert_called_once()
    ocean.handover((low + high) / 2)  # inside the band: unchanged
    assert not ocean.swell
    ocean.handover(low - 1.0)
    assert ocean.swell
    ocean.terrain.set_sea_surface.assert_called_with(False)
    ocean.ocean.refl_buffer.set_active.assert_called_with(True)


def test_ocean_waves_stay_anchored_then_re_anchor():
    ocean = _ocean()
    start = TangentFrame.at(10.0, 20.0).up
    assert ocean.anchor(start) == (0.0, 0.0)
    near = TangentFrame.at(10.0 + math.degrees(10e3 / R), 20.0).up  # 10 km north
    gx, gy = ocean.anchor(near)
    assert gx == pytest.approx(0.0, abs=1.0) and gy == pytest.approx(10e3, rel=1e-3)
    far = TangentFrame.at(10.0 + math.degrees(60e3 / R), 20.0).up  # past 50 km
    assert ocean.anchor(far) == (0.0, 0.0)


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
            ),
            BodyLook("moon", MOON, position=tuple(frame.up * 1e7)),
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

        assert planet.ocean.swell
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
