"""
World files (world_builder.world_file): the shipped worlds are the code's own
definitions written out, a file's rules are checked, and one world is loaded
on an offscreen window.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

import world_builder
from world_builder.config import ConfigError
from world_builder.shadows import SUN_ANGULAR_RADIUS
from world_builder.world import SunLook
from world_builder.world_file import (
    load_field,
    load_planet,
    load_world,
    parse_world,
    world_to_yaml,
)

WORLDS = Path(world_builder.__file__).parent / "worlds"
THREE = WORLDS / "three_planets.yaml"
DEEP = WORLDS / "deep_space.yaml"


@pytest.fixture(scope="module")
def three():
    return parse_world(THREE)


def body(world, name):
    return next(b for b in world.bodies if b.name == name)


# ── the shipped worlds are the code's definitions ───────────────────────────


def test_the_home_planet_is_the_600_km_demo_planet_with_its_stamps(three):
    planet = body(three, "planet")
    spec = planet.spec
    assert (spec.name, spec.seed, spec.radius) == ("earthlike", 19, 600e3)
    assert (spec.reference_lat, spec.reference_lon) == (20.91, 30.4)
    assert [s.blend for s in spec.stamps] == ["add", "max", "flatten", "replace"]
    assert len(spec.paints) == 3
    # The images sit beside the world files, and exist
    for image in [s.heightmap for s in spec.stamps] + [p.mask for p in spec.paints]:
        assert Path(image).parent == WORLDS / "stamps" and Path(image).exists()
    assert planet.mass == 5.97e24
    assert planet.atmosphere is not None and planet.ocean is not None
    assert planet.clouds.altitude == 2200.0 and planet.clouds.cirrus


def test_the_moon_is_airless_21000_km_out(three):
    moon = body(three, "moon")
    assert (moon.spec.seed, moon.spec.radius) == (7, 200e3)
    assert not moon.spec.has_ocean and not moon.spec.has_vegetation
    assert moon.atmosphere is None and moon.clouds is None and moon.ocean is None
    home = body(three, "planet")
    assert math.dist(moon.position, home.position) == pytest.approx(21.0e6, rel=1e-6)


def test_the_ice_planet_is_3000_km_across_with_air_and_no_sea(three):
    ice = body(three, "ice")
    assert ice.spec.radius * 2 == 3.0e6
    assert ice.spec.seed == 3 and ice.spec.climate.temperature_mean < 0.0
    assert ice.spec == load_planet("ice")  # linked, not copied
    assert not ice.spec.has_ocean and ice.ocean is None
    assert ice.atmosphere is not None and ice.clouds is not None
    assert ice.mass == pytest.approx(5.97e24 * 2.5**3)  # the home planet's density


def test_the_ring_is_the_demo_ring_around_the_ice_planet(three):
    (ring,) = three.fields
    ice = body(three, "ice")
    assert ring.name == "ring" and ring.position == ice.position
    shape = ring.spec.shape
    assert (shape.inner, shape.outer) == (5.0e6, 1.0e7)  # 5000 to 10,000 km
    assert ring.spec == load_field("ring")
    assert (shape.thickness, shape.radial_edge) == (4000.0, 5e5)
    assert len(shape.radial_profile) == 129  # ringlets, a gap and a dense band
    # The axis is the ring's normal: the quaternion turns +z onto it
    w, x, y, z = ring.orientation
    assert w * w + x * x + y * y + z * z == pytest.approx(1.0)
    normal = np.array(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
    )
    expected = np.array([0.25, 0.1, 1.0])
    assert normal == pytest.approx(expected / np.linalg.norm(expected))


def test_the_ring_clears_the_other_two_bodies(three):
    (ring,) = three.fields
    outer = ring.spec.shape.outer
    for name in ("planet", "moon"):
        b = body(three, name)
        assert math.dist(b.position, ring.position) > 2 * outer


def test_deep_space_is_the_cube_of_asteroids():
    world = parse_world(DEEP)
    assert world.bodies == ()
    (cube,) = world.fields
    assert cube.spec == load_field("asteroid_cube")
    assert world.sun == SunLook(
        direction=(-0.45, -0.85, 0.3),
        color=(2.6, 2.5, 2.35),
        angular_radius=SUN_ANGULAR_RADIUS,
    )
    assert world.space_ambient == (0.03, 0.03, 0.04)


def test_a_file_has_no_defaults_for_its_world():
    """No sun, no sky, no ambient in a file: none in the world."""
    world = parse_world(moon_world(mass=1.0))
    assert world.sun is None and world.sky is None and world.space_ambient is None
    assert parse_world({"sun": None, "sky": None}).sun is None
    text = yaml.safe_load(world_to_yaml(world))
    assert not set(text) & {"sun", "sky", "space_ambient"}


def test_a_world_file_has_no_run_time_values():
    for path in (THREE, DEEP):
        keys = set(yaml.safe_load(path.read_text()))
        assert not keys & {"quality", "settings", "camera", "backend", "clock"}


def test_a_world_written_in_full_reads_back():
    world = parse_world(DEEP)
    text = world_to_yaml(world)
    assert parse_world(yaml.safe_load(text)).fields == world.fields


def test_load_planet_and_load_field_read_files_by_name_or_path():
    assert load_planet("earth") is load_planet(WORLDS / "earth.yaml")  # cached
    assert load_planet("earth").radius == 6.371e6
    assert load_planet("desert").has_ocean is False
    assert load_planet("moon").radius == 200e3
    assert load_planet("ice").radius == 1.5e6
    assert load_planet("tiny").radius == 20e3
    assert load_field("asteroid_cube").keep_outs  # the corridor
    assert load_field(THREE).shape.inner == 5.0e6  # a world's first field
    assert load_planet(THREE).name == "earthlike"  # a world's first body


# ── the rules of a file ─────────────────────────────────────────────────────


def moon_body(**body):
    """The shipped moon's file, whole (a file has no defaults), as a body."""
    data = yaml.safe_load((WORLDS / "moon.yaml").read_text())
    data.pop("mass")
    return {"name": "m", "position": [0.0, 0.0, 0.0], **data, **body}


def moon_world(**body):
    return {"bodies": [moon_body(**body)]}


def test_a_body_has_a_mass_or_a_density():
    radius = 2e5
    by_density = parse_world(moon_world(density=3000.0)).bodies[0]
    assert by_density.mass == pytest.approx(3000.0 * 4 / 3 * math.pi * radius**3)
    assert parse_world(moon_world(mass=7.0)).bodies[0].mass == 7.0
    for given in ({}, {"mass": 1.0, "density": 2.0}):
        with pytest.raises(ConfigError, match="exactly one of mass"):
            parse_world(moon_world(**given))


@pytest.mark.parametrize(
    "world,message",
    [
        ({"quality": "high"}, "unknown keys.*quality"),
        ({"bodies": [{"name": "a", "mass": 1.0}]}, "planet: is required"),
        (moon_world(mass=1, x=1), "unknown keys.*x"),
        (moon_world(mass=1, exposure=None) | {"sun": {}}, "sun.direction: is required"),
        (
            moon_world(mass=1) | {"sky": {"stars": None}},
            "sky.star_intensity: is required",
        ),
        (
            {"bodies": [moon_body(mass=1, planet={"name": "a"})]},
            "planet.seed: is required",
        ),
        ({"fields": [{"name": "f", "spec": {}, "axis": [0, 0, 0]}]}, "zero vector"),
        (
            {"fields": [{"name": "f", "axis": [0, 0, 1], "orientation": [1, 0, 0, 0]}]},
            "axis or orientation",
        ),
    ],
)
def test_a_bad_world_says_what_and_where(world, message):
    with pytest.raises(ConfigError, match=message):
        parse_world(world)


def test_clouds_need_air_in_a_file_as_in_a_world(tmp_path):
    clouds = yaml.safe_load((WORLDS / "kerbin.yaml").read_text())["clouds"]
    text = moon_world(mass=1.0, clouds=clouds)
    # Parsing accepts it; the world refuses it when built (World's own check)
    world = parse_world(text)
    assert world.bodies[0].atmosphere is None and world.bodies[0].clouds is not None


def test_a_game_keeps_its_world_anywhere(tmp_path):
    """The links and the images are looked for next to the world file."""
    rock = yaml.safe_load((WORLDS / "moon.yaml").read_text())
    (tmp_path / "rock.yaml").write_text(yaml.safe_dump(rock))
    path = tmp_path / "mine.yaml"
    path.write_text("bodies:\n  - {name: rock, file: rock.yaml, position: [1, 2, 3]}\n")
    (rock,) = parse_world(path).bodies
    assert rock.position == (1.0, 2.0, 3.0) and rock.spec == load_planet("moon")


def test_load_world_is_exported_lazily():
    assert world_builder.load_world is load_world
    with pytest.raises(AttributeError):
        world_builder.nonsense


# ── loaded ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def window(showbase):
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


def test_deep_space_loads_and_updates(window):
    from world_builder.settings import WorldSettings

    world = load_world(
        window, DEEP, settings=WorldSettings.preset("low"), backend="inline"
    )
    try:
        assert window.camLens.get_far() >= 1.0
        assert world.field("cube") is world.world.field("cube")
        for t in range(3):
            world.update(np.array([0.0, -9000.0, 500.0 + t]))
        assert world.depth.far >= 1.0e6 * 0.5
        assert world.world.depth is world.depth
    finally:
        world.destroy()


def test_a_loaded_world_draws_with_one_reversed_camera(window):
    """load_world's camera is the reversed one (#57): the world's renderers
    take its depth mode, given back on destroy."""
    from panda3d.core import DepthTestAttrib

    from world_builder.depth import (
        DEPTH_STANDARD,
        ReversedDepthCamera,
        depth_mode_input,
    )
    from world_builder.settings import WorldSettings

    world = load_world(
        window, DEEP, settings=WorldSettings.preset("low"), backend="inline"
    )
    try:
        depth = world.depth
        assert isinstance(depth, ReversedDepthCamera)
        assert world.world.depth is depth
        assert depth_mode_input()[0] == depth.mode
        greater_equal = DepthTestAttrib.M_greater_equal
        haze = world.field("cube").haze.node
        assert haze.get_attrib(DepthTestAttrib).get_mode() == greater_equal
        if world.world.sky is not None:
            sky = world.world.sky.node
            assert sky.get_attrib(DepthTestAttrib).get_mode() == greater_equal
        world.update(np.array([0.0, -9000.0, 500.0]))
        assert depth.far >= 1.0 and depth.near >= 1.0
    finally:
        world.destroy()
    assert depth_mode_input()[0] == DEPTH_STANDARD


_WINDOWLESS = """
import sys

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type none\\naudio-library-name null")
from world_builder import load_world
from world_builder.settings import WorldSettings

app = ShowBase()
for path in sys.argv[1:]:
    world = load_world(
        app, path, settings=WorldSettings.preset("low"), backend="inline"
    )
    assert world.depth is None and world.world.depth is None
    assert all(body.ocean is None for body in world.world.bodies)
    origin = np.zeros(3)
    if world.world.bodies:
        body = world.world.bodies[0]
        origin = np.asarray(body.position) + (0.0, 0.0, body.radius + 2000.0)
    for _ in range(3):
        world.update(origin)
        app.taskMgr.step()
    world.destroy()
    print("loaded", path)
"""


def test_a_world_loads_without_a_window():
    """A game's headless runs (``window-type none``): no depth camera and no
    sea (both draw through the window), the rest built and updated as usual.
    ShowBase is one per process and the suite's has a window: a process of
    its own."""
    import subprocess
    import sys

    worlds = [WORLDS / "tiny_world.yaml", DEEP]  # a sea; an asteroid field
    run = subprocess.run(
        [sys.executable, "-c", _WINDOWLESS, *map(str, worlds)],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONPATH": str(Path(world_builder.__file__).parents[1])},
    )
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.stdout.count("loaded") == len(worlds)


def test_float_depth_props_ask_for_a_float_buffer():
    from world_builder.depth import float_depth_fbprops

    props = float_depth_fbprops()
    assert props.get_float_depth() and props.get_depth_bits() == 32
