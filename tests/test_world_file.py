"""
World files (world_builder.world_file): the shipped worlds are the code's own
definitions written out, a file's rules are checked, and one world is loaded
on an offscreen window.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from fields import PRESETS as FIELDS
from planets import PRESETS

import world_builder
from world_builder.config import ConfigError, to_config
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


def test_the_moon_is_the_airless_preset_21000_km_out(three):
    moon = body(three, "moon")
    assert moon.spec == replace(PRESETS["moon"], seed=7).resolved()
    assert moon.atmosphere is None and moon.clouds is None and moon.ocean is None
    home = body(three, "planet")
    assert math.dist(moon.position, home.position) == pytest.approx(21.0e6, rel=1e-6)


def test_the_ice_planet_is_3000_km_across_with_air_and_no_sea(three):
    ice = body(three, "ice")
    assert ice.spec.radius * 2 == 3.0e6
    expected = replace(PRESETS["ice"], radius=1.5e6, seed=3)
    assert ice.spec == expected.resolved()
    assert not ice.spec.has_ocean and ice.ocean is None
    assert ice.atmosphere is not None and ice.clouds is not None
    assert ice.mass == pytest.approx(5.97e24 * 2.5**3)  # the home planet's density


def test_the_ring_is_the_demo_ring_around_the_ice_planet(three):
    (ring,) = three.fields
    ice = body(three, "ice")
    assert ring.name == "ring" and ring.position == ice.position
    shape = ring.spec.shape
    assert (shape.inner, shape.outer) == (5.0e6, 1.0e7)  # 5000 to 10,000 km
    preset = FIELDS["ring"]
    # Everything but the radii (and the rim fade, scaled with the width)
    assert ring.spec == replace(
        preset, shape=replace(preset.shape, inner=5.0e6, outer=1.0e7, radial_edge=5e5)
    )
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
    assert cube.spec == FIELDS["cube"]
    assert world.sun == SunLook(direction=(-0.45, -0.85, 0.3), color=(2.6, 2.5, 2.35))
    assert world.space_ambient == (0.03, 0.03, 0.04)


def test_a_world_file_has_no_run_time_values():
    for path in (THREE, DEEP):
        keys = set(yaml.safe_load(path.read_text()))
        assert not keys & {"quality", "settings", "camera", "backend", "clock"}


def test_a_world_written_in_full_reads_back():
    world = parse_world(DEEP)
    text = world_to_yaml(world)
    assert parse_world(yaml.safe_load(text)).fields == world.fields


def test_the_shipped_planet_files_are_the_planets_the_tests_use():
    """earth, desert, forest, moon and ice written out in full: a planet read
    from its file is the catalogue's (the moon's and ice's differ by what the
    shipped worlds choose: seed 7, radius 1500 km, seed 3)."""
    for name in ("earth", "desert", "forest"):
        assert load_planet(name) == PRESETS[name].resolved(), name
    assert load_planet("moon") == replace(PRESETS["moon"], seed=7).resolved()
    assert (
        load_planet("ice") == replace(PRESETS["ice"], radius=1.5e6, seed=3).resolved()
    )
    assert load_planet(WORLDS / "earth.yaml").radius == 6.371e6
    assert load_field("asteroid_cube") == FIELDS["cube"]
    assert load_field("asteroid_icy_ball") == FIELDS["icy_ball"]
    assert load_field(THREE).shape.inner == 5.0e6  # a world's first field
    assert load_planet(THREE).name == "earthlike"  # a world's first body


# ── the rules of a file ─────────────────────────────────────────────────────


def moon_world(**body):
    return {
        "bodies": [{"name": "m", "planet": {"name": "m", "radius": 2e5}, **body}],
    }


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
        (
            {"bodies": [{"name": "a", "mass": 1, "planet": {"name": "a"}, "x": 1}]},
            "unknown keys.*x",
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
    text = {
        "bodies": [
            {
                "name": "a",
                "mass": 1.0,
                "planet": {"name": "a"},
                "clouds": {},
            }
        ]
    }
    # Parsing accepts it; the world refuses it when built (World's own check)
    world = parse_world(text)
    assert world.bodies[0].atmosphere is None and world.bodies[0].clouds is not None


def test_a_game_keeps_its_world_anywhere(tmp_path):
    """The links and the images are looked for next to the world file."""
    (tmp_path / "rock.yaml").write_text(
        yaml.safe_dump(
            {"planet": to_config(PRESETS["moon"]), "mass": 1.0},
        )
    )
    path = tmp_path / "mine.yaml"
    path.write_text("bodies:\n  - {name: rock, file: rock.yaml, position: [1, 2, 3]}\n")
    (rock,) = parse_world(path).bodies
    assert rock.position == (1.0, 2.0, 3.0) and rock.spec == PRESETS["moon"].resolved()


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
