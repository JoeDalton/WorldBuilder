"""The Earth-scale and rocky Jupiter-scale planets, and the weather on them."""

import os
import time

import pytest

from world_builder.clouds.synoptic import Scales
from world_builder.terrain.cubesphere import grid_for
from world_builder.terrain.planet import PRESETS


def test_presets_exist():
    earth = PRESETS["earth"]
    assert earth.radius == 6.371e6 and earth.day_length_h == 23.934
    assert earth.has_ocean
    jupiter = PRESETS["jupiter_rocky"]
    assert jupiter.radius == 69.911e6 and jupiter.day_length_h == 9.925


def test_earth_preset_has_earths_scales():
    s = Scales.from_planet(PRESETS["earth"])
    assert 27.0 < s.hadley_edge < 33.0
    assert 800e3 < s.deformation_radius < 1200e3


@pytest.mark.parametrize("name", ["earth", "jupiter_rocky"])
def test_grid_levels(name):
    grid = grid_for(PRESETS[name].radius)
    assert 1.0 <= grid.spacing0 <= 4.0, grid.spacing0


@pytest.mark.skipif(
    os.environ.get("WEATHER_TIMING") != "1",
    reason="a timing measurement: run with WEATHER_TIMING=1 on a quiet machine",
)
def test_jupiter_bake_bounded():
    from world_builder.clouds.weather import WeatherSpec, coverage_map

    start = time.perf_counter()
    coverage_map(WeatherSpec(PRESETS["earthlike"]))
    kerbin = time.perf_counter() - start
    start = time.perf_counter()
    coverage_map(WeatherSpec(PRESETS["jupiter_rocky"]))
    jupiter = time.perf_counter() - start
    assert jupiter < 4.0 * kerbin, (kerbin, jupiter)


@pytest.mark.parametrize("name", ["earth", "jupiter_rocky"])
def test_the_demo_level_keeps_the_presets_radius(name):
    """The demo level's 600 km radius must not override these presets'."""
    from world_builder.terrain.level import DEMO_LEVEL, level_for, load_level

    spec, pose = load_level(level_for(DEMO_LEVEL, name), name)
    assert spec.radius == PRESETS[name].radius
    assert spec.day_length_h == PRESETS[name].day_length_h
    assert pose is not None
