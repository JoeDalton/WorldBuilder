"""The Earth-scale and rocky Jupiter-scale planets, and the weather on them."""

from __future__ import annotations

import os
import time
from dataclasses import replace

import pytest

from world_builder.clouds.synoptic import Scales
from world_builder.terrain.cubesphere import grid_for
from world_builder.world_file import load_planet

#: A rocky world of Jupiter's radius and spin (no gas giant: the same terrain,
#: much larger).
JUPITER_ROCKY = replace(load_planet("earthlike"), radius=69.911e6, day_length_h=9.925)


def test_earth_exists():
    earth = load_planet("earth")
    assert earth.radius == 6.371e6 and earth.day_length_h == 23.934
    assert earth.has_ocean


def test_earth_preset_has_earths_scales():
    s = Scales.from_planet(load_planet("earth"))
    assert 27.0 < s.hadley_edge < 33.0
    assert 800e3 < s.deformation_radius < 1200e3


@pytest.mark.parametrize("spec", [load_planet("earth"), JUPITER_ROCKY])
def test_grid_levels(spec):
    grid = grid_for(spec.radius)
    assert 1.0 <= grid.spacing0 <= 4.0, grid.spacing0


@pytest.mark.skipif(
    os.environ.get("WEATHER_TIMING") != "1",
    reason="a timing measurement: run with WEATHER_TIMING=1 on a quiet machine",
)
def test_jupiter_bake_bounded():
    from world_builder.clouds.weather import WeatherSpec, coverage_map

    start = time.perf_counter()
    coverage_map(WeatherSpec(load_planet("earthlike")))
    kerbin = time.perf_counter() - start
    start = time.perf_counter()
    coverage_map(WeatherSpec(JUPITER_ROCKY))
    jupiter = time.perf_counter() - start
    assert jupiter < 4.0 * kerbin, (kerbin, jupiter)
