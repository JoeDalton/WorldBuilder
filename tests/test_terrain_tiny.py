"""The tiny test planet (PRESETS["tiny"])."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from planets import PRESETS

from world_builder.terrain import biomes
from world_builder.terrain.chart import dirs_from_latlon
from world_builder.terrain.cubesphere import grid_for
from world_builder.terrain.planet import Planet

#: The tiny planet over its test forest, grass and desert.
TINY = replace(PRESETS["tiny"], reference_lat=33.3, reference_lon=-147.0).resolved()


def test_tiny_is_a_cloudless_20_km_planet():
    """The test planet is small and drawn without clouds; the others keep theirs."""
    spec = TINY
    assert spec.name == "tiny" and spec.radius == 20e3 and not spec.has_clouds
    assert all(s.has_clouds for n, s in PRESETS.items() if n != "tiny")
    assert grid_for(spec.radius).max_level == 8


def test_forest_grass_and_desert_meet_near_the_start():
    """Within ~1.5 km of the level's reference, all three biomes are dominant
    somewhere, on low, gentle land."""
    spec = TINY
    planet = Planet(spec)
    lat, lon = np.meshgrid(
        spec.reference_lat + np.linspace(-4, 4, 41),
        spec.reference_lon + np.linspace(-4, 4, 41),
    )
    dirs = dirs_from_latlon(lat.ravel(), lon.ravel())
    e, t, h = planet.sample_dirs(dirs, None, apply_stamps=False)
    w = biomes.weights(t, h, e, np.zeros_like(e), spec.has_ocean)
    dom = biomes.dominant(w)
    for biome in (biomes.FOREST, biomes.GRASS, biomes.DESERT):
        assert (dom == biome).mean() > 0.1, biomes.BIOMES[biome]
    assert e.min() > 0.0 and e.max() < 300.0
