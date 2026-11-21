"""Clouds seen from afar (the march's far path): opaque masses with sharp,
ragged edges and clear gaps, not a translucent wash (cloud_march.glsl's
macroCover, macroThickness), over a weather whose mesoscale texture keeps
its physical size on any planet (WeatherSpec.meso_wavelength, metres)."""

import re
from pathlib import Path

import numpy as np

from world_builder.clouds.weather import WeatherSpec, coverage_at
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS, Planet

SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"


def _pairs(spec, separation: float, count: int = 400):
    """Coverage at *count* random points and at as many *separation* metres
    east of them, over the storm tracks (40-60 degrees)."""
    rng = np.random.default_rng(3)
    lat = rng.uniform(40.0, 60.0, count) * rng.choice([-1, 1], count)
    lon = rng.uniform(-180.0, 180.0, count)
    a, b = [], []
    for la, lo in zip(lat, lon):
        f = TangentFrame.at(float(la), float(lo))
        angle = separation / spec.planet.radius
        a.append(f.up)
        b.append(np.cos(angle) * f.up + np.sin(angle) * f.east)
    planet = Planet(spec.planet)
    return coverage_at(spec, np.array(a), planet), coverage_at(
        spec, np.array(b), planet
    )


def test_earths_mesoscale_texture_has_its_physical_size():
    """Points 120 km apart differ on Earth: the texture's largest feature
    is 160 km, as on the 600 km planet (scaled by the planet's feature
    scale, 10.6, it was 1700 km and the clouds came out as broad brushes from
    orbit: a correlation of 0.72 at 120 km then, 0.45 now)."""
    a, b = _pairs(WeatherSpec(PRESETS["earth"]), 120e3)
    assert np.corrcoef(a, b)[0, 1] < 0.6
    near_a, near_b = _pairs(WeatherSpec(PRESETS["earth"]), 10e3)
    assert np.corrcoef(near_a, near_b)[0, 1] > 0.8


def test_the_far_path_draws_masses_not_a_wash():
    march = (SHADERS / "cloud_march.glsl").read_text()
    for name in (
        "MACRO_FEATURES",
        "MACRO_WEIGHTS",
        "MACRO_AMOUNT",
        "MACRO_EDGE",
        "MACRO_THIN",
    ):
        assert re.search(rf"const \w+ {name} = ", march), name
    cover = march[march.index("float macroCover(") :]
    cover = cover[: cover.index("\n}\n")]
    # Each octave fades in as the view pixel resolves it; stepped at a half
    assert cover.count("resolved(MACRO_FEATURES") == 2
    assert "smoothstep(0.5 - MACRO_EDGE, 0.5 + MACRO_EDGE" in cover
    assert "16384" in cover  # the comparison's debug bit
    density = march[march.index("float genusDensity(") :]
    density = density[: density.index("\n}\n")]
    assert (
        "macroCover(farCoverage" in density and "macroThickness(farCoverage)" in density
    )
    # The far table's light keeps the pixel's own coverage
    assert re.search(r"farCoverage = genusCoverageAt\(g, p\);", density)
