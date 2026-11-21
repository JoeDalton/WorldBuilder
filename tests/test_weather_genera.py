"""The weather's genera in the march (synoptic weather, Task 7): the frontal
shield (altostratus thickening into nimbostratus, one family genus) and the
cirrostratus from the weather."""

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from world_builder.clouds import noise
from world_builder.clouds.genera import (
    FAMILY_SHARES,
    GENUS_PRESETS,
    HIGH_CHANNELS,
    WEATHER_GENERA,
    Feature,
    GeneraSpec,
    coverage_cube,
    genus_coverage,
    genus_share,
    genus_specs,
    has_coverage,
    high_genera,
    showcase_features,
)
from world_builder.clouds.weather import (
    CIRRUS_SHEET_GENERA,
    SYNOPTIC_CHANNELS,
    WeatherSpec,
    sample_cube,
    synoptic_cube,
)
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.planet import PRESETS

SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"
SIZE = 32


@pytest.fixture(scope="module")
def weather():
    return WeatherSpec(PRESETS["earthlike"])


@pytest.fixture(scope="module")
def genera(weather):
    return GeneraSpec(weather, features=(), cirrus_weather=True)


def _weather_at(weather, names, size=SIZE):
    dirs = cube_directions(size).reshape(-1, 3)
    picks = [SYNOPTIC_CHANNELS.index(n) for n in names]
    return sample_cube(synoptic_cube(weather)[..., picks], dirs)


# ── The family's profile ─────────────────────────────────────────────────────


def test_the_family_runs_from_a_thin_high_layer_to_a_deep_one():
    h = np.linspace(0.0, 1.0, 201)
    thin = noise.family_profile(h, 0.0)
    thick = noise.family_profile(h, 1.0)
    # Altostratus: only over its base (4.7 km of 0.6-6 km), thin
    assert thin[h < noise.FAMILY_THIN_BASE].max() == 0.0
    assert thin.max() == pytest.approx(noise.FAMILY_THIN_DENSITY, rel=1e-6)
    # Nimbostratus: from the bottom to its top (5 km), whole
    assert (
        thick[(h > noise.FAMILY_EDGE) & (h < noise.FAMILY_THICK_TOP - 0.05)].min()
        == 1.0
    )
    assert thick[h >= noise.FAMILY_THICK_TOP].max() == 0.0
    # In between: lower and denser as the share rises
    half = noise.family_profile(h, 0.5)
    assert thin.max() < half.max() < thick.max()
    assert h[np.argmax(half > 0)] < h[np.argmax(thin > 0)]


def test_a_family_genus_scales_its_recipe_density_by_its_profile():
    preset = GENUS_PRESETS["frontal_shield"]
    assert preset.recipe["family"] and preset.slab == (600.0, 5400.0)
    plain = noise.GenusField("ns", _resolved_field(), None, recipe=True)
    family = replace(plain, family=True)
    assert noise.recipe_flags(family) == noise.recipe_flags(plain) | noise.RECIPE_FAMILY
    h = np.array([0.3, 0.9, 0.9])
    value = np.full(3, 10.0)  # well inside the window: density 1 x profiles
    share = np.array([1.0, 1.0, 0.0])
    base = noise.recipe_density_from(plain, h, value, None, 0.0)
    got = noise.recipe_density_from(family, h, value, None, 0.0, share)
    np.testing.assert_allclose(got, base * noise.family_profile(h, share))
    assert got[0] > 0.0 and got[1] == 0.0 and got[2] > 0.0


def _resolved_field():
    """A resolved field (its threshold set) for the CPU's density."""
    return replace(noise.DensityField(), threshold=(0.0, 1.0))


def test_the_shaders_share_the_family_constants():
    field = (SHADERS / "cloud_field.glsl").read_text()
    assert int(re.search(r"const int RECIPE_FAMILY = (\d+);", field)[1]) == (
        noise.RECIPE_FAMILY
    )
    for name in ("THIN_BASE", "THICK_TOP", "THIN_DENSITY", "EDGE"):
        value = float(re.search(rf"const float FAMILY_{name} = ([\d.]+);", field)[1])
        assert value == getattr(noise, f"FAMILY_{name}"), name
    # The density and the weather read it
    density = field[field.index("float genusRecipeDensity(") :]
    assert "familyProfile(h, weather.z)" in density[: density.index("\n}\n")]
    at = field[field.index("vec4 genusWeatherAt(") :]
    assert "RECIPE_FAMILY" in at[: at.index("\n}\n")]


# ── The weather's genera ─────────────────────────────────────────────────────


def test_the_frontal_shield_is_the_weathers_altostratus_and_nimbostratus(
    weather, genera
):
    assert WEATHER_GENERA["frontal_shield"] == ("altostratus", "nimbostratus")
    assert FAMILY_SHARES["frontal_shield"] == "nimbostratus"
    at = _weather_at(weather, ("altostratus", "nimbostratus"))
    cover = genus_coverage(genera, "frontal_shield", SIZE).reshape(-1)
    np.testing.assert_allclose(cover, at.max(axis=1), atol=1e-6)
    share = genus_share(genera, "frontal_shield", SIZE).reshape(-1)
    some = at.sum(axis=1) > 0.05
    assert some.any()
    np.testing.assert_allclose(
        share[some], at[some, 1] / at[some].sum(axis=1), atol=1e-5
    )
    assert np.all((share >= 0.0) & (share <= 1.0))
    # The weather has both ends: thin shields and deep cores
    assert (share[some] < 0.3).any() and (share[some] > 0.7).any()


def test_cube_a_holds_the_shield_and_its_share(genera):
    cube = coverage_cube(genera, "A", SIZE)
    channel = GENUS_PRESETS["frontal_shield"].coverage[1]
    assert channel == 2
    np.testing.assert_array_equal(
        cube[..., 2], genus_coverage(genera, "frontal_shield", SIZE)
    )
    np.testing.assert_array_equal(
        cube[..., 3], genus_share(genera, "frontal_shield", SIZE)
    )


def test_the_march_takes_the_shield_and_the_cirrostratus(genera):
    """With the weather, the march draws one genus for the altostratus and
    the nimbostratus, and the cirrostratus in the high layer; not the plain
    altostratus, nimbostratus, stratocumulus or altocumulus (no feature)."""
    assert [s[0] for s in genus_specs(genera, covered=True)] == [
        "convective",
        "stratus",
        "frontal_shield",
    ]
    assert [n for n, _ in high_genera(genera, covered=True)] == [
        "cirrus",
        "cirrostratus",
    ]
    cube = coverage_cube(genera, "B", SIZE)
    weather = _weather_at(genera.weather, ("cirrostratus",))[:, 0]
    np.testing.assert_allclose(
        cube[..., HIGH_CHANNELS["cirrostratus"]].reshape(-1), weather, atol=1e-6
    )


def test_the_cirrus_cover_leaves_the_cirrostratus_to_its_genus():
    assert CIRRUS_SHEET_GENERA == ("cirrus", "cirrocumulus")
    assert "cirrostratus" in WEATHER_GENERA


def test_without_the_weather_the_showcase_keeps_its_own_genera(weather):
    """The showcase (no weather): the plain altostratus and nimbostratus
    features in their channels, no shield."""
    showcase = GeneraSpec(weather, features=showcase_features(), background=False)
    assert not has_coverage(showcase, "frontal_shield")
    names = [s[0] for s in genus_specs(showcase, covered=True)]
    assert "nimbostratus" in names and "altostratus" in names
    assert "frontal_shield" not in names
    cube = coverage_cube(showcase, "A", SIZE)
    np.testing.assert_array_equal(
        cube[..., 3], genus_coverage(showcase, "altostratus", SIZE)
    )
    # Without the cirrus weather, no weather cirrostratus
    assert [n for n, _ in high_genera(replace(showcase, features=()), True)] == []


def test_the_shield_and_an_altostratus_feature_cannot_share_its_channel(genera):
    spec = replace(genera, features=(Feature(0.0, 0.0, 20e3, "altostratus", 0.8),))
    with pytest.raises(ValueError, match="frontal shield"):
        coverage_cube(spec, "A", SIZE)
