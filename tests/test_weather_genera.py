"""The weather's genera in the march (synoptic weather, Task 7): the stratus
as a family (the weather's stratus thickening into nimbostratus, one genus,
under the convective clouds' band), gated by the cloud quality, and the
cirrostratus from the weather."""

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from world_builder.clouds import noise
from world_builder.clouds.cloud import CloudSettings
from world_builder.clouds.genera import (
    FAMILY_SHARES,
    GENUS_PRESETS,
    HIGH_CHANNELS,
    STRATUS_FAMILY,
    WEATHER_GENERA,
    Feature,
    GeneraSpec,
    coverage_cube,
    genus_coverage,
    genus_share,
    genus_specs,
    high_genera,
    showcase_features,
)
from world_builder.clouds.march import sky_bands
from world_builder.clouds.weather import (
    CIRRUS_SHEET_GENERA,
    SYNOPTIC_CHANNELS,
    WeatherSpec,
    sample_cube,
    synoptic_cube,
)
from world_builder.quality import RESTART, QualityLevel
from world_builder.sky.starfield import cube_directions
from world_builder.world_file import load_planet

SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"
SIZE = 32
SHARE = noise.FAMILY_SHARE_CHANNEL


@pytest.fixture(scope="module")
def weather():
    return WeatherSpec(load_planet("earthlike"))


@pytest.fixture(scope="module")
def genera(weather):
    return GeneraSpec(weather, features=(), cirrus_weather=True)


def _weather_at(weather, names, size=SIZE):
    dirs = cube_directions(size).reshape(-1, 3)
    picks = [SYNOPTIC_CHANNELS.index(n) for n in names]
    return sample_cube(synoptic_cube(weather)[..., picks], dirs)


def _resolved_field():
    """A resolved field (its threshold set) for the CPU's density."""
    return replace(noise.DensityField(), threshold=(0.0, 1.0))


# ── The family ───────────────────────────────────────────────────────────────


def test_the_family_runs_from_the_stratus_to_a_deep_nimbostratus():
    """Share 0: the stratus's 500 m over its base (0.9-1.4 km of the 0.6-2.15
    km slab) at its density; share 1: the whole slab at the nimbostratus's."""
    below, thickness = STRATUS_FAMILY
    base, top, factor = noise.family_range(0.0)
    assert base * thickness == pytest.approx(below, abs=0.5)
    assert top * thickness == pytest.approx(below + 500.0, abs=0.5)
    assert factor == pytest.approx(noise.FAMILY_THIN_DENSITY)
    preset = GENUS_PRESETS["stratus"]
    assert factor * preset.field["density"] == pytest.approx(0.05)  # the stratus's
    assert noise.family_range(1.0) == pytest.approx((0.0, 1.0, 1.0))
    # In between: lower, taller and denser as the share rises
    b, t, f = noise.family_range(0.5)
    assert 0.0 < b < base and top < t < 1.0 and factor < f < 1.0


def test_a_family_takes_its_profile_over_its_clouds_own_height():
    """Each end keeps its genus's vertical shape: the slab's heights remapped
    to the cloud's at the share, none outside it, the density scaled."""
    plain = noise.GenusField("st", _resolved_field(), None, recipe=True)
    family = replace(plain, family=True)
    assert noise.recipe_flags(family) == noise.recipe_flags(plain) | noise.RECIPE_FAMILY
    base, top, factor = noise.family_range(0.0)
    h = np.array([0.5 * base, 0.5 * (base + top), 0.5 * (top + 1.0)])
    local = noise.family_height(h, 0.0)
    assert local[0] < 0.0 and local[1] == pytest.approx(0.5) and local[2] > 1.0
    value = np.full(3, 10.0)  # well inside the window
    got = noise.recipe_density_from(family, local, value, None, 0.0, 0.0)
    assert got[0] == 0.0 and got[2] == 0.0
    assert got[1] == pytest.approx(factor * noise.recipe_profile(plain, 0.5))


def test_the_family_slab_stays_under_the_convective_band():
    """0.6-2.15 km under the cumulus's 2.2 km base: two bands of one genus
    each, the march's fast loop for both."""
    slabs = [(2200.0, 8300.0)]
    for name, _, base, thickness, _ in genus_specs(
        GeneraSpec(WeatherSpec(load_planet("earthlike")), features=()), 2200.0
    ):
        if name == "stratus":
            slabs.append((base, thickness))
            assert (base, thickness) == (900.0 - STRATUS_FAMILY[0], STRATUS_FAMILY[1])
    bands = sky_bands(slabs)
    assert len(bands) == 2


def test_the_shaders_share_the_family_constants():
    field = (SHADERS / "cloud_field.glsl").read_text()
    assert int(re.search(r"const int RECIPE_FAMILY = (\d+);", field)[1]) == (
        noise.RECIPE_FAMILY
    )
    for name in ("THIN_BASE", "THIN_TOP", "THIN_DENSITY"):
        value = float(re.search(rf"const float FAMILY_{name} = ([\d.]+);", field)[1])
        assert value == getattr(noise, f"FAMILY_{name}"), name
    assert int(re.search(r"const int FAMILY_SHARE_CHANNEL = (\d+);", field)[1]) == SHARE
    # The density and the weather read it
    density = field[field.index("float genusRecipeDensity(") :]
    assert "weather.z" in density[: density.index("\n}\n")]
    at = field[field.index("vec4 genusWeatherAt(") :]
    assert "cover[FAMILY_SHARE_CHANNEL]" in at[: at.index("\n}\n")]


# ── The weather's genera ─────────────────────────────────────────────────────


def test_the_stratus_is_the_weathers_stratus_stratocumulus_and_nimbostratus(
    weather, genera
):
    assert WEATHER_GENERA["stratus"] == ("stratus", "stratocumulus", "nimbostratus")
    assert FAMILY_SHARES["stratus"] == "nimbostratus"
    names = ("stratus", "stratocumulus", "nimbostratus")
    at = _weather_at(weather, names)
    cover = genus_coverage(genera, "stratus", SIZE).reshape(-1)
    np.testing.assert_allclose(cover, at.max(axis=1), atol=1e-6)
    share = genus_share(genera, "stratus", SIZE).reshape(-1)
    some = at.sum(axis=1) > 0.05
    assert some.any()
    np.testing.assert_allclose(
        share[some], at[some, 2] / at[some].sum(axis=1), atol=1e-5
    )
    assert np.all((share >= 0.0) & (share <= 1.0))
    # The weather has both ends: plain stratus and deep rain cloud
    assert (share[some] < 0.3).any() and (share[some] > 0.7).any()


def test_cube_a_holds_the_stratus_and_its_share(genera):
    cube = coverage_cube(genera, "A", SIZE)
    np.testing.assert_array_equal(cube[..., 0], genus_coverage(genera, "stratus", SIZE))
    np.testing.assert_array_equal(
        cube[..., SHARE], genus_share(genera, "stratus", SIZE)
    )
    assert not cube[..., 1].any() and not cube[..., 3].any()


def test_the_march_takes_the_stratus_and_the_cirrostratus(genera):
    """With the weather, the march draws the stratus (its nimbostratus in
    it), and the cirrostratus in the high layer; not the plain
    stratocumulus, nimbostratus, altostratus or altocumulus (no feature)."""
    assert [s[0] for s in genus_specs(genera, covered=True)] == [
        "convective",
        "stratus",
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


def test_without_the_nimbostratus_the_stratus_is_the_weathers_stratus_alone(
    weather, genera
):
    """weather_nimbostratus off (the cloud quality's, below high): the
    stratus covers the weather's stratus and stratocumulus, no share."""
    plain = replace(genera, weather_nimbostratus=False)
    at = _weather_at(weather, ("stratus", "stratocumulus"))
    cover = genus_coverage(plain, "stratus", SIZE).reshape(-1)
    np.testing.assert_allclose(cover, at.max(axis=1), atol=1e-6)
    assert not genus_share(plain, "stratus", SIZE).any()
    assert not coverage_cube(plain, "A", SIZE)[..., SHARE].any()


def test_the_cloud_quality_gates_the_weathers_nimbostratus():
    knob = dict(CloudSettings.knobs())["weather_nimbostratus"]
    assert knob.apply == RESTART
    assert CloudSettings().weather_nimbostratus
    for level, on in (
        (QualityLevel.LOW, False),
        (QualityLevel.MID, False),
        (QualityLevel.HIGH, True),
        (QualityLevel.ULTRA, True),
    ):
        assert CloudSettings.preset(level).weather_nimbostratus is on, level


def test_without_the_weather_the_showcase_keeps_its_own_genera(weather):
    """The showcase (no weather): the plain nimbostratus and altostratus
    features in their channels, the stratus no share."""
    showcase = GeneraSpec(weather, features=showcase_features(), background=False)
    names = [s[0] for s in genus_specs(showcase, covered=True)]
    assert "nimbostratus" in names and "altostratus" in names
    cube = coverage_cube(showcase, "A", SIZE)
    np.testing.assert_array_equal(
        cube[..., SHARE], genus_coverage(showcase, "nimbostratus", SIZE)
    )
    # Without the cirrus weather, no weather cirrostratus
    assert [n for n, _ in high_genera(replace(showcase, features=()), True)] == []


def test_the_weathers_nimbostratus_and_a_nimbostratus_feature_cannot_share(genera):
    spec = replace(genera, features=(Feature(0.0, 0.0, 20e3, "nimbostratus", 0.8),))
    with pytest.raises(ValueError, match="nimbostratus"):
        coverage_cube(spec, "A", SIZE)
    # Without the weather's nimbostratus, the feature has its channel
    coverage_cube(replace(spec, weather_nimbostratus=False), "A", SIZE)
