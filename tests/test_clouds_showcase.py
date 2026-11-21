"""
The separated-genera showcase (clouds.genera.showcase_features,
GeneraSpec.background; CloudMarch.set_genus_visible): every genus on its own
disc, the cumulus in a row along the northern horizon from the start, the
stratiform and mid-level genera along the southern, each at its real
altitude.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from world_builder.clouds.genera import (
    CONVECTIVE_SPECIES,
    GENUS_DEPTHS,
    HIGH_CHANNELS,
    SHOWCASE_ROW,
    SHOWCASE_SOUTH,
    Feature,
    GeneraSpec,
    genus_coverage,
    genus_planes,
    showcase_features,
)
from world_builder.clouds.weather import WeatherSpec
from world_builder.terrain.planet import PlanetSpec

R = 600e3
SHADERS = Path(__file__).parent.parent / "src/world_builder/clouds/shaders"


def _reach(feature) -> float:
    """How far a feature's cloud spreads from its centre: a cumulonimbus's
    anvil shelf to 3 radii (its plume longer downwind: 4 to be safe)."""
    return feature.radius * (4.0 if feature.genus == "cumulonimbus" else 1.0)


def test_the_showcase_has_every_genus_of_the_march_once():
    names = [f.genus for f in showcase_features()]
    assert sorted(names) == sorted(
        ["humilis", "mediocris", "congestus", "cumulonimbus", "stratus",
         "stratocumulus", "nimbostratus", "altostratus", "altocumulus",
         "cirrus", "cirrostratus", "cirrocumulus"]
    )  # fmt: skip
    # Every genus of the march but the convective one (its species) is there
    for name in GeneraSpec.__dataclass_fields__["genera"].default[1:]:
        assert name in names, name


def _slots(row) -> list[Feature]:
    """Every slot of a showcase row as a feature, those still to come too."""
    return [
        Feature(
            distance * math.sin(math.radians(bearing)),
            distance * math.cos(math.radians(bearing)),
            radius,
            genus,
            coverage,
        )
        for genus, bearing, distance, radius, coverage in row
    ]


def test_the_showcase_is_two_rows_in_view_from_the_start():
    """30-50 km out (near enough to read through the haze), the cumulus
    within 75 degrees of north, the stratiform and mid-level genera of
    south (one sweep of the view each), and no feature's cloud reaching
    another's, the slots of the genera still to come (nimbostratus,
    altostratus, altocumulus) counted."""
    north, south = _slots(SHOWCASE_ROW), _slots(SHOWCASE_SOUTH)
    assert all(f.genus in CONVECTIVE_SPECIES for f in north)
    assert not any(f.genus in CONVECTIVE_SPECIES for f in south)
    assert {"nimbostratus", "altostratus", "altocumulus"} <= {f.genus for f in south}
    for f in north + south:
        distance = math.hypot(f.east, f.north)
        assert 30e3 <= distance <= 50e3, f
        bearing = math.degrees(math.atan2(f.east, f.north))
        off = abs(bearing) if f in north else 180.0 - abs(bearing)
        assert off <= 75.0, f
    features = north + south
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            gap = math.hypot(a.east - b.east, a.north - b.north)
            assert gap > _reach(a) + _reach(b), (a.genus, b.genus, gap)
    # The high genera's slots (overhead: test_clouds_high's) and the rows'
    low = {f.genus for f in showcase_features()} - set(HIGH_CHANNELS)
    assert low <= {f.genus for f in features}


def test_the_southern_slots_fit_sheets_15_km_wide():
    """Each southern slot holds a sheet 10-15 km wide (its radius 5-7.5
    km), and sheets 15 km wide in every slot would not touch."""
    south = _slots(SHOWCASE_SOUTH)
    for f in south:
        assert 5e3 <= f.radius <= 7.5e3, f
    for i, a in enumerate(south):
        for b in south[i + 1 :]:
            assert math.hypot(a.east - b.east, a.north - b.north) > 15e3


def _spec(**kwargs):
    return GeneraSpec(
        WeatherSpec(PlanetSpec(name="clouds", radius=R)), focus=(10.0, 20.0), **kwargs
    )


def test_without_background_only_the_features_hold_convective_cloud():
    planes = genus_planes(_spec(features=showcase_features(), background=False))
    with_bg = genus_planes(_spec(features=showcase_features()))
    # Most of the planet is far from the features: no fair-weather cumulus
    assert (planes[..., 0] > 0.0).mean() < 0.01
    assert (with_bg[..., 0] > 0.0).mean() > 0.3


def test_each_stratiform_genus_fills_its_own_map():
    """A stratiform feature's coverage is its genus's map alone
    (genera.genus_coverage): its coverage at its centre, nothing in the
    convective coverage or another genus's map."""
    from world_builder.clouds.genera import _feature_dirs
    from world_builder.clouds.weather import sample_cube

    spec = _spec(features=showcase_features(), background=False)
    planes = genus_planes(spec)
    maps = {
        n: genus_coverage(spec, n)
        for n in (
            "stratus",
            "stratocumulus",
            "nimbostratus",
            "altostratus",
            "altocumulus",
        )
    }
    for feature, centre, _ in _feature_dirs(spec):
        if feature.genus not in maps:
            continue
        for name, plane in maps.items():
            at = float(sample_cube(plane, centre[None])[0])
            expected = feature.coverage if name == feature.genus else 0.0
            assert at == pytest.approx(expected, abs=0.02), (feature.genus, name)
        assert float(sample_cube(planes[..., 0], centre[None])[0]) == 0.0
    np.testing.assert_array_equal(planes[..., 2], maps["stratus"])


def test_the_stratiform_genera_fill_cube_a_a_channel_each():
    """Coverage cube A: the stratus channel 0, the stratocumulus 1, the
    nimbostratus 2, the altostratus 3; cube B: the altocumulus channel 0,
    the showcase's high genera theirs (the cirrus 1, the cirrostratus 2,
    the cirrocumulus 3) (genera.coverage_cube)."""
    from world_builder.clouds.genera import GENUS_PRESETS, coverage_cube

    spec = _spec(features=showcase_features(), background=False)
    cube = coverage_cube(spec, "A")
    for channel, name in enumerate(
        ("stratus", "stratocumulus", "nimbostratus", "altostratus")
    ):
        assert GENUS_PRESETS[name].coverage == ("A", channel)
        np.testing.assert_array_equal(cube[..., channel], genus_coverage(spec, name))
    assert GENUS_PRESETS["altocumulus"].coverage == ("B", 0)
    cube_b = coverage_cube(spec, "B")
    own = genus_coverage(spec, "altocumulus")
    np.testing.assert_array_equal(cube_b[..., 0], own)
    assert own.max() > 0.5
    for name in ("cirrus", "cirrostratus", "cirrocumulus"):
        channel = HIGH_CHANNELS[name]
        np.testing.assert_array_equal(cube_b[..., channel], genus_coverage(spec, name))


def test_each_cumulus_species_has_its_convective_depth():
    from world_builder.clouds.genera import _feature_dirs
    from world_builder.clouds.weather import sample_cube

    spec = _spec(features=showcase_features(), background=False)
    planes = genus_planes(spec)
    for feature, centre, _ in _feature_dirs(spec):
        if feature.genus in GENUS_DEPTHS:
            depth = float(sample_cube(planes[..., 1], centre[None])[0])
            assert abs(depth - GENUS_DEPTHS[feature.genus]) < 0.05, feature.genus
    assert (
        GENUS_DEPTHS["humilis"] < GENUS_DEPTHS["mediocris"] < GENUS_DEPTHS["congestus"]
    )
    assert np.isclose(GENUS_DEPTHS["congestus"], 0.5)


def test_the_march_skips_a_hidden_genus():
    text = (SHADERS / "cloud_march.glsl").read_text()
    body = text[text.index("float genusDensity(") :]
    body = body[: body.index("\n}\n")]
    assert re.search(r"\(uGenusMask >> g\) & 1\) == 0", body)
    assert "uniform int   uGenusMask;" in text
