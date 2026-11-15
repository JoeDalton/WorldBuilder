"""
The separated-genera showcase (clouds.genera.showcase_features,
GeneraSpec.background; CloudMarch.set_genus_visible): every genus on its own
disc, in a row along one horizon from the start, each at its real altitude.
"""

import math
import re
from pathlib import Path

import numpy as np

from world_builder.clouds.genera import (
    GENUS_DEPTHS,
    GeneraSpec,
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
        ["stratus", "humilis", "mediocris", "congestus", "cumulonimbus"]
    )


def test_the_showcase_is_a_row_in_view_from_the_start():
    """30-50 km out (near enough to read through the haze), all within 75
    degrees of north (one sweep of the view), and no feature's cloud
    reaching another's."""
    features = showcase_features()
    for f in features:
        distance = math.hypot(f.east, f.north)
        assert 30e3 <= distance <= 50e3, f
        assert abs(math.degrees(math.atan2(f.east, f.north))) <= 75.0, f
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            gap = math.hypot(a.east - b.east, a.north - b.north)
            assert gap > _reach(a) + _reach(b), (a.genus, b.genus, gap)


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
    text = (SHADERS / "cloud_march.frag").read_text()
    body = text[text.index("float genusDensity(") :]
    body = body[: body.index("\n}\n")]
    assert re.search(r"\(uGenusMask >> g\) & 1\) == 0", body)
    assert "uniform int   uGenusMask;" in text
