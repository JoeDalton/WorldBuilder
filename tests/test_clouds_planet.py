"""
Unit tests for a planet's clouds as one part (world_builder.clouds.planet):
the rule its layers must keep, the volume driven through it, and the
weather's cirrus in the march's high layer (``cirrus=True``). The march
itself: test_clouds_march.py; the high layer: test_clouds_high.py.
"""

import numpy as np
import pytest

from world_builder.atmosphere import AtmosphereParams, Atmospheres
from world_builder.clouds.cloud import CloudSettings, CloudType
from world_builder.clouds.genera import HIGH_SLAB
from world_builder.clouds.march import HIGH_MASK_SHIFT, CloudMarch
from world_builder.clouds.planet import PlanetClouds, check_layers
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS as PLANETS
from world_builder.terrain.workers import ChunkExecutor

R = 600e3


@pytest.mark.parametrize(
    "cloud_type,altitude,air_top,message",
    [
        # The cumulus's slab through the high layer's base (11 km)
        (CloudType.CUMULUS, 10800.0, None, "above the clouds"),
        (CloudType.CUMULUS, 2200.0, 12000.0, "inside the air"),
    ],
)
def test_the_cirrus_must_lie_above_the_volume_and_inside_the_air(
    cloud_type, altitude, air_top, message
):
    with pytest.raises(ValueError, match=message):
        check_layers(cloud_type, altitude, True, air_top)


def test_layers_that_fit_pass():
    check_layers(CloudType.CUMULUS, 2200.0, True, 60e3)
    check_layers(CloudType.CUMULUS, 2200.0, False, 1000.0)  # nothing to check


@pytest.fixture
def executor():
    made = ChunkExecutor(backend="inline", inline_budget=64)
    yield made
    made.shutdown()


def _clouds(showbase, executor, **kwargs):
    return PlanetClouds(
        showbase.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(octave_bands=2),
        planet=PLANETS["forest"],
        **kwargs,
    )


def test_the_air_bounds_the_cirrus_before_anything_is_built(showbase, executor):
    atmospheres = Atmospheres()
    index = atmospheres.add(R, AtmosphereParams(height=8500.0))
    before = showbase.render.get_num_children()
    with pytest.raises(ValueError, match="inside the air"):
        _clouds(
            showbase,
            executor,
            atmospheres=atmospheres,
            atmosphere=index,
            cirrus=True,
        )
    assert showbase.render.get_num_children() == before


def test_no_cirrus_is_the_volume_alone(showbase, executor):
    clouds = _clouds(showbase, executor)
    try:
        assert isinstance(clouds.volume, CloudMarch)
        assert clouds.volume.root.get_parent() == clouds.root
        assert clouds.volume.high_genera == []
        assert clouds.high_slab is None
        band = clouds.volume.sky_bands[-1]
        assert clouds.top_radius == pytest.approx(R + band[0] + band[1])
    finally:
        clouds.remove()


def test_the_cirrus_is_the_marchs_high_layer(showbase, executor):
    """Without genera, cirrus=True gives the one-type march a high layer:
    the weather's cirrus alone (genera.weather_cirrus), on the clouds' own
    weather; the C key's toggle is its bit of uGenusMask."""
    clouds = _clouds(showbase, executor, cirrus=True, weather_seed=5)
    try:
        march = clouds.volume
        assert march.genus_names == ["clouds"]
        assert march.high_genera == ["cirrus"]
        assert clouds.high_slab == HIGH_SLAB
        assert clouds.top_radius == pytest.approx(R + sum(HIGH_SLAB))
        assert march._high_spec.weather is clouds.weather
        assert march._high_spec.cirrus_weather
        assert clouds.genus_visible("cirrus")
        everything = march._genus_mask[0]
        clouds.set_genus_visible("cirrus", False)
        assert not clouds.genus_visible("cirrus")
        assert march._genus_mask[0] == everything & ~(1 << HIGH_MASK_SHIFT)
        assert clouds.genus_visible(0)  # the low clouds untouched
        clouds.set_genus_visible("cirrus", True)
        assert march._genus_mask[0] == everything
    finally:
        clouds.remove()


def test_the_volume_is_driven_through_it(showbase, executor):
    clouds = _clouds(showbase, executor, cirrus=True)
    try:
        volume = clouds.volume
        clouds.set_sun((0.0, 0.0, 2.0))
        assert tuple(volume._sun[0]) == (0.0, 0.0, 1.0)
        clouds.set_sky_light(0.7)
        assert volume._sky_light == 0.7
        # The volume takes the pixel angle at its scale
        clouds.set_pixel_angle(0.001, volume_scale=2.0)
        assert volume._pixel_angle == pytest.approx(0.002)
        reflected = CloudSettings(octave_bands=2, reflected=False)
        clouds.set_settings(reflected)
        assert clouds.settings is reflected and volume.settings is reflected
        camera = TangentFrame.at(20.0, 40.0).up * (R + 1000.0)
        for _ in range(4):
            clouds.update(camera)
        assert clouds.stats == volume.stats
        # Drawn camera-relative: the planet's centre less the camera
        np.testing.assert_allclose(
            volume.deck.get_pos(showbase.render), -camera, atol=1.0
        )
    finally:
        clouds.remove()
    assert clouds.root.is_empty()
