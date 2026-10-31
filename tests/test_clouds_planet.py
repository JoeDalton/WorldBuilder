"""
Unit tests for a planet's clouds as one part (world_builder.clouds.planet):
the rule its two layers must keep, and the volume and the sheet driven as
one. The layers themselves: test_clouds_sphere.py and test_clouds_cirrus.py.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from world_builder.atmosphere import AtmosphereParams, Atmospheres
from world_builder.clouds.cirrus import CirrusSpec, CloudSheet, cirrus_stamps
from world_builder.clouds.cloud import CloudSettings, CloudType
from world_builder.clouds.planet import PlanetClouds, check_layers
from world_builder.clouds.sphere import CloudVolume
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS as PLANETS
from world_builder.terrain.workers import ChunkExecutor

R = 600e3


@pytest.mark.parametrize(
    "cloud_type,altitude,cirrus,air_top,message",
    [
        (CloudType.CUMULUS, 2200.0, CirrusSpec(base=3200.0), None, "above the clouds"),
        # The type's own slab: cumulonimbus towers to 3.2 km from 800 m
        (CloudType.CUMULONIMBUS, None, CirrusSpec(base=3000.0), None, "above"),
        (CloudType.CUMULUS, 2200.0, CirrusSpec(), 8500.0, "inside the air"),
    ],
)
def test_a_sheet_must_lie_above_the_volume_and_inside_the_air(
    cloud_type, altitude, cirrus, air_top, message
):
    with pytest.raises(ValueError, match=message):
        check_layers(cloud_type, altitude, cirrus, air_top)


def test_layers_that_fit_pass():
    check_layers(CloudType.CUMULUS, 2200.0, CirrusSpec(), 60e3)
    check_layers(CloudType.CUMULUS, 2200.0, None, 1000.0)  # nothing to check


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
        settings=CloudSettings(shells=2),
        planet=PLANETS["forest"],
        **kwargs,
    )


def test_the_air_bounds_the_sheet_before_anything_is_built(showbase, executor):
    atmospheres = Atmospheres()
    index = atmospheres.add(R, AtmosphereParams(height=8500.0))
    before = showbase.render.get_num_children()
    with pytest.raises(ValueError, match="inside the air"):
        _clouds(
            showbase,
            executor,
            atmospheres=atmospheres,
            atmosphere=index,
            cirrus=CirrusSpec(),
        )
    assert showbase.render.get_num_children() == before


def test_no_cirrus_is_the_volume_alone(showbase, executor):
    clouds = _clouds(showbase, executor)
    try:
        assert isinstance(clouds.volume, CloudVolume)
        assert clouds.sheet is None
        assert clouds.volume.root.get_parent() == clouds.root
    finally:
        clouds.remove()


def test_both_layers_are_driven_as_one(showbase, executor):
    clouds = _clouds(showbase, executor, cirrus=CirrusSpec())
    try:
        volume, sheet = clouds.volume, clouds.sheet
        assert isinstance(sheet, CloudSheet)
        assert volume.root.get_parent() == clouds.root
        assert sheet.root.get_parent() == clouds.root

        # One sun, one sky light
        clouds.set_sun((0.0, 0.0, 2.0))
        for layer in (volume, sheet):
            assert tuple(layer._sun[0]) == (0.0, 0.0, 1.0)
        clouds.set_sky_light(0.7)
        assert volume._sky_light == sheet._sky_light == 0.7

        # The volume may be at half resolution, the sheet never is (its
        # atlas is mipmapped: it takes no pixel angle)
        clouds.set_pixel_angle(0.001, volume_scale=2.0)
        assert volume._pixel_angle == pytest.approx(0.002)
        half_res = MagicMock()
        clouds.use_half_res(half_res)
        half_res.add.assert_called_once_with(volume.root)
        half_res.exclude.assert_called_once_with(sheet.root)

        # One settings section
        reflected = CloudSettings(shells=2, reflected=False)
        clouds.set_settings(reflected)
        assert clouds.settings is reflected
        assert volume.settings is reflected and sheet.settings is reflected

        # The sheet's set-up counts as pending until it is in; both are
        # placed at the planet's centre
        camera = TangentFrame.at(20.0, 40.0).up * (R + 1000.0)
        for _ in range(4):
            clouds.update(camera)
        assert sheet.stats["pending"] == 0
        assert clouds.stats["pending"] == volume.stats["pending"]
        assert set(clouds.stats) == set(volume.stats)
        np.testing.assert_allclose(
            sheet.root.get_pos(showbase.render),
            volume.deck.get_pos(showbase.render),
            atol=1.0,
        )
    finally:
        clouds.remove()
    assert clouds.root.is_empty()


def test_the_layers_share_one_weather(showbase, executor):
    clouds = _clouds(showbase, executor, cirrus=CirrusSpec(), weather_seed=5)
    try:
        assert clouds.weather.seed == 5
        assert clouds.weather.planet == PLANETS["forest"]
        camera = TangentFrame.at(20.0, 40.0).up * (R + 1000.0)
        clouds.update(camera)
        # The sheet's stamps follow the same weather the volume's do
        assert clouds.sheet.stamps is cirrus_stamps(CirrusSpec(), clouds.weather, R)
    finally:
        clouds.remove()
