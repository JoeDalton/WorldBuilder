"""The weather's cloud cover: clear skies and decks, not a thin cover
everywhere (the background's contrast, weather.contrast), its tunables
(WeatherSpec.cloud_amount, cloud_contrast, cirrus_amount), clear
subtropics (synoptic.rules.zonal_cover), against observed cloud amounts on
the Earth-scale planet (scripts/weather_report.py's bands)."""

from dataclasses import replace

import numpy as np
import pytest

from world_builder.clouds import weather
from world_builder.clouds.synoptic import Scales
from world_builder.clouds.synoptic.rules import zonal_cover
from world_builder.clouds.weather import (
    SYNOPTIC_CHANNELS,
    WeatherSpec,
    cirrus_cover_map,
    contrast,
    sample_cube,
    synoptic_cube,
)
from world_builder.terrain.chart import latlon
from world_builder.world_file import load_planet

_DIRS = np.random.default_rng(1).normal(size=(20000, 3))
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)
_LAT, _ = latlon(_DIRS)


def _drawn(spec):
    """The cumulus and the stratus genus's union (scripts/weather_report.py's
    drawn_cover)."""
    at = sample_cube(synoptic_cube(spec, 64), _DIRS)
    channel = SYNOPTIC_CHANNELS.index
    stratus = np.maximum.reduce(
        [at[:, channel(n)] for n in ("stratus", "stratocumulus", "nimbostratus")]
    )
    return 1.0 - (1.0 - at[:, channel("convective")]) * (1.0 - stratus)


@pytest.fixture(scope="module")
def earth():
    return WeatherSpec(load_planet("earth"))


def test_the_contrast_is_a_smooth_step_about_its_edge():
    x = np.linspace(0.0, 1.0, 101)
    y = contrast(x, 0.3, 0.1)
    assert y[x < 0.199].max() == 0.0 and y[x > 0.401].min() == 1.0
    assert contrast(np.array([0.3]), 0.3, 0.1)[0] == pytest.approx(0.5)
    assert np.all(np.diff(y) >= 0.0)
    assert list(contrast(np.array([0.29, 0.31]), 0.3, 0.0)) == [0.0, 1.0]


def test_the_subtropics_are_the_backgrounds_minimum():
    scales = Scales.from_planet(load_planet("earth"))
    lat = np.linspace(0.0, 90.0, 181)
    cover = zonal_cover(lat, scales)
    sub = (lat > 20.0) & (lat < 35.0)
    assert cover[sub].min() < 0.2
    assert cover[sub].min() < 0.5 * cover[lat < 5].max()
    assert cover[sub].min() < 0.5 * cover[(lat > 45) & (lat < 60)].max()


def test_earths_cover_is_clear_or_cloudy(earth):
    """The report's bands at one sample: a mean under the observed total
    (the high and middle clouds are not drawn), a U-shaped distribution,
    clear subtropics."""
    cover = _drawn(earth)
    sub = (np.abs(_LAT) > 20.0) & (np.abs(_LAT) < 35.0)
    assert 0.35 <= cover.mean() <= 0.55
    assert (cover < 0.1).mean() >= 0.30
    assert (cover > 0.7).mean() >= 0.20
    assert (cover[sub] < 0.1).mean() >= 0.50


def test_the_cloud_amount_moves_the_cover(earth):
    means = [_drawn(replace(earth, cloud_amount=a)).mean() for a in (-0.1, 0.0, 0.1)]
    assert means[0] < means[1] < means[2]
    assert means[2] - means[0] > 0.1


def test_a_wider_contrast_softens_the_cover(earth):
    sharp = _drawn(earth)
    soft = _drawn(replace(earth, cloud_contrast=0.4))
    middle = lambda c: ((c > 0.1) & (c < 0.7)).mean()  # noqa: E731
    assert middle(soft) > middle(sharp)


def test_the_cirrus_is_patches_not_a_veil(earth):
    cover = sample_cube(cirrus_cover_map(earth), _DIRS)
    assert (cover < 0.1).mean() > 0.5
    assert weather.CIRRUS_FLOOR < 0.05
    less = sample_cube(cirrus_cover_map(replace(earth, cirrus_amount=-0.1)), _DIRS)
    assert less.mean() < cover.mean()
