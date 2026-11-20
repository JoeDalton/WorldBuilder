"""The weather preview (terrain.preview --weather) and the pins' command
line parsing."""

import numpy as np
import pytest

from world_builder.clouds.synoptic import Scales
from world_builder.clouds.synoptic.presets import Preset, WeatherPin, parse_pin
from world_builder.terrain.planet import PRESETS, Planet

H = 72


@pytest.fixture(scope="module")
def planet():
    return Planet(PRESETS["earthlike"])


@pytest.fixture(scope="module")
def sheet(planet):
    from world_builder.terrain.preview import render_weather

    return render_weather(planet, H, seed=11)


def _panel(sheet, k):
    """The k-th of the three panels (each 2H wide, 4 px gutters)."""
    x = k * (2 * H + 4)
    return sheet[:, x : x + 2 * H].astype(np.int32)


def test_three_panels(sheet):
    assert sheet.dtype == np.uint8
    assert sheet.shape == (H, 3 * 2 * H + 2 * 4, 3)


def test_the_chart_draws_isobars_fronts_and_jets(sheet):
    chart = _panel(sheet, 2)
    r, g, b = chart[..., 0], chart[..., 1], chart[..., 2]
    isobar = (r < 70) & (g < 70) & (b < 70)
    cold = (b > 180) & (r < 90) & (g < 140)
    warm = (r > 180) & (g < 80) & (b < 80)
    jet = (r > 200) & (g > 160) & (b < 80)
    assert isobar.mean() > 0.01
    assert cold.sum() > 10 and warm.sum() > 10
    assert jet.sum() > 2 * H


def test_pins_change_the_weather(planet, sheet):
    from world_builder.terrain.preview import render_weather

    lat = 0.5 * sum(Scales.from_planet(planet.spec).storm_bands[0])
    pinned = render_weather(
        planet, H, seed=11, pins=(WeatherPin(lat, 10.0, Preset.COLD_FRONT),)
    )
    assert pinned.shape == sheet.shape
    assert np.any(pinned != sheet)


def test_parse_pin():
    pin = parse_pin("62", "10.5", "Cold_Front")
    assert pin == WeatherPin(62.0, 10.5, Preset.COLD_FRONT)
    with pytest.raises(ValueError, match="warm_front_approach"):
        parse_pin("62", "10", "blizzard")
