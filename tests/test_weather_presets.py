"""Local weather presets pinned at coordinates, the global weather arranged
around them (docs/literature/synoptic-weather-review.tex, "Local presets")."""

import math
from dataclasses import replace

import numpy as np
import pytest
from planets import PRESETS

from world_builder.clouds.synoptic import Scales
from world_builder.clouds.synoptic.jets import troughs
from world_builder.clouds.synoptic.presets import (
    PRESET_GENUS,
    Preset,
    WeatherPin,
    dominant_genus,
    solve,
)
from world_builder.clouds.synoptic.rules import instability
from world_builder.clouds.synoptic.state import synoptic_state
from world_builder.clouds.synoptic.systems import Kind, position
from world_builder.clouds.weather import (
    WeatherSpec,
    _climate_direct,
    _sampler,
    genus_fields,
)
from world_builder.terrain.chart import dirs_from_latlon
from world_builder.terrain.planet import Planet

PLANET = PRESETS["earthlike"]
SCALES = Scales.from_planet(PLANET)
LOW_LAT = 0.5 * (SCALES.storm_bands[0][0] + SCALES.storm_bands[0][1])


@pytest.fixture(scope="module")
def planet():
    return Planet(PLANET)


def _search(planet, lat_range, accept):
    """The first (lat, lon) on a grid where *accept(elevation, T, H)*."""
    weather = WeatherSpec(PLANET)
    sample, _ = _sampler(weather, planet)
    lats = np.repeat(np.linspace(*lat_range, 21), 72)
    lons = np.tile(np.linspace(-180.0, 175.0, 72), 21)
    elevation, temperature, humidity = sample(dirs_from_latlon(lats, lons))
    ok = accept(elevation, temperature, humidity)
    assert ok.any()
    i = int(np.argmax(ok))
    return float(lats[i]), float(lons[i])


@pytest.fixture(scope="module")
def spots(planet):
    lo, hi = SCALES.tc_band
    return {
        Preset.FAIR_HIGH: (35.0, 20.0),
        Preset.WARM_FRONT_APPROACH: (LOW_LAT, 10.0),
        Preset.COLD_FRONT: (LOW_LAT, 10.0),
        Preset.SQUALL_LINE: (LOW_LAT, 10.0),
        Preset.OVERCAST_DRIZZLE: (LOW_LAT, 10.0),
        Preset.HURRICANE: _search(
            planet, (lo + 2.0, hi - 2.0), lambda e, t, h: (e < -500.0) & (t >= 26.0)
        ),
        Preset.THUNDERSTORMS: _search(
            planet, (-30.0, 30.0), lambda e, t, h: instability(h, t) > 0.6
        ),
    }


def _at(spec, lat, lon, planet):
    return genus_fields(
        spec, dirs_from_latlon(lat, lon), planet, _climate_direct(spec, planet)
    )


@pytest.mark.parametrize("preset", list(Preset))
def test_each_preset_overhead(preset, spots, planet):
    lat, lon = spots[preset]
    spec = WeatherSpec(PLANET, pins=(WeatherPin(lat, lon, preset),))
    fields = _at(spec, lat, lon, planet)
    expected = PRESET_GENUS[preset]
    got = dominant_genus(fields, 0)
    if preset is Preset.OVERCAST_DRIZZLE:
        assert got in ("stratus", "stratocumulus"), got
    else:
        assert got == expected, (preset, got)
    if preset is Preset.HURRICANE:
        # Its eyewall just round the clear eye
        state = synoptic_state(spec)
        tc = next(o for o in state.systems if o.kind is Kind.TROPICAL)
        ring = 0.1 * tc.radius / PLANET.radius
        frame_lat = np.full(16, lat) + math.degrees(ring) * np.cos(
            np.linspace(0, 2 * np.pi, 16)
        )
        frame_lon = np.full(16, lon) + math.degrees(ring) * np.sin(
            np.linspace(0, 2 * np.pi, 16)
        ) / math.cos(math.radians(lat))
        wall = genus_fields(
            spec,
            dirs_from_latlon(frame_lat, frame_lon),
            planet,
            _climate_direct(spec, planet),
        )
        assert np.median(wall.depth) > 0.8


def test_pin_consistent_neighbourhood(spots, planet):
    lat, lon = spots[Preset.WARM_FRONT_APPROACH]
    spec = WeatherSpec(PLANET, pins=(WeatherPin(lat, lon, Preset.WARM_FRONT_APPROACH),))
    pinned, jets = solve(spec, SCALES, planet)
    low = pinned[0]
    assert low.kind is Kind.LOW
    clat, clon = position(low, spec.time_s)
    jet = min(
        (j for j in jets if not j.subtropical and j.hemisphere == 1),
        key=lambda j: abs(j.lat0 - abs(clat)),
    )
    west = (clon - troughs(jet, spec.time_s)) % 360.0
    metres = np.radians(west) * PLANET.radius * math.cos(math.radians(clat))
    assert np.min(metres) < 0.5 * SCALES.wavelength
    # The free systems keep out of its way
    state = synoptic_state(spec)
    assert state.systems[0] is low or state.systems[0] == low
    centre = dirs_from_latlon(clat, clon)[0]
    for obj in state.systems[1:]:
        if obj.kind in (Kind.LOW, Kind.CUTOFF, Kind.TROPICAL):
            other = dirs_from_latlon(*position(obj, spec.time_s))[0]
            gap = PLANET.radius * math.acos(np.clip(other @ centre, -1, 1))
            assert gap > 0.5 * (obj.radius + low.radius)


def test_infeasible_raises(planet):
    with pytest.raises(ValueError, match="latitude"):
        solve(
            WeatherSpec(PLANET, pins=(WeatherPin(60.0, 0.0, Preset.HURRICANE),)),
            SCALES,
            planet,
        )
    desert = PRESETS["desert"]
    with pytest.raises(ValueError, match="ocean"):
        solve(
            WeatherSpec(desert, pins=(WeatherPin(15.0, 0.0, Preset.HURRICANE),)),
            Scales.from_planet(desert),
            Planet(desert),
        )
    far = (
        SCALES.storm_bands[0][0]
        - math.degrees(SCALES.deformation_radius / PLANET.radius)
        - 5
    )
    with pytest.raises(ValueError, match="storm track"):
        solve(
            WeatherSpec(PLANET, pins=(WeatherPin(far, 0.0, Preset.COLD_FRONT),)),
            SCALES,
            planet,
        )


def test_wrap_and_poles(planet):
    from world_builder.clouds.synoptic.rules import apply_systems

    def overhead(lon):
        spec = WeatherSpec(PLANET, pins=(WeatherPin(LOW_LAT, lon, Preset.COLD_FRONT),))
        pinned, _ = solve(spec, SCALES, planet)
        d = dirs_from_latlon(LOW_LAT, lon)
        out = apply_systems(
            d,
            pinned,
            SCALES,
            spec.time_s,
            base=np.zeros(1),
            sea=np.zeros(1),
            temperature=np.full(1, 10.0),
            elevation=np.zeros(1),
        )
        return np.array(
            [out.convective[0], out.depth[0]] + [v[0] for v in out.layers.values()]
        )

    assert np.allclose(overhead(179.9), overhead(0.0), atol=0.1)
    spec = WeatherSpec(PLANET, pins=(WeatherPin(85.0, 0.0, Preset.FAIR_HIGH),))
    solve(spec, SCALES, planet)
    synoptic_state(spec)


def test_close_pins(planet):
    shift = math.degrees(0.3 * SCALES.deformation_radius / PLANET.radius)
    spec = WeatherSpec(
        PLANET,
        pins=(
            WeatherPin(LOW_LAT, 10.0, Preset.COLD_FRONT),
            WeatherPin(
                LOW_LAT,
                10.0 + shift / math.cos(math.radians(LOW_LAT)),
                Preset.SQUALL_LINE,
            ),
        ),
    )
    pinned, _ = solve(spec, SCALES, planet)
    assert sum(o.kind is Kind.LOW for o in pinned) == 1
    lat, lon = 15.0, 0.0
    step = math.degrees(0.5 * SCALES.tc_radius / PLANET.radius)
    two = WeatherSpec(
        PLANET,
        pins=(
            WeatherPin(lat, lon, Preset.HURRICANE),
            WeatherPin(lat + step, lon, Preset.HURRICANE),
        ),
    )
    with pytest.raises(ValueError):
        solve(two, SCALES, planet)


def test_pins_hold_at_time(spots, planet):
    lat, lon = spots[Preset.COLD_FRONT]
    for t in (0.0, 1e6, 3.7e7):
        spec = WeatherSpec(
            PLANET, time_s=t, pins=(WeatherPin(lat, lon, Preset.COLD_FRONT),)
        )
        assert dominant_genus(_at(spec, lat, lon, planet), 0) == "cumulonimbus", t
        pinned, _ = solve(spec, SCALES, planet)
        assert pinned[0].birth <= t <= pinned[0].birth + pinned[0].lifetime


def test_unpinned_weather_unchanged():
    spec = WeatherSpec(PLANET)
    assert replace(spec, pins=()) == spec


def test_free_systems_keep_clear_of_pinned_ones(planet):
    """Any free system, high or low, keeps the full spacing from a pinned
    one: what the pin asks for is what is seen."""
    spec = WeatherSpec(PLANET, pins=(WeatherPin(LOW_LAT, 10.0, Preset.COLD_FRONT),))
    state = synoptic_state(spec)
    low = state.systems[0]
    centre = dirs_from_latlon(*position(low, spec.time_s))[0]
    from world_builder.clouds.synoptic.systems import spacing_radius

    for obj in state.systems[1:]:
        other = dirs_from_latlon(*position(obj, spec.time_s))[0]
        gap = PLANET.radius * math.acos(np.clip(other @ centre, -1, 1))
        need = 0.75 * (
            spacing_radius(obj, spec.time_s) + spacing_radius(low, spec.time_s)
        )
        assert gap >= need - 1.0, (obj.kind, gap, need)
