"""Jets, synoptic systems on closed-form tracks, pressure and wind
(docs/literature/synoptic-weather-review.tex, sections 11.2-11.5)."""

import math
from dataclasses import replace

import numpy as np
import pytest
from planets import PRESETS

from world_builder.clouds.synoptic import Scales
from world_builder.clouds.synoptic.fields import pressure, wind
from world_builder.clouds.synoptic.jets import jet_lat, jet_waves, troughs
from world_builder.clouds.synoptic.systems import (
    PRESSURE_SIGN,
    Kind,
    SynopticObject,
    active,
    population,
    position,
    spacing_radius,
    stage,
    storm_sectors,
)
from world_builder.clouds.weather import WeatherSpec, _sampler
from world_builder.terrain.planet import Planet, PlanetSpec

EARTH = PlanetSpec("e", radius=6.371e6, day_length_h=23.934)
DAY = 86400.0


def _dir(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _gc(a, b, radius):
    return radius * np.arccos(np.clip(np.sum(a * b, -1), -1.0, 1.0))


@pytest.fixture(scope="module")
def earth():
    spec = WeatherSpec(EARTH)
    scales = Scales.from_planet(EARTH)
    return spec, scales, jet_waves(scales, spec.seed), Planet(EARTH)


@pytest.fixture(scope="module")
def kerbin():
    planet_spec = PRESETS["earthlike"]
    spec = WeatherSpec(planet_spec)
    scales = Scales.from_planet(planet_spec)
    return spec, scales, jet_waves(scales, spec.seed), Planet(planet_spec)


def _eddy(jets, scales):
    return [j for j in jets if j.lat0 > scales.hadley_edge + 0.1]


def test_jet_meanders(earth):
    _, scales, jets, _ = earth
    lon = np.linspace(-180.0, 180.0, 721)
    a_tot = min(8.0, 0.25 * math.degrees(scales.jet_spacing / EARTH.radius))
    for jet in _eddy(jets, scales):
        lat = jet_lat(jet, lon, 0.0)
        assert np.all(np.sign(lat) == jet.hemisphere)
        assert 3.0 < np.ptp(lat) <= 2.0 * a_tot + 1e-9


def test_short_wave_troughs_move_east(earth):
    _, scales, jets, _ = earth
    for jet in _eddy(jets, scales):
        m = int(np.argmax(jet.k))
        assert jet.omega[m] > 0.0
        single = replace(
            jet,
            k=jet.k[m : m + 1],
            amp=jet.amp[m : m + 1],
            omega=jet.omega[m : m + 1],
            phase=jet.phase[m : m + 1],
        )
        hour = 3600.0  # well under a trough spacing: no aliasing
        before, after = troughs(single, 0.0), troughs(single, hour)
        shift = (after[:, None] - before[None, :] + 180.0) % 360.0 - 180.0
        moved = shift[np.arange(len(after)), np.argmin(np.abs(shift), axis=1)]
        expected = math.degrees(jet.omega[m] * hour / jet.k[m])
        assert np.allclose(moved, expected, rtol=0.05, atol=0.05)


def test_track_continuous(earth):
    spec, scales, jets, planet = earth
    objects = population(spec, scales, jets, planet, 3.0 * DAY)
    assert objects
    limit = 60.0 * 3.0 * scales.steering
    for obj in objects:
        a = _dir(*position(obj, 3.0 * DAY))
        b = _dir(*position(obj, 3.0 * DAY + 60.0))
        assert _gc(a, b, EARTH.radius) < limit


def _lows_per_hemisphere(objects):
    lows = [o for o in objects if o.kind is Kind.LOW]
    north = sum(1 for o in lows if o.lat0 > 0.0)
    return north, len(lows) - north


@pytest.mark.parametrize("which", ["earth", "kerbin"])
def test_population_steady(which, request):
    spec, scales, jets, planet = request.getfixturevalue(which)
    # The band's own wavenumber (k* is the 45 degree one).
    target = sum(
        storm_sectors(scales, j.lat0) for j in _eddy(jets, scales) if j.hemisphere == 1
    )
    lifetime = 2.5 * scales.wavelength / scales.steering
    times = list(np.linspace(0.0, 100.0 * lifetime, 20)) + [1e9]
    for t in times:
        for count in _lows_per_hemisphere(population(spec, scales, jets, planet, t)):
            assert 0.5 * target <= count <= 1.5 * target, (t, count, target)


def test_lows_downstream_of_troughs(earth):
    spec, scales, jets, planet = earth
    radius = EARTH.radius
    hits = total = 0
    for t in np.linspace(0.0, 20.0 * DAY, 10):
        for obj in population(spec, scales, jets, planet, t):
            if obj.kind is not Kind.LOW or stage(obj, t) >= 0.5:
                continue
            lat, lon = position(obj, t)
            band = [j for j in _eddy(jets, scales) if j.hemisphere == np.sign(lat)]
            jet = min(band, key=lambda j: abs(abs(lat) - j.lat0))
            west = (lon - troughs(jet, t)) % 360.0
            metres = np.radians(west) * radius * math.cos(math.radians(lat))
            total += 1
            hits += bool(np.any(metres <= 0.5 * scales.wavelength))
    assert total > 10
    assert hits >= 0.7 * total


def test_tc_over_warm_sea_only(kerbin):
    spec, scales, jets, planet = kerbin
    sample, _ = _sampler(spec, planet)
    seen = 0
    for t in np.linspace(0.0, 200.0 * DAY, 40):
        for obj in population(spec, scales, jets, planet, t):
            if obj.kind is not Kind.TROPICAL:
                continue
            seen += 1
            elevation, _, _ = sample(_dir(obj.lat0, obj.lon0)[None])
            assert elevation[0] < -200.0
            assert scales.tc_band[0] <= abs(obj.lat0) <= scales.tc_band[1]
    assert seen > 0


def test_no_tc_without_ocean():
    planet_spec = PRESETS["desert"]
    spec = WeatherSpec(planet_spec)
    scales = Scales.from_planet(planet_spec)
    jets = jet_waves(scales, spec.seed)
    planet = Planet(planet_spec)
    for t in np.linspace(0.0, 50.0 * DAY, 10):
        objects = population(spec, scales, jets, planet, t)
        assert not [o for o in objects if o.kind is Kind.TROPICAL]


@pytest.mark.parametrize("hemisphere", [1, -1])
def test_circulation_sense(earth, hemisphere):
    _, scales, _, _ = earth
    low = SynopticObject(
        kind=Kind.LOW,
        birth=-0.6 * DAY * 5,
        lifetime=DAY * 5,
        lat0=45.0 * hemisphere,
        lon0=10.0,
        radius=scales.cyclone_radius,
        strength=25.0,
        variant="norwegian",
        heading=0.0,
        seed=1,
    )
    centre = _dir(45.0 * hemisphere, 10.0)
    ring_angle = 0.5 * scales.cyclone_radius / EARTH.radius
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, centre)
    east /= np.linalg.norm(east)
    north = np.cross(centre, east)
    az = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    offsets = np.cos(az)[:, None] * east + np.sin(az)[:, None] * north
    ring = np.cos(ring_angle) * centre + np.sin(ring_angle) * offsets
    uv = wind(ring, [low], scales, 0.0, surface=True)
    tangent_ccw = np.stack([-np.sin(az), np.cos(az)], -1)  # (east, north)
    circulation = np.sum(np.sum(uv * tangent_ccw, -1))
    assert circulation * hemisphere > 0.0
    # Toward low pressure, across the isobars by the friction angle: the
    # angle between the wind and the isobar (perpendicular to grad p).
    step = 1e-4
    grads = []
    for d in ring:
        e, n = east, north  # near enough at the ring for a gradient's sign
        probes = np.stack([d + step * e, d - step * e, d + step * n, d - step * n])
        probes /= np.linalg.norm(probes, axis=1, keepdims=True)
        pe, pw, pn, ps = pressure(probes, [low], scales, 0.0)
        grads.append([pe - pw, pn - ps])
    down = -np.array(grads)
    down /= np.linalg.norm(down, axis=-1, keepdims=True)
    across = np.degrees(np.arcsin(np.sum(uv * down, -1) / np.linalg.norm(uv, axis=-1)))
    assert np.all((10.0 <= across) & (across <= 40.0)), across


def test_pressure_low_and_high(earth):
    _, scales, _, _ = earth
    common = dict(
        birth=-3.0 * DAY, lifetime=5.0 * DAY, lon0=0.0, variant="", heading=0.0, seed=1
    )
    low = SynopticObject(
        Kind.LOW, lat0=50.0, radius=scales.cyclone_radius, strength=25.0, **common
    )
    high = SynopticObject(
        Kind.HIGH, lat0=-50.0, radius=scales.cyclone_radius, strength=15.0, **common
    )
    p = pressure(
        np.stack([_dir(50.0, 0.0), _dir(-50.0, 0.0)]), [low, high], scales, 0.0
    )
    base = pressure(np.stack([_dir(50.0, 0.0), _dir(-50.0, 0.0)]), [], scales, 0.0)
    assert p[0] - base[0] < -15.0 and p[1] - base[1] > 8.0


def test_spacing(earth):
    spec, scales, jets, planet = earth
    for t in np.linspace(0.0, 30.0 * DAY, 6):
        objects = [o for o in population(spec, scales, jets, planet, t) if active(o, t)]
        centres = np.array([_dir(*position(o, t)) for o in objects])
        radii = np.array([spacing_radius(o, t) for o in objects])
        signs = np.array([PRESSURE_SIGN[o.kind] for o in objects])
        gc = _gc(centres[:, None], centres[None, :], EARTH.radius)
        same = signs[:, None] == signs[None, :]
        need = np.where(
            same,
            0.75 * (radii[:, None] + radii[None, :]),
            0.75 * np.minimum(radii[:, None], radii[None, :]),
        )
        np.fill_diagonal(gc, np.inf)
        assert np.all(gc >= need - 1.0)


def test_weather_spec_has_a_time():
    spec = WeatherSpec(EARTH)
    assert spec.time_s == 0.0
    assert replace(spec, time_s=3.0 * DAY).time_s == 3.0 * DAY


def test_upper_wind_on_a_tilted_axis_is_the_jet_speed(earth):
    """Across a meandering axis the profile is measured perpendicular to
    it: the wind on the axis is the jet's speed wherever it tilts."""
    from world_builder.clouds.synoptic.jets import upper_stream

    _, scales, jets, _ = earth
    jet = _eddy(jets, scales)[0]
    lon = np.linspace(-180.0, 180.0, 37)
    lat = jet_lat(jet, lon, 0.0)
    on_axis = _dir(lat, lon)
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, on_axis)
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(on_axis, east)
    step = 1000.0 / EARTH.radius
    probes = np.concatenate(
        [
            on_axis + step * east,
            on_axis - step * east,
            on_axis + step * north,
            on_axis - step * north,
        ]
    )
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    psi = upper_stream([jet], [], 0.0, probes).reshape(4, -1)
    grad = np.stack([psi[0] - psi[1], psi[2] - psi[3]], -1) / 2000.0
    speed = np.linalg.norm(grad, axis=-1)
    np.testing.assert_allclose(speed, jet.speed / 30.0, rtol=0.03)


@pytest.mark.parametrize("which", ["earth", "kerbin"])
def test_jets_do_not_overlap(which, request):
    """A jet is narrower than the gap to its neighbour: on each axis the
    other jets add little to its speed."""
    _, scales, jets, _ = request.getfixturevalue(which)
    north = [j for j in jets if j.hemisphere == 1]
    for jet in north:
        for other in north:
            if other is jet:
                continue
            gap = scales.radius * math.radians(abs(jet.lat0 - other.lat0))
            share = other.speed / jet.speed * math.exp(-((gap / other.width) ** 2))
            assert share < 0.05, (jet.lat0, other.lat0)
