"""
Unit tests for a planet's high cirrus (world_builder.clouds.cirrus, and its
upper wind and cover in world_builder.clouds.weather). The shaders need a
window (see scripts/terrain_shots.py: the cirrus_* poses); these check the
density's calibration, the fibres' direction, the weather it follows and
the draw order its nodes are given.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from panda3d.core import DepthWriteAttrib

from world_builder.clouds.cirrus import (
    BEHIND_DRAW_ORDER,
    FRONT_BIN,
    CirrusSpec,
    CloudSheet,
    along,
    calibrate,
    cirrus_density,
    cirrus_weather,
    threshold_for,
    value_noise,
)
from world_builder.clouds.cloud import CloudSettings
from world_builder.clouds.weather import (
    WeatherSpec,
    cirrus_cover_map,
    sample_cube,
    stream_function,
    stream_table,
    stream_vortices,
    upper_wind,
    vortices,
)
from world_builder.sky.sky import SKY_BIN
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.chart import TangentFrame, dirs_from_latlon, latlon
from world_builder.terrain.planet import PRESETS as PLANETS
from world_builder.terrain.workers import ChunkExecutor

R = 600e3
SPEC = CirrusSpec()


@pytest.fixture(scope="module")
def weather():
    return WeatherSpec(PLANETS["earthlike"])


@pytest.fixture(scope="module")
def data(weather):
    return cirrus_weather(SPEC, weather)


def _sphere_points(count, seed=1):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(count, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs, dirs * (R + SPEC.mid)


# ── The noise ──────────────────────────────────────────────────────────────────


def test_value_noise_is_smooth_bounded_and_stationary():
    rng = np.random.default_rng(3)
    points = rng.uniform(-1e4, 1e4, (20000, 3))
    values = value_noise(points)
    assert values.min() >= 0.0 and values.max() < 1.0
    assert abs(values.mean() - 0.5) < 0.01
    np.testing.assert_array_equal(values, value_noise(points))  # deterministic
    # Continuous across lattice cells: a step of 1e-4 of a cell moves it by
    # far less than a crease would.
    step = value_noise(points + 1e-4)
    assert np.abs(step - values).max() < 1e-3


def test_the_fibres_coordinates_leave_x_to_the_stream_function():
    points = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 7.0]])
    slow = along(points)
    assert np.all(slow[:, 0] == 0.0)
    # Blind only along (1, -0.7, -0.7), never an east-west direction.
    np.testing.assert_allclose(along(np.array([[1.0, -0.7, -0.7]])), 0.0, atol=1e-12)
    east = np.array([[-math.sin(0.3), math.cos(0.3), 0.0]])
    assert np.linalg.norm(along(east)) > 0.5


# ── Coverage ───────────────────────────────────────────────────────────────────


def test_the_threshold_falls_with_the_coverage():
    calibration = calibrate(SPEC)
    table = calibration.thresholds
    assert np.all(np.diff(table) < 0.0)
    assert calibration.edge > 0.0
    assert 0.0 < calibration.mean_density < 1.0
    assert abs(calibration.octave_mean - 0.5) < 0.01


@pytest.mark.parametrize("band", [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)])
def test_the_sheet_holds_the_cover_the_map_asks_for(data, band):
    """The share of the sheet with cirrus is the map's cover, within 3 %,
    measured on the density the shader draws."""
    dirs, points = _sphere_points(200_000)
    cover = sample_cube(data.cube, dirs)[:, 0]
    inside = (cover >= band[0]) & (cover < band[1])
    assert inside.sum() > 2000
    density = cirrus_density(SPEC, data, points[inside])
    assert abs((density > 0.0).mean() - cover[inside].mean()) < 0.03


def test_no_cover_is_clear_sky_and_full_cover_is_overcast(data):
    _, points = _sphere_points(5000, seed=2)
    calibration = data.calibration
    for coverage, expected in ((0.0, 0.0), (1.0, 1.0)):
        cube = data.cube.copy()
        cube[..., 0] = coverage
        cube[..., 1] = threshold_for(coverage, calibration)
        density = cirrus_density(SPEC, replace(data, cube=cube), points)
        assert ((density > 0.0).mean()) == pytest.approx(expected, abs=0.005)


def test_amount_scales_the_cover(weather):
    half = cirrus_weather(CirrusSpec(amount=0.5), weather)
    full = cirrus_cover_map(weather)
    np.testing.assert_allclose(half.cube[..., 0], np.clip(0.5 * full, 0, 1), atol=1e-6)


# ── The upper wind ─────────────────────────────────────────────────────────────


def test_the_zonal_stream_function_is_the_wind(weather):
    """psi = -R * integral(u dlat): its slope along a meridian is the
    eastward wind (the fibres run east-west, as fast as the jets are)."""
    table = stream_table(weather)
    lat = np.linspace(-90.0, 90.0, len(table))
    slope = -np.gradient(table, np.radians(lat)) / R
    inner = slice(10, -10)
    np.testing.assert_allclose(slope[inner], upper_wind(weather, lat)[inner], atol=0.02)
    # Westerly jets in both hemispheres at 30 degrees
    assert upper_wind(weather, np.array([30.0, -30.0])) == pytest.approx([1.0, 1.0])


def test_zonal_fibres_run_east_west(weather):
    """Without weather systems, the stream function is constant along a
    parallel: the fibres follow it."""
    table = stream_table(weather)
    none = np.zeros((0, 5))
    lat = np.full(36, 45.0)
    lon = np.linspace(-180.0, 170.0, 36)
    psi = stream_function(table, none, dirs_from_latlon(lat, lon))
    np.testing.assert_allclose(psi, psi[0], rtol=1e-9)


def test_the_fibres_turn_around_the_weather_systems(weather):
    """Each system's share is constant on circles around it (the wind turns
    around a low), a low's a minimum in the north, and the strongest wind it
    adds is the configured share of the jets'."""
    rows = stream_vortices(weather)
    systems = vortices(weather)
    assert len(rows) == len(systems) > 0
    for row, v in zip(rows, systems):
        centre, k, amplitude = row[:3], row[3], row[4]
        frame = TangentFrame.at(*(float(x[0]) for x in latlon(centre[None, :])))
        ring = np.array(
            [
                math.cos(0.5 / math.sqrt(k)) * frame.up
                + math.sin(0.5 / math.sqrt(k))
                * (math.cos(a) * frame.east + math.sin(a) * frame.north)
                for a in np.linspace(0.0, 2.0 * math.pi, 12)
            ]
        )
        psi = stream_function(np.zeros(4), row[None, :], ring)
        np.testing.assert_allclose(psi, psi[0], rtol=1e-6)
        if not v.tropical:
            assert (amplitude < 0.0) == (centre[2] > 0.0)
        # Its fastest wind: the steepest slope across the rings
        r = np.linspace(0.01, 3.0, 400) / math.sqrt(k)
        profile = amplitude * np.exp(-((2.0 * np.sin(r / 2.0)) ** 2) * k)
        speed = np.abs(np.gradient(profile, r * R)).max()
        assert speed == pytest.approx(weather.jet_vortex, rel=0.02)


def test_the_cirrus_cover_follows_the_jets_and_the_cyclones(weather):
    cover = cirrus_cover_map(weather)
    assert cover.shape == (6, 128, 128)
    assert 0.0 <= cover.min() and cover.max() <= 1.0
    np.testing.assert_array_equal(cover, cirrus_cover_map(weather))  # cached
    dirs = cube_directions(128).reshape(-1, 3)
    lat, _ = latlon(dirs)
    flat = cover.reshape(-1)
    # The wet tropics' anvils and the subtropical jet against the dry
    # air between them
    tropics = flat[np.abs(lat) < 5].mean()
    between = flat[(np.abs(lat) > 18) & (np.abs(lat) < 22)].mean()
    assert tropics > between + 0.1
    # A shield poleward and east of the extratropical lows, against their
    # cold sector behind
    ahead, behind = [], []
    for v in vortices(weather):
        if v.tropical:
            continue
        frame = TangentFrame.at(*(float(x[0]) for x in latlon(np.array([v.centre]))))
        poleward = frame.north if v.centre[2] > 0 else -frame.north
        ahead.append(frame.up + v.radius * (0.9 * frame.east + 1.0 * poleward))
        behind.append(frame.up + v.radius * (-1.5 * frame.east - 0.9 * poleward))
    assert len(ahead) >= 3
    assert (
        sample_cube(cover, np.array(ahead)).mean()
        > sample_cube(cover, np.array(behind)).mean() + 0.2
    )


# ── The scene ──────────────────────────────────────────────────────────────────


def test_the_sheet_draws_behind_and_in_front_of_the_clouds(showbase, weather):
    executor = ChunkExecutor(backend="inline")
    cirrus = CloudSheet(showbase.render, R, executor, planet=PLANETS["earthlike"])
    try:
        behind, front = cirrus.behind, cirrus.front
        assert behind.get_bin_name() == SKY_BIN
        assert behind.get_bin_draw_order() == BEHIND_DRAW_ORDER
        assert front.get_bin_name() == FRONT_BIN
        write = cirrus.root.get_attrib(DepthWriteAttrib)
        assert write.get_mode() == DepthWriteAttrib.M_off
        # Hidden until the set-up is in, then placed at the planet's centre
        assert cirrus.root.is_stashed()
        camera = TangentFrame.at(20.0, 40.0).up * (R + 1000.0)
        cirrus.update(camera)
        assert not cirrus.root.is_stashed()
        assert cirrus.stats["pending"] == 0
        np.testing.assert_allclose(
            cirrus.root.get_pos(showbase.render), -camera, atol=1.0
        )
        assert len(cirrus.data.vortices) > 0
        assert cirrus.root.get_shader_input("cirrusMap").get_texture() is not None
        # Out of the sea's mirror when the settings say so
        cirrus.set_settings(CloudSettings(reflected=False))
        from world_builder.depth import REFLECTION_CAMERA_MASK

        assert cirrus.root.is_hidden(REFLECTION_CAMERA_MASK)
    finally:
        cirrus.remove()
        executor.shutdown()
