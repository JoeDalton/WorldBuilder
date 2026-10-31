"""
Unit tests for a planet's high cirrus (world_builder.clouds.cirrus, and its
upper wind and cover in world_builder.clouds.weather). The shaders need a
window (see scripts/terrain_shots.py: the cirrus_* poses; and
scripts/cirrus_preview.py for the stamps' look); these check the baked atlas,
the stamps' placement against the weather, their geometry, and the draw order
their nodes are given.
"""

import math

import numpy as np
import pytest
from panda3d.core import DepthWriteAttrib, SamplerState

from world_builder.clouds.cirrus import (
    ATLAS_GRID,
    BEHIND_DRAW_ORDER,
    FOOTPRINT_THRESHOLD,
    FRONT_BIN,
    MAX_COVER,
    STAMP_PIXELS,
    STAMP_QUADS,
    CirrusSpec,
    CloudSheet,
    atlas_array,
    atlas_records,
    cirrus_stamps,
    load_cirrus_atlas,
    stamp_cover,
    stamp_thickness,
    stamp_vertices,
    upper_wind_at,
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
from world_builder.depth import REFLECTION_CAMERA_MASK
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
def stamps(weather):
    return cirrus_stamps(SPEC, weather, R)


def _sphere_dirs(count, seed=1):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(count, 3))
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


# ── The noise ──────────────────────────────────────────────────────────────────


def test_value_noise_is_smooth_bounded_and_stationary():
    rng = np.random.default_rng(3)
    points = rng.uniform(-1e4, 1e4, (20000, 3))
    values = value_noise(points)
    assert values.min() >= 0.0 and values.max() < 1.0
    assert abs(values.mean() - 0.5) < 0.01
    np.testing.assert_array_equal(values, value_noise(points))  # deterministic
    # Continuous across lattice cells
    step = value_noise(points + 1e-4)
    assert np.abs(step - values).max() < 1e-3


# ── The baked atlas ────────────────────────────────────────────────────────────


def test_the_atlas_holds_its_described_stamps():
    """The JSON describes the PNG: one rect per grid cell, its footprint and
    mean thickness measured from the pixels, and every stamp zero at its
    border (no quad edge ever shows)."""
    atlas = atlas_array()
    records = atlas_records()
    columns, rows = STAMP_PIXELS
    assert atlas.shape == (ATLAS_GRID[1] * rows, ATLAS_GRID[0] * columns)
    assert len(records) == ATLAS_GRID[0] * ATLAS_GRID[1]
    for r in records:
        c0 = round(r["u_min"] * atlas.shape[1])
        r0 = round(r["v_min"] * atlas.shape[0])
        stamp = atlas[r0 : r0 + rows, c0 : c0 + columns]
        assert stamp.max() > 0.5
        assert (stamp > FOOTPRINT_THRESHOLD).mean() == pytest.approx(
            r["footprint"], abs=0.01
        )
        assert stamp.mean() == pytest.approx(r["mean_thickness"], abs=0.01)
        border = np.concatenate([stamp[0], stamp[-1], stamp[:, 0], stamp[:, -1]])
        assert border.max() < FOOTPRINT_THRESHOLD


def test_tufts_lead_downwind():
    """Over the atlas, the tufts (the dense parts) sit toward +x and the
    tails (the thin parts) upwind of them. Some patches scatter tufts
    upwind too, so this holds on average, not stamp by stamp."""
    atlas = atlas_array()
    columns, rows = STAMP_PIXELS
    dense, thin = [], []
    for r in atlas_records():
        c0 = round(r["u_min"] * atlas.shape[1])
        r0 = round(r["v_min"] * atlas.shape[0])
        stamp = atlas[r0 : r0 + rows, c0 : c0 + columns]
        dense.append(np.nonzero(stamp > 0.6)[1].mean() / columns)
        thin.append(np.nonzero((stamp > 0.05) & (stamp < 0.3))[1].mean() / columns)
    assert np.mean(dense) > 0.6
    assert np.mean(dense) > np.mean(thin) + 0.04


def test_the_atlas_texture_is_mipmapped_and_clamped():
    tex = load_cirrus_atlas()
    assert tex.uses_mipmaps()
    assert tex.get_anisotropic_degree() >= 4
    assert tex.get_wrap_u() == SamplerState.WM_clamp


# ── The stamps ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("band", [(0.1, 0.25), (0.25, 0.4), (0.4, 0.6)])
def test_the_stamps_cover_what_the_map_asks_for(weather, stamps, band):
    """The share of the sheet under a stamp is the map's cover, within 5 %:
    a Poisson field at density -ln(1 - c) / A covers c."""
    dirs = _sphere_dirs(120_000)
    cover = np.clip(sample_cube(cirrus_cover_map(weather), dirs), 0.0, MAX_COVER)
    inside = (cover >= band[0]) & (cover < band[1])
    assert inside.sum() > 2000
    covered = stamp_cover(stamps, dirs[inside] * stamps.sheet)
    assert covered.mean() == pytest.approx(cover[inside].mean(), abs=0.05)


def test_the_stamps_are_placed_the_same_every_time(weather, stamps):
    cirrus_stamps.cache_clear()
    again = cirrus_stamps(SPEC, weather, R)
    np.testing.assert_array_equal(again.centres, stamps.centres)
    np.testing.assert_array_equal(again.index, stamps.index)


def test_no_cover_no_stamps(weather):
    assert len(cirrus_stamps(CirrusSpec(amount=0.0), weather, R)) == 0


def test_the_stamps_lie_on_the_sheet_turned_downwind(weather, stamps):
    """Centred on the sheet's sphere, tangent to it; where the jets blow
    their fastest, within 20 degrees of the wind (the tufts downwind)."""
    radius = np.linalg.norm(stamps.centres, axis=1)
    np.testing.assert_allclose(radius, R + SPEC.mid, rtol=1e-9)
    up = stamps.centres / radius[:, None]
    assert np.abs(np.einsum("ij,ij->i", stamps.along, up)).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", stamps.across, up)).max() < 1e-9
    wind = upper_wind_at(SPEC, weather, up, stamps.sheet)
    speed = np.linalg.norm(wind, axis=1)
    fast = speed > 0.9
    assert fast.sum() > 100
    cos = np.einsum("ij,ij->i", stamps.along[fast], wind[fast]) / speed[fast]
    assert np.mean(cos > math.cos(math.radians(20.0))) > 0.95


def test_overlapping_stamps_stay_parallel(stamps):
    """Real streaks side by side never cross at an angle: stamps within a
    stamp's width of each other point the same way, within 20 degrees but
    for a few in a thousand (they once turned each on its own, and crossed
    in more than half the pairs)."""
    from scipy.spatial import cKDTree

    pairs = np.array(sorted(cKDTree(stamps.centres).query_pairs(20000.0)))
    assert len(pairs) > 1000
    cos = np.einsum("ij,ij->i", stamps.along[pairs[:, 0]], stamps.along[pairs[:, 1]])
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    assert np.median(angle) < 5.0
    assert np.mean(angle > 20.0) < 0.005
    assert np.percentile(angle, 99.9) < 30.0


def test_the_wind_blows_east_in_the_westerlies(weather):
    dirs = dirs_from_latlon(np.full(8, 30.0), np.linspace(-180.0, 135.0, 8))
    wind = upper_wind_at(CirrusSpec(meander=0.0), weather, dirs, R)
    frame_east = np.cross([0.0, 0.0, 1.0], dirs)
    frame_east /= np.linalg.norm(frame_east, axis=1, keepdims=True)
    # Far from any system: the jet's own speed, eastward
    eastward = np.einsum("ij,ij->i", wind, frame_east)
    assert np.median(eastward) == pytest.approx(1.0, abs=0.15)


def test_the_stamps_thickness_is_their_atlas_texels(stamps):
    """At a stamp's centre, nothing but its own texel (or more, from its
    neighbours): never less than its own thickness x opacity."""
    i = int(np.argmax(stamps.opacity))
    own = stamp_thickness(stamps, stamps.centres[i : i + 1])[0]
    assert own >= 0.0
    far_away = -stamps.centres[i] * 3.0  # off the sheet altogether
    assert stamp_thickness(stamps, far_away[None, :])[0] == 0.0


def test_the_geometry_lies_on_the_sheet(stamps):
    rows, tris = stamp_vertices(stamps)
    per = (STAMP_QUADS[0] + 1) * (STAMP_QUADS[1] + 1)
    assert rows.shape == (len(stamps) * per, 6)
    assert len(tris) == len(stamps) * STAMP_QUADS[0] * STAMP_QUADS[1] * 6
    assert tris.max() < len(rows)
    np.testing.assert_allclose(
        np.linalg.norm(rows[:, 0:3].astype(np.float64), axis=1),
        stamps.sheet,
        rtol=1e-6,
    )
    # The corners map to the corners of the stamp's rect in the atlas
    records = atlas_records()
    first = records[int(stamps.index[0])]
    u, v = rows[0, 3], rows[0, 4]
    assert u == pytest.approx(first["u_min"]) and v == pytest.approx(first["v_min"])
    u, v = rows[per - 1, 3], rows[per - 1, 4]
    assert u == pytest.approx(first["u_min"] + first["u_size"])
    assert v == pytest.approx(first["v_min"] + first["v_size"])


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


def test_the_sheet_draws_behind_and_in_front_of_the_clouds(showbase):
    executor = ChunkExecutor(backend="inline")
    cirrus = CloudSheet(showbase.render, R, executor, planet=PLANETS["earthlike"])
    try:
        behind, front = cirrus.behind, cirrus.front
        assert behind.get_bin_name() == SKY_BIN
        assert behind.get_bin_draw_order() == BEHIND_DRAW_ORDER
        assert front.get_bin_name() == FRONT_BIN
        write = cirrus.root.get_attrib(DepthWriteAttrib)
        assert write.get_mode() == DepthWriteAttrib.M_off
        # Hidden until the stamps are placed, then built and placed at the
        # planet's centre
        assert cirrus.root.is_stashed()
        camera = TangentFrame.at(20.0, 40.0).up * (R + 1000.0)
        cirrus.update(camera)
        assert not cirrus.root.is_stashed()
        assert cirrus.stats["pending"] == 0
        assert cirrus.stats["stamps"] == len(cirrus.stamps) > 0
        np.testing.assert_allclose(
            cirrus.root.get_pos(showbase.render), -camera, atol=1.0
        )
        # Each group drawn by both nodes, from the same vertices
        assert behind.get_num_children() == front.get_num_children() > 1
        for a, b in zip(behind.get_children(), front.get_children()):
            assert (
                a.node().get_geom(0).get_vertex_data().get_num_rows()
                == b.node().get_geom(0).get_vertex_data().get_num_rows()
            )
        drawn = sum(
            c.node().get_geom(0).get_vertex_data().get_num_rows()
            for c in behind.get_children()
        )
        assert drawn == len(stamp_vertices(cirrus.stamps)[0])
        # Out of the sea's mirror when the settings say so
        cirrus.set_settings(CloudSettings(reflected=False))
        assert cirrus.root.is_hidden(REFLECTION_CAMERA_MASK)
    finally:
        cirrus.remove()
        executor.shutdown()
