"""
Unit tests for planet-wide clouds (world_builder.clouds.sphere and the
planet-mode density in world_builder.clouds.noise). The shaders need a window
(see scripts/terrain_shots.py); these check placement and bookkeeping.
"""

from dataclasses import replace

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.noise import (
    column_peaks,
    density_on_sphere,
    fbm_sigma,
    sample_weather,
    weather_volume,
    with_threshold,
)
from world_builder.clouds.sphere import (
    ShellSpec,
    SphereCloudSpec,
    SphericalClouds,
    _noise,
    generate_cloud_tile,
)
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.cubesphere import face_dirs, grid_for
from world_builder.terrain.workers import ChunkExecutor

R = 600e3


@pytest.fixture(scope="module")
def field():
    raw = replace(PRESETS[CloudType.CUMULUS].field, slab=(2200.0, 1000.0))
    volume = _noise(7)
    return with_threshold(raw, column_peaks(volume, raw), fbm_sigma(volume, raw))


@pytest.fixture(scope="module")
def spec(field):
    grid = grid_for(R)
    base = round(np.log2(8000.0 / grid.chunk_size(0)))
    shells = tuple(ShellSpec(base + k, (70 * 2**k, 140 * 2**k), 2.5) for k in range(3))
    return SphereCloudSpec(R, field, 7, 11, 0.55, shells)


def test_weather_samples_like_the_gpu():
    """Trilinear with the half-texel offset: texel centres read their texel."""
    volume = weather_volume(3, size=8)
    size = volume.shape[0]
    centre = (np.array([2, 5, 1]) + 0.5) / size * 2.0 - 1.0  # x, y, z
    value = sample_weather(volume, centre[None, :])[0]
    assert value == pytest.approx(volume[1, 5, 2])  # [z, y, x]
    assert 0.0 <= volume.min() and volume.max() <= 1.0


def test_density_lives_in_the_slab_above_the_sphere(field):
    volume = _noise(7)
    weather = weather_volume(11)
    up = TangentFrame.at(20.0, 40.0).up
    heights = np.array([0.0, 1500.0, 2700.0, 3500.0, 9000.0])
    points = up[None, :] * (R + heights)[:, None]
    values = density_on_sphere(volume, weather, field, points, R, 0.55)
    assert values[0] == 0.0 and values[1] == 0.0  # below the base
    assert values[3] == 0.0 and values[4] == 0.0  # above the top
    assert 0.0 <= values[2] <= 1.0


def test_weather_opens_and_closes_the_deck(field):
    """Some regions clear, some overcast: the gain shifts the coverage."""
    volume = _noise(7)
    weather = weather_volume(11)
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = dirs * (R + 2700.0)
    w = sample_weather(weather, dirs)
    cloudy = density_on_sphere(volume, weather, field, points, R, 0.55) > 0.0
    wet, dry = w > np.quantile(w, 0.8), w < np.quantile(w, 0.2)
    assert cloudy[wet].mean() > cloudy[dry].mean() + 0.2


def test_tiles_place_billboards_on_cloud(spec):
    grid = grid_for(R)
    level = spec.shells[0].level
    dirs = TangentFrame.at(20.0, 40.0).up[None, :]
    face, i, j, _, _ = grid.locate(dirs, level)
    tile = generate_cloud_tile(spec, (level, int(face[0]), int(i[0]), int(j[0])))
    p = tile.particles
    assert p.dtype == np.float32 and p.shape[1] == 9
    assert len(p) > 0
    positions = p[:, 0:3].astype(np.float64) + tile.centre
    # In the slab, on the field the shader draws.
    height = np.linalg.norm(positions, axis=1) - R
    base, thickness = spec.field.slab
    assert np.all((height > base - 1.0) & (height < base + thickness + 1.0))
    weather = weather_volume(spec.weather_seed)
    values = density_on_sphere(_noise(7), weather, spec.field, positions, R, 0.55)
    assert np.all(values > 0.0)
    # Within the tile's patch.
    tile_face, ti, tj, _, _ = grid.locate(positions, level)
    assert np.all(tile_face == face[0]) and np.all(ti == i[0]) and np.all(tj == j[0])
    radii = p[:, 3]
    assert radii.min() >= 70.0 and radii.max() <= 140.0
    assert np.all(p[:, 8] == 0)  # shell index
    # Deterministic.
    again = generate_cloud_tile(spec, tile.key)
    np.testing.assert_array_equal(again.particles, tile.particles)


def test_coarser_shells_use_bigger_billboards(spec):
    grid = grid_for(R)
    dirs = TangentFrame.at(20.0, 40.0).up[None, :]
    for k, shell in enumerate(spec.shells):
        face, i, j, _, _ = grid.locate(dirs, shell.level)
        tile = generate_cloud_tile(
            spec, (shell.level, int(face[0]), int(i[0]), int(j[0]))
        )
        if len(tile.particles):
            assert tile.particles[:, 3].min() >= shell.radius[0] - 1e-3
            assert np.all(tile.particles[:, 8] == k)


def test_empty_sky_gives_an_empty_tile(field, spec):
    """A coverage of zero leaves nothing to place (and no crash)."""
    clear = replace(spec, field=replace(field, threshold=(9.0, 9.5)))
    grid = grid_for(R)
    level = clear.shells[0].level
    tile = generate_cloud_tile(
        clear, (level, 0, grid.side(level) // 2, grid.side(level) // 2)
    )
    assert len(tile.particles) == 0


def test_clouds_draw_tiles_around_the_camera(showbase):
    executor = ChunkExecutor(backend="inline", inline_budget=64)
    clouds = SphericalClouds(showbase.render, R, executor, altitude=2200.0, shells=2)
    try:
        up = TangentFrame.at(20.0, 40.0).up
        camera = up * (R + 1000.0)
        for _ in range(12):
            clouds.update(camera)
        assert clouds.stats["tiles"] > 0
        assert clouds.stats["billboards"] > 0
        # Every shown tile sits where it is on the planet, relative to the
        # camera at the render origin.
        for key in clouds._shown:
            tile = clouds._tiles[key]
            expected = tile.data.centre - camera
            np.testing.assert_allclose(
                tile.node.get_pos(showbase.render), expected, atol=1.0
            )
        # Only the near shells' levels, near the camera.
        levels = {k[0] for k in clouds._shown}
        assert levels <= {s.level for s in clouds.spec.shells}
        # The deck sits at the planet's centre.
        np.testing.assert_allclose(
            clouds.deck.get_pos(showbase.render), -camera, atol=1.0
        )
    finally:
        clouds.remove()
        executor.shutdown()


def test_deck_mesh_is_a_sphere():
    from world_builder.clouds.sphere import _deck_mesh

    node = _deck_mesh(R + 2700.0, quads=4)
    vdata = node.get_geom(0).get_vertex_data()
    data = np.frombuffer(bytes(memoryview(vdata.get_array(0))), np.float32).reshape(
        -1, 3
    )
    np.testing.assert_allclose(np.linalg.norm(data, axis=1), R + 2700.0, rtol=1e-6)
    assert len(data) == 6 * 5 * 5
    # Same corners as the terrain's cube-sphere.
    corner = face_dirs(0, np.array([1.0]), np.array([1.0]))[0] * (R + 2700.0)
    assert np.min(np.linalg.norm(data - corner, axis=1)) < 1.0
