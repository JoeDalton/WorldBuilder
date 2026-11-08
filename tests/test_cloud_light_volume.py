"""
The clouds' light volume (clouds.light_volume, RENDERING_PLAN.md part C):
its chart, its bake read back against the CPU twin (each voxel the cloud
shadow march from itself, toward the sun and up), and its change-driven
rules (nothing a frame with all still; baked again into the back atlas, the
front swapped once whole).
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from world_builder.clouds.light_volume import (
    COLUMNS,
    REACH_MAX,
    REACH_MIN,
    RECENTRE,
    SLICES,
    STEP_SCALE,
    CloudLightVolumes,
    light_chart,
    light_depth_on_sphere,
    light_reach,
    smooth_volume,
    voxel_positions,
)
from world_builder.clouds.sphere import _noise
from world_builder.clouds.weather import sample_cube
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS as PLANETS

R = 600e3


@pytest.fixture(scope="module")
def window(showbase):
    """The session's ShowBase with an offscreen window; skipped where none
    can be opened."""
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


def test_the_reach_is_the_longest_sight_line_in_the_slab():
    base, top = R + 2000.0, R + 12000.0
    under = light_reach(R + 1000.0, base, top)
    assert under == pytest.approx(math.sqrt(top**2 - base**2))
    assert light_reach(R + 30e3, base, top) > under  # higher sees farther
    assert light_reach(R + 1e6, base, top) == REACH_MAX
    assert light_reach(R + 1000.0, base, base + 50.0) == REACH_MIN


def test_the_chart_is_finer_near_its_centre():
    frame = TangentFrame.at(10.0, 20.0)
    tile, reach = 64, 100e3
    chart = light_chart(frame.up, R + 2000.0, 1000.0, reach, tile)
    voxels = voxel_positions(chart, tile)
    assert voxels.shape == (SLICES, tile, tile, 3)
    radii = np.linalg.norm(voxels, axis=-1)
    np.testing.assert_allclose(
        radii[:, 0, 0], R + 2000.0 + (np.arange(SLICES) + 0.5) / SLICES * 1000.0
    )
    row = voxels[0, tile // 2]  # across the middle, west to east
    steps = np.linalg.norm(np.diff(row, axis=0), axis=-1)
    middle = steps[tile // 2 - 2 : tile // 2 + 1].max()
    assert middle < 0.1 * steps[0]  # the edge's texels are much wider
    # The farthest node along an axis is about the reach away (on the base)
    far = np.arccos(np.clip(row[-1] @ frame.up / np.linalg.norm(row[-1]), -1, 1))
    assert far * (R + 2000.0) == pytest.approx(reach * (1 - 1 / tile) ** 2, rel=1e-3)


def _read_atlas(atlas, tile):
    texture = atlas.texture
    dtype = np.float32 if texture.get_component_width() == 4 else np.float16
    ram = np.frombuffer(texture.get_ram_image(), dtype)
    height, width = SLICES // COLUMNS * tile, COLUMNS * tile
    texels = ram.reshape(height, width, -1).astype(np.float64)
    # (slice, row, column, channel): tile k at column k % COLUMNS, row k // COLUMNS
    slices = texels.reshape(SLICES // COLUMNS, tile, COLUMNS, tile, -1)
    return slices.transpose(0, 2, 1, 3, 4).reshape(SLICES, tile, tile, -1)


def test_the_bake_is_the_cpus_march_from_each_voxel(window):
    """Every voxel of a whole volume: the depth toward the sun (r) and up (g)
    as the CPU twin marches them from it; the readers' input on once whole;
    nothing baked again with all still, again on a turned sun or a camera
    moved past the recentring share."""
    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.sphere import CloudVolume
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    clouds = CloudVolume(
        window.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(shells=2),
        planet=PLANETS["forest"],
    )
    tile, steps = 64, 16
    volumes = CloudLightVolumes(
        window.win, clouds, tile=tile, per_frame=8, steps=steps, to_ram=True
    )
    try:
        clouds.update(TangentFrame.at(20.0, 40.0).up * (R + 1000.0))
        assert clouds.weather is not None
        places = [(lat, lon) for lat in range(-60, 61, 20) for lon in range(0, 360, 30)]
        cover = [
            sample_cube(clouds.weather[..., 0], TangentFrame.at(*place).up[None])[0]
            for place in places
        ]
        frame = TangentFrame.at(*places[int(np.argmax(cover))])
        camera = frame.up * (R + 1000.0)
        clouds.update(camera)
        sun = 0.8 * frame.up + 0.6 * frame.east
        sun /= np.linalg.norm(sun)
        clouds.set_sun(sun)
        volume = volumes.volumes[0]
        frames = SLICES // 8 + 1  # the slices, then their smoothing
        for _ in range(frames):
            volumes.update(camera, sun)
            assert volumes._on[0][0] == 0  # nothing whole yet
            window.graphicsEngine.render_frame()
        volumes.update(camera, sun)
        assert volumes._on[0][0] == 1 and volume.front is not None
        got = _read_atlas(volume.raw, tile)
        # The front: the raw bake smoothed (the CPU's tent, to half floats)
        front = _read_atlas(volume.front, tile)
        smoothed = smooth_volume(got)
        assert np.allclose(front, smoothed, rtol=2e-3, atol=2e-3)
        assert np.abs(front - got).max() > 0.1  # smoothing did something

        chart = [tuple(volume.front.chart[k]) for k in range(4)]
        points = voxel_positions(chart, tile).reshape(-1, 3)
        field, optics = clouds._deck_field, clouds._deck_optics
        optics = replace(optics, sun_step_length=optics.sun_step_length * STEP_SCALE)

        def march(dirs):
            return light_depth_on_sphere(
                _noise(7),
                field,
                optics,
                points,
                dirs,
                R,
                steps,
                shift_at=lambda p: sample_cube(clouds.weather[..., 1], p),
            ).reshape(SLICES, tile, tile)

        direct = march(sun)
        sky = march(points / np.linalg.norm(points, axis=-1, keepdims=True))
        assert direct.max() > 1.0 and sky.max() > 1.0  # cloud in it
        for k, expected in ((0, direct), (1, sky)):
            error = np.abs(np.exp2(-got[..., k]) - np.exp2(-expected))
            assert np.median(error) < 2e-3
            assert np.percentile(error, 99) < 0.05

        # Still: nothing baked; the front stays.
        front = volume.front
        volumes.update(camera, sun)
        assert not volume.baking and volume.front is front
        # A turned sun: a bake into the other atlas, the front kept meanwhile.
        turned = sun + 0.1 * frame.north
        turned /= np.linalg.norm(turned)
        volumes.update(camera, turned)
        assert volume.baking and volume.front is front
        for _ in range(frames):
            window.graphicsEngine.render_frame()
            volumes.update(camera, turned)
        assert not volume.baking and volume.front is not front
        assert np.allclose(volume.front.baked[1], turned)
        # A camera moved past the share: again; less: not.
        front = volume.front
        reach = front.baked[4]
        base = front.baked[5]
        near = (frame.up + 0.5 * RECENTRE * reach / base * frame.east) * (R + 1000.0)
        volumes.update(near, turned)
        assert not volume.baking
        far = (frame.up + 2.0 * RECENTRE * reach / base * frame.east) * (R + 1000.0)
        volumes.update(far, turned)
        assert volume.baking
    finally:
        volumes.destroy()
        clouds.remove()
        executor.shutdown()


@pytest.mark.parametrize("family", [0, 1])
def test_the_genera_bake_is_the_cpus_march_for_each_family(window, family):
    """With genera each family has its own volume from its own field and
    weather: towers (the convective depth and the anvil's spread taken at
    every step) and the stratus deck, against the CPU twin."""
    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.genera import GeneraSpec
    from world_builder.clouds.planet import PlanetClouds
    from world_builder.clouds.weather import WeatherSpec
    from world_builder.terrain.planet import PlanetSpec
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    spec = PlanetSpec(name="clouds", radius=R)
    genera = GeneraSpec(weather=WeatherSpec(spec, seed=11))
    clouds = PlanetClouds(
        window.render,
        R,
        executor,
        settings=CloudSettings(shells=2, renderer="march"),
        genera=genera,
    )
    tile, steps = 64, 24
    volumes = CloudLightVolumes(
        window.win, clouds, tile=tile, per_frame=8, steps=steps, to_ram=True
    )
    try:
        # Over the family's own clouds (genera.DEFAULT_FEATURES)
        frame = TangentFrame.at(-2.86 if family else 0.0, 0.0)
        camera = frame.up * (R + 1000.0)
        for _ in range(400):
            clouds.update(camera)
            if clouds.volume.weather is not None:
                break
        sun = 0.6 * frame.up + 0.8 * frame.east
        sun /= np.linalg.norm(sun)
        clouds.set_sun(sun)
        volume = volumes.volumes[family]
        for _ in range(SLICES // 8 + 1):  # the slices, then their smoothing
            volumes.update(camera, sun)
            window.graphicsEngine.render_frame()
        volumes.update(camera, sun)
        assert volume.front is not None
        got = _read_atlas(volume.raw, tile)
        caster = clouds.shadow_caster(family)
        weather = caster.weather

        def channel(k):
            if weather.shape[-1] <= k:
                return None
            return lambda p: sample_cube(weather[..., k], p)

        chart = [tuple(volume.front.chart[k]) for k in range(4)]
        # Every fourth slice: the twin's cost
        points = voxel_positions(chart, tile)[::4].reshape(-1, 3)
        optics = caster._deck_optics
        optics = replace(optics, sun_step_length=optics.sun_step_length * STEP_SCALE)
        for k, dirs in (
            (0, sun),
            (1, points / np.linalg.norm(points, axis=-1, keepdims=True)),
        ):
            expected = light_depth_on_sphere(
                _noise(7),
                caster.shadow_field,
                optics,
                points,
                dirs,
                R,
                steps,
                shift_at=channel(1),
                depth_at=channel(2),
                spread_at=channel(3),
            ).reshape(SLICES // 4, tile, tile)
            assert expected.max() > 0.5  # cloud in it
            error = np.abs(np.exp2(-got[::4, ..., k]) - np.exp2(-expected))
            assert np.median(error) < 5e-3
            assert np.percentile(error, 99) < 0.08
    finally:
        volumes.destroy()
        clouds.remove()
        executor.shutdown()
