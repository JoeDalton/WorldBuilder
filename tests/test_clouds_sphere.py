"""
Unit tests for planet-wide clouds (world_builder.clouds.sphere, the
planet-mode density in world_builder.clouds.noise, and the weather map in
world_builder.clouds.weather). The shaders need a window (see
scripts/terrain_shots.py); these check placement, calibration and bookkeeping.
"""

from dataclasses import replace

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudSettings, CloudType
from world_builder.clouds.noise import (
    DECK_LIGHT_COS_MIN,
    deck_light_table,
    density_on_sphere,
    fbm_sigma,
    sphere_column_peaks,
    sun_optical_depth_on_sphere,
    with_threshold,
)
from world_builder.clouds.sphere import (
    ShellSpec,
    SphereCloudSpec,
    SphericalClouds,
    _noise,
    generate_cloud_tile,
    octave_offset,
)
from world_builder.clouds.weather import (
    WeatherSpec,
    coverage_at,
    coverage_map,
    coverage_to_shift,
    orographic,
    sample_cube,
    shift_table,
    vortices,
    weather_cube,
)
from world_builder.quality import QualityLevel
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.chart import TangentFrame, dirs_from_latlon
from world_builder.terrain.cubesphere import face_dirs, grid_for
from world_builder.terrain.planet import PRESETS as PLANETS
from world_builder.terrain.planet import Planet, PlanetSpec
from world_builder.terrain.workers import ChunkExecutor, SharedArray

R = 600e3


RAW = replace(PRESETS[CloudType.CUMULUS].field, slab=(2200.0, 1000.0))


@pytest.fixture(scope="module")
def peaks():
    return sphere_column_peaks(_noise(7), RAW, R)


@pytest.fixture(scope="module")
def field(peaks):
    return with_threshold(RAW, peaks, fbm_sigma(_noise(7), RAW))


@pytest.fixture(scope="module")
def table(peaks, field):
    return shift_table(peaks, field.threshold[0])


@pytest.fixture(scope="module")
def weather():
    return WeatherSpec(PLANETS["earthlike"])


@pytest.fixture(scope="module")
def cube(weather, table):
    return weather_cube(weather, table)


@pytest.fixture(scope="module")
def spec(field, weather, table):
    grid = grid_for(R)
    base = round(np.log2(8000.0 / grid.chunk_size(0)))
    shells = tuple(
        ShellSpec(base + k, (70 * 2**k, 140 * 2**k), 2.0, (4, 3, 2)[k])
        for k in range(3)
    )
    return SphereCloudSpec(R, field, 7, weather, tuple(table), shells)


def _uniform(coverage: float, table: np.ndarray) -> np.ndarray:
    """A weather cube with the same coverage everywhere."""
    cube = np.empty((6, 4, 4, 2), dtype=np.float32)
    cube[..., 0] = coverage
    cube[..., 1] = coverage_to_shift(coverage, table)
    return cube


def _column_coverage(field, weather, octaves=4, columns=6000, levels=32, seed=0):
    """Share of random radial columns through the slab holding any cloud."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(columns, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    base, thickness = field.slab
    heights = R + base + (np.arange(levels) + 0.5) / levels * thickness
    points = dirs[:, None, :] * heights[None, :, None]
    values = density_on_sphere(_noise(7), weather, field, points, R, octaves)
    return float((values > 0.0).any(axis=1).mean())


# ── The cube map ───────────────────────────────────────────────────────────────


def test_cube_sampler_reads_texel_centres():
    """Bilinear with the half-texel offset: a texel centre reads its texel,
    on every face and in the GPU's face layout."""
    rng = np.random.default_rng(1)
    cube = rng.random((6, 8, 8, 2))
    dirs = cube_directions(8)
    for face, row, column in [(0, 2, 5), (3, 7, 0), (5, 4, 4), (2, 0, 7)]:
        value = sample_cube(cube, dirs[face, row, column])
        np.testing.assert_allclose(value, cube[face, row, column], atol=1e-9)
    # Between two centres, halfway.
    mid = dirs[1, 3, 2] + dirs[1, 3, 3]
    np.testing.assert_allclose(
        sample_cube(cube[..., 0], mid), cube[1, 3, 2:4, 0].mean(), atol=0.01
    )


# ── Coverage to shift, and the density ────────────────────────────────────────


@pytest.mark.parametrize("coverage", [0.1, 0.3, 0.6, 0.9])
def test_shift_table_gives_the_coverage(field, table, coverage):
    """The table is the inverse of the coverage function: shifting the field
    by its entry for c puts cloud in a share c of the columns."""
    measured = _column_coverage(field, _uniform(coverage, table))
    assert measured == pytest.approx(coverage, abs=0.03)


def test_shift_table_ends_are_clear_and_overcast(field, table):
    assert _column_coverage(field, _uniform(0.0, table), columns=1500) == 0.0
    assert _column_coverage(field, _uniform(1.0, table), columns=1500) == 1.0


@pytest.mark.parametrize("coverage", [0.15, 0.35, 0.7])
def test_dropping_octaves_keeps_the_coverage(field, peaks, table, coverage):
    """Far shells evaluate fewer octaves; the dropped ones count as their
    mean, and the shell's threshold offset makes up for the lost spread: the
    share of cloudy columns holds within 2 %."""
    weather = _uniform(coverage, table)
    full = _column_coverage(field, weather, octaves=4)
    for octaves in (3, 2):
        offset = octave_offset(peaks, sphere_column_peaks(_noise(7), RAW, R, octaves))
        shell = ShellSpec(0, (1.0, 1.0), 1.0, octaves, offset)
        measured = _column_coverage(shell.field_of(field), weather, octaves)
        assert measured == pytest.approx(full, abs=0.02), octaves


@pytest.mark.parametrize("octaves", [4, 3, 2])
def test_early_out_changes_nothing(field, cube, octaves):
    """The shader stops after one or two octaves where the rest cannot reach
    the threshold; the CPU mirror of it gives the very same densities."""
    rng = np.random.default_rng(octaves)
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    base, thickness = field.slab
    points = dirs * (R + base + thickness * rng.uniform(size=20000))[:, None]
    plain = density_on_sphere(_noise(7), cube, field, points, R, octaves)
    early = density_on_sphere(
        _noise(7), cube, field, points, R, octaves, early_out=True
    )
    np.testing.assert_array_equal(early, plain)
    assert 0.02 < (plain > 0.0).mean() < 0.9  # both sides are exercised


def test_density_lives_in_the_slab_above_the_sphere(field, cube):
    up = TangentFrame.at(20.0, 40.0).up
    heights = np.array([0.0, 1500.0, 2700.0, 3500.0, 9000.0])
    points = up[None, :] * (R + heights)[:, None]
    values = density_on_sphere(_noise(7), cube, field, points, R)
    assert values[0] == 0.0 and values[1] == 0.0  # below the base
    assert values[3] == 0.0 and values[4] == 0.0  # above the top
    assert 0.0 <= values[2] <= 1.0


def test_the_weather_opens_and_closes_the_deck(field, cube):
    """Where the map says overcast there is more cloud than where it says
    clear."""
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = dirs * (R + 2700.0)
    cover = sample_cube(cube[..., 0], dirs)
    cloudy = density_on_sphere(_noise(7), cube, field, points, R) > 0.0
    wet, dry = cover > 0.7, cover < 0.15
    assert wet.sum() > 200 and dry.sum() > 200
    assert cloudy[wet].mean() > cloudy[dry].mean() + 0.3


# ── The far deck's lighting ───────────────────────────────────────────────────

OPTICS = PRESETS[CloudType.CUMULUS].optics


def test_the_sun_march_counts_only_the_slab(field, table):
    """Above the slab, looking up, the sun is never shadowed; and with no
    steps, nothing is."""
    up = TangentFrame.at(20.0, 40.0).up
    overcast = coverage_to_shift(1.0, table)
    above = up * (R + sum(field.slab) + 10.0)
    depth = sun_optical_depth_on_sphere(
        _noise(7), field, OPTICS, above[None], up, R, overcast
    )
    assert depth[0] == 0.0
    inside = up * (R + field.slab[0] + 0.5 * field.slab[1])
    none = sun_optical_depth_on_sphere(
        _noise(7), field, replace(OPTICS, sun_steps=0), inside[None], up, R, overcast
    )
    assert none[0] == 0.0


def test_the_sun_march_along_overcast_grows_with_its_steps(field, table):
    """Marching along the middle of an overcast slab, each step adds depth."""
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(500, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    level = np.cross(dirs, [0.0, 0.0, 1.0])
    level /= np.linalg.norm(level, axis=1, keepdims=True)
    points = dirs * (R + field.slab[0] + 0.5 * field.slab[1])
    overcast = coverage_to_shift(1.0, table)
    short, long = (
        sun_optical_depth_on_sphere(
            _noise(7), field, replace(OPTICS, sun_steps=n), points, level, R, overcast
        )
        for n in (2, 4)
    )
    assert (short > 0.0).mean() > 0.9
    assert np.all(long >= short)
    assert long.mean() > 1.5 * short.mean()


@pytest.fixture(scope="module")
def deck_light(field, table):
    return deck_light_table(
        _noise(7), field, OPTICS, R, table, cosines=16, columns=1024
    )


def test_the_deck_light_is_a_shadowing(deck_light):
    """Shares of the sunlight: in [0, 1], the powder only darkens, and a
    clear sky is unshadowed."""
    assert deck_light.shape == (17, 16, 2)
    assert np.all((deck_light >= 0.0) & (deck_light <= 1.0))
    assert np.all(deck_light[..., 0] <= deck_light[..., 1] + 1e-6)
    np.testing.assert_array_equal(deck_light[0], 1.0)


def test_a_low_sun_shadows_the_deck_more(deck_light):
    """Toward the terminator the sun marches along the slab through the
    clouds around: the deck darkens gradually, never to nothing (the far
    deck used to drop its sunlight to black at the sun's horizon)."""
    mu = np.linspace(DECK_LIGHT_COS_MIN, 1.0, deck_light.shape[1])
    low, high = np.argmin(np.abs(mu - 0.05)), -1
    for row in range(4, 11):  # coverage 0.25 to 0.625
        visibility = deck_light[row, :, 1]
        assert visibility[low] < visibility[high] - 0.05, row
        assert visibility[low] > 0.2, row


def test_the_deck_light_is_deterministic(field, table):
    args = (_noise(7), field, OPTICS, R, table)
    first = deck_light_table(*args, cosines=4, columns=256)
    np.testing.assert_array_equal(
        first, deck_light_table(*args, cosines=4, columns=256)
    )


# ── The weather map ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def earth_map(weather):
    return coverage_map(weather, 64)


def test_weather_map_is_a_deterministic_coverage(weather, earth_map):
    assert earth_map.shape == (6, 64, 64)
    assert earth_map.min() >= 0.0 and earth_map.max() <= 1.0
    again = coverage_map.__wrapped__(weather, 64)
    np.testing.assert_array_equal(again, earth_map)
    assert 0.1 < earth_map.mean() < 0.7


def _band_mean(cover, lat_lo, lat_hi):
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.linspace(-180.0, 180.0, 360, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    values = sample_cube(cover, dirs_from_latlon(la.ravel(), lo.ravel()))
    band = (np.abs(la.ravel()) >= lat_lo) & (np.abs(la.ravel()) < lat_hi)
    return float(values[band].mean())


def test_weather_follows_the_zonal_circulation(earth_map):
    """Wet tropics, dry subtropics, cloudy storm tracks."""
    equator = _band_mean(earth_map, 0.0, 6.0)
    subtropics = _band_mean(earth_map, 22.0, 30.0)
    storm_tracks = _band_mean(earth_map, 48.0, 60.0)
    assert equator > subtropics + 0.15
    assert storm_tracks > subtropics + 0.15


def test_cyclone_cores_are_cloudier_than_their_surroundings(weather):
    planet = Planet(weather.planet)
    systems = vortices(weather, planet)
    assert sum(not v.tropical for v in systems) >= 4
    assert any(v.tropical for v in systems)  # the earthlike planet has sea
    for v in systems:
        c = np.asarray(v.centre)
        frame_east = np.cross([0.0, 0.0, 1.0], c)
        frame_east /= np.linalg.norm(frame_east)
        north = np.cross(c, frame_east)
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        ring = c + 4.0 * v.radius * (
            np.cos(angles)[:, None] * frame_east + np.sin(angles)[:, None] * north
        )
        ring /= np.linalg.norm(ring, axis=1, keepdims=True)
        # Just off the centre: a tropical cyclone's eye is clear.
        core = c + 0.4 * v.radius * frame_east
        values = coverage_at(weather, np.vstack([core, ring]), planet)
        # The median: the ring may cross a neighbouring system.
        assert values[0] > np.median(values[1:]) + 0.2, v


def test_tropical_cyclones_stay_over_the_sea(weather):
    planet = Planet(weather.planet)
    tropical = [v for v in vortices(weather, planet) if v.tropical]
    centres = np.array([v.centre for v in tropical])
    elevation, _, _ = planet.sample_dirs(centres, min_wavelength=60e3)
    assert np.all(elevation < 0.0)
    dry = WeatherSpec(PLANETS["desert"])
    assert not any(v.tropical for v in vortices(dry))


def test_a_desert_planet_is_clearer(earth_map):
    desert = coverage_map(WeatherSpec(PLANETS["desert"]), 64)
    assert desert.mean() < 0.6 * earth_map.mean()


def test_windward_slopes_are_cloudier_than_the_lee():
    """A north-south ridge at 45 degrees north under the westerlies: the
    west side is lifted, the east side is in its lee."""
    spec = WeatherSpec(PlanetSpec(name="ridge", radius=R))
    centre = dirs_from_latlon(45.0, 0.0)[0]
    east = np.cross([0.0, 0.0, 1.0], centre)
    east /= np.linalg.norm(east)

    def ridge(dirs):
        x = (dirs @ east) * R  # metres east of the crest
        return 2000.0 * np.exp(-((x / 20e3) ** 2))

    west_side = centre - east * (15e3 / R)
    east_side = centre + east * (15e3 / R)
    sides = np.stack([west_side, east_side])
    sides /= np.linalg.norm(sides, axis=1, keepdims=True)
    lift = orographic(sides, ridge, spec, step=10e3)
    assert lift[0] > 0.1 and lift[1] < -0.1


# ── Tiles ──────────────────────────────────────────────────────────────────────


def _cloudy_dir(cube) -> np.ndarray:
    """A direction the weather keeps well covered."""
    dirs = cube_directions(cube.shape[1])
    face, row, column = np.argwhere(cube[..., 0] > 0.7)[0]
    return dirs[face, row, column][None, :]


def test_tiles_place_billboards_on_cloud(spec, cube):
    grid = grid_for(R)
    level = spec.shells[0].level
    dirs = _cloudy_dir(cube)
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
    values = density_on_sphere(_noise(7), cube, spec.field, positions, R)
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


def test_tiles_read_the_shared_weather(spec, cube):
    """Workers read the weather's shift from shared memory instead of each
    building the map (~0.6 s): the same tiles, in this process and in a
    worker process."""
    grid = grid_for(R)
    level = spec.shells[0].level
    face, i, j, _, _ = grid.locate(_cloudy_dir(cube), level)
    key = (level, int(face[0]), int(i[0]), int(j[0]))
    built = generate_cloud_tile(spec, key)
    shared = SharedArray(cube[..., 1])
    try:
        with_shared = replace(spec, shift=shared.handle)
        here = generate_cloud_tile(with_shared, key)
        np.testing.assert_array_equal(here.particles, built.particles)
        executor = ChunkExecutor(backend="process", workers=1)
        try:
            there = executor.submit_once(generate_cloud_tile, with_shared, key)
            np.testing.assert_array_equal(
                there.result(timeout=120).particles, built.particles
            )
        finally:
            executor.shutdown()
    finally:
        shared.release()


def test_coarser_shells_use_bigger_billboards(spec, cube):
    grid = grid_for(R)
    dirs = _cloudy_dir(cube)
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


# ── The scene ──────────────────────────────────────────────────────────────────


def test_clouds_draw_tiles_around_the_camera(showbase):
    executor = ChunkExecutor(backend="inline", inline_budget=64)
    clouds = SphericalClouds(
        showbase.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(shells=2),
        planet=PLANETS["forest"],
    )
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
        # The deck sits at the planet's centre, its far lighting bound.
        np.testing.assert_allclose(
            clouds.deck.get_pos(showbase.render), -camera, atol=1.0
        )
        light = clouds.root.get_shader_input("deckLight").get_texture()
        assert light.get_x_size() == 32 and light.get_y_size() == 17
    finally:
        clouds.remove()
        executor.shutdown()


def test_lower_quality_places_fewer_billboards(showbase):
    """The quality presets (the demo's F4): bigger billboards and less
    overlap, so the same tile holds fewer of them; the far shells evaluate
    fewer octaves."""
    executor = ChunkExecutor(backend="inline")
    counts = {}
    try:
        for quality in (QualityLevel.HIGH, QualityLevel.LOW):
            clouds = SphericalClouds(
                showbase.render,
                R,
                executor,
                altitude=2200.0,
                planet=PLANETS["forest"],
                settings=CloudSettings.preset(quality),
                channel="clouds",
            )
            spec = clouds.spec
            shell = spec.shells[0]
            dirs = TangentFrame.at(20.0, 40.0).up[None, :]
            face, i, j, _, _ = grid_for(R).locate(dirs, shell.level)
            key = (shell.level, int(face[0]), int(i[0]), int(j[0]))
            counts[quality] = len(generate_cloud_tile(spec, key).particles)
            expected = {QualityLevel.HIGH: [4, 4, 4, 3], QualityLevel.LOW: [4, 3, 3, 2]}
            assert [s.octaves for s in spec.shells] == expected[quality]
            clouds.remove()
    finally:
        executor.shutdown()
    assert counts[QualityLevel.HIGH] > 0
    assert counts[QualityLevel.LOW] < 0.5 * counts[QualityLevel.HIGH]


def test_deck_mesh_is_a_sphere():
    from world_builder.clouds.sphere import deck_mesh

    node = deck_mesh(R + 2700.0, quads=4)
    vdata = node.get_geom(0).get_vertex_data()
    data = np.frombuffer(bytes(memoryview(vdata.get_array(0))), np.float32).reshape(
        -1, 3
    )
    np.testing.assert_allclose(np.linalg.norm(data, axis=1), R + 2700.0, rtol=1e-6)
    assert len(data) == 6 * 5 * 5
    # Same corners as the terrain's cube-sphere.
    corner = face_dirs(0, np.array([1.0]), np.array([1.0]))[0] * (R + 2700.0)
    assert np.min(np.linalg.norm(data - corner, axis=1)) < 1.0
