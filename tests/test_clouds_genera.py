"""
Unit tests for the genera (world_builder.clouds.genera) and the convective
field's growth with depth (world_builder.clouds.noise): the top, the anvil,
the cell octave, the calibration by depth and the placeholder maps. The
shaders need a window (see scripts/cloud_bench.py).
"""

import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.genera import (
    CONVECTIVE,
    DEPTHS,
    STRATIFORM,
    Feature,
    GeneraSpec,
    depth_shift,
    depth_shift_tables,
    genus_cube,
    genus_planes,
)
from world_builder.clouds.noise import (
    ANVIL_BAND,
    ANVIL_DEPTH,
    CELL_OFFSET,
    CELL_SCALE,
    SHELF_BAND,
    SHELF_CLEAR,
    SHELF_COVERAGE,
    SHELF_EDGE,
    cell_shift,
    density_on_sphere,
    fbm_sigma,
    pack_layer_params,
    shelf_underside,
    sphere_column_peaks,
    threshold_lift,
    top_fraction,
    with_shelf,
    with_threshold,
)
from world_builder.clouds.sphere import _noise
from world_builder.clouds.weather import WeatherSpec, shift_table
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS as PLANETS

R = 600e3
SHADERS = Path(__file__).parent.parent / "src/world_builder/clouds/shaders"


@pytest.fixture(scope="module")
def convective():
    raw = PRESETS[CloudType.CONVECTIVE].field
    volume = _noise(7)
    peaks = sphere_column_peaks(volume, raw, R)
    field = with_shelf(with_threshold(raw, peaks, fbm_sigma(volume, raw)), volume, R)
    return field, depth_shift_tables(volume, field, R)


def _cube(coverage, depth, tables, shelf=None):
    """A three-channel weather cube, the same everywhere; four with an
    anvil shelf of spread *shelf*."""
    cube = np.empty((6, 4, 4, 3 if shelf is None else 4), dtype=np.float32)
    cube[..., 0] = coverage
    cube[..., 1] = depth_shift(
        np.full((6, 4, 4), coverage), np.full((6, 4, 4), depth), tables
    )
    cube[..., 2] = depth
    if shelf is not None:
        cube[..., 3] = shelf
    return cube


def _columns(field, cube, columns=4000, levels=96, seed=0):
    """(columns, levels) density through the whole slab of random columns."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(columns, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    base, thickness = field.slab
    heights = R + base + (np.arange(levels) + 0.5) / levels * thickness
    points = dirs[:, None, :] * heights[None, :, None]
    return density_on_sphere(_noise(7), cube, field, points, R), heights - R


# ── A field without depth is unchanged ───────────────────────────────────────


def test_a_field_without_depth_ignores_the_depth_map():
    raw = replace(PRESETS[CloudType.CUMULUS].field, slab=(2200.0, 1000.0))
    volume = _noise(7)
    field = with_threshold(
        raw, sphere_column_peaks(volume, raw, R), fbm_sigma(volume, raw)
    )
    rng = np.random.default_rng(1)
    two = rng.uniform(-0.1, 0.1, (6, 4, 4, 2)).astype(np.float32)
    three = np.concatenate(
        [two, rng.uniform(0, 1, (6, 4, 4, 1)).astype(np.float32)], -1
    )
    d = rng.normal(size=(3000, 3))
    points = d / np.linalg.norm(d, axis=1, keepdims=True) * (R + 2700.0)
    assert np.array_equal(
        density_on_sphere(volume, two, field, points, R),
        density_on_sphere(volume, three, field, points, R),
    )
    assert top_fraction(field, 0.7) == 1.0
    assert cell_shift(volume, field, points, 1.0) == 0.0


# ── Growth with depth ────────────────────────────────────────────────────────


def test_the_top_rises_with_depth_from_the_cumulus_to_the_cap(convective):
    field, tables = convective
    base, thickness = field.slab
    tops = []
    for depth in (0.0, 0.5, 1.0):
        density, heights = _columns(field, _cube(0.5, depth, tables))
        cloudy = heights[(density > 0.0).any(axis=0)]
        top = base + thickness * top_fraction(field, depth)
        assert cloudy.max() <= top
        tops.append(cloudy.max())
    # Depth 0 is the cumulus: a kilometre; depth 1 nearly reaches the cap
    assert tops[0] <= base + 1000.0
    assert tops[0] < tops[1] < tops[2]
    assert tops[2] > base + 0.8 * thickness


def test_nothing_is_cloud_above_the_local_top(convective):
    field, _ = convective
    h = np.array([0.1, 0.119, 0.13, 0.5])
    lift = threshold_lift(field, h, 0.0)
    assert np.isfinite(lift[:2]).all()
    assert np.isinf(lift[2:]).all()
    assert np.isfinite(threshold_lift(field, h, 1.0)).all()


def test_the_deepest_clouds_spread_under_the_cap(convective):
    field, tables = convective
    base, thickness = field.slab
    flat = replace(field, anvil=0.0)
    cube = _cube(0.5, 1.0, tables)
    with_anvil, heights = _columns(field, cube)
    without, _ = _columns(flat, cube)
    band = (heights - base) / thickness
    anvil = (band > ANVIL_BAND[0]) & (band < ANVIL_BAND[1])
    assert (with_anvil[:, anvil] > 0).mean() > 1.5 * (without[:, anvil] > 0).mean()
    # Below it nothing changes
    low = band < ANVIL_BAND[0] * 0.9
    assert np.array_equal(with_anvil[:, low], without[:, low])


def test_the_anvil_shelf_spreads_over_shallow_columns_in_its_band(convective):
    field, tables = convective
    base, thickness = field.slab
    plain, heights = _columns(field, _cube(0.3, 0.0, tables))
    shelf, _ = _columns(field, _cube(0.3, 0.0, tables, shelf=1.0))
    band = (heights - base) / thickness
    # Over cumulus a kilometre tall, a sheet of cloud in its band
    assert not (plain[:, band > 0.2] > 0).any()
    assert (shelf[:, band > SHELF_BAND[0]] > 0).any(axis=1).mean() > 0.97
    # Not below it, and the cumulus under it are kept
    under = band < SHELF_BAND[0]
    assert not (shelf[:, under & (band > 0.2)] > 0).any()
    assert np.array_equal(shelf[:, band < 0.2], plain[:, band < 0.2])
    # And none at spread 0: the three-channel field
    none, _ = _columns(field, _cube(0.3, 0.0, tables, shelf=0.0))
    assert np.array_equal(none, plain)


def test_the_shelf_is_a_sheet_of_its_coverage_in_its_middle(convective):
    field, tables = convective
    base, thickness = field.slab
    rng = np.random.default_rng(4)
    d = rng.normal(size=(8000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    middle = 0.5 * (SHELF_BAND[1] + SHELF_BAND[2])
    points = d * (R + base + middle * thickness)
    cube = _cube(0.3, 0.0, tables, shelf=1.0)
    density = density_on_sphere(_noise(7), cube, field, points, R)
    assert (density > 0).mean() == pytest.approx(SHELF_COVERAGE, abs=0.03)


def test_the_shelf_is_a_lens_thinning_with_its_spread(convective):
    field, tables = convective
    base, thickness = field.slab
    thick = []
    for spread in (1.0, 0.6, 0.3):
        density, heights = _columns(field, _cube(0.3, 0.0, tables, shelf=spread))
        band = (heights - base) / thickness
        sheet = band > 0.2
        cloudy = band[sheet][(density[:, sheet] > 0).any(axis=0)]
        assert cloudy.min() > shelf_underside(spread)
        thick.append((density[:, sheet] > 0).mean())
    assert thick[0] > thick[1] > thick[2] > 0.0
    assert shelf_underside(0.0) == SHELF_EDGE


def test_the_shelf_is_smooth_its_fine_octaves_their_mean(convective):
    field, tables = convective
    base, thickness = field.slab
    rng = np.random.default_rng(5)
    d = rng.normal(size=(8000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    points = d * (R + base + 0.8 * thickness)  # over the cumulus: the shelf
    cube = _cube(0.3, 0.0, tables, shelf=1.0)
    full = density_on_sphere(_noise(7), cube, field, points, R, octaves=4)
    coarse = density_on_sphere(_noise(7), cube, field, points, R, octaves=2)
    assert (full > 0).mean() > 0.5
    assert np.array_equal(full, coarse)


def test_the_shelf_top_fades_in_density_not_cut(convective):
    field, tables = convective
    base, thickness = field.slab
    up = np.array([0.0, 0.6, 0.8])
    h = np.linspace(SHELF_BAND[1], 0.999, 400)
    points = up * (R + base + h * thickness)[:, None]
    cube = _cube(0.3, 0.0, tables, shelf=1.0)
    density = density_on_sphere(_noise(7), cube, field, points, R)
    # Along a column through its top, the density falls smoothly to 0:
    # no step from dense to clear
    assert density[0] > 0.5 and density[-1] < 0.01
    assert np.abs(np.diff(density)).max() < 0.05


def test_the_shelf_shift_is_resolved_and_packed(convective):
    field, _ = convective
    assert field.shelf_shift != 0.0
    assert pack_layer_params(field, 0.03)[10, 1] == pytest.approx(field.shelf_shift)
    cumulus = PRESETS[CloudType.CUMULUS].field
    volume = _noise(7)
    resolved = with_threshold(
        cumulus, sphere_column_peaks(volume, cumulus, R), fbm_sigma(volume, cumulus)
    )
    assert with_shelf(resolved, volume, R) is resolved


def test_over_a_deep_column_the_shelf_changes_nothing_below_its_band(convective):
    field, tables = convective
    base, thickness = field.slab
    deep, heights = _columns(field, _cube(0.6, 1.0, tables))
    shelf, _ = _columns(field, _cube(0.6, 1.0, tables, shelf=1.0))
    band = (heights - base) / thickness
    assert np.array_equal(deep[:, band < SHELF_BAND[0]], shelf[:, band < SHELF_BAND[0]])
    assert np.all(shelf >= deep)


@pytest.mark.parametrize("spread", [0.4, 1.0])
def test_the_shelf_early_out_is_exact(convective, spread):
    field, tables = convective
    cube = _cube(0.4, 0.3, tables, shelf=spread)
    rng = np.random.default_rng(3)
    d = rng.normal(size=(20000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    base, thickness = field.slab
    h = rng.uniform(0.5, 1.0, len(d))
    points = d * (R + base + h * thickness)[:, None]
    full = density_on_sphere(_noise(7), cube, field, points, R)
    assert (full > 0).mean() > 0.05
    assert np.allclose(
        density_on_sphere(_noise(7), cube, field, points, R, early_out=True), full
    )


def test_the_cell_octave_grows_with_depth_about_its_mean(convective):
    field, _ = convective
    volume = _noise(7)
    coords = np.random.default_rng(2).uniform(0, 64, (20000, 3))
    assert np.all(cell_shift(volume, field, coords, 0.0) == 0.0)
    deep = cell_shift(volume, field, coords, 1.0)
    assert abs(deep.mean()) < 0.01
    assert deep.std() > 0.5 * field.cell_weight * volume.std()
    assert np.allclose(cell_shift(volume, field, coords, 0.5), 0.5 * deep)


# ── Coverage by depth ────────────────────────────────────────────────────────


@pytest.mark.parametrize("depth", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("coverage", [0.3, 0.6])
def test_a_requested_coverage_is_the_coverage_you_get_at_any_depth(
    convective, depth, coverage
):
    field, tables = convective
    density, _ = _columns(field, _cube(coverage, depth, tables), levels=192)
    measured = (density > 0.0).any(axis=1).mean()
    assert measured == pytest.approx(coverage, abs=0.04)


def test_a_deeper_column_needs_less_shift(convective):
    _, tables = convective
    tables = tables[: len(DEPTHS)]
    shifts = depth_shift(np.full(5, 0.4), np.array(DEPTHS), tables)
    assert np.all(np.diff(shifts) < 0.0)
    # And the tables are read exactly at their depths
    for depth, table in zip(DEPTHS, tables):
        expected = np.interp(0.4, np.linspace(0, 1, len(table)), table)
        assert depth_shift(0.4, depth, tables) == pytest.approx(expected)


# ── The placeholder maps ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def genera():
    return GeneraSpec(WeatherSpec(PLANETS["earthlike"]), focus=(20.0, 30.0))


def _at(genera, feature: Feature, east=0.0, north=0.0):
    """The maps' direction at a feature's centre moved east and north."""
    frame = TangentFrame.at(*genera.focus)
    radius = genera.weather.planet.radius
    x, y = feature.east + east, feature.north + north
    angle = math.hypot(x, y) / radius
    heading = (x * frame.east + y * frame.north) / max(math.hypot(x, y), 1e-9)
    return math.cos(angle) * frame.up + math.sin(angle) * heading


def _sample(genera, direction):
    from world_builder.clouds.weather import sample_cube

    return sample_cube(genus_planes(genera), np.asarray(direction)[None])[0]


def test_the_features_are_where_they_are_placed(genera):
    for feature in genera.features:
        convective, depth, stratiform, shelf = _sample(genera, _at(genera, feature))
        if feature.genus == "cumulonimbus":
            assert depth == pytest.approx(1.0, abs=0.05)
            assert convective >= feature.coverage - 0.05
            # Its anvil's shelf, spread past the tower, gone by 3 radii
            assert shelf == pytest.approx(1.0)
            middle = _sample(genera, _at(genera, feature, east=2.0 * feature.radius))
            assert middle[3] == pytest.approx(0.5, abs=0.05)
            beyond = _at(genera, feature, north=-3.3 * feature.radius)
            assert _sample(genera, beyond)[3] == 0.0
            # Its anvil's overhang, deep, then the congestus around it
            # (past the core's edge the depth stays high, the coverage
            # falls: the map's texels are 3.7 km, a radius or less)
            overhang = _sample(genera, _at(genera, feature, east=1.1 * feature.radius))
            assert overhang[1] > 0.75
            assert overhang[0] < convective - 0.2
            ring = _sample(genera, _at(genera, feature, east=1.7 * feature.radius))
            assert 0.3 < ring[1] < 0.7
        elif feature.genus == "congestus":
            assert depth == pytest.approx(0.5, abs=0.05)
        else:
            assert stratiform == pytest.approx(feature.coverage, abs=0.05)


def test_away_from_the_features_it_is_the_weather(genera):
    from world_builder.clouds.weather import coverage_map, sample_cube

    dirs = cube_directions(128)
    planes = genus_planes(genera, 128)
    far = (dirs @ TangentFrame.at(*genera.focus).up) < 0.0  # the other side
    assert np.all(planes[..., 1][far] == 0.0)
    assert np.all(planes[..., 2][far] == 0.0)
    assert np.all(planes[..., 3][far] == 0.0)
    weather = sample_cube(coverage_map(genera.weather), dirs[far])
    assert np.allclose(planes[..., 0][far], weather, atol=1e-6)


def test_the_families_cubes(genera, convective):
    field, tables = convective
    cube = genus_cube(genera, CONVECTIVE, tuple(map(tuple, tables)), 64)
    planes = genus_planes(genera, 64)
    assert cube.shape == (6, 64, 64, 4)
    assert np.array_equal(cube[..., 0], planes[..., 0])
    assert np.array_equal(cube[..., 2], planes[..., 1])
    assert np.array_equal(cube[..., 3], planes[..., 3])
    stratus = PRESETS[CloudType.STRATUS].field
    volume = _noise(7)
    peaks = sphere_column_peaks(volume, stratus, R)
    resolved = with_threshold(stratus, peaks, fbm_sigma(volume, stratus))
    table = shift_table(peaks, resolved.threshold[0])
    flat = genus_cube(genera, STRATIFORM, tuple(table), 64)
    assert flat.shape == (6, 64, 64, 3)
    assert np.array_equal(flat[..., 0], planes[..., 2])
    assert np.all(flat[..., 2] == 0.0)
    with pytest.raises(ValueError):
        genus_cube(genera, "nimbus", tuple(table), 64)


# ── The shader and Python agree ──────────────────────────────────────────────


def test_the_shader_shares_the_cell_octave_and_the_anvil():
    text = (SHADERS / "cloud_field.glsl").read_text()
    assert float(re.search(r"CELL_SCALE = ([\d.]+);", text).group(1)) == CELL_SCALE
    offset = re.search(r"CELL_OFFSET = vec3\(([^)]*)\);", text).group(1)
    assert tuple(float(v) for v in offset.split(",")) == CELL_OFFSET
    lift = text[text.index("float thresholdLift(Field f, float h, float depth)") :]
    numbers = re.search(
        r"smoothstep\(([\d.]+), ([\d.]+), depth\)"
        r" \* smoothstep\(([\d.]+), ([\d.]+), local\)",
        lift,
    )
    assert tuple(map(float, numbers.groups())) == (*ANVIL_DEPTH, *ANVIL_BAND)
    band = re.search(r"SHELF_BAND = vec4\(([^)]*)\);", text).group(1)
    assert tuple(float(v) for v in band.split(",")) == SHELF_BAND
    for name, value in (
        ("SHELF_EDGE", SHELF_EDGE),
        ("SHELF_CLEAR", SHELF_CLEAR),
    ):
        assert float(re.search(rf"{name} = ([\d.]+);", text).group(1)) == value


# ── Placement ────────────────────────────────────────────────────────────────


def _tile_spec(genera, convective, family=CONVECTIVE):
    from world_builder.clouds.sphere import ShellSpec, SphereCloudSpec
    from world_builder.terrain.cubesphere import grid_for

    field, tables = convective
    grid = grid_for(R)
    base = round(np.log2(8000.0 / grid.chunk_size(0)))
    shells = (ShellSpec(base, (70.0, 140.0), 2.0, 4),)
    return SphereCloudSpec(
        R,
        field,
        7,
        genera.weather,
        tuple(tables[0]),
        shells,
        genera=(genera, family, tuple(map(tuple, tables))),
        deep_radius=PRESETS[CloudType.CONVECTIVE].deep_radius,
    ), base


def _tile_under(direction, level):
    from world_builder.terrain.cubesphere import grid_for

    face, i, j, _, _ = grid_for(R).locate(np.asarray(direction)[None], level)
    return (level, int(face[0]), int(i[0]), int(j[0]))


def test_a_cumulonimbus_tile_holds_tall_clouds_drawn_by_larger_billboards(
    genera, convective
):
    from world_builder.clouds.sphere import generate_cloud_tile

    spec, level = _tile_spec(genera, convective)
    field = spec.field
    cb = next(f for f in genera.features if f.genus == "cumulonimbus")
    deep = generate_cloud_tile(spec, _tile_under(_at(genera, cb), level))
    far = generate_cloud_tile(spec, _tile_under(-_at(genera, cb), level))

    def heights(tile):
        return np.linalg.norm(tile.particles[:, :3] + tile.centre, axis=1) - R

    base = field.slab[0]
    assert len(deep.particles) > 0 and len(far.particles) > 0
    assert heights(deep).max() > base + 5000.0
    assert heights(far).max() <= base + 1000.0 + 140.0
    # Larger where deep, as the cube root of their fewer number
    assert deep.particles[:, 3].mean() > 1.5 * far.particles[:, 3].mean()
    assert far.particles[:, 3].max() <= 140.0


def test_an_anvil_shelf_tile_holds_large_billboards_over_shallow_clouds(
    genera, convective
):
    from world_builder.clouds.sphere import generate_cloud_tile
    from world_builder.clouds.weather import sample_cube

    spec, level = _tile_spec(genera, convective)
    base, thickness = spec.field.slab
    cb = next(f for f in genera.features if f.genus == "cumulonimbus")
    # Past the congestus ring, under the shelf
    where = _at(genera, cb, north=-2.4 * cb.radius)
    assert sample_cube(genus_planes(genera), where[None])[0, 1] < 0.05
    tile = generate_cloud_tile(spec, _tile_under(where, level))
    points = tile.particles[:, :3] + tile.centre
    h = (np.linalg.norm(points, axis=1) - R - base) / thickness
    high = h > SHELF_BAND[0]
    assert high.sum() > 0
    assert tile.particles[high, 3].min() > 140.0  # as large as a tower's


# ── A planet's clouds by genus ───────────────────────────────────────────────


@pytest.fixture
def executor():
    from world_builder.terrain.workers import ChunkExecutor

    made = ChunkExecutor(backend="inline", inline_budget=64)
    yield made
    made.shutdown()


def test_the_genera_make_two_volumes_driven_as_one(showbase, executor, genera):
    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.planet import PlanetClouds
    from world_builder.clouds.sphere import CloudVolume

    clouds = PlanetClouds(
        showbase.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(shells=2, renderer="march"),
        genera=genera,
    )
    try:
        # Billboards whatever the renderer: the raymarch has no depth
        assert isinstance(clouds.volume, CloudVolume)
        assert isinstance(clouds.stratus, CloudVolume)
        assert clouds.volume.field.min_top < 1.0
        assert clouds.stratus.field.slab[0] == genera.stratus_altitude
        clouds.set_sun((0.0, 0.0, 3.0))
        assert tuple(clouds.stratus._sun[0]) == (0.0, 0.0, 1.0)
        # The decks: the farther first
        up = np.array([0.0, 0.0, 1.0])
        clouds._order_decks(up * (R + 100.0))  # under both
        assert clouds._deck_order == (clouds.volume, clouds.stratus)
        clouds._order_decks(up * (3.0 * R))  # outside both
        assert clouds._deck_order == (clouds.stratus, clouds.volume)
    finally:
        clouds.remove()


def test_the_stratus_must_stay_under_the_convective_clouds(genera):
    from world_builder.clouds.planet import check_layers

    check_layers(CloudType.CUMULUS, 2200.0, None, None, genera)
    high = replace(genera, stratus_altitude=2000.0)
    with pytest.raises(ValueError, match="under the convective"):
        check_layers(CloudType.CUMULUS, 2200.0, None, None, high)
    # And the cirrus over the cumulonimbus's cap
    from world_builder.clouds.cirrus import CirrusSpec

    check_layers(CloudType.CUMULUS, 2200.0, CirrusSpec(), None, genera)
    with pytest.raises(ValueError, match="above the clouds"):
        check_layers(CloudType.CUMULUS, 2200.0, CirrusSpec(base=9000.0), None, genera)


def test_inside_a_cloud_billboards_grow_fewer_as_the_cube(genera, convective):
    from world_builder.clouds.sphere import (
        INTERIOR_GROWTH,
        _noise,
        generate_cloud_tile,
        interior_levels,
    )

    spec, level = _tile_spec(genera, convective)
    cb = next(f for f in genera.features if f.genus == "cumulonimbus")
    key = _tile_under(_at(genera, cb), level)
    plain = generate_cloud_tile(spec, key)
    grown = generate_cloud_tile(replace(spec, interior_growth=True), key)
    assert len(grown.particles) < 0.95 * len(plain.particles)
    assert grown.particles[:, 3].max() > plain.particles[:, 3].max()
    # Only where the cloud reaches every way at the grown radius
    weather = genus_cube(genera, CONVECTIVE, spec.genera[2])
    points = plain.particles[:, :3] + plain.centre
    radii = np.full(len(points), 140.0)
    levels = interior_levels(_noise(7), weather, spec.field, points, radii, R)
    assert levels.max() >= 1 and (levels == 0).any()
    clear = np.array([[0.0, 0.0, R + 2200.0 + 500.0]])  # over the pole: cumulus
    far = generate_cloud_tile(
        replace(spec, interior_growth=True),
        _tile_under(clear[0] / np.linalg.norm(clear[0]), level),
    )
    assert far.particles[:, 3].max() <= 140.0 * INTERIOR_GROWTH
