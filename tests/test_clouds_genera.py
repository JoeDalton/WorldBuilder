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
from panda3d.core import ShaderAttrib

from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.field_setup import _noise
from world_builder.clouds.genera import (
    CONVECTIVE,
    DEPTHS,
    STRATUS_FAMILY,
    Feature,
    GeneraSpec,
    coverage_cube,
    depth_shift,
    depth_shift_tables,
    genus_cube,
    genus_planes,
)
from world_builder.clouds.noise import (
    ANVIL_BAND,
    ANVIL_DEPTH,
    ANVIL_SMOOTH_BAND,
    CELL_OFFSET,
    CELL_SCALE,
    GENUS_VEC4S,
    LAYER_VEC4S,
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
from world_builder.clouds.weather import WeatherSpec
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
    # Below it, and below where it smooths the tower first, nothing changes
    low = band < ANVIL_SMOOTH_BAND[0] * 0.9
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
    h = np.linspace(SHELF_BAND[1], 0.999, 400)
    cube = _cube(0.3, 0.0, tables, shelf=1.0)
    # The densest of a few hundred columns at the bottom of the fade (the
    # shelf's noise is low-frequency: noise.ANVIL_SCALE)
    rng = np.random.default_rng(4)
    ups = rng.normal(size=(300, 3))
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    bottom = density_on_sphere(
        _noise(7), cube, field, ups * (R + base + h[0] * thickness), R
    )
    up = ups[np.argmax(bottom)]
    points = up * (R + base + h * thickness)[:, None]
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
    # About its mean: within 2 % of its weight (a random sample's mean)
    assert abs(deep.mean()) < 0.02 * field.cell_weight
    assert deep.std() > 0.5 * field.cell_weight * volume.std()
    assert np.allclose(
        cell_shift(volume, field, coords, 0.5), 0.5**field.cell_power * deep
    )


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
    # The features alone: the weather's own cover there (overcast at the
    # fixture's focus since the background's contrast) would hide their shapes
    genera = replace(genera, background=False)
    for feature in genera.features:
        convective, depth, stratiform, shelf = _sample(genera, _at(genera, feature))
        if feature.genus == "cumulonimbus":
            assert depth == pytest.approx(1.0, abs=0.05)
            assert convective >= feature.coverage - 0.05
            # Its anvil's shelf, spread past the tower
            assert shelf == pytest.approx(1.0)
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


def test_in_calm_air_the_anvil_is_round(genera):
    calm = replace(genera, wind=(0.0, 0.0))
    for feature in calm.features:
        if feature.genus != "cumulonimbus":
            continue
        for east, north in ((2.0, 0.0), (0.0, -2.0), (-2.0, 0.0)):
            at = _at(calm, feature, east * feature.radius, north * feature.radius)
            assert _sample(calm, at)[3] == pytest.approx(0.5, abs=0.05)
        beyond = _at(calm, feature, north=-3.3 * feature.radius)
        assert _sample(calm, beyond)[3] == 0.0


def test_the_anvil_streams_downwind(genera):
    from world_builder.clouds.genera import ANVIL_DOWNWIND

    windy = replace(genera, wind=(0.0, 1.0))  # the fastest jet, northward
    cb = next(f for f in windy.features if f.genus == "cumulonimbus")
    reach = 3.0 * (1.0 + ANVIL_DOWNWIND)

    def spread(east, north):
        return _sample(windy, _at(windy, cb, east * cb.radius, north * cb.radius))[3]

    # Far downwind, not upwind nor across at the calm reach's half
    assert spread(0.0, 0.8 * reach) > 0.0
    assert spread(0.0, 0.5 * reach) > 0.5
    assert spread(0.0, -2.0) < 0.05
    assert spread(2.0, 0.0) < spread(0.0, 2.0)
    assert spread(0.0, 1.05 * reach) == 0.0


def test_the_upper_wind_is_the_jets():
    """With no systems' swirl, the upper wind on a jet's meandering axis
    blows along it at the jet's speed (the fastest jet's 1)."""
    from world_builder.clouds.genera import upper_wind_at
    from world_builder.clouds.synoptic.jets import jet_lat, meander_slope
    from world_builder.clouds.synoptic.state import synoptic_state
    from world_builder.terrain.planet import PlanetSpec

    earth = PlanetSpec("earth", radius=6.371e6, day_length_h=23.934)
    weather = replace(WeatherSpec(earth), jet_vortex=0.0)
    state = synoptic_state(weather)
    for jet in [j for j in state.jets if j.hemisphere == 1]:
        lon = 10.0
        lat = float(jet_lat(jet, np.array([lon]), state.t)[0])
        wind = upper_wind_at(weather, TangentFrame.at(lat, lon).up)
        slope = float(meander_slope(jet, np.array([lon]), state.t)[0])
        heading = np.degrees(np.arctan2(slope, np.cos(np.radians(lat))))
        assert np.hypot(*wind) == pytest.approx(jet.speed / 30.0, abs=0.05)
        assert np.degrees(np.arctan2(wind[1], wind[0])) == pytest.approx(
            heading, abs=5.0
        )


def test_away_from_the_features_it_is_the_weather(genera):
    from world_builder.clouds.weather import (
        SYNOPTIC_CHANNELS,
        coverage_map,
        sample_cube,
        synoptic_cube,
    )

    dirs = cube_directions(128)
    planes = genus_planes(genera, 128)
    far = (dirs @ TangentFrame.at(*genera.focus).up) < 0.0  # the other side
    weather = sample_cube(coverage_map(genera.weather), dirs[far])
    assert np.allclose(planes[..., 0][far], weather, atol=1e-6)
    cube = synoptic_cube(genera.weather)
    at = sample_cube(cube, dirs[far])
    channel = SYNOPTIC_CHANNELS.index
    assert np.allclose(planes[..., 1][far], at[..., channel("depth")], atol=1e-6)
    # The stratus's: the weather's stratus, stratocumulus and nimbostratus
    # (its family's thick end: genera.WEATHER_GENERA)
    low = np.maximum.reduce(
        [at[..., channel(n)] for n in ("stratus", "stratocumulus", "nimbostratus")]
    )
    assert np.allclose(planes[..., 2][far], low, atol=1e-6)
    assert np.allclose(planes[..., 3][far], at[..., channel("shelf")], atol=1e-6)
    # The weather now holds more than fair-weather cumulus there.
    assert planes[..., 2][far].max() > 0.3


def test_the_weather_cube_and_coverage_cube_a(genera, convective):
    """The primary genus's weather cube (coverage, shift, depth, spread),
    and cube A with the stratus's coverage in its channel 0 (its shift its
    own table's: genusShift)."""
    field, tables = convective
    cube = genus_cube(genera, CONVECTIVE, tuple(map(tuple, tables)), 64)
    planes = genus_planes(genera, 64)
    assert cube.shape == (6, 64, 64, 4)
    assert np.array_equal(cube[..., 0], planes[..., 0])
    assert np.array_equal(cube[..., 2], planes[..., 1])
    assert np.array_equal(cube[..., 3], planes[..., 3])
    covers = coverage_cube(genera, "A", 64)
    assert covers.shape == (6, 64, 64, 4)
    assert np.array_equal(covers[..., 0], planes[..., 2])
    # The stratocumulus none (no feature), 2 the stratus family's
    # nimbostratus share (noise.FAMILY_SHARE_CHANNEL), 3 none
    assert not covers[..., 1].any() and not covers[..., 3].any()
    assert covers[..., 2].any()
    assert not coverage_cube(genera, "B", 64).any()
    with pytest.raises(ValueError):
        genus_cube(genera, "stratus", tuple(map(tuple, tables)), 64)


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


# ── A planet's clouds by genus ───────────────────────────────────────────────


@pytest.fixture
def executor():
    from world_builder.terrain.workers import ChunkExecutor

    made = ChunkExecutor(backend="inline", inline_budget=64)
    yield made
    made.shutdown()


def test_the_raymarch_marches_both_genera_and_casts_one_summed_shadow(
    showbase, executor, genera
):
    """With genera a raymarch is one volume of two genera (no stratus
    volume): each genus its own field and weather (the stratus's from cube
    A and its shift table), one shadow of both (one caster, its layer genus
    0's far level, the sky bands, the union's shade the product of the
    genera's); every genus has all the detail levels asked for (the
    default's stratocumulus, no feature of its own, is not marched)."""
    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.march import CloudMarch
    from world_builder.clouds.planet import PlanetClouds

    assert "stratocumulus" in genera.genera
    clouds = PlanetClouds(
        showbase.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(octave_bands=6),
        genera=genera,
    )
    try:
        march = clouds.volume
        assert isinstance(march, CloudMarch)
        assert march.genera_count == 2
        assert clouds.shadow_caster() is march.shadow_caster()
        convective, stratiform = (f.field for f in march._genera)
        assert convective.min_top < 1.0
        # The stratus's family slab, under its base (genera.STRATUS_FAMILY)
        assert stratiform.slab == (
            genera.stratus_altitude - STRATUS_FAMILY[0],
            STRATUS_FAMILY[1],
        )
        # Each genus its 6 bands and its far level (DETAIL_LEVELS slots each)
        assert march._layers_used == 7
        assert march._layer_count[0] == march._layers_used
        assert "uFamilies" not in march._inputs
        for _ in range(400):  # the set-up, in the pool
            clouds.update((0.0, 0.0, R + 1000.0))
            if march.weather is not None:
                break
        stride = march._layers_used
        caster = clouds.shadow_caster()
        # Genus 0's far level (genus g's is g * DETAIL_LEVELS past it)
        assert caster.shadow_layer == stride - 1
        stratus, convective = (f.field.slab for f in reversed(march._genera))
        assert caster.sky_bands == [stratus, convective]  # disjoint: two bands
        assert clouds.top_radius == R + sum(convective)  # the union's top
        # far_shadow in row 13's w: the stratus under the towers takes their
        # depth beyond its window, the convective clouds none
        rows = np.array([tuple(march._layer_params[i]) for i in range(32)])
        assert rows[LAYER_VEC4S, 3] == 0.0
        assert rows[GENUS_VEC4S + LAYER_VEC4S, 3] == 1.0
        # The union's shade is the product of the genera's (the sum of
        # their depths), under the towers and under the stratus
        points = np.array([[0.0, 0.0, R + 10.0], [0.0, 40e3, R + 10.0]])
        points = np.vstack([points, TangentFrame.at(-2.86, 0.0).up * (R + 10.0)])
        sun = np.array([0.0, 0.3, 1.0])
        shade = clouds.sun_visibility(points, sun, 8)
        both = np.ones(len(points))
        for genus in (0, 1):
            both *= march.genus_caster(genus).sun_visibility(points, sun, 8)
        np.testing.assert_allclose(shade, both)
        np.testing.assert_allclose(caster.sun_visibility(points, sun, 8), both)
        assert np.all((shade >= 0.0) & (shade <= 1.0))
        node = showbase.render.attach_new_node("bake")
        clouds.bind_shadow_inputs(node)
        attrib = node.get_attrib(ShaderAttrib)
        for name in ("uCloudShadowLayer", "uCloudGenera", "uSkyBands", "genusCoverA"):
            assert attrib.has_shader_input(name)
        assert march._shadow_inputs()["uCloudGenera"] == 2
        # The stratus's weather has no depth channel; the convective one has
        assert march._genera[0].weather.shape[-1] == 4
        assert march._genera[1].weather.shape[-1] == 3
        # The stratus's coverage from cube A's channel 0, its shift from it
        stratus = march.genus_caster(1)
        cover, channel, table = stratus.coverage_source
        assert channel == 0 and march.genus_caster(0).coverage_source is None
        np.testing.assert_array_equal(march._genera[1].weather[..., 0], cover[..., 0])
        for name in ("genusCoverA", "genusShift", "genusDeckLight"):
            assert attrib.get_shader_input(name).get_texture() == march._textures[name]
    finally:
        clouds.remove()


def test_the_march_takes_the_genera_that_hold_cloud(genera):
    """genus_specs(covered=True), the march's: the primary genus always,
    another only where a feature is its (an empty genus would cost the
    march as a full one)."""
    from world_builder.clouds.genera import (
        genus_specs,
        has_coverage,
        showcase_features,
    )

    assert [s[0] for s in genus_specs(genera)] == list(genera.genera)
    assert [s[0] for s in genus_specs(genera, covered=True)] == [
        "convective",
        "stratus",
    ]
    # The default's genera with no feature (normal scenes pay nothing)
    for name in ("stratocumulus", "nimbostratus", "altostratus", "altocumulus"):
        assert name in genera.genera and not has_coverage(genera, name)
    showcase = replace(genera, features=showcase_features(), background=False)
    assert [s[0] for s in genus_specs(showcase, covered=True)] == list(genera.genera)
    # No feature: the weather's own stratus, stratocumulus and nimbostratus
    # still make the stratus's family sheet (genera.WEATHER_GENERA); without
    # the weather, nothing
    alone = replace(genera, features=())
    assert [s[0] for s in genus_specs(alone, covered=True)] == [
        "convective",
        "stratus",
    ]
    bare = replace(alone, background=False)
    assert [s[0] for s in genus_specs(bare, covered=True)] == ["convective"]


def test_the_genera_may_overlap(genera):
    from world_builder.clouds.planet import check_layers

    check_layers(CloudType.CUMULUS, 2200.0, False, None, genera)
    # The stratus inside the convective clouds' slab: one band of sky
    check_layers(
        CloudType.CUMULUS, 2200.0, False, None, replace(genera, stratus_altitude=2000.0)
    )


def test_the_genera_must_fit_the_march(genera, monkeypatch):
    from world_builder.clouds.genera import GENUS_PRESETS, GenusPreset
    from world_builder.clouds.march import MAX_SKY_BANDS
    from world_builder.clouds.noise import MAX_GENERA
    from world_builder.clouds.planet import check_layers

    many = replace(genera, genera=("convective",) + ("stratus",) * MAX_GENERA)
    with pytest.raises(ValueError, match="genera: the march takes"):
        check_layers(CloudType.CUMULUS, 2200.0, False, None, many)
    # Disjoint thin slabs under the convective clouds, a band of sky each
    names = []
    for k in range(MAX_SKY_BANDS):
        name = f"thin{k}"
        slab = (100.0 + 400.0 * k, 100.0)
        monkeypatch.setitem(
            GENUS_PRESETS, name, GenusPreset(CloudType.STRATUS, ("A", 0), slab=slab)
        )
        names.append(name)
    fits = replace(genera, genera=("convective", *names[:-1]))
    check_layers(CloudType.CUMULUS, 2200.0, False, None, fits)  # 4 bands
    over = replace(genera, genera=("convective", *names))
    with pytest.raises(ValueError, match="bands of sky"):
        check_layers(CloudType.CUMULUS, 2200.0, False, None, over)


def test_the_cirrus_must_clear_the_union_of_the_genera(genera, monkeypatch):
    """The weather's cirrus (``cirrus=True``) is the high layer's: its slab
    must clear the genera's union and stay in the air, even with no high
    genus listed."""
    from world_builder.clouds.genera import GENUS_PRESETS, GenusPreset
    from world_builder.clouds.planet import check_layers

    check_layers(CloudType.CUMULUS, 2200.0, True, None, genera)
    # A genus topping the convective clouds (10.5 km) sets the union's top
    monkeypatch.setitem(
        GENUS_PRESETS,
        "tall",
        GenusPreset(CloudType.STRATUS, ("A", 0), slab=(500.0, 10800.0)),
    )
    tall = replace(genera, genera=("convective", "tall"), high=())
    check_layers(CloudType.CUMULUS, 2200.0, False, None, tall)  # no high layer
    with pytest.raises(
        ValueError, match=r"high layer's base .* clouds' top \(11300 m\)"
    ):
        check_layers(CloudType.CUMULUS, 2200.0, True, None, tall)
    # The air's top bounds the cirrus
    with pytest.raises(ValueError, match="high layer's top .* inside the air"):
        check_layers(CloudType.CUMULUS, 2200.0, True, 11500.0, replace(genera, high=()))
