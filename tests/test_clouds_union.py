"""
The union march's geometry and genus model (clouds.march.sky_bands,
band_segments; noise.GenusField, pack_genus_params; genera.GENUS_PRESETS):
genera whose slabs may overlap, marched as one.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.genera import GENUS_PRESETS
from world_builder.clouds.march import (
    band_masks,
    band_segments,
    sky_bands,
    slab_segments,
)
from world_builder.clouds.noise import (
    COVERAGE_SOURCES,
    DETAIL_LEVELS,
    GENUS_VEC4S,
    MAX_GENERA,
    GenusField,
    pack_genus_params,
    pack_layer_params,
)

R = 600e3


# ── Sky bands ────────────────────────────────────────────────────────────────


def test_disjoint_slabs_stay_apart_sorted_by_base():
    assert sky_bands([(2200.0, 8300.0), (900.0, 500.0)]) == [
        (900.0, 500.0),
        (2200.0, 8300.0),
    ]


def test_overlapping_slabs_merge():
    assert sky_bands([(2200.0, 8300.0), (3000.0, 1000.0), (9000.0, 3000.0)]) == [
        (2200.0, 9800.0)
    ]


def test_nested_and_touching_slabs_merge():
    assert sky_bands([(1000.0, 1000.0), (2000.0, 500.0)]) == [(1000.0, 1500.0)]
    assert sky_bands([(1000.0, 5000.0), (2000.0, 500.0)]) == [(1000.0, 5000.0)]


def test_a_bridging_slab_splits_its_band_at_the_gap():
    """A slab joining two groups (the nimbostratus between the low sheets
    and the convective clouds) splits the band at the gap's middle, each
    part holding its side's genera and the bridge; the thickest slab never
    bridges (the convective one holding mid-level sheets)."""
    slabs = [(2200.0, 8300.0), (900.0, 500.0), (800.0, 1000.0), (600.0, 4400.0)]
    bands = sky_bands(slabs)
    assert bands == [(600.0, 1400.0), (2000.0, 8500.0)]
    assert band_masks(slabs, bands) == [0b1110, 0b1001]
    nested = [(2200.0, 8300.0), (3500.0, 1000.0), (4700.0, 1300.0)]
    assert sky_bands(nested) == [(2200.0, 8300.0)]
    assert band_masks(nested, sky_bands(nested)) == [0b111]


def test_more_than_four_bands_is_refused():
    slabs = [(1000.0 * k, 100.0) for k in range(1, 6)]
    with pytest.raises(ValueError):
        sky_bands(slabs)


@pytest.mark.parametrize("height, sign", [(100.0, 1.0), (20e3, -1.0), (1500.0, 1.0)])
def test_band_segments_are_the_slabs_own_on_disjoint_slabs(height, sign):
    slabs = [(2200.0, 8300.0), (900.0, 500.0)]
    up = np.array([0.0, 0.6, 0.8])
    tilt = np.array([0.8, 0.0, -0.6])
    for direction in (sign * up, sign * up * 0.2 + tilt):
        direction = direction / np.linalg.norm(direction)
        origin = up * (R + height)
        old = sorted(
            (float(t0), float(t1))
            for slab in slabs
            for t0, t1 in slab_segments(origin, direction, R, slab)
            if t1 > t0
        )
        new = [(t0, t1) for t0, t1, _ in band_segments(origin, direction, R, slabs)]
        np.testing.assert_allclose(new, old)


def test_a_nested_slab_shares_its_bands_segments():
    """A genus inside another's slab adds no stretch: the ray crosses the
    outer slab's band once (the genus loop finds both in it)."""
    up = np.array([0.0, 0.0, 1.0])
    slabs = [(2200.0, 8300.0), (900.0, 500.0), (3000.0, 1000.0)]
    segments = band_segments(up * (R + 100.0), up, R, slabs)
    assert [band for *_, band in segments] == [0, 1]
    assert segments[1][:2] == pytest.approx((2100.0, 10400.0))


def test_band_segments_name_their_band():
    up = np.array([0.0, 0.0, 1.0])
    segments = band_segments(
        up * (R + 100.0), up, R, [(2200.0, 8300.0), (900.0, 500.0)]
    )
    # Looking up from under both: the stratus's band (0, the lower) first
    assert [band for *_, band in segments] == [0, 1]


# ── The genus model ──────────────────────────────────────────────────────────


def _resolved(kind=CloudType.STRATUS):
    from dataclasses import replace

    return replace(PRESETS[kind].field, threshold=(0.45, 0.55))


def test_a_genus_packs_its_field_rows_then_its_coverage_row():
    field = _resolved()
    genus = GenusField(
        "stratus", field, PRESETS[CloudType.STRATUS].optics, ("A", 2), 1.0
    )
    rows = pack_genus_params(genus, 0.02, octave_mean=0.5, octaves=4)
    assert rows.shape == (GENUS_VEC4S, 4)
    expected = pack_layer_params(
        field, 0.02, octave_mean=0.5, optics=genus.optics, octaves=4
    )
    np.testing.assert_array_equal(rows[: len(expected)], expected)
    assert tuple(rows[13]) == (COVERAGE_SOURCES["A"], 2.0, 0.0, 1.0)
    assert not rows[14:].any()


def test_the_primary_genus_reads_the_weather_cube():
    genus = GenusField(
        "convective",
        _resolved(CloudType.CONVECTIVE),
        PRESETS[CloudType.CONVECTIVE].optics,
        ("weather", 0),
        0.0,
    )
    rows = pack_genus_params(genus, 0.02, shift_row=0)
    assert rows[13, 0] == COVERAGE_SOURCES["weather"] == 0


def test_the_uniforms_fit_the_gl_minimum():
    """Every genus's rows and the deck's, the per-(genus, level) threshold
    offsets (4 a vec4) and a few vec4s of detail bands: under GL 3.3's
    guaranteed 1024 components (256 vec4) of a stage's uniforms."""
    params = (MAX_GENERA + 1) * GENUS_VEC4S
    offsets = MAX_GENERA * DETAIL_LEVELS // 4
    assert params + offsets + 16 <= 256


def test_the_presets_name_the_two_genera_of_phase_3():
    convective = GENUS_PRESETS["convective"]
    stratus = GENUS_PRESETS["stratus"]
    assert convective.cloud_type is CloudType.CONVECTIVE
    assert convective.coverage == ("weather", 0) and convective.far_shadow == 0.0
    assert stratus.cloud_type is CloudType.STRATUS
    assert stratus.coverage == ("A", 0) and stratus.far_shadow == 1.0


# ── Coverage cubes, the shift texture, the deck light array ─────────────────


@pytest.fixture(scope="module")
def genera_spec():
    from world_builder.clouds.genera import GeneraSpec
    from world_builder.clouds.weather import WeatherSpec
    from world_builder.terrain.planet import PlanetSpec

    return GeneraSpec(WeatherSpec(PlanetSpec(name="clouds", radius=R)))


def test_cube_a_holds_the_stratus_coverage_in_its_channel(genera_spec):
    from world_builder.clouds.genera import coverage_cube, genus_planes

    cube = coverage_cube(genera_spec, "A")
    planes = genus_planes(genera_spec)
    assert cube.shape == planes.shape[:-1] + (4,) and cube.dtype == np.float32
    np.testing.assert_array_equal(cube[..., 0], planes[..., 2])
    # The stratocumulus's and the altostratus's: their features' alone
    # (none); channel 2 the stratus family's nimbostratus share
    assert not cube[..., 1].any() and not cube[..., 3].any()


def test_a_genus_coverage_turns_into_its_shift_through_its_table():
    """genus_weather_at: the coverage read off its channel, the shift its
    table's at that coverage; the coverage that shift gives is the one asked
    for (within 5 %)."""
    from dataclasses import replace

    from world_builder.clouds.field_setup import _noise
    from world_builder.clouds.genera import genus_weather_at
    from world_builder.clouds.noise import (
        density_on_sphere,
        fbm_sigma,
        sphere_column_peaks,
        with_threshold,
    )
    from world_builder.clouds.weather import shift_table

    raw = replace(PRESETS[CloudType.STRATUS].field, slab=(900.0, 500.0))
    volume = _noise(7)
    peaks = sphere_column_peaks(volume, raw, R)
    field = with_threshold(raw, peaks, fbm_sigma(volume, raw))
    table = shift_table(peaks, field.threshold[0])
    for coverage in (0.3, 0.7):
        cube = np.zeros((6, 4, 4, 4), dtype=np.float32)
        cube[..., 1] = coverage  # channel 1
        rng = np.random.default_rng(2)
        dirs = rng.normal(size=(4000, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        got, shift = genus_weather_at(cube, 1, table, dirs)
        np.testing.assert_allclose(got, coverage, atol=1e-6)
        weather = np.stack([got, shift], -1)
        weather = np.broadcast_to(weather[0], (6, 4, 4, 2)).astype(np.float32)
        heights = R + 900.0 + (np.arange(24) + 0.5) / 24 * 500.0
        points = dirs[:, None, :] * heights[None, :, None]
        cloudy = (density_on_sphere(volume, weather, field, points, R) > 0).any(1)
        assert abs(cloudy.mean() - coverage) < 0.05 * max(coverage, 0.1) + 0.02


def test_the_shift_texture_holds_one_table_a_row():
    from world_builder.clouds.weather import SHIFT_TABLE_SIZE, build_shift_texture

    tables = [
        np.linspace(-0.3, 0.2, SHIFT_TABLE_SIZE),
        np.linspace(0.1, 0.4, SHIFT_TABLE_SIZE),
    ]
    tex = build_shift_texture(tables)
    assert (tex.get_x_size(), tex.get_y_size()) == (SHIFT_TABLE_SIZE, MAX_GENERA)
    ram = np.frombuffer(tex.get_ram_image(), np.float32).reshape(MAX_GENERA, -1)
    np.testing.assert_allclose(ram[0], tables[0], rtol=1e-6)
    np.testing.assert_allclose(ram[1], tables[1], rtol=1e-6)
    assert not ram[2:].any()


def test_the_deck_light_array_holds_one_genus_a_page():
    from world_builder.clouds.noise import build_deck_light_array

    tables = [
        np.full((8, 16, 4), 0.25, np.float32),
        np.full((8, 16, 4), 0.75, np.float32),
    ]
    tex = build_deck_light_array(tables)
    assert (tex.get_x_size(), tex.get_y_size(), tex.get_z_size()) == (16, 8, 2)
    ram = np.frombuffer(tex.get_ram_image(), np.float32).reshape(2, 8, 16, 4)
    np.testing.assert_allclose(ram[0], 0.25)
    np.testing.assert_allclose(ram[1], 0.75)


# ── The shaders and Python agree ─────────────────────────────────────────────

SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"


def test_the_shaders_share_the_genus_and_band_sizes():
    from world_builder.clouds.march import MAX_SKY_BANDS
    from world_builder.clouds.noise import LAYER_VEC4S
    from world_builder.clouds.weather import SHIFT_TABLE_SIZE

    field = (SHADERS / "cloud_field.glsl").read_text()
    for name, value in (
        ("MAX_SKY_BANDS", MAX_SKY_BANDS),
        ("MAX_GENERA", MAX_GENERA),
        ("GENUS_VEC4S", GENUS_VEC4S),
        ("LAYER_VEC4S", LAYER_VEC4S),
        ("DETAIL_LEVELS", DETAIL_LEVELS),
        ("SHIFT_TABLE_SIZE", SHIFT_TABLE_SIZE),
    ):
        assert int(re.search(rf"const int {name} = (\d+);", field).group(1)) == value
    assert f"uniform vec4  uSkyBands[{MAX_SKY_BANDS}];" in field
    march = (SHADERS / "cloud_march.glsl").read_text()
    # Two segments a band, sorted; a genus loop per sample, one lighting path
    assert "const int MAX_SEGMENTS = 2 * MAX_SKY_BANDS;" in march
    assert "vec2 segments[MAX_SEGMENTS];" in march
    assert "weather = genusWeatherAt(g, p);" in march  # the genus loop
    assert "bool alone = (genera & (genera - 1)) == 0;" in march  # its fast path
    for gone in ("uFamilies", "otherWeather", "otherDeckLight", "cloudRadianceOther"):
        for name in ("cloud_march.glsl", "cloud_field.glsl", "cloud_shading.glsl"):
            assert gone not in (SHADERS / name).read_text(), (gone, name)


# ── A third genus, inside the convective slab (test-only) ───────────────────

#: A stratus-like genus 3-4 km up, inside the convective clouds' slab, its
#: coverage cube A's second channel (the stratus features' map).
MID = "stratus_mid"


@pytest.fixture
def mid_genus(monkeypatch):
    from world_builder.clouds.genera import GenusPreset

    monkeypatch.setitem(
        GENUS_PRESETS,
        MID,
        GenusPreset(CloudType.STRATUS, ("A", 1), far_shadow=1.0, slab=(3000.0, 1000.0)),
    )
    from world_builder.clouds.genera import DEFAULT_FEATURES, Feature, GeneraSpec
    from world_builder.clouds.weather import WeatherSpec
    from world_builder.terrain.planet import PlanetSpec

    spec = PlanetSpec(name="clouds", radius=R)
    features = DEFAULT_FEATURES + (Feature(20e3, -30e3, 10e3, MID, 0.8),)
    return GeneraSpec(
        WeatherSpec(spec, seed=11), features=features,
        genera=("convective", "stratus", MID),
    )  # fmt: skip


def test_a_third_genus_takes_its_slab_and_its_channel(mid_genus):
    from world_builder.clouds.genera import (
        coverage_cube,
        genus_coverage,
        genus_planes,
        genus_specs,
    )

    specs = genus_specs(mid_genus, 2200.0)
    assert [s[0] for s in specs] == ["convective", "stratus", MID]
    # The stratus: its family's slab, 300 m under its base (genera.STRATUS_FAMILY)
    assert specs[0][2] == 2200.0 and specs[1][2] == mid_genus.stratus_altitude - 300.0
    assert specs[2][2:] == (3000.0, 1000.0, ("A", 1))
    cube = coverage_cube(mid_genus, "A", 256)
    planes = genus_planes(mid_genus, 256)
    np.testing.assert_array_equal(cube[..., 0], planes[..., 2])
    # Its own feature's map, not the stratus's
    np.testing.assert_array_equal(cube[..., 1], genus_coverage(mid_genus, MID, 256))
    assert cube[..., 1].max() > 0.5 and not np.array_equal(cube[..., 1], cube[..., 0])
    # Channel 2 the stratus family's nimbostratus share, 3 none
    assert not cube[..., 3].any()


@pytest.fixture
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


def _pixels(texture):
    """A float texture's RAM image as (n, components) float32."""
    from panda3d.core import Texture

    dtype = {Texture.T_float: np.float32, Texture.T_half_float: np.float16}[
        texture.get_component_type()
    ]
    image = np.frombuffer(texture.get_ram_image(), dtype).astype(np.float32)
    return image.reshape(-1, texture.get_num_components())


def test_a_genus_inside_another_s_slab_renders(window, mid_genus):
    """The union march with overlapping genera: three genera in two sky
    bands (the stratus's; the convective one holding the third genus too),
    each genus its own parameters, coverage and far table; looking up from
    under the stratus deck the march draws cloud, and nothing that is not
    finite."""
    from panda3d.core import Point3

    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.planet import PlanetClouds
    from world_builder.depth import SplitDepthCameras
    from world_builder.terrain.chart import TangentFrame
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    clouds = PlanetClouds(
        window.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(octave_bands=2),
        genera=mid_genus,
    )
    depth = SplitDepthCameras(window, near=1.0)
    camera = window.camera
    pos, hpr = camera.get_pos(), camera.get_hpr()
    try:
        march = clouds.volume
        assert march.genera_count == 3
        # The stratus's family slab (0.6-2.15 km), the convective's 2.2-10.5 km
        assert march.sky_bands == [(600.0, 1550.0), (2200.0, 8300.0)]
        assert [march._sky_bands[k][2] for k in range(2)] == [0b010, 0b101]
        rows = np.array([tuple(march._layer_params[i]) for i in range(3 * GENUS_VEC4S)])
        assert tuple(rows[2 * GENUS_VEC4S + 13]) == (COVERAGE_SOURCES["A"], 1, 2, 1)
        assert tuple(rows[2 * GENUS_VEC4S + 4][:2]) == (3000.0, 1000.0)
        clouds.use_split_depth(window, depth)
        march._finish()
        assert march._genera[2].weather.shape[-1] == 3
        for name in ("genusCoverA", "genusShift", "genusDeckLight"):
            assert name in march._textures
        assert march._textures["genusDeckLight"].get_z_size() == 3
        # Under the stratus deck (genera.DEFAULT_FEATURES: 30 km south),
        # looking up through every band
        frame = TangentFrame.at(-2.86, 0.0)
        here = frame.up * (R + 600.0)
        camera.set_pos(0.0, 0.0, 0.0)
        camera.look_at(Point3(*frame.up))
        clouds.set_sun(frame.up + 0.5 * frame.east)
        (layer,) = march._layers
        gsg = window.win.get_gsg()
        alphas = []
        for _ in range(6):
            depth.update(*SplitDepthCameras.planet_ranges(R, R + 600.0))
            clouds.update(here)
            window.graphicsEngine.render_frame()
            window.graphicsEngine.extract_texture_data(layer.colour, gsg)
            image = _pixels(layer.colour)
            assert np.all(np.isfinite(image))
            alphas.append(image[:, 3].max())
        assert max(alphas) > 0.05
    finally:
        camera.set_pos(pos)
        camera.set_hpr(hpr)
        clouds.remove()
        depth.destroy()
        executor.shutdown()
