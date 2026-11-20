"""
The high layer in the march (clouds.md, "The high layer"): its slab in a sky
band of its own above the union of the genera (genera.HIGH_SLAB,
planet.check_layers), its genera on coverage cube B's channels 1-3
(genera.HIGH_CHANNELS, high_genera), the thin-layer crossings sorted among
the bands' stretches (march.band_segments' high slab: its crossings first
and last, thin_optical_depth), the shader variant compiled only with a high
genus covered (march.march_fragment), the thin-slab lighting cirrus.frag and
the march share (cloud_shading.glsl's thinSlabRadiance), the cirrus's flow
streaks (Task 23: the upper wind on the GPU, genera.high_wind_cube; the
streaks' CPU mirror, march.cirrus_streaks; the weather's cirrus by the march,
GeneraSpec.cirrus_weather), and a GPU smoke test.
"""

import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from world_builder.clouds.cloud import CloudType
from world_builder.clouds.genera import (
    DEFAULT_FEATURES,
    GENUS_PRESETS,
    HIGH_CHANNELS,
    HIGH_OPTICS,
    HIGH_SLAB,
    SHOWCASE_HIGH,
    SHOWCASE_HIGH_EYE,
    SHOWCASE_HIGH_PITCH,
    Feature,
    GeneraSpec,
    GenusPreset,
    coverage_cube,
    genus_coverage,
    high_genera,
    high_jet,
    high_wind_cube,
    showcase_features,
    showcase_slots,
    upper_stream,
    upper_wind_at,
    upper_wind_field,
)
from world_builder.clouds.march import (
    CC_CELL,
    CC_GRAIN,
    CC_RIPPLE,
    CI_CLUMP,
    CI_CLUMP_WINDOW,
    CI_COVER_EDGE,
    CI_FAN,
    CI_FIBRE,
    CI_FIBRE_GAIN,
    CI_FIBRE_WINDOW,
    CI_HEAD_FLOOR,
    CI_HEAD_FRAY,
    CI_HEAD_GAIN,
    CI_HEAD_WINDOW,
    CI_HOOK,
    CI_MAX_DENSITY,
    CI_OFFSETS,
    CI_PATCH,
    CI_PATCH_COVER,
    CI_PATCH_WINDOW,
    CI_PSI_AXIS,
    CI_SHEEN,
    CI_STRAND_FLOOR,
    CI_STRAND_WINDOW,
    CI_TAIL_FADE,
    CI_TAIL_GAIN,
    CI_TAIL_STEP,
    CI_TAPS,
    CI_TUFT,
    CI_TUFT_WINDOW,
    CS_FEATURE,
    CS_FIBRE,
    CS_STRETCH,
    HIGH_BAND,
    HIGH_MIN_INCIDENCE,
    THIN_GRAZING_PATH,
    band_segments,
    cirrocumulus_grain_mean,
    cirrocumulus_grains,
    cirrus_detail,
    cirrus_means,
    cirrus_patch_mean,
    cirrus_patches,
    cirrus_streaks,
    cirrus_tufts,
    footprint_along,
    high_optical_depth,
    march_fragment,
    sheet_crossing,
    sky_bands,
    slab_footprint,
    slab_segments,
    thin_optical_depth,
)
from world_builder.clouds.planet import check_layers, layer_slabs
from world_builder.clouds.weather import WeatherSpec
from world_builder.terrain.planet import PlanetSpec

R = 600e3
SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"


@pytest.fixture(scope="module")
def weather():
    return WeatherSpec(PlanetSpec(name="clouds", radius=R), seed=11)


#: A placeholder high feature over the default cumulonimbus (40 km north).
HIGH = Feature(0.0, 40e3, 30e3, "cirrus", 0.8)


# ── The slab and its band ────────────────────────────────────────────────────


@pytest.mark.parametrize("showcase", [False, True])
def test_the_high_slab_is_a_band_of_its_own_above_the_union(weather, showcase):
    """Added to the genera's slabs, the high slab neither merges with the
    convective clouds' band (10.5 km top) nor bridges anything: the bands
    below are unchanged and it is the last, whole."""
    features = showcase_features() if showcase else DEFAULT_FEATURES
    spec = GeneraSpec(weather, features=features)
    slabs = layer_slabs(CloudType.CONVECTIVE, 2200.0, spec)
    below = sky_bands(slabs)
    bands = sky_bands([*slabs, spec.high_slab])
    assert bands[:-1] == below
    assert bands[-1] == spec.high_slab == HIGH_SLAB
    assert HIGH_SLAB[0] > below[-1][0] + below[-1][1]  # a gap: never merged


def test_the_high_slab_must_clear_the_union_and_stay_in_the_air(weather, monkeypatch):
    spec = GeneraSpec(weather)
    check_layers(CloudType.CUMULUS, 2200.0, None, None, spec)
    monkeypatch.setitem(
        GENUS_PRESETS,
        "tall",
        GenusPreset(CloudType.STRATUS, ("A", 0), slab=(500.0, 10800.0)),
    )
    tall = replace(spec, genera=("convective", "tall"))
    with pytest.raises(ValueError, match=r"high layer's base \(11000 m\)"):
        check_layers(CloudType.CUMULUS, 2200.0, None, None, tall)
    # No high genera listed: no high layer to check
    check_layers(CloudType.CUMULUS, 2200.0, None, None, replace(tall, high=()))
    with pytest.raises(ValueError, match="high layer's top"):
        check_layers(CloudType.CUMULUS, 2200.0, None, 12000.0, spec)


def test_the_old_cirrus_sheet_may_share_the_high_slab(weather):
    """Until Task 24 deletes it, the sheet (11-12 km) sits inside the high
    slab; only the union of the genera bounds its base."""
    from world_builder.clouds.cirrus import CirrusSpec

    sheet = CirrusSpec()
    assert HIGH_SLAB[0] <= sheet.base < sheet.top <= HIGH_SLAB[0] + HIGH_SLAB[1]
    check_layers(CloudType.CUMULUS, 2200.0, sheet, None, GeneraSpec(weather))


# ── Its genera and their coverage ────────────────────────────────────────────


def test_the_high_genera_take_cube_b_channels_one_to_three():
    assert sorted(HIGH_CHANNELS.values()) == [1, 2, 3]
    assert set(HIGH_CHANNELS) == {"cirrus", "cirrostratus", "cirrocumulus"}
    assert HIGH_CHANNELS["cirrus"] == 1
    # No genus of the march reads them
    for preset in GENUS_PRESETS.values():
        assert preset.coverage[0] != "B" or preset.coverage[1] == 0


def test_only_covered_high_genera_are_drawn(weather):
    """As genus_specs(covered=True): a high genus without a feature costs
    nothing (the demo's default has none)."""
    assert high_genera(GeneraSpec(weather), covered=True) == []
    assert [n for n, _ in high_genera(GeneraSpec(weather))] == list(HIGH_CHANNELS)
    spec = GeneraSpec(weather, features=(*DEFAULT_FEATURES, HIGH))
    assert high_genera(spec, covered=True) == [("cirrus", 1)]
    assert high_genera(replace(spec, high=()), covered=True) == []


def test_cube_b_holds_the_high_coverage(weather):
    spec = GeneraSpec(weather, features=(*DEFAULT_FEATURES, HIGH))
    cube = coverage_cube(spec, "B", 64)
    np.testing.assert_array_equal(cube[..., 1], genus_coverage(spec, "cirrus", 64))
    assert cube[..., 1].max() == pytest.approx(0.8)
    assert not cube[..., 2:].any() and not cube[..., 0].any()


# ── The thin-layer crossings ─────────────────────────────────────────────────

UP = np.array([0.0, 0.0, 1.0])
LOW = [(2200.0, 8300.0), (900.0, 500.0)]


def test_from_below_the_high_crossing_comes_last():
    segments = band_segments(UP * (R + 100.0), UP, R, LOW, high=HIGH_SLAB)
    assert [band for *_, band in segments] == [0, 1, HIGH_BAND]
    assert segments[-1][:2] == pytest.approx((10900.0, 12400.0))
    # The bands' own stretches are unchanged by it
    assert segments[:-1] == band_segments(UP * (R + 100.0), UP, R, LOW)


def test_from_above_the_high_crossing_comes_first():
    """A camera over the high slab looking down: the veil in front of the
    towers, composited over them."""
    tilt = np.array([0.6, 0.0, -0.8])
    segments = band_segments(UP * (R + 14e3), tilt, R, LOW, high=HIGH_SLAB)
    assert segments[0][2] == HIGH_BAND
    assert all(band != HIGH_BAND for *_, band in segments[1:])
    assert segments == sorted(segments)


def test_the_high_crossing_is_clipped_like_any_band():
    origin = UP * (R + 100.0)
    # The scene (a mountain, the far plane) ends the ray before it
    segments = band_segments(origin, UP, R, LOW, t_end=11000.0, high=HIGH_SLAB)
    assert segments[-1] == (pytest.approx(10900.0), pytest.approx(11000.0), HIGH_BAND)
    segments = band_segments(origin, UP, R, LOW, t_end=10000.0, high=HIGH_SLAB)
    assert all(band != HIGH_BAND for *_, band in segments)
    # A camera inside the slab: the crossing starts at the near plane
    inside = UP * (R + 11500.0)
    level = np.array([1.0, 0.0, 0.0])
    (crossing,) = band_segments(inside, level, R, LOW, t_start=1.0, high=HIGH_SLAB)
    expected = slab_segments(inside, level, R, HIGH_SLAB, t_start=1.0)[0]
    assert crossing == (1.0, pytest.approx(expected[1]), HIGH_BAND)


def test_the_high_crossings_come_before_and_after_every_band():
    """The shader takes the high slab's crossings out of the sorted
    segments: a ray's height is convex and the slab is above every band, so
    its crossing on the way down (slab_segments' first) comes before every
    band's stretch and its crossing on the way up (the second) after."""
    rng = np.random.default_rng(4)
    for _ in range(400):
        height = rng.uniform(100.0, 20e3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        origin = UP * (R + height)
        segments = band_segments(origin, direction, R, LOW, high=HIGH_SLAB)
        down, up = slab_segments(origin, direction, R, HIGH_SLAB)
        low = [seg for seg in segments if seg[2] != HIGH_BAND]
        if down[1] > down[0]:
            assert segments[0] == (down[0], down[1], HIGH_BAND)
            assert all(t0 >= down[1] - 1e-6 for t0, _, _ in low)
        if up[1] > up[0]:
            assert segments[-1] == (up[0], up[1], HIGH_BAND)
            assert all(t1 <= up[0] + 1e-6 for _, t1, _ in low)


def test_the_optical_depth_is_capped_at_grazing_and_saturates():
    thickness, extinction, cap = 1500.0, 2e-4, 2.5
    vertical = thin_optical_depth(1.0, thickness, thickness, extinction, cap)
    assert vertical == pytest.approx(cap * (1 - np.exp(-0.3 / cap)))
    # Past THIN_GRAZING_PATH slab depths a longer path adds nothing
    far = THIN_GRAZING_PATH * thickness
    assert thin_optical_depth(1.0, 10 * far, thickness, extinction, cap) == (
        pytest.approx(thin_optical_depth(1.0, far, thickness, extinction, cap))
    )
    assert thin_optical_depth(50.0, far, thickness, extinction, cap) < cap
    assert thin_optical_depth(0.0, far, thickness, extinction, cap) == 0.0


# ── The shaders ──────────────────────────────────────────────────────────────


def test_the_high_variant_is_compiled_only_with_a_high_genus():
    assert march_fragment(False, False) == "cloud_march.frag"
    assert march_fragment(True, False) == "cloud_march_recipes.frag"
    assert march_fragment(False, True) == "cloud_march_high.frag"
    assert march_fragment(True, True) == "cloud_march_recipes_high.frag"
    for recipes in (False, True):
        for high in (False, True):
            text = (SHADERS / march_fragment(recipes, high)).read_text()
            assert ("#define CLOUD_RECIPES" in text) == recipes
            assert ("#define CLOUD_HIGH" in text) == high
            assert '#pragma include "cloud_march.glsl"' in text
    march = (SHADERS / "cloud_march.glsl").read_text()
    assert "#ifdef CLOUD_HIGH" in march
    assert "vec2 segments[MAX_SEGMENTS];" in march
    # The high crossings outside the sorted segments: before and after them
    assert "const int MAX_SEGMENTS = 2 * MAX_SKY_BANDS;" in march
    main = march[march.index("void main()") :]
    down = main.index("highCrossing(highDown")
    loop = main.index("for (int s = 0; s < MAX_SEGMENTS; ++s)")
    up = main.index("highCrossing(highUp")
    assert down < loop < up
    assert "HIGH_BAND" not in march


def test_the_march_and_the_sheet_share_the_thin_slab_lighting():
    shading = (SHADERS / "cloud_shading.glsl").read_text()
    assert "vec3 thinSlabRadiance(" in shading
    cirrus = (SHADERS / "cirrus.frag").read_text()
    assert '#pragma include "cloud_shading.glsl"' in cirrus
    assert "thinSlabRadiance(" in cirrus
    assert "lobePhase(" not in cirrus  # its own copy gone
    march = (SHADERS / "cloud_march.glsl").read_text()
    high = march[march.index("void highCrossing(") :]
    assert "thinSlabRadiance(" in high[: high.index("\n}\n")]
    value = float(re.search(r"const float THIN_GRAZING_PATH = ([\d.]+);", march)[1])
    assert value == THIN_GRAZING_PATH


# ── On the GPU ───────────────────────────────────────────────────────────────


def _pixels(texture):
    from panda3d.core import Texture

    dtype = {Texture.T_float: np.float32, Texture.T_half_float: np.float16}[
        texture.get_component_type()
    ]
    image = np.frombuffer(texture.get_ram_image(), dtype).astype(np.float32)
    return image.reshape(-1, texture.get_num_components())


@pytest.fixture
def window(showbase):
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


@pytest.mark.parametrize(
    "genus", [None, "cirrus", "cirrostratus", "cirrocumulus"], ids=str
)
def test_the_high_layer_renders_only_when_covered(window, weather, genus):
    """Looking up from under a high feature of each genus (its stratus deck
    and cumulus hidden): the high variant draws it (the cirrocumulus's
    grains, the cirrostratus's veil, the cirrus's streaks); without a
    feature the plain variant is compiled and the sky stays clear."""
    covered = genus is not None
    from panda3d.core import Point3, Shader

    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.planet import PlanetClouds
    from world_builder.depth import SplitDepthCameras
    from world_builder.terrain.chart import TangentFrame
    from world_builder.terrain.workers import ChunkExecutor

    over = Feature(0.0, -30e3, 25e3, genus or "cirrus", 0.9)  # over the stratus deck
    spec = GeneraSpec(
        weather,
        features=(*DEFAULT_FEATURES, over) if covered else DEFAULT_FEATURES,
        background=False,
    )
    executor = ChunkExecutor(backend="inline", inline_budget=64)
    clouds = PlanetClouds(
        window.render, R, executor, altitude=2200.0,
        settings=CloudSettings(octave_bands=2), genera=spec,
    )  # fmt: skip
    depth = SplitDepthCameras(window, near=1.0)
    camera = window.camera
    pos, hpr = camera.get_pos(), camera.get_hpr()
    try:
        march = clouds.volume
        assert march.high_genera == ([genus] if covered else [])
        clouds.use_split_depth(window, depth)
        march._finish()
        frame = TangentFrame.at(-2.86, 0.0)
        here = frame.up * (R + 600.0)
        camera.set_pos(0.0, 0.0, 0.0)
        camera.look_at(Point3(*frame.up))
        clouds.set_sun(frame.up + 0.5 * frame.east)
        (layer,) = march._layers
        fragment = str(layer.card.get_shader().get_filename(Shader.ST_fragment))
        assert fragment.endswith(march_fragment(True, covered))
        for genus in range(march.genera_count):
            march.set_genus_visible(genus, False)
        gsg = window.win.get_gsg()
        alphas = []
        for _ in range(6):
            depth.update(*SplitDepthCameras.planet_ranges(R, R + 13000.0))
            clouds.update(here)
            window.graphicsEngine.render_frame()
            window.graphicsEngine.extract_texture_data(layer.colour, gsg)
            image = _pixels(layer.colour)
            assert np.all(np.isfinite(image))
            alphas.append(image[:, 3].max())
        if covered:
            assert max(alphas) > 0.02
        else:
            assert max(alphas) == 0.0
    finally:
        camera.set_pos(pos)
        camera.set_hpr(hpr)
        clouds.remove()
        depth.destroy()
        executor.shutdown()


# ── The cirrostratus and the cirrocumulus (Tasks 21-22) ──────────────────────


@pytest.mark.parametrize("genus", ["cirrus", "cirrostratus", "cirrocumulus"])
def test_each_high_genus_fills_its_own_channel_of_cube_b(weather, genus):
    feature = Feature(0.0, 40e3, 30e3, genus, 0.7)
    spec = GeneraSpec(weather, features=(*DEFAULT_FEATURES, feature))
    assert high_genera(spec, covered=True) == [(genus, HIGH_CHANNELS[genus])]
    cube = coverage_cube(spec, "B", 64)
    channel = HIGH_CHANNELS[genus]
    np.testing.assert_array_equal(cube[..., channel], genus_coverage(spec, genus, 64))
    assert cube[..., channel].max() == pytest.approx(0.7)
    others = [c for c in range(4) if c != channel]
    assert not cube[..., others].any()


def test_the_default_spec_has_no_high_coverage(weather):
    """The demo's scene pays nothing for the high layer: no high genus is
    covered, so the march compiles its plain variant (the GPU test below
    checks the shader)."""
    spec = GeneraSpec(weather)
    assert not any(f.genus in HIGH_CHANNELS for f in DEFAULT_FEATURES)
    assert high_genera(spec, covered=True) == []
    assert not coverage_cube(spec, "B", 32)[..., 1:].any()


def test_the_high_genera_have_optics_of_their_own():
    """Each its full veil's optical depth straight through the slab and what
    a crossing's saturates toward; a cirrostratus 0.03-0.3 (the research),
    its halo; the cirrus's streaks 0.1-3 (a tuft about its depth, SIRTA's
    0.03-3), a fainter halo (its fragments); the cirrocumulus's grain cores
    denser than the veil, no halo."""
    assert set(HIGH_OPTICS) == set(HIGH_CHANNELS)
    ci = HIGH_OPTICS["cirrus"]
    cs, cc = HIGH_OPTICS["cirrostratus"], HIGH_OPTICS["cirrocumulus"]
    assert 0.03 <= cs.depth <= 0.3
    assert cs.halo > 0.0
    assert 0.1 <= ci.depth <= 3.0
    assert 0.0 < ci.halo < cs.halo
    assert cc.halo == 0.0
    assert cc.depth > cs.depth
    for optics in HIGH_OPTICS.values():
        assert optics.cap > optics.depth


def test_the_high_optics_are_packed_per_channel(weather):
    """uHighExtinction and uHighCap hold the channels' (cirrus,
    cirrostratus, cirrocumulus: cube B's 1-3) in x, y, z."""
    from world_builder.clouds.march import CloudMarch

    spec = GeneraSpec(weather)
    inputs = CloudMarch._high_inputs(spec)
    base, thickness = spec.high_slab
    assert tuple(inputs["uHighSlab"])[:2] == (base, thickness)
    order = sorted(HIGH_CHANNELS, key=HIGH_CHANNELS.get)
    extinction = [HIGH_OPTICS[name].depth / thickness for name in order]
    np.testing.assert_allclose(tuple(inputs["uHighExtinction"]), extinction, rtol=1e-6)
    caps = [HIGH_OPTICS[name].cap for name in order]
    np.testing.assert_allclose(tuple(inputs["uHighCap"]), caps, rtol=1e-6)
    shape = tuple(inputs["uHighShape"])
    assert shape[0] == pytest.approx(cirrocumulus_grain_mean())
    assert shape[1] == pytest.approx(HIGH_OPTICS["cirrostratus"].halo)
    assert shape[2] == pytest.approx(HIGH_OPTICS["cirrus"].halo)
    means = cirrus_means(7)
    assert shape[3] == pytest.approx(means[4])
    np.testing.assert_allclose(tuple(inputs["uHighCirrus"]), means[:4], rtol=1e-6)
    np.testing.assert_allclose(tuple(inputs["uHighJet"]), high_jet(spec), atol=1e-6)


def test_the_shader_reads_the_optics_per_channel():
    march = (SHADERS / "cloud_march.glsl").read_text()
    high = march[march.index("void highCrossing(") :]
    body = high[: high.index("\n}\n")]
    for name in ("uHighExtinction", "uHighCap", "uHighShape"):
        assert f"uniform vec{4 if name == 'uHighShape' else 3} {name};" in march
    assert "uHighExtinction" in body and "uHighCap" in body
    assert "uHighSlab.z" not in march and "uHighSlab.w" not in march
    for name, value in (
        ("CC_CELL", CC_CELL),
        ("CC_RIPPLE", CC_RIPPLE),
        ("CS_FEATURE", CS_FEATURE),
        ("CS_STRETCH", CS_STRETCH),
        ("CS_FIBRE", CS_FIBRE),
        ("HIGH_MIN_INCIDENCE", HIGH_MIN_INCIDENCE),
    ):
        assert float(re.search(rf"const float {name} = ([\d.]+);", march)[1]) == value
    lo, hi = re.search(
        r"const vec2 CC_GRAIN = vec2\(([\d.]+), ([\d.]+)\);", march
    ).groups()
    assert (float(lo), float(hi)) == CC_GRAIN


def test_the_optical_depth_of_mixed_genera():
    """The channels' optical depths add; the cap is theirs weighted by their
    shares; one channel alone is thin_optical_depth's."""
    thickness = 1500.0
    ext = np.array([0.3, 0.2, 1.0]) / thickness
    caps = np.array([2.5, 1.2, 2.5])
    for k in range(3):
        density = np.zeros(3)
        density[k] = 0.7
        assert high_optical_depth(density, 4000.0, thickness, ext, caps) == (
            pytest.approx(thin_optical_depth(0.7, 4000.0, thickness, ext[k], caps[k]))
        )
    assert high_optical_depth(np.zeros(3), 4000.0, thickness, ext, caps) == 0.0
    both = high_optical_depth(np.array([0.0, 1.0, 1.0]), 1500.0, thickness, ext, caps)
    assert 0.2 < both < 1.2


def test_the_sheet_point_is_where_the_ray_crosses_the_slab_middle():
    """The cirrostratus's and cirrocumulus's patterns are read where the
    ray crosses the slab's middle, not at the jittered samples: a thin
    sheet's pattern holds still over the visits."""
    middle = R + HIGH_SLAB[0] + 0.5 * HIGH_SLAB[1]
    from_below = UP * (R + 100.0)
    ((t0, t1, _),) = band_segments(from_below, UP, R, [], high=HIGH_SLAB)
    t = sheet_crossing(from_below, UP, middle, t0, t1)
    assert t == pytest.approx(middle - R - 100.0)
    tilt = np.array([0.6, 0.0, -0.8])
    from_above = UP * (R + 14e3)
    t0, t1, _ = band_segments(from_above, tilt, R, [], high=HIGH_SLAB)[0]
    t = sheet_crossing(from_above, tilt, middle, t0, t1)
    assert t0 < t < t1
    assert np.linalg.norm(from_above + t * tilt) == pytest.approx(middle)
    # Inside the slab under its middle, looking down: the middle is never
    # crossed in the segment; its middle instead
    inside = UP * (R + 11200.0)
    t0, t1, _ = band_segments(inside, -UP, R, [], t_start=1.0, high=HIGH_SLAB)[0]
    assert sheet_crossing(inside, -UP, middle, t0, t1) == pytest.approx(0.5 * (t0 + t1))


def test_the_footprint_on_the_slab_grows_toward_grazing():
    assert slab_footprint(10.0, 1.0) == pytest.approx(10.0)
    assert slab_footprint(10.0, 0.5) == pytest.approx(20.0)
    assert slab_footprint(10.0, 0.0) == pytest.approx(10.0 / HIGH_MIN_INCIDENCE)


def test_the_cirrocumulus_grains_fade_to_their_mean():
    """Grains a pixel cannot resolve (2 footprints a cell: none of them,
    4: whole) are their mean: the far patch keeps its optical depth, and
    the mean is the resolved grains' own."""
    from world_builder.clouds.field_setup import _cells

    cells = _cells()
    rng = np.random.default_rng(3)
    at = rng.uniform(0.0, 50e3, (20000, 3))
    mean = cirrocumulus_grain_mean()
    sharp = cirrocumulus_grains(cells, at, 1.0)
    assert 0.05 < mean < 0.5
    assert sharp.mean() == pytest.approx(mean, abs=0.02)
    assert sharp.min() == 0.0 and sharp.max() > 0.9  # grains with sky between
    far = cirrocumulus_grains(cells, at, CC_CELL / 2.0)
    np.testing.assert_allclose(far, mean, atol=1e-6)
    half = cirrocumulus_grains(cells, at, CC_CELL / 3.0)
    assert 0.0 < np.abs(half - mean).mean() < np.abs(sharp - mean).mean()


# ── The showcase's high row ──────────────────────────────────────────────────

#: The view's half-angles from the showcase's poses (70 degrees across at
#: 16:9: 21.5 degrees up and down).
HALF_ACROSS, HALF_UP = 35.0, 21.5
#: The low features' cloud tops, metres (the cumulus species and the stratus;
#: the others their presets' slabs' tops).
LOW_TOPS = {"humilis": 3500.0, "mediocris": 4500.0, "congestus": 7000.0,
            "cumulonimbus": 10500.0, "stratus": 2000.0}  # fmt: skip


def _feature(slot):
    genus, bearing, distance, radius, coverage = slot
    b = math.radians(bearing)
    return Feature(
        distance * math.sin(b), distance * math.cos(b), radius, genus, coverage
    )


def test_the_showcase_has_the_high_genera_apart():
    """The cirrus, the cirrostratus and the cirrocumulus, each its own disc,
    the discs apart (each high genus is seen alone)."""
    names = [f.genus for f in showcase_features()]
    assert names.count("cirrostratus") == names.count("cirrocumulus") == 1
    assert names.count("cirrus") == 1
    high = [_feature(slot) for slot in SHOWCASE_HIGH]
    assert {f.genus for f in high} <= set(HIGH_CHANNELS)
    for i, a in enumerate(high):
        for b in high[i + 1 :]:
            gap = math.hypot(a.east - b.east, a.north - b.north)
            assert gap > a.radius + b.radius, (a.genus, b.genus)
    # The cirrostratus is a wide veil, the cirrocumulus a patch
    radii = {f.genus: f.radius for f in high}
    assert radii["cirrostratus"] >= 25e3 and radii["cirrocumulus"] <= 10e3


@pytest.mark.parametrize("slot", SHOWCASE_HIGH, ids=lambda s: s[0])
def test_each_high_genus_is_shot_with_no_low_cloud_in_view(slot):
    """From the ground (the start's, or SHOWCASE_HIGH_EYE metres along its
    bearing; about sea level), looking along the high feature's bearing at
    its pose's pitch (SHOWCASE_HIGH_PITCH): every low feature is either off
    the view's sides or under its bottom edge (its top, at its near edge,
    lower than the view's lowest angle), or behind the eye."""
    genus, bearing, distance, radius, _ = slot
    pitch = SHOWCASE_HIGH_PITCH[genus]
    middle = HIGH_SLAB[0] + 0.5 * HIGH_SLAB[1]
    # The view's centre meets the slab's middle over the feature's core
    reach = middle / math.tan(math.radians(pitch))
    b = math.radians(bearing)
    heading = np.array([math.sin(b), math.cos(b)])
    eye = SHOWCASE_HIGH_EYE.get(genus, 0.0) * heading
    centre = distance * heading
    hit = eye + reach * heading
    assert np.linalg.norm(hit - centre) < 2.0 / 3.0 * radius
    for low in showcase_slots():
        name, low_bearing, low_distance, low_radius, _ = low
        if name in HIGH_CHANNELS:
            continue
        lb = math.radians(low_bearing)
        offset = low_distance * np.array([math.sin(lb), math.cos(lb)]) - eye
        low_distance = float(np.linalg.norm(offset))
        low_bearing = math.degrees(math.atan2(offset[0], offset[1]))
        spread = low_radius * (3.0 if name == "cumulonimbus" else 1.0)
        if offset @ heading < -spread:
            continue  # behind the eye
        width = math.degrees(math.asin(min(spread / low_distance, 1.0)))
        off = abs((low_bearing - bearing + 180.0) % 360.0 - 180.0) - width
        top = LOW_TOPS.get(name) or sum(GENUS_PRESETS[name].slab)
        elevation = math.degrees(math.atan2(top, low_distance - spread))
        assert off > HALF_ACROSS or elevation < pitch - HALF_UP, (genus, name)


# ── The cirrus's flow streaks (Task 23) ──────────────────────────────────────


@pytest.fixture(scope="module")
def wind_spec(weather):
    return GeneraSpec(weather, focus=(29.7, 29.8))


def test_the_gpu_wind_is_the_upper_wind(wind_spec):
    """high_wind_cube's rgb is upper_wind_at's wind (east, north) in the
    planet frame, its alpha the stream function; read off the cube as the
    GPU filters it, the direction within a few degrees of the CPU's
    everywhere the wind blows."""
    from world_builder.clouds.weather import sample_cube
    from world_builder.terrain.chart import TangentFrame

    cube = high_wind_cube(wind_spec, 64)
    assert cube.shape == (6, 64, 64, 4) and cube.dtype == np.float32
    rng = np.random.default_rng(2)
    angles = []
    for lat, lon in zip(rng.uniform(-70, 70, 60), rng.uniform(-180, 180, 60)):
        frame = TangentFrame.at(lat, lon)
        up = np.asarray(frame.up, np.float64)
        east, north = upper_wind_at(wind_spec.weather, up)
        cpu = east * np.asarray(frame.east) + north * np.asarray(frame.north)
        gpu = sample_cube(cube, up[None])[0]
        wind = gpu[:3] - (gpu[:3] @ up) * up
        if np.linalg.norm(cpu) < 0.05:
            continue  # near calm: no direction to speak of
        assert np.linalg.norm(wind) == pytest.approx(
            np.linalg.norm(cpu), rel=0.1, abs=0.02
        )
        cos = wind @ cpu / (np.linalg.norm(wind) * np.linalg.norm(cpu))
        angles.append(math.degrees(math.acos(min(cos, 1.0))))
        # The stream function: the wind blows along its contours
        assert gpu[3] == pytest.approx(upper_stream(wind_spec, up[None])[0], abs=2000.0)
    assert len(angles) > 30
    assert max(angles) < 5.0


def test_the_wind_turns_over_the_planet(wind_spec):
    """Not one jet axis: the wind's direction varies (around the lows),
    so the streaks curve with it."""
    from world_builder.sky.starfield import cube_directions

    dirs = cube_directions(16).reshape(-1, 3)
    wind, _ = upper_wind_field(wind_spec, dirs)
    speed = np.linalg.norm(wind, axis=1)
    east = np.cross([0.0, 0.0, 1.0], dirs)
    east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-9)
    cos = np.sum(wind * east, axis=1) / np.maximum(speed, 1e-9)
    assert (cos[speed > 0.05] < 0.7).mean() > 0.05  # many far off the zonal jet
    # The wind is tangent and blows along the stream function's contours
    np.testing.assert_allclose(np.sum(wind * dirs, axis=1), 0.0, atol=1e-9)


def test_a_set_wind_is_uniform_at_the_focus(weather):
    from world_builder.terrain.chart import TangentFrame

    spec = GeneraSpec(weather, focus=(10.0, 20.0), wind=(0.0, 0.8))
    frame = TangentFrame.at(10.0, 20.0)
    wind, _ = upper_wind_field(spec, np.asarray(frame.up)[None])
    np.testing.assert_allclose(wind[0], 0.8 * np.asarray(frame.north), atol=1e-3)
    np.testing.assert_allclose(high_jet(spec), frame.north, atol=1e-6)


def test_the_cirrus_weather_is_the_marchs_only_when_asked(weather):
    """--cirrus march: the weather's cirrus cover (the sheet's) fills cube
    B's channel 1 under the features, and compiles the high layer; by
    default the cirrus has only its features' coverage (none in the demo's
    default: the sheet draws the weather's)."""
    from world_builder.clouds.weather import cirrus_cover_map, sample_cube
    from world_builder.sky.starfield import cube_directions

    plain = GeneraSpec(weather)
    assert high_genera(plain, covered=True) == []
    assert not coverage_cube(plain, "B", 32)[..., 1].any()
    march = replace(plain, cirrus_weather=True)
    assert high_genera(march, covered=True) == [("cirrus", 1)]
    cube = coverage_cube(march, "B", 32)
    dirs = cube_directions(32).reshape(-1, 3)
    expected = sample_cube(cirrus_cover_map(weather), dirs).reshape(6, 32, 32)
    np.testing.assert_allclose(cube[..., 1], expected, atol=1e-6)
    assert not cube[..., [0, 2, 3]].any()
    # A feature over it: the larger of the two
    spec = replace(march, features=(*DEFAULT_FEATURES, HIGH))
    np.testing.assert_allclose(
        coverage_cube(spec, "B", 32)[..., 1],
        np.maximum(expected, genus_coverage(spec, "cirrus", 32)),
        atol=1e-6,
    )
    # No cirrus among the high genera: none
    assert not coverage_cube(replace(march, high=()), "B", 32)[..., 1].any()


def test_the_cirrus_showcase_slot_is_clear_of_every_feature():
    """Its disc clear of every other feature's (a cumulonimbus's to its
    anvil's 3 radii), the high ones' included."""
    (ci,) = [_feature(s) for s in SHOWCASE_HIGH if s[0] == "cirrus"]
    for f in showcase_features():
        if f.genus == "cirrus":
            continue
        reach = f.radius * (3.0 if f.genus == "cumulonimbus" else 1.0)
        gap = math.hypot(ci.east - f.east, ci.north - f.north)
        assert gap > ci.radius + reach, f.genus


@pytest.fixture(scope="module")
def streak_sample():
    from world_builder.clouds.march import _cirrus_sample

    return _cirrus_sample(7, n=20000, rng_seed=11)  # other points than the means'


def test_the_cirrus_means_are_the_resolved_streaks(streak_sample):
    """What the parts fade to is their own mean over the sheet: a far
    stretch keeps its optical depth."""
    volume, q, psi, along, across = streak_sample
    detail, _, _, strands, tufts = cirrus_means()
    assert 0.03 < detail < 0.5 and 0.0 < tufts < detail
    assert cirrus_detail(volume, q, psi, along, across).mean() == pytest.approx(
        detail, rel=0.1
    )
    assert cirrus_tufts(volume, q, psi, along, across).mean() == pytest.approx(
        tufts, rel=0.1
    )
    # The patches' share of the sheet about the coverage (the cirrus cover
    # map's meaning), and their mean's line close to it
    for cover in (0.2, 0.5, 0.8):
        share = cirrus_patches(volume, q, psi, across, cover).mean()
        assert share == pytest.approx(cover, abs=0.08)
        assert cirrus_patch_mean(cover) == pytest.approx(share, abs=0.05)


def test_the_streaks_are_detached_filaments(streak_sample):
    """Resolved, the streaks are mostly sky with dense heads (a tuft about
    1) and fibres between: no veil."""
    volume, q, psi, along, across = streak_sample
    streaks = cirrus_streaks(volume, q, psi, along, across, 5.0, 0.5)
    assert (streaks < 0.02).mean() > 0.5
    assert streaks.max() > 0.9


def test_the_streaks_fade_to_their_mean_by_footprint(streak_sample):
    """Sub-pixel fibres go to their mean (resolved: 2 footprints to a
    feature none of it, 4 all): from far away the streaks are their mean
    times the patches' mean, without aliasing."""
    volume, q, psi, along, across = streak_sample
    detail = cirrus_means()[0]
    far = cirrus_streaks(volume, q, psi, along, across, 1e5, 0.5)
    np.testing.assert_allclose(far, detail * cirrus_patch_mean(0.5), rtol=1e-6)
    sharp = cirrus_streaks(volume, q, psi, along, across, 5.0, 0.5)
    half = cirrus_streaks(volume, q, psi, along, across, CI_FIBRE[0] / 3.0, 0.5)
    mean = sharp.mean()
    assert half.mean() == pytest.approx(mean, rel=0.15)
    assert np.abs(half - mean).mean() < np.abs(sharp - mean).mean()
    # Coverage 0: nothing, whatever the footprint
    assert not cirrus_streaks(volume, q, psi, along, across, 5.0, 0.0).any()


def test_fibres_across_a_grazing_view_stay_resolved(streak_sample):
    """The slant stretches the footprint along the view: fibres running
    across it keep their detail where the view's footprint would fade them,
    fibres running along it fade (footprint_along)."""
    volume, q, psi, along, across = streak_sample
    view = np.array([0.0, 1.0, 0.0])  # looking across the wind (along y)
    incidence, pixel = 0.06, 20.0
    across_fp = footprint_along([0.0, 1.0, 0.0], view, pixel, incidence)
    along_fp = footprint_along([1.0, 0.0, 0.0], view, pixel, incidence)
    assert across_fp == pytest.approx(pixel / incidence)
    assert along_fp == pytest.approx(pixel)
    # Viewed along the wind instead: the fibres (across it) keep their width
    view = np.array([1.0, 0.0, 0.0])
    across_fp = float(footprint_along([0.0, 1.0, 0.0], view, pixel, incidence))
    along_fp = float(footprint_along([1.0, 0.0, 0.0], view, pixel, incidence))
    kept = cirrus_streaks(volume, q, psi, along, across, pixel / incidence, 0.5,
                          footprints=(across_fp, along_fp))  # fmt: skip
    lost = cirrus_streaks(volume, q, psi, along, across, pixel / incidence, 0.5)
    assert kept.std() > 1.4 * lost.std()  # the patches vary in both


def test_the_shader_takes_the_cirrus_constants():
    march = (SHADERS / "cloud_march.glsl").read_text()

    def const(name):
        return float(re.search(rf"const float {name} = ([\d.]+);", march)[1])

    def vec2(name):
        found = re.search(rf"const vec2 {name} = vec2\(([\d.]+), ([\d.]+)\);", march)
        return tuple(float(v) for v in found.groups())

    for name, (width, length) in (
        ("FIBRE", CI_FIBRE), ("SHEEN", CI_SHEEN), ("TUFT", CI_TUFT),
        ("PATCH", CI_PATCH), ("CLUMP", CI_CLUMP),
    ):  # fmt: skip
        assert const(f"CI_{name}_WIDTH") == width
        assert const(f"CI_{name}_LENGTH") == length
    for name, window in (
        ("CI_FIBRE", CI_FIBRE_WINDOW), ("CI_CLUMP", CI_CLUMP_WINDOW),
        ("CI_STRAND", CI_STRAND_WINDOW), ("CI_TUFT", CI_TUFT_WINDOW),
        ("CI_PATCH", CI_PATCH_WINDOW), ("CI_HEAD", CI_HEAD_WINDOW),
    ):  # fmt: skip
        assert vec2(name) == window
    for name, value in (
        ("CI_PATCH_COVER", CI_PATCH_COVER), ("CI_STRAND_FLOOR", CI_STRAND_FLOOR),
        ("CI_TAIL_GAIN", CI_TAIL_GAIN), ("CI_FIBRE_GAIN", CI_FIBRE_GAIN),
        ("CI_MAX_DENSITY", CI_MAX_DENSITY), ("CI_TAIL_STEP", CI_TAIL_STEP),
        ("CI_HOOK", CI_HOOK), ("CI_TAIL_FADE", CI_TAIL_FADE),
        ("CI_COVER_EDGE", CI_COVER_EDGE), ("CI_HEAD_FRAY", CI_HEAD_FRAY),
        ("CI_HEAD_FLOOR", CI_HEAD_FLOOR), ("CI_HEAD_GAIN", CI_HEAD_GAIN),
        ("CI_FAN", CI_FAN),
    ):  # fmt: skip
        assert const(name) == value
    assert int(re.search(r"const int CI_TAPS = (\d+);", march)[1]) == CI_TAPS
    axis = re.search(r"const vec3 CI_PSI_AXIS = vec3\(([\d.]+), ([\d.]+), ([\d.]+)\);",
                     march)  # fmt: skip
    assert tuple(float(v) for v in axis.groups()) == CI_PSI_AXIS
    body = march[march.index("float cirrusStreaks(") :]
    body = body[: body.index("\n}\n")]
    for name, offset in CI_OFFSETS.items():
        assert "vec3({}, {}, {})".format(*offset) in body, name
    # Read once at the sheet point, not at jittered samples
    high = march[march.index("void highCrossing(") :]
    assert "cirrusStreaks(q, dir" in high[: high.index("\n}\n")]
    assert "uniform samplerCube highWind;" in march


def test_the_heads_are_fibrous_not_blobs(streak_sample):
    """The user's complaint after Task 23: solid rounded heads. Where the
    tuft noise peaks (a head's core), the heads are striated by the strands
    (no uniform fill) and translucent (no density near the cap), and the
    taps' fan splays the fibres there: without it the same points differ."""
    import world_builder.clouds.march as march
    from world_builder.clouds.march import CI_OFFSETS, CI_TUFT, cirrus_flow
    from world_builder.clouds.noise import sample_volume

    volume, q, psi, along, across = streak_sample
    core = (
        sample_volume(
            volume,
            cirrus_flow(
                q, psi, across, 1.0, np.zeros_like(q), CI_TUFT, CI_OFFSETS["tuft"]
            ),
        )
        > 0.8
    )
    assert core.sum() > 100
    tufts = cirrus_tufts(volume, q[core], psi[core], along[core], across[core])
    assert tufts.std() / tufts.mean() > 0.25  # striated
    assert np.quantile(tufts, 0.5) < 0.6 * CI_HEAD_GAIN + 0.5  # translucent
    assert CI_HEAD_GAIN <= 1.2 and CI_HEAD_WINDOW[1] - CI_HEAD_WINDOW[0] >= 0.3
    fanned = cirrus_tufts(volume, q, psi, along, across)
    saved = march.CI_FAN
    try:
        march.CI_FAN = 0.0
        unfanned = cirrus_tufts(volume, q, psi, along, across)
    finally:
        march.CI_FAN = saved
    assert np.abs(fanned - unfanned).mean() > 0.01
