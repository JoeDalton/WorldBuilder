"""
Unit tests for the raymarched clouds (world_builder.clouds.march): the
NumPy mirror of the shader's geometry (the slab's segments, the steps and
their compositing), the constants the shader shares with Python, and the
renderer choice in PlanetClouds. The shader itself needs a window (see
scripts/cloud_bench.py).
"""

import math
import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudSettings, CloudType
from world_builder.clouds.march import (
    MIN_STEP_FEATURE,
    SEGMENT_SAMPLES,
    STEP_GROWTH,
    VIEW_EXTINCTION_SCALE,
    CloudMarch,
    bayer_offset,
    block_of,
    march_steps,
    march_transmittance,
    min_step,
    slab_segments,
    sphere_interval,
    step_scale,
    visit_of,
)
from world_builder.clouds.noise import (
    MAX_LAYERS,
    NOISE_FEATURE,
    density_on_sphere,
    fbm_sigma,
    sphere_column_peaks,
    with_threshold,
)
from world_builder.clouds.planet import PlanetClouds
from world_builder.clouds.sphere import CloudVolume, _noise
from world_builder.clouds.weather import coverage_to_shift, shift_table
from world_builder.quality import QualityLevel
from world_builder.terrain.planet import PRESETS as PLANETS
from world_builder.terrain.workers import ChunkExecutor

R = 600e3
SLAB = (2200.0, 1000.0)
SHADER = (
    Path(__file__).parent.parent / "src/world_builder/clouds/shaders/cloud_march.frag"
)
UP = np.array([0.0, 0.0, 1.0])


def _ray(altitude: float, elevation: float) -> tuple[np.ndarray, np.ndarray]:
    """A camera *altitude* above the north pole looking *elevation* degrees
    above its horizon."""
    e = math.radians(elevation)
    return UP * (R + altitude), np.array([math.cos(e), 0.0, math.sin(e)])


def _length(segments: np.ndarray) -> float:
    return float(np.clip(segments[:, 1] - segments[:, 0], 0.0, None).sum())


def _nonempty(segments: np.ndarray) -> np.ndarray:
    return np.array([s for s in segments if s[1] > s[0]])


# ── The slab's segments ──────────────────────────────────────────────────────


def test_sphere_interval_bounds_the_points_inside():
    rng = np.random.default_rng(1)
    origin = UP * (R + 3000.0)
    for _ in range(50):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        t0, t1 = sphere_interval(origin, d, R + 2500.0)
        ts = np.linspace(-1e5, 1e5, 20001)
        inside = np.linalg.norm(origin + ts[:, None] * d, axis=1) < R + 2500.0
        if t0 >= t1:
            assert not inside.any()
        else:
            assert inside[(ts > t0 + 20) & (ts < t1 - 20)].all()
            assert not inside[(ts < t0 - 20) | (ts > t1 + 20)].any()


def test_a_ray_from_below_crosses_the_slab_once():
    origin, d = _ray(300.0, 90.0)
    segments = _nonempty(slab_segments(origin, d, R, SLAB))
    assert len(segments) == 1
    assert segments[0] == pytest.approx([1900.0, 2900.0], abs=0.01)


def test_a_ray_from_inside_starts_at_the_camera():
    origin, d = _ray(2700.0, 0.0)
    segments = _nonempty(slab_segments(origin, d, R, SLAB))
    # Out through the top, at the horizontal distance to a sphere 500 m up
    top = math.sqrt((R + 3200.0) ** 2 - (R + 2700.0) ** 2)
    assert segments == pytest.approx(np.array([[0.0, top]]), rel=1e-6)


def test_a_ray_from_above_stops_at_the_base_looking_down():
    origin, d = _ray(5000.0, -90.0)
    segments = _nonempty(slab_segments(origin, d, R, SLAB))
    # It meets the planet: nothing behind the slab's underside
    assert segments == pytest.approx(np.array([[1800.0, 2800.0]]), abs=0.01)


def test_a_ray_skimming_under_the_slab_comes_back_into_it():
    # From inside the slab, a little down: it leaves through the base, passes
    # above the ground, and climbs back in far away.
    origin, d = _ray(2700.0, -3.0)
    (a0, a1), (b0, b1) = _nonempty(slab_segments(origin, d, R, SLAB))
    assert a0 == 0.0
    assert 0.0 < a1 < b0 < b1


def test_the_scene_clips_the_segments():
    origin, d = _ray(300.0, 90.0)
    assert _length(slab_segments(origin, d, R, SLAB, t_end=2400.0)) == pytest.approx(
        500.0, abs=0.01
    )
    assert _length(slab_segments(origin, d, R, SLAB, t_end=1000.0)) == 0.0
    # The far range starts at its near plane
    assert _length(slab_segments(origin, d, R, SLAB, t_start=2500.0)) == pytest.approx(
        400.0, abs=0.01
    )


# ── The steps ────────────────────────────────────────────────────────────────


def test_the_shortest_step_is_a_share_of_the_finest_feature():
    field = PRESETS[CloudType.CUMULUS].field
    finest = NOISE_FEATURE / (field.noise_scale[0] * field.octave_scales[-1])
    assert min_step(field) == pytest.approx(MIN_STEP_FEATURE * finest)


@pytest.mark.parametrize(
    "altitude,elevation", [(300, 90), (300, 3), (2700, 0), (5000, -20)]
)
def test_the_steps_cover_the_segments_within_the_budget(altitude, elevation):
    origin, d = _ray(altitude, elevation)
    segments = slab_segments(origin, d, R, SLAB)
    for budget in (24, 64, 96):
        t, dt = march_steps(segments, 50.0, STEP_GROWTH, budget)
        assert len(t) <= budget + 8
        assert dt.sum() == pytest.approx(_length(segments), rel=1e-6)


def test_the_steps_lengthen_with_distance_only_beyond_the_knee():
    t, dt = march_steps(np.array([[0.0, 200e3], [1.0, -1.0]]), 50.0, 0.01, 10**6)
    assert dt[t < 4000.0] == pytest.approx(50.0)
    far = t > 10e3
    far[-1] = False  # the last step ends with the segment
    assert np.all(dt[far] >= 0.01 * (t[far] - dt[far]) * 0.999)


def test_a_ray_that_fits_is_not_stretched():
    segments = np.array([[0.0, 1000.0], [1.0, -1.0]])
    assert step_scale(segments, 50.0, 0.01, 64) == 1.0
    assert step_scale(segments, 50.0, 0.01, 10) > 1.0


def test_a_short_segment_still_takes_its_fewest_samples():
    # From orbit, the slab is a sliver of a ray thousands of km long
    origin, d = _ray(2 * R, -90.0)
    segments = slab_segments(origin, d, R, SLAB)
    t, dt = march_steps(segments, 50.0, 0.5, 64)
    assert len(t) >= SEGMENT_SAMPLES
    assert dt.max() <= (segments[0, 1] - segments[0, 0]) / SEGMENT_SAMPLES + 1e-6


# ── Compositing ──────────────────────────────────────────────────────────────


def test_an_empty_ray_lets_everything_through():
    transmittance, weights = march_transmittance(np.zeros(5), np.full(5, 10.0))
    assert transmittance == 1.0
    assert weights.sum() == 0.0


def test_the_weights_and_the_transmittance_share_the_pixel():
    rng = np.random.default_rng(2)
    extinction = rng.uniform(0.0, 0.02, 40)
    transmittance, weights = march_transmittance(extinction, np.full(40, 50.0))
    assert weights.sum() + transmittance == pytest.approx(1.0)


def test_a_uniform_medium_is_exact_at_any_step():
    # exp2 of the summed depth, whatever the steps: the integration is
    # energy-conserving (each step's alpha, not depth times length)
    for n in (3, 17, 200):
        transmittance, _ = march_transmittance(np.full(n, 0.004), np.full(n, 600.0 / n))
        assert transmittance == pytest.approx(2.0**-2.4)


@pytest.fixture(scope="module")
def field():
    raw = replace(PRESETS[CloudType.CUMULUS].field, slab=SLAB)
    peaks = sphere_column_peaks(_noise(7), raw, R)
    return with_threshold(raw, peaks, fbm_sigma(_noise(7), raw)), peaks


def test_the_march_converges_on_the_fine_integral(field):
    field, peaks = field
    table = shift_table(peaks, field.threshold[0])
    cube = np.full((6, 4, 4), coverage_to_shift(0.5, table), dtype=np.float32)
    rng = np.random.default_rng(3)
    errors = []
    for _ in range(40):
        lon = rng.uniform(0, 2 * math.pi)
        origin = np.array([math.cos(lon), math.sin(lon), 0.0]) * (R + 1300.0)
        up = origin / np.linalg.norm(origin)
        side = np.cross(up, [0.0, 0.0, 1.0])
        e = math.radians(rng.uniform(20.0, 90.0))
        d = math.cos(e) * side / np.linalg.norm(side) + math.sin(e) * up
        segments = slab_segments(origin, d, R, SLAB)

        def transmittance(budget, jitter=0.5):
            t, dt = march_steps(segments, 50.0, STEP_GROWTH, budget, jitter)
            points = origin + t[:, None] * d
            density = density_on_sphere(_noise(7), cube, field, points, R)
            return march_transmittance(field.density * density, dt)[0]

        reference = transmittance(4096)
        # The shader's jitter: a pixel's estimate is unbiased over it
        mean = np.mean([transmittance(64, j) for j in (0.125, 0.375, 0.625, 0.875)])
        errors.append(abs(mean - reference))
    assert np.mean(errors) < 0.03
    assert np.max(errors) < 0.15


# ── The shader and Python agree ──────────────────────────────────────────────


def test_the_shader_shares_the_layer_count_and_the_feature_size():
    text = SHADER.read_text()
    assert int(re.search(r"MAX_MARCH_LAYERS = (\d+);", text).group(1)) == MAX_LAYERS
    units = float(re.search(r"NOISE_FEATURE_UNITS = ([\d.]+);", text).group(1))
    assert units == NOISE_FEATURE
    fewest = float(re.search(r"SEGMENT_SAMPLES = ([\d.]+);", text).group(1))
    assert fewest == SEGMENT_SAMPLES


# ── Settings and the renderer choice ─────────────────────────────────────────


def test_the_presets_trade_steps_and_resolution():
    levels = [CloudSettings.preset(level) for level in QualityLevel]
    steps = [s.march_steps for s in levels]
    assert steps == sorted(steps)
    assert all(s.renderer == "march" for s in levels)
    assert CloudSettings().renderer == "march"
    with pytest.raises(ValueError):
        CloudSettings(renderer="splats")


@pytest.fixture
def executor():
    made = ChunkExecutor(backend="inline", inline_budget=64)
    yield made
    made.shutdown()


def _clouds(showbase, executor, renderer):
    return PlanetClouds(
        showbase.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(shells=2, renderer=renderer),
        planet=PLANETS["forest"],
    )


def test_the_settings_pick_the_renderer(showbase, executor):
    clouds = _clouds(showbase, executor, "march")
    try:
        assert isinstance(clouds.volume, CloudMarch)
        assert clouds.volume.root.get_parent() == clouds.root
    finally:
        clouds.remove()


def test_a_change_of_renderer_rebuilds_the_volume_with_its_state(showbase, executor):
    clouds = _clouds(showbase, executor, "billboards")
    try:
        clouds.set_sun((0.0, 2.0, 0.0))
        clouds.set_sky_light(0.2)
        clouds.set_debug(1)
        old = clouds.volume
        clouds.set_settings(replace(clouds.settings, renderer="march"))
        assert isinstance(clouds.volume, CloudMarch)
        assert old.root.is_empty()  # removed
        assert tuple(clouds.volume._sun[0]) == pytest.approx((0.0, 1.0, 0.0))
        assert clouds.volume._inputs["skyStrength"] == 0.2
        assert clouds.volume._inputs["uCloudDebug"] == 1
        clouds.set_settings(replace(clouds.settings, renderer="billboards"))
        assert isinstance(clouds.volume, CloudVolume)
    finally:
        clouds.remove()


def test_the_march_packs_a_layer_per_shell_and_the_far_one(showbase, executor):
    clouds = _clouds(showbase, executor, "march")
    try:
        march = clouds.volume
        assert march._layers_used == 3
        assert march._inputs["uDeckLayer"] == 2
        far = [march._layer_far[i] for i in range(2)]
        assert 0.0 < far[0] < far[1]
        # The steps are live
        clouds.set_settings(replace(clouds.settings, march_steps=17))
        assert march._steps[0] == 17
        assert clouds.volume is march
    finally:
        clouds.remove()


def test_the_march_casts_a_shadow_and_reads_the_maps_for_its_own_light(
    showbase, executor
):
    """A raymarched volume is a shadow caster as the billboards' is (its far
    layer's field, the layer bound once the set-up is done) and takes the
    shadow maps' inputs on its passes' nodes, kept across a change of
    renderer; the shader's maps path is compiled in for the march alone."""
    clouds = _clouds(showbase, executor, "march")
    try:
        march = clouds.volume
        assert clouds.shadow_caster(0) is march.shadow_caster(0)
        assert clouds.shadow_families == march.families == 1
        assert march.shadow_field.slab == march.field.slab
        node = showbase.render.attach_new_node("receiver")
        clouds.bind_shadow_inputs(node)
        clouds.update((0.0, 0.0, 2000.0 + 600e3))
        for _ in range(200):  # the set-up, in the pool
            if clouds.volume.weather is not None:
                break
            clouds.update((0.0, 0.0, 2000.0 + 600e3))
        assert march.shadow_layer == march._layers_used - 1
        assert clouds.shadow_layer == march.shadow_layer
        bound = []
        clouds.bind_maps(lambda n: bound.append(n))
        assert bound == march._nodes
        clouds.set_settings(replace(clouds.settings, renderer="billboards"))
        assert clouds.shadow_caster(0) is clouds.volume
        assert not isinstance(clouds.volume, CloudMarch)
    finally:
        clouds.remove()
    shaders = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"
    assert "#define CLOUD_MAP_LIGHTING" in (shaders / "cloud_march.frag").read_text()
    for name in ("cloud.frag", "cloud_deck.frag"):
        path = shaders / name
        if path.exists():
            assert "CLOUD_MAP_LIGHTING" not in path.read_text()


def test_two_families_slabs_make_up_to_four_sorted_stretches():
    """The genera's stratus under the convective clouds: a ray from the ground
    crosses the stratus then the convective slab; from above, the convective
    one then the stratus, then (the planet not in the way) nothing; a ray
    skimming under both leaves and re-enters each."""
    from world_builder.clouds.march import family_segments

    radius = 600e3
    slabs = [(2200.0, 10500.0), (900.0, 1000.0)]  # convective, stratus
    up = np.array([0.0, 0.0, 1.0])
    below = family_segments(up * (radius + 100.0), up, radius, slabs)
    assert [f for _, _, f in below] == [1, 0]
    assert below[0][0] == pytest.approx(800.0) and below[0][1] == pytest.approx(1800.0)
    assert below[1][0] == pytest.approx(2100.0)
    assert below[1][1] == pytest.approx(12600.0)
    above = family_segments(up * (radius + 20e3), -up, radius, slabs)
    assert [f for _, _, f in above] == [0, 1]
    assert above[0][1] <= above[1][0]
    # A ray passing under the base of both (its closest approach 500 m up):
    # each slab twice, four stretches in order along it, the stratus's
    # outside the convective ones' (it is lower)
    side = np.array([1.0, 0.0, 0.0])
    skimming = family_segments(
        np.array([-200e3, 0.0, radius + 500.0]), side, radius, slabs
    )
    assert [f for _, _, f in skimming] == [0, 1, 1, 0]
    starts = [t0 for t0, _, _ in skimming]
    assert starts == sorted(starts)
    # The planet cuts the far side, and the view's range the near
    clipped = family_segments(
        up * (radius + 20e3), -up, radius, slabs, t_start=9e3, t_end=15e3
    )
    assert all(9e3 <= t0 < t1 <= 15e3 for t0, t1, _ in clipped)


def test_the_jitters_blue_noise_has_every_rank_once_and_no_low_frequencies():
    """The steps' jitter reads a tileable blue noise (void and cluster): each
    value once, almost no power at low frequencies (white noise has a few
    per cent of it in the lowest tenth of the band), the same pattern wrapped."""
    from world_builder.clouds.march import BLUE_NOISE_SIZE, blue_noise

    noise = blue_noise()
    assert noise.shape == (BLUE_NOISE_SIZE, BLUE_NOISE_SIZE)
    assert len(np.unique(noise)) == BLUE_NOISE_SIZE**2
    assert 0.0 <= noise.min() and noise.max() < 1.0
    power = np.abs(np.fft.fft2(noise - noise.mean())) ** 2
    freq = np.fft.fftfreq(BLUE_NOISE_SIZE)
    low = (freq[:, None] ** 2 + freq[None, :] ** 2) < 0.01
    assert power[low].sum() / power.sum() < 1e-3


def test_the_far_detail_fades_more_the_fewer_pixels_a_frame_is_traced():
    from world_builder.clouds.march import step_fade_of

    assert step_fade_of(1) == 0.0
    assert 0.0 <= step_fade_of(2) < step_fade_of(4) < 0.4


def test_the_resolve_smooths_more_the_fewer_pixels_a_frame_is_traced():
    from world_builder.clouds.march import smoothing_of

    assert smoothing_of(1) == 0.0
    assert 0.0 < smoothing_of(2) < smoothing_of(4) == 1.0


# ── Gathering over the frames ────────────────────────────────────────────────


@pytest.mark.parametrize("resolution,block", [(1.0, 1), (0.5, 2), (0.25, 4)])
def test_each_frame_traces_one_pixel_of_a_block_and_a_cycle_all(resolution, block):
    assert block_of(resolution) == block
    cycle = [bayer_offset(block, frame) for frame in range(block * block)]
    assert sorted(cycle) == [(x, y) for x in range(block) for y in range(block)]
    # And then again, in the same order
    assert bayer_offset(block, block * block) == cycle[0]


@pytest.mark.parametrize("block", [1, 2, 4])
def test_each_visit_moves_the_jitter_on_once(block):
    """A pixel traced once a cycle of block x block frames runs through the
    jitter's sequence one step a visit, not one a frame (it would skip
    through it, its visits half a step apart and never stratified)."""
    cycle = block * block
    traced = [f for f in range(8 * cycle) if bayer_offset(block, f) == (0, 0)]
    assert [visit_of(block, f) for f in traced] == list(range(8))
    golden = 0.618034
    seen = sorted((visit * golden) % 1.0 for visit in range(8))
    assert max(np.diff(seen)) < 0.25  # eight visits spread over the step
    assert visit_of(block, 7, gather=False) == 7


def test_neighbours_come_far_apart_in_time():
    cycle = [bayer_offset(4, frame) for frame in range(16)]
    for a, b in zip(cycle, cycle[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 1
        assert a != b
    # Consecutive frames never trace side by side within one row
    assert all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1 for a, b in zip(cycle, cycle[1:])
    )


def test_the_view_ray_takes_its_own_extinction(showbase, executor):
    clouds = _clouds(showbase, executor, "march")
    try:
        march = clouds.volume
        # layerParams row 5: (view extinction, the field's own, ...)
        view, field = march._layer_params[5][0], march._layer_params[5][1]
        assert view == pytest.approx(field * VIEW_EXTINCTION_SCALE)
    finally:
        clouds.remove()


def test_a_forgotten_half_res_pass_is_not_given_the_next_volume(showbase, executor):
    clouds = _clouds(showbase, executor, "billboards")
    try:
        half_res = MagicMock()
        clouds.use_half_res(half_res)
        half_res.add.assert_called_once_with(clouds.volume.root)
        clouds.use_half_res(None)
        half_res.remove.assert_called_once()
        clouds.set_settings(replace(clouds.settings, renderer="march"))
        clouds.set_settings(replace(clouds.settings, renderer="billboards"))
        assert half_res.add.call_count == 1
    finally:
        clouds.remove()
