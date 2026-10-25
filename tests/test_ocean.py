"""
Unit tests for the CPU halves of the sea's shading (world_builder.ocean): the
decoding of its reflection, its screen-space wave filtering and the wave
phases computed in float64. Headless: no window or GPU context.
"""

import math

import numpy as np
import pytest

from world_builder.ocean import OceanSettings, aces_tonemap
from world_builder.ocean.ocean import (
    C_WAVE,
    DEFAULT_EXPOSURE,
    FRESNEL_LUT_SLOPE_MAX,
    WAVE_SCALE,
    rough_fresnel_table,
    wave_directions,
    wave_phases,
    wave_tail_variance,
    wave_weight_sum,
)
from world_builder.quality import QualityLevel

# sea.glsl's aces_inverse, row-major (its GLSL literals are column-major).
_M1_INV = np.array(
    [
        [1.764741, -0.675778, -0.088963],
        [-0.147028, 1.160252, -0.013224],
        [-0.036337, -0.162436, 1.198773],
    ]
)
_M2_INV = np.array(
    [
        [0.643038, 0.311187, 0.045775],
        [0.059269, 0.931436, 0.009295],
        [0.005962, 0.063929, 0.930118],
    ]
)


def _aces_inverse(display):
    """A port of sea.glsl's aces_inverse: display colour to linear light."""
    w = np.minimum(_M2_INV @ np.clip(display, 0.0, 1.0) ** 2.2, 1.0)
    qa = 1.0 - 0.983729 * w
    qb = 0.0245786 - 0.4329510 * w
    qc = -(0.000090537 + 0.238081 * w)
    v = (-qb + np.sqrt(np.maximum(qb * qb - 4.0 * qa * qc, 0.0))) / (2.0 * qa)
    return np.maximum(_M1_INV @ v, 0.0)


@pytest.mark.parametrize("level", [0.01, 0.05, 0.13, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("tint", [(1.0, 1.0, 1.0), (0.6, 0.7, 1.0), (1.0, 0.5, 0.3)])
def test_reflection_decode_round_trips_the_display_colour(level, tint):
    """The reflection holds display colours (the sky tonemaps itself). Decoding
    one and tonemapping it again must give it back, or a full mirror shows a
    different sky: the night sky's reflection was twice as bright as the sky
    when the sample was tonemapped twice. (A tinted colour at full display is
    clipped, so not invertible; white at 1 is, below.)"""
    display = level * np.array(tint)
    linear = _aces_inverse(display) / DEFAULT_EXPOSURE
    np.testing.assert_allclose(aces_tonemap(linear), display, atol=2e-3)


def test_reflection_decode_keeps_white_highlights_bright():
    """Full white decodes to a finite, bright radiance that maps back to it."""
    linear = _aces_inverse(np.ones(3)) / DEFAULT_EXPOSURE
    assert np.all(np.isfinite(linear)) and np.all(linear > 10.0)
    np.testing.assert_allclose(aces_tonemap(linear), 1.0, atol=2e-3)


def test_double_tonemap_would_brighten_a_night_sky():
    """The bug the decode fixes: tonemapping the display colour again."""
    night = np.full(3, 0.13)
    assert aces_tonemap(night)[0] > 1.8 * night[0]


# ── Screen-space wave filtering ────────────────────────────────────────────────


def test_c_wave_is_the_mean_square_octave_slope():
    """sea.glsl's C_WAVE: (exp(sin x - 1) cos x)^2 averaged over a period."""
    x = np.linspace(0.0, 2.0 * math.pi, 200001)
    assert np.mean((np.exp(np.sin(x) - 1.0) * np.cos(x)) ** 2) == pytest.approx(
        C_WAVE, rel=1e-5
    )


@pytest.mark.parametrize("iterations", [1, 16, 36, 64])
def test_wave_weight_sum_matches_the_shader_loop(iterations):
    weight, total = 1.0, 0.0
    for _ in range(iterations):
        total += weight
        weight *= 0.8  # mix(weight, 0.0, 0.2)
    assert wave_weight_sum(iterations) == pytest.approx(total)


@pytest.mark.parametrize("iterations", [16, 36])
def test_tail_variance_is_the_unresolved_octaves_slope_variance(iterations):
    """uTailVar[i]: the normalised slope variance of octaves i.. n - 1."""
    norm = (10.0 / wave_weight_sum(iterations)) ** 2  # WAVE_HEIGHT / sum of weights
    per = [C_WAVE * (1.18**i * 0.8**i) ** 2 * norm for i in range(iterations)]
    tail = wave_tail_variance(iterations)
    for i in (0, 1, iterations // 2, iterations - 1):
        assert tail[i] == pytest.approx(sum(per[i:]))
    assert np.all(tail[iterations:] == 0.0)
    assert np.all(np.diff(tail) <= 0.0)
    # Its spread must fit the fresnel table, or distant water clamps to it.
    assert math.sqrt(tail[0]) <= FRESNEL_LUT_SLOPE_MAX


def test_rough_fresnel_is_schlick_on_a_smooth_surface():
    table = rough_fresnel_table()
    cos_v = np.linspace(0.0, 1.0, table.shape[1])
    np.testing.assert_allclose(table[0], 0.04 + 0.96 * (1.0 - cos_v) ** 5, atol=1e-6)


def test_rough_fresnel_softens_the_grazing_mirror():
    """Seen at a grazing angle, a rough sea shows its facets that face the eye,
    so it reflects less than a mirror; seen from above, its tilted facets
    reflect a little more."""
    table = rough_fresnel_table()
    assert np.all(np.diff(table[:, 0]) < 0.0)  # grazing: less with roughness
    assert table[-1, -1] > table[0, -1]  # from above: more
    assert np.all((table >= 0.04 - 1e-6) & (table <= 1.0))


def test_quality_presets_order_their_cost():
    levels = list(QualityLevel)
    assert OceanSettings.preset(QualityLevel.HIGH) == OceanSettings()
    for cheaper, dearer in zip(levels, levels[1:]):
        a, b = OceanSettings.preset(cheaper), OceanSettings.preset(dearer)
        assert a.wave_iterations <= b.wave_iterations
        # A cheaper preset drops octaves sooner (more pixels per wavelength).
        assert a.aa_gone_px >= b.aa_gone_px and a.aa_full_px >= b.aa_full_px
        assert a.reflection_scale <= b.reflection_scale
    for level in levels:
        settings = OceanSettings.preset(level)
        assert settings.aa_full_px > settings.aa_gone_px >= 2.0  # Nyquist


def test_wave_phases_carry_the_far_part_of_every_octave():
    """sea.glsl adds each octave's phase at the chart point (from here) to
    the local part it computes in float32: together they are the octave's
    full phase, frequency * direction . position + speed * time, mod 2 pi."""
    directions = wave_directions((0.6, 0.8), 0.5)
    origin = np.array([31234.5, -27890.25])  # tens of km from the anchor
    local = np.array([3.5, -12.0])
    time = 4321.0
    far = wave_phases(directions, origin, time)
    i = np.arange(len(directions))
    frequency = 1.18**i
    near = frequency * (directions @ (local * WAVE_SCALE))
    full = frequency * (directions @ ((origin + local) * WAVE_SCALE))
    full += 2.0 * 1.07**i * time
    np.testing.assert_allclose(
        np.mod(far + near - full + math.pi, 2.0 * math.pi) - math.pi, 0.0, atol=1e-6
    )
    assert np.all((far >= 0.0) & (far < 2.0 * math.pi))
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
