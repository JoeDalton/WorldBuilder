"""
The sea's look and what it costs, shared by :class:`world_builder.ocean.Sea`
(which draws it on the terrain's chunks) and the shaders.

:class:`OceanSettings` is what the sea costs, with LOW / MID / HIGH / ULTRA
presets (:mod:`world_builder.quality`). The rest are the CPU halves of the
shading: the wave directions, the variance of the octaves a pixel cannot
resolve, the rough fresnel table and the tonemap the reflection is decoded
with. :func:`bind_sea_defaults` gives a node every input sea.glsl reads, so the
terrain's shader runs without a sea.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    LMatrix4f,
    LVecBase2f,
    LVecBase3f,
    NodePath,
    PTA_float,
    PTA_LVecBase2f,
    SamplerState,
    Texture,
)

from world_builder.quality import REBUILD, QualityLevel, QualitySettings, knob

# Max wave iterations the shader's iWaveDirs[] array can hold (must match the
# MAX_WAVE_ITER #define in sea.glsl).
_MAX_WAVE_ITER = 64
# sea.glsl's small-wave constants the CPU side needs (keep them equal).
_WAVE_HEIGHT = 10.0
_WEIGHT_DECAY = 0.8  # weight = mix(weight, 0, 0.2) per octave
_FREQ_GROWTH = 1.18
_TIME_GROWTH = 1.07  # timeMultiplier *= 1.07 per octave, from 2
#: sea.glsl's WAVE_SCALE: the waves' domain per metre of the sea
WAVE_SCALE = 0.5
#: Mean square of one octave's slope, (exp(sin x - 1) cos x)^2 over a period
#: (sea.glsl's C_WAVE).
C_WAVE = 0.1076346
#: The fresnel table's size and the slope spread its last row is at
#: (sea.glsl's FRESNEL_LUT_SIZE and FRESNEL_SLOPE_MAX).
FRESNEL_LUT_SIZE = 32
FRESNEL_LUT_SLOPE_MAX = 2.0


@dataclass(frozen=True)
class OceanSettings(QualitySettings):
    """What the sea costs (:mod:`world_builder.quality`). All live but
    ``reflection_scale``, which rebuilds the reflection buffer.

    :ivar wave_iterations: most small-wave octaves a pixel evaluates (the
        filter stops sooner where they are sub-pixel)
    :ivar aa_full_px: an octave is at full strength while its wavelength
        spans at least this many pixels
    :ivar aa_gone_px: and is gone below this many (Nyquist is 2; at most
        ``aa_full_px``). Larger values drop octaves sooner: softer, cheaper
    :ivar rough_fresnel: average the fresnel over the slopes the filter took
        away (one small texture lookup); off, distant water is shaded as a
        flatter mirror
    :ivar reflection_blur: blur the reflection by those slopes, through its
        mipmaps (regenerated every frame); 0 = a sharp mirror and no mipmaps,
        1 = over all the ripples would have moved it (0.25 keeps the
        reflections readable at distance)
    :ivar depth_colour: tint the water over shallow seabed
    :ivar foam: shoreline foam
    :ivar foam_noise: break the foam up with noise; off, plain lines
    :ivar swell_by_depth: the swell dies out over the shallows
    :ivar reflection_scale: the reflection's resolution, as a fraction of
        the view's. Rebuild
    :ivar reflection_fov_scale: the reflection camera's field of view,
        relative to the main camera's (a margin against culling at the edges
        when turning)
    """

    wave_iterations: int = knob(
        36,
        "Wave detail",
        unit="octaves",
        range=(1, _MAX_WAVE_ITER),
        help="Most wave octaves a pixel evaluates near the camera.",
    )
    aa_full_px: float = knob(
        8.0,
        "Wave sharpness",
        unit="px",
        range=(2.0, 32.0),
        help="A wave keeps full strength while its wavelength spans this many "
        "pixels: higher is softer and cheaper.",
    )
    aa_gone_px: float = knob(
        4.0,
        "Wave cut-off",
        unit="px",
        range=(2.0, 32.0),
        advanced=True,
        help="And fades out by this many (at most the sharpness).",
    )
    rough_fresnel: bool = knob(
        True,
        "Distant water roughness",
        help="Distant water blends sky and water colour as rough water does "
        "(one texture read); off, it is a flatter mirror.",
    )
    reflection_blur: float = knob(
        0.25,
        "Reflection blur",
        range=(0.0, 1.0),
        help="Blur the reflection where the waves are too small to see (its "
        "mipmaps, rebuilt every frame); 0 = a sharp mirror.",
    )
    depth_colour: bool = knob(True, "Shallow water colour")
    foam: bool = knob(True, "Shoreline foam")
    foam_noise: bool = knob(True, "Foam detail", advanced=True)
    swell_by_depth: bool = knob(
        True, "Swell calms in shallows", help="The swell dies out near the shore."
    )
    reflection_scale: float = knob(
        0.5,
        "Reflection quality",
        REBUILD,
        unit="x",
        range=(0.1, 1.0),
        help="The sea reflection's resolution, as a fraction of the view's.",
    )
    reflection_fov_scale: float = knob(
        1.1,
        "Reflection margin",
        unit="x",
        range=(1.0, 2.0),
        advanced=True,
        help="The reflection camera's field of view over the main camera's: "
        "a margin against missing reflections at the screen's edges.",
    )

    LABEL: ClassVar = "Ocean"
    PRESETS: ClassVar = {
        QualityLevel.LOW: {
            "wave_iterations": 16,
            "aa_full_px": 12.0,
            "aa_gone_px": 6.0,
            "rough_fresnel": False,
            "reflection_blur": 0.0,
            "foam_noise": False,
            "reflection_scale": 0.25,
        },
        QualityLevel.MID: {
            "wave_iterations": 24,
            "aa_full_px": 10.0,
            "aa_gone_px": 5.0,
            "reflection_scale": 0.35,
        },
        QualityLevel.ULTRA: {
            "wave_iterations": 48,
            "aa_full_px": 6.0,
            "aa_gone_px": 3.0,
            "reflection_scale": 0.75,
        },
    }


#: Pre-tonemap exposure; the sky's and the terrain's default, which the
#: reflection's clear colour assumes.
DEFAULT_EXPOSURE = 0.9

# The shaders' ACES fit, as row-major matrices (GLSL's mat3 literals are
# column-major, so these are their transposes).
_ACES_M1 = np.array(
    [
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ]
)
_ACES_M2 = np.array(
    [
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602],
    ]
)


def aces_tonemap(color: ArrayLike, exposure: float = DEFAULT_EXPOSURE) -> np.ndarray:
    """The shaders' ``aces_tonemap(color * exposure)``: linear light to the
    display colour (gamma 1/2.2) that the sky and terrain write.

    :param color: linear (r, g, b)
    :param exposure: pre-tonemap exposure
    :return: display (r, g, b) in [0, 1]
    """
    v = _ACES_M1 @ (np.asarray(color, dtype=np.float64) * exposure)
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return np.clip(_ACES_M2 @ (a / b), 0.0, 1.0) ** (1.0 / 2.2)


def compute_wave_dirs(wind_dir: LVecBase2f, wind_strength: float) -> PTA_LVecBase2f:
    """Precompute the per-iteration wave directions the fragment shader uses.

    These depend only on the iteration index and the wind, not on the pixel, so
    we build them once on the CPU and pass them as a uniform array instead of
    recomputing sin/cos/normalize/mix per pixel per iteration in the shader.
    Mirrors the original in-shader formula exactly.

    :param wind_dir: Wind direction in the XY plane (normalised).
    :param wind_strength: How strongly directions align to the wind; 0 =
        random, 1 = fully aligned.
    :return: One normalised wave direction per shader iteration.
    """
    pta = PTA_LVecBase2f()
    it = 0.0
    ws = wind_strength
    for _ in range(_MAX_WAVE_ITER):
        rx, ry = math.sin(it), math.cos(it)
        # mix(rndDir, wind_dir, ws)
        mx = rx * (1.0 - ws) + wind_dir.x * ws
        my = ry * (1.0 - ws) + wind_dir.y * ws
        length = max(math.hypot(mx, my), 1e-5)
        pta.pushBack(LVecBase2f(mx / length, my / length))
        it += 1232.399963
    return pta


def wave_directions(wind_dir: ArrayLike, wind_strength: float) -> np.ndarray:
    """:func:`compute_wave_dirs` as an array, for the CPU's wave phases.

    :returns: (_MAX_WAVE_ITER, 2) float64 unit directions
    """
    dirs = compute_wave_dirs(LVecBase2f(*wind_dir), wind_strength)
    return np.array([(d.x, d.y) for d in dirs], dtype=np.float64)


def wave_phases(directions: np.ndarray, chart_xy: ArrayLike, time: float) -> np.ndarray:
    """Each small-wave octave's phase at a point of the sea's chart, at a time:
    what sea.glsl adds to its local part (``uWavePhase``), in float64 so the
    finest octaves keep their phase far from the chart's anchor.

    :param directions: (_MAX_WAVE_ITER, 2) the octaves' directions
        (:func:`wave_directions`)
    :param chart_xy: the point's chart coordinates, metres
    :param time: seconds
    :returns: (_MAX_WAVE_ITER,) radians, in [0, 2 pi)
    """
    i = np.arange(len(directions))
    frequency = _FREQ_GROWTH**i
    speed = 2.0 * _TIME_GROWTH**i
    point = np.asarray(chart_xy, dtype=np.float64) * WAVE_SCALE
    return np.mod(frequency * (directions @ point) + speed * time, 2.0 * math.pi)


def wave_weight_sum(iterations: int) -> float:
    """:returns: the sum of the small-wave octave weights over *iterations*
    octaves (sea.glsl's ``uWaveWeightSum``)"""
    return (1.0 - _WEIGHT_DECAY**iterations) / (1.0 - _WEIGHT_DECAY)


def wave_tail_variance(iterations: int) -> np.ndarray:
    """The slope variance of small-wave octaves i.. *iterations* - 1, for
    each i, normalised as sea.glsl normalises the normal: what a pixel
    whose filter stops at octave i leaves to the rough fresnel.

    :param iterations: the octave count (``OceanSettings.wave_iterations``)
    :returns: (_MAX_WAVE_ITER,) float64, zero from *iterations* on
    """
    i = np.arange(_MAX_WAVE_ITER)
    amp = (_FREQ_GROWTH * _WEIGHT_DECAY) ** i  # frequency * weight
    per = C_WAVE * amp**2 * (_WAVE_HEIGHT / wave_weight_sum(iterations)) ** 2
    per[iterations:] = 0.0
    return np.cumsum(per[::-1])[::-1]


def rough_fresnel_table(
    size: int = FRESNEL_LUT_SIZE,
    slope_max: float = FRESNEL_LUT_SLOPE_MAX,
    nodes: int = 24,
) -> np.ndarray:
    """Schlick's fresnel averaged over the facets of a rough surface, as seen:
    slopes Gaussian about the resolved normal (spread s, the rms slope
    magnitude), each facet weighted by its area projected toward the eye
    (facets turned away are hidden). sea.glsl reads it by (N.V, s).

    :param size: entries per axis
    :param slope_max: the spread of the last row
    :param nodes: Gauss-Hermite nodes per slope axis
    :returns: (size, size) float32, rows by spread, columns by N.V
    """
    x, w = np.polynomial.hermite.hermgauss(nodes)
    sx, sy = np.meshgrid(x, x)
    weight = np.outer(w, w)
    cos_v = np.linspace(0.0, 1.0, size)
    table = np.empty((size, size))
    for row, spread in enumerate(np.linspace(0.0, slope_max, size)):
        # Gauss-Hermite integrates exp(-x^2): scale to a per-axis variance
        # of spread^2 / 2.
        dx, dy = sx * spread, sy * spread
        inv = 1.0 / np.sqrt(1.0 + dx * dx + dy * dy)
        for col, c in enumerate(cos_v):
            # Facet normal (-dx, -dy, 1) * inv; eye toward (sin, 0, cos).
            cos_f = (-dx * math.sqrt(1.0 - c * c) + c) * inv
            seen = weight * np.maximum(cos_f, 0.0) / inv
            total = seen.sum()
            f = 0.04 + 0.96 * (1.0 - np.clip(cos_f, 0.0, 1.0)) ** 5
            table[row, col] = (seen * f).sum() / total if total > 1e-12 else 1.0
    return table.astype(np.float32)


def make_fresnel_texture() -> Texture:
    """:returns: :func:`rough_fresnel_table` as a linear, clamped float texture"""
    table = rough_fresnel_table()
    tex = Texture("ocean_fresnel")
    tex.setup_2d_texture(table.shape[1], table.shape[0], Texture.T_float, Texture.F_r32)
    tex.set_ram_image(np.ascontiguousarray(table).tobytes())
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex


def bind_sea_defaults(node: NodePath) -> None:
    """Give *node* every input sea.glsl reads, with the sea's near look off
    (``uSeaNear`` 0): the terrain's shader then runs without a
    :class:`world_builder.ocean.Sea`, which sets them when there is one."""
    blank = Texture("sea_blank")
    blank.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_rgb8)
    blank.set_ram_image(bytes(3))
    zeros = PTA_float()
    for _ in range(_MAX_WAVE_ITER):
        zeros.push_back(0.0)
    dirs = compute_wave_dirs(LVecBase2f(1.0, 0.0), 0.5)
    values = {
        "iTime": 0.0,
        "uSeaOrigin": LVecBase3f(0.0, 0.0, 0.0),
        "uSeaEast": LVecBase3f(1.0, 0.0, 0.0),
        "uSeaNorth": LVecBase3f(0.0, 1.0, 0.0),
        "uSeaUp": LVecBase3f(0.0, 0.0, 1.0),
        "uSeaChart": LVecBase2f(0.0, 0.0),
        "uSeaCamForward": LVecBase3f(0.0, 1.0, 0.0),
        "uSeaPixelAngle": 0.001,
        "uSeaNear": 0.0,
        "iWaterColor": LVecBase3f(0.02, 0.06, 0.14),
        "iWaterLight": LVecBase3f(1.0, 1.0, 1.0),
        "iReflectionTex": blank,
        "iRippleStrength": 0.0,
        "uReflUVScale": LVecBase2f(1.0, 1.0),
        "uSeaReflMVP": LMatrix4f.ident_mat(),
        "iWindDir": LVecBase2f(1.0, 0.0),
        "iIterationsNormal": 1,
        "iWaveDirs": dirs,
        "uWavePhase": zeros,
        "uAAStart": 1.0,
        "uAAEnd": 2.0,
        "uWaveWeightSum": 1.0,
        "uTailVar": zeros,
        "uRoughFresnel": 0,
        "uFresnelLUT": blank,
        "uReflBlur": 0.0,
        "iSwellScale": 0.004,
        "uSwellRotA": LMatrix4f.ident_mat(),  # a mat3: Panda takes a mat4's
        "uSwellRotB": LMatrix4f.ident_mat(),
        "uSwellAmplitude": 0.0,
        "uFmDepth": 0.0,
        "uSeaDebug": 0,
        "uWaveOff": 0,
        "uDepthColourOn": 0,
        "uFoamOn": 0,
        "uFoamNoiseOn": 0,
        "uFoamColor": LVecBase3f(0.8, 0.8, 0.8),
        "uFoamDepth": 2.5,
        "uFoamSpacing": 0.6,
        "uFoamSpeed": 1.2,
        "uFoamStrength": 0.0,
        "uSwellByDepth": 0,
        "uSwellShallow": 5.0,
        "uSwellDeep": 60.0,
    }
    for name, value in values.items():
        node.set_shader_input(name, value)


def swell_rotation(axis: ArrayLike, angle: float) -> np.ndarray:
    """:returns: (3, 3) the rotation by *angle* (radians) about the unit
    *axis* (Rodrigues): what turns a swell layer, float64"""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross


def swell_rotations(
    up: ArrayLike,
    east: ArrayLike,
    north: ArrayLike,
    wind_dir: ArrayLike,
    scale: float,
    drift: float,
    radius: float,
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The swell layers' rotations at a time (swell.glsl's uSwellRotA, B):
    each turns about an axis through the planet's centre, so its pattern
    drifts across a reference point at the speed the flat swell drifted at,
    the first layer along the wind there, the second across it.

    :param up: the reference point's vertical (unit), planet frame
    :param east: its east
    :param north: its north
    :param wind_dir: the wind there, (east, north), unit
    :param scale: the first layer's frequency, cycles per metre
    :param drift: the drift, noise units a second
    :param radius: the planet's, metres
    :param time: seconds
    :returns: the two (3, 3) rotations, float64, applied as ``R @ point``
    """
    up = np.asarray(up, dtype=np.float64)
    wind = wind_dir[0] * np.asarray(east) + wind_dir[1] * np.asarray(north)
    across = np.cross(up, wind)
    a = swell_rotation(np.cross(up, wind), drift / scale / radius * time)
    b = swell_rotation(
        np.cross(up, across), 0.6 * drift / (1.7 * scale) / radius * time
    )
    return a, b
