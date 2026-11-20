"""
A planet's low clouds, raymarched (:class:`CloudMarch`): the same density
field, weather, lighting and far field as the billboards and the deck
(:mod:`sphere`), drawn without their overdraw.

Billboards are a raymarch whose steps are quads: every one of them is drawn,
shaded and blended, so their cost grows with the cloud's volume on screen
(15 ms inside the slab on a laptop's GPU). Here each pixel steps along its
own ray instead (``shaders/cloud_march.frag``):

* through the **slab** between two spheres, less the planet: up to two
  segments, from the near plane to the scene's depth, so the ground
  and a game's models inside a cloud are hidden by just the cloud in front
  of them, and the camera inside a cloud sees it all around;
* in **steps** :data:`MIN_STEP_FEATURE` of the finest feature long near the
  camera, a share of the distance (:data:`STEP_GROWTH`) farther, all
  lengthened alike when a ray would need more than the budget
  (``CloudSettings.march_steps``), each jittered within its length. No
  step is longer in clear air: one longer than the finest features samples
  thin cloud by chance, a chance the jitter changes every frame, which reads
  as flicker and ragged edges (and the field's fine octaves leave no bound
  that would prove clear air clear);
* sampling the **field** with the octaves the billboard shells would have
  there (``CloudSettings.shell_octaves`` over the shells' distance bands),
  and stopping once the cloud is opaque;
* lit per sample as the billboards are (``shaders/cloud_shading.glsl``), the
  air applied once per pixel at the cloud's mean distance;
* **far** away, where a pixel spans more than the features, the deck's
  expected coverage as a homogeneous slab, lit by the deck's averages: the
  march reaches orbit with no deck.

**The passes.** One march over both depth ranges of
:class:`~world_builder.depth.SplitDepthCameras` (as EVE-Redux marches over
KSP's two cameras' depths), at a fraction of the view's resolution
(``CloudSettings.march_block``): a camera per range, sharing its lens,
draws the scene's depth into a texture (everything but the clouds and the
sky, the terrain through its depth-only stage, as :mod:`half_res` does); a
fullscreen card marches from the near range's near plane to the near
range's surface where there is one, else the far range's, into a second
buffer, and stops once opaque (a near cloud spares the march what lies
behind it). The result is gathered at full resolution, then laid over the
view by two cards: the near range's over the pixels its depth pass drew
(after the near camera's own surfaces, which the clouds may be in front
of), the far range's over the rest in the far pass's clouds' slot (after
the sky, before the cirrus). The near range's depth pass and card run only
while the slab reaches into it.

The sea's reflection sees the deck alone (:func:`deck.deck_mesh`, the
billboards' far field), the far layer packed for it.

Design notes: docs/source/clouds.md, "The raymarch".
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    Camera,
    CardMaker,
    ColorBlendAttrib,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    Lens,
    LMatrix4f,
    LVecBase2f,
    LVecBase2i,
    LVecBase3f,
    LVecBase4f,
    NodePath,
    OmniBoundingVolume,
    OrthographicLens,
    PTA_float,
    PTA_int,
    PTA_LMatrix4f,
    PTA_LVecBase2i,
    PTA_LVecBase3f,
    PTA_LVecBase4f,
    RenderState,
    SamplerState,
    Shader,
    ShaderAttrib,
    Texture,
    TransparencyAttrib,
    Vec3,
    Vec4,
    WindowProperties,
)

from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.clouds.cloud import (
    PRESETS,
    CloudSettings,
    CloudType,
    scale_march,
)
from world_builder.clouds.deck import (
    CLOUD_BIN,
    _ensure_bin,
    deck_mesh,
    deck_shader,
)
from world_builder.clouds.field_setup import (
    DECK_MAX_OPTICAL_DEPTH,
    OctaveBand,
    _cells,
    _coverage_cube,
    _deck_light,
    _depth_tables,
    _genus_weather,
    _noise,
    _peaks,
    _recipe_deck_light,
    _weather,
    genus_field,
    octave_bands,
    octave_offset,
    planet_field,
    planet_weather,
    recipe_peaks,
    sun_visibility,
)
from world_builder.clouds.genera import (
    CONVECTIVE,
    GENUS_PRESETS,
    HIGH_CHANNELS,
    HIGH_OPTICS,
    HIGH_SLAB,
    GeneraSpec,
    genus_specs,
    high_genera,
    high_jet,
    high_wind_cube,
    weather_cirrus,
)
from world_builder.clouds.noise import (
    DECK_LAYER,
    DECK_LIGHT_COS_MIN,
    DETAIL_LEVELS,
    GENUS_VEC4S,
    LAYER_VEC4S,
    MAX_GENERA,
    NOISE_FEATURE,
    NOISE_SIZE,
    WORLEY_CELLS,
    DensityField,
    GenusField,
    _smoothstep,
    build_cells_texture,
    build_curl_texture,
    build_deck_light_array,
    build_deck_light_texture,
    build_noise_texture,
    cell_means,
    clamped_anisotropy,
    curl_noise_volume,
    pack_genus_params,
    pack_layer_params,
    pack_recipe_cells,
    pack_recipe_frame,
    phase_weights,
    recipe_resolved,
    sample_cells,
    sample_volume,
)
from world_builder.clouds.passes import composite_card, pass_camera, pass_states
from world_builder.clouds.shadow_defaults import (
    _no_coverage,
    _no_deck_light_array,
    _no_shift,
    cloud_shadow_defaults,
)
from world_builder.clouds.weather import (
    WeatherSpec,
    build_coverage_texture,
    build_high_wind_texture,
    build_shift_texture,
    build_weather_texture,
    coverage_to_shift,
    shift_table,
)
from world_builder.depth import (
    CLOUD_FAR_BIT,
    CLOUD_NEAR_BIT,
    FAR_BIT,
    HIDE_FROM_CLOUD_PASS,
    HIDE_FROM_FAR,
    HIDE_FROM_NEAR,
    NEAR_BIT,
    REFLECTION_CAMERA_MASK,
    SplitDepthCameras,
)
from world_builder.shadows import bind_defaults as bind_shadow_defaults
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.cubesphere import grid_for
from world_builder.terrain.planet import PlanetSpec

if TYPE_CHECKING:
    from world_builder.terrain.workers import ChunkExecutor

_SHADERS = Path(__file__).parent / "shaders"
#: The shortest step, near the camera, as a share of the finest octave's
#: feature (~125 m for cumulus: 50 m steps).
MIN_STEP_FEATURE = 0.4
#: Farther, a step's length per metre of distance from the camera.
STEP_GROWTH = 0.01
#: The fewest samples a stretch of the slab takes (cloud_march.frag's
#: SEGMENT_SAMPLES): a far ray crossing it steeply, in a few steps longer
#: than its clouds' edges are thick, would hit or miss thin cloud by the
#: jitter's chance, frame after frame (sparkle seen from 20 km up).
SEGMENT_SAMPLES = 8
#: The march stops at this transmittance and counts the cloud opaque
#: (cloud_march.frag's OPAQUE).
OPAQUE = 0.04


def opaque_residual(transmittance: float) -> float:
    """The light a ray lets through once composited (cloud_march.glsl's
    ``opaqueResidual``): its transmittance, eased to 0 over [:data:`OPAQUE`,
    2 :data:`OPAQUE`] (value and slope continuous), 0 below. Snapped from
    OPAQUE to 0, a thin sheet seen at a grazing angle, whose rays cross the
    cutoff gradually across the view, took a step of 0.04 in alpha there: a
    contour across the altostratus (clouds.md, "The recipe genera").
    """
    if transmittance <= OPAQUE:
        return 0.0
    if transmittance >= 2.0 * OPAQUE:
        return float(transmittance)
    x = transmittance - OPAQUE
    return float(3.0 * x - x * x / OPAQUE)


#: The view ray's extinction over the field's (the sun march keeps the
#: field's own). The billboards' look: each of their quads carries its
#: sample's density over a whole chord, which keeps a cloud's edge crisp;
#: the field's own soft edge, integrated, reads as a blur.
VIEW_EXTINCTION_SCALE = 2.0
#: The edges' billows, which the billboards' sprites carry (their alpha is a
#: lumpy thickness): a billowy noise this many times finer than the field's
#: first octave (~85 m for cumulus)...
DETAIL_SCALE = 24.0
#: ...moving the threshold by up to this share of its window about its mean
#: (the coverage holds). 1 breaks the edges into rags.
DETAIL_EROSION = 0.5
#: The billows bent by a curl noise (EVE-Redux bends its detail noise so):
#: how far, in the billows' noise units (a billow is NOISE_FEATURE of them,
#: about 4), and the curl's frequency over the billows' (its features some
#: two billows across). 0 = straight billows.
CURL_STRENGTH = 2.0
CURL_FREQUENCY = 2.0
#: The view's pixels per pixel of the depth passes, per axis: 2 draws the
#: scene's depth for the march at half resolution (the march reads it at
#: the pixels it traces, the composites to tell the near range's pixels).
DEPTH_SCALE = 2


def depth_size(view: tuple[int, int], scale: int = DEPTH_SCALE) -> tuple[int, int]:
    """:returns: the depth passes' size for a view of *view* pixels"""
    return tuple(max(1, math.ceil(v / scale)) for v in view)


#: How far along a ray the weather is read once, metres (cloud_march.frag's
#: ``uWeatherStretch``): the fair-weather shift varies over kilometres; with
#: genera, a tower's depth over tens of metres, so every sample reads it.
WEATHER_STRETCH = 1000.0
WEATHER_STRETCH_GENERA = 1.0
#: The march's buffers' sorts: before the view they are laid over (the
#: half-resolution pass's are -50 and -49).
_FIRST_SORT = -60


# ── The march's geometry (cloud_march.frag's, for the tests) ─────────────────


def sphere_interval(
    origin: ArrayLike, direction: ArrayLike, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Where rays ``origin + t * direction`` are inside a sphere of *radius*
    around the planet's centre (the shader's ``sphereInterval``).

    :param origin: (..., 3) planet frame, metres
    :param direction: (..., 3) unit
    :returns: (t0, t1), each (...): empty where t0 >= t1
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    b = np.einsum("...i,...i->...", origin, direction)
    q = np.linalg.norm(origin - direction * b[..., None], axis=-1)
    inside = q < radius
    h = np.sqrt(np.where(inside, (radius - q) * (radius + q), 0.0))
    return np.where(inside, -b - h, 1.0), np.where(inside, -b + h, -1.0)


def slab_segments(
    origin: ArrayLike,
    direction: ArrayLike,
    radius: float,
    slab: tuple[float, float],
    t_start: float = 0.0,
    t_end: float = math.inf,
) -> np.ndarray:
    """The stretches of a ray in a slab of sky: between the spheres at its
    base and top, before the ray meets the planet, within [*t_start*,
    *t_end*] (the shader's segments).

    :param origin: (3,) planet frame, metres
    :param direction: (3,) unit
    :param radius: the planet's sea-level radius, metres
    :param slab: (base, thickness) above sea level, metres
    :returns: (2, 2) [segment, (t0, t1)]: the second is empty (t0 >= t1)
        unless the ray leaves the slab downward and comes back
    """
    base, thickness = slab
    top = np.array(sphere_interval(origin, direction, radius + base + thickness))
    inner = np.array(sphere_interval(origin, direction, radius + base))
    ground = np.array(sphere_interval(origin, direction, radius))
    if ground[0] < ground[1] and ground[1] > 0.0:
        t_end = min(t_end, max(ground[0], 0.0))
    if inner[0] < inner[1]:
        segments = np.array([[top[0], inner[0]], [inner[1], top[1]]])
    else:
        segments = np.array([top, [1.0, -1.0]])
    segments[:, 0] = np.maximum(segments[:, 0], t_start)
    segments[:, 1] = np.minimum(segments[:, 1], t_end)
    return segments


#: The most sky bands the march takes a ray through (each up to two
#: stretches: cloud_march.frag's eight segments). MUST match the shader.
MAX_SKY_BANDS = 4


def _merged(slabs: list[tuple[float, float]]) -> list[list[float]]:
    """Slabs merged where they overlap or touch: [base, top] each, lowest
    first."""
    out: list[list[float]] = []
    for base, thickness in sorted(slabs):
        top = base + thickness
        if out and base <= out[-1][1]:
            out[-1][1] = max(out[-1][1], top)
        else:
            out.append([base, top])
    return out


def sky_bands(slabs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The genera's slabs merged into the bands of sky the march takes a ray
    through (overlapping or touching slabs are one band; a gap between two
    costs no steps), lowest first.

    A slab that bridges a gap (the nimbostratus's 0.6-5 km, between the low
    sheets' band and the convective clouds') does not merge the slabs on
    either side: its band is split at the gap's middle, each part taking
    the genera on its side and the bridge. A band's thickest slab is never
    a bridge (the convective clouds' 2.2-10.5 km holding mid-level sheets
    stays one band). Merged, the low sheets took a
    9.9 km band's shares, its light volume's slices (310 m, not 31) and its
    bakes' steps: their shading turned grainy and their ground shadow
    patchy (clouds.md, "The nimbostratus").

    :param slabs: each genus's (base, thickness) above sea level, metres
    :returns: each band's (base, thickness)
    :raises ValueError: for more than :data:`MAX_SKY_BANDS` bands
    """
    bands: list[tuple[float, float]] = []
    for base, top in _merged(slabs):
        inside = [s for s in slabs if s[0] >= base - 1e-6 and s[0] + s[1] <= top + 1e-6]
        # The bridges: the slabs without which the band falls apart, but
        # its thickest (a convective slab holding mid-level sheets: cut, a
        # tower would take two bands' steps, a seam up its face)
        thickest = max(inside, key=lambda s: s[1])
        bridges = [
            s
            for k, s in enumerate(inside)
            if s is not thickest and len(_merged(inside[:k] + inside[k + 1 :])) > 1
        ]
        cores = _merged([s for s in inside if s not in bridges]) if bridges else []
        cuts = [0.5 * (a[1] + b[0]) for a, b in zip(cores, cores[1:])]
        edges = [base, *cuts, top]
        bands += [(float(lo), float(hi - lo)) for lo, hi in zip(edges, edges[1:])]
    if len(bands) > MAX_SKY_BANDS:
        raise ValueError(f"{len(bands)} bands of sky; the march takes {MAX_SKY_BANDS}")
    return bands


def band_masks(
    slabs: list[tuple[float, float]], bands: list[tuple[float, float]]
) -> list[int]:
    """Which genera each sky band holds, as bit masks (bit *g*: genus *g*;
    cloud_field.glsl's ``uSkyBands`` z): those whose slab reaches into it
    (a bridge, :func:`sky_bands`, both of its bands).

    :param slabs: each genus's (base, thickness) above sea level, metres
    :param bands: :func:`sky_bands` of *slabs*
    """
    masks = []
    for base, thickness in bands:
        top = base + thickness
        masks.append(
            sum(
                1 << g
                for g, (b, t) in enumerate(slabs)
                if b < top - 1e-6 and b + t > base + 1e-6
            )
        )
    return masks


#: The high genera's bits of ``uGenusMask`` (:meth:`CloudMarch.set_genus_visible`),
#: past the low genera's: channel c of cube B (``genera.HIGH_CHANNELS``) has
#: bit ``HIGH_MASK_SHIFT + c - 1`` (``cloud_march.glsl``'s ``highCrossing``).
HIGH_MASK_SHIFT = MAX_GENERA
#: Every genus drawn: the low genera's bits and the high genera's.
ALL_GENERA_MASK = (1 << (HIGH_MASK_SHIFT + len(HIGH_CHANNELS))) - 1

#: The band a high-layer crossing is tagged with (:func:`band_segments`;
#: ``cloud_march.glsl``'s ``HIGH_BAND``). MUST match the shader.
HIGH_BAND = -1
#: The longest path through the high slab, in slab depths
#: (``THIN_GRAZING_PATH``): past it a veil
#: would saturate toward the horizon into a flat plate. MUST match the shader.
THIN_GRAZING_PATH = 4.0


def band_segments(
    origin: ArrayLike,
    direction: ArrayLike,
    radius: float,
    slabs: list[tuple[float, float]],
    t_start: float = 0.0,
    t_end: float = math.inf,
    high: tuple[float, float] | None = None,
) -> list[tuple[float, float, int]]:
    """The stretches of a ray in the genera's bands of sky (:func:`sky_bands`
    of *slabs*), and in the high layer's slab, sorted along the ray (the
    order the march composites them in). The high slab's crossings are
    :func:`slab_segments`' of it: above every band, its first (on the way
    down) is always before every band's stretch and its second (on the way
    up) after every one, so the shader takes them before and after its
    sorted segments, not among them.

    :param slabs: each genus's (base, thickness) above sea level, metres
    :param high: the high layer's slab (``genera.HIGH_SLAB``), above every
        band; None = none
    :returns: (t0, t1, band) of each non-empty stretch, the band its index in
        :func:`sky_bands`, or :data:`HIGH_BAND` for the high slab's
    """
    out = []
    tagged = list(enumerate(sky_bands(slabs)))
    if high is not None:
        tagged.append((HIGH_BAND, high))
    for band, slab in tagged:
        for t0, t1 in slab_segments(origin, direction, radius, slab, t_start, t_end):
            if t1 > t0:
                out.append((float(t0), float(t1), band))
    return sorted(out)


def thin_optical_depth(
    density: float, length: float, thickness: float, extinction: float, cap: float
) -> float:
    """A high-layer crossing's optical depth (exp2 units), as the cirrus
    sheet's: its mean density times the path, at most
    :data:`THIN_GRAZING_PATH` slab depths, saturating softly toward *cap*
    (``highCrossing``).

    :param density: the samples' mean density (1: a full veil)
    :param length: the crossing's length, metres
    :param extinction: optical depth per metre at density 1
    :param cap: what it saturates toward (``genera.HIGH_OPTICS``' cap)
    """
    path = min(length, THIN_GRAZING_PATH * thickness)
    return cap * (1.0 - math.exp(-extinction * density * path / cap))


def high_optical_depth(
    density: ArrayLike,
    length: float,
    thickness: float,
    extinction: ArrayLike,
    cap: ArrayLike,
) -> float:
    """A high-layer crossing's optical depth with every genus in it
    (``highCrossing``): the channels' depths before the cap add, and the
    cap is theirs weighted by their shares (one genus alone:
    :func:`thin_optical_depth`).

    :param density: (3,) each channel's mean density (cirrus,
        cirrostratus, cirrocumulus)
    :param extinction: (3,) each one's optical depth per metre at density 1
        (``genera.HIGH_OPTICS``' depth over the slab's thickness)
    :param cap: (3,) what each one's saturates toward
    """
    path = min(length, THIN_GRAZING_PATH * thickness)
    parts = np.asarray(extinction, np.float64) * np.asarray(density, np.float64) * path
    total = float(parts.sum())
    if total <= 0.0:
        return 0.0
    mixed = float(parts @ np.asarray(cap, np.float64)) / total
    return mixed * (1.0 - math.exp(-total / mixed))


def sheet_crossing(
    origin: ArrayLike, direction: ArrayLike, middle: float, t0: float, t1: float
) -> float | None:
    """Where a crossing (t0, t1) of the high slab meets its middle, the
    sphere of radius *middle* around the planet's centre (``highCrossing``'s
    sheet point): every high genus's pattern is read there, once, as a thin
    sheet's, not at jittered samples (a slant ray's samples would wander
    over the slab's 1.5 km across a grain's 200 m or a fibre's 180 m, and
    the visits would blur them). Where the crossing does not meet it (a
    camera inside the slab, looking away from its middle), None: the ray
    crosses no sheet, and the crossing draws nothing (the crossing's middle,
    once read there, drew a second, offset copy of the cirrus from inside
    the slab)."""
    c = np.asarray(origin, np.float64)
    d = np.asarray(direction, np.float64)
    b = float(c @ d)
    q = float(np.linalg.norm(c - d * b))
    if q < middle:
        h = math.sqrt((middle - q) * (middle + q))
        for t in (-b - h, -b + h):
            if t0 <= t <= t1:
                return t
    return None


#: The steepest a ray may meet the high slab in the footprint's stretch
#: (:func:`slab_footprint`): the cosine of its angle to the vertical. MUST
#: match the shader.
HIGH_MIN_INCIDENCE = 0.05


def slab_footprint(pixel: float, incidence: float) -> float:
    """A view pixel's footprint on the high layer's sheet, metres: its own
    stretched along the view by the slant, ``pixel / cos`` (the cosine of
    the ray's angle to the vertical, at least :data:`HIGH_MIN_INCIDENCE`).
    The cirrocumulus's grains and the cirrostratus's fibres fade by it: at a
    grazing angle a pixel spans many grains along the view though few
    across it (Task 20's view-pixel footprint alone aliased there)."""
    return pixel / max(abs(incidence), HIGH_MIN_INCIDENCE)


#: The cirrostratus's veil (``highSheets``; clouds.md, "The high layer"):
#: features this many metres across the jet and CS_STRETCH times longer
#: along it (the research's 50-200 km, stretched 5-10 times), faint fibres
#: CS_FIBRE metres across, CS_FIBRE_STRETCH times longer. MUST match the
#: shader.
CS_FEATURE = 25000.0
CS_STRETCH = 6.0
CS_FIBRE = 1500.0
CS_FIBRE_STRETCH = 12.0
#: The cirrocumulus's grains: Worley cells this many metres across (the
#: research's 100-300 m: under 1 degree from 11 km below, the WMO's), each
#: grain ``smoothstep(CC_GRAIN, 1 - F1)`` (round flakes, sky between), the
#: grains' fill moved by a sine ripple across the jet's crests, CC_RIPPLE
#: metres apart (the research's 0.5-1 km: undulatus). MUST match the shader.
CC_CELL = 200.0
CC_GRAIN = (0.5, 0.75)
CC_RIPPLE = 600.0


def cirrocumulus_grain_value(cells: np.ndarray, at: ArrayLike) -> np.ndarray:
    """The cirrocumulus's grains at *at* (metres, planet frame), resolved:
    ``smoothstep(CC_GRAIN, 1 - F1)`` of the Worley cells CC_CELL metres
    across (``highSheets``)."""
    f1 = sample_cells(cells, np.asarray(at, np.float64) / CC_CELL)
    return _smoothstep(*CC_GRAIN, 1.0 - f1)


@functools.cache
def cirrocumulus_grain_mean() -> float:
    """The grains' mean (``uHighShape.x``): what they fade to where a pixel
    does not resolve them, so a far patch keeps its optical depth. Over a
    lattice of points through the Worley volume's tile, offset off its
    texels."""
    cells = _cells()
    n = 48
    grid = (np.arange(n) + 0.37) * (WORLEY_CELLS / n)
    at = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    return float(cirrocumulus_grain_value(cells, at * CC_CELL).mean())


def cirrocumulus_grains(
    cells: np.ndarray, at: ArrayLike, footprint: ArrayLike
) -> np.ndarray:
    """The cirrocumulus's grains as the shader takes them: faded to their
    mean (:func:`cirrocumulus_grain_mean`) where a footprint does not
    resolve a cell (``resolved``: 2 footprints a cell none, 4 whole).

    :param footprint: metres (:func:`slab_footprint`)
    """
    grain = cirrocumulus_grain_value(cells, at)
    mean = cirrocumulus_grain_mean()
    return mean + (grain - mean) * recipe_resolved(CC_CELL, footprint)


#: The cirrus (``cirrusStreaks``; clouds.md, "The cirrus"): its noise in the
#: upper wind's frame, (width across the wind, length along it), metres: the
#: fibres (14-33 times longer than wide: the research's 8-20, and more), the
#: bundles they come in, their strands (the fibrous texture), the tufts (the
#: heads), the patches. MUST match the shader.
CI_FIBRE = (180.0, 6000.0)
CI_CLUMP = (1200.0, 4500.0)
CI_SHEEN = (70.0, 900.0)
CI_TUFT = (400.0, 650.0)
CI_PATCH = (9000.0, 30000.0)
#: Each one's smoothstep window on its noise (the patches' moved down by
#: CI_PATCH_COVER x the coverage: more coverage, more patches).
CI_FIBRE_WINDOW = (0.45, 0.80)
CI_CLUMP_WINDOW = (0.45, 0.70)
CI_STRAND_WINDOW = (0.48, 0.68)
CI_TUFT_WINDOW = (0.66, 0.90)
CI_PATCH_WINDOW = (0.70, 0.85)
CI_PATCH_COVER = 0.55
#: The strands' least share of the fibrous texture, the tails' and the
#: fibres' gains against a head's 1, and the most a streak's density.
CI_STRAND_FLOOR = 0.15
CI_TAIL_GAIN = 1.6
CI_FIBRE_GAIN = 1.0
CI_MAX_DENSITY = 1.3
#: The heads (uncinus, floccus; the WMO's are fibrous themselves, the
#: fibres splaying out at them, :data:`CI_FAN`): a wide
#: window on the tuft noise (a soft, fraying edge), moved by CI_HEAD_FRAY x
#: the fibre noise about its mean (the outline frays along the fibres),
#: CI_HEAD_FLOOR of it between the strands (striated), at most CI_HEAD_GAIN
#: (translucent). Task 23's heads, a solid smoothstep(0.66, 0.90) at 1, read
#: as blobs and smeared chevrons. MUST match the shader.
CI_HEAD_WINDOW = (0.60, 1.00)
CI_HEAD_FRAY = 0.25
CI_HEAD_FLOOR = 0.2
CI_HEAD_GAIN = 1.2
#: The taps' fan toward the head (:func:`cirrus_tufts`): at most this many
#: metres across the wind at tap 0, a fibre's share by its noise. MUST
#: match the shader.
CI_FAN = 500.0
#: The tails: CI_TAPS taps CI_TAIL_STEP metres apart downwind (the heads
#: there), CI_HOOK x k^2 metres across the wind at tap k (the fall streak's
#: shear: a hook), weighted (1 - k / CI_TAPS)^CI_TAIL_FADE.
CI_TAPS = 6
CI_TAIL_STEP = 380.0
CI_HOOK = 35.0
CI_TAIL_FADE = 1.3
#: The stream function's axis in the noise (off its lattice). MUST match.
CI_PSI_AXIS = (0.5257, 0.6180, 0.5848)
#: Each noise's offset, in tiles of the noise volume. MUST match.
CI_OFFSETS = {
    "patch": (0.13, 0.71, 0.37),
    "fibre": (0.47, 0.19, 0.83),
    "clump": (0.23, 0.89, 0.61),
    "sheen": (0.91, 0.37, 0.53),
    "tuft": (0.67, 0.43, 0.09),
}


def cirrus_flow(
    q: ArrayLike,
    psi: ArrayLike,
    across: ArrayLike,
    speed: ArrayLike,
    delta: ArrayLike,
    size: tuple[float, float],
    offset: tuple[float, float, float],
) -> np.ndarray:
    """The noise coordinates (``sample_volume``'s) of the point q + delta of
    the upper wind's flow (``cirrusFlow``): its features ``size`` (width
    across the wind, length along it), metres. The stream function *psi* at
    q is the across coordinate (constant along a streamline), the position
    the along one; ``grad psi = -speed * across``.

    :param q: (..., 3) metres, planet frame; *across*, *delta* likewise
    """
    q = np.asarray(q, np.float64)
    delta = np.asarray(delta, np.float64)
    width, length = size
    stream = np.asarray(psi, np.float64) - np.asarray(speed) * np.sum(
        np.asarray(across, np.float64) * delta, axis=-1
    )
    axis = np.asarray(CI_PSI_AXIS)
    return (
        (q + delta) / length + axis * (stream / width)[..., None]
    ) * NOISE_FEATURE + np.asarray(offset) * NOISE_SIZE


def _ci_noise(volume, q, psi, across, speed, delta, size, name, window):
    coords = cirrus_flow(q, psi, across, speed, delta, size, CI_OFFSETS[name])
    return _smoothstep(*window, sample_volume(volume, coords))


def _ci_strands(volume, q, psi, across, speed, warp, strands):
    """The strands' value: given (their mean where a pixel does not resolve
    them), or their noise's."""
    if strands is None:
        return _ci_noise(volume, q, psi, across, speed, warp, CI_SHEEN, "sheen",
                         CI_STRAND_WINDOW)  # fmt: skip
    return np.asarray(strands, np.float64)


def _ci_fibrous(volume, q, psi, across, speed, warp, strands):
    strands = _ci_strands(volume, q, psi, across, speed, warp, strands)
    return CI_STRAND_FLOOR + (1.0 - CI_STRAND_FLOOR) * strands


def _ci_raw(volume, q, psi, across, speed, delta, size, name):
    coords = cirrus_flow(q, psi, across, speed, delta, size, CI_OFFSETS[name])
    return sample_volume(volume, coords)


def cirrus_tufts(
    volume: np.ndarray,
    q: ArrayLike,
    psi: ArrayLike,
    along: ArrayLike,
    across: ArrayLike,
    speed: ArrayLike = 1.0,
    strands: ArrayLike | None = None,
    warp: ArrayLike = 0.0,
) -> np.ndarray:
    """The cirrus's heads and tails, resolved (``cirrusStreaks``). The taps
    along the hooked line fan out toward the head: each fibre shifts them
    across the wind by its share of :data:`CI_FAN` (the fibre noise about
    its mean, signed), most at the head (tap 0), none at the tail's end, so
    the fibres splay out at the head and run together downstream (the
    fibres' noise, not the strands': 180 m, not 70 m, keeps neighbouring
    pixels' taps together in the texture cache). The head: a wide window
    (:data:`CI_HEAD_WINDOW`) on the tuft
    noise at tap 0, its outline frayed along the fibres
    (:data:`CI_HEAD_FRAY`), striated by the strands, at most
    :data:`CI_HEAD_GAIN` (translucent: no blob). The tail: the tufts at the
    other taps averaged (a line integral), textured by the strands.

    :param strands: the strands' value in place of their window on their
        noise (their mean where a pixel does not resolve them); None =
        resolved
    """
    q = np.asarray(q, np.float64)
    along = np.asarray(along, np.float64)
    across = np.asarray(across, np.float64)
    warp = np.broadcast_to(np.asarray(warp, np.float64), q.shape)
    args = (volume, q, psi, across, speed)
    mean = float(np.mean(volume))
    strands = _ci_strands(*args, warp, strands)
    fibre = _ci_raw(*args, warp, CI_FIBRE, "fibre")
    fan = CI_FAN * 2.0 * (fibre - mean)
    head = tail = weights = 0.0
    for k in range(CI_TAPS):
        spread = fan * (1.0 - k / CI_TAPS)
        delta = (
            warp
            + CI_TAIL_STEP * k * along
            + (CI_HOOK * k * k + spread)[..., None] * across
        )
        n = _ci_raw(*args, delta, CI_TUFT, "tuft")
        if k == 0:
            frayed = n + CI_HEAD_FRAY * (fibre - mean)
            head = (
                CI_HEAD_GAIN
                * _smoothstep(*CI_HEAD_WINDOW, frayed)
                * (CI_HEAD_FLOOR + (1.0 - CI_HEAD_FLOOR) * strands)
            )
        else:
            w = (1.0 - k / CI_TAPS) ** CI_TAIL_FADE
            tail = tail + w * _smoothstep(*CI_TUFT_WINDOW, n)
            weights += w
    fibrous = CI_STRAND_FLOOR + (1.0 - CI_STRAND_FLOOR) * strands
    return head + CI_TAIL_GAIN * tail / weights * fibrous


def cirrus_detail(
    volume: np.ndarray,
    q: ArrayLike,
    psi: ArrayLike,
    along: ArrayLike,
    across: ArrayLike,
    speed: ArrayLike = 1.0,
    strands: ArrayLike | None = None,
    tufts: ArrayLike | None = None,
    warp: ArrayLike = 0.0,
) -> np.ndarray:
    """The cirrus's streaks (``cirrusStreaks``' detail, its domain warp
    *warp* given: the curl noise's on the GPU): the tufts
    (:func:`cirrus_tufts`) and the fibres in their bundles, textured by the
    strands; at most :data:`CI_MAX_DENSITY`.

    :param strands: the strands' value (None: resolved)
    :param tufts: the tufts' value (their mean where a pixel does not
        resolve them); None = resolved
    """
    q = np.asarray(q, np.float64)
    warp = np.broadcast_to(np.asarray(warp, np.float64), q.shape)
    args = (volume, q, psi, across, speed)
    fibre = _ci_noise(*args, warp, CI_FIBRE, "fibre", CI_FIBRE_WINDOW)
    fibre = fibre * _ci_noise(*args, warp, CI_CLUMP, "clump", CI_CLUMP_WINDOW)
    if tufts is None:
        tufts = cirrus_tufts(volume, q, psi, along, across, speed, strands, warp)
    fibrous = _ci_fibrous(volume, q, psi, across, speed, warp, strands)
    return np.minimum(tufts + CI_FIBRE_GAIN * fibre * fibrous, CI_MAX_DENSITY)


def cirrus_patches(
    volume: np.ndarray,
    q: ArrayLike,
    psi: ArrayLike,
    across: ArrayLike,
    cover: ArrayLike,
    speed: ArrayLike = 1.0,
    warp: ArrayLike = 0.0,
) -> np.ndarray:
    """The patches the cirrus's streaks come in (``cirrusStreaks``): their
    share of the sheet about the coverage (the weather's cirrus cover is
    that share, :func:`weather.cirrus_cover_map`)."""
    q = np.asarray(q, np.float64)
    warp = np.broadcast_to(np.asarray(warp, np.float64), q.shape)
    coords = cirrus_flow(q, psi, across, speed, warp, CI_PATCH, CI_OFFSETS["patch"])
    return _smoothstep(
        *CI_PATCH_WINDOW,
        sample_volume(volume, coords) + CI_PATCH_COVER * np.asarray(cover),
    )


def _cirrus_sample(seed: int, n: int = 40000, rng_seed: int = 5):
    """Points of a flat stretch of the sheet with the wind along x (speed
    1): (volume, q, psi, along, across)."""
    rng = np.random.default_rng(rng_seed)
    volume = _noise(seed)
    xy = rng.uniform(0.0, 400e3, (n, 2))
    q = np.stack([xy[:, 0], xy[:, 1], np.full(n, 6.1e6)], -1)
    along = np.broadcast_to([1.0, 0.0, 0.0], q.shape)
    across = np.broadcast_to([0.0, 1.0, 0.0], q.shape)  # up x along
    psi = 2e5 - xy[:, 1]  # grad psi = -across
    return volume, q, psi, along, across


@functools.cache
def cirrus_means(seed: int = 7) -> tuple[float, float, float, float, float]:
    """The cirrus's means, what its parts fade to where a pixel does not
    resolve them (``uHighCirrus``, ``uHighShape.w``), over a stretch of the
    sheet: the streaks' detail (:func:`cirrus_detail`), the patches' at
    coverage 0 and their slope with coverage (a line fitted over coverage
    0-1: :func:`cirrus_patch_mean`), the strands', the tufts'
    (:func:`cirrus_tufts`).

    :param seed: the noise volume's (the march's)
    """
    volume, q, psi, along, across = _cirrus_sample(seed)
    strands = _ci_noise(volume, q, psi, across, 1.0, np.zeros_like(q), CI_SHEEN,
                        "sheen", CI_STRAND_WINDOW)  # fmt: skip
    tufts = cirrus_tufts(volume, q, psi, along, across)
    detail = cirrus_detail(volume, q, psi, along, across, tufts=tufts)
    covers = np.linspace(0.0, 1.0, 6)
    patches = [cirrus_patches(volume, q, psi, across, c).mean() for c in covers]
    slope, intercept = np.polyfit(covers, patches, 1)
    return (
        float(detail.mean()),
        float(intercept),
        float(slope),
        float(strands.mean()),
        float(tufts.mean()),
    )


def cirrus_patch_mean(cover: ArrayLike, seed: int = 7) -> np.ndarray:
    """The patches' mean at *cover* as the shader takes it (the line of
    :func:`cirrus_means`, clamped to [0, 1])."""
    _, intercept, slope, _, _ = cirrus_means(seed)
    return np.clip(intercept + slope * np.asarray(cover, np.float64), 0.0, 1.0)


#: The cirrus's density fades in over this much coverage (``CI_COVER_EDGE``):
#: the coverage thins its patches, it does not scale its density. MUST
#: match the shader.
CI_COVER_EDGE = 0.1


def footprint_along(
    axis: ArrayLike, view: ArrayLike, pixel: ArrayLike, incidence: ArrayLike
) -> np.ndarray:
    """A view pixel's footprint on the high sheet along its unit direction
    *axis* (``footprintAlong``): *pixel* metres across the view, stretched
    by the slant (:func:`slab_footprint`) along the view's own direction on
    the sheet, *view* (unit). Across the view a grazing pixel stays small:
    the cirrus's fibres running across it stay resolved."""
    a = np.sum(np.asarray(axis, np.float64) * np.asarray(view, np.float64), axis=-1)
    inc = np.maximum(np.abs(np.asarray(incidence, np.float64)), HIGH_MIN_INCIDENCE)
    return np.asarray(pixel) * np.sqrt(
        a * a / (inc * inc) + np.maximum(1.0 - a * a, 0.0)
    )


def cirrus_streaks(
    volume: np.ndarray,
    q: ArrayLike,
    psi: ArrayLike,
    along: ArrayLike,
    across: ArrayLike,
    footprint: ArrayLike,
    cover: ArrayLike,
    seed: int = 7,
    footprints: tuple[ArrayLike, ArrayLike] | None = None,
) -> np.ndarray:
    """The cirrus's density as the shader takes it (``cirrusStreaks``, no
    domain warp, speed 1): faded in by the coverage, the patches times the
    streaks, each part faded to its mean (:func:`cirrus_means`) where a
    footprint does not resolve it (``resolved``: 2 footprints to its size
    none of it, 4 all).

    :param footprint: the slant's (metres: :func:`slab_footprint`), the
        tufts'
    :param footprints: (across the wind, along it) (:func:`footprint_along`),
        the patches', the fibres' and the strands'; None = *footprint* both
    """
    detail_mean, _, _, strand_mean, tuft_mean = cirrus_means(seed)
    footprint = np.asarray(footprint, np.float64)
    across_fp, along_fp = footprints if footprints is not None else (footprint,) * 2
    patches = cirrus_patches(volume, q, psi, across, cover)
    mean = cirrus_patch_mean(cover, seed)
    patches = mean + (patches - mean) * (
        recipe_resolved(CI_PATCH[0], across_fp) * recipe_resolved(CI_PATCH[1], along_fp)
    )
    zero = np.zeros_like(np.asarray(q, np.float64))
    strands = _ci_noise(volume, q, psi, across, 1.0, zero, CI_SHEEN, "sheen",
                        CI_STRAND_WINDOW)  # fmt: skip
    strand_seen = recipe_resolved(CI_SHEEN[0], across_fp) * recipe_resolved(
        CI_SHEEN[1], along_fp
    )
    strands = strand_mean + (strands - strand_mean) * strand_seen
    tufts = cirrus_tufts(volume, q, psi, along, across, strands=strands)
    tufts = tuft_mean + (tufts - tuft_mean) * recipe_resolved(CI_TUFT[0], footprint)
    detail = cirrus_detail(volume, q, psi, along, across, strands=strands, tufts=tufts)
    seen = recipe_resolved(CI_FIBRE[0], across_fp) * recipe_resolved(
        CI_FIBRE[1], along_fp
    )
    detail = detail_mean + (detail - detail_mean) * seen
    edge = _smoothstep(0.0, CI_COVER_EDGE, np.asarray(cover, np.float64))
    return edge * patches * detail


def march_fragment(recipes: bool, high: bool) -> str:
    """The march's fragment shader: ``cloud_march.glsl`` with the recipe
    genera's density (``CLOUD_RECIPES``) and the high layer's crossings
    (``CLOUD_HIGH``) compiled in only where the march has them: a branch
    never taken still costs the convective clouds (0.15-0.2 ms for the
    recipes')."""
    return (
        "cloud_march"
        + ("_recipes" if recipes else "")
        + ("_high" if high else "")
        + ".frag"
    )


def steps_over(segment: ArrayLike, min_step: float, growth: float) -> float:
    """About how many steps cover a segment: *min_step* long up to the knee
    (``min_step / growth``), ``growth * t`` past it (``stepsOver``)."""
    t0, t1 = segment
    if t1 <= t0:
        return 0.0
    knee = min_step / growth
    n = 1.0
    if t0 < knee:
        n += (min(t1, knee) - t0) / min_step
    if t1 > knee:
        n += math.log(t1 / max(t0, knee)) / growth
    return n


def step_scale(
    segments: ArrayLike, min_step: float, growth: float, budget: int
) -> float:
    """What every step is lengthened by so a ray's steps fit *budget*.

    :returns: 1 when they fit
    """
    needed = sum(steps_over(s, min_step, growth) for s in segments)
    return max(1.0, needed / budget)


def step_scales(
    segments: ArrayLike, min_step: float, growth: float, budget: int, split: float
) -> tuple[float, float]:
    """:func:`step_scale` for the near range (to *split* metres) and the far
    range apart: each has the budget (``cloud_march.frag``'s ``stretch``).

    :returns: (near, far), 1 when their steps fit
    """
    near = sum(
        steps_over((t0, min(t1, split)), min_step, growth) for t0, t1 in segments
    )
    far = sum(steps_over((max(t0, split), t1), min_step, growth) for t0, t1 in segments)
    return max(1.0, near / budget), max(1.0, far / budget)


def _step_cap(budget: int, split: float) -> int:
    """The most samples a ray takes: the budget per range, and a margin."""
    return (budget if math.isinf(split) else 2 * budget) + 8


def march_steps(
    segments: ArrayLike,
    min_step: float,
    growth: float,
    budget: int,
    jitter: float = 0.5,
    split: float = math.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """The samples the march takes along a ray, as the shader takes them
    (before it stops at an opaque cloud).

    :param segments: (n, 2) the ray's stretches in the slab
    :param jitter: where in its step each sample lies, [0, 1)
    :param split: where the near range ends, metres: each range has the
        budget (:func:`step_scales`)
    :returns: (t, dt): each sample's distance and the length it stands for
    """
    scales = step_scales(segments, min_step, growth, budget, split)
    cap = _step_cap(budget, split)
    ts, dts = [], []
    for t0, t1 in segments:
        if t1 <= t0:
            continue
        least = (t1 - t0) / SEGMENT_SAMPLES
        a = t0
        while a < t1 and len(ts) < cap:
            k = scales[0] if a < split else scales[1]
            dt = min(k * max(min_step, a * growth), least, t1 - a)
            ts.append(a + dt * jitter)
            dts.append(dt)
            a += dt
    return np.array(ts), np.array(dts)


def march_transmittance(
    extinction: np.ndarray, dt: np.ndarray
) -> tuple[float, np.ndarray]:
    """Composite a march's samples front to back, as the shader does.

    :param extinction: (n,) optical depth per metre at each sample (the
        field's extinction times its density)
    :param dt: (n,) the length each stands for
    :returns: the transmittance left, and each sample's weight (its share
        of the pixel: transmittance in front times its own alpha)
    """
    alpha = 1.0 - np.exp2(-extinction * dt)
    transmittance = np.concatenate([[1.0], np.cumprod(1.0 - alpha)])
    return float(transmittance[-1]), transmittance[:-1] * alpha


def march_ray(
    segments: ArrayLike,
    min_step: float,
    growth: float,
    budget: int,
    extinction_at: Callable[[float], float],
    jitter: float = 0.5,
    split: float = math.inf,
) -> tuple[float, int]:
    """One ray's march as the shader runs it: the steps of :func:`march_steps`,
    whatever the light already spent (a step lengthened by it made the steps'
    grid depend on the samples, and the visits' jitter no longer averaged its
    contours away: bands on a tower's face), stopping at :data:`OPAQUE`.

    :param extinction_at: the optical depth per metre at a distance along the ray
    :returns: the transmittance left as composited (:func:`opaque_residual`:
        0 once the cloud is counted opaque), and the samples taken
    """
    scales = step_scales(segments, min_step, growth, budget, split)
    cap = _step_cap(budget, split)
    transmittance, samples = 1.0, 0
    for t0, t1 in segments:
        if t1 <= t0:
            continue
        least = (t1 - t0) / SEGMENT_SAMPLES
        a = t0
        while a < t1 and samples < cap and transmittance > OPAQUE:
            k = scales[0] if a < split else scales[1]
            dt = min(k * max(min_step, a * growth), least, t1 - a)
            samples += 1
            alpha = 1.0 - 2.0 ** (-extinction_at(a + dt * jitter) * dt)
            transmittance *= 1.0 - alpha
            a += dt
    return opaque_residual(transmittance), samples


def min_step(field: DensityField) -> float:
    """:returns: the march's shortest step for a field, metres
    (:data:`MIN_STEP_FEATURE` of its finest octave's feature)"""
    finest = max(field.noise_scale) * field.octave_scales[-1]
    return MIN_STEP_FEATURE * NOISE_FEATURE / finest


# ── Drawing ────────────────────────────────────────────────────────────────


#: Which pixel (x, y) of each block the march traces, frame after frame, by
#: block (across, up): consecutive frames never trace touching pixels, even
#: across the blocks' edges, and each cycle's first frames spread over the
#: block. 4 x 4: a Bayer order (EVE-Redux's ``samplingSequence16`` read as
#: ranks). 4 x 2 and 3 x 3 (the eighth and the ninth, EVE's default range):
#: found by an exhaustive search over the orders (consecutive pixels at least
#: a diagonal apart on the torus, then the spread of every prefix), as EVE's own
#: 4 x 2 and 3 x 3 sequences trace touching pixels on consecutive frames.
_ORDERS = {
    (1, 1): ((0, 0),),
    (2, 2): ((0, 0), (1, 1), (1, 0), (0, 1)),
    (4, 2): ((0, 0), (2, 0), (1, 1), (3, 1), (1, 0), (3, 0), (0, 1), (2, 1)),
    (3, 3): (
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 0),
        (0, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
    ),
    (4, 4): tuple(
        divmod(int(v), 4)[::-1]
        for v in np.argsort(
            [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5], kind="stable"
        )
    ),
}
#: The march's block shapes (``CloudSettings.march_block``'s choices).
BLOCKS = tuple(_ORDERS)
#: This frame's least share in a pixel that has gathered its full count of
#: samples (cloud_resolve.frag): it averages the steps' jitter over about
#: this many of its visits' inverse.
MIN_WEIGHT = 0.03
#: The camera may move this far between frames and keep its history, metres
#: (more is a jump: a new pose, a teleport), or :data:`HISTORY_JUMP_SHARE` of
#: its distance to the clouds' slab if more: from orbit a fast descent moves
#: kilometres a frame, a small change to clouds hundreds of kilometres away,
#: and with its history thrown away every frame the clouds were this frame's
#: few samples, smoothed (the small ones faded out).
HISTORY_JUMP = 2000.0
HISTORY_JUMP_SHARE = 0.02
#: The camera's motion (metres, plus :data:`TURN_METRES` per unit of its
#: rotation matrix's change) that takes the clamp's noise floor to nothing
#: (cloud_resolve.frag's uStill): a floor holds a still view steady, but in
#: motion it would keep what a cloud was as it comes closer.
STILL_METRES = 2.0
TURN_METRES = 400.0


#: The finer billow layer's weight over the first's (cloud_march.frag).
BILLOW_FINER_WEIGHT = 0.5


def billow_gain(volume: np.ndarray) -> float:
    """:returns: what scales two layers of the noise itself (the second
    :data:`BILLOW_FINER_WEIGHT` of the first, independent) to the spread of the
    ridged noise ``1 - |2n - 1|`` the billows once were: the same depth of
    erosion, in lumps"""
    ridged = float(np.std(1.0 - np.abs(2.0 * volume - 1.0)))
    layers = float(np.std(volume)) * math.sqrt(1.0 + BILLOW_FINER_WEIGHT**2)
    return ridged / max(layers, 1e-6)


#: The blue noise's side, texels (cloud_march.frag's jitter reads it repeated).
BLUE_NOISE_SIZE = 64


@functools.cache
def blue_noise(size: int = BLUE_NOISE_SIZE, sigma: float = 1.9) -> np.ndarray:
    """A tileable blue noise (void and cluster: Ulichney's): *size* x *size* in
    [0, 1), each value once, neighbours as unlike as can be, so the march's
    jitter has no lattice and no clumps at any scale.

    :returns: (size, size) float32, the ranks over their count
    """
    n = size * size
    axis = np.minimum(np.arange(size), size - np.arange(size))
    kernel = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2.0 * sigma**2))
    spectrum = np.fft.rfft2(kernel)

    def energy(pattern: np.ndarray) -> np.ndarray:
        return np.fft.irfft2(np.fft.rfft2(pattern) * spectrum, s=pattern.shape)

    def shifted(y: int, x: int) -> np.ndarray:
        return np.roll(kernel, (y, x), axis=(0, 1))

    rng = np.random.default_rng(3)
    pattern = np.zeros((size, size))
    ones = max(n // 10, 1)
    pattern.flat[rng.choice(n, ones, replace=False)] = 1.0
    field = energy(pattern)
    # Relax: the tightest cluster's point moves to the largest void, until it
    # lands where it was.
    for _ in range(4 * n):
        tight = int(np.argmax(np.where(pattern > 0, field, -np.inf)))
        pattern.flat[tight] = 0.0
        field -= shifted(*divmod(tight, size))
        void = int(np.argmin(np.where(pattern > 0, np.inf, field)))
        pattern.flat[void] = 1.0
        field += shifted(*divmod(void, size))
        if void == tight:
            break
    rank = np.zeros(n)
    # The ones, the tightest cluster's first, rank down from their count
    work, work_field = pattern.copy(), field.copy()
    for r in range(ones - 1, -1, -1):
        tight = int(np.argmax(np.where(work > 0, work_field, -np.inf)))
        rank[tight] = r
        work.flat[tight] = 0.0
        work_field -= shifted(*divmod(tight, size))
    # The rest, the largest void's first, rank up from there
    work, work_field = pattern.copy(), field.copy()
    for r in range(ones, n):
        void = int(np.argmin(np.where(work > 0, np.inf, work_field)))
        rank[void] = r
        work.flat[void] = 1.0
        work_field += shifted(*divmod(void, size))
    return (rank.reshape(size, size) / n).astype(np.float32)


@functools.cache
def _cells_texture() -> Texture:
    """The recipe genera's Worley cells (``cloudCells``)."""
    return build_cells_texture(_cells())


@functools.cache
def _curl_texture() -> Texture:
    return build_curl_texture(curl_noise_volume())


@functools.cache
def _blue_noise_texture() -> Texture:
    texture = Texture("march_blue_noise")
    texture.setup_2d_texture(
        BLUE_NOISE_SIZE, BLUE_NOISE_SIZE, Texture.T_unsigned_byte, Texture.F_red
    )
    texture.set_ram_image(
        np.ascontiguousarray(np.round(blue_noise() * 255.0), np.uint8).tobytes()
    )
    texture.set_wrap_u(SamplerState.WM_repeat)
    texture.set_wrap_v(SamplerState.WM_repeat)
    texture.set_minfilter(SamplerState.FT_nearest)
    texture.set_magfilter(SamplerState.FT_nearest)
    return texture


#: How far a pixel on fresh samples takes their smooth mean instead of their
#: bilinear mix (cloud_resolve.frag's uSmooth), by the march's block: all of it
#: at a sixteenth of the pixels a frame, half at a quarter, none at all of them.
SMOOTHING = {(1, 1): 0.0, (2, 2): 0.5, (4, 2): 0.75, (3, 3): 0.75, (4, 4): 1.0}


def smoothing_of(block: tuple[int, int]) -> float:
    """:returns: ``uSmooth`` for a march block (across, up)"""
    return SMOOTHING[block]


#: This frame's least share in a pixel while the camera moves
#: (cloud_resolve.frag's uMovingWeight, by its uStill), by the block the
#: resolve gathers: about one cycle of the block's pixels (each pixel's own
#: sample once), a quarter at most (a 1 x 1 march's samples alone would be raw
#: noise). With :data:`MIN_WEIGHT` alone, ~33 frames of history lagged the
#: view of a cloud whose look changes fast as the camera flies (a lumpy
#: anvil's base seen at a grazing angle from inside the slab), and the clamp
#: cut the lag wherever its box of noisy samples happened to be narrow: grain
#: on the cloud's shaded side in motion (performance.md §30; Unreal's clouds
#: refresh each pixel once a cycle for flying through them, their docs).
MOVING_WEIGHTS = {
    (1, 1): 0.25,
    (2, 2): 0.25,
    (4, 2): 0.125,
    (3, 3): 1.0 / 9.0,
    (4, 4): 0.0625,
}


def moving_weight_of(block: tuple[int, int]) -> float:
    """:returns: ``uMovingWeight`` for the block a resolve gathers (across, up)"""
    return MOVING_WEIGHTS[block]


#: The reach of the resolve's statistics, samples (cloud_resolve.frag's
#: STATS_REACH): each sample's weight a bump of its distance to the pixel,
#: falling to nothing this far.
STATS_REACH = 2.0


def stats_window(f: tuple[float, float]) -> dict[tuple[int, int], float]:
    """cloud_resolve.frag's statistics window, mirrored: each sample's
    weight (by its place in the march's grid) for a pixel at *f* (the
    pixel's place in that grid): the 4 x 4 samples from ``floor(f) - 1``,
    each ``(1 - (d / STATS_REACH)^2)^2`` of its distance d, none beyond."""
    base = (math.floor(f[0]), math.floor(f[1]))
    weights = {}
    for y in range(-1, 3):
        for x in range(-1, 3):
            tap = (base[0] + x, base[1] + y)
            bump = 1.0 - ((tap[0] - f[0]) ** 2 + (tap[1] - f[1]) ** 2) / STATS_REACH**2
            if bump > 0.0:
                weights[tap] = bump * bump
    return weights


#: What share of a step's length fades the field's finer features to their means
#: (cloud_march.frag's uStepFade), by the march's block: the fewer the samples a
#: frame has, the more shimmer, the more it must fade. It is the clouds' detail
#: at a distance against their steadiness: Phase 4.1 had 0.4 everywhere, which
#: made far clouds soft lumps at every preset.
STEP_FADES = {(1, 1): 0.0, (2, 2): 0.1, (4, 2): 0.15, (3, 3): 0.15, (4, 4): 0.25}


#: Past this view footprint (a pixel's share of the clouds' feature, as the
#: march's shading measures it) a sample takes its sunlight from the light
#: volume alone, without the sun march's two local steps (cloud_march.frag's
#: uLightLocalFootprint; RENDERING_PLAN.md C.3's alternate path).
LIGHT_LOCAL_FOOTPRINT = 0.04


#: The block the march takes while the camera moves, by the still one's: a
#: pixel traced twice as often (its history's grain in motion is the clamp
#: on fewer, noisier samples), back to the still block once the view has
#: been still :data:`MOVING_HOLD` frames. Costs the march's share per traced
#: pixel while moving (performance.md §28).
MOVING_BLOCKS = {(4, 4): (4, 2), (4, 2): (2, 2), (3, 3): (2, 2)}
MOVING_HOLD = 8
#: The camera's motion (cloud_resolve.frag's uStill below this) that counts as
#: moving for :data:`MOVING_BLOCKS`.
MOVING_STILL = 0.5


def step_fade_of(block: tuple[int, int]) -> float:
    """:returns: ``uStepFade`` for a march block (across, up)"""
    return STEP_FADES[block]


#: The old ``CloudSettings.march_resolution`` (per axis) as a block.
_RESOLUTIONS = {1.0: (1, 1), 0.5: (2, 2), 0.25: (4, 4)}


def block_of(value: Any) -> tuple[int, int]:
    """:returns: the march's block (across, up) for ``CloudSettings.march_block``
    (a block), or an old ``march_resolution`` (0.25, 0.5 or 1.0) or "BXxBY"
    (the bench's command line)

    :raises ValueError: not a block of :data:`BLOCKS`
    """
    if isinstance(value, str):
        value = tuple(int(v) for v in value.lower().split("x"))
    elif isinstance(value, (int, float)):
        value = _RESOLUTIONS.get(float(value), value)
    block = tuple(int(v) for v in value) if isinstance(value, (tuple, list)) else None
    if block not in _ORDERS:
        raise ValueError(f"not a march block: {value!r} (one of {list(BLOCKS)})")
    return block


def pixels_of(block: tuple[int, int]) -> int:
    """:returns: the view's pixels per march pixel (a block's area)"""
    return block[0] * block[1]


def visit_of(block: tuple[int, int], frame: int, gather: bool = True) -> int:
    """:returns: how many times the pixel traced on *frame* has been traced
    before (once a cycle of the block's pixels' frames when gathering, every
    frame otherwise): what moves its steps' jitter on, so that each pixel's
    visits, not the frames, run through the jitter's sequence"""
    return frame // pixels_of(block) if gather else frame


def block_order(block: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """:returns: the pixels (x, y) of a block, in the order the march traces them"""
    return _ORDERS[block]


def bayer_offset(block: tuple[int, int], frame: int) -> tuple[int, int]:
    """:returns: the pixel (x, y) of each block the march traces on *frame*"""
    order = _ORDERS[block]
    return order[frame % len(order)]


@functools.cache
def _shader(vertex: str, fragment: str) -> Shader:
    return Shader.load(
        Shader.SL_GLSL, vertex=_SHADERS / vertex, fragment=_SHADERS / fragment
    )


def _target(name: str, linear: bool = False) -> Texture:
    texture = Texture(name)
    texture.set_wrap_u(SamplerState.WM_clamp)
    texture.set_wrap_v(SamplerState.WM_clamp)
    filtering = SamplerState.FT_linear if linear else SamplerState.FT_nearest
    texture.set_minfilter(filtering)
    texture.set_magfilter(filtering)
    return texture


def _buffer(
    app: Any,
    output: Any,
    name: str,
    size: tuple[int, int],
    sort: int,
    targets: dict[int, Texture],
) -> GraphicsOutput:
    """An offscreen buffer drawing into *targets* (render texture plane ->
    texture): half-float colour and a half-float aux plane (the cloud's
    distance, in kilometres), or a depth texture."""
    props = FrameBufferProperties()
    if GraphicsOutput.RTP_color in targets:
        props.set_rgba_bits(16, 16, 16, 16)
        props.set_float_color(True)
    else:
        props.set_rgba_bits(8, 8, 8, 8)
    if GraphicsOutput.RTP_aux_hrgba_0 in targets:
        props.set_aux_hrgba(1)
    props.set_depth_bits(24 if GraphicsOutput.RTP_depth in targets else 0)
    buffer = app.graphicsEngine.make_output(
        app.pipe,
        name,
        sort,
        props,
        WindowProperties.size(*size),
        GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_resizeable,
        output.get_gsg(),
        output,
    )
    if buffer is None:
        raise RuntimeError(f"the clouds' raymarch could not make its {name} buffer")
    for plane, texture in targets.items():
        buffer.add_render_texture(texture, GraphicsOutput.RTM_bind_or_copy, plane)
    return buffer


def _fullscreen(buffer: GraphicsOutput, name: str, shader: Shader) -> NodePath:
    """A fullscreen card drawn into *buffer*, in a scene of its own.

    :returns: the card
    """
    scene = NodePath(f"{name}_scene")
    camera = scene.attach_new_node(Camera(f"{name}_camera", OrthographicLens()))
    buffer.make_display_region().set_camera(camera)
    maker = CardMaker(name)
    maker.set_frame(-1.0, 1.0, -1.0, 1.0)
    card = scene.attach_new_node(maker.generate())
    card.node().set_bounds(OmniBoundingVolume())
    card.node().set_final(True)
    card.set_shader(shader)
    card.set_depth_test(False)
    card.set_depth_write(False)
    card.set_two_sided(True)
    return card


class _Resolve:
    """One of a range's two full-resolution results (they alternate: each
    frame's reads the other's as its history)."""

    def __init__(self, app: Any, output: Any, name: str, size, sort: int) -> None:
        self.colour = _target(f"{name}_colour", linear=True)
        self.distance = _target(f"{name}_distance", linear=True)
        self.buffer = _buffer(
            app,
            output,
            name,
            size,
            sort,
            {
                GraphicsOutput.RTP_color: self.colour,
                GraphicsOutput.RTP_aux_hrgba_0: self.distance,
            },
        )
        self.buffer.set_clear_color_active(False)
        self.card = _fullscreen(
            self.buffer, name, _shader("cloud_march.vert", "cloud_resolve.frag")
        )


class _ResolveInputs:
    """A resolve card's per-frame inputs, written in place
    (:meth:`_MarchLayer.update`)."""

    def __init__(self) -> None:
        self.clip_to_rel = PTA_LMatrix4f.empty_array(1)
        self.previous = PTA_LMatrix4f.empty_array(1)
        self.offset = PTA_LVecBase2i.empty_array(1)
        self.valid = PTA_int.empty_array(1)
        self.still = PTA_float.empty_array(1)


class _DepthPass:
    """The scene's depth in one depth range, at the view's resolution, for
    the march to stop at (and, the near range's, for the composites to tell
    whose pixel it is)."""

    def __init__(
        self,
        app: Any,
        output: Any,
        name: str,
        lens: Lens,
        bit: int,
        view: tuple[int, int],
        sort: int,
        states: tuple[RenderState, RenderState],
    ) -> None:
        self.lens = lens
        self.depth = _target(f"{name}_depth")
        self.buffer = _buffer(
            app,
            output,
            f"{name}_depth",
            view,
            sort,
            {GraphicsOutput.RTP_depth: self.depth},
        )
        self.buffer.set_clear_color_active(False)
        self.buffer.set_clear_depth_active(True)
        self.camera = pass_camera(
            app, f"{name}_depth", lens, bit, *states, 1.0 / DEPTH_SCALE
        )
        self.buffer.make_display_region().set_camera(self.camera)


class _GroundShadow:
    """The clouds' shadow on the view's scene, at the depth passes'
    resolution (ground_shadow.frag): what the terrain and the sea read back
    by their distance instead of shading it per pixel, where the ground's
    small triangles had every 2 x 2 quad run it (0.8-0.9 ms a frame at 2560
    x 1440, performance.md §33). Its texture is the march's, kept across
    rebuilds, so the readers bind it once."""

    def __init__(
        self,
        app: Any,
        output: Any,
        name: str,
        size: tuple[int, int],
        sort: int,
        texture: Texture,
    ) -> None:
        self.texture = texture
        self.buffer = _buffer(
            app, output, name, size, sort, {GraphicsOutput.RTP_color: texture}
        )
        self.buffer.set_clear_color_active(False)
        self.card = _fullscreen(
            self.buffer, name, _shader("cloud_march.vert", "ground_shadow.frag")
        )


class _MarchLayer:
    """The march over both depth ranges (EVE-Redux's single march over the
    split cameras' depths): a depth pass per range, one march, its two
    alternating results, and a card per range laying them over that range's
    pass (cloud_composite.frag: the near range's where its depth pass drew
    something, the far range's elsewhere).

    The near range's depth pass and card run only while the slab reaches
    into the near range (:meth:`set_near`); the march then stops at the far
    range's scene, and whatever the near camera draws covers the clouds
    behind it.

    A *gathering* layer traces a different pixel of each block every frame
    and gathers them into the view's resolution; the other traces the same
    one and only averages its jitter over the frames, at the march's own
    resolution, upsampled by the composite (a full-resolution pass less, but
    4 x 4 squares at a quarter resolution).
    """

    def __init__(
        self,
        app: Any,
        output: Any,
        name: str,
        near_lens: Lens,
        far_lens: Lens,
        view: tuple[int, int],
        block: tuple[int, int],
        sort: int,
        states: tuple[RenderState, RenderState],
        gather: bool = True,
        moving_block: tuple[int, int] | None = None,
        ground: Texture | None = None,
        recipes: bool = False,
        high: bool = False,
    ) -> None:
        self.lens = near_lens
        self.far_lens = far_lens
        self.block = block
        self.gather = gather
        #: (still, moving): the block traced still and while moving
        self.blocks = (block, moving_block if gather and moving_block else block)
        self._still_frames = MOVING_HOLD
        # The block's inputs, written in place as it changes (no new render
        # state a frame): uBlock (march, resolves), uStepFade, uSmooth.
        self._block_input = PTA_LVecBase2i.empty_array(1)
        self._resolve_block = PTA_LVecBase2i.empty_array(1)
        self._step_fade = PTA_float.empty_array(1)
        self._smooth = PTA_float.empty_array(1)
        self._moving_weight = PTA_float.empty_array(1)
        self.far_depth = _DepthPass(
            app,
            output,
            f"{name}_far",
            far_lens,
            CLOUD_FAR_BIT,
            depth_size(view),
            sort,
            states,
        )
        self.near_depth = _DepthPass(
            app,
            output,
            f"{name}_near",
            near_lens,
            CLOUD_NEAR_BIT,
            depth_size(view),
            sort + 1,
            states,
        )
        self.camera = self.near_depth.camera  # both follow the app's camera
        # Whether the near range's depth pass runs (the march and the cards
        # read it only then).
        self.near_on = PTA_int.empty_array(1)
        self.near_on[0] = 1
        self.near = True
        #: 1: the composite's depth no farther than the scene's (A/B: 0, the
        #: resolved distance's alone; cloud_composite.frag)
        self.scene_cut = PTA_int.empty_array(1)
        self.scene_cut[0] = 1
        #: The clouds' shadow on the scene (None: its readers shade it per pixel)
        self.ground = (
            None
            if ground is None
            else _GroundShadow(
                app, output, f"{name}_ground", depth_size(view), sort + 2, ground
            )
        )

        # The march: one pixel of every block.
        self.colour = _target(f"{name}_colour")
        self.distance = _target(f"{name}_distance")
        self.buffer = _buffer(
            app,
            output,
            name,
            self._march_size(view),
            sort + 2,
            {
                GraphicsOutput.RTP_color: self.colour,
                GraphicsOutput.RTP_aux_hrgba_0: self.distance,
            },
        )
        self.buffer.set_clear_color_active(True)
        self.buffer.set_clear_color((0.0, 0.0, 0.0, 0.0))
        self.buffer.set_clear_value(
            GraphicsOutput.RTP_aux_hrgba_0, (0.0, 0.0, 0.0, 0.0)
        )
        self.buffer.set_clear_active(GraphicsOutput.RTP_aux_hrgba_0, True)
        # The recipe genera's and the high layer's variants only for a march
        # that has them (cloud_march.glsl's CLOUD_RECIPES, CLOUD_HIGH).
        fragment = march_fragment(recipes, high)
        self.card = _fullscreen(
            self.buffer, name, _shader("cloud_march.vert", fragment)
        )
        #: The march's region: the corner of its buffer this frame's block fills
        self.march_region = self.buffer.get_display_region(
            self.buffer.get_num_display_regions() - 1
        )
        self.card.set_shader_inputs(
            uSceneDepthNear=self.near_depth.depth,
            uSceneDepthFar=self.far_depth.depth,
            uNearDepthOn=self.near_on,
            uDepthScale=DEPTH_SCALE,
            uBlueNoise=_blue_noise_texture(),
        )
        # The jitter's index: this layer's visits (visit_of).
        self.visit = PTA_int.empty_array(1)
        self.card.set_shader_input("uFrame", self.visit)

        # The result, gathered over the frames.
        self.resolves = [
            _Resolve(
                app, output, f"{name}_resolve{i}", self._resolve_size(view), sort + 3
            )
            for i in (0, 1)
        ]
        for resolve, history in zip(self.resolves, self.resolves[::-1]):
            resolve.card.set_shader_inputs(
                uMarch=self.colour,
                uMarchDistance=self.distance,
                uHistory=history.colour,
                uHistoryDistance=history.distance,
                uMinWeight=MIN_WEIGHT,
                uMovingWeight=self._moving_weight,
                uSmooth=self._smooth,
            )

        # Laid over each range's pass at the cloud's depth: (far, near).
        self.composites = []
        for side, shown_to in (("far", FAR_BIT), ("near", NEAR_BIT)):
            card = composite_card(
                app, f"{name}_composite_{side}", self.colour, shown_to
            )
            card.set_shader(_shader("cloud_composite.vert", "cloud_composite.frag"))
            card.set_depth_test(True)
            own = self.near_depth if side == "near" else self.far_depth
            card.set_shader_inputs(
                uNearDepth=self.near_depth.depth,
                uNearDepthOn=self.near_on,
                uDepthScale=DEPTH_SCALE,
                uNearCard=int(side == "near"),
                uSceneDepth=own.depth,
                uSceneDepthCut=self.scene_cut,
            )
            self.composites.append(card)
        # The per-frame inputs, bound once and written in place by update:
        # new values given as inputs would make each card a new render state
        # every frame. Until the first update they hold neutral values.
        identity = LMatrix4f.ident_mat()
        self._clip_to_render = PTA_LMatrix4f.empty_array(1)
        self._clip_to_render[0] = identity
        self._clip_to_render_far = PTA_LMatrix4f.empty_array(1)
        self._clip_to_render_far[0] = identity
        self._march_offset = PTA_LVecBase2i.empty_array(1)
        self.card.set_shader_inputs(
            uClipToRender=self._clip_to_render,
            uClipToRenderFar=self._clip_to_render_far,
            uOffset=self._march_offset,
        )
        if self.ground is not None:
            self.ground.card.set_shader_inputs(
                uSceneDepthNear=self.near_depth.depth,
                uSceneDepthFar=self.far_depth.depth,
                uNearDepthOn=self.near_on,
                uDepthScale=DEPTH_SCALE,
                uClipToRender=self._clip_to_render,
                uClipToRenderFar=self._clip_to_render_far,
            )
        self._resolve_inputs = []
        for resolve in self.resolves:
            inputs = _ResolveInputs()
            inputs.clip_to_rel[0] = identity
            inputs.previous[0] = identity
            resolve.card.set_shader_inputs(
                uClipToRel=inputs.clip_to_rel,
                uPrevRelToClip=inputs.previous,
                uOffset=inputs.offset,
                uHistoryValid=inputs.valid,
                uStill=inputs.still,
                # Depth-aware: the scene each sample stopped at.
                uSceneDepthNear=self.near_depth.depth,
                uSceneDepthFar=self.far_depth.depth,
                uNearDepthOn=self.near_on,
                uDepthScale=DEPTH_SCALE,
                uClipToRender=self._clip_to_render,
                uClipToRenderFar=self._clip_to_render_far,
            )
            self._resolve_inputs.append(inputs)
        for card in self.composites:
            card.set_shader_input("uDistance", self.distance)
        # Each composite's two states (showing one resolve or the other)
        self._composite_attribs: list[list[ShaderAttrib]] = []
        self._view = view
        self.active = True
        self._previous: tuple[Any, np.ndarray, Any] | None = None
        self._set_sizes(view)

    def _march_size(self, view: tuple[int, int]) -> tuple[int, int]:
        """The march's buffer: room for the block with the most pixels
        (the moving one's), the still one's drawn in a corner of it."""
        return tuple(
            max(max(1, math.ceil(v / block[k])) for block in self.blocks)
            for k, v in enumerate(view)
        )

    def _resolve_size(self, view: tuple[int, int]) -> tuple[int, int]:
        return view if self.gather else self._march_size(view)

    def _offset(self, frame: int) -> tuple[int, int]:
        """:returns: the pixel of each block traced on *frame*: in turn when
        gathering, else its middle"""
        if self.gather:
            return bayer_offset(self.block, frame)
        return (self.block[0] // 2, self.block[1] // 2)

    def _set_block(self, block: tuple[int, int]) -> None:
        """Trace this block from this frame: its inputs, and the march's
        region the corner of its buffer the block fills."""
        self.block = block
        self._block_input[0] = LVecBase2i(*block)
        self._resolve_block[0] = LVecBase2i(*block) if self.gather else LVecBase2i(1, 1)
        self._step_fade[0] = step_fade_of(block)
        self._smooth[0] = smoothing_of(block)
        self._moving_weight[0] = moving_weight_of(block if self.gather else (1, 1))
        used = [max(1, math.ceil(v / b)) for v, b in zip(self._view, block)]
        size = self._march_size(self._view)
        self.march_region.set_dimensions(0.0, used[0] / size[0], 0.0, used[1] / size[1])

    def _set_sizes(self, view: tuple[int, int]) -> None:
        if self.ground is not None:
            self.ground.card.set_shader_input(
                "uViewSize", LVecBase2f(*depth_size(view))
            )
        self.card.set_shader_inputs(
            uBlock=self._block_input,
            uViewSize=LVecBase2f(*view),
            uStepFade=self._step_fade,
            uLightLocalFootprint=LIGHT_LOCAL_FOOTPRINT,
        )
        size = self._resolve_size(view)
        for resolve in self.resolves:
            resolve.card.set_shader_inputs(
                uBlock=self._resolve_block,
                uViewSize=LVecBase2f(*size),
            )
        self._set_block(self.block)
        # Made once (again after a resize): update switches between them
        self._composite_attribs = []
        for card in self.composites:
            card.set_shader_input(
                "uDownscale",
                LVecBase2f(1.0, 1.0) if self.gather else LVecBase2f(*self.block),
            )
            attribs = []
            for resolve in self.resolves:
                card.set_shader_inputs(
                    uClouds=resolve.colour, uDistance=resolve.distance
                )
                attribs.append(card.get_attrib(ShaderAttrib))
            self._composite_attribs.append(attribs)

    @property
    def depth_buffers(self) -> list[GraphicsOutput]:
        """The two ranges' depth passes' buffers: (far, near)."""
        return [self.far_depth.buffer, self.near_depth.buffer]

    def resize(self, view: tuple[int, int]) -> None:
        self._view = view
        for buffer in self.depth_buffers:
            buffer.set_size(*depth_size(view))
        if self.ground is not None:
            self.ground.buffer.set_size(*depth_size(view))
        self.buffer.set_size(*self._march_size(view))
        for resolve in self.resolves:
            resolve.buffer.set_size(*self._resolve_size(view))
        self._set_sizes(view)
        self._previous = None

    def set_active(self, active: bool) -> None:
        """Run the passes or stop them (and hide the composites); the history
        starts over."""
        self.active = active
        for buffer in (self.far_depth.buffer, self.buffer):
            buffer.set_active(active)
        if self.ground is not None:
            self.ground.buffer.set_active(active)
        for resolve in self.resolves:
            resolve.buffer.set_active(active)
        self._show_near()
        if active:
            self.composites[0].show()
        else:
            self.composites[0].hide()
        self._previous = None

    def set_near(self, near: bool) -> None:
        """Draw the near range's depth pass and lay its card, or not (when the
        slab is out of the near range: nothing of the clouds can be in front
        of what the near camera draws). The history goes on."""
        self.near = near
        self._show_near()

    def _show_near(self) -> None:
        on = self.active and self.near
        self.near_on[0] = int(on)
        self.near_depth.buffer.set_active(on)
        if on:
            self.composites[1].show()
        else:
            self.composites[1].hide()

    def update(
        self, app: Any, frame: int, camera: np.ndarray, slab_distance: float = 0.0
    ) -> None:
        """Per frame: this frame's pixel of each block, the ranges' clip
        spaces to the render frame (the march) and the near range's to the
        camera's surroundings now and a frame ago (the reprojection: both
        lenses look the same way), and which result is written.

        :param frame: the march's frame count
        :param camera: the camera in the planet frame, float64
        :param slab_distance: how far the clouds' slab is, metres (the
            history's jump limit grows with it)
        """
        projection = self.lens.get_projection_mat()
        unproject = self.lens.get_projection_mat_inv()
        placed = self.camera.get_mat(app.render)
        turned = LMatrix4f(placed)
        turned.set_row(3, LVecBase3f(0.0, 0.0, 0.0))
        rel_to_clip = LMatrix4f(turned)
        rel_to_clip.invert_in_place()
        rel_to_clip = rel_to_clip * projection
        self._clip_to_render[0] = unproject * placed
        self._clip_to_render_far[0] = self.far_lens.get_projection_mat_inv() * placed

        valid = self._previous is not None
        previous = rel_to_clip
        still = 0.0
        if valid:
            last_rel_to_clip, last_camera, last_turned = self._previous
            moved = camera - last_camera
            turn = sum(
                abs(turned.get_cell(i, j) - last_turned.get_cell(i, j))
                for i in range(3)
                for j in range(3)
            )
            motion = (float(np.linalg.norm(moved)) + turn * TURN_METRES) / STILL_METRES
            still = 1.0 - min(motion, 1.0)
            valid = float(np.linalg.norm(moved)) <= max(
                HISTORY_JUMP, HISTORY_JUMP_SHARE * slab_distance
            )
            # A point relative to the camera now, as it was relative to the
            # camera then (float64 for the move, float32 only near the camera).
            previous = LMatrix4f.translate_mat(LVecBase3f(*moved)) * last_rel_to_clip
        self._previous = (rel_to_clip, camera.copy(), LMatrix4f(turned))

        # The block: the moving one as soon as the view moves, the still one
        # once it has been still MOVING_HOLD frames.
        self._still_frames = self._still_frames + 1 if still >= MOVING_STILL else 0
        block = self.blocks[1] if self._still_frames < MOVING_HOLD else self.blocks[0]
        if block != self.block:
            self._set_block(block)
        offset = self._offset(frame)
        self.visit[0] = visit_of(self.block, frame, self.gather)
        self._march_offset[0] = LVecBase2i(*offset)

        write = self.resolves[frame % 2]
        read = self.resolves[1 - frame % 2]
        write.buffer.set_active(True)
        read.buffer.set_active(False)
        inputs = self._resolve_inputs[frame % 2]
        inputs.clip_to_rel[0] = unproject * turned
        inputs.previous[0] = previous
        inputs.offset[0] = LVecBase2i(*offset) if self.gather else LVecBase2i(0, 0)
        inputs.valid[0] = int(valid)
        inputs.still[0] = still
        for card, attribs in zip(self.composites, self._composite_attribs):
            card.set_attrib(attribs[frame % 2])

    def destroy(self, app: Any) -> None:
        for card in self.composites:
            card.remove_node()
        for depth in (self.far_depth, self.near_depth):
            depth.camera.remove_node()
        ground = [] if self.ground is None else [self.ground]
        for card in [self.card] + [p.card for p in self.resolves + ground]:
            card.get_top().remove_node()
        for buffer in (
            self.depth_buffers
            + [self.buffer]
            + [p.buffer for p in self.resolves + ground]
        ):
            app.graphicsEngine.remove_window(buffer)


@dataclass
class _Genus:
    """One genus of the march's clouds: a density field over its own slab
    (which may overlap another genus's) and its weather. A planet has one;
    with genera, one per name of :attr:`genera.GeneraSpec.genera`, the first
    (the primary genus) reading the weather cube, the others a channel of
    coverage cube A (``coverage``: the channel; ``weather_job`` the cube's)."""

    name: str | None
    preset: Any
    field: DensityField
    peaks: np.ndarray
    table: tuple
    weather_job: Any
    weather: np.ndarray | None = None
    deck_job: Any = None
    deck_field: DensityField | None = None
    deck_optics: Any = None
    coverage: tuple = ("weather", 0)
    cover: np.ndarray | None = None
    #: A recipe genus's resolved :class:`noise.GenusField` (its optics the
    #: preset's until _configure); None for the others
    recipe: GenusField | None = None
    #: The far level's octaves (its shadow's field's: _configure)
    deck_octaves: int = 4


class _GenusCaster:
    """A genus's part of the clouds' shadow (one genus of the march: its
    weather, its deck field and optics), as the CPU marches it
    (:func:`field_setup.genus_sun_depth`; :class:`_UnionCaster` sums them)."""

    def __init__(self, march: CloudMarch, index: int) -> None:
        self._march, self._index = march, index
        self._shade_key: tuple | None = None
        self._shades: dict[tuple, float] = {}

    @property
    def _genus(self) -> _Genus:
        return self._march._genera[self._index]

    @property
    def weather(self):
        return self._genus.weather

    @property
    def position(self):
        return self._march.position

    @property
    def radius(self):
        return self._march.radius

    @property
    def _seed(self):
        return self._march._seed

    @property
    def _octave_mean(self):
        return self._march._octave_mean

    @property
    def coverage_source(self) -> tuple | None:
        """(coverage cube, channel, shift table) of a genus that reads one
        (:func:`field_setup.genus_sun_depth`); None for the primary genus."""
        genus = self._genus
        if genus.cover is None:
            return None
        return genus.cover, genus.coverage[1], np.asarray(genus.table)

    @property
    def recipe(self) -> GenusField | None:
        """A recipe genus's field (:func:`field_setup.genus_sun_depth` skips
        it), None for the others."""
        return self._genus.recipe

    @property
    def _deck_field(self):
        return self._genus.deck_field

    @property
    def _deck_optics(self):
        return self._genus.deck_optics

    @property
    def _deck_octaves(self) -> int:
        return self._genus.deck_octaves

    @property
    def shadow_field(self) -> DensityField:
        return self._genus.deck_field

    #: Its shade alone (the union's is the product of the genera's).
    sun_visibility = sun_visibility


class _UnionCaster(_GenusCaster):
    """The clouds' shadow, all their genera's together (what
    :class:`shadow_map.CloudShadowMaps`, :class:`light_volume.CloudLightVolumes`
    and :class:`planet.PlanetClouds` ask of a caster): one shadow layer, the
    sky bands its resources span, and ``exp2(-sum of the genera's depths)``
    on the CPU. Its weather, field and optics are the first genus's."""

    def __init__(self, march: CloudMarch) -> None:
        super().__init__(march, 0)

    @property
    def shadow_genera(self) -> list[_GenusCaster]:
        """The genera it sums (:func:`field_setup.sun_visibility`)."""
        return self._march._genus_casters

    @property
    def shadow_layer(self) -> int:
        return int(self._march._shadow_layer[0])

    @property
    def sky_bands(self) -> list[tuple[float, float]]:
        """The sky bands (:func:`sky_bands`) the light volume and the maps
        span: their (base, thickness) above sea level, lowest first."""
        return self._march.sky_bands

    def bind_shadow_inputs(self, node: NodePath) -> None:
        self._march.bind_shadow_inputs(node)

    #: The genera's together.
    sun_visibility = sun_visibility


class CloudMarch:
    """A planet's low clouds, raymarched (see the module): the
    ``CloudSettings.renderer == "march"`` volume layer of
    :class:`planet.PlanetClouds`, with :class:`sphere.CloudVolume`'s
    interface and arguments.

    It draws nothing until :meth:`attach` gives it the app and its split
    cameras (``PlanetClouds.use_split_depth``, which a ``World`` calls).

    :ivar root: on the scene: the sea's reflection's deck (hide it to hide
        the reflection; :meth:`set_visible` hides everything)
    :ivar field: the density field, threshold resolved for the planet
    :ivar weather: (6, n, n, 2) cube map of coverage and fbm shift, once set
        up
    :ivar stats: as :class:`sphere.CloudVolume`'s (no tiles nor billboards:
        ``pending`` is 1 until the set-up is done)
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = CloudSettings

    def __init__(
        self,
        parent: NodePath,
        radius: float,
        executor: ChunkExecutor,
        position: ArrayLike = (0.0, 0.0, 0.0),
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        cloud_type: CloudType = CloudType.CUMULUS,
        altitude: float | None = None,
        planet: PlanetSpec | None = None,
        weather: WeatherSpec | None = None,
        seed: int = 7,
        weather_seed: int = 11,
        settings: CloudSettings | None = None,
        sun_brightness: float = 2.8,
        sky_strength: float = 0.35,
        sky_occlusion: float = 0.55,
        exposure: float = 0.9,
        genera: GeneraSpec | None = None,
        cirrus: bool = False,
    ):
        """As :class:`planet.PlanetClouds`' arguments of the same names.
        With *genera* it marches every genus of
        :attr:`genera.GeneraSpec.genera` (:data:`genera.GENUS_PRESETS`) as
        one: the first, the primary genus (the convective clouds), on the
        weather cube, the others on coverage cube A. With *cirrus* its high
        layer draws the weather's cirrus (:attr:`genera.GeneraSpec.cirrus_weather`;
        without genera, :func:`genera.weather_cirrus`'s high layer alone)."""
        settings = settings or CloudSettings()
        _ensure_bin()
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=np.float64)
        self.executor = executor
        self.grid = grid_for(self.radius)
        self.stats = {"pending": 1}
        self._atmospheres = atmospheres

        volume = _noise(seed)
        self._seed = seed
        self._octave_mean = float(volume.mean())
        if genera is not None and cirrus and not genera.cirrus_weather:
            genera = replace(genera, cirrus_weather=True)
        self._genera_spec = genera
        specs = (
            genus_specs(genera, altitude, covered=True)
            if genera is not None
            else [(None, cloud_type, altitude, None, ("weather", 0))]
        )
        if genera is None:
            weather = planet_weather(weather, planet, self.radius, weather_seed)
        if len(specs) > MAX_GENERA:
            raise ValueError(f"{len(specs)} genera; the march takes {MAX_GENERA}")
        self._genera: list[_Genus] = []
        cover_jobs: dict[str, Any] = {}
        for index, (name, kind, base, thickness, coverage) in enumerate(specs):
            if (coverage[0] == "weather") != (index == 0):
                raise ValueError(
                    f"genus {name!r}: the first genus, and it alone, reads the weather"
                )
            if genera is None:
                preset, field, peaks = planet_field(
                    kind, base, seed, self.radius, thickness
                )
                recipe = None
            else:
                east = tuple(float(v) for v in TangentFrame.at(*genera.focus).east)
                preset, field, peaks, recipe = genus_field(
                    name, base, thickness, seed, self.radius, east
                )
            table = tuple(float(v) for v in shift_table(peaks, field.threshold[0]))
            if genera is None:
                job = executor.submit_once(_weather, weather, table)
            elif index == 0:
                # The primary genus's weather cube: its shift by depth too.
                tables = table
                if field.min_top < 1.0:
                    tables = tuple(
                        tuple(float(v) for v in row)
                        for row in _depth_tables(seed, field, self.radius)
                    )
                job = executor.submit_once(_genus_weather, genera, CONVECTIVE, tables)
            else:
                if coverage[0] not in cover_jobs:
                    cover_jobs[coverage[0]] = executor.submit_once(
                        _coverage_cube, genera, coverage[0]
                    )
                job = cover_jobs[coverage[0]]
            self._genera.append(
                _Genus(name, preset, field, peaks, table, job, coverage=coverage,
                       recipe=recipe)
            )  # fmt: skip
        #: What the high layer follows: the genera, or without them the
        #: weather's cirrus alone (with *cirrus*); None = no high layer
        self._high_spec = (
            genera
            if genera is not None
            else (weather_cirrus(weather) if cirrus else None)
        )
        #: The high genera drawn (genera.high_genera, those a feature covers
        #: and the weather's cirrus: none, no high layer compiled) and their
        #: coverage cube (B)
        self._high = (
            [] if self._high_spec is None else high_genera(self._high_spec, True)
        )
        self._high_job = None
        self._wind_job = None
        if self._high:
            self._high_job = cover_jobs.get("B") or executor.submit_once(
                _coverage_cube, self._high_spec, "B"
            )
            # The upper wind on the GPU (highWind): the cirrus's flow
            self._wind_job = executor.submit_once(high_wind_cube, self._high_spec)
        first = self._genera[0]
        self.field = first.field
        self._preset = first.preset
        self._peaks = first.peaks
        self._table = first.table
        self._weather_job = first.weather_job
        self.weather: np.ndarray | None = None
        self._settings = settings
        #: The shadow's layer (uCloudShadowLayer: genus 0's far level; genus
        #: g's is g * DETAIL_LEVELS past it), -1 until ready
        self._shadow_layer = PTA_int.empty_array(1)
        self._shadow_layer[0] = -1
        #: The sky bands (uSkyBands): (base, top, genus mask, 0)
        slabs = [tuple(f.field.slab) for f in self._genera]
        self.sky_bands = sky_bands(slabs)
        self._sky_bands = PTA_LVecBase4f.empty_array(MAX_SKY_BANDS)
        for k, ((base, thickness), mask) in enumerate(
            zip(self.sky_bands, band_masks(slabs, self.sky_bands))
        ):
            self._sky_bands[k] = LVecBase4f(base, base + thickness, mask, 0.0)
        #: The nodes the shadow is bound to
        self._shadow_nodes: list[NodePath] = []
        self._map_binders: list[Any] = []
        self._genus_casters = [_GenusCaster(self, i) for i in range(len(self._genera))]
        self._union = _UnionCaster(self)
        self._textures: dict[str, Any] = {}
        self._layer_far = PTA_float.empty_array(DETAIL_LEVELS)
        self._layer_params = PTA_LVecBase4f.empty_array((MAX_GENERA + 1) * GENUS_VEC4S)
        self._level_offsets = PTA_LVecBase4f.empty_array(
            MAX_GENERA * DETAIL_LEVELS // 4
        )
        self._level_octaves = PTA_LVecBase4f.empty_array(DETAIL_LEVELS // 4)
        params, self._deck_jobs = self._configure(settings)
        self._write_params(params)

        # The sea's reflection: the deck alone.
        self.root = parent.attach_new_node("cloud_march")
        self.root.stash()  # out of every camera until the set-up is done
        self.root.hide(HIDE_FROM_NEAR | HIDE_FROM_FAR | HIDE_FROM_CLOUD_PASS)
        self.deck = self.root.attach_new_node(
            deck_mesh(self.radius + self.field.slab[0] + 0.5 * self.field.slab[1])
        )
        self.deck.set_shader(deck_shader())
        self.deck.set_two_sided(True)
        self.root.set_transparency(TransparencyAttrib.M_none)
        self.root.set_depth_test(True)
        self.root.set_depth_write(False)
        self.root.set_bin(CLOUD_BIN, 0)
        self.root.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_one,
                ColorBlendAttrib.O_one_minus_incoming_alpha,
            )
        )

        self._camera_planet = PTA_LVecBase3f.empty_array(1)
        self._camera_render = PTA_LVecBase3f.empty_array(1)
        self._planet_render = PTA_LVecBase3f.empty_array(1)
        self._sun = PTA_LVecBase3f.empty_array(1)
        self._sun[0] = LVecBase3f(0.0, 0.0, 1.0)
        self._steps = PTA_int.empty_array(1)
        self._split = PTA_float.empty_array(1)  # the near range's end (uSplit)
        self._frame = PTA_int.empty_array(1)
        #: The genera drawn, a bit each, the high genera's past the low
        #: genera's (set_genus_visible, HIGH_MASK_SHIFT)
        self._genus_mask = PTA_int.empty_array(1)
        self._genus_mask[0] = ALL_GENERA_MASK
        self._layer_count = PTA_int.empty_array(1)
        self._layer_count[0] = self._layers_used
        self._inputs: dict[str, Any] = {
            "layerParams": self._layer_params,
            "genusLevelOffset": self._level_offsets,
            "detailOctaves": self._level_octaves,
            "cloudNoise": build_noise_texture(volume),
            "cloudCells": _cells_texture(),
            "noiseSize": float(NOISE_SIZE),
            "noiseOffset": Vec3(0, 0, 0),
            "deckLightCosMin": DECK_LIGHT_COS_MIN,
            "planetRadius": self.radius,
            "uOwnAtmosphere": -1 if atmosphere is None else int(atmosphere),
            "camPos": self._camera_planet,
            "uCamRender": self._camera_render,
            "uPlanetCentre": self._planet_render,
            "sunDir": self._sun,
            # White: the planet's air colours it (planetCloudLight).
            "sunColor": Vec3(1.0, 1.0, 1.0),
            "skyColor": Vec3(0.45, 0.6, 0.85),
            "sunBrightness": float(sun_brightness),
            "skyStrength": float(sky_strength),
            "skyOcclusion": float(sky_occlusion),
            "exposure": float(exposure),
            "uPixelAngle": 0.001,
            "uCloudDebug": 0,
            "uMarchSteps": self._steps,
            "uSplit": self._split,
            "uMinStep": min(min_step(f.field) for f in self._genera),
            "uStepGrowth": STEP_GROWTH,
            "uLayerFar": self._layer_far,
            "uLayerCount": self._layer_count,
            "uGenusMask": self._genus_mask,
            "uWeatherStretch": WEATHER_STRETCH_GENERA
            if genera is not None
            else WEATHER_STRETCH,
            "uDetailScale": DETAIL_SCALE,
            "uDetailErosion": DETAIL_EROSION,
            "uBillowMean": float(np.mean(1.0 - np.abs(2.0 * volume - 1.0))),
            "uBillowGain": billow_gain(volume),
            "uBillowRawMean": float(np.mean(volume)),
            "uCurlNoise": _curl_texture(),
            "uCurlStrength": CURL_STRENGTH,
            "uCurlFrequency": CURL_FREQUENCY,
            # The deck's (cloud_deck.vert, cloud.frag): its own entry
            "uDeckLayer": DECK_LAYER,
            # The high layer's (cloud_march.glsl's highCrossing)
            **self._high_inputs(self._high_spec, seed),
            # The shadow's genera and sky bands: the light volume's chart and
            # the maps' chord (cloud_field.glsl)
            **self._shadow_inputs(),
        }
        self._nodes: list[NodePath] = []
        self._bind(self.root)
        # The clouds' shadow on the scene (bind_ground_shadow): its texture
        # kept across the passes' rebuilds, on while a pass draws it.
        self._ground_texture = _target("cloud_ground_shadow")
        self._ground_on = PTA_int.empty_array(1)
        self._ground_binders: list[Any] = []
        self._app: Any = None
        self._depth: SplitDepthCameras | None = None
        self._layers: list[_MarchLayer] = []
        self._size = (0, 0)
        self._visible = True
        self._pixel_angle = 0.001
        self._sky_light: float | None = None
        self._apply_live(settings)

    @staticmethod
    def _high_inputs(genera: GeneraSpec | None, seed: int = 7) -> dict[str, Any]:
        """The high layer's slab, optics and light (``uHighSlab``,
        ``uHighPhase``, ``uHighLight``): ice (``CloudType.CIRRUS``'s
        optics)."""
        base, thickness = genera.high_slab if genera is not None else HIGH_SLAB
        o = PRESETS[CloudType.CIRRUS].optics
        forward, backward, isotropic = phase_weights(o)
        order = sorted(HIGH_CHANNELS, key=HIGH_CHANNELS.get)
        optics = [HIGH_OPTICS[name] for name in order]
        jet = high_jet(genera) if genera is not None else np.array([1.0, 0.0, 0.0])
        return {
            "uHighSlab": Vec4(base, thickness, 0.0, 0.0),
            "uHighExtinction": Vec3(*(h.depth / thickness for h in optics)),
            "uHighCap": Vec3(*(h.cap for h in optics)),
            "uHighShape": Vec4(
                cirrocumulus_grain_mean(),
                HIGH_OPTICS["cirrostratus"].halo,
                HIGH_OPTICS["cirrus"].halo,
                cirrus_means(seed)[4],
            ),
            "uHighCirrus": Vec4(*cirrus_means(seed)[:4]),
            "uHighJet": Vec3(*(float(v) for v in jet)),
            "uHighPhase": Vec4(forward, backward, isotropic, clamped_anisotropy(o)),
            "uHighLight": Vec4(o.multiple_scattering, o.shadow_strength, 0.0, 0.0),
        }

    @property
    def high_genera(self) -> list[str]:
        """The high genera the march draws (those a feature covers, and the
        cirrus when it draws the weather's)."""
        return [name for name, _ in self._high]

    @property
    def high_slab(self) -> tuple[float, float] | None:
        """The high layer's slab, (base, thickness) above sea level, metres,
        while it draws a genus; None = no high layer."""
        return self._high_spec.high_slab if self._high else None

    # ── wiring ────────────────────────────────────────────────────────────

    def _bind(self, node: NodePath) -> None:
        """Give a node every input the march and the deck read (kept in step
        by :meth:`_set_inputs`)."""
        bind_defaults(node)
        bind_shadow_defaults(node)
        cloud_shadow_defaults(node)  # the maps' inputs, off, until bind_maps
        if self._atmospheres is not None:
            self._atmospheres.bind(node)
        node.set_shader_inputs(**self._inputs)
        if self.weather is not None:
            node.set_shader_inputs(**self._textures)
        for bind in self._map_binders:
            bind(node)
        self._nodes.append(node)

    def bind_maps(self, bind: Any) -> None:
        """Have the march's passes read a body's cloud shadow maps for their
        own light (cloud_field.glsl's ``cloudSunDepth``): *bind* is called
        with each pass's node now and with the next ones'
        (:meth:`shadow_map.CloudShadowMaps.bind`).

        :param bind: sets the maps' inputs on a node
        """
        self._map_binders.append(bind)
        for node in self._nodes:
            bind(node)

    def bind_ground_shadow(self, node: NodePath, binders: list[Any]) -> None:
        """Let a node's shaders (the terrain's: terrain_frag.glsl's
        ``uGroundShadow``) read the clouds' shadow on the view's scene from
        the march's pass (:class:`_GroundShadow`) while it draws, and shade it
        per pixel otherwise.

        :param node: the node that reads it
        :param binders: set the pass's own inputs beyond the clouds' (the
            world's ``Shadows.bind``: the shadow's steps), now and on the
            passes' rebuilds
        """
        node.set_shader_inputs(
            uGroundShadow=self._ground_texture,
            uGroundShadowOn=self._ground_on,
            uGroundShadowScale=DEPTH_SCALE,
        )
        self._ground_binders = list(binders)
        for layer in self._layers:
            self._bind_ground_pass(layer)

    def _bind_ground_pass(self, layer: _MarchLayer) -> None:
        for bind in self._ground_binders:
            bind(layer.ground.card)

    def _set_inputs(self, **inputs: Any) -> None:
        self._inputs.update(inputs)
        for node in self._nodes:
            node.set_shader_inputs(**inputs)

    def attach(self, app: Any, depth: SplitDepthCameras) -> None:
        """Build the march's passes, over both depth ranges of *depth* (see the
        module). Build the split cameras first; call again after a new set.

        :param app: the ShowBase (``camera``, ``render``, ``graphicsEngine``,
            ``pipe``)
        :param depth: the game's split cameras
        """
        self.detach()
        self._app, self._depth = app, depth
        self._build_layers()

    def detach(self) -> None:
        """Remove the march's passes (it draws nothing until :meth:`attach`)."""
        for layer in self._layers:
            self._nodes.remove(layer.card)
            self._nodes.remove(layer.ground.card)
            layer.destroy(self._app)
        self._layers = []
        self._ground_on[0] = 0

    def _build_layers(self) -> None:
        app, depth = self._app, self._depth
        self._size = self._view_size()
        block = block_of(self._settings.march_block)
        layer = _MarchLayer(
            app,
            depth.output,
            "cloud_march",
            app.camLens,
            depth.far_lens,
            self._size,
            block,
            _FIRST_SORT,
            pass_states(),
            gather=True,
            moving_block=(
                MOVING_BLOCKS.get(block)
                if self._settings.march_refresh_moving
                else None
            ),
            ground=self._ground_texture,
            recipes=any(genus.recipe is not None for genus in self._genera),
            high=bool(self._high),
        )
        self._bind(layer.card)
        self._bind(layer.ground.card)
        self.bind_shadow_inputs(layer.ground.card)
        self._bind_ground_pass(layer)
        # Nothing to march until the weather is in (_finish).
        layer.set_active(self._visible and self.weather is not None)
        self._layers.append(layer)

    def _view_size(self) -> tuple[int, int]:
        region = self._depth.near_region
        return (max(region.get_pixel_width(), 1), max(region.get_pixel_height(), 1))

    # ── settings ──────────────────────────────────────────────────────────

    @property
    def settings(self) -> CloudSettings:
        """What the clouds cost (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: CloudSettings) -> None:
        """Change what the clouds cost: the steps apply at once; the
        octaves per distance and the sun march repack the layers (the
        deck's light is remade for its octaves, ~0.4 s); the resolution
        resizes the passes. A change of renderer is
        :class:`planet.PlanetClouds`' to make."""
        if not self._settings.changes(settings):
            return
        self._finish()
        old = self._settings
        self._settings = settings
        params, deck_jobs = self._configure(settings)
        self._write_params(params)
        self._set_inputs(uDeckLayer=DECK_LAYER)
        if old.shell_octaves[-1:] != settings.shell_octaves[-1:]:
            lights = self._deck_textures([job.result() for job in deck_jobs])
            self._set_inputs(**lights)
            if self.weather is not None:
                self._textures.update(lights)
        self._update_shadow_nodes()
        if (
            old.march_block != settings.march_block
            or old.march_refresh_moving != settings.march_refresh_moving
        ):
            self.set_pixel_angle(self._pixel_angle)
            if self._layers:
                self.attach(self._app, self._depth)
        self._apply_live(settings)

    def _apply_live(self, settings: CloudSettings) -> None:
        self._steps[0] = int(settings.march_steps)
        if settings.reflected:
            self.root.show(REFLECTION_CAMERA_MASK)
        else:
            self.root.hide(REFLECTION_CAMERA_MASK)

    def _configure(self, settings: CloudSettings) -> tuple[np.ndarray, list[Any]]:
        """Pack the march's parameters: an entry per genus (its field at
        its full octaves), each detail level's octaves and, per genus, the
        threshold offset that keeps its coverage with them (the octave bands
        by distance, then the far level, which the shadows are marched in),
        and the deck's own entry (the first genus's far level, its optical
        depth capped). The genera share the levels' distances, each its
        own ``DETAIL_LEVELS`` slots.

        :returns: the packed parameters, and each genus's far level's light
            table being made in the pool (futures)
        """
        seed = self._seed
        shells = max(min(settings.octave_bands, DETAIL_LEVELS - 1), 1)
        schedule = settings.shell_octaves
        _, bands = octave_bands(self.grid, shells)
        params = np.zeros((MAX_GENERA + 1, GENUS_VEC4S, 4), dtype=np.float32)
        offsets = np.zeros(MAX_GENERA * DETAIL_LEVELS, dtype=np.float32)
        octave_counts = np.zeros(DETAIL_LEVELS, dtype=np.float32)
        far = np.zeros(DETAIL_LEVELS, dtype=np.float32)
        stride = shells + 1
        jobs = []
        for k in range(stride):
            octave_counts[k] = schedule[min(k, len(schedule) - 1)]
            if k < shells:
                far[k] = 0.5 * sum(bands[k])  # mid-band: where the levels cross
        for index, genus in enumerate(self._genera):
            field, peaks = genus.field, genus.peaks
            optics = scale_march(genus.preset, settings.march_scale).optics
            # Per metre of the view's path
            extinction = field.density * VIEW_EXTINCTION_SCALE
            params[index] = pack_genus_params(
                self._genus_field(index, optics), extinction=extinction,
                octave_mean=self._octave_mean, octaves=4, shift_row=index,
            )  # fmt: skip
            for k in range(stride):
                octaves = int(octave_counts[k])
                fewer = (
                    recipe_peaks(seed, genus.recipe, self.radius, octaves)
                    if genus.recipe is not None
                    else _peaks(seed, field, self.radius, octaves)
                )
                layer = OctaveBand(
                    0, (0.0, 0.0), 1.0, octaves, octave_offset(peaks, fewer)
                )
                offsets[index * DETAIL_LEVELS + k] = layer.threshold_offset
            # The far level is the reflection's deck too: its cap, no fade.
            layer_optics = replace(optics, max_optical_depth=DECK_MAX_OPTICAL_DEPTH)
            deck = layer.field_of(field)
            if index == 0:
                params[MAX_GENERA, :LAYER_VEC4S] = pack_layer_params(
                    deck, extinction=extinction, octave_mean=self._octave_mean,
                    optics=layer_optics, octaves=octaves,
                )  # fmt: skip
            # The deck's field and optics: the one its shadow is marched in.
            genus.deck_field = deck
            genus.deck_optics = layer_optics
            genus.deck_octaves = octaves
            if genus.recipe is not None:
                # A recipe genus's far averages from its recipe's density
                # at its far level (the bakes' field and octaves).
                far_genus = replace(genus.recipe, field=deck, optics=layer_optics)
                jobs.append(
                    self.executor.submit_once(
                        _recipe_deck_light,
                        seed,
                        far_genus,
                        self.radius,
                        genus.table,
                        octaves,
                    )  # fmt: skip
                )
                continue
            jobs.append(
                self.executor.submit_once(
                    _deck_light,
                    seed,
                    deck,
                    layer_optics,
                    self.radius,
                    genus.table,
                    schedule[-1],
                )
            )
        # The recipe genera's shared frame, in the deck entry's free row 15:
        # the stretch's axis (the focus's east) and the cells' mean F1.
        east = (0.0, 1.0, 0.0)
        if self._genera_spec is not None:
            east = TangentFrame.at(*self._genera_spec.focus).east
        f1_mean, edges_mean = cell_means(_cells())
        params[MAX_GENERA, LAYER_VEC4S + 2] = pack_recipe_frame(east, f1_mean)
        params[MAX_GENERA, LAYER_VEC4S + 1] = pack_recipe_cells(edges_mean)
        self._layers_used = stride
        self._far_values = far
        self._offset_values = offsets
        self._octave_values = octave_counts
        return params, jobs

    def _genus_field(self, index: int, optics: Any) -> GenusField:
        """Genus *index* as a :class:`noise.GenusField`: its field, optics,
        coverage source and far shadow (:data:`genera.GENUS_PRESETS`; a
        planet without genera is one primary genus)."""
        genus = self._genera[index]
        if genus.recipe is not None:
            return replace(genus.recipe, optics=optics)
        if self._genera_spec is None:
            return GenusField("primary", genus.field, optics)
        name = genus.name
        preset = GENUS_PRESETS[name]
        return GenusField(name, genus.field, optics, preset.coverage, preset.far_shadow)

    def _write_params(self, params: np.ndarray) -> None:
        for i, row in enumerate(params.reshape(-1, 4)):
            self._layer_params[i] = LVecBase4f(*(float(v) for v in row))
        for i, value in enumerate(self._far_values):
            self._layer_far[i] = float(value)
        for i, row in enumerate(self._offset_values.reshape(-1, 4)):
            self._level_offsets[i] = LVecBase4f(*(float(v) for v in row))
        for i, row in enumerate(self._octave_values.reshape(-1, 4)):
            self._level_octaves[i] = LVecBase4f(*(float(v) for v in row))
        if hasattr(self, "_layer_count"):
            self._layer_count[0] = self._layers_used

    # ── the clouds' shadow ────────────────────────────────────────────────

    @property
    def genera_count(self) -> int:
        """How many genera of clouds it marches (1 without genera)."""
        return len(self._genera)

    def shadow_caster(self) -> _UnionCaster:
        """The clouds' shadow, all their genera's together (one light volume,
        one pair of shadow maps)."""
        return self._union

    def genus_caster(self, genus: int) -> _GenusCaster:
        """One genus's part of the shadow, as the CPU marches it (0: the
        convective clouds, or the only ones; 1: the stratus)."""
        return self._genus_casters[genus]

    @property
    def shadow_layer(self) -> int:
        """The layer of ``layerParams`` the first genus's shadow is marched in
        (its far one; genus g's is g * ``DETAIL_LEVELS`` past it), -1 until
        the set-up is done or after :meth:`remove`."""
        return self._union.shadow_layer

    @property
    def shadow_field(self) -> DensityField:
        """The field the first genus's shadow is marched in (the deck's)."""
        return self._genera[0].deck_field

    def _deck_textures(self, tables: list[np.ndarray]) -> dict[str, Any]:
        """The genera's far tables: genus 0's as ``deckLight`` (the deck's
        and genus 0's), every genus's a page of ``genusDeckLight``."""
        out = {"deckLight": build_deck_light_texture(tables[0])}
        if len(tables) > 1:
            out["genusDeckLight"] = build_deck_light_array(tables)
        return out

    def _shadow_inputs(self) -> dict[str, Any]:
        """The shadow's layer, genera and sky bands (cloud_field.glsl)."""
        return {
            "uCloudShadowLayer": self._shadow_layer,
            "uCloudGenera": len(self._genera),
            "uSkyBands": self._sky_bands,
            "uSkyBandCount": len(self.sky_bands),
        }

    def bind_shadow_inputs(self, node: NodePath) -> None:
        """Let the shaders on a node (the planet's terrain, the light
        resources' bakes) darken the sun under these clouds: all the genera's
        shadow (``planetCloudShadow``), through their deck layers.
        """
        node.set_shader_inputs(
            cloudNoise=self._inputs["cloudNoise"],
            cloudCells=self._inputs["cloudCells"],
            layerParams=self._layer_params,
            genusLevelOffset=self._level_offsets,
            detailOctaves=self._level_octaves,
            planetRadius=self.radius,
            sunDir=self._sun,
            noiseSize=float(NOISE_SIZE),
            noiseOffset=Vec3(0, 0, 0),
            deckLightCosMin=DECK_LIGHT_COS_MIN,
            uCloudDebug=0,
            genusCoverA=_no_coverage(),
            genusCoverB=_no_coverage(),
            genusShift=_no_shift(),
            genusDeckLight=_no_deck_light_array(),
            **self._shadow_inputs(),
        )
        self._shadow_nodes.append(node)
        self._update_shadow_nodes()

    def _update_shadow_nodes(self) -> None:
        """The shaded nodes' weather and layer: none until the set-up."""
        ready = self.weather is not None
        self._shadow_layer[0] = self._layers_used - 1 if ready else -1
        self._shadow_nodes = [n for n in self._shadow_nodes if not n.is_empty()]
        if ready:
            for node in self._shadow_nodes:
                node.set_shader_inputs(**self._textures)

    def sun_visibility(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """The share of the sun the clouds leave points (all the genera's:
        :func:`field_setup.sun_visibility`)."""
        return self._union.sun_visibility(*args, **kwargs)

    # ── public ────────────────────────────────────────────────────────────

    def set_sun(
        self,
        direction: ArrayLike,
        sun_color: ArrayLike | None = None,
        sky_color: ArrayLike | None = None,
    ) -> None:
        """As :meth:`sphere.CloudVolume.set_sun`."""
        d = np.asarray(direction, dtype=float)
        self._sun[0] = LVecBase3f(*(d / np.linalg.norm(d)))
        if sun_color is not None:
            self._set_inputs(sunColor=Vec3(*sun_color))
        if sky_color is not None:
            self._set_inputs(skyColor=Vec3(*sky_color))

    def set_sky_light(self, strength: float) -> None:
        """As :meth:`sphere.CloudVolume.set_sky_light`."""
        if strength != self._sky_light:
            self._sky_light = strength
            self._set_inputs(skyStrength=float(strength))

    def set_pixel_angle(self, radians: float) -> None:
        """The angle one pixel of the VIEW spans (vertical fov / its height):
        the march's own pixels are larger by its resolution.

        :param radians: the angle per pixel of the view
        """
        self._pixel_angle = radians
        value = float(
            radians * math.sqrt(pixels_of(block_of(self._settings.march_block)))
        )
        if value != self._inputs["uPixelAngle"]:
            self._set_inputs(uPixelAngle=value)

    @property
    def genus_names(self) -> list[str]:
        """The genera's names, by index (the convective clouds, or the one
        type, first)."""
        return [genus.name or "clouds" for genus in self._genera]

    def set_genus_visible(self, genus: int | str, visible: bool) -> None:
        """Draw a genus or not, live (a look-development toggle: the
        showcase's keys, the demo's C key for the cirrus). Only the view
        hides it: it still shades the others, and the ground.

        :param genus: its index or its name (:attr:`genus_names`), or a high
            genus's name (``genera.HIGH_CHANNELS``: ``"cirrus"`` hides all
            the cirrus, the weather's and the features')
        """
        bit = 1 << self._genus_bit(genus)
        mask = self._genus_mask[0]
        self._genus_mask[0] = (mask | bit) if visible else (mask & ~bit)

    def genus_visible(self, genus: int | str) -> bool:
        """:returns: whether a genus is drawn (:meth:`set_genus_visible`)"""
        return bool(self._genus_mask[0] >> self._genus_bit(genus) & 1)

    def _genus_bit(self, genus: int | str) -> int:
        """A genus's bit of ``uGenusMask`` (:data:`HIGH_MASK_SHIFT`)."""
        if isinstance(genus, str) and genus in HIGH_CHANNELS:
            return HIGH_MASK_SHIFT + HIGH_CHANNELS[genus] - 1
        index = self.genus_names.index(genus) if isinstance(genus, str) else genus
        return int(index)

    def set_debug(self, flags: int) -> None:
        """Profiling switches (cloud_field.glsl's ``uCloudDebug`` bits)."""
        self._set_inputs(uCloudDebug=int(flags))

    def set_visible(self, visible: bool) -> None:
        """Show or hide everything: the march (its passes stop) and the
        reflection's deck."""
        self._visible = visible
        if visible:
            self.root.show()
        else:
            self.root.hide()
        self.root.hide(HIDE_FROM_NEAR | HIDE_FROM_FAR | HIDE_FROM_CLOUD_PASS)
        for layer in self._layers:
            layer.set_active(visible and self.weather is not None)
        if not visible:
            self._ground_on[0] = 0

    def update(
        self, camera_world: ArrayLike, camera_render: ArrayLike = (0.0, 0.0, 0.0)
    ) -> None:
        """Per frame, once the camera and the split cameras are placed.

        :param camera_world: camera in the world frame, metres (float64)
        :param camera_render: camera in the render frame
        """
        if self._weather_job is not None:
            if not all(
                job.done()
                for job in [f.weather_job for f in self._genera] + self._deck_jobs
            ):
                self.stats["pending"] = 1
                return
            self._finish()
        cam = np.asarray(camera_world, dtype=np.float64) - self.position
        render_cam = np.asarray(camera_render, dtype=np.float64)
        self._camera_planet[0] = LVecBase3f(*cam)
        self._camera_render[0] = LVecBase3f(*render_cam)
        self._planet_render[0] = LVecBase3f(*(render_cam - cam))
        self.deck.set_pos(*(render_cam - cam))
        if self._layers and self._visible:
            size = self._view_size()
            if size != self._size:
                self._size = size
                for layer in self._layers:
                    layer.resize(size)
            self._frame[0] += 1
            # The near range's depth pass and card run only when the slab
            # reaches into it.
            slab_distance = self._slab_distance(cam)
            near_wanted = slab_distance <= self._depth.split * 1.01
            self._split[0] = self._depth.split
            for layer in self._layers:
                if near_wanted != layer.near:
                    layer.set_near(near_wanted)
            for layer in self._layers:
                if layer.active:
                    layer.update(self._app, self._frame[0], cam, slab_distance)
        self._ground_on[0] = int(
            self._visible and any(layer.active for layer in self._layers)
        )

    def _slab_distance(self, camera: np.ndarray) -> float:
        """:returns: how far the slab's nearest point is from *camera*
        (planet frame), metres"""
        altitude = float(np.linalg.norm(camera)) - self.radius
        nearest = math.inf
        for genus in self._genera:
            base, thickness = genus.field.slab
            nearest = min(
                nearest, max(base - altitude, altitude - base - thickness, 0.0)
            )
        return nearest

    def remove(self) -> None:
        """Take the clouds off the scene (and their shadow off the nodes it
        shaded)."""
        self._shadow_layer[0] = -1
        self._shadow_nodes = []
        self.detach()
        self.root.remove_node()

    def _finish(self) -> None:
        """Finish the set-up begun in the pool (waiting for it if need be)."""
        if self._weather_job is None:
            return
        first = self._genera[0]
        first.weather = first.weather_job.result()
        self._textures = {"weatherMap": build_weather_texture(first.weather)}
        tables = [()] * len(self._genera)  # genusShift: genus 0's row unused
        for index, genus in enumerate(self._genera[1:], start=1):
            # A genus on coverage cube A: its coverage and the shift its
            # table gives it (genera.genus_weather_at, the shader's
            # genusWeatherAt) as the CPU's weather.
            genus.cover = genus.weather_job.result()
            coverage = genus.cover[..., genus.coverage[1]]
            shift = coverage_to_shift(coverage, np.asarray(genus.table))
            depth = np.zeros_like(coverage)
            genus.weather = np.stack([coverage, shift, depth], -1).astype(np.float32)
            tables[index] = genus.table
            # Each coverage cube once (genusCoverA, genusCoverB).
            name = "genusCover" + genus.coverage[0]
            if name not in self._textures:
                self._textures[name] = build_coverage_texture(genus.cover)
        if self._high_job is not None and "genusCoverB" not in self._textures:
            # The high genera's channels of cube B, with no genus on it
            self._textures["genusCoverB"] = build_coverage_texture(
                self._high_job.result()
            )
        self._high_job = None
        if self._wind_job is not None:
            self._textures["highWind"] = build_high_wind_texture(
                self._wind_job.result()
            )
        self._wind_job = None
        for genus in self._genera:
            genus.weather_job = None
        if len(self._genera) > 1:
            self._textures["genusShift"] = build_shift_texture(
                [t if t else np.zeros(len(self._genera[1].table)) for t in tables]
            )
        self._textures.update(
            self._deck_textures([job.result() for job in self._deck_jobs])
        )
        self.weather = first.weather
        self._weather_job = self._deck_jobs = None
        for node in self._nodes:
            node.set_shader_inputs(**self._textures)
        self._inputs.update(self._textures)
        self.stats["pending"] = 0
        self._update_shadow_nodes()
        self.root.unstash()
        for layer in self._layers:
            layer.set_active(self._visible)
