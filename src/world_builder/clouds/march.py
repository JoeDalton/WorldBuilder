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

The sea's reflection sees the deck alone (:func:`sphere.deck_mesh`, the
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
    WindowProperties,
)

from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.clouds.cloud import (
    CloudSettings,
    CloudType,
    load_cloud_atlas,
    scale_march,
)
from world_builder.clouds.genera import CONVECTIVE, STRATIFORM, GeneraSpec
from world_builder.clouds.half_res import composite_card, pass_camera, pass_states
from world_builder.clouds.noise import (
    DECK_LIGHT_COS_MIN,
    LAYER_VEC4S,
    MAX_LAYERS,
    NOISE_FEATURE,
    NOISE_SIZE,
    DensityField,
    build_curl_texture,
    build_deck_light_texture,
    build_noise_texture,
    curl_noise_volume,
    pack_layer_params,
)
from world_builder.clouds.shadow_defaults import (
    _no_deck_light,
    _no_weather,
    cloud_shadow_defaults,
)
from world_builder.clouds.sphere import (
    CLOUD_BIN,
    DECK_MAX_OPTICAL_DEPTH,
    CloudVolume,
    ShellSpec,
    _deck_light,
    _depth_tables,
    _ensure_bin,
    _genus_weather,
    _noise,
    _peaks,
    _shaders,
    _weather,
    deck_mesh,
    octave_offset,
    planet_field,
    planet_weather,
    shell_bands,
)
from world_builder.clouds.weather import WeatherSpec, build_weather_texture, shift_table
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


def family_segments(
    origin: ArrayLike,
    direction: ArrayLike,
    radius: float,
    slabs: list[tuple[float, float]],
    t_start: float = 0.0,
    t_end: float = math.inf,
) -> list[tuple[float, float, int]]:
    """The stretches of a ray in several families' slabs (the genera's
    convective clouds, then the stratus under them), sorted along the ray (the
    shader's four segments): the slabs do not overlap.

    :param slabs: each family's (base, thickness) above sea level, metres
    :returns: (t0, t1, family) of each non-empty stretch
    """
    out = []
    for family, slab in enumerate(slabs):
        for t0, t1 in slab_segments(origin, direction, radius, slab, t_start, t_end):
            if t1 > t0:
                out.append((float(t0), float(t1), family))
    return sorted(out)


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
    :returns: the transmittance left (0 once the cloud is counted opaque), and
        the samples taken
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
    return (0.0 if transmittance <= OPAQUE else transmittance), samples


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
        self.card = _fullscreen(
            self.buffer, name, _shader("cloud_march.vert", "cloud_march.frag")
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
class _Family:
    """One family of the march's clouds: a density field and its weather. A
    planet has one; with genera, two (the convective clouds, then the stratus
    under them), whose slabs do not overlap."""

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


class _FamilyCaster:
    """A family's shadow, as :class:`sphere.CloudVolume`'s (what
    :class:`shadow_map.CloudShadowMaps` and :class:`planet.PlanetClouds` ask
    of a caster)."""

    def __init__(self, march: CloudMarch, index: int) -> None:
        self._march, self._index = march, index
        self._shade_key: tuple | None = None
        self._shades: dict[tuple, float] = {}

    @property
    def _family(self) -> _Family:
        return self._march._families[self._index]

    @property
    def weather(self):
        return self._family.weather

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
    def _deck_field(self):
        return self._family.deck_field

    @property
    def _deck_optics(self):
        return self._family.deck_optics

    @property
    def shadow_layer(self) -> int:
        return int(self._march._shadow_layers[self._index][0])

    @property
    def shadow_field(self) -> DensityField:
        return self._family.deck_field

    def bind_shadow_inputs(self, node: NodePath) -> None:
        self._march.bind_shadow_inputs(node, family=self._index)

    sun_visibility = CloudVolume.sun_visibility


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
        near_fade_radii: float = 2.5,
        channel: str | None = None,
        game: Any = None,
        genera: GeneraSpec | None = None,
        family: str | None = None,
    ):
        """As :class:`sphere.CloudVolume`'s (*near_fade_radii* and
        *channel* are unused: nothing fades near the camera, nothing is
        generated per tile). With *genera* it marches both families, the
        convective clouds and the stratus under them (*family* is unused)."""
        settings = settings or CloudSettings()
        _ensure_bin()
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=np.float64)
        self.executor = executor
        self.grid = grid_for(self.radius)
        self.stats = {"tiles": 0, "billboards": 0, "pending": 1, "gen_ms": 0.0}
        self._atmospheres = atmospheres

        volume = _noise(seed)
        self._seed = seed
        self._octave_mean = float(volume.mean())
        self._genera = genera
        if genera is not None:
            specs = [
                (CONVECTIVE, CloudType.CONVECTIVE, altitude),
                (STRATIFORM, CloudType.STRATUS, genera.stratus_altitude),
            ]
        else:
            specs = [(None, cloud_type, altitude)]
            weather = planet_weather(weather, planet, self.radius, weather_seed)
        self._families: list[_Family] = []
        for name, kind, base in specs:
            preset, field, peaks = planet_field(kind, base, seed, self.radius)
            table = tuple(float(v) for v in shift_table(peaks, field.threshold[0]))
            if genera is not None:
                # The family's maps; a convective field's shift by depth too.
                tables = table
                if field.min_top < 1.0:
                    tables = tuple(
                        tuple(float(v) for v in row)
                        for row in _depth_tables(seed, field, self.radius)
                    )
                job = executor.submit_once(_genus_weather, genera, name, tables)
            else:
                job = executor.submit_once(_weather, weather, table)
            self._families.append(_Family(name, preset, field, peaks, table, job))
        first = self._families[0]
        self.field = first.field
        self._preset = first.preset
        self._peaks = first.peaks
        self._table = first.table
        self._weather_job = first.weather_job
        self.weather: np.ndarray | None = None
        self._settings = settings
        #: Each family's shadow's layer (uCloudShadowLayer), -1 until ready
        self._shadow_layers = [PTA_int.empty_array(1) for _ in self._families]
        for layer in self._shadow_layers:
            layer[0] = -1
        #: (node, its first family, its second or None) the shadow is bound to
        self._shadow_nodes: list[tuple[NodePath, int, int | None]] = []
        self._map_binders: list[Any] = []
        self._casters = [_FamilyCaster(self, i) for i in range(len(self._families))]
        self._family_textures: list[tuple[Any, Any]] = []
        self._layer_far = PTA_float.empty_array(MAX_LAYERS)
        self._layer_params = PTA_LVecBase4f.empty_array(MAX_LAYERS * LAYER_VEC4S)
        params, self._deck_jobs = self._configure(settings)
        self._write_params(params)

        # The sea's reflection: the deck alone.
        self.root = parent.attach_new_node("cloud_march")
        self.root.stash()  # out of every camera until the set-up is done
        self.root.hide(HIDE_FROM_NEAR | HIDE_FROM_FAR | HIDE_FROM_CLOUD_PASS)
        self.deck = self.root.attach_new_node(
            deck_mesh(self.radius + self.field.slab[0] + 0.5 * self.field.slab[1])
        )
        self.deck.set_shader(_shaders()[1])
        self.deck.set_two_sided(True)
        self.deck.set_texture(load_cloud_atlas(game)[0])
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
        self._layer_count = PTA_int.empty_array(1)
        self._layer_count[0] = self._layers_used
        self._inputs: dict[str, Any] = {
            "layerParams": self._layer_params,
            "cloudNoise": build_noise_texture(volume),
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
            "uMinStep": min(min_step(f.field) for f in self._families),
            "uStepGrowth": STEP_GROWTH,
            "uLayerFar": self._layer_far,
            "uLayerCount": self._layer_count,
            "uFamilies": len(self._families),
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
            # The deck's (cloud_deck.vert, cloud.frag)
            "uDeck": 1,
            "uDeckLayer": self._layers_used - 1,
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
        self._set_inputs(uDeckLayer=self._layers_used - 1)
        if old.shell_octaves[-1:] != settings.shell_octaves[-1:]:
            names = ("deckLight", "otherDeckLight")
            for index, job in enumerate(deck_jobs):
                light = build_deck_light_texture(job.result())
                self._set_inputs(**{names[index]: light})
                if self.weather is not None:
                    self._textures[names[index]] = light
                    weather = self._family_textures[index][0]
                    self._family_textures[index] = (weather, light)
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
        """Pack the march's layers: for each family, one per billboard
        shell's distance band (its octaves), then its far one, which the
        reflection's deck shares (the first family's). The families share the
        bands' distances; with two, the layers (at most :data:`MAX_LAYERS`
        between them) are fewer bands each.

        :returns: the packed layer parameters, and each family's far layer's
            light table being made in the pool (futures)
        """
        seed = self._seed
        families = len(self._families)
        shells = max(min(settings.shells, MAX_LAYERS // families - 1), 1)
        schedule = settings.shell_octaves
        _, bands = shell_bands(self.grid, shells)
        params = np.zeros((MAX_LAYERS, LAYER_VEC4S, 4), dtype=np.float32)
        far = np.zeros(MAX_LAYERS, dtype=np.float32)
        stride = shells + 1
        jobs = []
        for index, family in enumerate(self._families):
            field, peaks = family.field, family.peaks
            optics = scale_march(family.preset, settings.march_scale).optics
            for k in range(stride):
                octaves = schedule[min(k, len(schedule) - 1)]
                layer = ShellSpec(
                    0,
                    (0.0, 0.0),
                    1.0,
                    octaves,
                    octave_offset(peaks, _peaks(seed, field, self.radius, octaves)),
                )
                # The far layer is the reflection's deck too: its cap, no fade.
                layer_optics = (
                    replace(optics, max_optical_depth=DECK_MAX_OPTICAL_DEPTH)
                    if k == shells
                    else optics
                )
                params[index * stride + k] = pack_layer_params(
                    layer.field_of(field),
                    # Per metre of the view's path
                    extinction=field.density * VIEW_EXTINCTION_SCALE,
                    octave_mean=self._octave_mean,
                    optics=layer_optics,
                    octaves=octaves,
                )
                if k < shells and index == 0:
                    far[k] = 0.5 * sum(bands[k])  # mid-band: where the shells cross
            deck = layer.field_of(field)
            # The deck's field and optics: the one its shadow is marched in.
            family.deck_field = deck
            family.deck_optics = layer_optics
            jobs.append(
                self.executor.submit_once(
                    _deck_light,
                    seed,
                    deck,
                    layer_optics,
                    self.radius,
                    family.table,
                    schedule[-1],
                )
            )
        self._layers_used = stride
        self._far_values = far
        return params, jobs

    def _write_params(self, params: np.ndarray) -> None:
        for i, row in enumerate(params.reshape(-1, 4)):
            self._layer_params[i] = LVecBase4f(*(float(v) for v in row))
        for i, value in enumerate(self._far_values):
            self._layer_far[i] = float(value)
        if hasattr(self, "_layer_count"):
            self._layer_count[0] = self._layers_used

    # ── the clouds' shadow (as sphere.CloudVolume's) ──────────────────────

    @property
    def families(self) -> int:
        """How many families of clouds it marches: 2 with genera."""
        return len(self._families)

    def shadow_caster(self, family: int = 0) -> _FamilyCaster:
        """The shadow of a family of the clouds (0: the convective clouds, or
        the only ones; 1: the stratus), as a volume's."""
        return self._casters[family]

    @property
    def shadow_layer(self) -> int:
        """The layer of ``layerParams`` the (first) family's shadow is marched
        in (its far one, the deck's), -1 until the set-up is done or after
        :meth:`remove`."""
        return self._casters[0].shadow_layer

    @property
    def shadow_field(self) -> DensityField:
        """The field the (first) family's shadow is marched in (the deck's)."""
        return self._families[0].deck_field

    def bind_shadow_inputs(self, node: NodePath, family: int | None = None) -> None:
        """As :meth:`sphere.CloudVolume.bind_shadow_inputs`: let the shaders
        on a node (the planet's terrain, the maps' bake) darken the sun under
        these clouds, through the deck's layers.

        :param family: bind this family's alone, as the first (the maps'
            bake); None = the first's, and the second's too as the other
            (``uCloudShadowOther``)
        """
        first = 0 if family is None else family
        other = 1 if family is None and len(self._families) > 1 else None
        node.set_shader_inputs(
            cloudNoise=self._inputs["cloudNoise"],
            layerParams=self._layer_params,
            planetRadius=self.radius,
            sunDir=self._sun,
            noiseSize=float(NOISE_SIZE),
            noiseOffset=Vec3(0, 0, 0),
            deckLightCosMin=DECK_LIGHT_COS_MIN,
            uCloudDebug=0,
            uCloudShadowLayer=self._shadow_layers[first],
            uCloudShadowOther=-1 if other is None else self._shadow_layers[other],
            otherWeather=_no_weather(),
            otherDeckLight=_no_deck_light(),
        )
        self._shadow_nodes.append((node, first, other))
        self._update_shadow_nodes()

    def _textures_of(self, index: int) -> dict[str, Any]:
        """A family's weather and light table, as the names the first
        family's shaders read them by."""
        return {
            "weatherMap": self._family_textures[index][0],
            "deckLight": self._family_textures[index][1],
        }

    def _update_shadow_nodes(self) -> None:
        """The shaded nodes' weather and layers: none until the set-up."""
        ready = self.weather is not None
        stride = self._layers_used
        for index, layer in enumerate(self._shadow_layers):
            layer[0] = index * stride + stride - 1 if ready else -1
        self._shadow_nodes = [n for n in self._shadow_nodes if not n[0].is_empty()]
        if ready:
            for node, first, other in self._shadow_nodes:
                node.set_shader_inputs(**self._textures_of(first))
                if other is not None:
                    node.set_shader_inputs(
                        otherWeather=self._family_textures[other][0],
                        otherDeckLight=self._family_textures[other][1],
                    )

    def sun_visibility(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """The share of the sun the first family's clouds leave points
        (:meth:`sphere.CloudVolume.sun_visibility`)."""
        return self._casters[0].sun_visibility(*args, **kwargs)

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
                for job in [f.weather_job for f in self._families] + self._deck_jobs
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
        for family in self._families:
            base, thickness = family.field.slab
            nearest = min(
                nearest, max(base - altitude, altitude - base - thickness, 0.0)
            )
        return nearest

    def remove(self) -> None:
        """Take the clouds off the scene (and their shadow off the nodes it
        shaded)."""
        for layer in self._shadow_layers:
            layer[0] = -1
        self._shadow_nodes = []
        self.detach()
        self.root.remove_node()

    def _finish(self) -> None:
        """Finish the set-up begun in the pool (waiting for it if need be)."""
        if self._weather_job is None:
            return
        self._family_textures = []
        self._textures = {}
        names = (("weatherMap", "deckLight"), ("otherWeather", "otherDeckLight"))
        for index, family in enumerate(self._families):
            family.weather = family.weather_job.result()
            weather = build_weather_texture(family.weather)
            light = build_deck_light_texture(self._deck_jobs[index].result())
            self._family_textures.append((weather, light))
            self._textures[names[index][0]] = weather
            self._textures[names[index][1]] = light
            family.weather_job = None
        self.weather = self._families[0].weather
        self._weather_job = self._deck_jobs = None
        for node in self._nodes:
            node.set_shader_inputs(**self._textures)
        self._inputs.update(self._textures)
        self.stats["pending"] = 0
        self._update_shadow_nodes()
        self.root.unstash()
        for layer in self._layers:
            layer.set_active(self._visible)
