"""
noise.py — The cloud density field, as data.

A cloud type's shape is ONE continuous world-space density function that every
billboard fragment of that type samples (shaders/cloud_field.glsl, included by
cloud.frag). This module owns that function: the tileable noise octave it is built
from, the per-type parameters that shape it (:class:`DensityField`), and a NumPy
evaluator that reproduces the shader's result exactly over a planet
(:func:`density_on_sphere`), so placement can be driven by the very field that is
drawn. It also owns the coverage calibration (:func:`sphere_column_peaks`), the
phase function and the packing of a type into the shaders' ``layerParams``
(:func:`pack_layer_params`).

Density must be a function of world position ALONE — never per cloud — or
overlapping billboards disagree about where the boundary is. Design notes:
docs/source/clouds.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import SamplerState, Texture

from world_builder.terrain.noise import native_library
from world_builder.utils import clamp

# Volume resolution, and the random lattice it is upsampled from. NOISE_SIZE must be
# a multiple of NOISE_LATTICE so the result tiles.
NOISE_SIZE = 64
NOISE_LATTICE = 16

# Noise units per lattice cell: one feature of one octave. Divide by noise_scale
# for metres.
NOISE_FEATURE = NOISE_SIZE / NOISE_LATTICE

# Per-octave offsets, so the octaves never re-align into a grid. MUST match
# OCTAVE_n_OFFSET in cloud_field.glsl (a test reads the shader), or billboards get
# placed where the shader draws no cloud.
OCTAVE_OFFSETS = (
    (0.0, 0.0, 0.0),
    (0.31, 0.73, 0.19),
    (-0.57, 0.11, 0.83),
    (0.67, -0.29, 0.41),
)
#: One octave of weight 1 at scale 1, unshifted: :func:`sample_volume` through
#: the native fbm (``x * 1.0 + 0.0`` is ``x``: the same samples).
_ONE = (1.0,)
_ZERO = ((0.0, 0.0, 0.0),)

#: Coverage calibration resolution (see :func:`sphere_column_peaks`): columns
#: (COVERAGE_GRID squared) and levels through the slab.
COVERAGE_GRID = 96
COVERAGE_LEVELS = 32

#: vec4s per cloud type in the layerParams uniform array, and the most layers one
#: field may mix (LOD shells, and a planet's deck, count as layers). Must match
#: cloud_field.glsl / cloud_sphere.vert.
LAYER_VEC4S = 13
MAX_LAYERS = 8

#: A convective field's cell octave (:attr:`DensityField.cell_weight`): its
#: scale over the first octave's (four times its features: 8 km for cumulus),
#: and its offset. MUST match CELL_SCALE / CELL_OFFSET in cloud_field.glsl.
CELL_SCALE = 0.25
CELL_OFFSET = (0.13, 0.59, 0.37)


@dataclass(frozen=True)
class DensityField:
    """The density field of ONE cloud type: a slab of sky and how it is carved.

    Frozen, because the GPU copy is packed once and must not drift from it.

    slab            (base_z, thickness) metres, shared by every cloud of the type
    density         optical density per metre of fully-dense cloud
    noise_scale     world metres → noise coords, per axis (anisotropy → cirrus)
    octave_scales   the four fbm octave frequencies
    octave_weights  the four octave amplitudes
    coverage        fraction of the slab's columns (zenith-projected) that hold
                    cloud at any height — the authored amount of cloud
    edge_softness   smoothstep window width, in standard deviations of this field's
                    fbm (see :func:`fbm_sigma`); narrow = crisp cauliflower edge
    threshold       (lo, hi) window DERIVED from coverage and edge_softness against
                    a noise volume; None until :func:`with_threshold`
    base_ramp       how abruptly density fills in above the base (flat undersides)
    top_erosion     how far the threshold rises by the slab top (bulging tops)
    top_exponent    the curve of that rise
    offset          noise-space translation decorrelating types that share one
                    volume (see :func:`field_offset`)

    A convective field's clouds grow with the weather's convective DEPTH (0
    to 1, a cube map: :mod:`genera`), from fair-weather cumulus to
    cumulonimbus; the three are neutral by default (no depth):

    min_top         the clouds' top at depth 0, as a share of the slab; the
                    top rises with depth to the slab's (the profile above,
                    lift and base, follows the local top: :func:`top_fraction`)
    anvil           at depth 1, how far the threshold falls just under the
                    top, in fbm units: the cloud spreads there, an anvil
    cell_weight     at depth 1, the weight of a cell octave :data:`CELL_SCALE`
                    of the first: towers kilometres wide
    """

    slab: tuple = (1000.0, 600.0)
    density: float = 0.03
    noise_scale: tuple = (0.001, 0.001, 0.001)
    octave_scales: tuple = (1.0, 2.0, 7.0, 16.0)
    octave_weights: tuple = (0.5, 0.25, 0.125, 0.0625)
    coverage: float = 0.29
    edge_softness: float = 0.48
    threshold: tuple | None = None
    base_ramp: float = 5.0
    top_erosion: float = 0.40
    top_exponent: float = 3.0
    offset: tuple = (0.0, 0.0, 0.0)
    min_top: float = 1.0
    anvil: float = 0.0
    cell_weight: float = 0.0

    @property
    def feature_size(self) -> tuple:
        """:returns: per-axis world size in metres of the field's largest feature."""
        return tuple(NOISE_FEATURE / s for s in self.noise_scale)


@dataclass(frozen=True)
class CloudOptics:
    """How light travels through ONE cloud type. Per-fragment only, so unlike the
    field it can be changed live.

    sun_steps           steps of the per-fragment sun march (0 = no self-shadow)
    sun_step_length     metres per step; steps x length ≈ one cloud's depth
    shadow_strength     scales the sunward optical depth (1 = physical)
    multiple_scattering sky-coloured, NOT phase-weighted fill for shadowed cloud
    powder_strength     empirical darkening of optically thin cloud (0 = off)
    powder_length       metres turning LOCAL density into the powder's optical depth
    max_optical_depth   cap on what one billboard contributes, so a quad never
                        becomes a hard-edged opaque blob
    forward_gain        brightness looking into the sun, relative to side-on (1)
    backward_gain       brightness with the sun behind you, relative to side-on
    forward_anisotropy  forward lobe's Henyey-Greenstein g: the falloff SHAPE only;
                        the gains are pinned by :func:`phase_weights`
    """

    sun_steps: int = 4
    sun_step_length: float = 160.0
    shadow_strength: float = 1.0
    multiple_scattering: float = 0.12
    powder_strength: float = 0.6
    powder_length: float = 200.0
    max_optical_depth: float = 1.5
    forward_gain: float = 4.72
    backward_gain: float = 1.52
    forward_anisotropy: float = 0.4


#: Anisotropy of the phase function's fixed backward lobe.
BACKWARD_ANISOTROPY = -0.4
#: The HG peak diverges as g → 1. The CLAMPED value is what gets packed, so the
#: shader evaluates the same basis the weights were solved against.
FORWARD_ANISOTROPY_MAX = 0.95
#: Single-scattering Mie asymmetry of cloud droplets (OPAC); see delta_eddington.
MIE_ASYMMETRY_WATER = 0.85
MIE_ASYMMETRY_ICE = 0.80


def delta_eddington(asymmetry: float, applicability: float = 1.0) -> float:
    """The delta-Eddington similarity scaling, ``g -> g / (1 + g)``.

    Raw Mie asymmetry fed to a single-scattering shader saturates the sunward sky;
    the scaled value stands in for the flatter phase function multiple scattering
    produces (water 0.85 → 0.46). Thin cloud is near single scattering, so
    *applicability* (1 = thick, 0 = thin) blends back toward the raw value.
    """
    scaled = asymmetry / (1.0 + asymmetry)
    return asymmetry + (scaled - asymmetry) * applicability


def henyey_greenstein(cos_angle: ArrayLike, anisotropy: float) -> np.ndarray | float:
    """The Henyey-Greenstein phase function, matching the shader's ``hgPhase``.

    :param cos_angle: cosine between the light and the view ray (1 = into the light)
    :param anisotropy: the lobe's g; positive scatters forward
    """
    g = anisotropy
    g2 = g * g
    return 0.25 * (1.0 - g2) * (1.0 + g2 - 2.0 * g * np.asarray(cos_angle)) ** -1.5


def clamped_anisotropy(optics: CloudOptics) -> float:
    """:returns: the forward anisotropy in force, shared by solve, pack and CPU."""
    return clamp(optics.forward_anisotropy, 0.0, FORWARD_ANISOTROPY_MAX)


def _basis(g: float, cos_angle: ArrayLike) -> np.ndarray:
    """:returns: (..., 3) forward-lobe, backward-lobe and isotropic phase terms."""
    cos_angle = np.asarray(cos_angle, dtype=np.float64)
    return np.stack(
        [
            henyey_greenstein(cos_angle, g),
            henyey_greenstein(cos_angle, BACKWARD_ANISOTROPY),
            np.full_like(cos_angle, 0.25),
        ],
        axis=-1,
    )


def phase_weights(optics: CloudOptics) -> tuple[float, ...]:
    """Solve the three lobe weights that hit this type's gains exactly.

    Two lobes alone leave one degree of freedom for two observables, so the forward
    and backward brightness fight each other. The isotropic term makes it an exact
    3x3 solve pinning 0° → forward_gain, 180° → backward_gain, 90° → 1.

    :returns: (forward, backward, isotropic) weights
    :raises ValueError: if the phase would go negative anywhere (a dark hole)
    """
    g = clamped_anisotropy(optics)
    weights = np.linalg.solve(
        _basis(g, [1.0, -1.0, 0.0]),
        np.array([optics.forward_gain, optics.backward_gain, 1.0]),
    )
    # Check the whole range: a dip would fall between the pinned angles.
    sweep = _basis(g, np.linspace(-1.0, 1.0, 361)) @ weights
    if sweep.min() < 0.0:
        raise ValueError(
            f"forward_gain={optics.forward_gain} with "
            f"backward_gain={optics.backward_gain} makes the phase function go "
            f"negative ({sweep.min():.3f}) at some angle, which renders as a dark "
            "hole. Move the gains closer together, or raise forward_anisotropy to "
            "concentrate the forward lobe."
        )
    return tuple(float(w) for w in weights)


def phase(optics: CloudOptics, cos_angle: ArrayLike) -> np.ndarray | float:
    """The shader's phase function on the CPU; 1 when side-lit (90°).

    :param cos_angle: cosine between the sun and the view ray (1 = into the sun)
    """
    basis = _basis(clamped_anisotropy(optics), cos_angle)
    forward, backward, isotropic = phase_weights(optics)
    return (
        basis[..., 0] * forward + basis[..., 1] * backward + basis[..., 2] * isotropic
    )


def field_offset(seed: int) -> tuple[float, float, float]:
    """:returns: a decorrelating (x, y, z) noise-space offset for a cloud type.

    A shifted copy of the stationary, repeat-wrapped volume is an independent field
    with identical statistics, for the cost of one add.
    """
    rng = np.random.default_rng(0x51EED ^ int(seed))
    return tuple(float(v) for v in rng.uniform(0.0, NOISE_SIZE, 3))


# ── The noise volume ───────────────────────────────────────────────────────────


def _axis_weights(size: int, lattice: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (i0, i1, t) wrapping lattice indices and smoothstep weights."""
    coord = np.arange(size, dtype=np.float64) * (lattice / size)
    i0 = np.floor(coord).astype(np.int64) % lattice
    frac = coord - np.floor(coord)
    # Smoothstep keeps the field C1 across lattice cells: a narrow threshold window
    # would show every crease of a plain linear upsample.
    return i0, (i0 + 1) % lattice, (frac * frac * (3.0 - 2.0 * frac)).astype(np.float32)


def value_noise_volume(
    size: int = NOISE_SIZE, lattice: int = NOISE_LATTICE, seed: int = 0
) -> np.ndarray:
    """One tileable octave of 3D value noise, normalised to [0,1].

    Value noise, as the reference shader uses: the lattice's blobbiness is what the
    low octaves carry the formations with.

    :returns: (size, size, size) float32, indexed [z, y, x]
    """
    if size % lattice != 0:
        raise ValueError(f"size {size} must be a whole multiple of lattice {lattice}")

    rng = np.random.default_rng(seed)
    grid = rng.random((lattice, lattice, lattice), dtype=np.float32)
    i0, i1, t = _axis_weights(size, lattice)

    # Separable lerps, x then y then z: identical arithmetic to a per-corner blend.
    gx = grid[:, :, i0] + (grid[:, :, i1] - grid[:, :, i0]) * t[None, None, :]
    gy = gx[:, i0, :] + (gx[:, i1, :] - gx[:, i0, :]) * t[None, :, None]
    volume = gy[i0] + (gy[i1] - gy[i0]) * t[:, None, None]

    # Span the full [0,1]: the threshold window is absolute, and the upload is 8-bit.
    lo, hi = float(volume.min()), float(volume.max())
    return ((volume - lo) / max(hi - lo, 1e-9)).astype(np.float32)


def quantise(volume: np.ndarray) -> np.ndarray:
    """:returns: *volume* rounded to the 8 bits per texel the GPU copy stores, so
    CPU placement and GPU drawing cannot drift apart at the silhouette."""
    return (np.round(volume * 255.0) / 255.0).astype(np.float32)


def build_noise_texture(volume: np.ndarray) -> Texture:
    """Wrap a noise volume in a repeat-wrapped, linearly filtered 3D texture.

    Needs no graphics context. A 3D RAM image is page-major, i.e. numpy's [z, y, x].

    :param volume: a (n, n, n) volume in [0,1], quantised or not
    """
    size = volume.shape[0]
    texture = Texture("cloud_noise")
    texture.setup_3d_texture(size, size, size, Texture.T_unsigned_byte, Texture.F_red)
    texture.set_ram_image(
        np.ascontiguousarray(np.round(volume * 255.0), np.uint8).tobytes()
    )
    for set_wrap in (texture.set_wrap_u, texture.set_wrap_v, texture.set_wrap_w):
        set_wrap(Texture.WM_repeat)
    texture.set_magfilter(SamplerState.FT_linear)
    # NOT mipmapped: a 3D mip chain averages the 16x octave away to a constant.
    texture.set_minfilter(SamplerState.FT_linear)
    return texture


# ── The field, evaluated on the CPU exactly as the shader evaluates it ─────────


def sample_volume(volume: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample *volume* as ``texture(cloudNoise, coord / NOISE_SIZE)`` does:
    trilinear and repeat-wrapped, including hardware linear's half-texel offset
    (coordinate u addresses texel centre u*size - 0.5). Getting that wrong shifts
    every octave by half its texel (250 m for the lowest cumulus octave), moving
    placement off the drawn cloud.

    :param volume: (n, n, n) float array indexed [z, y, x]
    :param coords: (..., 3) sample positions in noise coords
    """
    native = native_library()
    if native is not None and volume.dtype == np.float32:
        return native.volume_fbm(volume, coords, _ONE, _ONE, _ZERO, 1, 0.0)
    size = volume.shape[0]
    coords = np.asarray(coords, dtype=np.float64)
    texel = coords - 0.5
    base = np.floor(texel)
    frac = texel - base
    i0 = base.astype(np.int64) % size
    i1 = (i0 + 1) % size
    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
    x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2]
    tx, ty, tz = frac[..., 0], frac[..., 1], frac[..., 2]

    c000, c100 = volume[z0, y0, x0], volume[z0, y0, x1]
    c010, c110 = volume[z0, y1, x0], volume[z0, y1, x1]
    c001, c101 = volume[z1, y0, x0], volume[z1, y0, x1]
    c011, c111 = volume[z1, y1, x0], volume[z1, y1, x1]
    c00 = c000 + (c100 - c000) * tx
    c01 = c010 + (c110 - c010) * tx
    c10 = c001 + (c101 - c001) * tx
    c11 = c011 + (c111 - c011) * tx
    c0 = c00 + (c01 - c00) * ty
    c1 = c10 + (c11 - c10) * ty
    return c0 + (c1 - c0) * tz


def fbm(
    volume: np.ndarray,
    coords: np.ndarray,
    spec: DensityField,
    octaves: int = 4,
    octave_mean: float | None = None,
) -> np.ndarray:
    """The four-octave fbm of the shader's ``cloudFbm``.

    :param volume: the noise octave, indexed [z, y, x]
    :param coords: (..., 3) noise coords (world * noise_scale + offset)
    :param spec: the field (its octave scales and weights)
    :param octaves: octaves evaluated (2-4); the finer ones count as their
        mean, so dropping them keeps the fbm's mean (the far shells' saving;
        see :func:`sphere.octave_offset` for the spread)
    :param octave_mean: the noise's mean; None = *volume*'s
    :returns: (...) raw fbm, in [0, sum of the weights]
    """
    if octave_mean is None:
        octave_mean = float(volume.mean())
    native = native_library()
    if native is not None and volume.dtype == np.float32:
        # zip() below: as many octaves as the shortest of the three
        count = min(
            len(spec.octave_scales), len(spec.octave_weights), len(OCTAVE_OFFSETS)
        )
        return native.volume_fbm(
            volume,
            coords,
            spec.octave_scales[:count],
            spec.octave_weights[:count],
            OCTAVE_OFFSETS[:count],
            min(octaves, count),
            octave_mean,
        )
    total = np.zeros(np.shape(coords)[:-1], dtype=np.float64)
    for index, (scale, weight, offset) in enumerate(
        zip(spec.octave_scales, spec.octave_weights, OCTAVE_OFFSETS)
    ):
        if index < octaves:
            sample = sample_volume(volume, coords * scale + np.asarray(offset))
            total += sample * weight
        else:
            total += octave_mean * weight
    return total


def top_fraction(spec: DensityField, depth: ArrayLike = 0.0) -> np.ndarray | float:
    """The clouds' local top, as a share of the slab, at convective *depth*
    (:attr:`DensityField.min_top`: 1 for a field with no depth)."""
    return spec.min_top + (1.0 - spec.min_top) * np.clip(depth, 0.0, 1.0)


def _smoothstep(lo: float, hi: float, x: ArrayLike) -> np.ndarray:
    t = np.clip((np.asarray(x) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


#: The anvil (:attr:`DensityField.anvil`): the convective depth it grows
#: over, and the band under the local top it spreads in. MUST match
#: cloud_field.glsl's thresholdLift.
ANVIL_DEPTH = (0.6, 1.0)
ANVIL_BAND = (0.7, 0.95)


def threshold_lift(
    spec: DensityField, h: np.ndarray, depth: ArrayLike = 0.0
) -> np.ndarray:
    """How far the threshold has risen at normalised slab height *h*.

    Raising the THRESHOLD, not fading density, is what makes tops bulge: strong
    columns keep clearing the rising bar and tower, weak ones stop short. Fading
    density could only slice every cloud off at one altitude. Once the lift passes
    the fbm's maximum nothing is cloud, which bounds the slab with no hard clamp.

    The profile follows the local top at convective *depth*
    (:func:`top_fraction`); above it nothing is cloud (an infinite lift). A
    deep cloud's threshold falls again just under its top: the anvil.
    """
    local = np.asarray(h) / top_fraction(spec, depth)
    lift = spec.top_erosion * np.clip(local, 0.0, 1.0) ** spec.top_exponent
    if spec.anvil:
        lift = lift - spec.anvil * _smoothstep(*ANVIL_DEPTH, depth) * _smoothstep(
            *ANVIL_BAND, local
        )
    if spec.min_top < 1.0:
        lift = np.where(local < 1.0, lift, np.inf)
    return lift


def cell_shift(
    volume: np.ndarray,
    spec: DensityField,
    coords: np.ndarray,
    depth: ArrayLike,
    octave_mean: float | None = None,
) -> np.ndarray | float:
    """What the cell octave adds to the fbm at convective *depth*
    (:attr:`DensityField.cell_weight`): its noise about its mean, so the
    coverage barely moves. 0 for a field without one.

    :param coords: (..., 3) noise coords (world * noise_scale + offset)
    """
    if not spec.cell_weight:
        return 0.0
    if octave_mean is None:
        octave_mean = float(volume.mean())
    cell = sample_volume(
        volume, np.asarray(coords) * CELL_SCALE + np.asarray(CELL_OFFSET)
    )
    return spec.cell_weight * np.clip(depth, 0.0, 1.0) * (cell - octave_mean)


def fbm_sigma(volume: np.ndarray, spec: DensityField) -> float:
    """Standard deviation of the field's fbm, the unit of ``edge_softness``.

    Treating octaves as independent samples: std(volume) * sqrt(sum(w²)). Derived
    so retuning the octave weights does not silently change edge crispness.
    """
    weights = np.asarray(spec.octave_weights, dtype=np.float64)
    return float(volume.std() * np.sqrt((weights**2).sum()))


def sphere_column_peaks(
    volume: np.ndarray,
    spec: DensityField,
    radius: float,
    octaves: int = 4,
    columns: int = COVERAGE_GRID * COVERAGE_GRID,
    levels: int = COVERAGE_LEVELS,
    depth: float = 0.0,
) -> np.ndarray:
    """The per-column peak of ``fbm - threshold_lift``, SORTED ascending:
    radial columns through the slab above a sphere, spread evenly over it (a
    Fibonacci lattice).

    A column holds cloud exactly when its peak exceeds threshold_lo, so this
    sorted array is the inverse of the coverage function: the (1 - coverage)
    quantile IS the threshold. (base_ramp scales density, never whether a
    column is cloud, so it drops out.) Columns cross the noise volume at every
    phase and angle; the levels must resolve the finest octave (~125 m
    features at the cumulus scale against a 1000 m slab), or thin peaks are
    missed and the coverage comes out low.

    :param radius: sea-level radius, metres
    :param octaves: octaves evaluated (see :func:`fbm`)
    :param depth: a convective field's depth, the same everywhere: the levels
        span its local top, as many more as it is taller than at depth 0 (so
        they still resolve the finest octave)
    :returns: (columns,) sorted peaks, in raw fbm units
    """
    dirs = _fibonacci_dirs(columns)
    base, thickness = spec.slab
    top = top_fraction(spec, depth)
    levels = int(math.ceil(levels * top / top_fraction(spec, 0.0)))
    h = (np.arange(levels) + 0.5) / levels * top
    points = dirs[:, None, :] * (radius + base + h * thickness)[None, :, None]
    coords = points * np.asarray(spec.noise_scale) + np.asarray(spec.offset)
    raw = fbm(volume, coords, spec, octaves) + cell_shift(volume, spec, coords, depth)
    peaks = (raw - threshold_lift(spec, h, depth)[None, :]).max(axis=1)
    peaks.sort()
    return peaks


def threshold_from_peaks(
    peaks: np.ndarray, coverage: float, edge_softness: float, sigma: float
) -> tuple[float, float]:
    """Read the (lo, hi) threshold window off measured, sorted column peaks.

    The peaks never change for a field, so caching them makes a coverage change an
    array index — which is what lets coverage be swept live.

    :param peaks: sorted column peaks (:func:`sphere_column_peaks`)
    :param coverage: the share of columns to hold cloud, in [0, 1]
    :param edge_softness: the window's width, in *sigma*
    :param sigma: the fbm's standard deviation (:func:`fbm_sigma`)
    :returns: (lo, hi) raw fbm values
    """
    width = float(edge_softness) * float(sigma)
    # Exact endpoints: a 0/1 quantile lands ON an observed peak, and the test is
    # strict, so "everything" would miss the strongest column.
    if coverage >= 1.0:
        lo = float(peaks[0]) - width
    elif coverage <= 0.0:
        lo = float(peaks[-1]) + width
    else:
        index = (1.0 - coverage) * (len(peaks) - 1)
        low = int(np.floor(index))
        frac = index - low
        lo = float(
            peaks[low] * (1.0 - frac) + peaks[min(low + 1, len(peaks) - 1)] * frac
        )
    return lo, lo + width


def with_threshold(spec: DensityField, peaks: np.ndarray, sigma: float) -> DensityField:
    """:returns: *spec* with its threshold derived from already-measured peaks."""
    return replace(
        spec,
        threshold=threshold_from_peaks(peaks, spec.coverage, spec.edge_softness, sigma),
    )


def _require_resolved(spec: DensityField):
    """Fail loudly: a None threshold would otherwise surface far away, or read as 0
    and turn the whole slab into solid cloud."""
    if spec.threshold is None:
        raise ValueError(
            "this DensityField's threshold is still derived-but-unresolved; derive "
            "it with noise.with_threshold(spec, sphere_column_peaks(...), "
            "fbm_sigma(...)) first (the threshold for a given coverage depends on "
            "the noise volume, so a field alone cannot know it)"
        )


def shadow_calibration(spec: DensityField, octave_mean: float) -> tuple[float, float]:
    """Affine correction making the sun march's 2-octave fbm match the full field.

    The two lowest octaves correlate with the full fbm at 0.97 but have the wrong
    mean and spread, so uncorrected the march sees 1.65x the cloud (grey tops in
    full sun). Treating octaves as independent (mean m, sd s): the gain matches the
    spreads sqrt(Σw²)/sqrt(w0²+w1²), the bias then matches the means.

    :returns: (gain, bias) for ``gain * two_octave_sum + bias``
    """
    weights = np.asarray(spec.octave_weights, dtype=np.float64)
    full_sd = np.sqrt((weights**2).sum())
    pair_sd = np.sqrt((weights[:2] ** 2).sum())
    gain = full_sd / max(pair_sd, 1e-9)
    bias = octave_mean * (weights.sum() - gain * weights[:2].sum())
    return float(gain), float(bias)


def pack_layer_params(
    spec: DensityField,
    extinction: float,
    octave_mean: float = 0.5,
    optics: CloudOptics = CloudOptics(),
    aspect: float = 1.0,
    fade_in: tuple | None = None,
    fade_out: tuple | None = None,
    octaves: int = 4,
) -> np.ndarray:
    """Flatten one type into its LAYER_VEC4S rows of the layerParams array.

    The row layout IS the contract with cloud_field.glsl's getField (and the rows
    cloud_sphere.vert reads directly: 10, 11); change them together.

    :param spec: a resolved field
    :param extinction: per-BILLBOARD extinction (density / (overlap x the
        sprites' mean alpha), so a ray through overlapping billboards takes the
        field's own optical depth), or per metre of path for a planet's deck
    :param octave_mean: noise volume mean, for :func:`shadow_calibration` and the
        octaves not evaluated
    :param optics: the type's optics (phase weights are solved here)
    :param aspect: billboard long:short ratio (area-preserving, so no recalibration)
    :param fade_in: (lo, hi) camera distances the layer fades in over, None = none
    :param fade_out: (lo, hi) it fades out over, None = never (the deck; the
        air hazes it)
    :param octaves: octaves evaluated (the rest count as *octave_mean*; see
        :func:`fbm`)
    :returns: a (LAYER_VEC4S, 4) float32 array
    :raises ValueError: for an unresolved field, or gains with no valid phase
    """
    _require_resolved(spec)
    gain, bias = shadow_calibration(spec, octave_mean)
    forward, backward, isotropic = phase_weights(optics)
    # "No fade" sentinels make the shader's smoothsteps degenerate to 1 branch-free.
    fade_in = fade_in if fade_in is not None else (-1.0, 0.0)
    fade_out = fade_out if fade_out is not None else (1e9, 2e9)
    o = optics
    return np.array(
        [
            (*spec.noise_scale, spec.threshold[0]),
            spec.octave_scales,
            spec.octave_weights,
            (*spec.offset, spec.threshold[1]),
            (spec.slab[0], spec.slab[1], spec.base_ramp, spec.top_erosion),
            # Two extinctions: the billboard sum's, and the FIELD's for the sun
            # march, a continuous integral that must not be scaled by 1/phi.
            (extinction, spec.density, 0.0, 0.0),
            (spec.top_exponent, gain, bias, o.max_optical_depth),
            (o.sun_steps, o.sun_step_length, o.shadow_strength, o.multiple_scattering),
            (o.powder_strength, o.powder_length, forward, backward),
            (isotropic, clamped_anisotropy(o), octaves, octave_mean),
            (aspect, 0.0, 0.0, 0.0),
            (*fade_in, *fade_out),
            (spec.min_top, spec.anvil, spec.cell_weight, 0.0),
        ],
        dtype=np.float32,
    )


# ── On a planet ────────────────────────────────────────────────────────────────


def density_on_sphere(
    volume: np.ndarray,
    weather: np.ndarray,
    spec: DensityField,
    points: np.ndarray,
    radius: float,
    octaves: int = 4,
    early_out: bool = False,
) -> np.ndarray:
    """The shader's ``cloudDensity`` (cloud_field.glsl): the slab
    is a height above the sea-level sphere, the noise is sampled in 3D at the
    planet-frame position (seamless on a sphere), and the weather map shifts
    the fbm up (overcast) or down (clear) by direction.

    :param volume: the noise octave (quantised, to match the GPU)
    :param weather: (6, n, n, 2) cube map of coverage and shift
        (:func:`weather.weather_cube`), (6, n, n, 3) with the convective
        depth too (:mod:`genera`), or its (6, n, n) shift plane alone (depth
        0)
    :param spec: a resolved field (a shell's: :meth:`sphere.ShellSpec.field_of`)
    :param points: (..., 3) planet-frame positions, metres
    :param radius: sea-level radius, metres
    :param octaves: octaves evaluated (the shell's; see :func:`fbm`)
    :param early_out: evaluate as the shader does, stopping where the octaves
        still to come cannot reach the threshold (the same result, which a
        test checks; on the CPU only slower)
    :returns: (...) density in [0, 1]
    """
    from world_builder.clouds.weather import sample_cube

    _require_resolved(spec)
    points = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(points, axis=-1)
    base, thickness = spec.slab
    h = (r - radius - base) / max(thickness, 1e-3)
    coords = points * np.asarray(spec.noise_scale) + np.asarray(spec.offset)
    plane = weather[..., 1] if weather.ndim == 4 else weather
    shift = sample_cube(plane, points)
    depth = 0.0
    if weather.ndim == 4 and weather.shape[-1] > 2:
        depth = sample_cube(weather[..., 2], points)
    shift = shift + cell_shift(volume, spec, coords, depth)
    raw = fbm(volume, coords, spec, octaves) + shift
    if early_out:
        # cloud_field.glsl's cloudDensity: what is still to come at most.
        lo = spec.threshold[0] + threshold_lift(spec, h, depth)
        mean = float(volume.mean())
        w = spec.octave_weights
        rest = sum((1.0 if k < octaves else mean) * w[k] for k in (2, 3))
        first = sample_volume(volume, coords * spec.octave_scales[0]) * w[0] + shift
        second = (
            sample_volume(
                volume, coords * spec.octave_scales[1] + np.asarray(OCTAVE_OFFSETS[1])
            )
            * w[1]
        )
        alive = (first + w[1] + rest > lo) & (first + second + rest > lo)
        raw = np.where(alive, raw, -np.inf)
    return _slab_density(spec, h, raw, depth)


def _slab_density(
    spec: DensityField, h: np.ndarray, raw: np.ndarray, depth: ArrayLike = 0.0
) -> np.ndarray:
    """cloud_field.glsl's ``slabDensity``, zero outside the slab: the threshold
    window raised by :func:`threshold_lift`, times the base's fill, both
    following the local top at convective *depth*.

    :param h: normalised slab height
    :param raw: the fbm, shift included
    """
    lift = threshold_lift(spec, h, depth)
    above = np.isinf(lift)  # over the local top: no cloud
    lift = np.where(above, 0.0, lift)
    lo, hi = spec.threshold[0] + lift, spec.threshold[1] + lift
    t = np.clip((raw - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    t = np.where(above, 0.0, t)
    coverage = t * t * (3.0 - 2.0 * t)
    local = np.clip(h / top_fraction(spec, depth), 0.0, 1.0)
    base_fill = 1.0 - np.exp2(-spec.base_ramp * local)
    inside = (h > 0.0) & (h < 1.0)
    return np.where(inside, coverage * base_fill, 0.0)


def _sun_march_samples(
    volume: np.ndarray,
    spec: DensityField,
    optics: CloudOptics,
    points: np.ndarray,
    sun_dirs: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The sun march's steps before the weather and the calibration: their
    slab heights and two-octave sums, (..., sun_steps) each."""
    steps = np.arange(optics.sun_steps) + 0.5
    positions = points[..., None, :] + sun_dirs[..., None, :] * (
        steps[:, None] * optics.sun_step_length
    )
    base, thickness = spec.slab
    h = (np.linalg.norm(positions, axis=-1) - radius - base) / max(thickness, 1e-3)
    coords = positions * np.asarray(spec.noise_scale) + np.asarray(spec.offset)
    return h, fbm(volume, coords, spec, octaves=2, octave_mean=0.0)


def _sun_depth(
    spec: DensityField,
    optics: CloudOptics,
    h: np.ndarray,
    pair: np.ndarray,
    shift: ArrayLike,
    octave_mean: float,
    depth: ArrayLike = 0.0,
) -> np.ndarray:
    """:func:`sun_optical_depth_on_sphere` from :func:`_sun_march_samples`."""
    gain, bias = shadow_calibration(spec, octave_mean)
    shift = np.asarray(shift, dtype=np.float64)[..., None]
    depth = np.asarray(depth, dtype=np.float64)[..., None]
    density = _slab_density(spec, h, pair * gain + shift + bias, depth).sum(axis=-1)
    return density * optics.sun_step_length * spec.density * optics.shadow_strength


def sun_optical_depth_on_sphere(
    volume: np.ndarray,
    spec: DensityField,
    optics: CloudOptics,
    points: np.ndarray,
    sun_dirs: np.ndarray,
    radius: float,
    shift: ArrayLike = 0.0,
    octave_mean: float | None = None,
    depth: ArrayLike = 0.0,
) -> np.ndarray:
    """The shader's ``sunOpticalDepth`` in planet mode: *sun_steps* steps of
    *sun_step_length* toward the sun, from half a step out, through the two
    lowest octaves calibrated onto the full field (:func:`shadow_calibration`).
    Its early-out is exact, so it is left out.

    :param volume: the noise octave (quantised, to match the GPU)
    :param spec: a resolved field (the layer's)
    :param optics: the layer's optics (its steps, step length and strength)
    :param points: (..., 3) planet-frame positions the march starts from
    :param sun_dirs: (..., 3) unit directions toward the sun
    :param radius: sea-level radius, metres
    :param shift: the weather's fbm shift at *points* (broadcast), the cell
        octave's there included (:func:`cell_shift`)
    :param octave_mean: the noise's mean; None = *volume*'s
    :param depth: the convective depth at *points* (broadcast): the march
        keeps it, as it keeps the shift
    :returns: (...) optical depth toward the sun
    """
    _require_resolved(spec)
    points = np.asarray(points, dtype=np.float64)
    if optics.sun_steps <= 0:
        return np.zeros(points.shape[:-1])
    if octave_mean is None:
        octave_mean = float(volume.mean())
    sun_dirs = np.broadcast_to(np.asarray(sun_dirs, dtype=np.float64), points.shape)
    h, pair = _sun_march_samples(volume, spec, optics, points, sun_dirs, radius)
    return _sun_depth(spec, optics, h, pair, shift, octave_mean, depth)


#: The lowest sun cosine (at the cloud) of :func:`deck_light_table`: below
#: it the planet's shadow has taken the sunlight at any cloud height.
DECK_LIGHT_COS_MIN = -0.3


def _fibonacci_dirs(count: int) -> np.ndarray:
    """:returns: (count, 3) unit directions spread evenly over the sphere"""
    index = np.arange(count) + 0.5
    z = 1.0 - 2.0 * index / count
    angle = np.pi * (1.0 + 5.0**0.5) * index
    ring = np.sqrt(1.0 - z * z)
    return np.stack([ring * np.cos(angle), ring * np.sin(angle), z], axis=1)


def deck_light_table(
    volume: np.ndarray,
    spec: DensityField,
    optics: CloudOptics,
    radius: float,
    shift_table: ArrayLike,
    octaves: int = 4,
    octave_mean: float | None = None,
    cosines: int = 32,
    coverages: int = 17,
    columns: int = 2048,
    seed: int = 0,
) -> np.ndarray:
    """What the planet's deck shows on average where its pixels are too big
    for the clouds' self-shadowing (cloud.frag, ``shading`` 0): the mean of
    its own close-up lighting, by the sun's elevation and the weather's
    coverage.

    Close up, a deck pixel is lit by ``vis = exp2(-sunOpticalDepth)`` at
    mid-slab, its sun term scaled by the powder term as well. Averaged over
    columns, weighted by their alpha seen from overhead, those make the far
    deck the blur of the near one: a low sun marches along the slab through
    the neighbouring clouds, which darkens it gradually to the terminator,
    in the sunlight's own (reddened) colour.

    :param volume: the noise octave (quantised, to match the GPU)
    :param spec: the deck's resolved field (its shell's
        :meth:`sphere.ShellSpec.field_of`)
    :param optics: the deck's optics (its ``max_optical_depth`` caps the alpha)
    :param radius: sea-level radius, metres
    :param shift_table: coverage to fbm shift (:func:`weather.shift_table`)
    :param octaves: octaves the deck evaluates
    :param octave_mean: the noise's mean; None = *volume*'s
    :param cosines: sun cosines, evenly over [:data:`DECK_LIGHT_COS_MIN`, 1]
    :param coverages: weather coverages, evenly over [0, 1]
    :param columns: columns averaged, spread over the planet
    :param seed: the sun's azimuth at each column
    :returns: (coverages, cosines, 2) float32: the mean of ``vis * powder
        term`` (the sun term's) and of ``vis`` (the multiple-scattering
        fill's); 1 where there is no cloud
    """
    from world_builder.clouds.weather import coverage_to_shift

    _require_resolved(spec)
    if octave_mean is None:
        octave_mean = float(volume.mean())
    up = _fibonacci_dirs(columns)
    base, thickness = spec.slab
    reference = np.where(
        (np.abs(up[:, 2]) < 0.9)[:, None], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    )
    east = np.cross(reference, up)
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(up, east)
    azimuth = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, columns)
    level = np.cos(azimuth)[:, None] * east + np.sin(azimuth)[:, None] * north

    # The deck's sample: its density is the mean of two through the slab,
    # its sun march starts at mid-slab.
    heights = np.array([0.3, 0.7])
    layers = up[:, None, :] * (radius + base + heights * thickness)[None, :, None]
    coords = layers * np.asarray(spec.noise_scale) + np.asarray(spec.offset)
    view_raw = fbm(volume, coords, spec, octaves, octave_mean)
    mid = up * (radius + base + 0.5 * thickness)
    mu = np.linspace(DECK_LIGHT_COS_MIN, 1.0, cosines)
    sun = (
        mu[:, None, None] * up[None]
        + np.sqrt(1.0 - mu * mu)[:, None, None] * level[None]
    )
    if optics.sun_steps > 0:
        march = _sun_march_samples(volume, spec, optics, mid[None], sun, radius)

    table = np.ones((coverages, cosines, 2), dtype=np.float32)
    shifts = coverage_to_shift(
        np.linspace(0.0, 1.0, coverages), np.asarray(shift_table)
    )
    for row, shift in enumerate(shifts):
        density = _slab_density(spec, heights, view_raw + shift).mean(axis=1)
        alpha = 1.0 - np.exp2(
            -np.minimum(spec.density * density * thickness, optics.max_optical_depth)
        )
        # Plus one clear, unshadowed column: an almost clear row would
        # otherwise rest on its few wisps (the far deck shows next to none
        # of them, but filters between the rows).
        total = alpha.sum() + 1.0
        if total <= 1.0 + 1e-9:
            continue
        if optics.sun_steps > 0:
            vis = np.exp2(-_sun_depth(spec, optics, *march, shift, octave_mean))
        else:
            vis = np.ones((cosines, columns))
        powder = 1.0 - np.exp2(-spec.density * density * optics.powder_length * 2.0)
        powder_term = 1.0 + optics.powder_strength * (powder - 1.0)
        table[row, :, 0] = ((vis * (alpha * powder_term)).sum(axis=1) + 1.0) / total
        table[row, :, 1] = ((vis * alpha).sum(axis=1) + 1.0) / total
    return table


def build_deck_light_texture(table: np.ndarray) -> Texture:
    """A :func:`deck_light_table` as an RG32F 2D texture (x: the sun's
    cosine, y: the coverage), linearly filtered and clamped."""
    coverages, cosines, _ = table.shape
    tex = Texture("cloud_deck_light")
    tex.setup_2d_texture(cosines, coverages, Texture.T_float, Texture.F_rg32)
    tex.set_ram_image(np.ascontiguousarray(table, dtype=np.float32).tobytes())
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex
