"""
Deterministic, vectorised 3D simplex noise over arbitrary float64 points.

Everything here is pure numpy (no Panda) so worker processes and tests can use
it. The lattice hash is a 64-bit integer mix of the cell coordinates and the
seed rather than a 256-entry permutation table, which would repeat every 256
cells (every kilometre for a 4 m octave): the planet is sampled over millions
of cells per octave and must never visibly tile.

Each octave of :func:`fbm` / :func:`ridged` is sampled in its own randomly
rotated and offset frame, so octaves do not share lattice axes (which shows as
grid-aligned ridges) and the planet's poles are not special.
"""

from __future__ import annotations

import functools

import numpy as np

# Skew / unskew factors of the 3D simplex lattice.
_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0
# Kernel radius squared. 0.5 is the largest value that keeps every corner's
# contribution inside its own simplex neighbourhood, so the noise is continuous
# (Gustavson's 0.6 leaves small discontinuities).
_R2 = 0.5
# Brings the output to about [-1, 1]; measured over 4M points (max |n| ~ 0.99).
_SCALE = 76.0

# The 12 cube-edge gradients, padded to 16 so a 4-bit index picks one uniformly
# enough without a modulo.
_GRADIENTS = np.array(
    [
        (1, 1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (-1, 0, 1),
        (1, 0, -1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, -1, 1),
        (0, 1, -1),
        (0, -1, -1),
        (1, 1, 0),
        (-1, 1, 0),
        (0, -1, 1),
        (0, -1, -1),
    ],
    dtype=np.float64,
)

_PRIME_I = np.uint64(0x9E3779B185EBCA87)
_PRIME_J = np.uint64(0xC2B2AE3D27D4EB4F)
_PRIME_K = np.uint64(0x165667B19E3779F9)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)


def _hash(i: np.ndarray, j: np.ndarray, k: np.ndarray, seed: np.uint64) -> np.ndarray:
    """Mix integer lattice coordinates and a seed into well-spread 64-bit values.

    uint64 numpy arithmetic wraps silently, which is exactly what a hash wants.

    :param i: int64 lattice x
    :param j: int64 lattice y
    :param k: int64 lattice z
    :param seed: the noise seed, already mixed into a uint64
    :returns: uint64 hash per point
    """
    h = (i.view(np.uint64) * _PRIME_I) ^ (j.view(np.uint64) * _PRIME_J)
    h ^= k.view(np.uint64) * _PRIME_K
    h ^= seed
    # splitmix64 finaliser
    h ^= h >> np.uint64(30)
    h *= _MIX_1
    h ^= h >> np.uint64(27)
    h *= _MIX_2
    h ^= h >> np.uint64(31)
    return h


def _corner(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    seed: np.uint64,
) -> np.ndarray:
    """One simplex corner's contribution.

    :param x: offset from the corner, x
    :param y: offset from the corner, y
    :param z: offset from the corner, z
    :param i: corner lattice x
    :param j: corner lattice y
    :param k: corner lattice z
    :param seed: mixed seed
    :returns: the corner's kernel-weighted gradient dot product
    """
    t = _R2 - (x * x + y * y + z * z)
    np.maximum(t, 0.0, out=t)
    t *= t
    t *= t
    # take() is ~4x faster than fancy indexing for this gather.
    g = _GRADIENTS.take((_hash(i, j, k, seed) >> np.uint64(60)).astype(np.intp), axis=0)
    return t * (g[:, 0] * x + g[:, 1] * y + g[:, 2] * z)


@functools.cache
def _mix_seed(seed: int) -> np.uint64:
    """:returns: *seed* spread over 64 bits, so seeds 1 and 2 are unrelated"""
    s = np.array([seed], dtype=np.int64)
    return _hash(s, s, s, np.uint64(0x5EED))[0]


def simplex3(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """3D simplex noise.

    :param points: (N, 3) float64 sample positions, in lattice units
    :param seed: integer seed; each seed is an independent field
    :returns: (N,) float64 in about [-1, 1], continuous, zero mean
    """
    s = _mix_seed(seed)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    skew = (x + y + z) * _F3
    fi = np.floor(x + skew)
    fj = np.floor(y + skew)
    fk = np.floor(z + skew)
    unskew = (fi + fj + fk) * _G3
    x0 = x - fi + unskew
    y0 = y - fj + unskew
    z0 = z - fk + unskew
    i = fi.astype(np.int64)
    j = fj.astype(np.int64)
    k = fk.astype(np.int64)

    # Rank of each offset among the three picks the simplex: the largest axis
    # is stepped first, then the middle one.
    xy = x0 >= y0
    xz = x0 >= z0
    yz = y0 >= z0
    rank_x = xy.astype(np.int64) + xz
    rank_y = (~xy).astype(np.int64) + yz
    rank_z = (~xz).astype(np.int64) + ~yz
    # Each step as int (lattice) and float (offset): mixing bool arrays into
    # float arithmetic converts them on every use.
    i1, j1, k1 = (r >= 2 for r in (rank_x, rank_y, rank_z))
    i2, j2, k2 = (r >= 1 for r in (rank_x, rank_y, rank_z))
    f1 = [b.astype(np.float64) for b in (i1, j1, k1)]
    f2 = [b.astype(np.float64) for b in (i2, j2, k2)]

    n = _corner(x0, y0, z0, i, j, k, s)
    n += _corner(
        x0 - f1[0] + _G3,
        y0 - f1[1] + _G3,
        z0 - f1[2] + _G3,
        i + i1,
        j + j1,
        k + k1,
        s,
    )
    n += _corner(
        x0 - f2[0] + 2.0 * _G3,
        y0 - f2[1] + 2.0 * _G3,
        z0 - f2[2] + 2.0 * _G3,
        i + i2,
        j + j2,
        k + k2,
        s,
    )
    n += _corner(
        x0 - (1.0 - 3.0 * _G3),
        y0 - (1.0 - 3.0 * _G3),
        z0 - (1.0 - 3.0 * _G3),
        i + 1,
        j + 1,
        k + 1,
        s,
    )
    return n * _SCALE


@functools.lru_cache(maxsize=1024)
def octave_frame(seed: int, index: int) -> tuple[np.ndarray, np.ndarray]:
    """A random rotation and offset for one octave.

    :param seed: field seed
    :param index: octave index
    :returns: (3x3 rotation, (3,) offset in lattice units)
    """
    rng = np.random.default_rng([seed & 0xFFFFFFFF, index, 0x0C7A])
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))  # a uniform rotation, not just any orthogonal
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return q, rng.uniform(-1000.0, 1000.0, size=3)


def octave_weight(frequency: float, max_frequency: float | None) -> float:
    """Band-limit weight: 1 below half the cutoff, fading to 0 at the cutoff.

    A smooth fade instead of a hard cut keeps an octave from popping in when a
    chunk switches level.

    :param frequency: the octave's frequency, in cycles per metre
    :param max_frequency: the cutoff, or None for no limit
    :returns: weight in [0, 1]
    """
    if max_frequency is None:
        return 1.0
    t = (2.0 * max_frequency - 2.0 * frequency) / max_frequency
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _octave(points: np.ndarray, frequency: float, seed: int, index: int) -> np.ndarray:
    """:returns: simplex noise of one octave, in its own rotated frame"""
    rotation, offset = octave_frame(seed, index)
    return simplex3(points @ (rotation.T * frequency) + offset, seed + index)


def fbm(
    points: np.ndarray,
    wavelength: float,
    octaves: int,
    seed: int,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    min_wavelength: float | None = None,
) -> np.ndarray:
    """Fractional Brownian motion: a sum of octaves at rising frequency.

    :param points: (N, 3) positions in metres
    :param wavelength: wavelength of the first octave, in metres
    :param octaves: number of octaves before band limiting
    :param seed: field seed
    :param lacunarity: frequency ratio between octaves
    :param gain: amplitude ratio between octaves
    :param min_wavelength: octaves shorter than this are dropped (faded out),
        so coarse samples do not alias detail they cannot represent
    :returns: (N,) in about [-1, 1] (normalised by the full amplitude sum, so
        band limiting only removes detail and never rescales the field)
    """
    max_frequency = None if min_wavelength is None else 1.0 / min_wavelength
    total = np.zeros(len(points))
    frequency = 1.0 / wavelength
    for index in range(octaves):
        weight = octave_weight(frequency, max_frequency)
        if weight <= 0.0:
            break
        total += (gain**index * weight) * _octave(points, frequency, seed, index)
        frequency *= lacunarity
    # The full amplitude sum, skipped octaves included, so every level shares
    # one normalisation.
    norm = sum(gain**index for index in range(octaves))
    return total / norm


def ridged(
    points: np.ndarray,
    wavelength: float,
    octaves: int,
    seed: int,
    lacunarity: float = 2.0,
    gain: float = 2.0,
    offset: float = 1.0,
    persistence: float = 0.5,
    min_wavelength: float | None = None,
) -> np.ndarray:
    """Musgrave's ridged multifractal: sharp crests, smooth valleys.

    Each octave is weighted by the previous one's signal, so detail gathers on
    the ridges and the valleys stay smooth: a cheap stand-in for erosion.

    :param points: (N, 3) positions in metres
    :param wavelength: wavelength of the first octave, in metres
    :param octaves: number of octaves before band limiting
    :param seed: field seed
    :param lacunarity: frequency ratio between octaves
    :param gain: how strongly a ridge feeds detail into the next octave
    :param offset: ridge sharpness; 1 puts crests at the noise's zero crossings
    :param persistence: amplitude ratio between octaves
    :param min_wavelength: octaves shorter than this are faded out
    :returns: (N,) in about [0, 1]
    """
    max_frequency = None if min_wavelength is None else 1.0 / min_wavelength
    frequency = 1.0 / wavelength
    signal = offset - np.abs(_octave(points, frequency, seed, 0))
    signal *= signal
    total = signal.copy()
    amplitude, norm = 1.0, 1.0
    for index in range(1, octaves):
        frequency *= lacunarity
        amplitude *= persistence
        norm += amplitude
        weight = octave_weight(frequency, max_frequency)
        if weight <= 0.0:
            continue
        feed = np.clip(signal * gain, 0.0, 1.0)
        signal = offset - np.abs(_octave(points, frequency, seed, index))
        signal *= signal
        signal *= feed
        total += (amplitude * weight) * signal
    return total / norm


def warp(
    points: np.ndarray,
    wavelength: float,
    amplitude: float,
    octaves: int,
    seed: int,
) -> np.ndarray:
    """Domain warp: displace points by a smooth random vector field.

    Feeding warped points to the other fields bends their straight-ish
    features into the meanders real coastlines and ranges have.

    :param points: (N, 3) positions in metres
    :param wavelength: wavelength of the displacement field, in metres
    :param amplitude: displacement magnitude, in metres
    :param octaves: octaves of the displacement field
    :param seed: field seed
    :returns: (N, 3) displaced positions
    """
    displacement = np.stack(
        [fbm(points, wavelength, octaves, seed + 7919 * axis) for axis in range(3)],
        axis=1,
    )
    return points + amplitude * displacement


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """GLSL smoothstep: 0 below edge0, 1 above edge1, cubic in between.

    Works for edge0 > edge1 too (a falling step).
    """
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
