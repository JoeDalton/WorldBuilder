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

numpy pays ~150 operations per octave, so :func:`fbm` and :func:`ridged` sum
their octaves in a small C library (``native/noise.c``) when it is built:
about 5x faster, the same noise (sums differ from numpy's in the last bits).
The wheels ship it; ``invoke native`` builds it in a source checkout. Without
it the numpy code below runs. ``WORLD_BUILDER_NO_NATIVE_NOISE`` forces numpy,
``WORLD_BUILDER_REQUIRE_NATIVE`` makes a missing library an error.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

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


class _Plan:
    """The octaves one :func:`fbm` or :func:`ridged` call evaluates.

    :ivar octaves: (index, frequency, weight) per evaluated octave, the
        weight multiplying its signal in the sum
    :ivar norm: what the sum is divided by
    :ivar frames: (len(octaves), 12) for the native library: each octave's
        ``rotation.T * frequency`` (row-major), then its offset
    :ivar seeds: (len(octaves),) uint64, mixed
    :ivar weights: (len(octaves),) float64
    """

    def __init__(self, seed: int, octaves: list[tuple[int, float, float]], norm: float):
        self.octaves = octaves
        self.norm = norm
        self.frames = np.zeros((len(octaves), 12))
        for row, (index, frequency, _) in enumerate(octaves):
            rotation, offset = octave_frame(seed, index)
            self.frames[row, :9] = (rotation.T * frequency).ravel()
            self.frames[row, 9:] = offset
        self.seeds = np.array(
            [_mix_seed(seed + index) for index, _, _ in octaves], dtype=np.uint64
        )
        self.weights = np.array([weight for _, _, weight in octaves])


@functools.lru_cache(maxsize=1024)
def _fbm_plan(
    wavelength: float,
    octaves: int,
    seed: int,
    lacunarity: float,
    gain: float,
    min_wavelength: float | None,
) -> _Plan:
    max_frequency = None if min_wavelength is None else 1.0 / min_wavelength
    evaluated = []
    frequency = 1.0 / wavelength
    for index in range(octaves):
        weight = octave_weight(frequency, max_frequency)
        if weight <= 0.0:
            break
        evaluated.append((index, frequency, gain**index * weight))
        frequency *= lacunarity
    # The full amplitude sum, skipped octaves included, so every level shares
    # one normalisation.
    norm = sum(gain**index for index in range(octaves))
    return _Plan(seed, evaluated, norm)


@functools.lru_cache(maxsize=1024)
def _ridged_plan(
    wavelength: float,
    octaves: int,
    seed: int,
    lacunarity: float,
    persistence: float,
    min_wavelength: float | None,
) -> _Plan:
    max_frequency = None if min_wavelength is None else 1.0 / min_wavelength
    frequency = 1.0 / wavelength
    evaluated = [(0, frequency, 1.0)]
    amplitude, norm = 1.0, 1.0
    for index in range(1, octaves):
        frequency *= lacunarity
        amplitude *= persistence
        norm += amplitude
        # An octave of zero weight adds nothing, and leaves the signal feeding
        # the next one as it was.
        weight = octave_weight(frequency, max_frequency)
        if weight > 0.0:
            evaluated.append((index, frequency, amplitude * weight))
    return _Plan(seed, evaluated, norm)


# ── native library ────────────────────────────────────────────────────────────

#: Interface version of the native library (``WB_NOISE_ABI`` in noise.c).
_NATIVE_ABI = 9
_SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
#: Where the native noise library is built and shipped. Found by path: the
#: ``native`` package's own imports lead back here.
NATIVE_LIBRARY = Path(__file__).parent / "native" / f"libwb_noise{_SUFFIX}"


class _NativeNoise:
    """ctypes binding of ``native/noise.c``."""

    def __init__(self, lib: ctypes.CDLL):
        ptr, i64, i32, f64 = (
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_double,
        )
        lib.wb_simplex3.argtypes = [ptr, i64, ctypes.c_uint64, ptr]
        lib.wb_fbm.argtypes = [ptr, i64, i32, ptr, ptr, ptr, ptr]
        lib.wb_ridged.argtypes = [ptr, i64, i32, ptr, ptr, ptr, f64, f64, ptr]
        lib.wb_volume_fbm.argtypes = [
            ptr,
            i64,
            ptr,
            i64,
            i32,
            i32,
            ptr,
            ptr,
            ptr,
            f64,
            ptr,
        ]
        lib.wb_swell_heights.argtypes = [ptr, ptr, ptr, ptr, i64, ptr]
        lib.wb_swell_heights.restype = None
        for function in (lib.wb_simplex3, lib.wb_fbm, lib.wb_ridged, lib.wb_volume_fbm):
            function.restype = None
        # The LOD selection (lod.c; world_builder.terrain.quadtree).
        lib.wb_lod_new.argtypes = [f64, i32, i64, f64]
        lib.wb_lod_new.restype = ptr
        lib.wb_lod_free.argtypes = [ptr]
        lib.wb_lod_set_params.argtypes = [ptr, f64, f64, f64, f64, i32]
        lib.wb_lod_reset.argtypes = [ptr]
        lib.wb_lod_set_bounds.argtypes = [ptr, ptr, ptr, i64]
        lib.wb_lod_set_ready.argtypes = [ptr, ptr, i32, i64]
        lib.wb_lod_select.argtypes = [ptr, ptr, ptr]
        lib.wb_lod_result.argtypes = [ptr, i32, ctypes.POINTER(i64)]
        lib.wb_lod_result.restype = ctypes.POINTER(i32)
        # A chunk's geometry (chunk.c; world_builder.terrain.chunkgen).
        lib.wb_face_grid.argtypes = [i32, ptr, i64, ptr, i64, ptr]
        lib.wb_chunk_surface.argtypes = [ptr, ptr, i64, i32, i32, f64, ptr, ptr]
        lib.wb_chunk_pack.argtypes = [
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            f64,
            f64,
            i32,
            ptr,
            ptr,
            ptr,
        ]
        # The planet clouds' back-to-front order (billboards.c).
        lib.wb_billboard_order.argtypes = [ptr, i64, i64, ptr, ptr, ptr]
        lib.wb_billboard_vertices.argtypes = [ptr, i64, ptr, ptr, ptr]
        for function in (
            lib.wb_billboard_order,
            lib.wb_billboard_vertices,
            lib.wb_face_grid,
            lib.wb_chunk_surface,
            lib.wb_chunk_pack,
            lib.wb_lod_free,
            lib.wb_lod_set_params,
            lib.wb_lod_reset,
            lib.wb_lod_set_bounds,
            lib.wb_lod_set_ready,
            lib.wb_lod_select,
        ):
            function.restype = None
        self._lib = lib

    @property
    def lib(self) -> ctypes.CDLL:
        """The loaded library, its signatures set (for the other native
        modules: ``quadtree.NativeLodSelector``)"""
        return self._lib

    def simplex3(self, points: np.ndarray, seed: np.uint64) -> np.ndarray:
        points = np.ascontiguousarray(points, dtype=np.float64)
        out = np.empty(len(points))
        self._lib.wb_simplex3(
            points.ctypes.data, len(points), int(seed), out.ctypes.data
        )
        return out

    def fbm(self, points: np.ndarray, plan: _Plan) -> np.ndarray:
        """:returns: the plan's weighted sum of octaves, not normalised"""
        points = np.ascontiguousarray(points, dtype=np.float64)
        out = np.empty(len(points))
        self._lib.wb_fbm(
            points.ctypes.data,
            len(points),
            len(plan.weights),
            plan.frames.ctypes.data,
            plan.seeds.ctypes.data,
            plan.weights.ctypes.data,
            out.ctypes.data,
        )
        return out

    def ridged(
        self, points: np.ndarray, plan: _Plan, offset: float, gain: float
    ) -> np.ndarray:
        """:returns: the ridged sum over the plan's octaves, not normalised"""
        points = np.ascontiguousarray(points, dtype=np.float64)
        out = np.empty(len(points))
        self._lib.wb_ridged(
            points.ctypes.data,
            len(points),
            len(plan.weights),
            plan.frames.ctypes.data,
            plan.seeds.ctypes.data,
            plan.weights.ctypes.data,
            offset,
            gain,
            out.ctypes.data,
        )
        return out

    def volume_fbm(
        self,
        volume: np.ndarray,
        coords: np.ndarray,
        scales: np.ndarray,
        weights: np.ndarray,
        offsets: np.ndarray,
        evaluated: int,
        octave_mean: float,
    ) -> np.ndarray:
        """``clouds.noise.fbm``: see ``wb_volume_fbm`` in noise.c.

        :param volume: (n, n, n) float32, indexed [z, y, x]
        :param coords: (..., 3) noise coords
        :returns: (...) float64
        """
        volume = np.ascontiguousarray(volume, dtype=np.float32)
        coords = np.asarray(coords, dtype=np.float64)
        shape = coords.shape[:-1]
        flat = np.ascontiguousarray(coords.reshape(-1, 3))
        scales = np.ascontiguousarray(scales, dtype=np.float64)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        offsets = np.ascontiguousarray(offsets, dtype=np.float64)
        out = np.empty(len(flat))
        self._lib.wb_volume_fbm(
            volume.ctypes.data,
            volume.shape[0],
            flat.ctypes.data,
            len(flat),
            len(weights),
            evaluated,
            scales.ctypes.data,
            weights.ctypes.data,
            offsets.ctypes.data,
            octave_mean,
            out.ctypes.data,
        )
        return out.reshape(shape)

    def billboard_order(
        self,
        particles: np.ndarray,
        offset: np.ndarray,
        order: np.ndarray,
        index: np.ndarray,
    ) -> None:
        """A cloud tile's billboards farthest first (``wb_billboard_order``).

        :param particles: (n, k) float32 rows, xyz relative to the tile's
            centre first
        :param offset: (3,) the tile's centre minus the camera
        :param order: (n,) uint32 the previous order, updated in place
            (stable: equal distances keep it)
        :param index: (6 n,) uint32, filled with the triangle indices
        """
        assert particles.dtype == np.float32 and particles.flags.c_contiguous
        assert order.dtype == np.uint32 and index.dtype == np.uint32
        offset = np.ascontiguousarray(offset, dtype=np.float64)
        self._lib.wb_billboard_order(
            particles.ctypes.data,
            particles.shape[1],
            len(particles),
            offset.ctypes.data,
            order.ctypes.data,
            index.ctypes.data,
        )

    def billboard_vertices(
        self, particles: np.ndarray, out: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """A cloud tile's vertices (``wb_billboard_vertices``).

        :param particles: (n, 9) float32, ``CloudTile.particles``
        :param out: (4 n * 11,) float32, filled (four corners a billboard)
        :returns: the tile's box (min xyz, max xyz; float32) and its reach
        """
        particles = np.ascontiguousarray(particles, dtype=np.float32)
        assert particles.shape[1] == 9 and out.dtype == np.float32
        assert out.size == 4 * 11 * len(particles)
        box = np.empty(6, dtype=np.float32)
        reach = ctypes.c_double()
        self._lib.wb_billboard_vertices(
            particles.ctypes.data,
            len(particles),
            out.ctypes.data,
            box.ctypes.data,
            ctypes.byref(reach),
        )
        return box, reach.value

    def face_grid(self, face: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``cubesphere.face_dirs(face, *np.meshgrid(a, b))``: (len(b) *
        len(a), 3) unit vectors, rows along *b*"""
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        out = np.empty((len(b) * len(a), 3))
        self._lib.wb_face_grid(
            int(face), a.ctypes.data, len(a), b.ctypes.data, len(b), out.ctypes.data
        )
        return out

    def chunk_surface(
        self,
        dirs: np.ndarray,
        elevation: np.ndarray,
        border: int,
        step: int,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """A chunk's normals and slope (``wb_chunk_surface`` in chunk.c).

        :param dirs: (n, n, 3) the sampled grid's directions
        :param elevation: (n, n) its heights, metres
        :returns: the inner grid's normals (VERTS^2, 3) and slope in degrees
            (VERTS^2,), from the normals at +-*step* samples
        """
        dirs = np.ascontiguousarray(dirs, dtype=np.float64)
        elevation = np.ascontiguousarray(elevation, dtype=np.float64)
        count = (elevation.shape[0] - 2 * border) ** 2
        normals = np.empty((count, 3))
        slope = np.empty(count)
        self._lib.wb_chunk_surface(
            dirs.ctypes.data,
            elevation.ctypes.data,
            elevation.shape[0],
            int(border),
            int(step),
            float(radius),
            normals.ctypes.data,
            slope.ctypes.data,
        )
        return normals, slope

    def chunk_pack(
        self,
        out: np.ndarray,
        dirs: np.ndarray,
        elevation: np.ndarray,
        normals: np.ndarray,
        weights: np.ndarray,
        centre: np.ndarray,
        radius: float,
        skirt_depth: float,
        sea: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fill a chunk's vertex buffer (``wb_chunk_pack`` in chunk.c).

        :param out: the buffer, ``chunkgen.VERTEX_DTYPE``, grid then skirt
        :param dirs: (VERTS^2, 3) the inner grid's directions
        :param elevation: (VERTS^2,) its heights
        :param normals: (VERTS^2, 3)
        :param weights: (VERTS^2, 6) uint8 biome weights
        :param sea: the box holds the water the shader lifts the seabed to
        :returns: the drawn vertices' box (min, max), centre-relative float32
        """
        assert out.dtype.itemsize == 36 and out.flags.c_contiguous
        dirs = np.ascontiguousarray(dirs, dtype=np.float64)
        elevation = np.ascontiguousarray(elevation, dtype=np.float64)
        normals = np.ascontiguousarray(normals, dtype=np.float64)
        weights = np.ascontiguousarray(weights, dtype=np.uint8)
        centre = np.ascontiguousarray(centre, dtype=np.float64)
        box_min = np.empty(3, dtype=np.float32)
        box_max = np.empty(3, dtype=np.float32)
        self._lib.wb_chunk_pack(
            dirs.ctypes.data,
            elevation.ctypes.data,
            normals.ctypes.data,
            weights.ctypes.data,
            centre.ctypes.data,
            float(radius),
            float(skirt_depth),
            int(sea),
            out.ctypes.data,
            box_min.ctypes.data,
            box_max.ctypes.data,
        )
        return box_min, box_max

    def swell_heights(
        self, swell: Any, points: np.ndarray, depths: np.ndarray, max_freqs: np.ndarray
    ) -> np.ndarray:
        """``wb_swell_heights`` (world_builder.ocean.swell): the water's
        height at (n, 3) sea-level points, metres

        :param swell: a ``WbSwell`` ctypes structure
        """
        points = np.ascontiguousarray(points, dtype=np.float64)
        depths = np.ascontiguousarray(depths, dtype=np.float64)
        max_freqs = np.ascontiguousarray(max_freqs, dtype=np.float64)
        out = np.empty(len(points))
        self._lib.wb_swell_heights(
            ctypes.addressof(swell),
            points.ctypes.data,
            depths.ctypes.data,
            max_freqs.ctypes.data,
            len(points),
            out.ctypes.data,
        )
        return out


def native_library() -> _NativeNoise | None:
    """:returns: the native noise library's binding, or None (numpy runs);
    for the other noise modules (``clouds.noise``)"""
    return _native()


@functools.cache
def _native() -> _NativeNoise | None:
    """:returns: the native noise library, or None to use numpy"""
    if os.environ.get("WORLD_BUILDER_NO_NATIVE_NOISE"):
        return None
    if not NATIVE_LIBRARY.exists():
        problem = f"{NATIVE_LIBRARY.name} is not built"
    else:
        try:
            lib = ctypes.CDLL(str(NATIVE_LIBRARY))
        except OSError as error:
            problem = f"cannot load {NATIVE_LIBRARY.name}: {error}"
        else:
            if lib.wb_noise_abi() == _NATIVE_ABI:
                return _NativeNoise(lib)
            problem = f"{NATIVE_LIBRARY.name} is out of date"
    message = (
        f"{problem}: noise runs in numpy, terrain generation is about 3x "
        "slower. Build it with `invoke native` (docs/source/native_build.md)."
    )
    if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
        raise RuntimeError(message)
    LOGGER.warning(message)
    return None


def native_noise() -> bool:
    """:returns: whether the native noise library is in use"""
    return _native() is not None


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
    plan = _fbm_plan(wavelength, octaves, seed, lacunarity, gain, min_wavelength)
    native = _native()
    if native is not None:
        return native.fbm(points, plan) / plan.norm
    total = np.zeros(len(points))
    for index, frequency, weight in plan.octaves:
        total += weight * _octave(points, frequency, seed, index)
    return total / plan.norm


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
    plan = _ridged_plan(
        wavelength, octaves, seed, lacunarity, persistence, min_wavelength
    )
    native = _native()
    if native is not None:
        return native.ridged(points, plan, offset, gain) / plan.norm
    (_, frequency, _), *rest = plan.octaves
    signal = offset - np.abs(_octave(points, frequency, seed, 0))
    signal *= signal
    total = signal.copy()
    for index, frequency, weight in rest:
        feed = np.clip(signal * gain, 0.0, 1.0)
        signal = offset - np.abs(_octave(points, frequency, seed, index))
        signal *= signal
        signal *= feed
        total += weight * signal
    return total / plan.norm


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
