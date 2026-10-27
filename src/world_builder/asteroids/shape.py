"""
A rock prototype's shape: its radius as a function of direction, in units of
the rock's bounding radius.

    radius(d) = scale * ellipsoid(d) * (1 + lumps + ridges + craters)

- **ellipsoid**: semi-axes (short, middle, 1), the long axis along body z
  (a tumbling rock's symmetry axis, see :mod:`motion`).
- **lumps**: fbm noise of the direction, the potato shape and its knobs.
- **ridges**: ridged noise, crests and grooves.
- **craters**: bowls with raised rims, of random sizes at random places.
- **scale**: makes the farthest point 1: the rock's ``radius`` in a field is
  its bounding radius. It is set from a fixed sampling of the full-detail
  shape, with a margin, so it does not depend on the meshes' resolution.

``min_wavelength`` band-limits the shape for a coarse sampling, as the
terrain does: noise octaves and craters smaller than it fade out, so a
coarse mesh is a smoothed rock, not an aliased one.

A radius function is star-shaped: no overhangs, arches or caves.

Where a rock's ice lies (:class:`IcePatches`) is a function of the direction
too, per prototype: a rock of ice fraction f is ice on the share f of its
surface where a smooth noise ranks highest.

Pure numpy (and the native noise when built), for worker processes.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import ArrayLike

from world_builder.asteroids.spec import ShapeSpec
from world_builder.terrain import noise
from world_builder.terrain.cubesphere import face_dirs

#: Octaves of the lumps and of the ridges.
LUMP_OCTAVES = 6
RIDGE_OCTAVES = 4
#: The ridges' first wavelength, over the lumps'.
RIDGE_SCALE = 0.45
#: A rim's height over its crater's diameter, and its width over the radius.
RIM_HEIGHT = 0.04
RIM_WIDTH = 0.35
#: Octaves of the ice patches' noise.
PATCH_OCTAVES = 3
#: Directions the patches' noise is ranked on (equal areas).
PATCH_SAMPLES = 8192
#: Half the rank over which ice turns to rock at a patch's edge.
PATCH_EDGE = 0.02
#: The sampling the bounding radius is measured on (per face side), and the
#: margin left over it for what that sampling misses.
SCALE_SAMPLES = 96
SCALE_MARGIN = 1.02


def face_grid_dirs(samples: int, centred: bool = False, border: int = 0) -> np.ndarray:
    """Directions of a regular grid on each cube face.

    :param samples: quads per face side (``centred``: texels per side)
    :param centred: at texel centres instead of grid nodes
    :param border: extra rows and columns past each edge (onto the extended
        face)
    :returns: (6, n, n, 3), n = samples + 1 (nodes) or samples (centred),
        plus 2 * border; [face, row (v), column (u)]
    """
    if centred:
        params = -1.0 + (2.0 * np.arange(-border, samples + border) + 1.0) / samples
    else:
        params = -1.0 + 2.0 * np.arange(-border, samples + 1 + border) / samples
    a, b = np.meshgrid(params, params)
    n = len(params)
    return np.stack([face_dirs(f, a, b).reshape(n, n, 3) for f in range(6)])


class PrototypeShape:
    """One prototype of a field: shape *index* of the field seeded *seed*.

    :ivar axes: the ellipsoid's semi-axes (x, y, z), z the longest (1)
    :ivar craters: (k, 4): each crater's direction and angular radius
    :ivar scale: makes the farthest point of the full-detail shape 1
    """

    def __init__(self, spec: ShapeSpec, seed: int, index: int):
        self.spec = spec
        self.seed = seed
        self.index = index
        rng = np.random.default_rng([seed & 0xFFFFFFFF, index, 0x5A9E])
        short = float(spec.elongation.sample(rng.random()))
        middle = short + (1.0 - short) * float(rng.random())
        self.axes = np.array([short, middle, 1.0])
        count = int(rng.poisson(spec.craters))
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        sizes = spec.crater_size.sample(rng.random(count))
        self.craters = np.column_stack((directions, sizes))
        self._noise_seed = (seed * 7919 + index * 104_729 + 17) & 0x7FFFFFFF
        self.scale = 1.0
        full = self.radius(face_grid_dirs(SCALE_SAMPLES).reshape(-1, 3))
        self.scale = 1.0 / (SCALE_MARGIN * float(full.max()))

    def radius(
        self, dirs: np.ndarray, min_wavelength: float | None = None
    ) -> np.ndarray:
        """:param dirs: (n, 3) unit directions, body frame
        :param min_wavelength: smaller detail is faded out (None: all of it)
        :returns: (n,) the radius in units of the bounding radius"""
        dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        s = self.spec
        q = dirs / self.axes
        ellipsoid = 1.0 / np.sqrt(np.sum(q * q, axis=1))
        relief = s.lumps * noise.fbm(
            dirs,
            s.lump_scale,
            LUMP_OCTAVES,
            self._noise_seed,
            min_wavelength=min_wavelength,
        )
        if s.ridges > 0.0:
            ridged = noise.ridged(
                dirs,
                s.lump_scale * RIDGE_SCALE,
                RIDGE_OCTAVES,
                self._noise_seed + 1,
                min_wavelength=min_wavelength,
            )
            relief += s.ridges * 2.0 * (ridged - 0.5)
        relief += self._craters(dirs, min_wavelength)
        radius = ellipsoid * np.maximum(1.0 + relief, 0.3)
        return self.scale * radius

    def _craters(self, dirs: np.ndarray, min_wavelength: float | None) -> np.ndarray:
        """The craters' relief, a share of the radius: a bowl of the spec's
        depth inside each, a rim around it."""
        total = np.zeros(len(dirs))
        depth = self.spec.crater_depth
        for cx, cy, cz, size in self.craters:
            # A crater shows on a mesh with about four samples across it:
            # whole down to a diameter of min_wavelength (four spacings),
            # gone at half that.
            weight = noise.octave_weight(
                1.0 / (2.0 * size),
                None if min_wavelength is None else 2.0 / min_wavelength,
            )
            if weight <= 0.0:
                continue
            cosine = np.clip(dirs @ np.array([cx, cy, cz]), -1.0, 1.0)
            x = np.arccos(cosine) / size
            near = x < 1.0 + 4.0 * RIM_WIDTH
            if not np.any(near):
                continue
            xs = x[near]
            # depth and rim are over the diameter, 2 * size (radians = radii)
            bowl = np.where(xs < 1.0, depth * 2.0 * size * (xs * xs - 1.0), 0.0)
            rim = RIM_HEIGHT * 2.0 * size * np.exp(-(((xs - 1.0) / RIM_WIDTH) ** 2))
            total[near] += weight * (bowl + rim)
        return total


@functools.lru_cache(maxsize=256)
def prototype_shape(spec: ShapeSpec, seed: int, index: int) -> PrototypeShape:
    """:returns: a cached :class:`PrototypeShape` (one per worker process)"""
    return PrototypeShape(spec, seed, index)


class IcePatches:
    """Where a prototype's ice lies: a smooth noise of the direction, ranked
    over the surface (0 to 1, uniform by area). A rock of ice fraction f is
    ice where the rank is above 1 - f: that share of its surface, in
    patches, a few of ice on a rocky body, a few of rock on an icy one. Its
    instances (the normal map's fourth channel), its terrain and its
    collision all read it.

    :ivar quantiles: (PATCH_SAMPLES,) the noise's values over the sphere,
        sorted: the rank's table
    """

    def __init__(self, seed: int, index: int, scale: float):
        self.scale = float(scale)
        self._seed = (seed * 6151 + index * 3079 + 41) & 0x7FFFFFFF
        self.quantiles = np.sort(self._noise(_fibonacci(PATCH_SAMPLES)))
        self._ranks = (np.arange(PATCH_SAMPLES) + 0.5) / PATCH_SAMPLES

    def _noise(self, dirs: np.ndarray) -> np.ndarray:
        return noise.fbm(dirs, self.scale, PATCH_OCTAVES, self._seed)

    def rank(self, dirs: ArrayLike) -> np.ndarray:
        """:param dirs: (n, 3) unit directions, body frame
        :returns: (n,) the share of the surface ranking lower, 0 to 1"""
        dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        return np.interp(self._noise(dirs), self.quantiles, self._ranks, 0.0, 1.0)


def ice_weight(rank: ArrayLike, ice: ArrayLike) -> np.ndarray:
    """:returns: how much of ice at surface ranks *rank* of a rock of ice
    fraction *ice*: 1 above 1 - ice, 0 below, a narrow ramp between
    (asteroid.frag's)"""
    rank = np.asarray(rank, dtype=np.float64)
    ice = np.asarray(ice, dtype=np.float64)
    t = np.clip((rank - (1.0 - ice - PATCH_EDGE)) / (2.0 * PATCH_EDGE), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    return np.where(ice <= 0.0, 0.0, np.where(ice >= 1.0, 1.0, w))


@functools.lru_cache(maxsize=256)
def ice_patches(seed: int, index: int, scale: float) -> IcePatches:
    """:returns: a cached :class:`IcePatches` (one per process)"""
    return IcePatches(seed, index, scale)


def _fibonacci(n: int) -> np.ndarray:
    """:returns: (n, 3) directions of equal areas (a Fibonacci sphere)"""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
