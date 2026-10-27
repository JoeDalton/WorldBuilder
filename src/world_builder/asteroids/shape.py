"""
A rock prototype's shape: its radius as a function of direction, in units of
the rock's bounding radius.

    radius(d) = scale * body(d) * (1 + lumps + ridges + craters)

- **body**, by the prototype's family (:meth:`ShapeSpec.family`):
  - round: an ellipsoid, semi-axes (short, middle, 1), the long axis along
    body z (a tumbling rock's symmetry axis, see :mod:`motion`), with big
    bulges (low-frequency noise);
  - binary: two ellipsoidal lobes along z, the origin in both where they
    meet, joined by a smooth maximum (a waist, not a crease), each with
    its own lumps;
  - bent: an elongated ellipsoid seen through bent and warped directions
    (its ends curve aside, it twists);
  - fragment: an ellipsoid clipped by a few planes, joined by a smooth
    minimum (flat faces, rounded edges), with half the lumps and craters.
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
import math

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
#: The round family's bulges: octaves and wavelength (radii).
BULGE_OCTAVES = 2
BULGE_SCALE = 2.0
#: A binary's waist: the smooth maximum's softness (radii).
WAIST = 0.06
#: The bent family's warp: octaves, and wavelength over the lumps'.
WARP_OCTAVES = 2
WARP_SCALE = 1.5
#: A fragment's edges: the smooth minimum's softness (radii).
EDGE = 0.03
#: The largest angle between a binary's lobes' axes, radians.
LOBE_TILT = 0.7
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
        self.family = spec.family(index)
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
        # The family's own draws: a stream of their own, so a family's
        # parameters never move the others'
        family = np.random.default_rng([seed & 0xFFFFFFFF, index, 0xFA41])
        if self.family == "binary":
            self._lobes = self._make_lobes(family)
        elif self.family == "bent":
            short = float(spec.crooked_elongation.sample(family.random()))
            middle = short + (1.0 - short) * float(family.random())
            self.axes = np.array([short, middle, 1.0])
            self._bend = float(spec.bend.sample(family.random()))
        elif self.family == "fragment":
            self._planes = self._make_planes(family)
            self.craters = self.craters[: len(self.craters) // 2]
        self.scale = 1.0
        full = self.radius(face_grid_dirs(SCALE_SAMPLES).reshape(-1, 3))
        self.scale = 1.0 / (SCALE_MARGIN * float(full.max()))

    # ── the families' bodies ──────────────────────────────────────────────

    def _make_lobes(self, rng: np.random.Generator) -> list[tuple]:
        """Two lobes, (centre, rotation, semi-axes) each: the big one along
        -z, the small one along an axis tilted from +z (up to
        :data:`LOBE_TILT`), the origin on both axes where they overlap."""
        s = self.spec
        ratio = float(s.lobe_ratio.sample(rng.random()))
        overlap = float(s.lobe_overlap.sample(rng.random()))
        big = self.axes * 1.0
        short = float(s.elongation.sample(rng.random()))
        small = np.array([short, short + (1.0 - short) * rng.random(), 1.0]) * ratio
        reach = overlap * small[2]
        lobes = [(np.array([0.0, 0.0, -(big[2] - reach)]), np.eye(3), big)]
        bearing = 2.0 * math.pi * float(rng.random())
        hinge = np.array([math.cos(bearing), math.sin(bearing), 0.0])
        turn = _rotation(hinge, LOBE_TILT * float(rng.random()))
        centre = turn @ np.array([0.0, 0.0, small[2] - reach])
        lobes.append((centre, turn, small))
        return lobes

    def _make_planes(self, rng: np.random.Generator) -> np.ndarray:
        """(k, 4): each clipping plane's unit normal and its distance from
        the centre, a share of the ellipsoid's radius that way."""
        s = self.spec
        count = int(s.facets.sample(rng.random()))
        normals = rng.normal(size=(count, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        depth = s.facet_depth.sample(rng.random(count))
        distance = depth * _ellipsoid(normals, self.axes)
        return np.column_stack((normals, distance))

    def _lumps(self, points, seed, min_wavelength, amplitude=1.0) -> np.ndarray:
        """The lumps and ridges at *points* of the body's surface (not at
        directions: on the long sides of an elongated rock, or a fragment's
        faces, a direction's noise would stretch into streaks)."""
        dirs = points
        s = self.spec
        relief = (
            amplitude
            * s.lumps
            * noise.fbm(
                dirs, s.lump_scale, LUMP_OCTAVES, seed, min_wavelength=min_wavelength
            )
        )
        if s.ridges > 0.0:
            ridged = noise.ridged(
                dirs,
                s.lump_scale * RIDGE_SCALE,
                RIDGE_OCTAVES,
                seed + 1,
                min_wavelength=min_wavelength,
            )
            relief = relief + amplitude * s.ridges * 2.0 * (ridged - 0.5)
        return relief

    def _body(self, dirs: np.ndarray, min_wavelength: float | None):
        """:returns: the family's body radius and its relief (lumps and
        ridges), before the craters"""
        s = self.spec
        seed = self._noise_seed
        if self.family == "binary":
            (ca, ra, sa), (cb, rb, sb) = self._lobes
            a = _lobe(dirs, ca, ra, sa)
            b = _lobe(dirs, cb, rb, sb)
            body = _smooth_max(a, b, WAIST)
            share = 1.0 / (1.0 + np.exp(-(a - b) / WAIST))  # the big lobe's
            points = dirs * body[:, None]
            relief = self._lumps(points, seed, min_wavelength)
            # the small lobe's own lumps, where it shows
            small = share < 0.999
            if np.any(small):
                own = self._lumps(points[small], seed + 7, min_wavelength)
                relief[small] = (
                    share[small] * relief[small] + (1.0 - share[small]) * own
                )
            return body, relief
        if self.family == "bent":
            bent = dirs - self._bend * (dirs[:, 2:3] ** 2 - 1.0 / 3.0) * [1.0, 0.0, 0.0]
            bent /= np.linalg.norm(bent, axis=1)[:, None]
            scale = s.lump_scale * WARP_SCALE
            twist = np.column_stack(
                [
                    noise.fbm(bent, scale, WARP_OCTAVES, seed + 11 + k,
                              min_wavelength=min_wavelength)
                    for k in range(3)
                ]
            )  # fmt: skip
            # faded out toward the tips, where the radius turns fastest
            taper = (1.0 - bent[:, 2] ** 2)[:, None]
            warped = bent + s.warp * taper * twist
            warped /= np.linalg.norm(warped, axis=1)[:, None]
            body = _ellipsoid(warped, self.axes)
            return body, self._lumps(dirs * body[:, None], seed, min_wavelength)
        if self.family == "fragment":
            radii = [_ellipsoid(dirs, self.axes)]
            for nx, ny, nz, distance in self._planes:
                facing = dirs @ np.array([nx, ny, nz])
                radii.append(
                    np.where(facing > 1e-9, distance / np.maximum(facing, 1e-9), 1e9)
                )
            body = _smooth_min(np.stack(radii), EDGE)
            points = dirs * body[:, None]
            return body, self._lumps(points, seed, min_wavelength, amplitude=0.5)
        body = _ellipsoid(dirs, self.axes)
        points = dirs * body[:, None]
        relief = self._lumps(points, seed, min_wavelength)
        if s.bulges > 0.0:
            relief = relief + s.bulges * noise.fbm(
                points,
                BULGE_SCALE,
                BULGE_OCTAVES,
                seed + 3,
                min_wavelength=min_wavelength,
            )
        return body, relief

    def radius(
        self, dirs: np.ndarray, min_wavelength: float | None = None
    ) -> np.ndarray:
        """:param dirs: (n, 3) unit directions, body frame
        :param min_wavelength: smaller detail is faded out (None: all of it)
        :returns: (n,) the radius in units of the bounding radius"""
        dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        body, relief = self._body(dirs, min_wavelength)
        relief = relief + self._craters(dirs, min_wavelength)
        radius = body * np.maximum(1.0 + relief, 0.3)
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


def _ellipsoid(dirs: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """:returns: an ellipsoid's radius along *dirs*, centred on the origin"""
    q = dirs / axes
    return 1.0 / np.sqrt(np.sum(q * q, axis=1))


def _lobe(dirs: np.ndarray, centre, turn, axes) -> np.ndarray:
    """:returns: how far along *dirs* the surface of an ellipsoid (offset by
    *centre*, turned by *turn*) is: the far root, the origin being inside"""
    d = (dirs @ turn) / axes
    c = (turn.T @ centre) / axes
    a = np.sum(d * d, axis=1)
    b = -(d @ c)
    cc = float(c @ c) - 1.0
    return (-b + np.sqrt(np.maximum(b * b - a * cc, 0.0))) / a


def _smooth_max(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(a, b) + k * np.log1p(np.exp(-np.abs(a - b) / k))


def _smooth_min(values: np.ndarray, k: float) -> np.ndarray:
    """:returns: a smooth minimum over the first axis (log-sum-exp)"""
    low = values.min(axis=0)
    return low - k * np.log(np.sum(np.exp(-(values - low) / k), axis=0))


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """:returns: the 3x3 rotation by *angle* about *axis*"""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


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
