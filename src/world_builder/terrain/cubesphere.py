"""
The cube-sphere: how the planet is cut into chunks.

The sphere is the six faces of a cube pushed out onto it. Each face is a
quadtree of square chunks; a chunk key is ``(level, face, i, j)``, level 0
being the finest and :attr:`CubeGrid.max_level` the whole face.

The cube is equal-angle (tangent-warped): face parameter ``a`` in [-1, 1] maps
to the cube coordinate ``tan(pi a / 4)``, so cells span equal angles along the
face's centre lines and grid lines are great circles. Cell sizes still vary
from 0.71x to 1x the nominal spacing (smaller toward the corners).

Sample positions are integer *level-0 indices* k in [0, n0] per face axis,
``a = (2k - n0) / n0``, exact in float64 (n0 is a power of two). A chunk at
level L samples every 2**L indices, so neighbouring chunks, of any level,
share their edge samples exactly. Across a face seam the two faces compute
bit-identical directions for their shared edge (see :func:`face_dirs`).

Face frames are right-handed (u x v = outward normal), so a grid wound
counter-clockwise in (u, v) is counter-clockwise seen from outside. On the four
equatorial faces u points east and v north.

Pure numpy, so it runs in worker processes.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

#: Quads per chunk side; vertices per side is one more.
QUADS = 64
VERTS = QUADS + 1
#: Aimed-for vertex spacing of the finest level, metres.
TARGET_SPACING = 2.0
#: Texture coordinates repeat with this period, metres (materials.UV_PERIOD).
UV_PERIOD = 4096.0

#: (outward normal, u, v) per face: +X, -X, +Y, -Y, +Z, -Z.
FACE_FRAMES = np.array(
    [
        [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(-1, 0, 0), (0, -1, 0), (0, 0, 1)],
        [(0, 1, 0), (-1, 0, 0), (0, 0, 1)],
        [(0, -1, 0), (1, 0, 0), (0, 0, 1)],
        [(0, 0, 1), (0, 1, 0), (-1, 0, 0)],
        [(0, 0, -1), (0, 1, 0), (1, 0, 0)],
    ],
    dtype=np.float64,
)
FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

Key = tuple[int, int, int, int]


def _warp(a: np.ndarray) -> np.ndarray:
    """Face parameter to cube coordinate, tan(pi a / 4).

    Odd by construction and exactly +-1 at the face edges, so the two faces
    of a seam compute identical coordinates.
    """
    a = np.asarray(a, dtype=np.float64)
    x = np.copysign(np.tan((math.pi / 4.0) * np.abs(a)), a)
    return np.where(np.abs(a) == 1.0, np.sign(a), x)


def face_dirs(face: int, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Directions of face points.

    Accepts parameters beyond [-1, 1] (the face extended past its edges, for
    border samples), up to |a| < 2.

    :param face: 0-5, see :data:`FACE_FRAMES`
    :param a: face u parameter, any shape
    :param b: face v parameter, same shape
    :returns: (N, 3) unit vectors, N = a.size
    """
    x = np.ravel(_warp(a))
    y = np.ravel(_warp(b))
    n, u, v = FACE_FRAMES[face]
    p = n + x[:, None] * u + y[:, None] * v
    # Written as 1 + (x^2 + y^2): addition commutes, so whichever of x, y is
    # the +-1 of a seam, both faces round identically.
    return p / np.sqrt(1.0 + (x * x + y * y))[:, None]


def dirs_to_face(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The face each direction lies on, and its face parameters.

    Points on a seam go to the first face in :data:`FACE_FRAMES` order.

    :param dirs: (N, 3) unit vectors (or any nonzero vectors)
    :returns: face (N,) int, a (N,), b (N,) in [-1, 1]
    """
    dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    axis = np.argmax(np.abs(dirs), axis=1)
    rows = np.arange(len(dirs))
    negative = dirs[rows, axis] < 0.0
    face = 2 * axis + negative
    frames = FACE_FRAMES[face]
    along_n = np.einsum("ij,ij->i", dirs, frames[:, 0])
    along_u = np.einsum("ij,ij->i", dirs, frames[:, 1])
    along_v = np.einsum("ij,ij->i", dirs, frames[:, 2])
    a = np.arctan(along_u / along_n) * (4.0 / math.pi)
    b = np.arctan(along_v / along_n) * (4.0 / math.pi)
    return face, np.clip(a, -1.0, 1.0), np.clip(b, -1.0, 1.0)


@dataclass(frozen=True)
class KeyBox:
    """Geometry of one chunk's patch, for LOD distances and culling.

    The box frame is the patch's tangent frame at its centre direction:
    ``t_u``, ``t_v`` along the face axes, ``up`` radial. Extents are per unit
    radius, so the box for any height range is cheap (:meth:`extents`).

    :ivar up: centre direction
    :ivar t_u: unit tangent along the face's u
    :ivar t_v: unit tangent along the face's v (orthogonalised)
    :ivar u_range: (min, max) of dot(dir, t_u) over the patch
    :ivar v_range: (min, max) of dot(dir, t_v) over the patch
    :ivar cos_radius: cosine of the patch's angular radius (centre to corner)
    :ivar sin_radius: its sine
    """

    up: tuple[float, float, float]
    t_u: tuple[float, float, float]
    t_v: tuple[float, float, float]
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    cos_radius: float
    sin_radius: float

    def extents(
        self, r_min: float, r_max: float
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Box extents along (t_u, t_v, up) for radii [r_min, r_max].

        :returns: ((u_lo, u_hi), (v_lo, v_hi), (w_lo, w_hi)), metres from the
            planet centre along each axis
        """
        u0, u1 = self.u_range
        v0, v1 = self.v_range
        return (
            (min(u0 * r_min, u0 * r_max), max(u1 * r_min, u1 * r_max)),
            (min(v0 * r_min, v0 * r_max), max(v1 * r_min, v1 * r_max)),
            (self.cos_radius * r_min, r_max),
        )


@dataclass(frozen=True)
class CubeGrid:
    """The chunk grid of a planet of one radius. Build with :func:`grid_for`.

    :ivar radius: planet radius, metres
    :ivar max_level: level of the six face roots
    :ivar n0: level-0 index steps per face edge (QUADS * 2**max_level)
    :ivar spacing0: nominal level-0 spacing (face arc / n0), metres
    :ivar uv_unit: texture metres per level-0 index step; a face edge is a
        whole number of :data:`UV_PERIOD`, so textures continue across seams
    """

    radius: float
    max_level: int
    n0: int
    spacing0: float
    uv_unit: float
    #: Memoised :class:`KeyBox` per key (keyed by the bare key: hashing the
    #: grid itself on every LOD visit would cost more than the box maths)
    _boxes: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    # ── sizes ──────────────────────────────────────────────────────────────────

    def spacing(self, level: int) -> float:
        """:returns: nominal vertex spacing at *level*, metres (same for every
        chunk of the level, so neighbours band-limit alike)"""
        return self.spacing0 * (1 << level)

    def chunk_size(self, level: int) -> float:
        """:returns: nominal chunk side at *level*, metres"""
        return self.spacing(level) * QUADS

    def side(self, level: int) -> int:
        """:returns: chunks per face edge at *level*"""
        return 1 << (self.max_level - level)

    def roots(self) -> list[Key]:
        """:returns: the six face roots"""
        return [(self.max_level, face, 0, 0) for face in range(6)]

    # ── indices and directions ─────────────────────────────────────────────────

    def param(self, k: ArrayLike) -> np.ndarray:
        """:returns: face parameter of level-0 indices (exact)"""
        return (2.0 * np.asarray(k, dtype=np.float64) - self.n0) / self.n0

    def index_coords(self, key: Key, border: int = 0) -> np.ndarray:
        """Level-0 indices of a chunk's sample columns (or rows).

        :param key: the chunk
        :param border: extra samples on each side
        :returns: (VERTS + 2 border,) int64 indices for the u axis; the v
            axis is the same with j for i, see :meth:`grid_indices`
        """
        level, _, i, _ = key
        return (i * QUADS + np.arange(-border, VERTS + border, dtype=np.int64)) << level

    def grid_indices(self, key: Key, border: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """:returns: (ka, kb), level-0 indices of the columns and rows, each
        (VERTS + 2 border,) int64"""
        level, _, i, j = key
        steps = np.arange(-border, VERTS + border, dtype=np.int64)
        return (i * QUADS + steps) << level, (j * QUADS + steps) << level

    def centre_dir(self, key: Key) -> np.ndarray:
        """:returns: (3,) direction of the chunk's centre"""
        level, face, i, j = key
        half = QUADS // 2
        a = self.param((i * QUADS + half) << level)
        b = self.param((j * QUADS + half) << level)
        return face_dirs(face, a, b)[0]

    def box(self, key: Key) -> KeyBox:
        """:returns: the chunk's :class:`KeyBox` (memoised)"""
        box = self._boxes.get(key)
        if box is None:
            if len(self._boxes) > _BOX_CACHE_LIMIT:
                self._boxes.clear()
            box = self._boxes[key] = _key_box(self, key)
        return box

    def uv_origin(self, key: Key) -> tuple[float, float]:
        """:returns: texture coordinate of the chunk's first vertex, wrapped
        to [0, UV_PERIOD), metres"""
        level, _, i, j = key
        return (
            math.fmod(((i * QUADS) << level) * self.uv_unit, UV_PERIOD),
            math.fmod(((j * QUADS) << level) * self.uv_unit, UV_PERIOD),
        )

    # ── tree ───────────────────────────────────────────────────────────────────

    @staticmethod
    def children(key: Key) -> list[Key]:
        """:returns: the four children, south-west first, rows then columns"""
        level, face, i, j = key
        return [
            (level - 1, face, 2 * i + di, 2 * j + dj) for dj in (0, 1) for di in (0, 1)
        ]

    @staticmethod
    def parent(key: Key) -> Key:
        """:returns: the chunk containing *key* one level up"""
        level, face, i, j = key
        return level + 1, face, i >> 1, j >> 1

    def locate(
        self, dirs: np.ndarray, level: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The chunk at *level* under each direction, and where in it.

        :param dirs: (N, 3) directions
        :param level: chunk level
        :returns: face, i, j (N,) int, and fu, fv (N,) float: position inside
            the chunk in quads, [0, QUADS]
        """
        face, a, b = dirs_to_face(dirs)
        per_chunk = QUADS << level
        ka = (a + 1.0) * (self.n0 / 2.0)
        kb = (b + 1.0) * (self.n0 / 2.0)
        last = self.side(level) - 1
        i = np.clip(np.floor(ka / per_chunk).astype(np.int64), 0, last)
        j = np.clip(np.floor(kb / per_chunk).astype(np.int64), 0, last)
        scale = float(1 << level)
        fu = ka / scale - i * QUADS
        fv = kb / scale - j * QUADS
        return face, i, j, fu, fv


@functools.lru_cache(maxsize=16)
def grid_for(radius: float) -> CubeGrid:
    """The grid for a planet radius.

    The level count is picked so the level-0 spacing is as near
    :data:`TARGET_SPACING` as a power of two allows: 1.8 m at 600 km (14
    levels), 2.4 m at 6371 km (17 levels).

    :param radius: metres
    :returns: the (cached) grid
    """
    arc = math.pi * radius / 2.0
    max_level = max(0, round(math.log2(arc / (QUADS * TARGET_SPACING))))
    n0 = QUADS << max_level
    spacing0 = arc / n0
    periods = max(1, round(arc / UV_PERIOD))
    return CubeGrid(
        radius=float(radius),
        max_level=max_level,
        n0=n0,
        spacing0=spacing0,
        uv_unit=periods * UV_PERIOD / n0,
    )


_BOX_CACHE_LIMIT = 200_000


_FRAMES_PY = tuple(
    tuple(tuple(float(c) for c in axis) for axis in frame) for frame in FACE_FRAMES
)


def _point(face: int, a: float, b: float) -> tuple[float, float, float]:
    """Scalar :func:`face_dirs` (not bit-exact with it; for bounds only)."""
    x = math.tan(math.pi / 4.0 * a)
    y = math.tan(math.pi / 4.0 * b)
    n, u, v = _FRAMES_PY[face]
    inv = 1.0 / math.sqrt(1.0 + x * x + y * y)
    return (
        (n[0] + x * u[0] + y * v[0]) * inv,
        (n[1] + x * u[1] + y * v[1]) * inv,
        (n[2] + x * u[2] + y * v[2]) * inv,
    )


def _key_box(grid: CubeGrid, key: Key) -> KeyBox:
    """Pure Python: the LOD pass builds these for new keys while flying, and
    numpy's per-call overhead would dominate on a dozen points."""
    level, face, i, j = key
    span = 2.0 * (QUADS << level) / grid.n0  # face parameter width
    a0, b0 = -1.0 + i * span, -1.0 + j * span
    # The patch boundary, densely enough that the extents of an edge's arc
    # (whose extreme can fall between samples) are bounded by the pad below.
    angle = (math.pi / 4.0) * span
    steps = min(16, max(2, math.ceil(angle / 0.05)))
    t = [k / steps for k in range(steps + 1)]
    boundary = (
        [_point(face, a0 + span * f, b0) for f in t]
        + [_point(face, a0 + span, b0 + span * f) for f in t]
        + [_point(face, a0 + span * f, b0 + span) for f in t]
        + [_point(face, a0, b0 + span * f) for f in t]
    )
    up = _point(face, a0 + 0.5 * span, b0 + 0.5 * span)
    _, u_axis, _ = _FRAMES_PY[face]
    along = u_axis[0] * up[0] + u_axis[1] * up[1] + u_axis[2] * up[2]
    t_u = [u_axis[k] - up[k] * along for k in range(3)]
    norm = math.sqrt(t_u[0] ** 2 + t_u[1] ** 2 + t_u[2] ** 2)
    t_u = (t_u[0] / norm, t_u[1] / norm, t_u[2] / norm)
    t_v = (
        up[1] * t_u[2] - up[2] * t_u[1],
        up[2] * t_u[0] - up[0] * t_u[2],
        up[0] * t_u[1] - up[1] * t_u[0],
    )
    du = [p[0] * t_u[0] + p[1] * t_u[1] + p[2] * t_u[2] for p in boundary]
    dv = [p[0] * t_v[0] + p[1] * t_v[1] + p[2] * t_v[2] for p in boundary]
    # A unit arc bulges at most (step angle)^2 / 8 past its chord.
    step_angle = angle / steps
    pad = step_angle * step_angle / 8.0 + 1e-9
    corners = (
        boundary[0],
        boundary[steps],
        boundary[2 * steps + 1],
        boundary[2 * steps + 2],
    )
    cos_radius = min(c[0] * up[0] + c[1] * up[1] + c[2] * up[2] for c in corners)
    return KeyBox(
        up=up,
        t_u=t_u,
        t_v=t_v,
        u_range=(min(du) - pad, max(du) + pad),
        v_range=(min(dv) - pad, max(dv) + pad),
        cos_radius=cos_radius,
        sin_radius=math.sqrt(max(0.0, 1.0 - cos_radius * cos_radius)),
    )
