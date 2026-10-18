"""
Terrain queries for collision, sensors and gameplay (pure numpy).

The physical surface is the chunk grid at :attr:`TerrainQuery.level`, the
finest level the renderer draws, interpolated on the same triangles, so near
the camera collisions match what is seen exactly (apart from the visual
curvature drop, 8 cm at 1 km). Physics stays flat, like the ocean's collision
plane.

Query results are shaped for the game's collision code when it is wired in:
:meth:`TerrainQuery.sphere_contact` returns the (hit point, surface normal)
pair that ``CollisionSystem.sensor_into_obstacle`` and
``ship_into_terrain_pushback`` read from a Panda collision entry.

Chunks come from the store the renderer fills. A missing chunk is generated
on the spot (about 25 ms) when ``generate=True``, else the finest stored
ancestor answers, which is cheaper and fine for distant bots.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping

import numpy as np

from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chunkgen import (
    MAX_LEVEL,
    QUADS,
    ChunkData,
    chunk_size,
    chunk_spacing,
    generate_chunk,
)
from world_builder.terrain.planet import Planet


class TerrainQuery:
    """Height, normal and contact queries against one planet's terrain."""

    def __init__(
        self,
        planet: Planet,
        store: MutableMapping[tuple, ChunkData] | None = None,
        version: int = 0,
        include_ocean: bool | None = None,
        level: int = 0,
    ):
        """
        :param planet: the planet
        :param store: shared chunk cache (the renderer's); a private one if None
        :param version: spec version stamped on chunks generated here
        :param include_ocean: treat the sea surface (z = 0) as solid; defaults
            to whether the planet has an ocean
        :param level: grid level queries read by default; keep it equal to
            the finest level drawn, so the physical ground is the visible one
        """
        self.level = level
        self.planet = planet
        self.store = store if store is not None else {}
        self.version = version
        self.include_ocean = (
            planet.spec.has_ocean if include_ocean is None else include_ocean
        )

    # ── point queries ──────────────────────────────────────────────────────────

    def height_at(
        self, x, y, level: int | None = None, generate: bool = True
    ) -> np.ndarray:
        """Surface height (sea surface included if :attr:`include_ocean`).

        :param x: chart x, metres (scalar or array)
        :param y: chart y, metres (same shape)
        :param level: grid level to query; None = :attr:`level`
        :param generate: generate missing chunks rather than fall back to
            coarser stored ones
        :returns: heights, metres, shaped like x
        """
        heights, _ = self._surface(x, y, self._level(level), generate)
        return heights

    def ground_height_at(
        self, x, y, level: int | None = None, generate: bool = True
    ) -> np.ndarray:
        """Like :meth:`height_at`, but ignoring the sea surface."""
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        h, _ = self._interpolate(x.ravel(), y.ravel(), self._level(level), generate)
        return h.reshape(x.shape)

    def normal_at(
        self, x, y, level: int | None = None, generate: bool = True
    ) -> np.ndarray:
        """Surface normal of the triangle under each point.

        :returns: (..., 3) unit normals
        """
        _, normals = self._surface(x, y, self._level(level), generate)
        return normals

    def biome_weights_at(self, x: float, y: float, generate: bool = True) -> np.ndarray:
        """:returns: (6,) biome weights at the nearest vertex of the query grid"""
        chunk = self._chunk(x, y, self.level, generate)
        if chunk is None:
            return np.zeros(len(BIOMES))
        col, row = self._nearest_vertex(chunk, x, y)
        return chunk.weights[row * (QUADS + 1) + col] / 255.0

    def material_at(self, x: float, y: float, generate: bool = True) -> str:
        """:returns: "water" over the sea, else the dominant biome's name"""
        if (
            self.include_ocean
            and float(self.ground_height_at(x, y, generate=generate)) < 0.0
        ):
            return "water"
        return BIOMES[int(np.argmax(self.biome_weights_at(x, y, generate)))]

    # ── shape queries ──────────────────────────────────────────────────────────

    def sphere_contact(self, center, radius: float, generate: bool = True):
        """Contact between a sphere and the surface.

        The grid is read at the level whose spacing is about radius / 4 (and
        never finer than :attr:`level`), so a 100 m sensor sphere costs about
        as much as a 2 m one.

        :param center: (3,) chart position, metres
        :param radius: metres
        :param generate: see :meth:`height_at`
        :returns: (hit_point (3,), normal (3,)) if the sphere touches or
            penetrates the surface, else None. The hit point is the surface
            point nearest the centre; the normal points out of the ground.
        """
        c = np.asarray(center, dtype=np.float64)
        level = int(
            np.clip(
                math.floor(math.log2(max(radius / 4.0, 1e-6) / chunk_spacing(0))),
                self.level,
                MAX_LEVEL,
            )
        )
        spacing = chunk_spacing(level)
        # Grid samples inside the sphere's footprint, plus one ring.
        reach = radius + spacing
        i0, i1 = (
            math.floor((c[0] - reach) / spacing),
            math.ceil((c[0] + reach) / spacing),
        )
        j0, j1 = (
            math.floor((c[1] - reach) / spacing),
            math.ceil((c[1] + reach) / spacing),
        )
        gx, gy = np.meshgrid(
            spacing * np.arange(i0, i1 + 1), spacing * np.arange(j0, j1 + 1)
        )
        gx, gy = gx.ravel(), gy.ravel()
        heights, normals = self._surface(gx, gy, level, generate)
        if c[2] - radius > heights.max():
            return None
        points = np.column_stack([gx, gy, heights])
        nearest = int(np.argmin(np.sum((points - c) ** 2, axis=1)))
        # Refine on the tangent plane of the triangle under the centre.
        h_c, n_c = self._surface(np.array([c[0]]), np.array([c[1]]), level, generate)
        normal = n_c[0]
        on_surface = np.array([c[0], c[1], h_c[0]])
        distance = float(normal @ (c - on_surface))
        if distance >= radius:
            # The plane under the centre is clear; a steeper neighbour may not be.
            p = points[nearest]
            if np.linalg.norm(c - p) >= radius:
                return None
            normal = normals[nearest]
            distance = float(normal @ (c - p))
            if distance >= radius:
                return None
            on_surface = p
        hit = c - normal * distance
        return hit, normal

    def segment_intersect(self, p0, p1, generate: bool = False):
        """First crossing of a segment (e.g. a laser) into the surface.

        :param p0: (3,) start, metres
        :param p1: (3,) end, metres
        :param generate: see :meth:`height_at`; off by default, since a long
            laser crosses many chunks
        :returns: (t in [0, 1], point (3,), normal (3,)), or None
        """
        a = np.asarray(p0, dtype=np.float64)
        b = np.asarray(p1, dtype=np.float64)
        length = float(np.linalg.norm(b - a))
        steps = max(2, math.ceil(length / chunk_spacing(self.level)))
        t = np.linspace(0.0, 1.0, steps + 1)
        pts = a + t[:, None] * (b - a)
        above = pts[:, 2] - self.height_at(pts[:, 0], pts[:, 1], generate=generate)
        below = np.nonzero(above <= 0.0)[0]
        if below.size == 0:
            return None
        k = int(below[0])
        if k == 0:
            n = self.normal_at(a[0], a[1], generate=generate)
            return 0.0, a.copy(), n
        lo, hi = t[k - 1], t[k]
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            p = a + mid * (b - a)
            if p[2] - float(self.height_at(p[0], p[1], generate=generate)) > 0.0:
                lo = mid
            else:
                hi = mid
        point = a + hi * (b - a)
        point[2] = float(self.height_at(point[0], point[1], generate=generate))
        return hi, point, self.normal_at(point[0], point[1], generate=generate)

    # ── internals ──────────────────────────────────────────────────────────────

    def _level(self, level: int | None) -> int:
        return self.level if level is None else level

    def _surface(self, x, y, level, generate):
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        shape = x.shape
        h, n = self._interpolate(x.ravel(), y.ravel(), level, generate)
        if self.include_ocean:
            wet = h < 0.0
            h = np.where(wet, 0.0, h)
            n[wet] = (0.0, 0.0, 1.0)
        return h.reshape(shape), n.reshape(*shape, 3)

    def _interpolate(self, x: np.ndarray, y: np.ndarray, level: int, generate: bool):
        """Height and triangle normal at flat arrays of points."""
        heights = np.zeros(x.size)
        normals = np.zeros((x.size, 3))
        size = chunk_size(level)
        ix = np.floor(x / size).astype(np.int64)
        iy = np.floor(y / size).astype(np.int64)
        keys, inverse = np.unique(
            np.column_stack([ix, iy]), axis=0, return_inverse=True
        )
        for k, (kx, ky) in enumerate(keys):
            sel = np.nonzero(inverse.ravel() == k)[0]
            chunk = self._chunk((kx + 0.5) * size, (ky + 0.5) * size, level, generate)
            if chunk is None:
                heights[sel] = self.planet.sample_points(x[sel], y[sel])[0]
                normals[sel] = (0.0, 0.0, 1.0)
                continue
            heights[sel], normals[sel] = _triangle_interpolate(chunk, x[sel], y[sel])
        return heights, normals

    def _chunk(
        self, x: float, y: float, level: int, generate: bool
    ) -> ChunkData | None:
        """The chunk at *level* containing (x, y), generated or an ancestor."""
        for lvl in range(level, MAX_LEVEL + 1):
            size = chunk_size(lvl)
            key = (lvl, math.floor(x / size), math.floor(y / size))
            chunk = self.store.get(key)
            if chunk is not None:
                return chunk
            if generate and lvl == level:
                chunk = generate_chunk(self.planet.spec, key, self.version)
                self.store[key] = chunk
                return chunk
        return None

    @staticmethod
    def _nearest_vertex(chunk: ChunkData, x: float, y: float) -> tuple[int, int]:
        level, kx, ky = chunk.key
        spacing = chunk_spacing(level)
        col = int(np.clip(round((x - kx * chunk_size(level)) / spacing), 0, QUADS))
        row = int(np.clip(round((y - ky * chunk_size(level)) / spacing), 0, QUADS))
        return col, row


def _triangle_interpolate(chunk: ChunkData, x: np.ndarray, y: np.ndarray):
    """Interpolate on the chunk's own triangles (diagonal (i, j)-(i+1, j+1)).

    :returns: heights (N,) and triangle normals (N, 3)
    """
    level, kx, ky = chunk.key
    size = chunk_size(level)
    spacing = chunk_spacing(level)
    u = (x - kx * size) / spacing
    v = (y - ky * size) / spacing
    col = np.clip(np.floor(u).astype(np.int64), 0, QUADS - 1)
    row = np.clip(np.floor(v).astype(np.int64), 0, QUADS - 1)
    fu = u - col
    fv = v - row
    e = chunk.elevation.astype(np.float64)
    h00 = e[row, col]
    h01 = e[row, col + 1]  # east
    h10 = e[row + 1, col]  # north
    h11 = e[row + 1, col + 1]
    lower = fu >= fv  # triangle (00, 01, 11)
    gx = np.where(lower, h01 - h00, h11 - h10)
    gy = np.where(lower, h11 - h01, h10 - h00)
    heights = h00 + fu * gx + fv * gy
    normals = np.column_stack([-gx / spacing, -gy / spacing, np.ones_like(gx)])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return heights, normals
