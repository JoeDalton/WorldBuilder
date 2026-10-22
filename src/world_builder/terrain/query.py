"""
Terrain queries for collision, sensors and gameplay (pure numpy).

Positions are float64 metres in the planet frame (centre at the origin, no
rotation). Heights are measured along the radial line through the query
point: ``height_at(p)`` is the surface's height above sea level straight
below (or above) p, ``altitude(p)`` how far p is above that surface.

The physical surface is the chunk grid at :attr:`TerrainQuery.level`, the
finest level the renderer draws: a radial ray is intersected with the very
triangle on screen (same vertices, same diagonal), so collisions match what
is seen exactly.

Query results are shaped for the game's collision code when it is wired in:
:meth:`TerrainQuery.sphere_contact` returns the (hit point, surface normal)
pair that ``CollisionSystem.sensor_into_obstacle`` and
``ship_into_terrain_pushback`` read from a Panda collision entry.

Chunks come from the store the renderer fills. A missing chunk is generated
on the spot (about 20 ms) when ``generate=True``, else the finest stored
ancestor answers, which is cheaper and fine for distant bots. With no
ancestor stored either, the planet is evaluated directly at the point (no
triangles: the radial direction stands in for the normal).
"""

from __future__ import annotations

import math
from collections.abc import Callable, MutableMapping

import numpy as np
from numpy.typing import ArrayLike

from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chunkgen import QUADS, VERTS, ChunkData, generate_chunk
from world_builder.terrain.cubesphere import (
    FACE_FRAMES,
    CubeGrid,
    dirs_to_face,
    face_dirs,
)
from world_builder.terrain.planet import Planet

#: Face-parameter margin of the sphere_contact sample window: cells near a
#: face's corners are down to 0.71x the nominal spacing.
_WINDOW_MARGIN = 1.5


class TerrainQuery:
    """Height, normal and contact queries against one planet's terrain."""

    def __init__(
        self,
        planet: Planet,
        store: MutableMapping[tuple, ChunkData] | None = None,
        version: int = 0,
        include_ocean: bool | None = None,
        level: int = 0,
        store_chunk: Callable[[ChunkData], None] | None = None,
    ):
        """
        :param planet: the planet
        :param store: shared chunk cache (the renderer's); a private one if None
        :param version: spec version stamped on chunks generated here
        :param include_ocean: treat the sea surface (sea level) as solid;
            defaults to whether the planet has an ocean
        :param level: grid level queries read by default; keep it equal to
            the finest level drawn, so the physical ground is the visible one
        :param store_chunk: called with each chunk generated here, to put it
            in *store*; the store's owner passes its own (the renderer keeps
            bookkeeping per chunk). None puts it in *store* directly.
        """
        self.level = level
        self.planet = planet
        self.grid: CubeGrid = planet.grid
        self.radius = planet.spec.radius
        self.store = store if store is not None else {}
        self._store_chunk = store_chunk or self._put
        self.version = version
        self.include_ocean = (
            planet.spec.has_ocean if include_ocean is None else include_ocean
        )

    # ── point queries ──────────────────────────────────────────────────────────

    def height_at(
        self, points: ArrayLike, level: int | None = None, generate: bool = True
    ) -> np.ndarray:
        """Surface height above sea level under (or over) each point.

        The sea surface counts if :attr:`include_ocean`.

        :param points: (..., 3) planet-frame positions, metres
        :param level: grid level to query; None = :attr:`level`
        :param generate: generate missing chunks rather than fall back to
            coarser stored ones
        :returns: heights, metres, shaped (...)
        """
        heights, _ = self._surface(points, self._level(level), generate)
        return heights

    def ground_height_at(
        self, points: ArrayLike, level: int | None = None, generate: bool = True
    ) -> np.ndarray:
        """Like :meth:`height_at`, but ignoring the sea surface."""
        dirs, shape = _directions(points)
        h, _ = self._interpolate(dirs, self._level(level), generate)
        return h.reshape(shape)

    def normal_at(
        self, points: ArrayLike, level: int | None = None, generate: bool = True
    ) -> np.ndarray:
        """Outward normal of the surface triangle under each point (the local
        vertical over the sea, if :attr:`include_ocean`).

        :param points: (..., 3) planet-frame positions, metres
        :param level: see :meth:`height_at`
        :param generate: see :meth:`height_at`
        :returns: (..., 3) unit normals, planet frame
        """
        _, normals = self._surface(points, self._level(level), generate)
        return normals

    def altitude(
        self, points: ArrayLike, level: int | None = None, generate: bool = True
    ) -> np.ndarray:
        """:returns: metres from each point down to the surface below it
        (negative inside the ground), shaped (...)"""
        p = np.asarray(points, dtype=np.float64)
        return (
            np.linalg.norm(p, axis=-1)
            - self.radius
            - self.height_at(p, level, generate)
        )

    def surface_point(
        self, points: ArrayLike, level: int | None = None, generate: bool = True
    ) -> np.ndarray:
        """:returns: (..., 3) the surface point on each point's radial line"""
        dirs, shape = _directions(points)
        heights, _ = self._surface(dirs, self._level(level), generate)
        return (dirs * (self.radius + heights.reshape(-1))[:, None]).reshape(*shape, 3)

    def biome_weights_at(self, point: ArrayLike, generate: bool = True) -> np.ndarray:
        """:returns: (6,) biome weights in [0, 1] at the nearest vertex of the
        query grid (a stored ancestor's if the chunk is missing; zeros if
        none is stored and *generate* is off)"""
        dirs, _ = _directions(point)
        chunk = self._chunk(self._key_of(dirs, self.level), generate)
        if chunk is None:
            return np.zeros(len(BIOMES))
        fu, fv = self._in_chunk(chunk, dirs)
        col = int(np.clip(np.rint(fu[0]), 0, QUADS))
        row = int(np.clip(np.rint(fv[0]), 0, QUADS))
        return chunk.weights[row * VERTS + col] / 255.0

    def material_at(self, point: ArrayLike, generate: bool = True) -> str:
        """:returns: "water" over the sea, else the dominant biome's name"""
        if (
            self.include_ocean
            and float(self.ground_height_at(point, generate=generate)) < 0.0
        ):
            return "water"
        return BIOMES[int(np.argmax(self.biome_weights_at(point, generate)))]

    # ── shape queries ──────────────────────────────────────────────────────────

    def sphere_contact(
        self, center: ArrayLike, radius: float, generate: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Contact between a sphere and the surface.

        The grid is read at the level whose spacing is about radius / 4 (and
        never finer than :attr:`level`), so a 100 m sensor sphere costs about
        as much as a 2 m one. Works across face seams and corners.

        :param center: (3,) planet-frame position, metres
        :param radius: metres
        :param generate: see :meth:`height_at`
        :returns: (hit_point (3,), normal (3,)) if the sphere touches or
            penetrates the surface, else None. The hit point is the surface
            point nearest the centre; the normal points out of the ground.
        """
        c = np.asarray(center, dtype=np.float64)
        grid = self.grid
        level = int(
            np.clip(
                math.floor(math.log2(max(radius / 4.0, 1e-6) / grid.spacing0)),
                self.level,
                grid.max_level,
            )
        )
        step = 1 << level
        # Grid samples inside the sphere's footprint, plus one ring, laid on
        # the centre's face (extended past its edges if need be); each sample
        # then finds its own face and chunk.
        dirs_c, _ = _directions(c)
        face, a, b = dirs_to_face(dirs_c)
        reach = (radius + grid.spacing(level)) / self.radius
        half = reach * (4.0 / math.pi) * _WINDOW_MARGIN * grid.n0 / 2.0
        ka_c, kb_c = (a[0] + 1.0) * grid.n0 / 2.0, (b[0] + 1.0) * grid.n0 / 2.0
        ka = step * np.arange(
            math.floor((ka_c - half) / step), math.ceil((ka_c + half) / step) + 1
        )
        kb = step * np.arange(
            math.floor((kb_c - half) / step), math.ceil((kb_c + half) / step) + 1
        )
        ga, gb = np.meshgrid(grid.param(ka), grid.param(kb))
        # Past 1.5 the extended face folds away from the sphere: clip.
        dirs = face_dirs(int(face[0]), np.clip(ga, -1.5, 1.5), np.clip(gb, -1.5, 1.5))
        heights, normals = self._surface(dirs, level, generate)
        if np.linalg.norm(c) - radius > self.radius + heights.max():
            return None
        points = dirs * (self.radius + heights)[:, None]
        nearest = int(np.argmin(np.sum((points - c) ** 2, axis=1)))
        # Refine on the plane of the triangle under the centre.
        h_c, n_c = self._surface(dirs_c, level, generate)
        normal = n_c[0]
        on_surface = dirs_c[0] * (self.radius + h_c[0])
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
        hit = c - normal * distance
        return hit, normal

    def segment_intersect(
        self, p0: ArrayLike, p1: ArrayLike, generate: bool = False
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        """First crossing of a segment (e.g. a laser) into the surface.

        :param p0: (3,) start, planet frame, metres
        :param p1: (3,) end
        :param generate: see :meth:`height_at`; off by default, since a long
            laser crosses many chunks
        :returns: (t in [0, 1], point (3,), normal (3,)), or None
        """
        a = np.asarray(p0, dtype=np.float64)
        b = np.asarray(p1, dtype=np.float64)
        length = float(np.linalg.norm(b - a))
        steps = max(2, math.ceil(length / self.grid.spacing(self.level)))
        t = np.linspace(0.0, 1.0, steps + 1)
        pts = a + t[:, None] * (b - a)
        below = np.nonzero(self.altitude(pts, generate=generate) <= 0.0)[0]
        if below.size == 0:
            return None
        k = int(below[0])
        if k == 0:
            return 0.0, a.copy(), self.normal_at(a, generate=generate)
        lo, hi = t[k - 1], t[k]
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if float(self.altitude(a + mid * (b - a), generate=generate)) > 0.0:
                lo = mid
            else:
                hi = mid
        point = self.surface_point(a + hi * (b - a), generate=generate)
        return hi, point, self.normal_at(point, generate=generate)

    # ── internals ──────────────────────────────────────────────────────────────

    def _level(self, level: int | None) -> int:
        return self.level if level is None else level

    def _surface(self, points, level, generate):
        dirs, shape = _directions(points)
        h, n = self._interpolate(dirs, level, generate)
        if self.include_ocean:
            wet = h < 0.0
            h = np.where(wet, 0.0, h)
            n[wet] = dirs[wet]
        return h.reshape(shape), n.reshape(*shape, 3)

    def _key_of(self, dirs: np.ndarray, level: int):
        face, i, j, _, _ = self.grid.locate(dirs[:1], level)
        return level, int(face[0]), int(i[0]), int(j[0])

    def _interpolate(self, dirs: np.ndarray, level: int, generate: bool):
        """Height and triangle normal at (N, 3) directions."""
        heights = np.zeros(len(dirs))
        normals = np.zeros((len(dirs), 3))
        if len(dirs) == 0:
            return heights, normals
        face, i, j, _, _ = self.grid.locate(dirs, level)
        if len(dirs) == 1:  # the common case (an altimeter): no grouping
            groups = [((int(face[0]), int(i[0]), int(j[0])), np.zeros(1, np.intp))]
        else:
            # One int64 key per chunk, sorted once and split: for many points
            # (the ocean's seabed) far cheaper than unique rows and a scan of
            # all points per chunk.
            side = self.grid.side(level)
            key = (np.asarray(face, np.int64) * side + i) * side + j
            order = np.argsort(key, kind="stable")
            sorted_key = key[order]
            starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
            groups = [
                (
                    (int(k // (side * side)), int(k // side % side), int(k % side)),
                    sel,
                )
                for k, sel in zip(sorted_key[starts], np.split(order, starts[1:]))
            ]
        for (f, ki, kj), sel in groups:
            chunk = self._chunk((level, f, ki, kj), generate)
            if chunk is None:
                heights[sel] = self.planet.sample_dirs(dirs[sel], apply_stamps=True)[0]
                normals[sel] = dirs[sel]
                continue
            heights[sel], normals[sel] = self._triangle_intersect(chunk, dirs[sel])
        return heights, normals

    def _chunk(self, key, generate: bool) -> ChunkData | None:
        """The chunk at *key*, generated, or its finest stored ancestor."""
        level, face, i, j = key
        for up in range(self.grid.max_level - level + 1):
            ancestor = (level + up, face, i >> up, j >> up)
            chunk = self.store.get(ancestor)
            if chunk is not None:
                return chunk
            if generate and up == 0:
                chunk = generate_chunk(self.planet.spec, key, self.version)
                self._store_chunk(chunk)
                return chunk
        return None

    def _put(self, chunk: ChunkData) -> None:
        self.store[chunk.key] = chunk

    def _in_chunk(self, chunk: ChunkData, dirs: np.ndarray):
        """:returns: fu, fv, position of *dirs* inside *chunk* in quads, on the
        chunk's own face (extended past its edges if need be)"""
        level, face, i, j = chunk.key
        frame_n, frame_u, frame_v = FACE_FRAMES[face]
        along_n = dirs @ frame_n
        a = np.arctan2(dirs @ frame_u, along_n) * (4.0 / math.pi)
        b = np.arctan2(dirs @ frame_v, along_n) * (4.0 / math.pi)
        scale = float(1 << level)
        fu = (a + 1.0) * (self.grid.n0 / 2.0) / scale - i * QUADS
        fv = (b + 1.0) * (self.grid.n0 / 2.0) / scale - j * QUADS
        return fu, fv

    def _triangle_intersect(self, chunk: ChunkData, dirs: np.ndarray):
        """Radial rays against the chunk's own triangles (diagonal (i, j)-(i+1, j+1)).

        :returns: heights (N,) and triangle normals (N, 3)
        """
        fu, fv = self._in_chunk(chunk, dirs)
        col = np.clip(np.floor(fu).astype(np.int64), 0, QUADS - 1)
        row = np.clip(np.floor(fv).astype(np.int64), 0, QUADS - 1)
        grid_vertices = chunk.vertices()["vertex"]
        base = row * VERTS + col

        def corner(index):  # only the needed vertices go to float64
            return grid_vertices[index].astype(np.float64) + chunk.centre

        v00 = corner(base)
        v01 = corner(base + 1)  # +u
        v10 = corner(base + VERTS)  # +v
        v11 = corner(base + VERTS + 1)
        lower = (fu - col) >= (fv - row)  # triangle (00, 01, 11), else (00, 11, 10)
        e1 = np.where(lower[:, None], v01 - v00, v11 - v00)
        e2 = np.where(lower[:, None], v11 - v00, v10 - v00)
        n = np.cross(e1, e2)  # counter-clockwise from outside: outward
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        t = np.einsum("ij,ij->i", n, v00) / np.einsum("ij,ij->i", n, dirs)
        return t - self.radius, n


def _directions(points) -> tuple[np.ndarray, tuple]:
    """:returns: (N, 3) unit directions of (..., 3) points, and the (...) shape"""
    p = np.asarray(points, dtype=np.float64)
    shape = p.shape[:-1]
    flat = p.reshape(-1, 3)
    return flat / np.linalg.norm(flat, axis=1, keepdims=True), shape
