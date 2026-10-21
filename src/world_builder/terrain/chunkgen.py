"""
Terrain chunk generation: one cube-sphere tile into ready-to-upload vertex bytes.

Pure numpy, so it runs in worker processes. A chunk ``(level, face, i, j)``
(see :mod:`cubesphere`) holds ``QUADS x QUADS`` quads of one cube face.
Vertices are sampled at integer level-0 indices, so neighbouring chunks
compute bit-identical edge directions and heights, across face seams too.

Positions are ``dir * (R + h)`` in the planet frame (float64), stored
relative to the chunk's float64 centre as float32: a leaf is resolved to
better than a millimetre, a whole face to a few centimetres.

The vertex layout is one interleaved 36-byte record (see :data:`VERTEX_DTYPE`),
packed here so the main thread only copies bytes. A skirt (one extra vertex
per edge vertex, hung radially down) closes the cracks between chunks of
different levels and the float-rounding seams between faces.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import numpy as np

from world_builder.terrain import biomes
from world_builder.terrain.cubesphere import QUADS, VERTS, Key
from world_builder.terrain.planet import Planet, PlanetSpec

__all__ = ["QUADS", "VERTS", "VERTEX_DTYPE", "ChunkData", "generate_chunk"]

#: Interleaved vertex record. Biome weights are normalised bytes:
#: a = snow, rock, grass, forest; b = desert, beach, unused, skirt flag.
#: ``height`` is metres above sea level (the skirt's own, below its edge).
VERTEX_DTYPE = np.dtype(
    [
        ("vertex", np.float32, 3),
        ("normal", np.float32, 3),
        ("height", np.float32),
        ("biome_a", np.uint8, 4),
        ("biome_b", np.uint8, 4),
    ]
)


@dataclass
class ChunkData:
    """One generated chunk; picklable, so it crosses process boundaries.

    :ivar key: (level, face, i, j)
    :ivar version: spec version it was generated for; stale ones are dropped
    :ivar centre: (3,) float64 planet-frame point the vertices are relative to
    :ivar vertex_bytes: interleaved :data:`VERTEX_DTYPE` records, grid first
        (row-major, VERTS x VERTS, rows along the face's v) then the skirt ring
    :ivar elevation: (VERTS, VERTS) float32 grid heights
    :ivar weights: (VERTS * VERTS, 6) uint8 biome weights
    :ivar aabb_min: (3,) float32 lower corner of all vertices, centre-relative
    :ivar aabb_max: (3,) float32 upper corner
    :ivar h_min: lowest grid height, metres
    :ivar h_max: highest grid height, metres
    :ivar skirt_depth: how far the skirt hangs below its edge, metres
    :ivar geometric_error: max height change against the parent's coarser
        interpolation, metres
    :ivar gen_ms: generation time, milliseconds
    """

    key: Key
    version: int
    centre: np.ndarray
    vertex_bytes: bytes
    elevation: np.ndarray
    weights: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    h_min: float
    h_max: float
    skirt_depth: float
    geometric_error: float
    gen_ms: float

    def vertices(self) -> np.ndarray:
        """:returns: the vertex records as a :data:`VERTEX_DTYPE` array (a view)"""
        return np.frombuffer(self.vertex_bytes, dtype=VERTEX_DTYPE)


def skirt_ring() -> np.ndarray:
    """Grid indices of the edge vertices, in order around the chunk.

    :returns: (4 * QUADS,) indices into the row-major grid, counter-clockwise
        from the south-west corner
    """
    n = QUADS
    south = [(0, i) for i in range(n)]
    east = [(j, n) for j in range(n)]
    north = [(n, n - i) for i in range(n)]
    west = [(n - j, 0) for j in range(n)]
    return np.array([r * VERTS + c for r, c in south + east + north + west])


@functools.cache
def triangle_indices() -> np.ndarray:
    """The index buffer every chunk shares: grid, then skirt.

    Each quad is cut along its (i, j)-(i+1, j+1) diagonal; :mod:`query`
    interpolates on the same triangles.

    :returns: uint16 indices, counter-clockwise seen from outside the planet
    """
    r, c = np.meshgrid(np.arange(QUADS), np.arange(QUADS), indexing="ij")
    v00 = (r * VERTS + c).ravel()
    v01 = v00 + 1  # +u
    v10 = v00 + VERTS  # +v
    v11 = v10 + 1
    grid = np.stack([v00, v01, v11, v00, v11, v10], axis=1).ravel()

    ring = skirt_ring()
    top_a = ring
    top_b = np.roll(ring, -1)
    bottom_a = VERTS * VERTS + np.arange(len(ring))
    bottom_b = np.roll(bottom_a, -1)
    # The ring runs counter-clockwise seen from outside, so outward faces wind
    # (top_a, bottom_a, bottom_b), (top_a, bottom_b, top_b). A crack can be
    # seen from either side, so the skirt is two-sided: both windings.
    skirt = np.stack(
        [
            top_a,
            bottom_a,
            bottom_b,
            top_a,
            bottom_b,
            top_b,
            top_a,
            bottom_b,
            bottom_a,
            top_a,
            top_b,
            bottom_b,
        ],
        axis=1,
    ).ravel()
    return np.concatenate([grid, skirt]).astype(np.uint16)


@functools.lru_cache(maxsize=4)
def planet_for(spec: PlanetSpec) -> Planet:
    """:returns: a cached :class:`Planet` (one per worker process and spec)"""
    return Planet(spec)


def generate_chunk(spec: PlanetSpec, key: Key, version: int = 0) -> ChunkData:
    """Generate one chunk. The worker-process entry point.

    :param spec: the planet, resolved (sea bias set)
    :param key: (level, face, i, j)
    :param version: spec version, carried through to the result
    :returns: the chunk
    """
    start = time.perf_counter()
    planet = planet_for(spec)
    grid = planet.grid
    level, face = key[0], key[1]
    spacing = grid.spacing(level)
    radius = spec.radius

    # A border wide enough for the biome slope (measured at a fixed scale) and
    # the normals (one sample). Past a face edge it samples the extended face,
    # which is still the sphere.
    step = max(1, round(biomes.SLOPE_SCALE / spacing))
    border = step
    n = VERTS + 2 * border
    ka, kb = grid.grid_indices(key, border)
    sample = planet.sample_face_grid(face, ka, kb, 1 << level)
    full = sample.elevation
    dirs = sample.dirs.reshape(n, n, 3)
    points = dirs * (radius + full)[:, :, None]
    inner = (slice(border, border + VERTS), slice(border, border + VERTS))
    elevation = full[inner]

    normals = _grid_normals(points, border, 1)
    slope = biomes.slope_from_normals(_grid_normals(points, border, step), dirs[inner])
    w = biomes.weights(
        sample.temperature[inner],
        sample.humidity[inner],
        elevation,
        slope,
        spec.has_ocean,
        spec.has_vegetation,
    )
    inner_dirs = dirs[inner].reshape(-1, 3)
    for paint in spec.paints:
        w = paint.apply(inner_dirs, w, radius)
    w_bytes = np.clip(np.rint(w * 255.0), 0, 255).astype(np.uint8)

    geometric_error = float(np.max(np.abs(elevation - parent_interpolation(elevation))))
    h_min, h_max = float(elevation.min()), float(elevation.max())
    skirt_depth = min(
        4.0 * spacing + 4.0 * geometric_error, 0.5 * grid.chunk_size(level)
    )
    centre = grid.centre_dir(key) * (radius + 0.5 * (h_min + h_max))

    vertices = _pack_vertices(
        points[inner], inner_dirs, centre, elevation, normals, w_bytes, skirt_depth
    )
    local = vertices["vertex"]
    return ChunkData(
        key=key,
        version=version,
        centre=centre,
        vertex_bytes=vertices.tobytes(),
        elevation=elevation.astype(np.float32),
        weights=w_bytes,
        aabb_min=local.min(axis=0),
        aabb_max=local.max(axis=0),
        h_min=h_min,
        h_max=h_max,
        skirt_depth=skirt_depth,
        geometric_error=geometric_error,
        gen_ms=(time.perf_counter() - start) * 1e3,
    )


def _grid_normals(points: np.ndarray, border: int, step: int) -> np.ndarray:
    """Outward unit normals of the inner VERTS x VERTS grid, by central
    differences over +-*step* samples (the border must be at least *step*).

    :param points: (n, n, 3) float64 positions, rows along v
    :returns: (VERTS, VERTS, 3)
    """
    b, s = border, step
    rows = slice(b, b + VERTS)
    d_u = points[rows, b + s : b + s + VERTS] - points[rows, b - s : b - s + VERTS]
    d_v = (
        points[b + s : b + s + VERTS, b : b + VERTS]
        - points[b - s : b - s + VERTS, b : b + VERTS]
    )
    normals = np.cross(d_u, d_v)  # u x v points out of the face
    return normals / np.linalg.norm(normals, axis=2, keepdims=True)


def parent_interpolation(elevation: np.ndarray) -> np.ndarray:
    """The grid as its parent chunk would show it (every other sample, linear).

    Used for the geometric error (and a later geomorphing pass).

    :param elevation: (VERTS, VERTS) heights
    :returns: same shape; the even samples unchanged, odd ones interpolated
        along the parent's triangle diagonals
    """
    m = elevation.copy()
    m[:, 1::2] = 0.5 * (m[:, 0:-1:2] + m[:, 2::2])  # odd columns on even rows
    m[1::2, :] = 0.5 * (m[0:-1:2, :] + m[2::2, :])  # odd rows
    # Odd-odd samples sit on the parent's (i, j)-(i+1, j+1) diagonal.
    m[1::2, 1::2] = 0.5 * (elevation[0:-1:2, 0:-1:2] + elevation[2::2, 2::2])
    return m


def _pack_vertices(
    points: np.ndarray,
    dirs: np.ndarray,
    centre: np.ndarray,
    elevation: np.ndarray,
    normals: np.ndarray,
    w_bytes: np.ndarray,
    skirt_depth: float,
) -> np.ndarray:
    """:returns: grid then skirt vertices as a :data:`VERTEX_DTYPE` array"""
    ring = skirt_ring()
    count = VERTS * VERTS
    out = np.zeros(count + len(ring), dtype=VERTEX_DTYPE)
    out["vertex"][:count] = points.reshape(-1, 3) - centre
    out["normal"][:count] = normals.reshape(-1, 3)
    out["height"][:count] = elevation.ravel()
    out["biome_a"][:count] = w_bytes[:, 0:4]
    out["biome_b"][:count, 0:2] = w_bytes[:, 4:6]

    skirt = out[count:]
    skirt[:] = out[ring]
    edge = points.reshape(-1, 3)[ring]
    skirt["vertex"] = edge - skirt_depth * dirs[ring] - centre
    skirt["height"] -= skirt_depth
    skirt["biome_b"][:, 3] = 255  # flag, for the skirt debug view
    return out
