"""
Terrain chunk generation: one quadtree tile into ready-to-upload vertex bytes.

Pure numpy, so it runs in worker processes. A chunk at level L is a square of
``LEAF_SIZE * 2**L`` metres on the flat chart, aligned to multiples of its own
size, holding ``QUADS x QUADS`` quads. Vertices are sampled at integer
multiples of the spacing from a world-aligned origin, so neighbouring chunks
compute bit-identical edge heights.

The vertex layout is one interleaved 36-byte record (see :data:`VERTEX_DTYPE`),
packed here so the main thread only copies bytes. A skirt (one extra vertex
per edge vertex, hung straight down) closes the cracks between chunks of
different levels. The camera-relative drop depends only on x and y, so skirts
stay vertical and attached under it.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import numpy as np

from world_builder.terrain import biomes
from world_builder.terrain.planet import Planet, PlanetSpec

#: Side of a level-0 (finest) chunk, metres.
LEAF_SIZE = 128.0
#: Quads per chunk side; vertices per side is one more.
QUADS = 64
VERTS = QUADS + 1
#: Coarsest level: 524 km chunks.
MAX_LEVEL = 12

#: Interleaved vertex record. Biome weights are normalised bytes:
#: a = snow, rock, grass, forest; b = desert, beach, unused, skirt flag.
VERTEX_DTYPE = np.dtype(
    [
        ("vertex", np.float32, 3),
        ("normal", np.float32, 3),
        ("morph_z", np.float32),
        ("biome_a", np.uint8, 4),
        ("biome_b", np.uint8, 4),
    ]
)


def chunk_size(level: int) -> float:
    """:returns: side of a chunk at *level*, metres"""
    return LEAF_SIZE * (1 << level)


def chunk_spacing(level: int) -> float:
    """:returns: vertex spacing of a chunk at *level*, metres"""
    return chunk_size(level) / QUADS


def chunk_origin(key: tuple[int, int, int]) -> tuple[float, float]:
    """:returns: chart (x, y) of the chunk's south-west corner"""
    level, ix, iy = key
    size = chunk_size(level)
    return ix * size, iy * size


def chunk_center(key: tuple[int, int, int]) -> tuple[float, float]:
    """:returns: chart (x, y) of the chunk's centre"""
    x, y = chunk_origin(key)
    half = chunk_size(key[0]) / 2.0
    return x + half, y + half


@dataclass
class ChunkData:
    """One generated chunk; picklable, so it crosses process boundaries.

    :ivar key: (level, ix, iy)
    :ivar version: spec version it was generated for; stale ones are dropped
    :ivar vertex_bytes: interleaved :data:`VERTEX_DTYPE` records, grid first
        (row-major, VERTS x VERTS) then the skirt ring
    :ivar elevation: (VERTS, VERTS) float32 grid heights, row = y
    :ivar weights: (VERTS * VERTS, 6) uint8 biome weights
    :ivar h_min: lowest grid height, metres
    :ivar h_max: highest grid height, metres
    :ivar skirt_depth: how far the skirt hangs below its edge, metres
    :ivar geometric_error: max height change against the parent's coarser
        interpolation, metres
    :ivar gen_ms: generation time, milliseconds
    """

    key: tuple[int, int, int]
    version: int
    vertex_bytes: bytes
    elevation: np.ndarray
    weights: np.ndarray
    h_min: float
    h_max: float
    skirt_depth: float
    geometric_error: float
    gen_ms: float


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

    :returns: uint16 indices, counter-clockwise seen from above
    """
    r, c = np.meshgrid(np.arange(QUADS), np.arange(QUADS), indexing="ij")
    v00 = (r * VERTS + c).ravel()
    v01 = v00 + 1  # east
    v10 = v00 + VERTS  # north
    v11 = v10 + 1
    grid = np.stack([v00, v01, v11, v00, v11, v10], axis=1).ravel()

    ring = skirt_ring()
    top_a = ring
    top_b = np.roll(ring, -1)
    bottom_a = VERTS * VERTS + np.arange(len(ring))
    bottom_b = np.roll(bottom_a, -1)
    # The ring runs counter-clockwise seen from above, so outward faces wind
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


def generate_chunk(
    spec: PlanetSpec, key: tuple[int, int, int], version: int = 0
) -> ChunkData:
    """Generate one chunk. The worker-process entry point.

    :param spec: the planet, resolved (sea bias set)
    :param key: (level, ix, iy)
    :param version: spec version, carried through to the result
    :returns: the chunk
    """
    start = time.perf_counter()
    planet = planet_for(spec)
    level = key[0]
    spacing = chunk_spacing(level)
    size = chunk_size(level)
    x0, y0 = chunk_origin(key)

    # A border wide enough for the biome slope (measured at a fixed scale) and
    # the normals (one sample).
    step = max(1, round(biomes.SLOPE_SCALE / spacing))
    border = step
    n = VERTS + 2 * border
    sample = planet.sample_grid(
        x0 - border * spacing, y0 - border * spacing, spacing, n
    )
    full = sample.elevation
    inner = (slice(border, border + VERTS), slice(border, border + VERTS))
    elevation = full[inner]

    # Normals by central differences, from the border samples at the edges.
    b = border
    dz_dx = (
        full[b : b + VERTS, b + 1 : b + VERTS + 1]
        - full[b : b + VERTS, b - 1 : b + VERTS - 1]
    ) / (2.0 * spacing)
    dz_dy = (
        full[b + 1 : b + VERTS + 1, b : b + VERTS]
        - full[b - 1 : b + VERTS - 1, b : b + VERTS]
    ) / (2.0 * spacing)
    normals = np.dstack([-dz_dx, -dz_dy, np.ones_like(dz_dx)])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    slope = biomes.slope_degrees(full, spacing, step)[inner]
    w = biomes.weights(
        sample.temperature[inner],
        sample.humidity[inner],
        elevation,
        slope,
        spec.has_ocean,
    )
    for paint in spec.paints:
        w = paint.apply(
            sample.dirs.reshape(n, n, 3)[inner].reshape(-1, 3), w, spec.radius
        )
    w_bytes = np.clip(np.rint(w * 255.0), 0, 255).astype(np.uint8)

    morph = parent_interpolation(elevation)
    geometric_error = float(np.max(np.abs(elevation - morph)))
    h_min, h_max = float(elevation.min()), float(elevation.max())
    skirt_depth = min(4.0 * spacing + 4.0 * geometric_error, 0.5 * size)

    vertices = _pack_vertices(
        spacing, size, elevation, normals, morph, w_bytes, skirt_depth
    )
    return ChunkData(
        key=key,
        version=version,
        vertex_bytes=vertices.tobytes(),
        elevation=elevation.astype(np.float32),
        weights=w_bytes,
        h_min=h_min,
        h_max=h_max,
        skirt_depth=skirt_depth,
        geometric_error=geometric_error,
        gen_ms=(time.perf_counter() - start) * 1e3,
    )


def parent_interpolation(elevation: np.ndarray) -> np.ndarray:
    """The grid as its parent chunk would show it (every other sample, linear).

    Used for the geometric error, and stored as ``morph_z`` for a later
    geomorphing pass.

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
    spacing: float,
    size: float,
    elevation: np.ndarray,
    normals: np.ndarray,
    morph: np.ndarray,
    w_bytes: np.ndarray,
    skirt_depth: float,
) -> np.ndarray:
    """:returns: grid then skirt vertices as a :data:`VERTEX_DTYPE` array"""
    ring = skirt_ring()
    count = VERTS * VERTS
    out = np.zeros(count + len(ring), dtype=VERTEX_DTYPE)
    local = spacing * np.arange(VERTS) - size / 2.0  # relative to the centre
    gx, gy = np.meshgrid(local, local)
    pos = np.column_stack([gx.ravel(), gy.ravel(), elevation.ravel()])
    out["vertex"][:count] = pos
    out["normal"][:count] = normals.reshape(-1, 3)
    out["morph_z"][:count] = morph.ravel()
    out["biome_a"][:count] = w_bytes[:, 0:4]
    out["biome_b"][:count, 0:2] = w_bytes[:, 4:6]

    skirt = out[count:]
    skirt[:] = out[ring]
    skirt["vertex"][:, 2] -= skirt_depth
    skirt["morph_z"] -= skirt_depth
    skirt["biome_b"][:, 3] = 255  # flag, for the skirt debug view
    return out
