"""
Trees and bushes on the terrain: dense trees in forest, sparse trees and
bushes on grassland and among rocks, very sparse dry, scorched bushes in
desert.

Nothing is modelled or instanced. Each cube face carries a square grid of
cells (:data:`CELL`, in the face's texture coordinates); a cell holds one
plant at most, chosen by an integer hash of the cell and by the biome
weights there (``shaders/vegetation_cells.glsl``; :func:`plants` is its
numpy twin, bit for bit on the hash). A plant is an ellipsoid crown, a
tree's on a trunk.

Three looks, by distance (``docs/source/vegetation.md``):

- **Near** (within ``VegetationSettings.range``): the canopy's shell. Each
  drawn chunk with plants, fine enough (vertex spacing at most
  :data:`MAX_SPACING`) and in range, draws its own vertices a second time
  (:class:`Canopy`): its grid lifted by the tallest plant its biomes grow,
  its skirt on its edge's ground, a closed volume. Its fragment stage
  (``canopy.frag``) walks the view ray through the cells inside, testing
  each cell's crown exactly, and writes the hit's colour and depth. The
  ground under the march, and the biome weights, come from the chunk's own
  heights (a *canopy layer* per chunk: :data:`LAYER_ROWS` rows of
  :data:`LAYER_SIDE`, in pages of :data:`PAGE_LAYERS`).
- **Shadows** (within ``shadow_range``): the ground's own pixels test the
  crowns between them and the sun (``vegetation_far.glsl``).
- **Far**: the canopy as a statistical cover in the ground's shading
  (``vegetation_far.glsl``): the crowns' share of a pixel and their shadow,
  by the view's and the sun's angles.
"""

from __future__ import annotations

import functools
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from panda3d.core import (
    BitMask32,
    BoundingBox,
    CullBinManager,
    CullFaceAttrib,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    NodePath,
    Point3,
    SamplerState,
    Shader,
    Texture,
)

from world_builder.quality import QualityLevel, QualitySettings, knob
from world_builder.terrain.biomes import DESERT, FOREST, GRASS, ROCK
from world_builder.terrain.chunkgen import skirt_ring
from world_builder.terrain.cubesphere import QUADS, UV_PERIOD, VERTS, face_dirs

LOGGER = logging.getLogger(__name__)
SHADER_PATH = Path(__file__).parent / "shaders"

# ── the plants (vegetation_cells.glsl's constants) ──────────────────────────

#: A cell's side, uv metres; a whole number of cells per UV_PERIOD.
CELL = 6.4
CELLS_PER_PERIOD = 640
NONE, TREE, BUSH, DRY = range(4)
TREE_FOREST = 0.85
TREE_GRASS = 0.04
BUSH_GRASS = 0.20
DRY_DESERT = 0.025
TREE_ROCK = 0.01
BUSH_ROCK = 0.03
TREE_HEIGHT = (9.0, 21.0)
TREE_RADIUS = (2.0, 3.0)
BUSH_HEIGHT = (0.8, 2.0)
BUSH_RADIUS = (0.6, 1.4)
DRY_HEIGHT = (0.5, 1.2)
DRY_RADIUS = (0.5, 1.0)
REACH = 0.5
HULL_TREE = 23.0
HULL_BUSH = 2.4

# ── the shell ───────────────────────────────────────────────────────────────

#: Chunks with a vertex spacing above this draw no shell (their plants are
#: the far look's), metres.
MAX_SPACING = 8.0
#: A canopy layer: the chunk's grid (VERTS x VERTS: ground and weights), a
#: row of its frame, then the grid again (the shell's lift per vertex).
LAYER_SIDE = VERTS
LAYER_FRAME_ROW = VERTS
LAYER_LIFT_ROW = VERTS + 1
LAYER_ROWS = 2 * VERTS + 1
#: The shell's lift over the tallest crown near a vertex, metres.
LIFT_MARGIN = 1.0
#: Layers per page (one texture array, uploaded whole on any change: ~1 MB),
#: and the most pages: 128 chunks with a shell at once.
PAGE_LAYERS = 16
MAX_PAGES = 8
#: The camera within this many metres of a chunk's shell (its box) draws
#: its inside too (the march from the camera).
INSIDE_MARGIN = 40.0
#: The most shells made per LOD pass (~0.25 ms each: their layer, node and
#: page upload), nearest first; the rest wait for the next pass, asked for
#: at once.
ADDS_PER_PASS = 8
#: Nothing grows on ground below this with a sea, metres.
SEA_FLOOR = 0.5
#: The shells' cull bin: front to back, before the ground's (19): the
#: ground hidden by the trees is not shaded (the early depth test).
CANOPY_BIN = "canopy"
CANOPY_BIN_SORT = 18
#: A chunk's box marking it as growing nothing.
_NO_PLANTS = np.zeros(0)


assert CELLS_PER_PERIOD * CELL == UV_PERIOD


@dataclass(frozen=True)
class VegetationSettings(QualitySettings):
    """What the vegetation costs (:mod:`world_builder.quality`). All live.

    :ivar enabled: trees and bushes at all (the far look too)
    :ivar range: the shell draws plants out to here, metres; 0 = none, the
        far look only
    :ivar shadow_range: the plants' shadows on the ground out to here,
        metres; 0 = none
    :ivar detail_range: trunks and leafy crowns out to here, metres
    :ivar max_steps: the most cells a view ray walks in a chunk's shell
    """

    enabled: bool = knob(True, "Vegetation", help="Trees and bushes, near and far.")
    range: float = knob(
        400.0,
        "Tree distance",
        unit="m",
        range=(0.0, 4000.0),
        help="How far trees and bushes are drawn one by one (beyond: their "
        "cover's colour and shadow on the ground).",
    )
    shadow_range: float = knob(
        250.0,
        "Tree shadow distance",
        unit="m",
        range=(0.0, 1000.0),
        help="How far the plants' shadows on the ground reach.",
    )
    detail_range: float = knob(
        150.0,
        "Tree detail distance",
        unit="m",
        range=(0.0, 1000.0),
        advanced=True,
        help="How far trunks and leafy crowns are drawn.",
    )
    max_steps: int = knob(
        48,
        "Tree march steps",
        range=(4, 160),
        advanced=True,
        help="The most cells a view ray walks among the plants of a chunk "
        "(grazing views walk the most).",
    )

    LABEL: ClassVar = "Vegetation"
    PRESETS: ClassVar = {
        QualityLevel.LOW: {
            "range": 0.0,
            "shadow_range": 0.0,
            "detail_range": 0.0,
            "max_steps": 16,
        },
        QualityLevel.MID: {
            "range": 250.0,
            "shadow_range": 120.0,
            "detail_range": 80.0,
            "max_steps": 32,
        },
        QualityLevel.ULTRA: {
            "range": 900.0,
            "shadow_range": 400.0,
            "detail_range": 250.0,
            "max_steps": 72,
        },
    }


# ── the placement, in numpy (vegetation_cells.glsl's twin) ──────────────────


def _hash(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
    """pcg3d on uint32 arrays (wrapping), as vegetation_cells.glsl's vegHash."""
    x = x.astype(np.uint32) * np.uint32(1664525) + np.uint32(1013904223)
    y = y.astype(np.uint32) * np.uint32(1664525) + np.uint32(1013904223)
    z = z.astype(np.uint32) * np.uint32(1664525) + np.uint32(1013904223)
    x = x + y * z
    y = y + z * x
    z = z + x * y
    x ^= x >> np.uint32(16)
    y ^= y >> np.uint32(16)
    z ^= z >> np.uint32(16)
    x = x + y * z
    y = y + z * x
    z = z + x * y
    return x, y, z


def _unit(h: np.ndarray) -> np.ndarray:
    return (h >> np.uint32(8)).astype(np.float64) / 16777216.0


def crown_area(radius: tuple[float, float]) -> float:
    """:returns: a kind's mean crown area over a cell's (r uniform)"""
    lo, hi = radius
    return math.pi * (hi**3 - lo**3) / (3.0 * (hi - lo)) / (CELL * CELL)


def plants(
    face: int,
    cu: np.ndarray,
    cv: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """The plants of cells (``plantAt``'s twin).

    :param face: the cube face
    :param cu: (N,) global cell indices along u (the period's index times
        :data:`CELLS_PER_PERIOD` plus the cell in it)
    :param cv: (N,) along v
    :param weights: (N, 4) forest, grass, desert and rock weights in [0, 1]
    :param seed: the planet's seed
    :returns: ``kind`` (N,) int (:data:`NONE`, :data:`TREE`, :data:`BUSH`,
        :data:`DRY`); ``height``, ``radius`` (N,) metres; ``offset`` (N, 2)
        the centre from the cell's, uv metres; ``tint`` (N,)
    """
    cu = np.asarray(cu, dtype=np.int64)
    cv = np.asarray(cv, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 4)
    zf = np.full(cu.shape, face + 8 * seed, dtype=np.int64)
    h1 = _hash(cu & 0xFFFFFFFF, cv & 0xFFFFFFFF, zf & 0xFFFFFFFF)
    h2 = _hash(
        h1[0] ^ np.uint32(0x9E3779B9),
        h1[1] ^ np.uint32(0x85EBCA6B),
        h1[2] ^ np.uint32(0xC2B2AE35),
    )
    r1 = [_unit(h) for h in h1]
    r2 = [_unit(h) for h in h2]
    p_tree = TREE_FOREST * w[:, 0] + TREE_GRASS * w[:, 1] + TREE_ROCK * w[:, 3]
    p_bush = BUSH_GRASS * w[:, 1] + BUSH_ROCK * w[:, 3]
    p_dry = DRY_DESERT * w[:, 2]
    u = r1[0]
    kind = np.select(
        [u < p_tree, u < p_tree + p_bush, u < p_tree + p_bush + p_dry],
        [TREE, BUSH, DRY],
        NONE,
    )
    heights = np.array([DRY_HEIGHT, TREE_HEIGHT, BUSH_HEIGHT, DRY_HEIGHT])[kind]
    radii = np.array([DRY_RADIUS, TREE_RADIUS, BUSH_RADIUS, DRY_RADIUS])[kind]
    height = heights[:, 0] + (heights[:, 1] - heights[:, 0]) * r1[1]
    radius = radii[:, 0] + (radii[:, 1] - radii[:, 0]) * r1[2]
    play = np.maximum(REACH * CELL - radius, 0.0)
    offset = (2.0 * np.stack([r2[0], r2[1]], axis=1) - 1.0) * play[:, None]
    return {
        "kind": kind,
        "height": height,
        "radius": radius,
        "offset": offset,
        "tint": r2[2],
    }


def coverage(weights: np.ndarray) -> np.ndarray:
    """:returns: (N,) the crowns' mean share of the ground seen from above
    where the weights (N, 4: forest, grass, desert, rock) are
    (``vegCoverage``'s)"""
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 4)
    trees = (TREE_FOREST * w[:, 0] + TREE_GRASS * w[:, 1] + TREE_ROCK * w[:, 3]) * (
        crown_area(TREE_RADIUS)
    )
    bushes = (BUSH_GRASS * w[:, 1] + BUSH_ROCK * w[:, 3]) * crown_area(
        BUSH_RADIUS
    ) + DRY_DESERT * w[:, 2] * crown_area(DRY_RADIUS)
    return trees + bushes


def chunk_weights(weights: np.ndarray) -> np.ndarray:
    """:returns: (n, 4) forest, grass, desert, rock in [0, 1] from a chunk's
    (n, 6) uint8 biome weights"""
    w = np.asarray(weights, dtype=np.float64) / 255.0
    return w[:, [FOREST, GRASS, DESERT, ROCK]]


def chunk_has_plants(weights: np.ndarray) -> bool:
    """:returns: whether a chunk's (n, 6) uint8 weights grow anything"""
    w = np.asarray(weights)
    return bool(
        (w[:, FOREST].astype(np.int32) + w[:, GRASS] + w[:, DESERT] + w[:, ROCK]).max()
        > 1
    )


# ── a chunk's canopy layer ──────────────────────────────────────────────────


def chunk_frame(grid: Any, key: tuple, centre: np.ndarray, radius: float) -> np.ndarray:
    """A chunk's tangent frame at its middle vertex O (sea level): the
    canopy layer's last row (``canopy_frag.glsl``).

    :returns: (6, 4) float32: E_u (metres per grid step along u, tangent;
        w the radius), E_v, F_u and F_v (the dual axes: grid steps per
        metre), O from the chunk's centre, the vertical at O
    """
    level, face, i, j = key
    half = QUADS // 2
    step = (1 << level) * 2.0 / grid.n0
    a = grid.param((i * QUADS + half) << level)
    b = grid.param((j * QUADS + half) << level)
    dirs = face_dirs(
        face,
        np.array([a, a + step, a - step, a, a]),
        np.array([b, b, b, b + step, b - step]),
    )
    up = dirs[0]
    origin = radius * up
    eu = 0.5 * radius * (dirs[1] - dirs[2])
    ev = 0.5 * radius * (dirs[3] - dirs[4])
    eu -= up * (eu @ up)
    ev -= up * (ev @ up)
    gram = np.array([[eu @ eu, eu @ ev], [ev @ eu, ev @ ev]])
    inv = np.linalg.inv(gram)
    fu = inv[0, 0] * eu + inv[0, 1] * ev
    fv = inv[1, 0] * eu + inv[1, 1] * ev
    out = np.zeros((6, 4), dtype=np.float64)
    out[0, :3], out[0, 3] = eu, radius
    out[1, :3] = ev
    out[2, :3] = fu
    out[3, :3] = fv
    out[4, :3] = origin - np.asarray(centre, dtype=np.float64)
    out[5, :3] = up
    return out.astype(np.float32)


def _bilinear(grid: np.ndarray, gu: np.ndarray, gv: np.ndarray) -> np.ndarray:
    """(rows v, columns u) sampled at fractional grid coordinates, clamped."""
    n = grid.shape[0] - 1
    gu = np.clip(gu, 0.0, n)
    gv = np.clip(gv, 0.0, n)
    u0 = np.minimum(np.floor(gu).astype(np.int64), n - 1)
    v0 = np.minimum(np.floor(gv).astype(np.int64), n - 1)
    fu, fv = gu - u0, gv - v0
    top = grid[v0, u0] * (1 - fu) + grid[v0, u0 + 1] * fu
    bottom = grid[v0 + 1, u0] * (1 - fu) + grid[v0 + 1, u0 + 1] * fu
    return top * (1 - fv) + bottom * fv


def chunk_plants(
    uv0: tuple[float, float],
    step: float,
    period: tuple[int, int],
    face: int,
    ground: np.ndarray,
    weights: np.ndarray,
    seed: int,
    sea_floor: float,
    margin: int = 1,
) -> dict[str, np.ndarray]:
    """Every plant a chunk's march can meet: its cells' and *margin* cells
    around (their crowns reach in), as ``canopy_frag.glsl`` places them.

    :param uv0: the chunk's first vertex's uv (wrapped), metres
    :param step: uv metres per grid step
    :param period: which UV_PERIOD the first vertex is in (u, v)
    :param ground: (VERTS, VERTS) float32 metres above sea level, rows v
    :param weights: (VERTS, VERTS, 4) float32 forest, grass, desert, rock as
        the canopy layer holds them
    :returns: :func:`plants`' fields for the cells with a plant, and ``g``
        (N, 2) its centre's grid coordinate, ``base`` (N,) its ground
    """
    span = QUADS * step
    lo = np.floor(np.asarray(uv0) / CELL).astype(np.int64) - margin
    hi = np.floor((np.asarray(uv0) + span) / CELL).astype(np.int64) + margin
    cu, cv = np.meshgrid(
        np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="xy"
    )
    cu, cv = cu.ravel(), cv.ravel()
    centre_u = ((cu + 0.5) * CELL - uv0[0]) / step
    centre_v = ((cv + 0.5) * CELL - uv0[1]) / step
    nu = np.clip(np.floor(centre_u + 0.5).astype(np.int64), 0, QUADS)
    nv = np.clip(np.floor(centre_v + 0.5).astype(np.int64), 0, QUADS)
    w = weights[nv, nu]
    p = plants(
        face,
        cu + period[0] * CELLS_PER_PERIOD,
        cv + period[1] * CELLS_PER_PERIOD,
        w,
        seed,
    )
    uv_u = (cu + 0.5) * CELL + p["offset"][:, 0]
    uv_v = (cv + 0.5) * CELL + p["offset"][:, 1]
    g = np.stack([(uv_u - uv0[0]) / step, (uv_v - uv0[1]) / step], axis=1)
    base = _bilinear(ground, g[:, 0], g[:, 1])
    grows = (p["kind"] != NONE) & (base >= sea_floor)
    out = {name: value[grows] for name, value in p.items()}
    out["g"] = g[grows]
    out["base"] = base[grows]
    return out


def shell_lift(
    found: dict[str, np.ndarray], ground: np.ndarray, step: float
) -> np.ndarray:
    """The shell's lift per vertex: over every crown whose footprint meets
    the triangles around the vertex (their tops linear between vertices),
    by :data:`LIFT_MARGIN`; 0 where none (the shell then lies on the
    ground and draws nothing).

    :param found: :func:`chunk_plants`' plants
    :param ground: (VERTS, VERTS) metres, rows v
    :param step: uv metres per grid step
    :returns: (VERTS, VERTS) float32 metres, rows v
    """
    lift = np.full(VERTS * VERTS, -np.inf)
    if len(found["base"]):
        top = found["base"] + found["height"]
        # The vertices within the crown's radius plus a triangle of it.
        reach = found["radius"] / step + 1.0
        span = int(np.ceil(reach.max()))
        offsets = np.arange(-span, span + 1)
        du, dv = np.meshgrid(offsets, offsets, indexing="xy")
        du, dv = du.ravel(), dv.ravel()
        gu = np.round(found["g"][:, 0]).astype(np.int64)
        gv = np.round(found["g"][:, 1]).astype(np.int64)
        u = gu[:, None] + du[None]
        v = gv[:, None] + dv[None]
        near = (
            (np.abs(u - found["g"][:, :1]) <= reach[:, None])
            & (np.abs(v - found["g"][:, 1:]) <= reach[:, None])
            & (u >= 0)
            & (u <= QUADS)
            & (v >= 0)
            & (v <= QUADS)
        )
        index = (v * VERTS + u)[near]
        np.maximum.at(lift, index, np.broadcast_to(top[:, None], u.shape)[near])
    lift = lift.reshape(VERTS, VERTS) - ground + LIFT_MARGIN
    return np.where(np.isfinite(lift), np.maximum(lift, LIFT_MARGIN), 0.0).astype(
        np.float32
    )


def chunk_uv(
    grid: Any, key: tuple
) -> tuple[tuple[float, float], float, tuple[int, int]]:
    """:returns: a chunk's first vertex's uv (wrapped, as the chunk table
    holds it), uv metres per grid step, and which UV_PERIOD the first vertex
    is in (u, v)"""
    level, _, i, j = key
    unit = grid.uv_unit
    period = (
        math.floor(((i * QUADS) << level) * unit / UV_PERIOD),
        math.floor(((j * QUADS) << level) * unit / UV_PERIOD),
    )
    return grid.uv_origin(key), (1 << level) * unit, period


def layer_weights(weights: np.ndarray) -> np.ndarray:
    """:returns: (VERTS, VERTS, 4) float32 forest, grass, desert, rock from a
    chunk's (n, 6) uint8 weights, as the canopy layer gives them the GPU"""
    w = np.asarray(weights).reshape(VERTS, VERTS, -1).astype(np.float32)
    return w[..., [FOREST, GRASS, DESERT, ROCK]] / np.float32(255.0)


def chunk_lift(
    grid: Any, key: tuple, elevation, weights, spec: Any
) -> np.ndarray | None:
    """The shell's lift of a chunk (:func:`shell_lift`), in the worker that
    generates it; None for a chunk too coarse to draw its plants
    (:data:`MAX_SPACING`) or growing none."""
    if grid.spacing(key[0]) > MAX_SPACING or not chunk_has_plants(weights):
        return None
    uv0, step, period = chunk_uv(grid, key)
    ground = np.asarray(elevation, dtype=np.float32).reshape(VERTS, VERTS)
    sea_floor = SEA_FLOOR if spec.has_ocean else -1.0e9
    found = chunk_plants(
        uv0,
        step,
        period,
        key[1],
        ground,
        layer_weights(weights),
        int(spec.seed) & 0xFFFF,
        sea_floor,
    )
    if not len(found["base"]):
        return None
    return shell_lift(found, ground, step)


def canopy_layer(
    grid: Any,
    key: tuple,
    centre: np.ndarray,
    radius: float,
    elevation,
    weights,
    lift: np.ndarray,
) -> np.ndarray:
    """A chunk's canopy layer (``uCanopy``): (LAYER_ROWS, LAYER_SIDE, 4)
    float32, rows along v. Rows 0-64 per vertex: the ground (metres above
    sea level), the forest and grass weights, and desert + 256 rock (their
    bytes); row 65: :func:`chunk_frame` in its first six texels; rows
    66-130 per vertex: the shell's *lift* (:func:`chunk_lift`, made with the
    chunk), then per quad the lowest ground and the shell's highest top at
    its corners (the march's exit test in one fetch)."""
    layer = np.zeros((LAYER_ROWS, LAYER_SIDE, 4), dtype=np.float32)
    ground = np.asarray(elevation, dtype=np.float32).reshape(VERTS, VERTS)
    layer[:VERTS, :, 0] = ground
    w = np.asarray(weights).reshape(VERTS, VERTS, -1).astype(np.float32)
    layer[:VERTS, :, 1] = w[..., FOREST] / np.float32(255.0)
    layer[:VERTS, :, 2] = w[..., GRASS] / np.float32(255.0)
    layer[:VERTS, :, 3] = w[..., DESERT] + 256.0 * w[..., ROCK]
    layer[LAYER_FRAME_ROW, :6] = chunk_frame(grid, key, centre, radius)
    layer[LAYER_LIFT_ROW:, :, 0] = lift
    # Per quad (its first corner's texel): the lowest ground and the shell's
    # highest top at its corners. The drawn (triangulated) shell never
    # leaves them, so the march's exit test, one fetch, never takes a ray
    # out of the shell early (a crown cut along a thin line).
    top = np.where(lift > 0.0, ground + lift, -1.0e9)
    low = np.minimum(
        np.minimum(ground[:-1, :-1], ground[1:, :-1]),
        np.minimum(ground[:-1, 1:], ground[1:, 1:]),
    )
    high = np.maximum(
        np.maximum(top[:-1, :-1], top[1:, :-1]), np.maximum(top[:-1, 1:], top[1:, 1:])
    )
    layer[LAYER_LIFT_ROW:, :, 1] = ground
    layer[LAYER_LIFT_ROW:, :, 2] = top
    layer[LAYER_LIFT_ROW : LAYER_LIFT_ROW + QUADS, :QUADS, 1] = low
    layer[LAYER_LIFT_ROW : LAYER_LIFT_ROW + QUADS, :QUADS, 2] = high
    return layer


@functools.cache
def shell_triangles() -> GeomTriangles:
    """:returns: the shell's index buffer (every chunk's): the grid (the
    lift hugs the crowns vertex by vertex), then the skirt as outward walls
    only (one winding: the shell is closed and drawn front faces out)"""
    stride = 1
    r, c = np.meshgrid(
        np.arange(0, QUADS, stride), np.arange(0, QUADS, stride), indexing="ij"
    )
    v00 = (r * VERTS + c).ravel()
    v01 = v00 + stride
    v10 = v00 + stride * VERTS
    v11 = v10 + stride
    top = np.stack([v00, v01, v11, v00, v11, v10], axis=1).ravel()
    ring = skirt_ring()[::stride]
    top_a, top_b = ring, np.roll(ring, -1)
    bottom_a = VERTS * VERTS + np.arange(0, 4 * QUADS, stride)
    bottom_b = np.roll(bottom_a, -1)
    walls = np.stack(
        [top_a, bottom_a, bottom_b, top_a, bottom_b, top_b], axis=1
    ).ravel()
    indices = np.concatenate([top, walls]).astype(np.uint16)
    tris = GeomTriangles(Geom.UH_static)
    tris.set_index_type(GeomEnums.NT_uint16)
    tris.add_next_vertices(indices.size)
    memoryview(tris.modify_vertices()).cast("B")[:] = indices.tobytes()
    return tris


@functools.cache
def _ground_shader() -> Shader:
    """:returns: the ground's shader with the plants' shadows (the chunks
    with a shell): ``terrain_veg.frag``"""
    return Shader.load(
        Shader.SL_GLSL,
        vertex=SHADER_PATH / "terrain.vert",
        fragment=SHADER_PATH / "terrain_veg.frag",
    )


@functools.cache
def _shader(inside: bool) -> Shader:
    """:returns: the shell's shader: the march from where the view ray enters
    it, or (*inside*) from the camera within it"""
    if inside:
        return Shader.load(
            Shader.SL_GLSL,
            vertex=SHADER_PATH / "canopy_inside.vert",
            fragment=SHADER_PATH / "canopy_inside.frag",
        )
    return Shader.load(
        Shader.SL_GLSL,
        vertex=SHADER_PATH / "canopy.vert",
        fragment=SHADER_PATH / "canopy.frag",
    )


def bind_defaults(node: NodePath) -> None:
    """The vegetation's inputs, off (a terrain without a :class:`Canopy`)."""
    node.set_shader_input("uVegOn", 0)
    node.set_shader_input("uVegRange", 0.0)
    node.set_shader_input("uVegShadowRange", 0.0)
    node.set_shader_input("uVegDetail", 0.0)
    node.set_shader_input("uVegSteps", 0)
    node.set_shader_input("uVegSeed", 0)
    node.set_shader_input("uVegSeaFloor", -1.0e9)
    node.set_shader_input("uVegLooks", 2)


class Canopy:
    """One terrain's vegetation: its inputs, and the shells of the drawn
    chunks near the camera (each with a canopy layer).

    Made by its :class:`~world_builder.terrain.terrain.Terrain`
    (``terrain.canopy``), which calls :meth:`update` after each LOD pass and
    :meth:`drop` for a chunk it lets go.
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = VegetationSettings

    def __init__(self, terrain: Any, settings: VegetationSettings | None = None):
        self._terrain = terrain
        self._settings = settings or VegetationSettings()
        self._pages: list[Texture] = []
        self._free: list[int] = []
        self._layers: dict[tuple, int] = {}  # chunk key -> layer id
        self._shells: dict[tuple, NodePath] = {}
        self._insides: dict[tuple, NodePath] = {}
        self._shadowed: dict[tuple, bool] = {}  # chunks given terrain_veg.frag
        # chunk key -> its box (planet frame: low, high), or _NO_PLANTS
        self._info: dict[tuple, np.ndarray] = {}
        self._dirty_pages: set[int] = set()
        self.waiting = False
        self.stats = {"shells": 0, "inside": 0, "added": 0, "update_ms": 0.0}
        bind_defaults(terrain.root)
        manager = CullBinManager.get_global_ptr()
        if manager.find_bin(CANOPY_BIN) == -1:
            manager.add_bin(
                CANOPY_BIN, CullBinManager.BT_front_to_back, CANOPY_BIN_SORT
            )
        self.reset()

    # ── settings ───────────────────────────────────────────────────────────

    @property
    def settings(self) -> VegetationSettings:
        """What the vegetation costs (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: VegetationSettings) -> None:
        """Change what the vegetation costs (live)."""
        if settings == self._settings:
            return
        self._settings = settings
        self._apply()
        self._terrain._last_select_pos = None  # a pass soon: the shells' set

    @property
    def seed(self) -> int:
        """The plants' seed (the planet's)."""
        return int(self._terrain.spec.seed) & 0xFFFF

    @property
    def active(self) -> bool:
        """Whether this planet grows anything and the vegetation is on."""
        return bool(self._settings.enabled and self._terrain.spec.has_vegetation)

    def _apply(self) -> None:
        s, root = self._settings, self._terrain.root
        on = self.active
        root.set_shader_input("uVegOn", int(on))
        root.set_shader_input("uVegRange", float(s.range) if on else 0.0)
        root.set_shader_input("uVegShadowRange", float(s.shadow_range) if on else 0.0)
        root.set_shader_input("uVegDetail", float(s.detail_range))
        root.set_shader_input("uVegSteps", int(s.max_steps))
        root.set_shader_input("uVegSeed", self.seed)
        root.set_shader_input(
            "uVegSeaFloor", SEA_FLOOR if self._terrain.spec.has_ocean else -1.0e9
        )
        if not on or s.range <= 0.0:
            for key in list(self._layers):
                self.drop(key)

    def reset(self) -> None:
        """Forget every shell (a new planet: :meth:`Terrain.set_spec`)."""
        for key in list(self._layers):
            self.drop(key)
        self._info.clear()
        grid = self._terrain.grid
        self._max_level = max(
            [
                level
                for level in range(grid.max_level + 1)
                if grid.spacing(level) <= MAX_SPACING
            ],
            default=-1,
        )
        self._apply()

    # ── per LOD pass ───────────────────────────────────────────────────────

    def update(self, draw: set, camera: np.ndarray) -> None:
        """The shells of the drawn chunks that grow plants within range
        (after each LOD pass: ~0.1 ms over a few hundred drawn chunks).

        :param draw: the drawn chunk keys
        :param camera: the camera, planet frame
        """
        start = time.perf_counter()
        added = 0
        terrain = self._terrain
        s = self._settings
        keys: list[tuple] = []
        boxes: list[np.ndarray] = []
        if self.active and s.range > 0.0:
            top = self._max_level
            info = self._info
            store, nodes = terrain.store, terrain.nodes
            for key in draw:
                if key[0] > top:
                    continue
                box = info.get(key)
                if box is None:
                    chunk = store.get(key)
                    if chunk is None or key not in nodes:
                        continue
                    if getattr(chunk, "canopy_lift", None) is None:
                        box = info[key] = _NO_PLANTS
                    else:
                        box = info[key] = np.concatenate(
                            [
                                chunk.centre + chunk.aabb_min,
                                chunk.centre + chunk.aabb_max,
                            ]
                        )
                if box is not _NO_PLANTS:
                    keys.append(key)
                    boxes.append(box)
        keep: list[tuple] = []
        inside: set[tuple] = set()
        if keys:
            b = np.array(boxes)
            gap = np.maximum(np.maximum(b[:, :3] - camera, camera - b[:, 3:]), 0.0)
            distance = np.sqrt((gap * gap).sum(axis=1))
            order = np.argsort(distance)
            capacity = MAX_PAGES * PAGE_LAYERS
            order = order[distance[order] <= s.range + HULL_TREE][:capacity]
            keep = [keys[i] for i in order]
            near = (b[:, :3] - INSIDE_MARGIN <= camera).all(axis=1) & (
                camera <= b[:, 3:] + INSIDE_MARGIN + HULL_TREE
            ).all(axis=1)
            inside = {keys[i] for i in order if near[i]}
        wanted = set(keep)
        for key in [k for k in self._layers if k not in wanted]:
            self.drop(key)
        slots = []
        waiting = False
        for key in keep:  # nearest first
            if key not in self._layers:
                if added >= ADDS_PER_PASS:
                    waiting = True  # the rest at the next pass, soon
                    break
                slot = self._add(key)
                if slot is not None:
                    slots.append(slot)
                    added += 1
        #: Shells in range wait to be made (ADDS_PER_PASS): a pass soon
        self.waiting = waiting
        if waiting:
            terrain._last_select_pos = None
        for key in [k for k in self._insides if k not in inside]:
            self._insides.pop(key).remove_node()
        for key in inside:
            if key in self._layers and key not in self._insides:
                page = self._pages[self._layers[key] // PAGE_LAYERS]
                chunk = terrain.store[key]
                self._insides[key] = self._shell_node(key, chunk, page, inside=True)
        if slots:
            terrain._write_chunk_table(slots)
        for page in self._dirty_pages:
            self._pages[page].modify_ram_image()  # uploaded whole, once
        self._dirty_pages.clear()
        if len(self._info) > 4 * len(terrain.store) + 64:
            self._info = {k: v for k, v in self._info.items() if k in terrain.store}
        self.stats["shells"] = len(self._shells)
        self.stats["inside"] = len(self._insides)
        self.stats["added"] = added
        self.stats["update_ms"] = (time.perf_counter() - start) * 1e3

    def drop(self, key: tuple) -> None:
        """Forget a chunk's shell (its node let go, or out of range)."""
        layer = self._layers.pop(key, None)
        if layer is None:
            return
        self._free.append(layer)
        for nodes in (self._shells, self._insides):
            node = nodes.pop(key, None)
            if node is not None:
                node.remove_node()
        chunk_node = self._terrain.nodes.get(key)
        if chunk_node is not None and self._shadowed.pop(key, False):
            chunk_node.clear_shader()
        slot = self._terrain._slots.get(key)
        if slot is not None:
            self._terrain._chunk_values[3, slot, 2:] = (0.0, 0.0)
            self._terrain._write_chunk_table([slot])

    # ── internals ──────────────────────────────────────────────────────────

    def _page(self, index: int) -> Texture:
        while len(self._pages) <= index:
            page = Texture(f"canopy{len(self._pages)}")
            page.setup_2d_texture_array(
                LAYER_SIDE, LAYER_ROWS, PAGE_LAYERS, Texture.T_float, Texture.F_rgba32
            )
            page.set_compression(Texture.CM_off)
            page.set_minfilter(SamplerState.FT_linear)
            page.set_magfilter(SamplerState.FT_linear)
            page.set_wrap_u(SamplerState.WM_clamp)
            page.set_wrap_v(SamplerState.WM_clamp)
            page.set_ram_image(
                np.zeros(
                    PAGE_LAYERS * LAYER_ROWS * LAYER_SIDE * 4, np.float32
                ).tobytes()
            )
            self._pages.append(page)
            base = (len(self._pages) - 1) * PAGE_LAYERS
            self._free.extend(range(base + PAGE_LAYERS - 1, base - 1, -1))
        return self._pages[index]

    def _take_layer(self) -> int | None:
        if not self._free:
            if len(self._pages) >= MAX_PAGES:
                return None
            self._page(len(self._pages))
        # The lowest first: fewer pages in use, fewer states to draw.
        self._free.sort(reverse=True)
        return self._free.pop()

    def _add(self, key: tuple) -> int | None:
        terrain = self._terrain
        chunk = terrain.store[key]
        slot = terrain._slots.get(key)
        if slot is None:
            return None
        layer = self._take_layer()
        if layer is None:
            return None
        page_index, index = divmod(layer, PAGE_LAYERS)
        page = self._page(page_index)
        data = canopy_layer(
            terrain.grid,
            key,
            chunk.centre,
            terrain.radius,
            chunk.elevation,
            chunk.weights,
            chunk.canopy_lift,
        )
        ram = np.frombuffer(memoryview(page.modify_ram_image()), np.float32).reshape(
            PAGE_LAYERS, LAYER_ROWS, LAYER_SIDE, 4
        )
        ram[index] = data[..., [2, 1, 0, 3]]  # Panda keeps RGBA as BGRA
        self._dirty_pages.add(page_index)
        self._layers[key] = layer
        terrain._chunk_values[3, slot, 2:] = (float(index), 1.0)
        self._shells[key] = self._shell_node(key, chunk, page, inside=False)
        node = terrain.nodes[key]
        if not node.get_python_tag("wet"):
            # The plants' shadows on its ground (the chunks with water keep
            # the sea's shader: none there).
            node.set_shader(_ground_shader())
            self._shadowed[key] = True
        return slot

    def _shell_node(
        self, key: tuple, chunk: Any, page: Texture, inside: bool
    ) -> NodePath:
        parent = self._terrain.nodes[key]
        vdata = parent.find("fine").node().get_geom(0).get_vertex_data()
        geom = Geom(vdata)
        geom.add_primitive(shell_triangles())
        lift = HULL_TREE + 1.0
        bounds = BoundingBox(
            Point3(*(chunk.aabb_min - lift)), Point3(*(chunk.aabb_max + lift))
        )
        geom.set_bounds(bounds)
        mesh = GeomNode("canopy_inside" if inside else "canopy")
        mesh.add_geom(geom)
        mesh.set_bounds(bounds)
        mesh.set_final(True)
        mesh.set_into_collide_mask(BitMask32.all_off())
        node = parent.attach_new_node(mesh)
        node.hide(BitMask32.all_on())
        node.show(_MAIN_CAMERAS)
        node.set_shader(_shader(inside), 10)
        node.set_shader_input("uCanopy", page)
        # Before the ground (its bin, sort 19), front to back: the ground
        # under the trees is then hidden before it is shaded.
        node.set_bin(CANOPY_BIN, 0, 1)
        if inside:
            node.set_attrib(CullFaceAttrib.make_reverse())
        return node


def _main_cameras() -> BitMask32:
    """The camera bits that draw the shells: all but the ocean reflection's,
    the clouds' passes' and the casters' map's."""
    from world_builder.depth import (
        CASTER_BIT,
        HIDE_FROM_CLOUD_PASS,
        REFLECTION_CAMERA_MASK,
    )

    mask = BitMask32.all_on()
    mask &= ~REFLECTION_CAMERA_MASK
    mask &= ~HIDE_FROM_CLOUD_PASS
    mask &= ~BitMask32.bit(CASTER_BIT)
    return mask


_MAIN_CAMERAS = _main_cameras()
