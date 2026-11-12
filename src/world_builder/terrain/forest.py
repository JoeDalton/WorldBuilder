"""
Where the trees, bushes and rocks stand (the tree assets of
:mod:`world_builder.datafiles`), chunk by chunk, in the terrain's workers:
Serious Sam 4's jittered grid and rules (VEGETATION_PLAN.md, part V1;
[SS4 s.22, s.29]).

**The grids.** Each cube face carries two square grids of cells in its
texture coordinates (uv metres; a whole number of cells per UV_PERIOD, so
cells match across the wrap): the trees' (:data:`TREE_CELL`) and the small
plants' and rocks' (:data:`SMALL_CELL`). Their sides derive from the baked
crown sizes (``trees/crowns.json``: :func:`crowns`), :data:`CELL_CROWNS`
mean crown diameters of the groups on them, never set by hand. A cell holds
one sample at most, by an integer hash of its global index (the same on the
CPU and, later, the GPU): kept or not, its group, its offset within the
cell, its turn, scale and tint.

**The rules.** Per group (:data:`SPECIES`), MSFS-style: instances per
hectare in each biome, spawn ratios among its assets, and ramps on the
slope, the convexity and the altitude [SS4 s.22]. They give each group's
probability per cell at every vertex (:func:`probabilities`), scaled by the
vegetation density (procedural, painted over by levels' density paints:
``chunkgen``) and, per sample, thinned by a clumping noise (groves and
clearings).

**What a chunk gets** (:func:`place`):

- its plants, as a list (:data:`COLUMNS`): the samples whose point lies in
  its uv square (neighbours of one level share none), on its own ground.
  Coarser chunks keep a share of theirs, the bigger (:func:`thinning`); past
  :data:`MAX_SPACING` of vertex spacing none (the far forest, part V2,
  stands for them). Small plants only grow on chunks up to
  :data:`SMALL_SPACING`: they are drawn near the camera only.
- per vertex, the far forest's data (``chunkgen`` packs it into the vertex,
  :data:`FOREST_BYTES`): the trees' cover (the crowns' summed area per
  ground area: the actual trees' on chunks fine enough to place them all,
  else its expected value) and each far group's tree density (its
  expected share of the trees' cells).

Pure numpy: it runs in the worker processes (``chunkgen.generate_chunk``).
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from world_builder.datafiles import TREES_PATH
from world_builder.terrain.biomes import DESERT, FOREST, GRASS, ROCK, SNOW
from world_builder.terrain.cubesphere import QUADS, UV_PERIOD, VERTS

#: The baked crowns (scripts/bake_tree_impostors.py): per asset its crown's
#: area seen from above (m^2) and its height (m).
CROWNS_PATH = TREES_PATH / "crowns.json"
#: A grid's cell side, in mean crown diameters of the groups growing on it
#: (weighted by their densities in their main biome): the trees' cells hold
#: about a crown each; the small plants', sparse, several.
CELL_CROWNS = {"tree": 1.0, "small": 2.5}
#: How far a sample may sit from its cell's centre, in cells (0.5: anywhere).
JITTER = 0.4
#: Chunks finer than this (vertex spacing, metres) keep every tree; each
#: doubling keeps half as many, a factor THINNING_GROWTH bigger, up to
#: MAX_SPACING; coarser chunks grow none.
FULL_SPACING = 8.0
MAX_SPACING = 32.0
THINNING_GROWTH = 1.25
#: Small plants and rocks grow on chunks up to this vertex spacing only
#: (they are drawn within the meshes' range only).
SMALL_SPACING = 4.0
#: The clumping noise's lattice, uv metres, and how much it thins the
#: groups (0 none, 1 clearings wholly bare, times the group's ``clumped``).
CLUMP_SCALE = 90.0
CLUMP_STRENGTH = 0.75
#: The vegetation density's procedural field (chunkgen): fbm of this first
#: wavelength (metres) and octaves, through a smoothstep: mostly dense, with
#: wide thinner stretches and clearings.
DENSITY_WAVELENGTH = 2400.0
DENSITY_OCTAVES = 3
DENSITY_RAMP = (-0.6, 0.15)
DENSITY_FLOOR = 0.15
#: Nothing grows on ground below this with a sea, metres.
SEA_FLOOR = 0.5
#: Below the ground the base sits, metres (the drawn ground is a little off
#: the bilinear one).
SINK = 0.25
#: The cover byte's scale: cover (summed crown area per area) / COVER_MAX.
COVER_MAX = 2.0
#: Bytes per vertex of the far forest's data: cover, then the far groups'
#: densities (FAR_GROUPS of them at most).
FOREST_BYTES = 4
MAX_FAR_GROUPS = FOREST_BYTES - 1

#: The columns of a chunk's plants (ChunkData.forest, float32): the base
#: from the chunk's centre (planet axes, x y z), the turn about the vertical
#: (radians), the scale, the asset's index (:data:`ASSETS`), a tint in
#: [0, 1) (0.75 and up: a dry, scorched plant).
COLUMNS = ("x", "y", "z", "yaw", "scale", "asset", "tint")

Ramp = tuple[float, float]


@dataclass(frozen=True)
class Group:
    """Assets that grow alike [MSFS species]: their names, spawn ratios and
    scale range, and how they take the biomes and the ground.

    :ivar density: instances per hectare in each biome (forest, grass,
        desert, rock, snow) at full vegetation density
    :ivar grid: "tree" or "small": the grid it grows on
    :ivar far: a tree the far forest shades (V2: the shader-map's tables
        are baked per far group, in :data:`SPECIES` order)
    :ivar clumped: how much the clumping noise thins it (0 none)
    :ivar dry: its plants are dry and scorched (a tint)
    :ivar slope, convexity, altitude: ramps (value where it grows none,
        value where it grows fully; smooth between, either way round) on
        the slope (degrees), the convexity (1/m: > 0 hilltops, < 0 hollows)
        and the altitude (metres above sea level); None: no rule
    """

    name: str
    assets: tuple[str, ...]
    weights: tuple[float, ...]
    scale: tuple[float, float]
    density: tuple[float, float, float, float, float]
    grid: str = "tree"
    far: bool = False
    clumped: float = 1.0
    dry: bool = False
    slope: Ramp | None = (40.0, 26.0)
    convexity: Ramp | None = None
    altitude: Ramp | None = None


#: What grows, by biome
# The order is forest, grass, desert, rock, snow:
# (instances per hectare: a dense conifer forest
#: ~400; the packs' crowns are sparse cards, ~15 m^2 seen from above).
SPECIES: tuple[Group, ...] = (
    Group(
        "conifers",
        (
            "forest_tree_far_0",
            "forest_tree_far_1",
            "forest_tree_far_2",
            "forest_tree_far_3",
            "forest_tree_far_4",
            "forest_tree_far_5",
            "forest_tree_far_6",
            "forest_tree_far_7",
            "forest_tree_near_0",
            "forest_tree_near_1",
            "forest_tree_near_2",
            "forest_tree_near_3",
        ),
        # far_3, far_6 and near_0 are autumn-yellow to orange: a few.
        (0.6, 0.6, 1.0, 0.25, 1.0, 0.8, 0.15, 1.0, 0.1, 1.2, 1.2, 1.2),
        (0.75, 1.15),
        (420.0, 0.0, 0.0, 0.2, 0.0),
        far=True,
        clumped=0.6,
        # Thinner on sharp crests (wind, thin soil).
        convexity=(0.03, 0.01),
    ),
    Group(
        "birches",
        ("birch_tree_0", "birch_tree_1", "birch_tree_2"),
        (1.0, 1.0, 1.0),
        (0.7, 1.05),
        (50.0, 1.0, 0.0, 0.0, 0.0),
        far=True,
        clumped=1.6,
        # In the hollows (wetter), not on the crests.
        convexity=(0.006, -0.002),
    ),
    Group(
        "snowy pines",
        ("snowy_tree_0", "snowy_tree_1", "snowy_tree_2"),
        (1.0, 1.0, 1.0),
        (0.8, 1.2),
        (0.0, 0.0, 0.0, 0.0, 6.0),
        far=True,
        slope=(36.0, 24.0),
    ),
    Group(
        "bushes",
        ("forest_plant_0", "forest_plant_1", "forest_plant_2", "forest_plant_3"),
        (1.0, 1.0, 1.0, 0.5),
        (0.6, 1.1),
        (0.0, 10.0, 0.0, 5.0, 0.0),
        grid="small",
        clumped=0.5,
        slope=(45.0, 30.0),
        convexity=(0.02, 0.0),
    ),
    Group(
        "dry bushes",
        ("forest_plant_0", "forest_plant_2"),
        (1.0, 1.0),
        (0.35, 0.7),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        grid="small",
        clumped=0.3,
        dry=True,
    ),
    Group(
        "rocks",
        tuple(f"forest_rock_{k}" for k in range(9)),
        (1.0,) * 9,
        (0.8, 2.2),
        (0.0, 4.0, 6.0, 60.0, 0.0),
        grid="small",
        clumped=0.2,
        # Scree and outcrops: more on slopes and crests.
        slope=(5.0, 30.0),
        convexity=(-0.004, 0.008),
    ),
)

GRIDS = ("tree", "small")


@functools.cache
def asset_names() -> tuple[str, ...]:
    """:returns: the assets in index order (the ``asset`` column)"""
    names: list[str] = []
    for group in SPECIES:
        for name in group.assets:
            if name not in names:
                names.append(name)
    return tuple(names)


ASSETS = asset_names()
#: The far groups' indices in SPECIES (the vertex's density bytes' order).
FAR_GROUPS = tuple(g for g, group in enumerate(SPECIES) if group.far)
assert len(FAR_GROUPS) <= MAX_FAR_GROUPS


# ── the crowns and the grids' cells ─────────────────────────────────────────


@functools.cache
def crowns() -> dict[str, tuple[float, float]]:
    """:returns: per asset (crown area seen from above, m^2; height, m), as
    baked (``crowns.json``), else estimated from the index's sizes"""
    out: dict[str, tuple[float, float]] = {}
    if CROWNS_PATH.exists():
        for name, e in json.loads(CROWNS_PATH.read_text()).items():
            out[name] = (float(e["area"]), float(e["height"]))
    else:  # an estimate: an ellipse in the bounding box, two thirds filled
        from world_builder.datafiles import tree_assets

        for a in tree_assets():
            w, d, h = a.size
            out[a.name] = (math.pi * w * d / 4.0 * 0.66, h)
    return out


def group_crown_area(group: Group) -> float:
    """:returns: the group's mean crown area, m^2 (its spawn ratios, its
    scale range: E[s^2] for s uniform in it)"""
    table = crowns()
    w = np.array(group.weights, dtype=np.float64)
    areas = np.array([table[a][0] for a in group.assets])
    lo, hi = group.scale
    return float((w @ areas) / w.sum() * (lo * lo + lo * hi + hi * hi) / 3.0)


def _main_density(group: Group) -> float:
    return max(group.density)


@functools.cache
def cells_per_period(grid: str) -> int:
    """:returns: the grid's cells per UV_PERIOD: its side CELL_CROWNS mean
    crown diameters of its groups"""
    groups = [g for g in SPECIES if g.grid == grid]
    weights = np.array([_main_density(g) for g in groups])
    diameter = 2.0 * math.sqrt(
        float(weights @ np.array([group_crown_area(g) for g in groups]))
        / weights.sum()
        / math.pi
    )
    return max(1, round(UV_PERIOD / (CELL_CROWNS[grid] * diameter)))


def cell_size(grid: str) -> float:
    """:returns: the grid's cell side, uv metres"""
    return UV_PERIOD / cells_per_period(grid)


TREE_CELL = cell_size("tree")
SMALL_CELL = cell_size("small")


# ── randoms and noise ───────────────────────────────────────────────────────


def _hash(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
    """pcg3d on uint32 arrays (wrapping)."""
    m = np.uint32(1664525)
    c = np.uint32(1013904223)
    x = x.astype(np.uint32) * m + c
    y = y.astype(np.uint32) * m + c
    z = z.astype(np.uint32) * m + c
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


def value_noise(
    u: np.ndarray, v: np.ndarray, scale: float, face: int, seed: int
) -> np.ndarray:
    """Smooth value noise in [0, 1) on a lattice of *scale* uv metres (u, v:
    global uv metres)."""
    x, y = u / scale, v / scale
    i, j = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - i, y - j
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    z = np.full(i.shape, 977 + 16 * face + 4096 * seed, dtype=np.int64)

    def lattice(a, b):
        return _unit(_hash(a & 0xFFFFFFFF, b & 0xFFFFFFFF, z & 0xFFFFFFFF)[0])

    top = lattice(i, j) * (1 - fx) + lattice(i + 1, j) * fx
    bottom = lattice(i, j + 1) * (1 - fx) + lattice(i + 1, j + 1) * fx
    return top * (1 - fy) + bottom * fy


def _clump_factor(clump01: np.ndarray, clumped: np.ndarray) -> np.ndarray:
    """:returns: (n, groups) the clumping's thinning (clump01: (n,) in [0, 1])"""
    return np.clip(
        1.0 - CLUMP_STRENGTH * clumped[None] * (1.0 - clump01[:, None]), 0.0, 1.0
    )


def _clump01(noise: np.ndarray) -> np.ndarray:
    return np.clip((noise - 0.25) / 0.5, 0.0, 1.0)


@functools.cache
def clump_means() -> np.ndarray:
    """:returns: (groups,) the clumping's mean thinning per group (over the
    noise's own distribution): what a coarse vertex expects"""
    rng = np.random.default_rng(5)
    u = rng.uniform(0.0, 1e6, 200_000)
    v = rng.uniform(0.0, 1e6, 200_000)
    noise = value_noise(u, v, CLUMP_SCALE, 0, 0)
    clumped = np.array([g.clumped for g in SPECIES])
    return _clump_factor(_clump01(noise), clumped).mean(axis=0)


def thinning(spacing: float) -> tuple[float, float]:
    """:returns: (the share of its trees a chunk of this vertex spacing
    keeps, the scale its trees grow by); (0, 0) for none"""
    if spacing > MAX_SPACING:
        return 0.0, 0.0
    doublings = max(0.0, math.log2(spacing / FULL_SPACING))
    return 0.5 ** round(doublings), THINNING_GROWTH ** round(doublings)


def _ramp(x: np.ndarray, ramp: Ramp | None) -> np.ndarray | float:
    if ramp is None:
        return 1.0
    zero, one = ramp
    t = np.clip((x - zero) / (one - zero), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def density_from_noise(noise: np.ndarray) -> np.ndarray:
    """:returns: the procedural vegetation density in [DENSITY_FLOOR, 1] from
    its fbm (about [-1, 1])"""
    lo, hi = DENSITY_RAMP
    t = np.clip((noise - lo) / (hi - lo), 0.0, 1.0)
    return DENSITY_FLOOR + (1.0 - DENSITY_FLOOR) * t * t * (3.0 - 2.0 * t)


# ── the rules ───────────────────────────────────────────────────────────────


@functools.cache
def _tables() -> dict[str, Any]:
    density = np.array([g.density for g in SPECIES])  # (groups, 5) per hectare
    cell_area = np.array([cell_size(g.grid) ** 2 for g in SPECIES])
    picks = []
    for g in SPECIES:
        w = np.array(g.weights, dtype=np.float64)
        picks.append(
            (np.cumsum(w) / w.sum(), np.array([ASSETS.index(a) for a in g.assets]))
        )
    return {
        "per_cell": density * (cell_area / 1e4)[:, None],  # (groups, 5)
        "clumped": np.array([g.clumped for g in SPECIES]),
        "scales": np.array([g.scale for g in SPECIES]),
        "dry": np.array([g.dry for g in SPECIES]),
        "grid": np.array([GRIDS.index(g.grid) for g in SPECIES]),
        "picks": picks,
        "crown_area": np.array([crowns()[a][0] for a in ASSETS]),
        "far_cover": np.array(
            [
                group_crown_area(SPECIES[g]) / cell_size(SPECIES[g].grid) ** 2
                for g in FAR_GROUPS
            ]
        ),
    }


def probabilities(
    weights: np.ndarray,
    density: np.ndarray,
    slope: np.ndarray,
    convexity: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    """Each group's probability per cell of its grid [SS4 s.22, MSFS].

    :param weights: (N, 6) biome weights in [0, 1] (:data:`biomes.BIOMES`)
    :param density: (N,) the vegetation density in [0, 1]
    :param slope: (N,) degrees
    :param convexity: (N,) minus the elevation's Laplacian, 1/m
    :param elevation: (N,) metres above sea level
    :returns: (N, groups) in [0, 1]; the groups of one grid sum to 1 at most
    """
    t = _tables()
    biome = np.asarray(weights, dtype=np.float64)[
        :, [FOREST, GRASS, DESERT, ROCK, SNOW]
    ]
    p = biome @ t["per_cell"].T * np.asarray(density, dtype=np.float64)[:, None]
    for g, group in enumerate(SPECIES):
        p[:, g] *= (
            _ramp(slope, group.slope)
            * _ramp(convexity, group.convexity)
            * _ramp(elevation, group.altitude)
        )
    for k in range(len(GRIDS)):
        mine = t["grid"] == k
        total = p[:, mine].sum(axis=1, keepdims=True)
        p[:, mine] /= np.maximum(total, 1.0)
    return p


def far_data(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:returns: ((N, len(FAR_GROUPS)) the far groups' expected densities
    (shares of the trees' cells, the clumping's mean thinning in), (N,) the
    trees' expected cover (summed crown area per area))"""
    far = p[:, FAR_GROUPS] * clump_means()[list(FAR_GROUPS)][None]
    return far, far @ _tables()["far_cover"]


def pack_far(far: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """:returns: (N, FOREST_BYTES) uint8: cover / COVER_MAX, then the far
    groups' densities"""
    out = np.zeros((len(cover), FOREST_BYTES), dtype=np.float64)
    out[:, 0] = cover / COVER_MAX
    out[:, 1 : 1 + far.shape[1]] = far
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


# ── the grids' samples ──────────────────────────────────────────────────────


def _bilinear(grid: np.ndarray, gu: np.ndarray, gv: np.ndarray) -> np.ndarray:
    """(rows v, columns u[, ...]) at fractional grid coordinates, clamped."""
    n = grid.shape[0] - 1
    gu = np.clip(gu, 0.0, n)
    gv = np.clip(gv, 0.0, n)
    u0 = np.minimum(np.floor(gu).astype(np.int64), n - 1)
    v0 = np.minimum(np.floor(gv).astype(np.int64), n - 1)
    fu, fv = gu - u0, gv - v0
    if grid.ndim == 3:
        fu, fv = fu[:, None], fv[:, None]
    top = grid[v0, u0] * (1 - fu) + grid[v0, u0 + 1] * fu
    bottom = grid[v0 + 1, u0] * (1 - fu) + grid[v0 + 1, u0 + 1] * fu
    return top * (1 - fv) + bottom * fv


def cell_samples(
    grid: str, cu: np.ndarray, cv: np.ndarray, face: int, seed: int
) -> dict[str, np.ndarray]:
    """The first randoms of cells (cu, cv: their global indices on the face:
    the index in the period plus the period times :func:`cells_per_period`):
    ``keep`` (its group, or none), ``pick`` (its asset), ``thin`` (kept by a
    coarse chunk), in [0, 1), and the hash :func:`cell_details` goes on from.
    Two steps: most cells are rejected on these."""
    salt = GRIDS.index(grid)
    z = (face + 8 * salt) * 65536 + (seed & 0xFFFF)
    z = np.full(np.shape(cu), z, dtype=np.int64)
    h1 = _hash(cu & 0xFFFFFFFF, cv & 0xFFFFFFFF, z & 0xFFFFFFFF)
    keep, pick, thin = (_unit(h) for h in h1)
    return {
        "keep": keep,
        "pick": pick,
        "thin": thin,
        "h0": h1[0],
        "h1": h1[1],
        "h2": h1[2],
    }


def cell_details(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The rest of cells' randoms (*s* from :func:`cell_samples`, filtered):
    their samples' point in the cell (``ju``, ``jv``: from its corner, in
    cells), ``yaw``, ``scale``, ``tint`` in [0, 1). :returns: *s* with them"""
    h2 = _hash(
        s["h0"] ^ np.uint32(0x9E3779B9),
        s["h1"] ^ np.uint32(0x85EBCA6B),
        s["h2"] ^ np.uint32(0xC2B2AE35),
    )
    h3 = _hash(
        h2[0] ^ np.uint32(0x27D4EB2F),
        h2[1] ^ np.uint32(0x165667B1),
        h2[2] ^ np.uint32(0xD3A2646C),
    )
    ju, jv, yaw = (_unit(h) for h in h2)
    scale, tint, _ = (_unit(h) for h in h3)
    return dict(
        s,
        ju=0.5 + JITTER * (2.0 * ju - 1.0),
        jv=0.5 + JITTER * (2.0 * jv - 1.0),
        yaw=yaw,
        scale=scale,
        tint=tint,
    )


def _choose(
    s: dict[str, np.ndarray], p: np.ndarray, clump01: np.ndarray, grid: str
) -> tuple[np.ndarray, np.ndarray]:
    """:returns: (the grown samples' mask, their groups) from their groups'
    probabilities (n, groups) and the clumping there"""
    t = _tables()
    share = p * _clump_factor(clump01, t["clumped"])
    share[:, t["grid"] != GRIDS.index(grid)] = 0.0
    cum = np.cumsum(share, axis=1)
    group = np.argmax(s["keep"][:, None] < cum, axis=1)
    return s["keep"] < cum[:, -1], group


def _assets(s: dict[str, np.ndarray], group: np.ndarray) -> tuple[np.ndarray, ...]:
    """:returns: the grown samples' assets, scales (before any growth) and
    tints"""
    t = _tables()
    asset = np.empty(len(group), dtype=np.int64)
    for g, (cdf, ids) in enumerate(t["picks"]):
        mask = group == g
        if mask.any():
            k = np.searchsorted(cdf, s["pick"][mask], side="right").clip(
                0, len(ids) - 1
            )
            asset[mask] = ids[k]
    lo, hi = t["scales"][group, 0], t["scales"][group, 1]
    scale = lo + (hi - lo) * s["scale"]
    tint = np.where(t["dry"][group], 0.75 + 0.25 * s["tint"], 0.75 * s["tint"])
    return asset, scale, tint


def _splat_cover(
    gu: np.ndarray, gv: np.ndarray, area: np.ndarray, step: float
) -> np.ndarray:
    """The crowns' summed area per area at each vertex: each crown a square
    of its area, centred on its tree (gu, gv: grid coordinates), its overlap
    with each vertex's footprint (a step-wide square) added.

    :returns: (VERTS, VERTS) rows v
    """
    cover = np.zeros((VERTS, VERTS))
    if not len(gu):
        return cover
    half = 0.5 * np.sqrt(area) / step  # in grid steps
    reach = int(math.ceil(float(half.max()) + 0.5))
    offsets = np.arange(-reach, reach + 1)
    ku = np.rint(gu).astype(np.int64)[:, None] + offsets[None]
    kv = np.rint(gv).astype(np.int64)[:, None] + offsets[None]

    def overlap(g, k):
        lo = np.maximum(g[:, None] - half[:, None], k - 0.5)
        hi = np.minimum(g[:, None] + half[:, None], k + 0.5)
        return np.maximum(hi - lo, 0.0)

    ou, ov = overlap(gu, ku), overlap(gv, kv)  # (n, 2 reach + 1), in steps
    weight = ov[:, :, None] * ou[:, None, :]
    rows = np.broadcast_to(kv[:, :, None], weight.shape)
    cols = np.broadcast_to(ku[:, None, :], weight.shape)
    inside = (rows >= 0) & (rows < VERTS) & (cols >= 0) & (cols < VERTS) & (weight > 0)
    np.add.at(cover, (rows[inside], cols[inside]), weight[inside])
    return cover


def place(
    uv0: tuple[float, float],
    step: float,
    period: tuple[int, int],
    face: int,
    dirs: np.ndarray,
    elevation: np.ndarray,
    p: np.ndarray,
    radius: float,
    centre: np.ndarray,
    seed: int,
    has_ocean: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """A chunk's plants, and its trees' actual cover where it places them all.

    :param uv0: its first vertex's uv (wrapped), metres
    :param step: uv metres per grid step (the vertex spacing's uv)
    :param period: which UV_PERIOD the first vertex is in (u, v)
    :param dirs: (VERTS, VERTS, 3) its vertices' directions, rows v
    :param elevation: (VERTS, VERTS) its ground, metres above sea level
    :param p: (VERTS * VERTS, groups) the groups' probabilities per cell
        (:func:`probabilities`)
    :param radius: the planet's, metres
    :param centre: (3,) the chunk's centre (its vertices are from it)
    :param seed: the planet's seed
    :param has_ocean: nothing grows under the sea level then
    :returns: ((n, len(COLUMNS)) float32 or None (none, or a chunk too
        coarse: :func:`thinning`), (VERTS, VERTS) the far groups' cover or
        None: its expected value stands)
    """
    keep_share, grow = thinning(step)
    if keep_share <= 0.0 or p.max() <= 0.0:
        return None, None
    pg = p.reshape(VERTS, VERTS, -1)
    span = QUADS * step
    t = _tables()
    rows = []
    actual_cover = None
    for grid in GRIDS:
        mine = t["grid"] == GRIDS.index(grid)
        if pg[..., mine].max() <= 0.0:
            if grid == "tree" and keep_share >= 1.0:
                actual_cover = np.zeros((VERTS, VERTS))
            continue
        if grid == "small" and step > SMALL_SPACING:
            continue
        n = cells_per_period(grid)
        c = UV_PERIOD / n
        # The cells around the chunk's uv square, a margin wide for the
        # cover of the vertices on its edge (their footprint, the crowns).
        margin = 0.0
        if grid == "tree" and keep_share >= 1.0:
            margin = step + 0.5 * math.sqrt(float(t["crown_area"].max())) * 1.3
        lo = np.floor((np.asarray(uv0) - margin) / c).astype(np.int64)
        hi = np.floor((np.asarray(uv0) + span + margin) / c).astype(np.int64)
        cu, cv = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        cu, cv = cu.ravel(), cv.ravel()
        s = cell_samples(grid, cu + period[0] * n, cv + period[1] * n, face, seed)
        # Rejected at once: the cells no group of the chunk can fill, and
        # those a coarse chunk thins out (the same at every level: a
        # chunk's kept trees are its finer children's, thinned).
        live = s["keep"] < pg[..., mine].sum(axis=-1).max()
        if grid == "tree" and keep_share < 1.0:
            live &= s["thin"] < keep_share
        if not live.any():
            continue
        s = cell_details({k: a[live] for k, a in s.items()})
        cu, cv = cu[live], cv[live]
        u = (cu + s["ju"]) * c
        v = (cv + s["jv"]) * c
        gu = (u - uv0[0]) / step
        gv = (v - uv0[1]) / step
        inside = (gu >= 0) & (gu < QUADS) & (gv >= 0) & (gv < QUADS)
        near = (
            (gu >= -margin / step)
            & (gu < QUADS + margin / step)
            & (gv >= -margin / step)
            & (gv < QUADS + margin / step)
        )
        near &= inside | (margin > 0.0)
        if not near.any():
            continue
        s = {k: a[near] for k, a in s.items()}
        u, v, gu, gv, inside = u[near], v[near], gu[near], gv[near], inside[near]
        # Groves and clearings.
        clump = _clump01(
            value_noise(
                u + period[0] * UV_PERIOD,
                v + period[1] * UV_PERIOD,
                CLUMP_SCALE,
                face,
                seed,
            )
        )
        grown, group = _choose(s, _bilinear(pg, gu, gv), clump, grid)
        if grid == "tree" and keep_share >= 1.0:
            far = grown & np.isin(group, FAR_GROUPS)
            asset, scale, _ = _assets({k: a[far] for k, a in s.items()}, group[far])
            actual_cover = _splat_cover(
                gu[far], gv[far], t["crown_area"][asset] * scale * scale, step
            )
        grown &= inside
        if not grown.any():
            continue
        s = {k: a[grown] for k, a in s.items()}
        gu, gv, group = gu[grown], gv[grown], group[grown]
        asset, scale, tint = _assets(s, group)
        if grid == "tree":
            scale = scale * grow
        # On the ground, between the vertices.
        ground = _bilinear(np.asarray(elevation, dtype=np.float64), gu, gv)
        keep = ground >= SEA_FLOOR if has_ocean else np.ones(len(ground), dtype=bool)
        if not keep.any():
            continue
        d = _bilinear(np.asarray(dirs, dtype=np.float64), gu, gv)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        base = d * (radius + ground - SINK)[:, None] - np.asarray(centre)[None]
        rows.append(
            np.stack(
                [
                    base[:, 0],
                    base[:, 1],
                    base[:, 2],
                    s["yaw"] * 2.0 * math.pi,
                    scale,
                    asset,
                    tint,
                ],
                axis=1,
            )[keep]
        )
    plants = np.concatenate(rows).astype(np.float32) if rows else None
    return plants, actual_cover


def chunk_period(grid: Any, key: tuple) -> tuple[int, int]:
    """:returns: which UV_PERIOD the chunk's first vertex is in (u, v)"""
    level, face, i, j = key
    unit = grid.uv_unit
    return (
        math.floor(((i * QUADS) << level) * unit / UV_PERIOD),
        math.floor(((j * QUADS) << level) * unit / UV_PERIOD),
    )


def chunk_forest(
    grid: Any,
    key: tuple,
    dirs: np.ndarray,
    elevation: np.ndarray,
    weights: np.ndarray,
    centre: np.ndarray,
    spec: Any,
    density: np.ndarray | None = None,
    slope: np.ndarray | None = None,
    convexity: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """:func:`place` for a chunk being generated (``chunkgen``).

    :param weights: (VERTS * VERTS, 6) uint8 biome weights
    :param density: (VERTS * VERTS,) the vegetation density (None: 1)
    :param slope: (VERTS * VERTS,) degrees (None: flat)
    :param convexity: (VERTS * VERTS,) 1/m (None: 0)
    :returns: (its plants or None, (VERTS * VERTS, FOREST_BYTES) uint8 the
        far forest's data per vertex)
    """
    count = VERTS * VERTS
    if not getattr(spec, "has_vegetation", False):
        return None, np.zeros((count, FOREST_BYTES), dtype=np.uint8)
    w = np.asarray(weights).reshape(count, -1).astype(np.float64) / 255.0
    ones = np.ones(count)
    p = probabilities(
        w,
        ones if density is None else density,
        0.0 * ones if slope is None else slope,
        0.0 * ones if convexity is None else convexity,
        np.asarray(elevation, dtype=np.float64).ravel(),
    )
    if spec.has_ocean:
        p[np.asarray(elevation).ravel() < SEA_FLOOR] = 0.0
    far, cover = far_data(p)
    level, face = key[0], key[1]
    step = (1 << level) * grid.uv_unit
    plants, actual = place(
        grid.uv_origin(key),
        step,
        chunk_period(grid, key),
        face,
        dirs,
        elevation,
        p,
        spec.radius,
        centre,
        int(spec.seed),
        spec.has_ocean,
    )
    if actual is not None:
        cover = actual.ravel()
    return plants, pack_far(far, cover)
