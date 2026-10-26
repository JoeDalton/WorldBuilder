"""
A field's rocks, from its spec alone: where each is, its size, how it turns,
its shape and material. Any part of space can be asked for, in any order, and
always gets the same rocks: nothing is stored.

How (per size class, then per cell):

- **Size classes.** Radii [lo, 2 lo), [2 lo, 4 lo), ... up to the sizes' hi.
  Class k scatters in cubic cells of edge ``cell_factor * lo * 2^k``, so big
  rocks are sparse in big cells and each class can be asked for out to its
  own distance.
- **Candidates.** A cell's 64-bit hash (splitmix64 chained over its
  coordinates, from the class's seed) draws a Poisson count, then each
  candidate's position, radius (from the class's own slice of the size
  distribution) and attributes, every draw from the hash of (cell,
  candidate, slot).
- **Thinning.** A candidate exists with probability = the density there: the
  field's profile (soft edges) times its clumping (a smoothstep of fbm noise).
  Candidates inside a keep-out do not exist.
- **Hard core (Matern type II).** A rock is dropped where it overlaps a rock
  that ranks above it: larger, or as large with a smaller id. The test is
  against existing candidates, never against what survived, so it is local:
  a cell's rocks depend on its 27 neighbours in its own class and in every
  larger one, nothing farther.

The native library (``native/scatter.c``) runs the same steps. Every choice
(count, existence, overlap, class, prototype) is made with the same
arithmetic on the same values in both, so they give the same rocks. Their
orientations and spin axes go through sin, cos and log, which may differ in
the last bits between numpy and C.
"""

from __future__ import annotations

import ctypes
import functools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from world_builder.asteroids.distributions import TABLE_SIZE, interpolate
from world_builder.asteroids.motion import quat_multiply
from world_builder.asteroids.spec import (
    AroundAxis,
    Box,
    FieldSpec,
    KeepOut,
    Sphere,
)
from world_builder.terrain import noise

#: One rock, as both implementations write it (``WbRock`` in scatter.c).
ROCK_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("cls", "<i4"),
        ("proto", "<i4"),
        ("centre", "<f8", 3),
        ("radius", "<f8"),
        ("q0", "<f8", 4),
        ("axis", "<f8", 3),
        ("precession", "<f8"),
        ("spin", "<f8"),
        ("ice", "<f8"),
    ],
    align=True,
)

#: Draws per candidate (slot numbers below), after the cell's count (slot 0).
SLOTS = 20
_POS, _THIN, _SIZE, _ID, _PROTO, _ICE, _RATE = 0, 3, 4, 5, 6, 7, 8
_AXIS_PHI, _AXIS_W, _TUMBLE, _CONE, _RATIO, _AZIMUTH, _PHASE, _Q0 = (
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_CLASS_STEP = np.uint64(0xD1B54A32D192ED03)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_UNIT = 2.0**-53
#: The clumping noise's seed, after the field's.
_CLUMP_SEED = 1013
#: The 27 cells around one (itself included).
_NEIGHBOURS = np.array(
    [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
    dtype=np.int64,
)


def _mix64(h: np.ndarray) -> np.ndarray:
    """The splitmix64 finaliser (``wb_mix64``), on uint64 arrays."""
    h = h ^ (h >> np.uint64(30))
    h = h * _MIX_1
    h = h ^ (h >> np.uint64(27))
    h = h * _MIX_2
    return h ^ (h >> np.uint64(31))


def _draw(h: np.ndarray, index: np.ndarray | int) -> np.ndarray:
    """The 64-bit draw number *index* of hash *h*."""
    with np.errstate(over="ignore"):
        return _mix64(h + np.asarray(index, dtype=np.uint64) * _GOLDEN)


def _cell_hash(cells: np.ndarray, seed: np.uint64) -> np.ndarray:
    """A cell's 64-bit hash (``cell_hash`` in scatter.c): splitmix64 chained
    over its three indices. Not the noise's lattice hash, whose xor of
    products collides on small coordinates (cells (-1, 1, 0) and (1, -1, 0)
    share one), which would repeat whole cells of rocks."""
    with np.errstate(over="ignore"):
        h = _mix64(seed + cells[:, 0].astype(np.int64).view(np.uint64))
        h = _mix64(h + cells[:, 1].astype(np.int64).view(np.uint64))
        return _mix64(h + cells[:, 2].astype(np.int64).view(np.uint64))


def _unit(bits: np.ndarray) -> np.ndarray:
    """:returns: 64 random bits as a uniform double in [0, 1), exactly"""
    return (bits >> np.uint64(11)).astype(np.float64) * _UNIT


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@functools.cache
def fbm_quantile(octaves: int, share: float) -> float:
    """The value of :func:`noise.fbm` (default lacunarity and gain) that a
    share *share* of space lies below, measured over 100,000 points."""
    rng = np.random.default_rng(0xC1A3)
    points = rng.uniform(0.0, 64.0, size=(100_000, 3))
    values = noise.fbm(points, 1.0, octaves, 0xC1A3)
    return float(np.quantile(values, share))


@dataclass(frozen=True)
class SizeClass:
    """One size class.

    :ivar lo: smallest radius
    :ivar hi: largest radius (exclusive but for the last class)
    :ivar cell: the cells' edge
    :ivar cdf_lo: the size distribution's CDF at lo
    :ivar cdf_hi: and at hi
    :ivar per_cell: the expected candidates per cell at full density
    """

    lo: float
    hi: float
    cell: float
    cdf_lo: float
    cdf_hi: float
    per_cell: float


class _Tables:
    """Everything a scatter draws with, from the spec (and any runtime
    keep-outs): the same numbers go to numpy and to C."""

    def __init__(self, spec: FieldSpec, keep_outs: tuple[KeepOut, ...]):
        self.spec = spec
        sizes = spec.sizes
        per_m3 = spec.density.count_per_km3 * 1e-9
        classes = []
        lo = sizes.lo
        while True:
            hi = min(2.0 * lo, sizes.hi)
            cdf_lo, cdf_hi = (float(c) for c in sizes.cdf([lo, hi]))
            if len(classes) == 0:
                cdf_lo = 0.0
            last = hi >= sizes.hi
            if last:
                cdf_hi = 1.0
            cell = spec.cell_factor * lo
            classes.append(
                SizeClass(
                    lo, hi, cell, cdf_lo, cdf_hi, per_m3 * (cdf_hi - cdf_lo) * cell**3
                )
            )
            if last:
                break
            lo = hi
        self.classes = classes
        # One inverse-CDF table per class, over its slice of the sizes: a
        # class holding a sliver of the distribution (the largest rocks)
        # still draws its sizes at full resolution, within its bounds.
        u = np.linspace(0.0, 1.0, TABLE_SIZE)
        tables = []
        for c in classes:
            table = sizes.ppf(c.cdf_lo + u * (c.cdf_hi - c.cdf_lo))
            table[0], table[-1] = c.lo, c.hi
            tables.append(np.clip(np.maximum.accumulate(table), c.lo, c.hi))
        self.size_tables = np.stack(tables)
        seed = noise._mix_seed(spec.seed)
        with np.errstate(over="ignore"):
            self.class_seeds = _mix64(
                seed + np.arange(1, len(classes) + 1, dtype=np.uint64) * _CLASS_STEP
            )
        # Cumulative Poisson probabilities per class, out to where the tail
        # is below 1e-15 (the same length for every class).
        most = max(c.per_cell for c in classes)
        self.poisson_max = int(most + 12.0 * math.sqrt(most) + 24.0)
        self.poisson = np.ones((len(classes), self.poisson_max))
        for k, c in enumerate(classes):
            n = np.arange(self.poisson_max, dtype=np.float64)
            log_p = (
                -c.per_cell + n * math.log(max(c.per_cell, 1e-300)) - _log_factorial(n)
            )
            if c.per_cell <= 0.0:
                log_p = np.where(n == 0, 0.0, -np.inf)
            self.poisson[k] = np.minimum(np.cumsum(np.exp(log_p)), 1.0)
        # Clumping.
        clump = spec.density.clumping
        if clump is None:
            self.clump_plan = None
            self.clump_low, self.clump_width = 0.0, 1.0
        else:
            if clump.softness <= 0.0:
                raise ValueError("clumping softness must be positive")
            self.clump_plan = noise._fbm_plan(
                clump.scale, clump.octaves, spec.seed + _CLUMP_SEED, 2.0, 0.5, None
            )
            threshold = fbm_quantile(clump.octaves, 1.0 - clump.coverage)
            self.clump_low = threshold - clump.softness
            self.clump_width = 2.0 * clump.softness
        # Spin.
        spin = spec.spin
        self.rate_table = spin.rate.table_values()
        self.rate_ref = sizes.lo
        if isinstance(spin.axis, AroundAxis) and spin.axis.kappa > 1e-9:
            a = np.asarray(spin.axis.axis, dtype=np.float64)
            a = a / np.linalg.norm(a)
            helper = np.array([1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
            e1 = np.cross(a, helper)
            e1 /= np.linalg.norm(e1)
            self.axis_frame = np.stack((e1, np.cross(a, e1), a))
            self.kappa = float(spin.axis.kappa)
        else:
            self.axis_frame = np.eye(3)
            self.kappa = 0.0
        tumble = spin.tumble
        self.tumble_fraction = 0.0 if tumble is None else float(tumble.fraction)
        zero = np.zeros(TABLE_SIZE)
        self.cone_table = zero if tumble is None else tumble.cone_angle.table_values()
        self.ratio_table = zero if tumble is None else tumble.ratio.table_values()
        self.ice_table = spec.material.ice_fraction.table_values()
        # Extent and keep-outs.
        self.bounds = spec.shape.bounds()
        self.keep_outs = np.array(
            [k.row() for k in (*spec.keep_outs, *keep_outs)], dtype=np.float64
        ).reshape(-1, 7)

    # ── density ───────────────────────────────────────────────────────────

    def profile(self, p: np.ndarray) -> np.ndarray:
        """:returns: the shape's density profile at field-frame points (n, 3)"""
        shape = self.spec.shape
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        if isinstance(shape, Box):
            hx, hy, hz = shape.half_size
            if shape.edge > 0.0:
                e = shape.edge
                sx = _smoothstep01((hx - np.abs(x)) / e)
                sy = _smoothstep01((hy - np.abs(y)) / e)
                sz = _smoothstep01((hz - np.abs(z)) / e)
                return sx * sy * sz
            inside = (np.abs(x) <= hx) & (np.abs(y) <= hy) & (np.abs(z) <= hz)
            return inside.astype(np.float64)
        if isinstance(shape, Sphere):
            d = np.sqrt(x * x + y * y + z * z)
            if shape.edge > 0.0:
                return _smoothstep01((shape.radius - d) / shape.edge)
            return (d <= shape.radius).astype(np.float64)
        raise TypeError(f"unknown field shape {shape!r}")

    def clump(self, p: np.ndarray) -> np.ndarray:
        """:returns: the clumping factor at field-frame points (n, 3)"""
        if self.clump_plan is None:
            return np.ones(len(p))
        c = self.spec.density.clumping
        v = noise.fbm(p, c.scale, c.octaves, self.spec.seed + _CLUMP_SEED)
        return _smoothstep01((v - self.clump_low) / self.clump_width)

    def density(self, p: np.ndarray) -> np.ndarray:
        """:returns: the share of candidates that exist, at field-frame points"""
        return self.profile(p) * self.clump(p)

    def kept_out(self, centres: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """:returns: (n,) whether each sphere enters a keep-out"""
        out = np.zeros(len(centres), dtype=bool)
        for row in self.keep_outs:
            a, b, clearance = row[:3], row[3:6], row[6]
            ab = b - a
            d = centres - a
            length2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]
            if length2 > 0.0:
                t = (d[:, 0] * ab[0] + d[:, 1] * ab[1] + d[:, 2] * ab[2]) / length2
                t = np.clip(t, 0.0, 1.0)
                d = d - t[:, None] * ab
            dist2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
            reach = clearance + radii
            out |= dist2 < reach * reach
        return out


def _log_factorial(n: np.ndarray) -> np.ndarray:
    return np.array([math.lgamma(v + 1.0) for v in n])


# ── the numpy reference ─────────────────────────────────────────────────────


def _candidates(tables: _Tables, cls: int, cells: np.ndarray) -> dict[str, np.ndarray]:
    """Every candidate of class *cls* in *cells* (n, 3), with whether it
    exists (thinning and keep-outs), before the hard core."""
    c = tables.classes[cls]
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    h = _cell_hash(cells, tables.class_seeds[cls])
    counts = np.searchsorted(tables.poisson[cls], _unit(_draw(h, 0)), side="right")
    owner = np.repeat(np.arange(len(cells)), counts)
    starts = np.cumsum(counts) - counts
    m = np.arange(len(owner)) - np.repeat(starts, counts)
    hh = h[owner]
    base = (1 + m * SLOTS).astype(np.uint64)

    def u(slot: int) -> np.ndarray:
        return _unit(_draw(hh, base + np.uint64(slot)))

    centre = np.empty((len(owner), 3))
    for axis in range(3):
        centre[:, axis] = (
            cells[owner, axis].astype(np.float64) + u(_POS + axis)
        ) * c.cell
    radius = interpolate(tables.size_tables[cls], u(_SIZE))
    alive = u(_THIN) < tables.density(centre)
    if len(tables.keep_outs):
        alive &= ~tables.kept_out(centre, radius)
    return {
        "cell": cells[owner],
        "h": hh,
        "base": base,
        "centre": centre,
        "radius": radius,
        "id": _draw(hh, base + np.uint64(_ID)),
        "alive": alive,
    }


def _alive(cand: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = cand["alive"]
    return {key: value[keep] for key, value in cand.items()}


def _unique_cells(cells: np.ndarray) -> np.ndarray:
    return (
        np.unique(cells.reshape(-1, 3), axis=0) if len(cells) else cells.reshape(0, 3)
    )


def _numpy_rocks(tables: _Tables, cls: int, cells: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    rocks = _alive(_candidates(tables, cls, cells))
    if len(rocks["id"]) == 0:
        return np.zeros(0, dtype=ROCK_DTYPE)
    # The candidates that may outrank them: the same class around their
    # cells, and every larger class around their positions.
    blockers = []
    occupied = _unique_cells(rocks["cell"])
    around = _unique_cells(occupied[:, None, :] + _NEIGHBOURS[None, :, :])
    blockers.append(_alive(_candidates(tables, cls, around)))
    for j in range(cls + 1, len(tables.classes)):
        own = np.floor(rocks["centre"] / tables.classes[j].cell).astype(np.int64)
        around = _unique_cells(own[:, None, :] + _NEIGHBOURS[None, :, :])
        blockers.append(_alive(_candidates(tables, j, around)))
    b_centre = np.concatenate([b["centre"] for b in blockers])
    b_radius = np.concatenate([b["radius"] for b in blockers])
    b_id = np.concatenate([b["id"] for b in blockers])
    dropped = np.zeros(len(rocks["id"]), dtype=bool)
    if len(b_id):
        tree = cKDTree(b_centre)
        reach = rocks["radius"] + b_radius.max()
        for a, near in enumerate(tree.query_ball_point(rocks["centre"], reach)):
            if not near:
                continue
            near = np.asarray(near)
            d = b_centre[near] - rocks["centre"][a]
            dist2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
            ra, rb = rocks["radius"][a], b_radius[near]
            touch = dist2 < (ra + rb) * (ra + rb)
            above = (rb > ra) | ((rb == ra) & (b_id[near] < rocks["id"][a]))
            dropped[a] = np.any(touch & above & (b_id[near] != rocks["id"][a]))
    rocks = {key: value[~dropped] for key, value in rocks.items()}
    return _attributes(tables, cls, rocks)


def _attributes(tables: _Tables, cls: int, rocks: dict[str, np.ndarray]) -> np.ndarray:
    """The surviving rocks' records."""
    n = len(rocks["id"])
    out = np.zeros(n, dtype=ROCK_DTYPE)
    if n == 0:
        return out
    hh, base = rocks["h"], rocks["base"]

    def u(slot: int) -> np.ndarray:
        return _unit(_draw(hh, base + np.uint64(slot)))

    out["id"] = rocks["id"]
    out["cls"] = cls
    out["centre"] = rocks["centre"]
    out["radius"] = rocks["radius"]
    prototypes = np.uint64(tables.spec.shapes.prototypes)
    out["proto"] = (_draw(hh, base + np.uint64(_PROTO)) % prototypes).astype(np.int32)
    out["ice"] = interpolate(tables.ice_table, u(_ICE))
    rate = interpolate(tables.rate_table, u(_RATE))
    rate = rate * np.power(
        rocks["radius"] / tables.rate_ref, -tables.spec.spin.size_exponent
    )
    # Axes: isotropic, or von Mises-Fisher about the spec's.
    phi = 2.0 * math.pi * u(_AXIS_PHI)
    v = u(_AXIS_W)
    if tables.kappa > 0.0:
        k = tables.kappa
        w = 1.0 + np.log(v + (1.0 - v) * math.exp(-2.0 * k)) / k
    else:
        w = 2.0 * v - 1.0
    s = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    e1, e2, a = tables.axis_frame
    axis = (
        (s * np.cos(phi))[:, None] * e1
        + (s * np.sin(phi))[:, None] * e2
        + w[:, None] * a
    )
    out["axis"] = axis
    out["precession"] = rate
    tumbling = u(_TUMBLE) < tables.tumble_fraction
    cone = interpolate(tables.cone_table, u(_CONE))
    ratio = interpolate(tables.ratio_table, u(_RATIO))
    out["spin"] = np.where(tumbling, rate * np.cos(cone) * ratio, 0.0)
    # q0: tumbling rocks have their body z on the cone; the others any way.
    q_tumble = _cone_orientation(axis, cone, u(_AZIMUTH), u(_PHASE))
    q_free = _uniform_rotation(u(_Q0), u(_Q0 + 1), u(_Q0 + 2))
    out["q0"] = np.where(tumbling[:, None], q_tumble, q_free)
    return out


def _cone_orientation(
    axis: np.ndarray, cone: np.ndarray, azimuth: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    """q0 of a tumbling rock: body z at *cone* from *axis* (at an azimuth),
    turned about itself by a phase (``cone_orientation`` in scatter.c)."""
    helper = np.where(
        (np.abs(axis[:, 0]) < 0.9)[:, None], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    )
    f1 = np.cross(axis, helper)
    f1 /= np.sqrt(np.sum(f1 * f1, axis=1))[:, None]
    f2 = np.cross(axis, f1)
    alpha = 2.0 * math.pi * azimuth
    sc, cc = np.sin(cone), np.cos(cone)
    zb = (
        cc[:, None] * axis
        + (sc * np.cos(alpha))[:, None] * f1
        + (sc * np.sin(alpha))[:, None] * f2
    )
    # The shortest turn from z to zb.
    align = np.stack((1.0 + zb[:, 2], -zb[:, 1], zb[:, 0], np.zeros(len(zb))), axis=1)
    norm = np.sqrt(np.sum(align * align, axis=1))
    flipped = align[:, 0] < 1e-12
    align = np.where(
        flipped[:, None],
        [0.0, 1.0, 0.0, 0.0],
        align / np.where(flipped, 1.0, norm)[:, None],
    )
    half = math.pi * phase
    turn = np.stack(
        (np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)), axis=1
    )
    return quat_multiply(align, turn)


def _uniform_rotation(u1: np.ndarray, u2: np.ndarray, u3: np.ndarray) -> np.ndarray:
    """Shoemake's uniform random unit quaternion (w, x, y, z)."""
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    t2, t3 = 2.0 * math.pi * u2, 2.0 * math.pi * u3
    return np.stack(
        (b * np.cos(t3), a * np.sin(t2), a * np.cos(t2), b * np.sin(t3)), axis=1
    )


# ── the native scatter ──────────────────────────────────────────────────────


class _WbScatter(ctypes.Structure):
    """``WbScatter`` in scatter.c: the tables, as pointers into numpy arrays
    the :class:`Scatter` keeps alive."""

    _fields_ = [
        ("shape", ctypes.c_int32),
        ("classes", ctypes.c_int32),
        ("half", ctypes.c_double * 3),
        ("edge", ctypes.c_double),
        ("clump_octaves", ctypes.c_int32),
        ("poisson_max", ctypes.c_int32),
        ("clump_frames", ctypes.c_void_p),
        ("clump_seeds", ctypes.c_void_p),
        ("clump_weights", ctypes.c_void_p),
        ("clump_norm", ctypes.c_double),
        ("clump_low", ctypes.c_double),
        ("clump_width", ctypes.c_double),
        ("class_lo", ctypes.c_void_p),
        ("class_hi", ctypes.c_void_p),
        ("class_cell", ctypes.c_void_p),
        ("class_seed", ctypes.c_void_p),
        ("poisson", ctypes.c_void_p),
        ("table_n", ctypes.c_int32),
        ("prototypes", ctypes.c_int32),
        ("size_table", ctypes.c_void_p),
        ("ice_table", ctypes.c_void_p),
        ("rate_table", ctypes.c_void_p),
        ("cone_table", ctypes.c_void_p),
        ("ratio_table", ctypes.c_void_p),
        ("rate_ref", ctypes.c_double),
        ("size_exponent", ctypes.c_double),
        ("axis_frame", ctypes.c_double * 9),
        ("kappa", ctypes.c_double),
        ("tumble_fraction", ctypes.c_double),
        ("keep_outs", ctypes.c_int32),
        ("keep_out", ctypes.c_void_p),
    ]


class _NativeScatter:
    """ctypes binding of ``native/scatter.c``."""

    def __init__(self, lib: ctypes.CDLL):
        self._call = lib.wb_scatter_cells
        self._call.restype = ctypes.c_int64
        self._call.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]

    def rocks(
        self, struct: _WbScatter, cls: int, cells: np.ndarray, guess: int
    ) -> np.ndarray:
        cells = np.ascontiguousarray(cells, dtype=np.int64).reshape(-1, 3)
        capacity = max(guess, 16)
        while True:
            out = np.zeros(capacity, dtype=ROCK_DTYPE)
            n = self._call(
                ctypes.addressof(struct),
                cls,
                cells.ctypes.data,
                len(cells),
                out.ctypes.data,
                capacity,
            )
            if n <= capacity:
                return out[:n]
            capacity = int(n)


def _native() -> _NativeScatter | None:
    """:returns: the native scatter, or None (numpy runs)"""
    library = noise.native_library()
    return None if library is None else _NativeScatter(library.lib)


class Scatter:
    """A field's rocks, generated on demand from its spec.

    :ivar spec: the field
    :ivar classes: its size classes
    """

    def __init__(
        self,
        spec: FieldSpec,
        keep_outs: tuple[KeepOut, ...] = (),
        native: bool | None = None,
    ):
        """
        :param spec: the field
        :param keep_outs: keep-outs added to the spec's (at runtime)
        :param native: True / False to force the native or the numpy scatter;
            None = native when the library is built
        """
        self.spec = spec
        self._tables = _Tables(spec, tuple(keep_outs))
        self.classes = self._tables.classes
        lib = _native() if native is not False else None
        if native and lib is None:
            raise RuntimeError("the native scatter is not built")
        self._lib = lib
        self._struct = self._make_struct() if lib is not None else None

    @property
    def native(self) -> bool:
        """Whether the native scatter runs."""
        return self._lib is not None

    def with_keep_outs(self, keep_outs: tuple[KeepOut, ...]) -> Scatter:
        """:returns: this field with *keep_outs* added to the spec's"""
        return Scatter(self.spec, keep_outs, native=self.native)

    def _make_struct(self) -> _WbScatter:
        t = self._tables
        s = _WbScatter()
        keep = self._keep = []

        def ptr(array: np.ndarray, dtype=np.float64) -> int:
            array = np.ascontiguousarray(array, dtype=dtype)
            keep.append(array)
            return array.ctypes.data

        shape = self.spec.shape
        if isinstance(shape, Box):
            s.shape = 0
            s.half[:] = shape.half_size
        else:
            s.shape = 1
            s.half[:] = (shape.radius, 0.0, 0.0)
        s.edge = shape.edge
        s.classes = len(t.classes)
        s.poisson_max = t.poisson_max
        if t.clump_plan is not None:
            plan = t.clump_plan
            s.clump_octaves = len(plan.octaves)
            s.clump_frames = ptr(plan.frames)
            s.clump_seeds = ptr(plan.seeds, np.uint64)
            s.clump_weights = ptr(plan.weights)
            s.clump_norm = plan.norm
        s.clump_low, s.clump_width = t.clump_low, t.clump_width
        s.class_lo = ptr([c.lo for c in t.classes])
        s.class_hi = ptr([c.hi for c in t.classes])
        s.class_cell = ptr([c.cell for c in t.classes])
        s.class_seed = ptr(t.class_seeds, np.uint64)
        s.poisson = ptr(t.poisson)
        s.table_n = TABLE_SIZE
        s.prototypes = self.spec.shapes.prototypes
        s.size_table = ptr(t.size_tables)
        s.ice_table = ptr(t.ice_table)
        s.rate_table = ptr(t.rate_table)
        s.cone_table = ptr(t.cone_table)
        s.ratio_table = ptr(t.ratio_table)
        s.rate_ref = t.rate_ref
        s.size_exponent = self.spec.spin.size_exponent
        s.axis_frame[:] = t.axis_frame.ravel()
        s.kappa = t.kappa
        s.tumble_fraction = t.tumble_fraction
        s.keep_outs = len(t.keep_outs)
        s.keep_out = ptr(t.keep_outs)
        return s

    # ── queries ───────────────────────────────────────────────────────────

    def density(self, points: ArrayLike) -> np.ndarray:
        """:returns: the share of candidates that exist at field-frame points
        (the profile times the clumping)"""
        return self._tables.density(np.asarray(points, dtype=np.float64).reshape(-1, 3))

    def rocks_in_cells(self, cls: int, cells: ArrayLike) -> np.ndarray:
        """:returns: the rocks of class *cls* whose centres are in *cells*
        ((n, 3) cell indices of that class), as :data:`ROCK_DTYPE` records"""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        if len(cells) == 0:
            return np.zeros(0, dtype=ROCK_DTYPE)
        if self._lib is not None:
            guess = int(len(cells) * self.classes[cls].per_cell * 1.5) + 16
            return self._lib.rocks(self._struct, cls, cells, guess)
        return _numpy_rocks(self._tables, cls, cells)

    def cells_in_box(
        self, cls: int, lo: ArrayLike, hi: ArrayLike, limit: int = 4_000_000
    ) -> np.ndarray:
        """:returns: (n, 3) the cells of class *cls* that overlap the box
        [lo, hi] and the field
        :raises ValueError: past *limit* cells"""
        b_lo, b_hi = self._tables.bounds
        lo = np.maximum(np.asarray(lo, dtype=np.float64), b_lo)
        hi = np.minimum(np.asarray(hi, dtype=np.float64), b_hi)
        if np.any(hi < lo):
            return np.zeros((0, 3), dtype=np.int64)
        cell = self.classes[cls].cell
        first = np.floor(lo / cell).astype(np.int64)
        last = np.floor(hi / cell).astype(np.int64)
        counts = last - first + 1
        if int(np.prod(counts)) > limit:
            raise ValueError(f"{int(np.prod(counts))} cells, more than {limit}")
        axes = [np.arange(a, b + 1) for a, b in zip(first, last)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return grid.reshape(-1, 3)

    def rocks_in_box(self, lo: ArrayLike, hi: ArrayLike) -> np.ndarray:
        """:returns: every rock whose centre is in the box [lo, hi]"""
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        found = []
        for cls in range(len(self.classes)):
            rocks = self.rocks_in_cells(cls, self.cells_in_box(cls, lo, hi))
            inside = np.all((rocks["centre"] >= lo) & (rocks["centre"] <= hi), axis=1)
            found.append(rocks[inside])
        return np.concatenate(found)

    def rocks_near(self, point: ArrayLike, radius: float) -> np.ndarray:
        """:returns: every rock that reaches within *radius* of *point*"""
        point = np.asarray(point, dtype=np.float64)
        found = []
        for cls, c in enumerate(self.classes):
            reach = radius + c.hi
            cells = self.cells_in_box(cls, point - reach, point + reach)
            rocks = self.rocks_in_cells(cls, cells)
            d = rocks["centre"] - point
            dist2 = np.sum(d * d, axis=1)
            found.append(rocks[dist2 < (radius + rocks["radius"]) ** 2])
        return np.concatenate(found)

    def is_clear(self, point: ArrayLike, radius: float) -> bool:
        """:returns: whether no rock reaches within *radius* of *point*"""
        return len(self.rocks_near(point, radius)) == 0

    def find_clear(
        self,
        point: ArrayLike,
        radius: float,
        search: float,
        step: float | None = None,
    ) -> np.ndarray | None:
        """The nearest place to *point* where a ball of *radius* touches no
        rock: *point* itself if clear, else the first clear point on spheres
        of growing radius around it (for spawning ships in the field).

        :param point: where wanted, field frame
        :param radius: the clearance needed
        :param search: how far from *point* to look
        :param step: the spacing of the points tried (default: *radius*)
        :returns: the point, or None if none within *search*
        """
        point = np.asarray(point, dtype=np.float64)
        if self.is_clear(point, radius):
            return point
        step = radius if step is None else step
        shell = step
        while shell <= search:
            count = max(
                8, int(math.ceil(4.0 * math.pi * shell * shell / (step * step)))
            )
            for direction in _fibonacci_sphere(count):
                candidate = point + shell * direction
                if self.is_clear(candidate, radius):
                    return candidate
            shell += step
        return None


def _fibonacci_sphere(n: int) -> np.ndarray:
    """:returns: (n, 3) unit vectors spread evenly over the sphere"""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1)
