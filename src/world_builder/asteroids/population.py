"""
The rocks around a camera: which are in range, kept in slots, and which are
drawn this frame at which level of detail. Pure numpy (the Panda side is
:mod:`world_builder.asteroids.field`).

- **Ranges.** A rock of radius r is drawn while its diameter covers at least
  ``min_pixels`` on screen: out to ``2 r * pixels_per_radian / min_pixels``.
  Each size class keeps the cells within its largest rock's range: a ball of
  cells around the camera (its **window**: every cell within the range of
  some point of the camera's cell), clipped to the field. Small rocks are
  kept near, big ones far.
- **Windows move by their difference.** A window is its centre cell and its
  radius: each row of cells (along x) is one interval, worked out from them.
  When the camera crosses a cell, only the cells entering the window are
  scattered, and the rocks of the cells leaving it are freed. A window grows
  toward its range at most about :data:`CELLS_PER_FRAME` new cells a frame
  (over all classes; moving with the camera is always done), so a first
  frame in a ring 20,000 km wide, or a budget's step, is spread over
  frames. :attr:`Population.settled` tells when every window has reached
  its range.
- **Slots.** Each rock of those cells holds a slot (a row of the instance
  table) until its cell leaves the window. Slots are reused; the table only
  grows.
- **Visible pass.** Per frame, over the rocks in slots: big enough on
  screen, inside some camera's view (the main one, a rear-view mirror), and
  at the coarsest level of detail whose error stays under
  ``lod_error_px`` pixels (with 15 % hysteresis against flicker). The result
  is the slots grouped by (prototype, level): one draw each.
"""

from __future__ import annotations

import ctypes
import functools
import math
from dataclasses import dataclass

import numpy as np

from world_builder.asteroids.motion import orientation
from world_builder.asteroids.scatter import ROCK_DTYPE, Scatter
from world_builder.terrain import noise

#: A window this near its range counts as settled (its last cells come in
#: over the next frames, the haze standing in for their rocks meanwhile).
SETTLED = 0.9
#: New cells scattered per frame to grow the windows, shared by the classes:
#: about 7 ms of native scatter.
CELLS_PER_FRAME = 30_000
#: Rocks fade out over the last this share of the field's ``far``, into its
#: haze (asteroid.vert, haze.frag).
FAR_FADE = 0.2
#: A coarser level needs its error this much under the limit to be taken.
LOD_HYSTERESIS = 1.15
#: What the visible pass reads of each slot (``WbSlot`` in visible.c): the
#: rock's centre, radius, class and prototype, 40 bytes against its record's
#: 128 (the pass reads every slot every frame).
SLOT_DTYPE = np.dtype(
    [("centre", "<f8", 3), ("radius", "<f8"), ("cls", "<i4"), ("proto", "<i4")]
)
#: The table's floats per slot: 4 RGBA texels.
SLOT_FLOATS = 16


@dataclass
class Visible:
    """One frame's visible pass.

    :ivar slots: the drawn slots, grouped by bucket (prototype * levels +
        level), in bucket order
    :ivar counts: (prototypes * levels,) slots per bucket
    :ivar in_range: rocks within their drawing distance (any direction)
    :ivar near: the slots held (drawn or not) of big rocks near the camera:
        radius at least the pass's ``near_radius``, closer than
        ``near_factor`` radii (those that may be drawn as terrain)
    """

    slots: np.ndarray
    counts: np.ndarray
    in_range: int
    near: np.ndarray


class Population:
    """The rocks of a field around a camera, in slots.

    :ivar scatter: the field's rocks
    :ivar capacity: slots allocated (the instance table's rows)
    """

    def __init__(self, scatter: Scatter, capacity: int = 1024):
        self.scatter = scatter
        self.capacity = 0
        self.live = np.zeros(0, dtype=bool)
        self.rocks = np.zeros(0, dtype=ROCK_DTYPE)
        #: per slot, what the visible pass reads (:data:`SLOT_DTYPE`)
        self.compact = np.zeros(0, dtype=SLOT_DTYPE)
        #: per slot, its size class (contiguous: scanned when a window moves)
        self.slot_cls = np.zeros(0, dtype=np.int8)
        self.cells = np.zeros((0, 3), dtype=np.int64)
        self.lod = np.zeros(0, dtype=np.int8)
        self._free: list[int] = []
        self._grow(capacity)
        classes = len(scatter.classes)
        #: per class: the cells held; None = none yet
        self._windows: list[_Ball | None] = [None] * classes
        #: whether every window has reached its range
        self.settled = False

    # ── slots ─────────────────────────────────────────────────────────────

    def _grow(self, capacity: int) -> None:
        old = self.capacity
        self.live = np.concatenate((self.live, np.zeros(capacity - old, dtype=bool)))
        self.rocks = np.concatenate(
            (self.rocks, np.zeros(capacity - old, dtype=ROCK_DTYPE))
        )
        self.compact = np.concatenate(
            (self.compact, np.zeros(capacity - old, dtype=SLOT_DTYPE))
        )
        self.slot_cls = np.concatenate(
            (self.slot_cls, np.zeros(capacity - old, dtype=np.int8))
        )
        self.cells = np.concatenate(
            (self.cells, np.zeros((capacity - old, 3), dtype=np.int64))
        )
        self.lod = np.concatenate(
            (self.lod, np.full(capacity - old, -1, dtype=np.int8))
        )
        self._free.extend(range(capacity - 1, old - 1, -1))
        self.capacity = capacity

    def _allocate(self, count: int) -> np.ndarray:
        if count > len(self._free):
            self._grow(max(2 * self.capacity, self.capacity + count))
        slots = (
            np.array(self._free[-count:][::-1], dtype=np.int64)
            if count
            else np.zeros(0, np.int64)
        )
        del self._free[len(self._free) - count :]
        return slots

    @property
    def count(self) -> int:
        """Slots in use."""
        return int(np.count_nonzero(self.live))

    # ── ranges ────────────────────────────────────────────────────────────

    def ranges(
        self, pixels_per_radian: float, min_pixels: float, far: float
    ) -> np.ndarray:
        """:returns: (classes,) how far each class's cells are kept: its
        largest rock's drawing distance (and its own size), at most *far*"""
        hi = np.array([c.hi for c in self.scatter.classes])
        reach = 2.0 * hi * pixels_per_radian / max(min_pixels, 1e-6)
        return np.minimum(reach, far) + hi

    def update(
        self,
        camera: np.ndarray,
        ranges: np.ndarray,
        cells_per_frame: int = CELLS_PER_FRAME,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move each class's window toward the cells within its range of the
        camera.

        :param camera: (3,) the camera, field frame
        :param ranges: (classes,) from :meth:`ranges`
        :param cells_per_frame: new cells to grow the windows, at most,
            shared by the classes (moving with the camera is always done)
        :returns: the slots filled and the slots freed
        """
        camera = np.asarray(camera, dtype=np.float64)
        b_lo, b_hi = self.scatter.spec.shape.bounds()
        added, removed = [], []
        settled = True
        left = cells_per_frame  # what a class leaves goes to the next
        classes = len(self.scatter.classes)
        for cls, c in enumerate(self.scatter.classes):
            quota = max(1, left // (classes - cls))
            first = np.floor(b_lo / c.cell).astype(np.int64)
            last = np.floor(b_hi / c.cell).astype(np.int64)
            old = self._windows[cls]
            home = np.floor(camera / c.cell).astype(np.int64)
            target = float(ranges[cls])
            window = _toward(old, target, home, c.cell, first, last, quota)
            left -= min(_count(window, old), quota)
            settled &= window.radius >= SETTLED * target
            if window == old:
                continue
            self._windows[cls] = window
            # Only a window that lost cells frees rocks
            grew = (
                old is not None
                and np.array_equal(old.home, window.home)
                and window.radius >= old.radius
            )
            held = (
                np.zeros(0, dtype=np.int64)
                if grew
                else np.flatnonzero(self.live & (self.slot_cls == cls))
            )
            if len(held):
                free = held[~window.contains(self.cells[held])]
                if len(free):
                    self.live[free] = False
                    self.lod[free] = -1
                    self._free.extend(free.tolist())
                    removed.append(free)
            new = _difference(window, old)
            if len(new):
                rocks = self.scatter.rocks_in_cells(cls, new)
                slots = self._allocate(len(rocks))
                self.rocks[slots] = rocks
                for name in ("centre", "radius", "cls", "proto"):
                    self.compact[name][slots] = rocks[name]
                self.slot_cls[slots] = cls
                self.cells[slots] = np.floor(rocks["centre"] / c.cell).astype(np.int64)
                self.live[slots] = True
                self.lod[slots] = -1
                added.append(slots)
        self.settled = settled
        empty = np.zeros(0, dtype=np.int64)
        return (
            np.concatenate(added) if added else empty,
            np.concatenate(removed) if removed else empty,
        )

    # ── the GPU's table ───────────────────────────────────────────────────

    def table_rows(self, slots: np.ndarray, epoch: float) -> np.ndarray:
        """The instance table's rows for *slots*: per slot, 4 RGBA texels
        (``asteroid.vert``):

        - the centre relative to its cell's corner, the radius;
        - the orientation at *epoch*;
        - the precession (axis times rate), the spin;
        - the cell's indices, then class, prototype and ice packed in one
          float (class << 18 | prototype << 6 | ice in 64ths).

        :returns: (n, 16) float32
        """
        rocks = self.rocks[slots]
        cell = np.array([c.cell for c in self.scatter.classes])[rocks["cls"]]
        own = np.floor(rocks["centre"] / cell[:, None])
        local = rocks["centre"] - own * cell[:, None]
        q = orientation(
            rocks["q0"], rocks["axis"], rocks["precession"], rocks["spin"], epoch
        )
        ice = np.rint(np.clip(rocks["ice"], 0.0, 1.0) * 63.0).astype(np.int64)
        packed = (
            (rocks["cls"].astype(np.int64) << 18)
            | (rocks["proto"].astype(np.int64) << 6)
            | ice
        )
        rows = np.empty((len(slots), SLOT_FLOATS), dtype=np.float32)
        rows[:, 0:3] = local
        rows[:, 3] = rocks["radius"]
        rows[:, 4:8] = q
        rows[:, 8:11] = rocks["axis"] * rocks["precession"][:, None]
        rows[:, 11] = rocks["spin"]
        rows[:, 12:15] = own
        rows[:, 15] = packed
        return rows

    # ── the visible pass ──────────────────────────────────────────────────

    def limits(self, far: float = math.inf) -> np.ndarray:
        """:returns: (classes,) how far each class's rocks are all held (its
        window's radius, while it grows), at most *far*: none farther is
        drawn, and they fade out over the last :data:`FAR_FADE` of it"""
        return np.array(
            [min(0.0 if w is None else w.radius, far) for w in self._windows]
        )

    def visible(
        self,
        camera: np.ndarray,
        planes: np.ndarray,
        pixels_per_radian: float,
        min_pixels: float,
        lod_error_px: float,
        errors: np.ndarray,
        ready: np.ndarray | None = None,
        far: float = math.inf,
        native: bool | None = None,
        near_radius: float = math.inf,
        near_factor: float = 0.0,
    ) -> Visible:
        """The rocks drawn this frame (``native/visible.c``, or numpy).

        :param camera: (3,) the camera, field frame
        :param planes: (cameras, 4, 3) each view's side planes' inward unit
            normals, field frame, through the camera
        :param pixels_per_radian: the main view's
        :param min_pixels: the smallest diameter drawn
        :param lod_error_px: the error allowed for a coarser level
        :param errors: (prototypes, levels) each level's error to the finest,
            in units of the radius, coarse to fine (the finest 0)
        :param ready: (n,) slots whose table rows are on the GPU; None = all
        :param far: no rock farther is drawn (see :meth:`limits`)
        :param native: True / False to force the native or the numpy pass;
            None = native when the library is built
        :param near_radius: list the rocks held at least this big ...
        :param near_factor: ... closer than this many radii
            (:attr:`Visible.near`)
        """
        usable = self.live if ready is None else self.live & ready
        limits = self.limits(far)
        lib = _native_visible() if native is not False else None
        if native and lib is None:
            raise RuntimeError("the native visible pass is not built")
        if lib is not None:
            return lib(
                self,
                usable,
                camera,
                planes,
                (pixels_per_radian, min_pixels, lod_error_px),
                errors,
                limits,
                (near_radius, near_factor),
            )
        prototypes, levels = errors.shape
        held = np.flatnonzero(self.live)
        centre = self.compact["centre"][held]
        x, y, z = (centre[:, i] - camera[i] for i in range(3))
        dist = np.sqrt(x * x + y * y + z * z)  # visible.c's order
        radius = self.compact["radius"][held]
        near = held[(radius >= near_radius) & (dist < near_factor * radius)]
        on = usable[held]
        slots, x, y, z, dist, radius = (a[on] for a in (held, x, y, z, dist, radius))
        scale = (
            radius * pixels_per_radian / np.maximum(dist, 1e-6)
        )  # px per unit radius
        cls = self.compact["cls"][slots]
        big = (2.0 * scale >= min_pixels) & (dist <= limits[cls])
        in_range = int(np.count_nonzero(big))
        # inside some view: no side plane has the whole rock behind it
        inside = np.zeros(len(slots), dtype=bool)
        for view in planes:
            ahead = np.ones(len(slots), dtype=bool)
            for n in view:
                ahead &= x * n[0] + y * n[1] + z * n[2] >= -radius
            inside |= ahead
        keep = big & inside
        slots, scale = slots[keep], scale[keep]
        proto = self.compact["proto"][slots]
        pixels = errors[proto] * scale[:, None]  # (n, levels)
        fits = pixels <= lod_error_px
        previous = self.lod[slots].astype(np.int64)
        coarser = np.arange(levels)[None, :] < previous[:, None]
        fits &= ~coarser | (pixels * LOD_HYSTERESIS <= lod_error_px)
        fits[:, -1] = True
        lod = np.argmax(fits, axis=1)
        self.lod[slots] = lod
        bucket = proto * levels + lod
        order = np.argsort(bucket, kind="stable")
        counts = np.bincount(bucket, minlength=prototypes * levels)
        return Visible(slots=slots[order], counts=counts, in_range=in_range, near=near)


# ── the native pass ─────────────────────────────────────────────────────────


class _NativeVisible:
    """ctypes binding of ``native/visible.c``."""

    def __init__(self, lib: ctypes.CDLL):
        self._call = lib.wb_asteroid_visible
        self._call.restype = ctypes.c_int64
        p, f8, i32, i64 = (
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_int64,
        )
        # slots live usable n lod camera planes views ppr min_px lod_error
        # hysteresis errors prototypes levels limits near_radius near_factor
        # near near_count out counts in_range
        self._call.argtypes = [p, p, p, i64, p, p, p, i32, f8, f8, f8, f8, p, i32,
                               i32, p, f8, f8, p, p, p, p, p]  # fmt: skip

    def __call__(
        self, pop, usable, camera, planes, scales, errors, limits, near_args
    ) -> Visible:
        ppr, min_pixels, lod_error_px = scales
        near_radius, near_factor = near_args
        prototypes, levels = errors.shape
        live = pop.live.view(np.uint8)
        usable = np.ascontiguousarray(usable, dtype=np.uint8)
        camera = np.ascontiguousarray(camera, dtype=np.float64)
        planes = np.ascontiguousarray(planes, dtype=np.float64).reshape(-1, 4, 3)
        errors = np.ascontiguousarray(errors, dtype=np.float64)
        limits = np.ascontiguousarray(limits, dtype=np.float64)
        out = np.empty(pop.capacity, dtype=np.int64)
        near = np.empty(pop.capacity, dtype=np.int64)
        counts = np.empty(prototypes * levels, dtype=np.int64)
        in_range, near_count = ctypes.c_int64(), ctypes.c_int64()
        drawn = self._call(
            pop.compact.ctypes.data, live.ctypes.data, usable.ctypes.data,
            pop.capacity, pop.lod.ctypes.data, camera.ctypes.data,
            planes.ctypes.data, len(planes), ppr, min_pixels, lod_error_px,
            LOD_HYSTERESIS, errors.ctypes.data, prototypes, levels,
            limits.ctypes.data, near_radius, near_factor, near.ctypes.data,
            ctypes.byref(near_count), out.ctypes.data, counts.ctypes.data,
            ctypes.byref(in_range),
        )  # fmt: skip
        if drawn < 0:
            raise MemoryError("the native visible pass ran out of memory")
        return Visible(
            slots=out[:drawn].copy(),
            counts=counts,
            in_range=in_range.value,
            near=near[: near_count.value].copy(),
        )


@functools.cache
def _native_visible() -> _NativeVisible | None:
    """:returns: the native visible pass, or None (numpy runs)"""
    library = noise.native_library()
    return None if library is None else _NativeVisible(library.lib)


# ── windows ─────────────────────────────────────────────────────────────────


class _Ball:
    """The cells (of edge *cell*) within *radius* of some point of the cell
    *home*, clipped to the cells *first* to *last*: a cell at index offset o
    from home is in if sum((max(|o| - 1, 0) * cell)^2) <= radius^2. Each row
    of cells along x is one interval."""

    __slots__ = ("home", "radius", "cell", "first", "last")

    def __init__(self, home, radius: float, cell: float, first, last):
        self.home = np.asarray(home, dtype=np.int64)
        self.radius = float(radius)
        self.cell = float(cell)
        self.first = np.asarray(first, dtype=np.int64)
        self.last = np.asarray(last, dtype=np.int64)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _Ball)
            and self.radius == other.radius
            and np.array_equal(self.home, other.home)
        )

    def span(self, j: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """:returns: the first and last x index of the rows (j, k) in the
        ball (first > last for none)"""
        dy = np.maximum(np.abs(j - self.home[1]) - 1, 0) * self.cell
        dz = np.maximum(np.abs(k - self.home[2]) - 1, 0) * self.cell
        rest = self.radius * self.radius - dy * dy - dz * dz
        reach = np.floor(np.sqrt(np.maximum(rest, 0.0)) / self.cell).astype(np.int64)
        a = np.maximum(self.home[0] - reach - 1, self.first[0])
        b = np.minimum(self.home[0] + reach + 1, self.last[0])
        out = (
            (rest < 0.0)
            | (j < self.first[1])
            | (j > self.last[1])
            | (k < self.first[2])
            | (k > self.last[2])
        )
        return np.where(out, 1, a), np.where(out, 0, b)

    def rows(self) -> tuple[np.ndarray, ...]:
        """:returns: (j, k, first, last) of its non-empty rows"""
        reach = int(self.radius // self.cell) + 1
        js = np.arange(
            max(self.home[1] - reach, self.first[1]),
            min(self.home[1] + reach, self.last[1]) + 1,
        )
        ks = np.arange(
            max(self.home[2] - reach, self.first[2]),
            min(self.home[2] + reach, self.last[2]) + 1,
        )
        j, k = (g.ravel() for g in np.meshgrid(js, ks, indexing="ij"))
        a, b = self.span(j, k)
        keep = a <= b
        return j[keep], k[keep], a[keep], b[keep]

    def contains(self, cells: np.ndarray) -> np.ndarray:
        """:returns: (n,) whether each of *cells* (n, 3) is in it"""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        a, b = self.span(cells[:, 1], cells[:, 2])
        return (cells[:, 0] >= a) & (cells[:, 0] <= b)

    def grid(self) -> np.ndarray:
        """:returns: (n, 3) its cells"""
        return _difference(self, None)


def _pieces(new: _Ball, old: _Ball | None) -> tuple[np.ndarray, ...]:
    """:returns: (j, k, first, last): the intervals of *new* not in *old*"""
    j, k, a, b = new.rows()
    if old is None:
        return j, k, a, b
    a_old, b_old = old.span(j, k)
    none = a_old > b_old
    # left of the old interval (or all of the row), and right of it
    left_b = np.where(none, b, np.minimum(b, a_old - 1))
    right_a = np.where(none, b + 1, np.maximum(a, b_old + 1))
    return (
        np.concatenate((j, j)),
        np.concatenate((k, k)),
        np.concatenate((a, right_a)),
        np.concatenate((left_b, b)),
    )


def _count(new: _Ball, old: _Ball | None) -> int:
    """:returns: how many cells of *new* are not in *old*"""
    _, _, a, b = _pieces(new, old)
    return int(np.maximum(b - a + 1, 0).sum())


def _difference(new: _Ball, old: _Ball | None) -> np.ndarray:
    """:returns: (n, 3) the cells of *new* not in *old*"""
    j, k, a, b = _pieces(new, old)
    length = np.maximum(b - a + 1, 0)
    keep = length > 0
    j, k, a, length = j[keep], k[keep], a[keep], length[keep]
    row = np.repeat(np.arange(len(length)), length)
    start = np.repeat(np.cumsum(length) - length, length)
    i = a[row] + (np.arange(int(length.sum())) - start)
    return np.stack((i, j[row], k[row]), axis=1)


def _toward(
    old: _Ball | None,
    target: float,
    home: np.ndarray,
    cell: float,
    first: np.ndarray,
    last: np.ndarray,
    budget: int,
) -> _Ball:
    """The window this frame: around the camera's cell (*home*), as wide as
    *old* was (no wider than *target*, the range), grown toward *target*
    while its new cells stay within *budget* (beyond those of the move
    itself), by a cell's width at least. Starts again from the camera's cell
    where it jumped farther than *old*'s radius (the camera's cell may lie
    outside the field: above a ring)."""
    radius = 0.0
    if old is not None:
        jump = float(np.max(np.abs(home - old.home))) * cell
        if jump <= old.radius:
            radius = min(old.radius, target)
    moved = _Ball(home, radius, cell, first, last)
    if radius >= target:
        return moved
    allowed = max(budget, _count(moved, old))
    grown = target
    while True:
        window = _Ball(home, grown, cell, first, last)
        if _count(window, old) <= allowed:
            return window
        if grown - radius <= cell:  # a cell's width at least
            return window
        grown = radius + 0.5 * (grown - radius)
