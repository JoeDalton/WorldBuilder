"""
The rocks around a camera: which are in range, kept in slots, and which are
drawn this frame at which level of detail. Pure numpy (the Panda side is
:mod:`world_builder.asteroids.field`).

- **Ranges.** A rock of radius r is drawn while its diameter covers at least
  ``min_pixels`` on screen: out to ``2 r * pixels_per_radian / min_pixels``.
  Each size class keeps the cells within its largest rock's range: a box of
  cells around the camera (its **window**), clipped to the field. Small rocks
  are kept near, big ones far.
- **Windows move by their difference.** When the camera crosses a cell, only
  the cells entering the window are scattered, and the rocks of the cells
  leaving it are freed. A window grows toward its range by whole layers of
  cells, at most about :data:`CELLS_PER_FRAME` new cells per frame (each
  class at least one layer), so a first frame in a ring 20,000 km wide, or a
  budget's step, is spread over frames. :attr:`Population.settled` tells
  when every window has reached its range.
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

from dataclasses import dataclass

import numpy as np

from world_builder.asteroids.motion import orientation
from world_builder.asteroids.scatter import ROCK_DTYPE, Scatter

#: New cells scattered per frame, shared by the classes (each always adds a
#: layer): about 7 ms of native scatter.
CELLS_PER_FRAME = 30_000
#: A coarser level needs its error this much under the limit to be taken.
LOD_HYSTERESIS = 1.15
#: The table's floats per slot: 4 RGBA texels.
SLOT_FLOATS = 16


@dataclass
class Visible:
    """One frame's visible pass.

    :ivar slots: the drawn slots, grouped by bucket (prototype * levels +
        level), in bucket order
    :ivar counts: (prototypes * levels,) slots per bucket
    :ivar in_range: rocks within their drawing distance (any direction)
    """

    slots: np.ndarray
    counts: np.ndarray
    in_range: int


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
        self.cells = np.zeros((0, 3), dtype=np.int64)
        self.lod = np.zeros(0, dtype=np.int8)
        self._free: list[int] = []
        self._grow(capacity)
        classes = len(scatter.classes)
        #: per class: the cells held, (first, last) cell indices; None = none
        self._windows: list[_Window | None] = [None] * classes
        #: whether every window has reached its range
        self.settled = False

    # ── slots ─────────────────────────────────────────────────────────────

    def _grow(self, capacity: int) -> None:
        old = self.capacity
        self.live = np.concatenate((self.live, np.zeros(capacity - old, dtype=bool)))
        self.rocks = np.concatenate(
            (self.rocks, np.zeros(capacity - old, dtype=ROCK_DTYPE))
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
        :param cells_per_frame: new cells, at most, shared by the classes
            (each adds at least one layer of its window)
        :returns: the slots filled and the slots freed
        """
        camera = np.asarray(camera, dtype=np.float64)
        b_lo, b_hi = self.scatter.spec.shape.bounds()
        added, removed = [], []
        settled = True
        quota = max(1, cells_per_frame // len(self.scatter.classes))
        for cls, c in enumerate(self.scatter.classes):
            lo = np.maximum(camera - ranges[cls], b_lo)
            hi = np.minimum(camera + ranges[cls], b_hi)
            target = (
                None
                if np.any(hi < lo)
                else _Window(
                    np.floor(lo / c.cell).astype(np.int64),
                    np.floor(hi / c.cell).astype(np.int64),
                )
            )
            old = self._windows[cls]
            home = np.floor(camera / c.cell).astype(np.int64)
            window = _toward(old, target, home, quota)
            settled &= window == target
            if window == old:
                continue
            self._windows[cls] = window
            # Only a window that lost cells frees rocks
            grew = old is not None and window is not None and window.meet(old) == old
            held = (
                np.zeros(0, dtype=np.int64)
                if grew
                else np.flatnonzero(self.live & (self.rocks["cls"] == cls))
            )
            if len(held):
                free = held[~_inside(self.cells[held], window)]
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

    def visible(
        self,
        camera: np.ndarray,
        planes: np.ndarray,
        pixels_per_radian: float,
        min_pixels: float,
        lod_error_px: float,
        errors: np.ndarray,
        ready: np.ndarray | None = None,
    ) -> Visible:
        """The rocks drawn this frame.

        :param camera: (3,) the camera, field frame
        :param planes: (cameras, 4, 3) each view's side planes' inward unit
            normals, field frame, through the camera
        :param pixels_per_radian: the main view's
        :param min_pixels: the smallest diameter drawn
        :param lod_error_px: the error allowed for a coarser level
        :param errors: (prototypes, levels) each level's error to the finest,
            in units of the radius, coarse to fine (the finest 0)
        :param ready: (n,) slots whose table rows are on the GPU; None = all
        """
        prototypes, levels = errors.shape
        slots = np.flatnonzero(self.live if ready is None else self.live & ready)
        rel = self.rocks["centre"][slots] - camera
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        radius = self.rocks["radius"][slots]
        scale = (
            radius * pixels_per_radian / np.maximum(dist, 1e-6)
        )  # px per unit radius
        big = 2.0 * scale >= min_pixels
        in_range = int(np.count_nonzero(big))
        # inside some view: no side plane has the whole rock behind it
        inside = np.zeros(len(slots), dtype=bool)
        for view in planes:
            ahead = np.ones(len(slots), dtype=bool)
            for normal in view:
                ahead &= rel @ normal >= -radius
            inside |= ahead
        keep = big & inside
        slots, scale = slots[keep], scale[keep]
        proto = self.rocks["proto"][slots]
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
        return Visible(slots=slots[order], counts=counts, in_range=in_range)


# ── windows ─────────────────────────────────────────────────────────────────


class _Window:
    """A box of cells: indices *first* to *last*, inclusive, per axis."""

    __slots__ = ("first", "last")

    def __init__(self, first: np.ndarray, last: np.ndarray):
        self.first = np.asarray(first, dtype=np.int64)
        self.last = np.asarray(last, dtype=np.int64)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _Window)
            and np.array_equal(self.first, other.first)
            and np.array_equal(self.last, other.last)
        )

    @property
    def cells(self) -> int:
        return int(np.prod(np.maximum(self.last - self.first + 1, 0)))

    def meet(self, other: _Window | None) -> _Window | None:
        """:returns: the cells in both, or None"""
        if other is None:
            return None
        first = np.maximum(self.first, other.first)
        last = np.minimum(self.last, other.last)
        return None if np.any(last < first) else _Window(first, last)

    def grid(self) -> np.ndarray:
        """:returns: (n, 3) its cells"""
        axes = [np.arange(a, b + 1) for a, b in zip(self.first, self.last)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _toward(
    old: _Window | None, target: _Window | None, home: np.ndarray, budget: int
) -> _Window | None:
    """The window this frame: what of *old* is still in *target*, grown
    toward it by whole layers while the new cells stay within *budget* (at
    least one layer). With nothing of *old* in it, it starts from the cell of
    *target* nearest the camera's (*home*)."""
    if target is None:
        return None
    start = np.clip(home, target.first, target.last)
    window = target.meet(old)
    added = 0
    if window is None or not _inside(start[None, :], window)[0]:
        window, added = _Window(start, start), 1
    while window != target:
        grown = target.meet(_Window(window.first - 1, window.last + 1))
        more = grown.cells - window.cells
        if added and added + more > budget:
            break
        window, added = grown, added + more
    return window


def _inside(cells: np.ndarray, window: _Window | None) -> np.ndarray:
    """:returns: (n,) whether each of *cells* (n, 3) is in *window*"""
    if window is None:
        return np.zeros(len(cells), dtype=bool)
    return np.all((cells >= window.first) & (cells <= window.last), axis=1)


def _difference(new: _Window | None, old: _Window | None) -> np.ndarray:
    """:returns: (n, 3) the cells of *new* not in *old*: up to six slabs"""
    if new is None:
        return np.zeros((0, 3), dtype=np.int64)
    common = new.meet(old)
    if common is None:
        return new.grid()
    slabs = []
    first, last = new.first.copy(), new.last.copy()
    for axis in range(3):
        if first[axis] < common.first[axis]:
            lo, hi = first.copy(), last.copy()
            hi[axis] = common.first[axis] - 1
            slabs.append(_Window(lo, hi).grid())
        if common.last[axis] < last[axis]:
            lo, hi = first.copy(), last.copy()
            lo[axis] = common.last[axis] + 1
            slabs.append(_Window(lo, hi).grid())
        first[axis], last[axis] = common.first[axis], common.last[axis]
    if not slabs:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(slabs)
