"""
The rocks around a camera: which are in range, kept in slots, and which are
drawn this frame at which level of detail. Pure numpy (the Panda side is
:mod:`world_builder.asteroids.field`).

- **Ranges.** A rock of radius r is drawn while its diameter covers at least
  ``min_pixels`` on screen: out to ``2 r * pixels_per_radian / min_pixels``.
  Each size class keeps the cells within its largest rock's range (a box of
  cells around the camera, clipped to the field): small rocks near, big ones
  far.
- **Slots.** Each rock of those cells holds a slot (a row of the instance
  table) until its cell leaves the range. Slots are reused; the table only
  grows.
- **Visible pass.** Per frame, over the rocks in slots: big enough on
  screen, inside some camera's view (the main one, a rear-view mirror), and
  at the coarsest level of detail whose error stays under
  ``lod_error_px`` pixels (with 15 % hysteresis against flicker). The result
  is the slots grouped by (prototype, level): one draw each.

Cells are keyed by their three indices packed into one int64 (21 bits each),
so cell sets are sorted integer arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_builder.asteroids.motion import orientation
from world_builder.asteroids.scatter import ROCK_DTYPE, Scatter

#: Cell indices must stay within +-2^20 per axis (keys pack 21 bits each).
_KEY_BITS = 21
_KEY_OFFSET = 1 << (_KEY_BITS - 1)
#: A coarser level needs its error this much under the limit to be taken.
LOD_HYSTERESIS = 1.15
#: The table's floats per slot: 4 RGBA texels.
SLOT_FLOATS = 16


def cell_keys(cells: np.ndarray) -> np.ndarray:
    """:returns: (n,) int64 keys of (n, 3) cell indices"""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if np.any(np.abs(cells) >= _KEY_OFFSET):
        raise ValueError("cell index out of range for the key packing")
    shifted = cells + _KEY_OFFSET
    return (
        (shifted[:, 0] << (2 * _KEY_BITS))
        | (shifted[:, 1] << _KEY_BITS)
        | shifted[:, 2]
    )


def key_cells(keys: np.ndarray) -> np.ndarray:
    """:returns: (n, 3) cell indices of :func:`cell_keys` keys"""
    keys = np.asarray(keys, dtype=np.int64)
    mask = (1 << _KEY_BITS) - 1
    cells = np.stack(
        (keys >> (2 * _KEY_BITS), (keys >> _KEY_BITS) & mask, keys & mask), 1
    )
    return cells - _KEY_OFFSET


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
        self.keys = np.zeros(0, dtype=np.int64)
        self.lod = np.zeros(0, dtype=np.int8)
        self._free: list[int] = []
        self._grow(capacity)
        classes = len(scatter.classes)
        #: per class: the sorted keys of the cells held, and their window
        self._cells = [np.zeros(0, dtype=np.int64) for _ in range(classes)]
        self._windows: list[tuple | None] = [None] * classes

    # ── slots ─────────────────────────────────────────────────────────────

    def _grow(self, capacity: int) -> None:
        old = self.capacity
        self.live = np.concatenate((self.live, np.zeros(capacity - old, dtype=bool)))
        self.rocks = np.concatenate(
            (self.rocks, np.zeros(capacity - old, dtype=ROCK_DTYPE))
        )
        self.keys = np.concatenate(
            (self.keys, np.zeros(capacity - old, dtype=np.int64))
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
        self, camera: np.ndarray, ranges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hold the cells within each class's range of the camera.

        :param camera: (3,) the camera, field frame
        :param ranges: (classes,) from :meth:`ranges`
        :returns: the slots filled and the slots freed
        """
        camera = np.asarray(camera, dtype=np.float64)
        b_lo, b_hi = self.scatter.spec.shape.bounds()
        added, removed = [], []
        for cls, c in enumerate(self.scatter.classes):
            lo = np.maximum(camera - ranges[cls], b_lo)
            hi = np.minimum(camera + ranges[cls], b_hi)
            if np.any(hi < lo):
                window = None
            else:
                window = (
                    tuple(np.floor(lo / c.cell).astype(np.int64)),
                    tuple(np.floor(hi / c.cell).astype(np.int64)),
                )
            if window == self._windows[cls]:
                continue
            self._windows[cls] = window
            if window is None:
                keys = np.zeros(0, dtype=np.int64)
            else:
                axes = [np.arange(a, b + 1) for a, b in zip(*window)]
                grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
                    -1, 3
                )
                keys = np.sort(cell_keys(grid))
            old = self._cells[cls]
            gone = np.setdiff1d(old, keys, assume_unique=True)
            new = np.setdiff1d(keys, old, assume_unique=True)
            self._cells[cls] = keys
            if len(gone):
                held = np.flatnonzero(self.live & (self.rocks["cls"] == cls))
                free = held[np.isin(self.keys[held], gone)]
                self.live[free] = False
                self.lod[free] = -1
                self._free.extend(free.tolist())
                removed.append(free)
            if len(new):
                rocks = self.scatter.rocks_in_cells(cls, key_cells(new))
                slots = self._allocate(len(rocks))
                self.rocks[slots] = rocks
                own = np.floor(rocks["centre"] / c.cell).astype(np.int64)
                self.keys[slots] = cell_keys(own)
                self.live[slots] = True
                self.lod[slots] = -1
                added.append(slots)
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
        rocks = self.rocks[slots]
        rel = rocks["centre"] - camera
        dist = np.sqrt(np.sum(rel * rel, axis=1))
        radius = rocks["radius"]
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
