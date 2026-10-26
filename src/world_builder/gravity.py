"""
The world's gravity field: point masses, summed.

    g(x) = -G * sum_i m_i * (x - p_i) / |x - p_i|^3

The raw acceleration, not normalised (``-g`` is "up" for whoever wants a
direction). Each body is a point mass at its centre, except that the field
inside a body stops growing: the distance is floored at the body's radius, so
the field is finite at a centre and equals the surface value anywhere inside.

Evaluated for many points at once, in float64 (world coordinates are large:
differences must be taken before anything is narrowed). The native kernel
(``native/gravity.c``) fuses the whole sum in one pass; numpy runs without it.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from world_builder.terrain import noise


class _HasMass(Protocol):
    position: np.ndarray
    mass: float
    radius: float


class _NativeGravity:
    """ctypes binding of ``native/gravity.c``."""

    def __init__(self, lib: ctypes.CDLL):
        self._call = lib.wb_gravity
        self._call.restype = None
        self._call.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]

    def __call__(self, points: np.ndarray, bodies: np.ndarray, out: np.ndarray) -> None:
        self._call(
            points.ctypes.data,
            len(points),
            bodies.ctypes.data,
            len(bodies),
            out.ctypes.data,
        )


def _native() -> _NativeGravity | None:
    """:returns: the native kernel, or None (numpy runs)"""
    library = noise.native_library()
    return None if library is None else _NativeGravity(library.lib)


class Gravity:
    """The gravity of a set of point masses.

    :ivar g_constant: the gravitational constant, in the caller's units
    """

    def __init__(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        radii: ArrayLike,
        g_constant: float = 1.0,
    ):
        """
        :param positions: (B, 3) centres, world frame
        :param masses: (B,) masses
        :param radii: (B,) radii: the field is capped inside them
        :param g_constant: G
        """
        self.positions = np.ascontiguousarray(positions, dtype=np.float64).reshape(
            -1, 3
        )
        self.masses = np.ascontiguousarray(masses, dtype=np.float64).reshape(-1)
        self.radii = np.ascontiguousarray(radii, dtype=np.float64).reshape(-1)
        if not len(self.positions) == len(self.masses) == len(self.radii):
            raise ValueError("positions, masses and radii must have the same length")
        if (self.radii <= 0).any():
            raise ValueError("radii must be positive")
        self.g_constant = float(g_constant)
        # The native kernel's records: x, y, z, G*m, radius squared.
        self._records = np.ascontiguousarray(
            np.column_stack(
                (self.positions, self.g_constant * self.masses, self.radii**2)
            )
        )
        self._native = _native()

    @classmethod
    def from_bodies(
        cls, bodies: Iterable[_HasMass], g_constant: float = 1.0
    ) -> Gravity:
        """:param bodies: anything with position, mass and radius (``world.Body``)"""
        bodies = list(bodies)
        return cls(
            [b.position for b in bodies],
            [b.mass for b in bodies],
            [b.radius for b in bodies],
            g_constant,
        )

    def acceleration(
        self, points: ArrayLike, out: np.ndarray | None = None
    ) -> np.ndarray:
        """
        :param points: (N, 3) or (3,) positions, world frame, metres
        :param out: (N, 3) float64 C-contiguous array to fill, for a caller
            that evaluates every frame (None = a new array)
        :returns: the gravity there, same shape as points
        """
        pts = np.asarray(points, dtype=np.float64)
        single = pts.ndim == 1
        pts = np.ascontiguousarray(pts.reshape(-1, 3))
        acc = self._accumulate(pts, out)
        return acc[0] if single else acc

    def _accumulate(self, pts: np.ndarray, out: np.ndarray | None) -> np.ndarray:
        shape = pts.shape
        if out is None:
            out = np.empty(shape)
        elif (
            out.shape != shape or out.dtype != np.float64 or not out.flags.c_contiguous
        ):
            raise ValueError(
                "out must be a C-contiguous float64 array of the points' shape"
            )
        if self._native is not None:
            self._native(pts, self._records, out)
            return out
        out.fill(0.0)
        delta = np.empty(shape)
        r2 = np.empty(len(pts))
        for centre, gm, floor2 in zip(
            self.positions, self.g_constant * self.masses, self.radii**2
        ):
            np.subtract(centre, pts, out=delta)  # toward the body
            np.einsum("ij,ij->i", delta, delta, out=r2)
            np.maximum(r2, floor2, out=r2)
            r2 *= np.sqrt(r2)
            np.divide(gm, r2, out=r2)
            delta *= r2[:, None]
            out += delta
        return out
