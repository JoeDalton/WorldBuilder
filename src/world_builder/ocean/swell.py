"""
The sea's swell on the CPU: the height of the water the terrain draws, for
collisions and queries (``TerrainQuery``, the native collision solid).

The field is ocean/shaders/swell.glsl's, computed again here in float64:
natively when the noise library is built (``wb_swell_heights``, from
``terrain/native/swell.h``), else in numpy, the same arithmetic. A
:class:`world_builder.ocean.Sea` makes a :class:`Swell` every frame (its
layers turn with time) and hands it to its terrain.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from world_builder.terrain.noise import native_library

#: The field's mean, which faded octaves tend to (swell.glsl's SWELL_MEAN).
SWELL_MEAN = 0.25


class _WbSwell(ctypes.Structure):
    """swell.h's ``WbSwell``."""

    _fields_ = [
        ("amplitude", ctypes.c_double),
        ("scale", ctypes.c_double),
        ("rot_a", ctypes.c_double * 9),
        ("rot_b", ctypes.c_double * 9),
        ("by_depth", ctypes.c_int32),
        ("shallow", ctypes.c_double),
        ("deep", ctypes.c_double),
    ]


@dataclass(frozen=True)
class Swell:
    """The swell at one moment.

    :ivar amplitude: metres of water per unit of field, deep water
    :ivar scale: the first layer's frequency, cycles per metre
    :ivar rot_a: (3, 3) the first layer's rotation (applied as ``R @ p``)
    :ivar rot_b: the second's
    :ivar by_depth: the swell dies out over the shallows
    :ivar shallow: the depth where it starts, metres...
    :ivar deep: ...and where it is whole
    """

    amplitude: float
    scale: float
    rot_a: np.ndarray
    rot_b: np.ndarray
    by_depth: bool = True
    shallow: float = 5.0
    deep: float = 60.0

    def as_struct(self) -> _WbSwell:
        """:returns: it as swell.h's ``WbSwell`` (for ctypes)"""
        out = _WbSwell()
        out.amplitude = self.amplitude
        out.scale = self.scale
        out.rot_a[:] = [float(v) for v in np.asarray(self.rot_a).ravel()]
        out.rot_b[:] = [float(v) for v in np.asarray(self.rot_b).ravel()]
        out.by_depth = int(self.by_depth)
        out.shallow = self.shallow
        out.deep = self.deep
        return out

    def heights(
        self, points: ArrayLike, depths: ArrayLike, max_freqs: ArrayLike
    ) -> np.ndarray:
        """The water's height above sea level at sea-level points.

        :param points: (N, 3) planet frame, metres
        :param depths: (N,) metres of water to the seabed under each (the
            swell dies out over the shallows)
        :param max_freqs: (N,) or scalar: the resolution each is drawn at,
            1 / (4 vertex spacing); octaves finer than that are faded
        :returns: (N,) metres
        """
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        n = len(points)
        depths = np.broadcast_to(np.asarray(depths, dtype=np.float64), (n,))
        max_freqs = np.broadcast_to(np.asarray(max_freqs, dtype=np.float64), (n,))
        if self.amplitude == 0.0 or n == 0:
            return np.zeros(n)
        native = native_library()
        if native is not None:
            return native.swell_heights(self.as_struct(), points, depths, max_freqs)
        factor = np.ones(n)
        if self.by_depth:
            t = np.clip((depths - self.shallow) / (self.deep - self.shallow), 0.0, 1.0)
            factor = t * t * (3.0 - 2.0 * t)
        field = swell_field(self, points, max_freqs)
        return self.amplitude * factor * (field - SWELL_MEAN)


def swell_field(swell: Swell, points: np.ndarray, max_freqs: np.ndarray) -> np.ndarray:
    """swell.h's ``wb_swell_field`` in numpy: the field in [0, 1] at (N, 3)
    planet-frame points, octaves above *max_freqs* faded to their mean."""
    k, kb = swell.scale, 1.7 * swell.scale
    qa = (points @ np.asarray(swell.rot_a).T) * k
    qb = (points @ np.asarray(swell.rot_b).T) * kb
    la = _layer(qa, _band(k, max_freqs), _band(2.1 * k, max_freqs))
    lb = _layer(qb, _band(kb, max_freqs), _band(3.57 * k, max_freqs))
    return la * lb


def _band(freq: float, max_freqs: np.ndarray) -> np.ndarray:
    limited = max_freqs > 0.0
    safe = np.where(limited, max_freqs, 1.0)
    t = np.clip((2.0 * safe - 2.0 * freq) / safe, 0.0, 1.0)
    return np.where(limited, t * t * (3.0 - 2.0 * t), 1.0)


def _layer(q: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    n0 = _noise(q)
    n1 = _noise(q * 2.1 + np.array([5.3, 1.7, 3.1]))
    return 0.6 * (0.5 + (n0 - 0.5) * w0) + 0.4 * (0.5 + (n1 - 0.5) * w1)


def _lattice(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """swell.h's lattice hash, in uint32 arithmetic (wraps as C's does)."""
    with np.errstate(over="ignore"):
        h = (
            (x.astype(np.int32).view(np.uint32) * np.uint32(0x8DA6B343))
            ^ (y.astype(np.int32).view(np.uint32) * np.uint32(0xD8163841))
            ^ (z.astype(np.int32).view(np.uint32) * np.uint32(0xCB1AB31F))
        )
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    return (h >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)


def _noise(q: np.ndarray) -> np.ndarray:
    i = np.floor(q)
    f = q - i
    u = f * f * (3.0 - 2.0 * f)
    x, y, z = (i[:, axis].astype(np.int64) for axis in range(3))
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    def lerp(a, b, t):
        return a + (b - a) * t

    x00 = lerp(_lattice(x, y, z), _lattice(x + 1, y, z), ux)
    x10 = lerp(_lattice(x, y + 1, z), _lattice(x + 1, y + 1, z), ux)
    x01 = lerp(_lattice(x, y, z + 1), _lattice(x + 1, y, z + 1), ux)
    x11 = lerp(_lattice(x, y + 1, z + 1), _lattice(x + 1, y + 1, z + 1), ux)
    return lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz)
