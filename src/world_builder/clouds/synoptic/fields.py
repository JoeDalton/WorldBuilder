"""
Pressure and wind from the weather systems.

Pressure: a zonal climatology plus one kernel per system (an anisotropic
Gaussian for lows and highs, holland1980's profile for tropical cyclones).
Wind: geostrophic from its gradient, the gradient wind near a tropical
cyclone's core, turned toward low pressure and slowed by friction at the
surface, blending into the trades near the equator.

Kernels are evaluated only near their system: directions are binned into
cube-face tiles once per call (:class:`DirIndex`).

docs/literature/synoptic-weather-review.tex, sections "Pressure" and "Wind".
"""

from __future__ import annotations

import math

import numpy as np

from world_builder.clouds.synoptic.scales import Scales
from world_builder.clouds.synoptic.systems import (
    Kind,
    SynopticObject,
    _smoothstep,
    intensity,
    position,
)
from world_builder.terrain.chart import dirs_from_latlon, latlon

RHO_AIR = 1.2
#: Surface friction: the turn toward low pressure, and the speed kept.
FRICTION_TURN_DEG = 25.0
FRICTION_KEEP = 0.7
#: The trades the wind blends into near the equator, m/s (east, poleward).
TRADE_EAST = -5.0
TRADE_EQUATORWARD = 1.5
#: Kernels reach this many radii (tapered over the last one).
KERNEL_REACH = 4.0
#: A tropical cyclone's radius of maximum wind, in its radii.
TC_MAX_WIND = 0.1


class DirIndex:
    """Directions binned into tiles on the cube's faces, for culling.

    :param dirs: (N, 3) unit directions
    :param per_tile: the directions a tile holds, roughly
    """

    def __init__(self, dirs: np.ndarray, per_tile: int = 64):
        self.dirs = dirs
        n = len(dirs)
        tiles = max(1, int(math.sqrt(max(n, 1) / (6.0 * per_tile))))
        major = np.argmax(np.abs(dirs), axis=1)
        sign = np.take_along_axis(dirs, major[:, None], 1)[:, 0] >= 0.0
        face = major * 2 + sign
        others = np.array([[1, 2], [0, 2], [0, 1]])[major]
        scale = np.maximum(
            np.abs(np.take_along_axis(dirs, major[:, None], 1)[:, 0]), 1e-12
        )
        u = np.take_along_axis(dirs, others[:, :1], 1)[:, 0] / scale
        v = np.take_along_axis(dirs, others[:, 1:], 1)[:, 0] / scale
        iu = np.clip(((u + 1.0) * 0.5 * tiles).astype(np.int64), 0, tiles - 1)
        iv = np.clip(((v + 1.0) * 0.5 * tiles).astype(np.int64), 0, tiles - 1)
        cell = (face * tiles + iv) * tiles + iu
        self.order = np.argsort(cell, kind="stable")
        sorted_cells = cell[self.order]
        cells, starts = np.unique(sorted_cells, return_index=True)
        self.starts = np.append(starts, n)
        sums = (
            np.add.reduceat(dirs[self.order], starts, axis=0) if n else np.zeros((0, 3))
        )
        centres = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
        self.centres = centres
        spread = np.einsum(
            "ij,ij->i", dirs[self.order], np.repeat(centres, np.diff(self.starts), 0)
        )
        lowest = np.minimum.reduceat(spread, starts) if n else np.zeros(0)
        self.angles = np.arccos(np.clip(lowest, -1.0, 1.0))

    def query(self, centre: np.ndarray, angle: float) -> np.ndarray:
        """:returns: the indices of the directions within *angle* of *centre*"""
        if len(self.centres) == 0:
            return np.zeros(0, dtype=np.int64)
        tile = np.arccos(np.clip(self.centres @ centre, -1.0, 1.0))
        hit = np.nonzero(tile <= angle + self.angles)[0]
        if len(hit) == 0:
            return np.zeros(0, dtype=np.int64)
        idx = np.concatenate(
            [self.order[self.starts[i] : self.starts[i + 1]] for i in hit]
        )
        close = self.dirs[idx] @ centre >= math.cos(min(angle, math.pi))
        return idx[close]


def near(dirs: np.ndarray, obj: SynopticObject, t: float, reach: float) -> np.ndarray:
    """:returns: the mask of *dirs* within *reach* radians of *obj*'s centre"""
    centre = dirs_from_latlon(*position(obj, t))[0]
    return np.asarray(dirs) @ centre >= math.cos(min(reach, math.pi))


def _gauss(x, x0, sigma):
    return np.exp(-(((x - x0) / sigma) ** 2))


def climatology(lat: np.ndarray, scales: Scales, itcz: float = 0.0) -> np.ndarray:
    """The zonal mean sea-level pressure, hPa: the equatorial trough, the
    subtropical highs, the subpolar lows and the polar highs."""
    a = np.abs(lat)
    track = 0.5 * (scales.storm_track[0] + scales.storm_track[1])
    return (
        1013.0
        - 4.0 * _gauss(lat, itcz, 8.0)
        + 8.0 * _gauss(a, scales.hadley_edge, 8.0)
        - 10.0 * _gauss(a, track, 10.0)
        + 4.0 * _gauss(a, 90.0, 10.0)
    )


def frame(centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:returns: (east, north) unit vectors at *centre* (any pair at a pole)"""
    east = np.cross([0.0, 0.0, 1.0], centre)
    norm = np.linalg.norm(east)
    east = east / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])
    return east, np.cross(centre, east)


def local(dirs: np.ndarray, obj: SynopticObject, t: float, radius: float):
    """Gnomonic (east, north) metres of *dirs* around *obj*'s centre, and
    the same turned so x runs along its heading.

    :returns: x_east, y_north, x_along, y_across
    """
    centre = dirs_from_latlon(*position(obj, t))[0]
    east, north = frame(centre)
    depth = np.maximum(dirs @ centre, 1e-6)
    x = radius * (dirs @ east) / depth
    y = radius * (dirs @ north) / depth
    c, s = math.cos(obj.heading), math.sin(obj.heading)
    return x, y, x * c + y * s, -x * s + y * c


def holland_b(obj: SynopticObject) -> float:
    """:returns: its Holland B, ~N(1.3, 0.35) clamped to [0.8, 2.2]
    (willoughby2004)"""
    return float(np.clip(np.random.default_rng(obj.seed).normal(1.3, 0.35), 0.8, 2.2))


def kernel(obj: SynopticObject, t: float, x, y, xa, ya) -> np.ndarray:
    """Its pressure anomaly at local coordinates, hPa (before its taper)."""
    r = obj.radius
    strength = obj.strength * intensity(obj, t)
    if obj.kind is Kind.HIGH:
        return strength * np.exp(-(x * x + y * y) / (1.5 * r) ** 2)
    if obj.kind is Kind.TROPICAL:
        rho = np.maximum(np.hypot(x, y), 1.0)
        return -strength * (1.0 - np.exp(-((TC_MAX_WIND * r / rho) ** holland_b(obj))))
    return -strength * np.exp(-(xa * xa) / (1.3 * r) ** 2 - (ya * ya) / (0.8 * r) ** 2)


def _anomalies(dirs, systems, scales, t, index=None):
    """Sum of the systems' kernels at *dirs*, hPa."""
    index = index or DirIndex(dirs)
    out = np.zeros(len(dirs))
    for obj in systems:
        if intensity(obj, t) <= 0.0:
            continue
        centre = dirs_from_latlon(*position(obj, t))[0]
        reach = KERNEL_REACH * obj.radius / scales.radius
        idx = index.query(centre, reach)
        if len(idx) == 0:
            continue
        x, y, xa, ya = local(dirs[idx], obj, t, scales.radius)
        rho = np.hypot(x, y)
        taper = 1.0 - _smoothstep(
            (KERNEL_REACH - 1.0) * obj.radius, KERNEL_REACH * obj.radius, rho
        )
        out[idx] += kernel(obj, t, x, y, xa, ya) * taper
    return out


def pressure(dirs: np.ndarray, systems, scales: Scales, t: float) -> np.ndarray:
    """Sea-level pressure at *dirs*, hPa from 1013.

    :param dirs: (N, 3) unit directions
    :param systems: the weather systems (:func:`systems.population`)
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    lat, _ = latlon(dirs)
    return climatology(lat, scales) - 1013.0 + _anomalies(dirs, systems, scales, t)


def _tangent_basis(dirs):
    east = np.cross([0.0, 0.0, 1.0], dirs)
    norm = np.linalg.norm(east, axis=1, keepdims=True)
    east = np.where(norm > 1e-9, east / np.maximum(norm, 1e-12), [0.0, 1.0, 0.0])
    return east, np.cross(dirs, east)


def _gradient_wind(obj, t, x, y, f, scales):
    """A tropical cyclone's gradient wind (holland1980), m/s, cyclonic."""
    r_max = TC_MAX_WIND * obj.radius
    rho = np.maximum(np.hypot(x, y), 1.0)
    b = holland_b(obj)
    dp = obj.strength * intensity(obj, t) * 100.0
    q = (r_max / rho) ** b
    v = (
        np.sqrt(b * dp / RHO_AIR * q * np.exp(-q) + (rho * f) ** 2 / 4.0)
        - rho * np.abs(f) / 2.0
    )
    h = 1.0 if position(obj, t)[0] >= 0.0 else -1.0
    return v * h * (-y / rho), v * h * (x / rho)


def wind(
    dirs: np.ndarray, systems, scales: Scales, t: float, surface: bool = True
) -> np.ndarray:
    """The wind at *dirs*, m/s (east, north): geostrophic from the pressure
    (f floored at its 5 degree value), a tropical cyclone's gradient wind
    near its core; at the *surface* turned toward low pressure and slowed,
    blending into the trades near the equator.

    :returns: (N, 2)
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    n = len(dirs)
    lat, _ = latlon(dirs)
    east, north = _tangent_basis(dirs)
    step = 0.05 * scales.cyclone_radius / scales.radius
    probes = np.concatenate(
        [
            dirs + step * east,
            dirs - step * east,
            dirs + step * north,
            dirs - step * north,
        ]
    )
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    p = pressure(probes, systems, scales, t) * 100.0  # Pa
    metres = 2.0 * step * scales.radius
    dpdx = (p[:n] - p[n : 2 * n]) / metres
    dpdy = (p[2 * n : 3 * n] - p[3 * n :]) / metres
    h = np.where(lat >= 0.0, 1.0, -1.0)
    f_floor = 2.0 * scales.omega * math.sin(math.radians(5.0))
    f = h * np.maximum(np.abs(2.0 * scales.omega * np.sin(np.radians(lat))), f_floor)
    u = -dpdy / (RHO_AIR * f)
    v = dpdx / (RHO_AIR * f)

    for obj in systems:
        if obj.kind is not Kind.TROPICAL or intensity(obj, t) <= 0.0:
            continue
        r_max = TC_MAX_WIND * obj.radius
        centre = dirs_from_latlon(*position(obj, t))[0]
        close = dirs @ centre >= math.cos(4.0 * r_max / scales.radius)
        if not close.any():
            continue
        x, y, _, _ = local(dirs[close], obj, t, scales.radius)
        gu, gv = _gradient_wind(obj, t, x, y, f[close], scales)
        w = 1.0 - _smoothstep(3.0 * r_max, 4.0 * r_max, np.hypot(x, y))
        u[close] = w * gu + (1.0 - w) * u[close]
        v[close] = w * gv + (1.0 - w) * v[close]

    if surface:
        turn = np.radians(FRICTION_TURN_DEG) * h
        c, s = np.cos(turn), np.sin(turn)
        u, v = FRICTION_KEEP * (u * c - v * s), FRICTION_KEEP * (u * s + v * c)
        w = _smoothstep(5.0, 12.0, np.abs(lat))
        u = w * u + (1.0 - w) * TRADE_EAST
        v = w * v + (1.0 - w) * (-TRADE_EQUATORWARD * h)
    return np.stack([u, v], axis=-1)
