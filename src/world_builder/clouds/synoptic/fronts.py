"""
A low's fronts: curves of linearly turning heading (circular arcs) in its
local frame, shaped by its stage.

The frame is the low's: x along its heading (the jet's direction, about
east), y poleward, so one layout serves both hemispheres (the southern one
is the mirror image, as its cyclones are). Norwegian lows (bjerknes1922)
occlude from the centre: the occluded front wraps poleward and the triple
point moves out along it. Shapiro-Keyser lows (shapiro1990) fracture
instead: the cold front starts away from the warm one, meeting it in a
T-bone, and the warm front bends back poleward and west round the centre.

Every front is stored with its cold side on its left, so a signed distance
(:func:`front_distance`) is positive on the cold side: the warm front runs
from the triple point east, the cold front from its tip back to the triple
point.

docs/literature/synoptic-weather-review.tex, section "Fronts".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from world_builder.clouds.synoptic.scales import Scales
from world_builder.clouds.synoptic.systems import SynopticObject, _smoothstep, stage

#: Points per front.
SAMPLES = 33
_DEG = math.pi / 180.0


@dataclass(frozen=True, eq=False)
class Fronts:
    """A low's fronts, (M, 2) polylines in its local (along, poleward)
    metres, each with its cold side on its left.

    :ivar warm: from the triple point east
    :ivar cold: from its tip back to the triple point
    :ivar occluded: from the centre to the triple point (empty for a
        Shapiro-Keyser low)
    :ivar triple: (2,) the triple point
    :ivar bent: a Shapiro-Keyser low's bent-back warm front, past its bend
        (empty otherwise): cloud on both sides of it (its cloud head), so
        it has no cold side
    """

    warm: np.ndarray
    cold: np.ndarray
    occluded: np.ndarray
    triple: np.ndarray
    bent: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def arc(start, heading: float, curvature: float, length: float, begin: float = 0.0):
    """A curve of heading ``heading + curvature * sigma`` (radians, per
    unit length) from *start*, sampled from *begin* to *length*.

    :returns: (SAMPLES, 2)
    """
    length = max(length, begin + 1e-3)
    sigma = np.linspace(0.0, length, 4 * SAMPLES)
    theta = heading + curvature * sigma
    step = sigma[1] - sigma[0]
    dx = np.concatenate(
        [[0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * step)]
    )
    dy = np.concatenate(
        [[0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * step)]
    )
    points = np.asarray(start, dtype=np.float64) + np.stack([dx, dy], axis=-1)
    keep = np.linspace(begin, length, SAMPLES)
    return np.stack(
        [np.interp(keep, sigma, points[:, 0]), np.interp(keep, sigma, points[:, 1])],
        axis=-1,
    )


def _bent_back(start, length: float, bend_at: float, bend: float):
    """The Shapiro-Keyser warm front: straight to *bend_at*, then turning
    *bend* radians per unit length (curling poleward and back west)."""
    sigma = np.linspace(0.0, length, 4 * SAMPLES)
    theta = -10.0 * _DEG + np.maximum(0.0, sigma - bend_at) * bend
    step = sigma[1] - sigma[0]
    dx = np.concatenate(
        [[0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * step)]
    )
    dy = np.concatenate(
        [[0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * step)]
    )
    points = np.asarray(start) + np.stack([dx, dy], axis=-1)
    keep = np.linspace(0.0, length, SAMPLES)
    return np.stack(
        [np.interp(keep, sigma, points[:, 0]), np.interp(keep, sigma, points[:, 1])],
        axis=-1,
    )


def fronts_in_radii(obj: SynopticObject, t: float) -> Fronts:
    """:func:`fronts` in units of the low's radius."""
    s = stage(obj, t)
    grown = float(_smoothstep(0.0, 0.2, s))
    if obj.variant == "sk":
        triple = np.array([0.05, 0.0])
        occluded = np.zeros((0, 2))
        bend = float(_smoothstep(0.3, 0.7, s))
        whole = _bent_back(triple, 1.5 * grown + 1.0 * bend, 1.0, 150.0 * _DEG)
        steps = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(whole, axis=0), axis=1)])
        k = max(int(np.searchsorted(steps, 1.0, side="right")), 2)
        warm, bent = whole[:k], whole[k - 1 :]
        if len(bent) < 2:
            bent = np.zeros((0, 2))
        fracture = 0.4 * float(_smoothstep(0.3, 0.6, s))
        cold = arc(triple, -90.0 * _DEG, -16.0 * _DEG, 2.5 * grown + fracture, fracture)
    else:
        wrap = float(_smoothstep(0.35, 0.9, s))
        occluded = arc((0.0, 0.0), -20.0 * _DEG, 60.0 * _DEG, max(1.2 * wrap, 1e-3))
        triple = occluded[-1].copy()
        warm = arc(triple, -10.0 * _DEG, -10.0 * _DEG, 1.5 * grown)
        cold = arc(triple, -80.0 * _DEG, -16.0 * _DEG, 2.5 * grown)
        bent = np.zeros((0, 2))
    return Fronts(
        warm=warm, cold=cold[::-1].copy(), occluded=occluded, triple=triple, bent=bent
    )


def fronts(obj: SynopticObject, t: float, scales: Scales) -> Fronts:
    """A low's fronts at *t*, in its local (along, poleward) metres."""
    f = fronts_in_radii(obj, t)
    r = obj.radius
    return Fronts(
        warm=f.warm * r,
        cold=f.cold * r,
        occluded=f.occluded * r,
        triple=f.triple * r,
        bent=f.bent * r,
    )


#: Beyond this many radii from a front's bounding box a point is "far on
#: its cold side": every rule gives nothing there.
FAR = 2.5


def front_query(
    points: np.ndarray, polyline: np.ndarray, far: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance to a polyline (positive on its left: the cold side)
    and the arc length of the closest point along it, from its start.

    :param points: (N, 2)
    :param polyline: (M, 2), M >= 2
    :param far: points farther than this from the polyline's bounding box
        get +inf (and arc length 0) without being measured
    :returns: (N,), (N,)
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    distance = np.full(n, np.inf)
    along = np.zeros(n)
    todo = np.arange(n)
    if far is not None:
        lo = polyline.min(axis=0) - far
        hi = polyline.max(axis=0) + far
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        todo = np.nonzero(inside)[0]
    if len(todo) == 0:
        return distance, along
    px, py = points[todo, 0][:, None], points[todo, 1][:, None]
    ax, ay = polyline[:-1, 0][None], polyline[:-1, 1][None]
    sx, sy = np.diff(polyline[:, 0])[None], np.diff(polyline[:, 1])[None]
    seg_len2 = np.maximum(sx * sx + sy * sy, 1e-18)
    rx, ry = px - ax, py - ay
    u = np.clip((rx * sx + ry * sy) / seg_len2, 0.0, 1.0)
    gx, gy = rx - u * sx, ry - u * sy
    dist2 = gx * gx + gy * gy
    best = np.argmin(dist2, axis=1)
    rows = np.arange(len(todo))
    cross = sx[0, best] * gy[rows, best] - sy[0, best] * gx[rows, best]
    sign = np.where(cross >= 0.0, 1.0, -1.0)
    measured = sign * np.sqrt(dist2[rows, best])
    # Past either end, the distance from the end segment's extended line:
    # it crosses 0 smoothly there, where the distance to the end point,
    # signed by side, would jump.
    last = len(polyline) - 2
    past = ((best == 0) & (u[rows, best] <= 0.0)) | (
        (best == last) & (u[rows, best] >= 1.0)
    )
    rel_cross = sx[0, best] * ry[rows, best] - sy[0, best] * rx[rows, best]
    perpendicular = rel_cross / np.sqrt(seg_len2[0, best])
    distance[todo] = np.where(past, perpendicular, measured)
    seg_len = np.sqrt(seg_len2[0])
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    along[todo] = cumulative[best] + u[rows, best] * seg_len[best]
    return distance, along


def overshoot(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """How far past either end of a polyline *points* lie, along its own
    direction there (0 alongside it).

    :returns: (N,), in the polyline's units
    """
    points = np.asarray(points, dtype=np.float64)
    head = polyline[0] - polyline[1]
    tail = polyline[-1] - polyline[-2]
    head /= max(np.linalg.norm(head), 1e-12)
    tail /= max(np.linalg.norm(tail), 1e-12)
    before = (points - polyline[0]) @ head
    after = (points - polyline[-1]) @ tail
    return np.maximum(np.maximum(before, after), 0.0)


def front_distance(points_local: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """:returns: the signed distance of *points_local* to the front, positive
    on its cold side (its left), in the polyline's units"""
    return front_query(points_local, polyline)[0]


def length(polyline: np.ndarray) -> float:
    """:returns: the polyline's arc length"""
    return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))
