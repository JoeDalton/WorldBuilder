"""
Weather maps for previews: the genera as colours, and a surface chart
(isobars, fronts with their symbols, the jets' axes, the systems' centres)
on an equirectangular grid, the southmost row first (as
:func:`terrain.preview._globe_dirs`).
"""

from __future__ import annotations

import math

import numpy as np

from world_builder.clouds.synoptic.fields import frame, pressure
from world_builder.clouds.synoptic.fronts import fronts_in_radii
from world_builder.clouds.synoptic.jets import jet_lat
from world_builder.clouds.synoptic.systems import Kind, intensity, position
from world_builder.terrain.chart import dirs_from_latlon, latlon

#: Isobars every this many hPa.
ISOBAR_STEP = 4.0
#: The chart's colours, sRGB.
ISOBAR = np.array([0.15, 0.15, 0.15])
COLD = np.array([0.15, 0.35, 1.0])
WARM = np.array([0.9, 0.15, 0.15])
OCCLUDED = np.array([0.6, 0.2, 0.75])
JET = np.array([1.0, 0.75, 0.15])
MARKS = {
    Kind.LOW: np.array([1.0, 0.0, 1.0]),
    Kind.CUTOFF: np.array([0.6, 0.0, 0.6]),
    Kind.HIGH: np.array([0.0, 0.8, 1.0]),
    Kind.TROPICAL: np.array([1.0, 0.55, 0.0]),
}
#: The genera's colours, linear: low, mid, high, the cumulus, deep
#: convection.
GENUS_COLOURS = (
    (("stratus", "stratocumulus"), np.array([0.95, 0.9, 0.55])),
    (("altostratus", "nimbostratus", "altocumulus"), np.array([0.6, 0.8, 1.0])),
    (("cirrus", "cirrostratus", "cirrocumulus"), np.array([0.85, 0.6, 0.95])),
)


def genus_colours(ground: np.ndarray, channels: dict) -> np.ndarray:
    """*ground* (linear colour) under the genera, by colour: low yellow, mid
    blue, high violet, cumulus white, cumulonimbus red.

    :param channels: :data:`weather.SYNOPTIC_CHANNELS` name -> array
    """
    img = ground.copy()
    for names, colour in GENUS_COLOURS[::-1]:
        cover = np.maximum.reduce([channels[n] for n in names])[..., None]
        img = img * (1.0 - 0.85 * cover) + 0.85 * cover * colour
    cover = channels["convective"][..., None]
    img = img * (1.0 - 0.85 * cover) + 0.85 * cover
    deep = (channels["depth"] * channels["convective"])[..., None]
    return img * (1.0 - deep) + deep * np.array([1.0, 0.2, 0.15])


class _Canvas:
    """An equirectangular sRGB image, south row first, drawn on by latitude
    and longitude."""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.h, self.w = image.shape[:2]

    def pixel(self, lat, lon):
        row = np.rint((np.asarray(lat) + 90.0) / 180.0 * (self.h - 1)).astype(int)
        col = np.rint(
            ((np.asarray(lon) + 180.0) % 360.0) / 360.0 * (self.w - 1)
        ).astype(int)
        return np.clip(row, 0, self.h - 1), np.clip(col, 0, self.w - 1)

    def dots(self, lat, lon, colour, size: int = 0):
        row, col = self.pixel(lat, lon)
        for dr in range(-size, size + 1):
            for dc in range(-size, size + 1):
                self.image[np.clip(row + dr, 0, self.h - 1), (col + dc) % self.w] = (
                    colour
                )

    def dirs(self, dirs, colour, size: int = 0):
        lat, lon = latlon(np.atleast_2d(dirs))
        self.dots(lat, lon, colour, size)


def _densify(dirs: np.ndarray, per_radian: float) -> np.ndarray:
    """Points along a polyline of directions, about *per_radian* a radian."""
    out = [dirs[:1]]
    for a, b in zip(dirs[:-1], dirs[1:]):
        n = max(2, int(math.acos(float(np.clip(a @ b, -1.0, 1.0))) * per_radian) + 1)
        t = np.linspace(0.0, 1.0, n)[1:, None]
        seg = a * (1.0 - t) + b * t
        out.append(seg / np.linalg.norm(seg, axis=1, keepdims=True))
    return np.concatenate(out)


def _to_dirs(obj, t: float, points: np.ndarray, radius: float) -> np.ndarray:
    """A low's local (along, poleward) points, in its radii, as directions."""
    lat, lon = position(obj, t)
    centre = dirs_from_latlon(lat, lon)[0]
    east, north = frame(centre)
    h = 1.0 if lat >= 0.0 else -1.0
    c, s = math.cos(obj.heading), math.sin(obj.heading)
    ahead = c * east + s * north
    left = -s * east + c * north
    scale = obj.radius / radius
    d = centre + scale * (points[:, :1] * ahead + h * points[:, 1:] * left)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _front(canvas, obj, t, points, radius, colour, symbol_side: float, per_radian):
    """A front's line, with a symbol every few pixels on *symbol_side* of it
    (+1 its left, its cold side; -1 its right; 0 both, alternating)."""
    if len(points) < 2:
        return
    line = _to_dirs(obj, t, points, radius)
    canvas.dirs(_densify(line, per_radian), colour)
    # Symbols: a blob a little off the line, every 8 pixels or so.
    steps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    pixel_radii = 8.0 / per_radian * radius / obj.radius
    for k, at in enumerate(np.arange(0.5 * pixel_radii, steps[-1], pixel_radii)):
        i = int(np.clip(np.searchsorted(steps, at), 1, len(points) - 1))
        tangent = points[i] - points[i - 1]
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        normal = np.array([-tangent[1], tangent[0]])
        side = symbol_side if symbol_side else (1.0 if k % 2 else -1.0)
        p = np.interp(at, steps, points[:, 0]), np.interp(at, steps, points[:, 1])
        blob = np.array(p) + side * normal * 1.5 / per_radian * radius / obj.radius
        canvas.dirs(_to_dirs(obj, t, blob[None], radius), colour, size=1)


def draw_chart(
    background: np.ndarray, dirs: np.ndarray, systems, jets, scales, t: float
) -> np.ndarray:
    """A surface chart over *background* (sRGB, south row first): isobars
    every :data:`ISOBAR_STEP` hPa, the jets' axes, the lows' fronts (cold
    blue, its triangles on the warm side; warm red, its half-discs on the
    cold side; occluded purple, both) and the systems' centres.

    :param dirs: (h, w, 3) the grid's directions
    :returns: (h, w, 3) sRGB, south row first
    """
    h, w = dirs.shape[:2]
    canvas = _Canvas(background.copy())
    p = pressure(dirs.reshape(-1, 3), systems, scales, t).reshape(h, w) + 1013.0
    level = np.floor(p / ISOBAR_STEP)
    edge = np.zeros((h, w), bool)
    edge[:, :-1] |= level[:, :-1] != level[:, 1:]
    edge[:-1, :] |= level[:-1, :] != level[1:, :]
    canvas.image[edge] = ISOBAR

    lon = np.linspace(-180.0, 180.0, 4 * w)
    for jet in jets:
        canvas.dots(jet_lat(jet, lon, t), lon, JET)

    per_radian = w / (2.0 * math.pi)
    radius = scales.radius
    for obj in systems:
        if intensity(obj, t) <= 0.0:
            continue
        if obj.kind is Kind.LOW:
            f = fronts_in_radii(obj, t)
            _front(canvas, obj, t, f.cold, radius, COLD, -1.0, per_radian)
            _front(canvas, obj, t, f.warm, radius, WARM, 1.0, per_radian)
            if len(f.bent) >= 2:
                _front(canvas, obj, t, f.bent, radius, WARM, 1.0, per_radian)
            if len(f.occluded) >= 2:
                _front(canvas, obj, t, f.occluded, radius, OCCLUDED, 0.0, per_radian)
        canvas.dots(*position(obj, t), MARKS[obj.kind], size=2)
    return canvas.image
