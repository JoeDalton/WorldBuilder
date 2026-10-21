"""
Local flat maps of the planet sphere, and latitude/longitude.

:class:`AzimuthalEquidistantChart` unrolls the sphere around a reference point
into flat metres (x east, y north): distances from that point are exact, and
the sideways stretch is c^2/6 at angular distance c (0.12 % at 50 km on a
600 km planet). Stamps and paints are each defined in their own chart; level
files give positions in chart metres around their reference point; the
demo's ocean is anchored to the ground through one. The terrain itself is
sampled on the cube-sphere (:mod:`cubesphere`), not on a chart.

All of it is float64 numpy: at 500 km from the origin, float32 cannot even
resolve a metre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class TangentFrame:
    """Local east / north / up axes at a point of the unit sphere.

    :ivar up: outward unit normal at the point
    :ivar east: unit vector toward increasing longitude
    :ivar north: unit vector toward increasing latitude
    """

    up: np.ndarray
    east: np.ndarray
    north: np.ndarray

    @classmethod
    def at(cls, lat_deg: float, lon_deg: float) -> TangentFrame:
        """:returns: the frame at latitude/longitude, in degrees"""
        lat, lon = math.radians(lat_deg), math.radians(lon_deg)
        up = np.array(
            [
                math.cos(lat) * math.cos(lon),
                math.cos(lat) * math.sin(lon),
                math.sin(lat),
            ]
        )
        east = np.array([-math.sin(lon), math.cos(lon), 0.0])
        north = np.cross(up, east)
        return cls(up=up, east=east, north=north)


def latlon(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:returns: latitude and longitude in degrees of (N, 3) unit vectors"""
    lat = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
    return lat, lon


def dirs_from_latlon(lat_deg: ArrayLike, lon_deg: ArrayLike) -> np.ndarray:
    """:returns: (N, 3) unit vectors at the given latitudes/longitudes (degrees)"""
    lat = np.radians(np.atleast_1d(np.asarray(lat_deg, dtype=np.float64)))
    lon = np.radians(np.atleast_1d(np.asarray(lon_deg, dtype=np.float64)))
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
    )


class AzimuthalEquidistantChart:
    """Flat (x east, y north) metres around a tangent point <-> sphere directions.

    Also used by stamps, each with its own tangent point, so a stamp keeps its
    shape wherever it is placed on the planet.
    """

    def __init__(self, lat_deg: float, lon_deg: float, radius: float):
        """
        :param lat_deg: latitude of the chart origin, degrees
        :param lon_deg: longitude of the chart origin, degrees
        :param radius: planet radius, metres
        """
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.radius = radius
        self.frame = TangentFrame.at(lat_deg, lon_deg)

    def to_dirs(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Directions of flat chart points.

        :param x: metres east of the origin, any shape
        :param y: metres north of the origin, same shape
        :returns: (N, 3) unit vectors, N = x.size
        """
        x = np.ravel(np.asarray(x, dtype=np.float64))
        y = np.ravel(np.asarray(y, dtype=np.float64))
        c = np.hypot(x, y) / self.radius  # angular distance from the origin
        # sin(c)/rho = sinc(c)/R, finite at the origin
        tangential = np.sinc(c / math.pi) / self.radius
        f = self.frame
        return (
            np.cos(c)[:, None] * f.up
            + (tangential * x)[:, None] * f.east
            + (tangential * y)[:, None] * f.north
        )

    def from_dirs(self, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat chart coordinates of directions.

        :param dirs: (N, 3) unit vectors
        :returns: x, y in metres
        """
        f = self.frame
        along_up = dirs @ f.up
        e = dirs @ f.east
        n = dirs @ f.north
        sin_c = np.hypot(e, n)
        c = np.arctan2(sin_c, along_up)
        # rho / sin(c) = R c / sin(c) = R / sinc(c)
        scale = self.radius / np.sinc(c / math.pi)
        return e * scale, n * scale

    def __getstate__(self):
        return (self.lat_deg, self.lon_deg, self.radius)

    def __setstate__(self, state):
        self.__init__(*state)
