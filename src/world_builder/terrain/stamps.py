"""
Hand-made terrain: heightmap stamps and biome paint masks.

A stamp is a 16-bit grayscale image laid on the planet at a latitude and
longitude, with its own size and rotation. It is defined in its own tangent
frame (an azimuthal-equidistant chart centred on it), so it keeps its shape
anywhere on the planet and does not depend on the game's chart. Blend modes:

``replace``  height = offset + scale * pixel
``add``      height += offset + scale * pixel
``max``      height = max(height, offset + scale * pixel)
``min``      height = min(height, offset + scale * pixel)
``flatten``  height = ground at the stamp's centre + offset + scale * pixel,
             i.e. a level pad (landing zone, base) carrying the image as relief

Each blend fades in over ``falloff`` metres inside the stamp's rectangle.
Images are box-filtered into a mip pyramid; a coarse chunk samples the level
matching its spacing, so it never aliases the stamp.

A paint mask is a colour image whose colours map to biomes (``colors``). It
forces those biomes where painted, blurred by ``falloff`` metres, blended by
``strength``. Pixels matching no colour leave the procedural biomes alone.

Images load through Panda's PNMImage (in workers too) and are cached per
process by path and modification time.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chart import AzimuthalEquidistantChart, TangentFrame
from world_builder.terrain.noise import smoothstep
from world_builder.terrain.pngio import read_image

BLENDS = ("replace", "add", "max", "min", "flatten")


@dataclass(frozen=True)
class Placement:
    """Where an image lies on the planet.

    :ivar lat: centre latitude, degrees
    :ivar lon: centre longitude, degrees
    :ivar width: east-west extent before rotation, metres
    :ivar height: north-south extent before rotation, metres
    :ivar rotation: counter-clockwise, degrees
    """

    lat: float
    lon: float
    width: float
    height: float
    rotation: float = 0.0

    def local(self, dirs: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """:returns: (u, v) in metres in the image's rotated frame, u east, v north"""
        x, y = _chart(self.lat, self.lon, radius).from_dirs(dirs)
        theta = math.radians(self.rotation)
        c, s = math.cos(theta), math.sin(theta)
        return x * c + y * s, -x * s + y * c

    def near(self, dirs: np.ndarray, radius: float) -> np.ndarray:
        """Indices of the directions within the image's bounding cap.

        Every chunk of the planet passes through the stamps, so this cheap
        test keeps the chart maths (singular at the antipode) to the few
        directions that can be inside.

        :returns: (M,) indices into dirs
        """
        up, cos_reach = _cap(self.lat, self.lon, self.width, self.height, radius)
        return np.nonzero(dirs @ up > cos_reach)[0]

    def inside(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """:returns: whether each (u, v) of :meth:`local` lies on the image"""
        return (np.abs(u) < self.width / 2.0) & (np.abs(v) < self.height / 2.0)

    def edge_weight(self, u: np.ndarray, v: np.ndarray, falloff: float) -> np.ndarray:
        """:returns: 1 deep inside, fading to 0 at the rectangle's edge"""
        depth = np.minimum(self.width / 2.0 - np.abs(u), self.height / 2.0 - np.abs(v))
        if falloff <= 0.0:
            return (depth > 0.0).astype(np.float64)
        return smoothstep(0.0, falloff, depth)

    def pixel(
        self, u: np.ndarray, v: np.ndarray, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """:returns: (row, column) fractional pixel coordinates, top row first"""
        rows, cols = shape
        col = (u / self.width + 0.5) * (cols - 1)
        row = (0.5 - v / self.height) * (rows - 1)
        return row, col


@dataclass(frozen=True)
class Stamp:
    """A heightmap stamp. See the module docstring for the blend modes.

    :ivar heightmap: image path (absolute, or relative to the working dir;
        level files resolve theirs against the level's folder). Any image
        Panda reads; a 16-bit grayscale PNG keeps the full precision
    :ivar placement: where it lies
    :ivar height_scale: metres per full-white pixel
    :ivar height_offset: metres added to every pixel
    :ivar blend: one of :data:`BLENDS`
    :ivar falloff: metres over which it fades in from its edges
    """

    heightmap: str
    placement: Placement
    height_scale: float = 1000.0
    height_offset: float = 0.0
    blend: str = "add"
    falloff: float = 0.0

    def __post_init__(self):
        if self.blend not in BLENDS:
            raise ValueError(f"unknown blend {self.blend!r}, expected one of {BLENDS}")

    def apply(
        self,
        dirs: np.ndarray,
        elevation: np.ndarray,
        spacing: float,
        radius: float,
        base: float = 0.0,
    ) -> np.ndarray:
        """Blend the stamp into sampled heights.

        :param dirs: (N, 3) sample directions
        :param elevation: (N,) heights so far, metres
        :param spacing: sample spacing, metres (picks the mip level)
        :param radius: planet radius, metres
        :param base: ground height at the stamp's centre, for ``flatten``
        :returns: new (N,) heights
        """
        near = self.placement.near(dirs, radius)
        if near.size == 0:
            return elevation
        u, v = self.placement.local(dirs[near], radius)
        within = self.placement.inside(u, v)
        if not np.any(within):
            return elevation
        inside = near[within]
        u, v = u[within], v[within]
        pyramid = _height_pyramid(self.heightmap, _mtime(self.heightmap))
        image = _pick_level(pyramid, self.placement.width, spacing)
        row, col = self.placement.pixel(u, v, image.shape)
        value = ndimage.map_coordinates(image, [row, col], order=1, mode="nearest")
        stamped = self.height_offset + self.height_scale * value
        current = elevation[inside]
        if self.blend == "replace":
            target = stamped
        elif self.blend == "add":
            target = current + stamped
        elif self.blend == "max":
            target = np.maximum(current, stamped)
        elif self.blend == "min":
            target = np.minimum(current, stamped)
        else:  # flatten
            target = base + stamped
        weight = self.placement.edge_weight(u, v, self.falloff)
        out = elevation.copy()
        out[inside] = current + weight * (target - current)
        return out


@dataclass(frozen=True)
class Paint:
    """A biome paint mask.

    :ivar mask: colour image path (as :attr:`Stamp.heightmap`)
    :ivar placement: where it lies
    :ivar colors: ((hex colour, biome name), ...)
    :ivar falloff: blur radius, metres, softening painted edges
    :ivar strength: 1 forces the painted biomes, less blends them in
    :ivar tolerance: colour distance (0-1 per channel, Euclidean) that counts
        as a match, for JPEG-ish or anti-aliased masks
    """

    mask: str
    placement: Placement
    colors: tuple[tuple[str, str], ...]
    falloff: float = 0.0
    strength: float = 1.0
    tolerance: float = 0.12

    def __post_init__(self):
        for _color, biome in self.colors:
            if biome not in BIOMES:
                raise ValueError(f"unknown biome {biome!r}, expected one of {BIOMES}")

    def apply(self, dirs: np.ndarray, weights: np.ndarray, radius: float) -> np.ndarray:
        """Force the painted biomes into (N, 6) weights.

        :param dirs: (N, 3) sample directions
        :param weights: (N, 6) normalised biome weights
        :param radius: planet radius, metres
        :returns: new (N, 6) weights, still normalised
        """
        near = self.placement.near(dirs, radius)
        if near.size == 0:
            return weights
        u, v = self.placement.local(dirs[near], radius)
        within = self.placement.inside(u, v)
        if not np.any(within):
            return weights
        inside = near[within]
        u, v = u[within], v[within]
        channels = _paint_channels(
            self.mask,
            _mtime(self.mask),
            self.colors,
            self.tolerance,
            self.falloff,
            self.placement.width,
        )
        row, col = self.placement.pixel(u, v, channels.shape[1:])
        paint = np.stack(
            [
                ndimage.map_coordinates(ch, [row, col], order=1, mode="constant")
                for ch in channels
            ],
            axis=1,
        )
        coverage = np.clip(paint.sum(axis=1, keepdims=True), 0.0, 1.0) * self.strength
        out = weights.astype(np.float64).copy()
        normalised = paint / np.maximum(paint.sum(axis=1, keepdims=True), 1e-6)
        out[inside] = out[inside] * (1.0 - coverage) + normalised * coverage
        return out.astype(np.float32)


# ── image caches (per process) ────────────────────────────────────────────────


@functools.lru_cache(maxsize=8)
def _chart(lat: float, lon: float, radius: float) -> AzimuthalEquidistantChart:
    return AzimuthalEquidistantChart(lat, lon, radius)


@functools.lru_cache(maxsize=64)
def _cap(lat: float, lon: float, width: float, height: float, radius: float):
    """:returns: (centre direction, cosine of a cap angle enclosing the image)"""
    # The half diagonal is a chart distance, i.e. an arc length; 1 % spare.
    angle = 1.01 * math.hypot(width, height) / 2.0 / radius + 1e-9
    return TangentFrame.at(lat, lon).up, math.cos(min(angle, math.pi))


def _mtime(path: str) -> float:
    return os.path.getmtime(path)


@functools.lru_cache(maxsize=32)
def _height_pyramid(path: str, _mtime_key: float) -> list[np.ndarray]:
    """:returns: the heightmap (gray, [0, 1]) and its 2x box-filtered mips"""
    image = read_image(path)[:, :, 0].astype(np.float64)
    pyramid = [image]
    while min(image.shape) >= 4:
        rows, cols = (image.shape[0] // 2) * 2, (image.shape[1] // 2) * 2
        image = 0.25 * (
            image[0:rows:2, 0:cols:2]
            + image[1:rows:2, 0:cols:2]
            + image[0:rows:2, 1:cols:2]
            + image[1:rows:2, 1:cols:2]
        )
        pyramid.append(image)
    return pyramid


def _pick_level(pyramid: list[np.ndarray], width: float, spacing: float) -> np.ndarray:
    """:returns: the finest mip whose texels are at least *spacing* apart"""
    texel = width / pyramid[0].shape[1]
    level = 0 if spacing <= texel else int(math.floor(math.log2(spacing / texel)))
    return pyramid[min(level, len(pyramid) - 1)]


@functools.lru_cache(maxsize=16)
def _paint_channels(path, _mtime_key, colors, tolerance, falloff, width) -> np.ndarray:
    """:returns: (6, rows, cols) per-biome coverage in [0, 1], blurred by falloff"""
    image = read_image(path)
    rgb = image[:, :, :3] if image.shape[2] >= 3 else np.repeat(image, 3, axis=2)
    channels = np.zeros((len(BIOMES), *rgb.shape[:2]))
    for color, biome in colors:
        target = np.array(
            [int(color.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
        )
        match = np.linalg.norm(rgb - target, axis=2) < tolerance
        channels[BIOMES.index(biome)] = np.maximum(channels[BIOMES.index(biome)], match)
    if falloff > 0.0:
        sigma = falloff / (width / rgb.shape[1])
        channels = np.stack(
            [ndimage.gaussian_filter(ch, sigma, mode="constant") for ch in channels]
        )
    return channels
