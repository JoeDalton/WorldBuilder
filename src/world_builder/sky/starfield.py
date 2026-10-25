"""
A procedural star field: point stars, a Milky Way band, nebulae.

Generated with numpy, so there are no assets to ship; :func:`load_cube_map`
takes a painted cube map instead (e.g. SpaceFlight's Spacescape skyboxes,
``models/skyboxes/purple_#.png``).

* **Stars** are a catalogue (:func:`make_stars`) drawn as points, each over
  at most 2x2 pixels at any resolution (a cube map would need ~4096 px a face
  to keep them that small at 1920x1080).
* **The diffuse light** (the band, its dust lanes, nebulae) is smooth, so a
  small cube map holds it (:func:`make_glow`, uploaded by
  :func:`glow_texture`), as parameters rather than colours: the sky shader
  colours them, so the palette and intensity (:class:`NebulaStyle`) are
  live.

:class:`Starfield` builds both for :class:`sky.Sky`.

Faces follow OpenGL's cube map convention (what ``texture(samplerCube, dir)``
reads), so :func:`cube_directions` gives the direction of every texel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import Filename, SamplerState, Texture

from world_builder.terrain.noise import fbm

#: (major axis, s axis, t axis) per face, OpenGL order +X -X +Y -Y +Z -Z:
#: direction = major + sc * s_axis + tc * t_axis, with sc, tc in [-1, 1].
_FACES = np.array(
    [
        [(1, 0, 0), (0, 0, -1), (0, -1, 0)],
        [(-1, 0, 0), (0, 0, 1), (0, -1, 0)],
        [(0, 1, 0), (1, 0, 0), (0, 0, 1)],
        [(0, -1, 0), (1, 0, 0), (0, 0, -1)],
        [(0, 0, 1), (1, 0, 0), (0, -1, 0)],
        [(0, 0, -1), (-1, 0, 0), (0, -1, 0)],
    ],
    dtype=np.float64,
)


def cube_directions(size: int) -> np.ndarray:
    """:returns: (6, size, size, 3) unit directions of every texel centre,
    [face, row (t), column (s)]"""
    c = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    tc, sc = np.meshgrid(c, c, indexing="ij")
    out = np.empty((6, size, size, 3))
    for face, (major, s_axis, t_axis) in enumerate(_FACES):
        d = major + sc[..., None] * s_axis + tc[..., None] * t_axis
        out[face] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return out


def directions_to_texels(
    dirs: ArrayLike, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cube map texel under each direction (the inverse of
    :func:`cube_directions`).

    :param dirs: (N, 3) directions (any length)
    :param size: texels per face edge
    :returns: face (N,) int, row and column (N,) float (texel units,
        centres at .5)
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    axis = np.argmax(np.abs(dirs), axis=1)
    rows = np.arange(len(dirs))
    face = 2 * axis + (dirs[rows, axis] < 0.0)
    major = np.abs(dirs[rows, axis])
    s_axis = _FACES[face, 1]
    t_axis = _FACES[face, 2]
    sc = np.einsum("ij,ij->i", dirs, s_axis) / major
    tc = np.einsum("ij,ij->i", dirs, t_axis) / major
    return face, (tc + 1.0) * 0.5 * size, (sc + 1.0) * 0.5 * size


#: Nebula colour palettes: three linear colour stops the nebula's tint runs
#: through (:attr:`NebulaStyle.palette`).
NEBULA_PALETTES = {
    "violet": ((0.55, 0.20, 0.50), (0.85, 0.25, 0.35), (0.35, 0.25, 0.75)),
    "emerald": ((0.15, 0.55, 0.45), (0.25, 0.45, 0.70), (0.45, 0.65, 0.30)),
    "ember": ((0.85, 0.35, 0.15), (0.75, 0.20, 0.25), (0.90, 0.60, 0.25)),
    "glacier": ((0.25, 0.45, 0.85), (0.35, 0.70, 0.80), (0.55, 0.40, 0.85)),
    "dust": ((0.60, 0.50, 0.40), (0.45, 0.40, 0.38), (0.70, 0.62, 0.50)),
}


@dataclass(frozen=True)
class NebulaStyle:
    """How the diffuse light is coloured: live uniforms, no regeneration.

    :ivar palette: three linear RGB stops across the nebula's tint
    :ivar intensity: the nebula's radiance scale (0 = none)
    :ivar milky_way: the Milky Way band's linear colour times its intensity
    """

    palette: tuple[tuple[float, float, float], ...] = NEBULA_PALETTES["violet"]
    intensity: float = 0.01
    milky_way: tuple[float, float, float] = (0.06, 0.057, 0.051)


@dataclass(frozen=True)
class StarCatalogue:
    """Point stars.

    :ivar dirs: (n, 3) float32 unit directions in the star map's frame
    :ivar flux: (n,) float32 brightness (linear radiance of a pixel the star
        covers whole, before :class:`Sky`'s ``star_brightness``)
    :ivar colour: (n, 3) float32 linear tint, about unit luminance
    """

    dirs: np.ndarray
    flux: np.ndarray
    colour: np.ndarray


#: Star tints, about unit luminance: slightly red, slightly blue.
STAR_RED = np.array([1.18, 0.95, 0.80])
STAR_BLUE = np.array([0.82, 0.98, 1.25])


def make_stars(
    seed: int = 3,
    count: int = 14000,
    galactic_pole: ArrayLike = (0.25, -0.55, 0.8),
    red: float = 0.25,
    blue: float = 0.25,
) -> StarCatalogue:
    """Stars, more of them toward the Milky Way, with a power law of
    brightness (few bright, many faint).

    :param seed: random seed
    :param count: how many stars
    :param galactic_pole: normal of the Milky Way's plane
    :param red: share of stars tinted slightly red (each by a random amount)
    :param blue: share tinted slightly blue; the rest are white
    :returns: the catalogue, in the star map's frame
    """
    rng = np.random.default_rng(seed)
    pole = np.asarray(galactic_pole, dtype=np.float64)
    pole /= np.linalg.norm(pole)
    picked = []
    while sum(len(p) for p in picked) < count:
        d = rng.normal(size=(count, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        keep = rng.uniform(size=count) < 0.35 + 0.65 * np.exp(
            -(((d @ pole) / 0.25) ** 2)
        )
        picked.append(d[keep])
    dirs = np.concatenate(picked)[:count]
    flux = np.clip(0.04 * rng.pareto(1.6, count) + 0.03, 0.0, 6.0)
    kind = rng.uniform(size=count)
    amount = rng.uniform(0.35, 1.0, count)[:, None]
    colour = np.ones((count, 3))
    reds, blues = kind < red, (kind >= red) & (kind < red + blue)
    colour[reds] += (STAR_RED - 1.0) * amount[reds]
    colour[blues] += (STAR_BLUE - 1.0) * amount[blues]
    return StarCatalogue(
        dirs.astype(np.float32), flux.astype(np.float32), colour.astype(np.float32)
    )


def make_glow(
    size: int = 256,
    seed: int = 3,
    galactic_pole: ArrayLike = (0.25, -0.55, 0.8),
    nebula_scale: float = 1.0,
    nebula_coverage: float = 0.5,
) -> np.ndarray:
    """The diffuse light's parameters as a cube map, in [0, 1]; :class:`Sky`
    colours them (:class:`NebulaStyle`), so the look changes live.

    * r: the Milky Way band (its clumps and dust lanes)
    * g: the nebulae's density
    * b: where the nebula is in its palette

    :param size: face resolution, texels (smooth light: 256 is plenty)
    :param seed: noise seed
    :param galactic_pole: normal of the Milky Way's plane
    :param nebula_scale: size of the nebulae (1 = features of ~0.4 rad)
    :param nebula_coverage: about the share of the sky with some nebula
    :returns: (6, size, size, 3) float32, in :func:`cube_directions`'
        layout
    """
    pole = np.asarray(galactic_pole, dtype=np.float64)
    pole /= np.linalg.norm(pole)
    dirs = cube_directions(size).reshape(-1, 3)
    latitude = dirs @ pole  # sine of the galactic latitude
    band = np.exp(-((latitude / 0.16) ** 2))
    radius = 1000.0  # noise is evaluated on a sphere of this size
    clumps = 0.5 + 0.5 * fbm(dirs * radius, 220.0, 5, seed * 13 + 1)
    dust = np.clip(fbm(dirs * radius, 120.0, 4, seed * 13 + 2) * 2.2, 0.0, 1.0)
    glow = band * clumps * (1.0 - 0.75 * dust * np.exp(-((latitude / 0.06) ** 2)))
    raw = fbm(dirs * radius, 400.0 * nebula_scale, 5, seed * 13 + 3)
    # The fbm is about symmetric around 0: shifting it sets how much of the
    # sky clears zero (about half at 0.5).
    nebula = np.clip(raw + (nebula_coverage - 0.5) * 0.6, 0.0, None) ** 2
    tint = fbm(dirs * radius, 700.0 * nebula_scale, 2, seed * 13 + 4)
    params = np.stack(
        [
            np.clip(glow, 0.0, 1.0),
            np.clip(nebula / 0.36, 0.0, 1.0),
            np.clip(0.5 + 0.9 * tint, 0.0, 1.0),
        ],
        axis=-1,
    )
    return params.reshape(6, size, size, 3).astype(np.float32)


def glow_texture(glow: np.ndarray) -> Texture:
    """:returns: :func:`make_glow` as a linear, mipmapped RGB8 cube map"""
    size = glow.shape[1]
    data = (np.clip(glow, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    tex = Texture("sky_glow")
    tex.setup_cube_map(size, Texture.T_unsigned_byte, Texture.F_rgb8)
    # Panda's RAM layout is BGR, pages in face order, rows bottom-up = t up.
    tex.set_ram_image(np.ascontiguousarray(data[..., ::-1]).tobytes())
    _filtered(tex)
    return tex


class Starfield:
    """A procedural sky for :class:`Sky`: point stars (drawn over at most
    2x2 pixels at any resolution) and the diffuse light's parameters
    (coloured live by a :class:`NebulaStyle`).

    Usage::

        sky = Sky(render, Starfield(seed=3), atmospheres)

    :ivar catalogue: the point stars (:func:`make_stars`)
    """

    def __init__(
        self,
        seed: int = 3,
        stars: int = 14000,
        galactic_pole: ArrayLike = (0.25, -0.55, 0.8),
        glow_size: int = 256,
        nebula_scale: float = 1.0,
        nebula_coverage: float = 0.5,
        red: float = 0.25,
        blue: float = 0.25,
    ) -> None:
        """See :func:`make_stars` and :func:`make_glow` for the parameters
        (*stars* is :func:`make_stars`' *count*, *glow_size* :func:`make_glow`'s
        *size*); both share *seed* and *galactic_pole*."""
        self.catalogue = make_stars(seed, stars, galactic_pole, red, blue)
        self._glow_args = (seed, galactic_pole, nebula_scale, nebula_coverage)
        self.glow_size = int(glow_size)
        # Made when first asked for (~0.4 s at 256), at the size the sky uses.
        self._glows: dict[int, Texture] = {}
        self._glow_jobs: dict[int, Any] = {}

    def prefetch_glow(self, executor: Any, size: int | None = None) -> None:
        """Start making the glow at *size* in a worker pool, so it is ready
        when the sky asks for it (:meth:`glow_at` takes it from there).

        :param executor: a :class:`world_builder.terrain.workers.ChunkExecutor`
        :param size: texels per face; None = ``glow_size``
        """
        size = int(size or self.glow_size)
        if size not in self._glows and size not in self._glow_jobs:
            self._glow_jobs[size] = executor.submit_once(
                make_glow, size, *self._glow_args
            )

    @property
    def glow(self) -> Texture:
        """The diffuse light's cube map (:func:`glow_texture`) at
        ``glow_size``."""
        return self.glow_at(self.glow_size)

    def glow_at(self, size: int) -> Texture:
        """:returns: the diffuse light's cube map at another size (the same
        sky; made once per size)"""
        size = int(size)
        if size not in self._glows:
            job = self._glow_jobs.pop(size, None)
            data = (
                job.result() if job is not None else make_glow(size, *self._glow_args)
            )
            self._glows[size] = glow_texture(data)
        return self._glows[size]


def load_cube_map(pattern: str, loader: Any) -> Texture:
    """A painted cube map (six images, ``#`` = 0-5 in OpenGL face order),
    mipmapped, for :class:`Sky`.

    :param pattern: e.g. ``".../skyboxes/purple_#.png"``
    :param loader: the ShowBase loader
    """
    tex = loader.load_cube_map(Filename.from_os_specific(pattern))
    _filtered(tex)
    return tex


def _filtered(tex: Texture) -> None:
    tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)


def angular_texel(size: int) -> float:
    """:returns: the angle a texel spans at a face centre, radians"""
    return 2.0 * math.atan(1.0 / size)
