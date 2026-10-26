"""
One rock as a terrain body: what lets a big rock near the camera be drawn by
a :class:`~world_builder.terrain.terrain.Terrain`, with the terrain's level of
detail, instead of an instanced mesh.

A :class:`RockSpec` names the rock (its field's shapes and seed, its
prototype, its radius, its ice, its own detail's seed); its evaluator,
:class:`RockBody`, follows the terrain's body protocol (see
:class:`~world_builder.terrain.planet.Planet`):

- **elevation** at a direction: the prototype's radius there
  (:mod:`world_builder.asteroids.shape`) times the rock's radius, plus
  metre-scale detail of its own (fbm, seeded per rock), less the radius;
  band-limited as a planet is, so a coarse chunk is a smoothed rock and the
  instanced meshes are the same rock, coarser;
- **biomes**: snow (ice) and rock in the rock's proportions, set directly;
- **textures** repeat every :data:`ROCK_UV_PERIOD` metres along a face, so a
  small body's textures are not stretched;
- **height bounds** from its shape: the LOD pass culls behind its smallest
  radius.

Positions are in the rock's body frame: the field turns it (the terrain's
orientation). Pure numpy, for worker processes.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from world_builder.asteroids.shape import face_grid_dirs, prototype_shape
from world_builder.asteroids.spec import ShapeSpec
from world_builder.terrain import noise
from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.cubesphere import grid_for
from world_builder.terrain.planet import GridSample, face_grid

#: Texture metres per period along a face: a multiple of the rock and snow
#: textures' tile times the macro scale (4 m x 8), dividing the planets'.
ROCK_UV_PERIOD = 128.0
#: Octaves of a rock's own detail.
DETAIL_OCTAVES = 5
_SNOW, _ROCK = BIOMES.index("snow"), BIOMES.index("rock")


@dataclass(frozen=True)
class RockSpec:
    """One rock of a field, as a terrain body.

    :ivar shape: the field's shapes
    :ivar seed: the field's seed
    :ivar index: the rock's prototype
    :ivar radius: its bounding radius (before its detail), metres
    :ivar ice: its share of ice
    :ivar detail_seed: its own detail's seed (its id, say)
    :ivar detail_m: that detail's amplitude, metres
    :ivar detail_wavelength_m: its largest wavelength, metres
    :ivar lowest: its shape's smallest radius, in units of the radius
        (``Prototype.lowest``); None = measured here (slower)
    :ivar name: for display
    """

    shape: ShapeSpec
    seed: int
    index: int
    radius: float
    ice: float = 0.0
    detail_seed: int = 0
    detail_m: float = 1.5
    detail_wavelength_m: float = 24.0
    lowest: float | None = None
    name: str = "rock"

    #: What a planet's spec has and a rock lacks (the body protocol)
    has_ocean: ClassVar[bool] = False
    #: Lit as an object, each face as it faces the sun (no day and night by
    #: the vertical, as a planet's)
    radial_light: ClassVar[bool] = False
    has_vegetation: ClassVar[bool] = False
    paints: ClassVar[tuple] = ()
    stamps: ClassVar[tuple] = ()

    def evaluator(self) -> RockBody:
        return rock_body(self)


@functools.lru_cache(maxsize=64)
def rock_body(spec: RockSpec) -> RockBody:
    """:returns: a cached :class:`RockBody` (one per process and spec)"""
    return RockBody(spec)


class RockBody:
    """Evaluates a :class:`RockSpec` (the terrain's body protocol).

    :ivar spec: the rock
    :ivar grid: its chunk grid
    :ivar height_bounds: (lowest, highest) ground, metres from the radius
    """

    def __init__(self, spec: RockSpec):
        self.spec = spec
        self.grid = grid_for(spec.radius, ROCK_UV_PERIOD)
        # The smallest radius with a margin: what can hide the far side.
        lowest = spec.lowest
        if lowest is None:
            lowest = float(self.shape.radius(face_grid_dirs(48).reshape(-1, 3)).min())
        r = spec.radius
        self.height_bounds = (r * (0.95 * lowest - 1.0) - spec.detail_m, spec.detail_m)
        self._detail_seed = int(spec.detail_seed) & 0x7FFFFFFF

    @functools.cached_property
    def shape(self):
        """The prototype's shape, made when first sampled: the main process
        only needs the grid and the bounds (the workers sample)."""
        return prototype_shape(self.spec.shape, self.spec.seed, self.spec.index)

    def elevation(
        self, dirs: np.ndarray, min_wavelength: float | None = None
    ) -> np.ndarray:
        """:param dirs: (n, 3) unit directions, body frame
        :param min_wavelength: smaller detail fades out, metres
        :returns: (n,) metres from the rock's radius"""
        r = self.spec.radius
        band = None if min_wavelength is None else min_wavelength / r
        surface = r * self.shape.radius(dirs, band)
        if self.spec.detail_m > 0.0:
            surface = surface + self.spec.detail_m * noise.fbm(
                dirs * r,
                self.spec.detail_wavelength_m,
                DETAIL_OCTAVES,
                self._detail_seed,
                min_wavelength=min_wavelength,
            )
        return surface - r

    def weights(self, count: int) -> np.ndarray:
        """:returns: (count, 6) the rock's biome weights"""
        w = np.zeros((count, len(BIOMES)))
        w[:, _SNOW] = self.spec.ice
        w[:, _ROCK] = 1.0 - self.spec.ice
        return w

    def sample_dirs(
        self,
        dirs: np.ndarray,
        min_wavelength: float | None = None,
        apply_stamps: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """:returns: elevation, temperature, humidity at directions (the last
        two unused: zeros)"""
        dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
        zeros = np.zeros(len(dirs))
        return self.elevation(dirs, min_wavelength), zeros, zeros

    def sample_face_grid(
        self,
        face: int,
        ka: np.ndarray,
        kb: np.ndarray,
        step: int,
        apply_stamps: bool = True,
    ) -> GridSample:
        """Sample a grid of one cube face, band-limited to four of its
        spacings (as :meth:`Planet.sample_face_grid`)."""
        grid = self.grid
        dirs = face_grid(face, grid.param(ka), grid.param(kb))
        elevation = self.elevation(dirs, 4.0 * grid.spacing0 * step)
        shape = (len(kb), len(ka))
        zeros = np.zeros(shape)
        return GridSample(
            dirs=dirs,
            elevation=elevation.reshape(shape),
            temperature=zeros,
            humidity=zeros,
            weights=self.weights(len(dirs)).reshape(*shape, len(BIOMES)),
        )
