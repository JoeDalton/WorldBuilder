"""
The mountains' shadow at planetary scale: a "shadow ceiling" cube map, baked
for one sun.

For every texel of a cube map over the planet (its directions, the
planet's own frame), the **ceiling** is the lowest altitude from which the
sun clears every occluder toward it, and the **distance** is how far the
occluder that sets it lies. A receiver at altitude ``a`` over that texel
sees the sun clear the occluder by ``(a - ceiling) / distance`` radians: the
penumbra is a smoothstep of that over the sun's angular radius
(:func:`terrain_visibility`, shadows.glsl's ``terrainShadow``). Shadow on a
heightfield is monotonic in altitude, so one ceiling says it all.

The bake does not march every texel toward the sun. A point's ceiling only
depends on the point a step toward the sub-solar point (along the great
circle to it): the ray that just clears everything from there, followed
back, is the one that does from here::

    ceiling(P) = max(height(P'), ceiling(P')) - gain(P)

``gain`` is how much the sun's ray rises over that step (curvature
included: closed form, :func:`ray_gain`). On a grid aligned with the sun,
columns along great circles through the sub-solar point and rows of equal
angle from it, ``P'`` is the node a row nearer the sun, exactly: the bake
is one pass over the rows, each an array operation over every column, with
no interpolation along the way (interpolating at every step rounded the
ceiling's crest down a little each time: the shadows came out ~20 % short).
The ground is read onto that grid once, and the ceiling back onto the cube
map once. The nearest occluder counted is :data:`STEP_TEXELS` away, which
keeps a texel's own slope from shadowing it (nearer occluders are part E's
local map).

Heights come from :meth:`Planet.sample_dirs` at the texel centres,
band-limited to two texels, stamps included; the sea is flat at 0.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import NodePath, PTA_int, SamplerState, Texture

from world_builder.clouds.weather import sample_cube
from world_builder.sky.starfield import cube_directions

#: The bake's step toward the sun, in texels at a face's centre (2 / n
#: radians each, the widest): its nearest occluder (a texel's own ground
#: never shadows it).
STEP_TEXELS = 2.0
#: :meth:`ShadowMap.visibility` keeps a point's shade per cube of this
#: side, metres (a shadow's edge is wider), at most this many.
SHADE_CELL = 25.0
SHADE_CACHE_SIZE = 4096
#: A ceiling where nothing occludes the sun, metres (finite: it is
#: interpolated).
NO_CEILING = -1.0e5
#: Past this angle from the sub-solar point no ground sees the sun, metres
#: above its highest terrain notwithstanding: the bake stops there and
#: leaves a ceiling over everything.
FULL_SHADOW = 1.0e5


def height_cube(planet, size: int, ocean: bool = True) -> np.ndarray:
    """The planet's ground at every texel centre of a cube map, band-limited
    to two texels (stamps included): what occludes the sun.

    :param planet: a :class:`~world_builder.terrain.planet.Planet`
    :param size: texels per face edge
    :param ocean: the sea's surface at 0 hides the seabed
    :returns: (6, size, size) float32, metres above sea level
    """
    dirs = cube_directions(size).reshape(-1, 3)
    texel = 0.5 * math.pi * planet.spec.radius / size
    heights = planet.sample_dirs(dirs, min_wavelength=2.0 * texel, apply_stamps=True)[0]
    if ocean:
        heights = np.maximum(heights, 0.0)
    return heights.reshape(6, size, size).astype(np.float32)


def ray_gain(zenith: ArrayLike, step: float, radius: ArrayLike) -> np.ndarray:
    """How far the sun's ray from radius *radius* rises above the ground's
    sphere over *step* radians of arc toward the sun, whose zenith angle
    there is *zenith*: ``r (sin z / sin(z - step) - 1)`` (the law of sines
    in the plane of the centre, the point and the sun). Huge once the sun
    is within *step* of the zenith.

    :returns: metres, broadcast
    """
    z = np.asarray(zenith, dtype=np.float64)
    ahead = np.sin(np.maximum(z - step, 1e-9))
    return np.asarray(radius) * (np.sin(z) / ahead - 1.0)


def ceiling_cube(
    heights: np.ndarray, radius: float, sun: ArrayLike, step_texels: float = STEP_TEXELS
) -> np.ndarray:
    """Bake the shadow ceiling for one sun (see the module).

    :param heights: (6, n, n) the ground (:func:`height_cube`), metres
    :param radius: sea-level radius, metres
    :param sun: toward the sun, the planet's frame (any length)
    :param step_texels: the nearest occluder counted, in texels
        (:data:`STEP_TEXELS`)
    :returns: (6, n, n, 2) float32: the ceiling (metres above sea level,
        :data:`NO_CEILING` where nothing occludes) and the distance to the
        occluder that sets it (metres)
    """
    size = heights.shape[1]
    sun = np.asarray(sun, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    texel = 2.0 / size  # radians at a face's centre, the widest
    # Rows half a texel apart: the march along each ray sees a sharp rim's
    # crest (a texel apart, it stepped past it: ceilings 200 m low).
    step = 0.5 * texel
    near = max(1, round(step_texels * texel / step))  # rows to the nearest counted
    top = max(float(heights.max()), 0.0)
    # Beyond this nothing sees the sun (the ray from the highest ground
    # grazes the sea-level sphere's horizon).
    last = 0.5 * math.pi + math.acos(radius / (radius + top + 1.0))
    rows = int(math.ceil(last / step)) + near + 1
    columns = int(math.ceil(2.0 * math.pi / texel))  # a texel apart at 90 degrees

    # The sun's frame and the grid's nodes.
    side = np.cross(sun, [1.0, 0.0, 0.0] if abs(sun[0]) < 0.9 else [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(sun, side)
    psi = np.arange(rows) * step
    phi = np.arange(columns) * (2.0 * math.pi / columns)
    ring = np.cos(phi)[:, None] * side + np.sin(phi)[:, None] * other
    ground = np.empty((rows, columns))
    for row, angle in enumerate(psi):  # row by row: (rows, columns, 3) is big
        nodes = math.cos(angle) * sun + math.sin(angle) * ring
        ground[row] = sample_cube(heights, nodes)

    # The pass: every column at once, row by row away from the sun.
    ceiling = np.full((rows, columns), NO_CEILING)
    distance = np.full((rows, columns), step * radius)
    for row in range(1, rows):
        gain = ray_gain(psi[row], step, radius + np.maximum(ground[row], 0.0))
        up_ground, up_ceiling = ground[row - 1], ceiling[row - 1]
        binding = up_ground >= up_ceiling
        ceiling[row] = np.maximum(np.maximum(up_ground, up_ceiling) - gain, NO_CEILING)
        distance[row] = np.where(binding, 0.0, distance[row - 1]) + step * radius
    # Not counting the nearest occluders: a node's ceiling from the node
    # near - 1 rows up, carried down over those rows.
    if near > 1:
        carried = np.full_like(ceiling, NO_CEILING)
        span = (near - 1) * step
        gain = ray_gain(
            psi[near - 1 :, None], span, radius + np.maximum(ground[near - 1 :], 0.0)
        )
        carried[near - 1 :] = np.maximum(ceiling[: rows - near + 1] - gain, NO_CEILING)
        far = np.full_like(distance, step * radius)
        far[near - 1 :] = distance[: rows - near + 1] + span * radius
        ceiling, distance = carried, far

    # Back onto the cube map: each texel between its four nodes (columns
    # wrap round the sun).
    dirs = cube_directions(size).reshape(-1, 3)
    angle = np.arccos(np.clip(dirs @ sun, -1.0, 1.0))
    around = np.arctan2(dirs @ other, dirs @ side) % (2.0 * math.pi)
    y = np.minimum(angle / step, rows - 1.0)
    x = around / (2.0 * math.pi) * columns
    y0 = np.minimum(np.floor(y).astype(np.int64), rows - 2)
    x0 = np.floor(x).astype(np.int64) % columns
    x1 = (x0 + 1) % columns
    ty, tx = y - y0, x - np.floor(x)
    out = np.empty((len(dirs), 2))
    for channel, grid in enumerate((ceiling, distance)):
        top_row = grid[y0, x0] * (1 - tx) + grid[y0, x1] * tx
        bottom = grid[y0 + 1, x0] * (1 - tx) + grid[y0 + 1, x1] * tx
        out[:, channel] = top_row * (1 - ty) + bottom * ty
    out[angle >= (rows - 1) * step, 0] = FULL_SHADOW
    return out.reshape(6, size, size, 2).astype(np.float32)


def terrain_visibility(
    ceiling: ArrayLike, distance: ArrayLike, altitude: ArrayLike, sun_radius: float
) -> np.ndarray:
    """The share of the sun the mountains leave a point *altitude* metres
    above sea level over a texel of *ceiling* and *distance*: it clears the
    occluder by ``(altitude - ceiling) / distance`` radians, a smoothstep of
    that over the sun's angular radius (shadows.glsl's ``terrainShadow``).

    :returns: (...) in [0, 1], broadcast
    """
    margin = (np.asarray(altitude) - np.asarray(ceiling)) / np.maximum(distance, 1.0)
    t = np.clip((margin + sun_radius) / (2.0 * sun_radius), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ── a body's map, baked in the pool ────────────────────────────────────────────


def _face_heights(spec, size: int, face: int, ocean: bool) -> np.ndarray:
    """One face of :func:`height_cube`, in a worker (the planet cached
    there): the slow part, ~2.5 us a texel."""
    from world_builder.terrain.chunkgen import planet_for

    planet = planet_for(spec)
    dirs = cube_directions(size)[face].reshape(-1, 3)
    texel = 0.5 * math.pi * planet.spec.radius / size
    heights = planet.sample_dirs(dirs, min_wavelength=2.0 * texel, apply_stamps=True)[0]
    if ocean:
        heights = np.maximum(heights, 0.0)
    return heights.reshape(size, size).astype(np.float32)


def _bake(heights: np.ndarray, radius: float, sun: np.ndarray) -> np.ndarray:
    """:func:`ceiling_cube`, in a worker (~2 s at 1024)."""
    return ceiling_cube(heights, radius, sun)


class ShadowMap:
    """A body's mountain shadow (see the module): its heights sampled once,
    its ceiling baked for each sun, both in the worker pool, and swapped in
    when ready. The shaders read it through :meth:`bind`
    (shadows.glsl's ``terrainShadow``), the CPU through :meth:`visibility`.

    The sun is static as far as the map goes: :meth:`set_sun` bakes again
    (~2 s at 1024 in a worker), and the last map stays until the new one is
    in. A swinging sun gets a new map as fast as the pool bakes them.

    :ivar size: texels per face edge (0: no map)
    """

    def __init__(
        self,
        terrain: Any,
        executor: Any,
        size: int,
        sun_direction: ArrayLike,
    ) -> None:
        """
        :param terrain: the body's :class:`~world_builder.terrain.terrain.Terrain`
            (its planet, radius, frame)
        :param executor: the world's :class:`ChunkExecutor`
        :param size: texels per face edge (``ShadowSettings.terrain_map_size``;
            0 = none)
        :param sun_direction: toward the sun, world frame
        """
        self._terrain = terrain
        self._executor = executor
        # A cube map from the start (the shaders' sampler), one texel until
        # the first bake.
        self._texture = Texture("terrain_shadow")
        self._texture.setup_cube_map(1, Texture.T_half_float, Texture.F_rg16)
        self._texture.set_ram_image(bytes(6 * 4))
        self._on = PTA_int.empty_array(1)
        self._on[0] = 0
        self.cube: np.ndarray | None = None  # (6, n, n, 2) metres, as baked
        self._heights_jobs: list = []
        self._heights: np.ndarray | None = None
        self._bake_job = None
        self._baking_sun: np.ndarray | None = None
        self._sun = np.zeros(3)
        self.bake_seconds = 0.0
        self._shades: dict[tuple, float] = {}  # visibility's, per cell
        self._shades_for = None  # the map they were read from
        self._bake_started = 0.0
        self.size = 0
        self.set_sun(sun_direction)
        self.set_size(size)

    @property
    def loading(self) -> bool:
        """Whether the first map is still being made."""
        return self.size > 0 and self.cube is None

    @property
    def current(self) -> bool:
        """Whether the map drawn is the one for the sun now (none wanted
        counts): a later sun is still baking otherwise."""
        if not self.size:
            return True
        baked = self._bake_job is None and self._baking_sun is not None
        return baked and np.array_equal(self._baking_sun, self._sun)

    def bind(self, node: NodePath) -> None:
        """Give a node's shaders this body's map (shadows.glsl's
        ``uTerrainShadow``, ``uTerrainShadowOn``: off until baked)."""
        node.set_shader_input("uTerrainShadow", self._texture)
        node.set_shader_input("uTerrainShadowOn", self._on)

    def set_size(self, size: int) -> None:
        """A new resolution: the heights are sampled again (0: no map)."""
        if size == self.size:
            return
        self.size = int(size)
        self.cube = self._heights = self._bake_job = None
        self._on[0] = 0
        spec = self._terrain.spec
        self._heights_jobs = (
            [
                self._executor.submit_once(
                    _face_heights, spec, self.size, face, bool(spec.has_ocean)
                )
                for face in range(6)
            ]
            if self.size
            else []
        )

    def set_sun(self, direction: ArrayLike) -> None:
        """Toward the sun, world frame: a new bake if it moved (at the next
        :meth:`update`)."""
        sun = np.asarray(direction, dtype=np.float64)
        sun = self._terrain.rotation.T @ (sun / np.linalg.norm(sun))
        self._sun = sun

    def update(self) -> None:
        """Per frame: collect what the pool made, start the next bake."""
        if not self.size:
            return
        if self._heights is None:
            if not all(job.done() for job in self._heights_jobs):
                return
            self._heights = np.stack([job.result() for job in self._heights_jobs])
            self._heights_jobs = []
        if self._bake_job is not None:
            if not self._bake_job.done():
                return
            self.cube = self._bake_job.result()
            self.bake_seconds = time.perf_counter() - self._bake_started
            self._upload(self.cube)
            self._bake_job = None
        if self._baking_sun is None or not np.array_equal(self._baking_sun, self._sun):
            self._baking_sun = self._sun.copy()
            self._bake_started = time.perf_counter()
            self._bake_job = self._executor.submit_once(
                _bake, self._heights, self._terrain.radius, self._baking_sun
            )

    def _upload(self, cube: np.ndarray) -> None:
        """The map as a half-float cube texture, in km (half floats reach
        65504)."""
        size = cube.shape[1]
        if self._texture.get_x_size() != size:
            self._texture.setup_cube_map(size, Texture.T_half_float, Texture.F_rg16)
            self._texture.set_magfilter(SamplerState.FT_linear)
            # Mipmapped: from orbit a whole face spans a few hundred pixels,
            # and reading it texel-wide apart thrashed the texture cache.
            self._texture.set_minfilter(SamplerState.FT_linear_mipmap_linear)
        self._texture.set_ram_image(
            np.ascontiguousarray(cube / 1e3, dtype=np.float16).tobytes()
        )
        self._on[0] = 1

    def visibility(self, points_world: ArrayLike, sun_radius: float) -> np.ndarray:
        """The share of the sun the mountains leave points: the shader's
        ``terrainShadow`` on the CPU (the map as last baked).

        :param points_world: (..., 3) world frame, metres
        :param sun_radius: the sun's angular radius, radians
        :returns: (...) in [0, 1]; 1 before the first map
        """
        points = np.asarray(points_world, dtype=np.float64)
        if self.cube is None:
            return np.ones(points.shape[:-1])
        # ~0.1 ms a call (numpy on a few points), every frame: a point's
        # shade is kept per SHADE_CELL, until the next bake.
        if self._shades_for is not self.cube or len(self._shades) > SHADE_CACHE_SIZE:
            self._shades_for, self._shades = self.cube, {}
        planet = self._terrain.to_planet(points.reshape(-1, 3))
        cells = [tuple(c) for c in np.round(planet / SHADE_CELL).astype(np.int64)]
        missing = [i for i, cell in enumerate(cells) if cell not in self._shades]
        if missing:
            rho = np.linalg.norm(planet[missing], axis=-1)
            ceiling, distance = np.moveaxis(
                sample_cube(self.cube, planet[missing]), -1, 0
            )
            shade = terrain_visibility(
                ceiling, distance, rho - self._terrain.radius, sun_radius
            )
            for i, value in zip(missing, np.atleast_1d(shade)):
                self._shades[cells[i]] = float(value)
        shades = np.array([self._shades[cell] for cell in cells])
        return shades.reshape(points.shape[:-1])
