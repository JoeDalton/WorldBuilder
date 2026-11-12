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
from panda3d.core import (
    LVecBase4f,
    NodePath,
    PTA_int,
    PTA_LVecBase4f,
    SamplerState,
    Texture,
)

from world_builder.clouds.weather import sample_cube
from world_builder.sky.starfield import (
    cube_directions,
    directions_to_texels,
    texel_directions,
)
from world_builder.terrain.chart import AzimuthalEquidistantChart, latlon
from world_builder.utils import sun_moved

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


def march_limits(
    size: int, radius: float, top: float, step_texels: float = STEP_TEXELS
) -> tuple[float, int, float, int]:
    """The march's constants for a cube map of *size* (:func:`march_ceiling`,
    ``shadow_bake.frag``).

    :returns: (step, radians: half a texel at a face's centre; the first
        step counted (the nearest occluder: *step_texels*); the angle from
        the sub-solar point past which no ground sees the sun; the most
        steps a texel takes)
    """
    step = 1.0 / size  # half of a face centre's texel (2 / size radians)
    near = max(1, round(step_texels * (2.0 / size) / step))
    horizon = math.acos(radius / (radius + max(top, 0.0) + 1.0))
    last = 0.5 * math.pi + horizon
    # Past the terminator a ray first dips (to twice the sun's depth below
    # the horizon), then climbs past the highest ground (a horizon's arc).
    steps = int(math.ceil(3.0 * horizon / step)) + near + 1
    return step, near, last, steps


#: The GPU bake's stop: a block of this many texels (a face's side over
#: it) holds the highest ground within TOPS_REACH_TEXELS of any of its
#: texels (:func:`local_tops`).
TOPS_BLOCK_TEXELS = 64
TOPS_REACH_TEXELS = 128


def local_tops(heights: np.ndarray) -> np.ndarray:
    """The highest ground near every block of a cube map: for each block of
    :data:`TOPS_BLOCK_TEXELS`, the highest ground within
    :data:`TOPS_REACH_TEXELS` (at a face's centre: an angle) of any of its
    texels, from a max-pooled copy (conservative: whole coarse cells that
    reach so far). The GPU's march stops on it, not on the planet's highest
    ground, while it is that near: a lowland texel stops after a few steps
    instead of climbing past the highest peak.

    :param heights: (6, n, n) the ground, metres
    :returns: (6, b, b) float32 (b = n / TOPS_BLOCK_TEXELS, at least 1)
    """
    size = heights.shape[1]
    block = min(TOPS_BLOCK_TEXELS, size)
    blocks = max(1, size // block)
    cell = max(1, block // 4)
    cells = size // cell
    coarse = (
        heights[:, : cells * cell, : cells * cell]
        .reshape(6, cells, cell, cells, cell)
        .max(axis=(2, 4))
    )
    coarse_dirs = cube_directions(cells).reshape(-1, 3)
    coarse_max = coarse.reshape(-1)
    centres = cube_directions(blocks)
    texel = 2.0 / size  # radians at a face's centre, the widest
    # A block's texels lie within half its diagonal of its centre, a coarse
    # cell's within half of its.
    reach = (TOPS_REACH_TEXELS + 0.75 * block + 0.75 * cell) * texel
    cos_reach = math.cos(min(reach, math.pi))
    out = np.empty((6, blocks, blocks), dtype=np.float32)
    for face in range(6):
        near = centres[face].reshape(-1, 3) @ coarse_dirs.T >= cos_reach
        out[face] = (
            np.where(near, coarse_max[None], -np.inf)
            .max(axis=1)
            .reshape(blocks, blocks)
        )
    return np.maximum(out, 0.0)


def march_ceiling(
    heights: np.ndarray,
    radius: float,
    sun: ArrayLike,
    dirs: ArrayLike,
    step_texels: float = STEP_TEXELS,
    top: float | None = None,
) -> np.ndarray:
    """The shadow ceiling at directions *dirs* by marching each toward the
    sun over *heights*: what ``shadow_bake.frag`` bakes on the GPU, on the
    CPU (its twin, for a few points). From the first counted step on (half
    a texel each, at a face's centre), a node *k* steps toward the sun
    along the great circle bounds the ceiling at its ground less the rise
    of the sun's ray from here (:func:`ray_gain`, from the ground's radius
    here); the march stops once no ground can rise above the ceiling found
    (or at the sub-solar point). Past the angle no ground sees the sun at,
    :data:`FULL_SHADOW`.

    :param heights: (6, n, n) the ground (:func:`height_cube`), metres
    :param radius: sea-level radius, metres
    :param sun: toward the sun, the planet's frame (any length)
    :param dirs: (..., 3) directions, the planet's frame (any length)
    :param top: the highest ground (None: found in *heights*, ~1 ms at
        1024: pass it for many calls)
    :returns: (..., 2) float64: ceiling and distance to the occluder,
        metres (:func:`ceiling_cube`'s)
    """
    size = heights.shape[1]
    sun = np.asarray(sun, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    dirs = np.asarray(dirs, dtype=np.float64)
    shape = dirs.shape[:-1]
    d = dirs.reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    top = max(float(heights.max() if top is None else top), 0.0)
    step, near, last, steps = march_limits(size, radius, top, step_texels)
    cos_z = np.clip(d @ sun, -1.0, 1.0)
    zenith = np.arccos(cos_z)
    toward = sun[None] - cos_z[:, None] * d
    length = np.linalg.norm(toward, axis=1)
    toward = toward / np.maximum(length, 1e-12)[:, None]
    lifted = radius + np.maximum(sample_cube(heights, d), 0.0)
    # Every step at once (a lookup for them all: the shader's early stop
    # only skips steps that cannot beat the ceiling found).
    angle = np.arange(near, steps) * step  # (k,)
    node = (
        np.cos(angle)[:, None, None] * d[None]
        + np.sin(angle)[:, None, None] * toward[None]
    )
    gain = ray_gain(zenith[None], angle[:, None], lifted[None])
    candidate = sample_cube(heights, node) - gain  # (k, n)
    valid = (angle[:, None] < zenith[None]) & (length > 1e-9)[None]
    candidate = np.where(valid, candidate, -np.inf)
    best = np.argmax(candidate, axis=0)  # the first of the highest, as the shader
    top_value = candidate[best, np.arange(len(d))]
    found = top_value > NO_CEILING
    ceiling = np.where(found, top_value, NO_CEILING)
    distance = np.where(found, angle[best] * radius, near * step * radius)
    ceiling = np.where(zenith >= last, FULL_SHADOW, ceiling)
    out = np.stack([ceiling, distance], axis=-1)
    return out.reshape(shape + (2,))


def global_reach(size: int, radius: float, step_texels: float = STEP_TEXELS) -> float:
    """:returns: the nearest occluder :func:`ceiling_cube` counts, metres
    (nearer ones are the local map's, :func:`local_ceiling`)"""
    return step_texels * (2.0 / size) * radius


def _bilinear(grid: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """*grid* (rows, columns, ...) at fractional (row, column) node
    coordinates, clamped at its edges."""
    rows, columns = grid.shape[:2]
    y = np.clip(y, 0.0, rows - 1.0)
    x = np.clip(x, 0.0, columns - 1.0)
    y0 = np.minimum(np.floor(y).astype(np.int64), rows - 2)
    x0 = np.minimum(np.floor(x).astype(np.int64), columns - 2)
    ty, tx = y - y0, x - x0
    if grid.ndim == 3:
        ty, tx = ty[..., None], tx[..., None]
    top = grid[y0, x0] * (1 - tx) + grid[y0, x0 + 1] * tx
    bottom = grid[y0 + 1, x0] * (1 - tx) + grid[y0 + 1, x0 + 1] * tx
    return top * (1 - ty) + bottom * ty


def chart_nodes(size: int, spacing: float) -> np.ndarray:
    """:returns: (size,) a local map's node coordinates along each chart
    axis, metres from its centre (node centres)"""
    return (np.arange(size) - 0.5 * (size - 1)) * spacing


def local_ceiling(
    heights: np.ndarray,
    chart: AzimuthalEquidistantChart,
    spacing: float,
    sun: ArrayLike,
    near: float,
    far: float,
) -> np.ndarray:
    """The local shadow map's bake: :func:`ceiling_cube`'s, over a square
    of ground around a chart's centre, counting only the occluders between
    *near* and *far* (the global map's reach, :func:`global_reach`): the
    two maps never count one occluder twice.

    The same pass over a grid aligned with the sun, over the window that
    covers the square (and *far* toward the sun beyond it), its nodes'
    ground read from *heights*; the result is read back at the chart's
    nodes. The ground beyond the square is unknown: near its sunward edges
    the map misses occluders (the shaders fade it out there). With the sun
    within the window's reach of the zenith the grid has no room: there is
    no local map (and its shadows are short).

    :param heights: (n, n) the ground at the chart's nodes
        (:func:`chart_nodes`), metres above sea level
    :param chart: the chart around the camera (planet frame)
    :param spacing: metres between nodes
    :param sun: toward the sun, the planet's frame
    :param near: the nearest occluder counted, metres
    :param far: the farthest (the global map's nearest), metres
    :returns: (n, n, 2) float32: ceiling (:data:`NO_CEILING` where nothing
        occludes) and distance, metres
    """
    size = heights.shape[0]
    radius = chart.radius
    sun = np.asarray(sun, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    out = np.empty((size, size, 2), dtype=np.float32)
    out[..., 0] = NO_CEILING
    out[..., 1] = far
    nodes = chart_nodes(size, spacing)
    x, y = np.meshgrid(nodes, nodes)
    dirs = chart.to_dirs(x, y).reshape(size, size, 3)
    zenith = np.arccos(np.clip(dirs @ sun, -1.0, 1.0))
    reach = (far + 2.0 * spacing) / radius
    if zenith.min() <= reach or zenith.min() >= 0.5 * math.pi + 0.2:
        return out  # the sun about overhead, or night over the whole square

    side = np.cross(sun, [1.0, 0.0, 0.0] if abs(sun[0]) < 0.9 else [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(sun, side)
    around = np.arctan2(dirs @ other, dirs @ side)
    centre = float(around[size // 2, size // 2])
    around = (around - centre + math.pi) % (2.0 * math.pi) - math.pi
    step = spacing / radius
    psi0 = float(zenith.min()) - reach
    rows = int(math.ceil((float(zenith.max()) - psi0) / step)) + 2
    middle = float(zenith[size // 2, size // 2])
    turn = step / math.sin(middle)  # a node's spacing across, at the centre
    phi0 = float(around.min()) - turn
    columns = int(math.ceil((float(around.max()) + turn - phi0) / turn)) + 1
    psi = psi0 + np.arange(rows) * step
    phi = centre + phi0 + np.arange(columns) * turn
    ring = np.cos(phi)[:, None] * side + np.sin(phi)[:, None] * other

    # The ground at the grid's nodes (none known outside the square: there
    # nothing occludes).
    ground = np.empty((rows, columns))
    half = 0.5 * (size - 1) * spacing
    for row, angle in enumerate(psi):
        node = math.cos(angle) * sun + math.sin(angle) * ring
        gx, gy = chart.from_dirs(node)
        inside = (np.abs(gx) <= half) & (np.abs(gy) <= half)
        value = _bilinear(heights, (gy + half) / spacing, (gx + half) / spacing)
        ground[row] = np.where(inside, value, NO_CEILING)

    # Every occluder between near and far, a row at a time.
    ceiling = np.full((rows, columns), NO_CEILING)
    distance = np.full((rows, columns), far)
    lifted = radius + np.maximum(ground, 0.0)
    for k in range(max(1, round(near / spacing)), int(math.ceil(far / spacing))):
        gain = ray_gain(psi[k:, None], k * step, lifted[k:])
        candidate = ground[:-k] - gain
        higher = candidate > ceiling[k:]
        ceiling[k:] = np.where(higher, candidate, ceiling[k:])
        distance[k:] = np.where(higher, k * spacing, distance[k:])

    # Back at the chart's nodes.
    gy = (zenith - psi0) / step
    gx = (around - phi0) / turn
    both = np.stack([np.maximum(ceiling, NO_CEILING), distance], axis=-1)
    out[...] = _bilinear(both, gy, gx)
    return out


def local_lit(
    chart: AzimuthalEquidistantChart,
    size: int,
    spacing: float,
    sun: ArrayLike,
    far: float,
) -> bool:
    """Whether a local map is made for this sun (:func:`local_ceiling`'s
    test): not with the sun about overhead somewhere in its square (no
    room toward it), nor with the whole square in the night."""
    sun = np.asarray(sun, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    nodes = chart_nodes(size, spacing)
    # Its nodes' zenith angles, on a coarser grid with its edges (smooth).
    grid = np.linspace(nodes[0], nodes[-1], 65)
    gx, gy = np.meshgrid(grid, grid)
    zenith = np.arccos(np.clip(chart.to_dirs(gx, gy).reshape(-1, 3) @ sun, -1.0, 1.0))
    reach = (far + 2.0 * spacing) / chart.radius
    return not (zenith.min() <= reach or zenith.min() >= 0.5 * math.pi + 0.2)


def march_local(
    heights: np.ndarray,
    chart: AzimuthalEquidistantChart,
    spacing: float,
    sun: ArrayLike,
    near: float,
    far: float,
    dirs: ArrayLike,
) -> np.ndarray:
    """The local map at directions *dirs* by marching each toward the sun
    over the chart's ground, the occluders between *near* and *far*: what
    ``shadow_bake_local.frag`` bakes on the GPU, on the CPU (its twin).
    :func:`local_ceiling`'s march, at each point directly (no grid aligned
    with the sun between): a node *k* spacings toward the sun bounds the
    ceiling at its ground less the rise of the ray from here; ground
    outside the chart's square is unknown (no occluder).

    :param heights: (n, n) the ground at the chart's nodes, metres
    :returns: (..., 2) float64: ceiling (:data:`NO_CEILING` where nothing
        occludes) and distance, metres
    """
    size = heights.shape[0]
    radius = chart.radius
    sun = np.asarray(sun, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    dirs = np.asarray(dirs, dtype=np.float64)
    shape = dirs.shape[:-1]
    d = dirs.reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    half = 0.5 * (size - 1) * spacing

    def ground(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = chart.from_dirs(points)
        inside = (np.abs(x) <= half) & (np.abs(y) <= half)
        value = _bilinear(heights, (y + half) / spacing, (x + half) / spacing)
        return value, inside

    cos_z = np.clip(d @ sun, -1.0, 1.0)
    zenith = np.arccos(cos_z)
    toward = sun[None] - cos_z[:, None] * d
    toward /= np.maximum(np.linalg.norm(toward, axis=1), 1e-12)[:, None]
    here, _ = ground(d)
    lifted = radius + np.maximum(here, 0.0)
    # Every step at once (one lookup).
    k = np.arange(max(1, round(near / spacing)), int(math.ceil(far / spacing)))
    if k.size == 0:
        # No step between near and far (a small planet: the global map's
        # reach under two spacings): nothing occludes, as the GPU's empty
        # march gives.
        out = np.stack([np.full(len(d), NO_CEILING), np.full(len(d), far)], axis=-1)
        return out.reshape(shape + (2,))
    angle = k * spacing / radius
    node = (
        np.cos(angle)[:, None, None] * d[None]
        + np.sin(angle)[:, None, None] * toward[None]
    )
    value, inside = ground(node.reshape(-1, 3))
    value, inside = value.reshape(node.shape[:2]), inside.reshape(node.shape[:2])
    gain = ray_gain(zenith[None], angle[:, None], lifted[None])
    candidate = np.where(inside, value - gain, -np.inf)
    best = np.argmax(candidate, axis=0)
    top_value = candidate[best, np.arange(len(d))]
    found = top_value > NO_CEILING
    ceiling = np.where(found, top_value, NO_CEILING)
    distance = np.where(found, k[best] * spacing, far)
    out = np.stack([ceiling, distance], axis=-1)
    return out.reshape(shape + (2,))


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
    """A body's mountain shadow (see the module): its heights sampled once
    in the worker pool, its ceiling baked for each sun, and swapped in when
    ready. The shaders read it through :meth:`bind` (shadows.glsl's
    ``terrainShadow``), the CPU through :meth:`visibility`.

    With a window the ceiling is baked on the GPU
    (:mod:`world_builder.terrain.shadow_bake`): a new sun's map is drawn
    with the next frame. Without one, in the pool (~2 s at 1024 in a
    worker), the last map staying until the new one is in.

    :ivar size: texels per face edge (0: no map)
    """

    def __init__(
        self,
        terrain: Any,
        executor: Any,
        size: int,
        sun_direction: ArrayLike,
        local_size: int = 0,
        local_spacing: float = 50.0,
        output: Any = None,
    ) -> None:
        """
        :param terrain: the body's :class:`~world_builder.terrain.terrain.Terrain`
            (its planet, radius, frame)
        :param executor: the world's :class:`ChunkExecutor`
        :param size: texels per face edge (``ShadowSettings.terrain_map_size``;
            0 = none)
        :param sun_direction: toward the sun, world frame
        :param local_size: the local map around the camera, nodes per side
            (``ShadowSettings.local_map_size``; 0 = none)
        :param local_spacing: its nodes' spacing, metres
        :param output: the window whose GSG bakes the maps (None: the pool)
        """
        self._terrain = terrain
        self._executor = executor
        self._output = output
        self._gpu = None  # its CeilingBake, at the size
        self._heights_texture = None
        self._top = 0.0  # the highest ground, once sampled
        self._tops_job = None  # local_tops, in the pool
        self._tops_texture = None
        # The GPU's map's texels marched again on the CPU, for its twin.
        self._texels: dict[tuple, tuple] = {}
        self._texels_for: tuple | None = None
        self._gpu_pending: np.ndarray | None = None  # the sun drawn next frame
        self._nodes: list[NodePath] = []
        self._drawn_sun: np.ndarray | None = None  # the map's shown
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
        #: The nearby ridges' map, around the camera
        self.local = LocalShadowMap(terrain, executor, output)
        self.set_sun(sun_direction)
        self.set_size(size)
        self.local.set_size(local_size, local_spacing)

    @property
    def loading(self) -> bool:
        """Whether the first global map is still being made."""
        return self.size > 0 and self._drawn_sun is None

    @property
    def current(self) -> bool:
        """Whether the map drawn is the one for the sun now (none wanted
        counts): a later sun is still baking otherwise."""
        if not self.local.current:
            return False
        if not self.size:
            return True
        return not sun_moved(self._drawn_sun, self._sun)

    def bind(self, node: NodePath) -> None:
        """Give a node's shaders this body's map (shadows.glsl's
        ``uTerrainShadow``, ``uTerrainShadowOn``: off until baked)."""
        self._nodes.append(node)
        node.set_shader_input("uTerrainShadow", self._texture)
        node.set_shader_input("uTerrainShadowOn", self._on)
        self.local.bind(node)

    def _set_texture(self, texture: Texture) -> None:
        """A new map texture (the GPU bake's, at a new size): rebound."""
        self._texture = texture
        self._nodes = [node for node in self._nodes if not node.is_empty()]
        for node in self._nodes:
            node.set_shader_input("uTerrainShadow", texture)

    def set_local(self, size: int, spacing: float) -> None:
        """The local map's size and spacing (0: none), made again."""
        self.local.set_size(size, spacing)

    def set_size(self, size: int) -> None:
        """A new resolution: the heights are sampled again (0: no map)."""
        if size == self.size:
            return
        self.size = int(size)
        self.cube = self._heights = self._bake_job = None
        self._heights_texture = self._gpu_pending = self._drawn_sun = None
        self._tops_job = self._tops_texture = None
        self._baking_sun = None
        self._on[0] = 0
        if self._gpu is not None:
            self._gpu.destroy()
            self._gpu = None
        if self._output is not None and self.size:
            from world_builder.terrain.shadow_bake import CeilingBake

            self._gpu = CeilingBake(self._output, self.size)
            self._set_texture(self._gpu.texture)
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

    def update(self, camera_world: ArrayLike | None = None) -> None:
        """Per frame: collect what the pool made, start the next bakes.

        :param camera_world: the camera, world frame: the local map follows
            it (None: no local map work)
        """
        self._update_global()
        if camera_world is not None:
            far = (
                global_reach(self.size, self._terrain.radius)
                if self.size
                else LOCAL_FAR
            )
            camera = self._terrain.to_planet(np.asarray(camera_world, float)[None])[0]
            self.local.update(camera, self._sun, far)

    def _update_global(self) -> None:
        if not self.size:
            return
        if self._heights is None:
            if not all(job.done() for job in self._heights_jobs):
                return
            self._heights = np.stack([job.result() for job in self._heights_jobs])
            self._top = float(self._heights.max())
            self._heights_jobs = []
        if self._gpu is not None:
            self._update_gpu()
            return
        if self._bake_job is not None:
            if not self._bake_job.done():
                return
            self.cube = self._bake_job.result()
            self.bake_seconds = time.perf_counter() - self._bake_started
            self._upload(self.cube)
            self._drawn_sun = self._baking_sun
            self._bake_job = None
        if sun_moved(self._baking_sun, self._sun):
            self._baking_sun = self._sun.copy()
            self._bake_started = time.perf_counter()
            self._bake_job = self._executor.submit_once(
                _bake, self._heights, self._terrain.radius, self._baking_sun
            )

    def _update_gpu(self) -> None:
        """The GPU's bake: what was asked last frame is drawn; a new sun is
        asked for."""
        from world_builder.terrain.shadow_bake import heights_texture, tops_texture

        if self._gpu_pending is not None:
            if not self._gpu.step():
                return  # a band more this frame
            self._drawn_sun = self._gpu_pending
            self._gpu_pending = None
            self.bake_seconds = time.perf_counter() - self._bake_started
            self._on[0] = 1
        if self._heights_texture is None:
            self._heights_texture = heights_texture(self._heights)
            self._tops_job = self._executor.submit_once(local_tops, self._heights)
        if self._tops_texture is None:
            if not self._tops_job.done():
                return
            self._tops_texture = tops_texture(self._tops_job.result())
        if sun_moved(self._drawn_sun, self._sun):
            self._bake_started = time.perf_counter()
            self._gpu.bake(
                self._heights_texture,
                self._tops_texture,
                self._sun,
                self._terrain.radius,
                self._top,
            )
            self._gpu_pending = self._sun.copy()

    def destroy(self) -> None:
        """Remove the GPU's bake buffers (with a window)."""
        if self._gpu is not None:
            self._gpu.destroy()
            self._gpu = None
        self.local.destroy()

    def _gpu_twin(self, points: np.ndarray) -> np.ndarray:
        """The GPU's map at points, as its sampler reads it: its texels
        around each (marched here once each, kept for the sun drawn),
        bilinear within a face (:func:`sample_cube`'s). A texel is ~1 km:
        a flying camera reaches a new one every few seconds."""
        drawn = tuple(self._drawn_sun)
        if self._texels_for != drawn or len(self._texels) > 4 * SHADE_CACHE_SIZE:
            self._texels, self._texels_for = {}, drawn
        size = self.size
        face, row, column = directions_to_texels(points, size)
        y = np.clip(row - 0.5, 0.0, size - 1.0)
        x = np.clip(column - 0.5, 0.0, size - 1.0)
        y0 = np.minimum(np.floor(y).astype(np.int64), size - 2)
        x0 = np.minimum(np.floor(x).astype(np.int64), size - 2)
        corners = [(y0 + dy, x0 + dx) for dy in (0, 1) for dx in (0, 1)]
        keys = [list(zip(face.tolist(), r.tolist(), c.tolist())) for r, c in corners]
        wanted = {key for corner in keys for key in corner}
        missing = sorted(wanted - set(self._texels))
        if missing:
            f, r, c = (np.array(v) for v in zip(*missing))
            marched = march_ceiling(
                self._heights,
                self._terrain.radius,
                self._drawn_sun,
                texel_directions(size, f, r, c),
                top=self._top,
            )
            for key, value in zip(missing, marched):
                self._texels[key] = (float(value[0]), float(value[1]))
        c00, c01, c10, c11 = (
            np.array([self._texels[key] for key in corner]) for corner in keys
        )
        ty, tx = (y - y0)[:, None], (x - x0)[:, None]
        top = c00 + (c01 - c00) * tx
        bottom = c10 + (c11 - c10) * tx
        return top + (bottom - top) * ty

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
        ``terrainShadow`` on the CPU (both maps as last baked).

        :param points_world: (..., 3) world frame, metres
        :param sun_radius: the sun's angular radius, radians
        :returns: (...) in [0, 1]; 1 before the first maps
        """
        points = np.asarray(points_world, dtype=np.float64)
        if self._drawn_sun is None and self.local.drawn is None:
            return np.ones(points.shape[:-1])
        # ~0.1 ms a map a call (numpy on a few points), every frame: a
        # point's shade is kept per SHADE_CELL, until either map changes.
        drawn = None if self._drawn_sun is None else tuple(self._drawn_sun)
        maps = (id(self.cube), drawn, id(self.local.drawn))
        if self._shades_for != maps or len(self._shades) > SHADE_CACHE_SIZE:
            self._shades_for, self._shades = maps, {}
        planet = self._terrain.to_planet(points.reshape(-1, 3))
        cells = [tuple(c) for c in np.round(planet / SHADE_CELL).astype(np.int64)]
        missing = [i for i, cell in enumerate(cells) if cell not in self._shades]
        if missing:
            at = planet[missing]
            shade = self.local.visibility(at, sun_radius)
            global_map = None
            if self.cube is not None:
                global_map = sample_cube(self.cube, at)
            elif self._drawn_sun is not None and self._heights is not None:
                global_map = self._gpu_twin(at)
            if global_map is not None:
                ceiling, distance = np.moveaxis(global_map, -1, 0)
                altitude = np.linalg.norm(at, axis=-1) - self._terrain.radius
                shade = np.minimum(
                    shade, terrain_visibility(ceiling, distance, altitude, sun_radius)
                )
            for i, value in zip(missing, np.atleast_1d(shade)):
                self._shades[cells[i]] = float(value)
        shades = np.array([self._shades[cell] for cell in cells])
        return shades.reshape(points.shape[:-1])


#: The local map's reach without a global map, metres.
LOCAL_FAR = 10e3
#: Above this altitude the local map is not made again (its detail is
#: under a pixel; the last one stays).
LOCAL_MAX_ALTITUDE = 20e3


def _local_heights(
    spec, chart_state: tuple, size: int, spacing: float, rows: tuple, ocean: bool
) -> np.ndarray:
    """Rows of the local map's ground, in a worker (~2.5 us a node)."""
    from world_builder.terrain.chunkgen import planet_for

    planet = planet_for(spec)
    chart = AzimuthalEquidistantChart(*chart_state)
    nodes = chart_nodes(size, spacing)
    x, y = np.meshgrid(nodes, nodes[rows[0] : rows[1]])
    heights = planet.sample_dirs(
        chart.to_dirs(x, y), min_wavelength=2.0 * spacing, apply_stamps=True
    )[0]
    if ocean:
        heights = np.maximum(heights, 0.0)
    return heights.reshape(x.shape).astype(np.float32)


class LocalShadowMap:
    """The nearby ridges' shadow (part E): :func:`local_ceiling` over a
    square chart around the camera, the occluders nearer than the global
    map's reach. Its ground is sampled when the camera has moved a quarter
    of the square from its centre (below :data:`LOCAL_MAX_ALTITUDE`), its
    ceiling baked for each sun and each ground, both in the pool; the last
    map stays until the next is in (the shadows where they overlap are the
    same: no pop). A :class:`ShadowMap` owns it.

    The shaders read it with the chart's frame (shadows.glsl's
    ``localShadow``), faded out toward its edges, where the ground beyond
    is unknown.

    With a window its ceiling is baked on the GPU
    (:class:`~world_builder.terrain.shadow_bake.LocalBake`, drawn with the
    next frame; :func:`march_local` its twin), its ground still sampled in
    the pool.
    """

    def __init__(self, terrain: Any, executor: Any, output: Any = None) -> None:
        self._terrain = terrain
        self._executor = executor
        self._output = output
        self._gpu = None  # its LocalBake, at the size
        self._gpu_pending: tuple | None = None  # drawn next frame
        self._heights_texture: tuple | None = None  # (ground id, texture)
        self._nodes: list[NodePath] = []
        # The GPU's map's nodes marched again on the CPU, for its twin.
        self._marched: dict[tuple, tuple] = {}
        self._marched_for: int | None = None
        self._texture = Texture("local_terrain_shadow")
        self._texture.setup_2d_texture(1, 1, Texture.T_float, Texture.F_rg32)
        self._texture.set_ram_image(bytes(8))
        self._on = PTA_int.empty_array(1)
        self._chart_inputs = PTA_LVecBase4f.empty_array(4)
        self.size = 0
        self.spacing = 50.0
        self.drawn: tuple | None = None  # (chart, (n, n, 2) map) as shown
        self._heights_jobs: list = []
        self._heights_chart: AzimuthalEquidistantChart | None = None  # sampled
        self._ground: tuple | None = None  # (chart, (n, n) heights), ready
        self._bake_job = None
        self._baked_for: tuple | None = None  # (ground id, sun) baked or baking
        self._far = 0.0  # the drawn map's reach

    @property
    def current(self) -> bool:
        """Whether nothing is being sampled or baked."""
        busy = self._heights_jobs or self._bake_job is not None
        return not self.size or not (busy or self._gpu_pending is not None)

    def bind(self, node: NodePath) -> None:
        """Give a node's shaders the local map (off until baked)."""
        self._nodes.append(node)
        node.set_shader_input("uLocalShadow", self._texture)
        node.set_shader_input("uLocalShadowOn", self._on)
        node.set_shader_input("uLocalChart", self._chart_inputs)

    def set_size(self, size: int, spacing: float) -> None:
        """Nodes per side and their spacing (0: none): made again."""
        if (size, spacing) == (self.size, self.spacing):
            return
        self.size, self.spacing = int(size), float(spacing)
        self._on[0] = 0
        self.drawn = self._ground = self._heights_chart = self._bake_job = None
        self._gpu_pending = self._heights_texture = None
        self._heights_jobs = []
        self._baked_for = None
        if self._gpu is not None:
            self._gpu.destroy()
            self._gpu = None
        if self._output is not None and self.size:
            from world_builder.terrain.shadow_bake import LocalBake

            self._gpu = LocalBake(self._output, self.size)
            self._texture = self._gpu.texture
            self._nodes = [node for node in self._nodes if not node.is_empty()]
            for node in self._nodes:
                node.set_shader_input("uLocalShadow", self._texture)

    def destroy(self) -> None:
        """Remove the GPU's bake buffer (with a window)."""
        if self._gpu is not None:
            self._gpu.destroy()
            self._gpu = None

    def update(self, camera: np.ndarray, sun: np.ndarray, far: float) -> None:
        """Per frame.

        :param camera: the camera, the planet's frame
        :param sun: toward the sun, the planet's frame (unit)
        :param far: the global map's reach (:func:`global_reach`)
        """
        if not self.size:
            return
        radius = self._terrain.radius
        extent = self.size * self.spacing
        rho = float(np.linalg.norm(camera))
        if not self._heights_jobs and rho - radius < LOCAL_MAX_ALTITUDE:
            moved = self._heights_chart is None
            if not moved:
                x, y = self._heights_chart.from_dirs(camera[None] / rho)
                moved = max(abs(x[0]), abs(y[0])) > 0.25 * extent
            if moved:
                self._sample(camera / rho)
        if self._heights_jobs and all(job.done() for job in self._heights_jobs):
            heights = np.concatenate([job.result() for job in self._heights_jobs])
            self._ground = (self._heights_chart, heights)
            self._heights_jobs = []
        if self._gpu is not None:
            self._update_gpu(sun, far)
            return
        if self._bake_job is not None and self._bake_job.done():
            self.drawn = self._bake_job.result()
            self._upload(*self.drawn)
            self._bake_job = None
        if self._bake_job is None and self._ground is not None:
            key = (id(self._ground), tuple(sun))
            if key != self._baked_for:
                self._baked_for = key
                chart, heights = self._ground
                near = 2.0 * self.spacing
                self._bake_job = self._executor.submit_once(
                    _local_bake,
                    heights,
                    (chart.lat_deg, chart.lon_deg, chart.radius),
                    self.spacing,
                    sun.copy(),
                    near,
                    far,
                )

    def _update_gpu(self, sun: np.ndarray, far: float) -> None:
        """The GPU's bake: what was asked last frame is drawn; a new ground
        or sun is asked for (none for a sun about overhead or a square all
        in the night: :func:`local_lit`)."""
        from world_builder.terrain.shadow_bake import local_heights_texture

        if self._gpu_pending is not None:
            if not self._gpu.step():
                return  # a band more this frame
            self.drawn = self._gpu_pending
            self._gpu_pending = None
            chart = self.drawn[0]
            self._set_chart(chart, self.size, self.drawn[5])
        if self._ground is None:
            return
        key = (id(self._ground), tuple(sun))
        if key == self._baked_for:
            return
        self._baked_for = key
        chart, heights = self._ground
        if not local_lit(chart, self.size, self.spacing, sun, far):
            self.drawn = None
            self._on[0] = 0
            return
        if self._heights_texture is None or self._heights_texture[0] != id(heights):
            self._heights_texture = (id(heights), local_heights_texture(heights))
        near = 2.0 * self.spacing
        texture = self._heights_texture[1]
        top = float(heights.max())
        self._gpu.bake(texture, chart, self.spacing, sun, near, far, top)
        # (chart, its ground, the sun, near, far, reach): march_local's
        self._gpu_pending = (chart, heights, sun.copy(), near, far, far)

    def _sample(self, up: np.ndarray) -> None:
        """Ask the pool for the ground of a new square centred under *up*."""
        lat, lon = latlon(up[None])
        chart = AzimuthalEquidistantChart(
            float(lat[0]), float(lon[0]), self._terrain.radius
        )
        spec = self._terrain.spec
        state = (chart.lat_deg, chart.lon_deg, chart.radius)
        bounds = np.linspace(0, self.size, 5).astype(int)
        self._heights_chart = chart
        self._heights_jobs = [
            self._executor.submit_once(
                _local_heights,
                spec,
                state,
                self.size,
                self.spacing,
                (int(a), int(b)),
                bool(spec.has_ocean),
            )
            for a, b in zip(bounds[:-1], bounds[1:])
        ]

    def _upload(self, chart: AzimuthalEquidistantChart, ceiling: np.ndarray) -> None:
        """The map as a float texture, and its chart's frame (the uniforms
        rewritten in place, with it)."""
        size = ceiling.shape[0]
        if self._texture.get_x_size() != size:
            self._texture.setup_2d_texture(size, size, Texture.T_float, Texture.F_rg32)
            self._texture.set_magfilter(SamplerState.FT_linear)
            # Mipmapped: toward its far side it is read minified (0.5 ms
            # without, at the ground poses: the texture cache).
            self._texture.set_minfilter(SamplerState.FT_linear_mipmap_linear)
            self._texture.set_wrap_u(SamplerState.WM_clamp)
            self._texture.set_wrap_v(SamplerState.WM_clamp)
        self._texture.set_ram_image(
            np.ascontiguousarray(ceiling, dtype=np.float32).tobytes()
        )
        # Its reach: no occluder is farther (once a map: the twin reads it).
        self._set_chart(chart, size, float(np.max(ceiling[..., 1])))

    def _set_chart(
        self, chart: AzimuthalEquidistantChart, size: int, far: float
    ) -> None:
        """The drawn map's chart frame (the uniforms rewritten in place)
        and its reach, and the map on."""
        frame = chart.frame
        half = 0.5 * (size - 1) * self.spacing
        self._far = far
        self._chart_inputs[0] = LVecBase4f(*frame.up, chart.radius)
        self._chart_inputs[1] = LVecBase4f(*frame.east, self.spacing)
        self._chart_inputs[2] = LVecBase4f(*frame.north, float(size))
        # Faded out over the band within the global reach of the edge.
        self._chart_inputs[3] = LVecBase4f(half - 2.0 * far, half - far, 0.0, 0.0)
        self._on[0] = 1

    def _gpu_twin(self, gy: np.ndarray, gx: np.ndarray, size: int) -> np.ndarray:
        """The GPU's map at fractional node coordinates, as its sampler
        reads it: the nodes around each (marched here once each, kept for
        the map drawn), bilinear (:func:`_bilinear`'s clamping)."""
        chart, heights, sun, near, far, _ = self.drawn
        full = len(self._marched) > 4 * SHADE_CACHE_SIZE
        if self._marched_for != id(self.drawn) or full:
            self._marched, self._marched_for = {}, id(self.drawn)
        gy = np.clip(gy, 0.0, size - 1.0)
        gx = np.clip(gx, 0.0, size - 1.0)
        y0 = np.minimum(np.floor(gy).astype(np.int64), size - 2)
        x0 = np.minimum(np.floor(gx).astype(np.int64), size - 2)
        corners = [(y0 + dy, x0 + dx) for dy in (0, 1) for dx in (0, 1)]
        keys = [list(zip(r.tolist(), c.tolist())) for r, c in corners]
        wanted = {key for corner in keys for key in corner}
        missing = sorted(wanted - set(self._marched))
        if missing:
            rows, columns = (np.array(v) for v in zip(*missing))
            nodes = chart_nodes(size, self.spacing)
            dirs = chart.to_dirs(nodes[columns], nodes[rows]).reshape(-1, 3)
            marched = march_local(heights, chart, self.spacing, sun, near, far, dirs)
            for key, value in zip(missing, marched):
                self._marched[key] = (float(value[0]), float(value[1]))
        c00, c01, c10, c11 = (
            np.array([self._marched[key] for key in corner]) for corner in keys
        )
        ty, tx = (gy - y0)[:, None], (gx - x0)[:, None]
        top = c00 + (c01 - c00) * tx
        bottom = c10 + (c11 - c10) * tx
        return top + (bottom - top) * ty

    def visibility(self, planet: np.ndarray, sun_radius: float) -> np.ndarray:
        """``localShadow`` on the CPU, at (m, 3) planet-frame points: 1
        outside the map (and before it), faded toward its edges."""
        planet = np.asarray(planet, dtype=np.float64).reshape(-1, 3)
        if self.drawn is None:
            return np.ones(len(planet))
        chart = self.drawn[0]
        rho = np.linalg.norm(planet, axis=1)
        x, y = chart.from_dirs(planet / rho[:, None])
        size = self.drawn[1].shape[0]  # its ground's, or the pool's map's
        half = 0.5 * (size - 1) * self.spacing
        far = self._far
        edge = np.maximum(np.abs(x), np.abs(y))
        t = np.clip((edge - (half - 2.0 * far)) / max(far, 1.0), 0.0, 1.0)
        keep = 1.0 - t * t * (3.0 - 2.0 * t)
        if len(self.drawn) == 2:  # the pool's map
            ceiling = self.drawn[1]
            value = _bilinear(
                ceiling, (y + half) / self.spacing, (x + half) / self.spacing
            )
        else:  # the GPU's: its nodes marched again here
            value = self._gpu_twin(
                (y + half) / self.spacing, (x + half) / self.spacing, size
            )
        seen = terrain_visibility(
            value[:, 0], value[:, 1], rho - chart.radius, sun_radius
        )
        return np.where(keep > 0.0, 1.0 - keep * (1.0 - seen), 1.0)


def _local_bake(heights, chart_state, spacing, sun, near, far) -> tuple:
    """:func:`local_ceiling`, in a worker (~0.9 s at 1024), returned with
    its chart (what the map shows)."""
    chart = AzimuthalEquidistantChart(*chart_state)
    return chart, local_ceiling(heights, chart, spacing, sun, near, far)
