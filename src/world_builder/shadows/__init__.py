"""
Shadows: what hides the sun from a point, for every drawn receiver.

Each shader that lights something by the sun multiplies its sunlight by a
product of independent visibility terms, all from ``shaders/shadows.glsl``.
This module feeds that include its uniforms, shared by every node bound, and
evaluates the same terms on the CPU (for Panda3D's lights and the sea's near
look, which are lit there).

The first term is the **eclipse**: spherical bodies (planets, moons) hiding
some or all of the sun's disc. The sun is a disc of
:data:`SUN_ANGULAR_RADIUS` (the sky draws the same disc), each body a disc
of its own angular radius as seen from the point, and the visibility is the
share of the sun's disc left uncovered: a soft penumbra, and an umbra where
the body looks wider than the sun. A body never eclipses itself (each
receiver names its own body, ``uShadowSelf``); a planet's own shadow is its
air's (``atmoSunlight``) or its day/night line.

The sky's light dims with the sunlight that makes it, down to
:data:`ECLIPSE_SKY_FLOOR` (the air outside the shadow still glows).

Most of the time no body's shadow is near another: :meth:`Shadows.set_sun`
works out, per body, which others' penumbra cones can reach what it holds
(its air, its mountains: its *reach*), and the shaders skip a body whose
list is empty, before any per-point work.

Usage::

    shadows = Shadows(sun_angular_radius)
    planet = shadows.add(planet_radius, planet_position, reach=air_top)
    moon = shadows.add(moon_radius, moon_position, reach=highest_ground)
    shadows.set_sun(sun_direction)           # and whenever it moves
    shadows.set_atmosphere(0, planet)        # the planet owns atmosphere 0
    shadows.bind(planet_terrain.root, planet)
    shadows.bind(sky.node)
    # every frame, camera in the world frame (float64):
    shadows.update(camera_world)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import LVecBase4f, NodePath, PTA_float, PTA_int, PTA_LVecBase4f

from world_builder.atmosphere import MAX_ATMOSPHERES

SHADER_PATH = Path(__file__).parent / "shaders"
#: Bodies one shader can take (must match MAX_SHADOW_BODIES in the GLSL).
MAX_SHADOW_BODIES = 8
#: The sun's angular radius, radians: as the sky draws it (about twice the
#: real one: a disc that reads at any resolution), and every penumbra.
SUN_ANGULAR_RADIUS = math.radians(0.55)
#: The sky's light left in a total eclipse, as a share of its light by day
#: (shadows.glsl's ECLIPSE_SKY_FLOOR).
ECLIPSE_SKY_FLOOR = 0.15


class Shadows:
    """The world's sun occluders, as shader inputs shared by every node bound.

    The inputs are uniform arrays (``MAX_SHADOW_BODIES`` entries) that
    :meth:`update` and :meth:`set_sun` rewrite in place, so every bound node
    sees each frame's values without being touched again (as
    :class:`~world_builder.atmosphere.Atmospheres`).
    """

    def __init__(self, sun_angular_radius: float = SUN_ANGULAR_RADIUS) -> None:
        """
        :param sun_angular_radius: the sun's angular radius, radians
        """
        self._bodies: list[list] = []  # [radius, position, reach]
        self._sun = np.array([0.0, 0.0, 1.0])
        self._count = PTA_int.empty_array(1)
        self._sun_angular_radius = float(sun_angular_radius)
        self._sun_radius = PTA_float.empty_array(1)
        self._sun_radius[0] = self._sun_angular_radius
        self._spheres = PTA_LVecBase4f.empty_array(MAX_SHADOW_BODIES)
        # Per body, a bit per body whose penumbra can reach it.
        self._casters = PTA_int.empty_array(MAX_SHADOW_BODIES)
        self._atmosphere_bodies = PTA_int.empty_array(MAX_ATMOSPHERES)
        for index in range(MAX_ATMOSPHERES):
            self._atmosphere_bodies[index] = -1

    @property
    def sun_angular_radius(self) -> float:
        """The sun's angular radius, radians."""
        return self._sun_angular_radius

    def add(
        self,
        radius: float,
        position: ArrayLike = (0, 0, 0),
        reach: float | None = None,
    ) -> int:
        """Add a spherical body that can hide the sun.

        :param radius: its radius, metres (sea level: its mountains are not
            in the disc)
        :param position: its centre in the world frame (float64)
        :param reach: from its centre, the farthest of what is drawn as its
            own (its air's top, its highest ground), metres: the shaders
            skip the eclipses that cannot reach that far. None = *radius*
        :returns: its index (a receiver's own body, :meth:`bind`)
        :raises ValueError: past :data:`MAX_SHADOW_BODIES`
        """
        if len(self._bodies) >= MAX_SHADOW_BODIES:
            raise ValueError(f"at most {MAX_SHADOW_BODIES} shadow bodies")
        reach = float(radius if reach is None else max(reach, radius))
        self._bodies.append([float(radius), np.asarray(position, float), reach])
        self._count[0] = len(self._bodies)
        self._cull()
        return len(self._bodies) - 1

    def set_position(self, index: int, position: ArrayLike) -> None:
        """Move a body: its centre in the world frame, float64. The shaders
        see it at the next :meth:`update`."""
        self._bodies[index][1] = np.asarray(position, float)
        self._cull()

    def set_sun(self, direction: ArrayLike) -> None:
        """Toward the sun, world frame (any length): which bodies' shadows
        can reach which (the shaders' ``uShadowCasters``)."""
        sun = np.asarray(direction, dtype=np.float64)
        sun = sun / np.linalg.norm(sun)
        if not np.array_equal(sun, self._sun):  # a game may set it every frame
            self._sun = sun
            self._cull()

    def casters(self, body: int) -> list[int]:
        """:returns: the bodies whose penumbra can reach *body*'s reach, for
        the sun last set"""
        mask = self._casters[body]
        return [k for k in range(len(self._bodies)) if mask & (1 << k)]

    def _cull(self) -> None:
        """Per receiving body, the bodies whose penumbra cone meets its
        reach. The cone holds every point a body hides any of the sun from:
        its apex R / sin(a) toward the sun from the body's centre, its
        half-angle the sun's radius a. A sphere meets it when its centre is
        nearer the cone than its radius (the apex, for a centre behind it),
        unless it lies wholly more than R sunward of the body's centre (the
        cone's sunward part, where nothing is hidden)."""
        a = self._sun_angular_radius
        for index, (_, centre, reach) in enumerate(self._bodies):
            mask = 0
            for k, (radius, position, _) in enumerate(self._bodies):
                if k == index or (centre - position) @ self._sun - reach > radius:
                    continue
                apex = position + self._sun * (radius / math.sin(a))
                offset = centre - apex
                behind = float(offset @ -self._sun)
                aside = float(np.linalg.norm(offset + self._sun * behind))
                if behind * math.cos(a) + aside * math.sin(a) < 0.0:
                    gap = float(np.linalg.norm(offset))  # nearest the apex
                else:
                    gap = aside * math.cos(a) - behind * math.sin(a)
                if gap < reach * 1.001 + 1.0:
                    mask |= 1 << k
            self._casters[index] = mask

    def set_atmosphere(self, atmosphere: int, body: int) -> None:
        """Say which body owns an atmosphere: its air is not eclipsed by
        its own planet (that is the air's own shadow, atmosphere.glsl).

        :param atmosphere: the atmosphere's index in the world's
            :class:`~world_builder.atmosphere.Atmospheres`
        :param body: the body's index here
        """
        self._atmosphere_bodies[atmosphere] = int(body)

    def __len__(self) -> int:
        return len(self._bodies)

    def bind(self, node: NodePath, body: int | None = None) -> None:
        """Give a node (and its subtree) the shadow uniforms.

        :param body: the node's own body (it does not eclipse itself), None
            = none (the sky, an object in space)
        """
        node.set_shader_input("uShadowCount", self._count)
        node.set_shader_input("uShadowBodies", self._spheres)
        node.set_shader_input("uShadowCasters", self._casters)
        node.set_shader_input("uSunAngularRadius", self._sun_radius)
        node.set_shader_input("uAtmoBody", self._atmosphere_bodies)
        node.set_shader_input("uShadowSelf", -1 if body is None else int(body))

    def update(self, camera_world: ArrayLike) -> None:
        """Per frame: each body's centre relative to the camera (float64
        maths, the render frame's axes).

        :param camera_world: the camera in the world frame, metres
        """
        camera = np.asarray(camera_world, dtype=np.float64)
        for index, (radius, position, _) in enumerate(self._bodies):
            self._spheres[index] = LVecBase4f(*(position - camera), radius)

    def eclipse(
        self,
        points_world: ArrayLike,
        sun_direction: ArrayLike,
        exclude: int | None = None,
    ) -> np.ndarray:
        """The share of the sun's disc every body leaves visible from each
        point: shadows.glsl's eclipseVisibility on the CPU, in float64.

        :param points_world: (..., 3) points in the world frame, metres
        :param sun_direction: toward the sun, world frame (any length)
        :param exclude: the points' own body (never in the way), None = none
        :returns: (...) visibilities in [0, 1]: 1 in full sun, 0 in an umbra
        """
        points = np.asarray(points_world, dtype=np.float64)
        sun = np.asarray(sun_direction, dtype=np.float64)
        sun = sun / np.linalg.norm(sun)
        visible = np.ones(points.shape[:-1])
        for index, (radius, position, _) in enumerate(self._bodies):
            if index == exclude:
                continue
            visible *= 1.0 - _body_coverage(
                position - points, radius, sun, self.sun_angular_radius
            )
        return visible


def _body_coverage(
    centre: np.ndarray, radius: float, sun: np.ndarray, sun_radius: float
) -> np.ndarray | float:
    """The share of the sun's disc a sphere covers, seen from points it is
    at *centre* from ((..., 3), world frame); 0.0 (a float) when it is
    clear of the sun from every point."""
    distance = np.linalg.norm(centre, axis=-1)
    along = centre @ sun
    across = np.linalg.norm(centre - along[..., None] * sun, axis=-1)
    # shadows.glsl's exact reject, before any inverse trigonometry: past
    # the sum of the radii, from their tangents.
    with np.errstate(divide="ignore", invalid="ignore"):
        body_tan = radius / np.sqrt(distance * distance - radius * radius)
    sun_tan = math.tan(sun_radius)
    sum_cos = 1.0 - body_tan * sun_tan
    clear = (sum_cos > 0.0) & (across * sum_cos >= along * (body_tan + sun_tan))
    if np.all(clear):
        return 0.0
    separation = np.arctan2(across, along)
    body = np.arcsin(np.minimum(radius / np.maximum(distance, 1e-9), 1.0))
    coverage = disc_coverage(sun_radius, body, separation)
    return np.where(distance <= radius, 1.0, coverage)  # inside it: no sun


def disc_coverage(sun: ArrayLike, body: ArrayLike, separation: ArrayLike) -> np.ndarray:
    """The share of a uniform disc (the sun) another disc covers: their
    overlap's area over the sun's (shadows.glsl's discCoverage). Angles are
    taken as flat (small) lengths.

    :param sun: the sun's angular radius, radians
    :param body: the occluder's angular radius, radians
    :param separation: the angle between their centres, radians
    :returns: shares in [0, 1], broadcast
    """
    a, b, s = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (sun, body, separation))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_a = (s * s + a * a - b * b) / (2.0 * s * a)
        cos_b = (s * s + b * b - a * a) / (2.0 * s * b)
        lens = (
            a * a * np.arccos(np.clip(cos_a, -1.0, 1.0))
            + b * b * np.arccos(np.clip(cos_b, -1.0, 1.0))
            - 0.5
            * np.sqrt(
                np.maximum((-s + a + b) * (s + a - b) * (s - a + b) * (s + a + b), 0.0)
            )
        ) / (math.pi * a * a)
        inside = np.where(b >= a, 1.0, (b * b) / (a * a))
    coverage = np.where(s <= np.abs(a - b), inside, np.clip(lens, 0.0, 1.0))
    return np.where(s >= a + b, 0.0, coverage)


_DEFAULTS: dict | None = None


def bind_defaults(node: NodePath) -> None:
    """Give a node "nothing casts a shadow" inputs, so shaders including
    shadows.glsl (every one including atmosphere.glsl) run unchanged until
    :meth:`Shadows.bind` overrides them (a GLSL uniform the shader reads must
    always be set)."""
    global _DEFAULTS
    if _DEFAULTS is None:
        count = PTA_int.empty_array(1)
        count[0] = 0
        spheres = PTA_LVecBase4f.empty_array(MAX_SHADOW_BODIES)
        casters = PTA_int.empty_array(MAX_SHADOW_BODIES)
        for index in range(MAX_SHADOW_BODIES):
            spheres[index] = LVecBase4f(0.0, 0.0, 0.0, 1.0)
            casters[index] = 0
        sun_radius = PTA_float.empty_array(1)
        sun_radius[0] = SUN_ANGULAR_RADIUS
        owners = PTA_int.empty_array(MAX_ATMOSPHERES)
        for index in range(MAX_ATMOSPHERES):
            owners[index] = -1
        _DEFAULTS = {
            "uShadowCount": count,
            "uShadowBodies": spheres,
            "uShadowCasters": casters,
            "uSunAngularRadius": sun_radius,
            "uAtmoBody": owners,
            "uShadowSelf": -1,
        }
    for name, value in _DEFAULTS.items():
        node.set_shader_input(name, value)
