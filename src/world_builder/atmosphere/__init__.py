"""
Atmospheres: single scattering of sunlight by air (Rayleigh) and haze (Mie).

Every shader that draws something seen through air includes
``shaders/atmosphere.glsl`` and calls ``applyAtmospheres`` on its linear
colour; :class:`Atmospheres` feeds it the uniforms of every atmosphere in the
world, once per frame, through arrays shared by all those shaders. A body seen
through another's air (a moon from the ground) is dimmed and tinted by it.

The model, per view ray: march ``ATMO_STEPS`` samples through each atmosphere
crossed; at each, the optical depth toward the sun is analytic (Chapman
function, erfcx form, ~1.5 % from a numerical integral), and only the lit
share of the sample's stretch gets sun (the planet's shadow is found
analytically along the ray, so its edge moves smoothly). That gives the sky
colour, sunsets, the limb seen from space, aerial perspective and the
reddened sunlight on the ground, with no precomputation. Light scattered more
than once is only a flat share of the sunlight that fades through twilight
(:attr:`AtmosphereParams.multiple_scattering`): see the docs for the
precomputed alternative (Bruneton). :attr:`AtmosphereParams.sunset_radius`
lengthens grazing sunlight's path, so a small planet's sunsets are as red as
Earth's.

The CPU side (:func:`depth_toward`, :meth:`Atmospheres.transmittance_to_sun`,
:meth:`Atmospheres.light_at`) evaluates the same formulas, to light what is
drawn without these shaders (Panda3D lights, the ocean).

Defaults are Earth's air (Bruneton 2017's constants) over a 60 km shell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    LVecBase4f,
    NodePath,
    PTA_int,
    PTA_LVecBase4f,
)

SHADER_PATH = Path(__file__).parent / "shaders"
#: Atmospheres one shader can take (must match MAX_ATMOSPHERES in the GLSL).
MAX_ATMOSPHERES = 2


@dataclass(frozen=True)
class AtmosphereParams:
    """One planet's air. Lengths in metres, coefficients per metre at the ground.

    :ivar height: top of the atmosphere above the ground radius
    :ivar rayleigh: Rayleigh scattering (r, g, b): blue sky, red sunsets
    :ivar rayleigh_height: its scale height
    :ivar mie: Mie (aerosol) scattering: haze, the glow around the sun
    :ivar mie_extinction: Mie extinction (scattering plus absorption)
    :ivar mie_height: its scale height
    :ivar mie_g: Mie anisotropy (forward scattering, 0-1)
    :ivar sun_intensity: radiance scale of the scattered sunlight, in the
        renderer's linear units (the terrain's sun is ~2.6)
    :ivar multiple_scattering: light scattered more than once, as a share of
        direct sunlight, reaching air the sun does not (the planet's shadow
        at dusk is blue-grey, not black); it fades out through twilight
    :ivar sunset_radius: sunlight's slant through the air is taken as over a
        planet of this radius, if larger than the real one: as red near the
        terminator as there (a 600 km planet's grazing sunlight crosses ~11
        air masses, Earth's ~38), the same overhead. None = physical
    """

    height: float = 60e3
    rayleigh: tuple[float, float, float] = (5.802e-6, 13.558e-6, 33.1e-6)
    rayleigh_height: float = 8000.0
    mie: float = 3.996e-6
    mie_extinction: float = 4.44e-6
    mie_height: float = 1200.0
    mie_g: float = 0.8
    sun_intensity: float = 22.0
    multiple_scattering: float = 0.12
    sunset_radius: float | None = 6371e3


#: The vec4 uniform arrays atmosphere.glsl reads, MAX_ATMOSPHERES each.
_ARRAYS = ("uAtmoCam", "uAtmoShape", "uAtmoScatter", "uAtmoMie", "uAtmoExtra")


class Atmospheres:
    """The atmospheres of a world, as shader inputs shared by every node bound.

    The inputs are uniform arrays (``MAX_ATMOSPHERES`` entries each) that
    :meth:`update` rewrites in place, so every bound node sees each frame's
    values without being touched again.

    Usage::

        atmospheres = Atmospheres()
        index = atmospheres.add(planet_radius, AtmosphereParams(), position)
        atmospheres.bind(terrain.root)        # and the sky, clouds...
        # every frame, camera in the world frame (float64):
        atmospheres.update(camera_world)
    """

    def __init__(self) -> None:
        self._bodies: list[list] = []  # [radius, params, position]
        self._count = PTA_int.empty_array(1)
        self._arrays = {
            name: PTA_LVecBase4f.empty_array(MAX_ATMOSPHERES) for name in _ARRAYS
        }

    def add(
        self, radius: float, params: AtmosphereParams, position: ArrayLike = (0, 0, 0)
    ) -> int:
        """Add a planet's atmosphere.

        :param radius: the planet's ground (sea-level) radius, metres
        :param params: the air
        :param position: the planet centre in the world frame (float64)
        :returns: its index (the GLSL's atmosphere index)
        :raises ValueError: past :data:`MAX_ATMOSPHERES`
        """
        if len(self._bodies) >= MAX_ATMOSPHERES:
            raise ValueError(f"at most {MAX_ATMOSPHERES} atmospheres")
        self._bodies.append([float(radius), params, np.asarray(position, float)])
        index = len(self._bodies) - 1
        p = params
        self._arrays["uAtmoShape"][index] = LVecBase4f(
            radius, radius + p.height, p.rayleigh_height, p.mie_height
        )
        self._arrays["uAtmoScatter"][index] = LVecBase4f(*p.rayleigh, p.mie)
        self._arrays["uAtmoExtra"][index] = LVecBase4f(
            p.multiple_scattering, _curve(radius, p), 0.0, 0.0
        )
        self._count[0] = len(self._bodies)
        return index

    def set_position(self, index: int, position: ArrayLike) -> None:
        """Move a planet (orbits): its centre in the world frame, float64."""
        self._bodies[index][2] = np.asarray(position, float)

    def params(self, index: int) -> AtmosphereParams:
        """:returns: the air of atmosphere *index*"""
        return self._bodies[index][1]

    def __len__(self) -> int:
        return len(self._bodies)

    def bind(self, node: NodePath) -> None:
        """Give a node (and its subtree) the atmosphere uniforms."""
        node.set_shader_input("uAtmoCount", self._count)
        for name, array in self._arrays.items():
            node.set_shader_input(name, array)

    def update(self, camera_world: ArrayLike) -> None:
        """Per frame: the camera's place in each atmosphere (float64 maths).

        :param camera_world: the camera in the world frame, metres
        """
        camera = np.asarray(camera_world, dtype=np.float64)
        for index, (radius, params, position) in enumerate(self._bodies):
            offset = camera - position
            rho2 = float(offset @ offset)
            top = radius + params.height
            self._arrays["uAtmoCam"][index] = LVecBase4f(*offset, (rho2 - top * top))
            self._arrays["uAtmoMie"][index] = LVecBase4f(
                params.mie_extinction,
                params.mie_g,
                params.sun_intensity,
                rho2 - radius * radius,
            )

    def transmittance_to_sun(
        self, index: int, point_world: ArrayLike, sun_direction: ArrayLike
    ) -> np.ndarray:
        """Sunlight reaching a point through atmosphere *index*: the GLSL's
        atmoSunlight on the CPU (the same formula, :func:`depth_toward`, with
        the same ``sunset_radius`` scale).

        :param point_world: the point in the world frame, metres
        :param sun_direction: toward the sun, world frame (any length)
        :returns: (3,) factors in [0, 1]: 1 above the air, 0 in the planet's
            shadow
        """
        radius, params, position = self._bodies[index]
        p = np.asarray(point_world, float) - position
        d = np.asarray(sun_direction, float)
        d = d / np.linalg.norm(d)
        r = max(float(np.linalg.norm(p)), radius)
        if r >= radius + params.height and float(p @ d) >= 0.0:
            return np.ones(3)  # above the air, the sun not behind it
        cos_sun = float(p @ d) / max(float(np.linalg.norm(p)), 1e-9)
        curve = _curve(radius, params)
        depth_r = depth_toward(r, cos_sun, radius, params.rayleigh_height, curve)
        depth_m = depth_toward(r, cos_sun, radius, params.mie_height, curve)
        if depth_r >= 1e9:
            return np.zeros(3)  # the planet is in the way
        return np.exp(
            -(np.asarray(params.rayleigh) * depth_r + params.mie_extinction * depth_m)
        )

    def light_at(
        self,
        index: int,
        point_world: ArrayLike,
        sun_direction: ArrayLike,
        sun_color: ArrayLike,
        sky_light: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The light at a point in atmosphere *index*, as the terrain's
        shader lights its ground: for lighting anything else there (Panda3D
        lights, the ocean...).

        :param point_world: the point in the world frame, metres
        :param sun_direction: toward the sun, world frame
        :param sun_color: sunlight above the air, linear RGB
        :param sky_light: the sky's light by day, linear RGB
        :returns: (direct, ambient): the sunlight through the air (0 in the
            planet's shadow), and the sky light, fading through twilight
            (:func:`twilight`)
        """
        radius, _, position = self._bodies[index]
        p = np.asarray(point_world, float) - position
        d = np.asarray(sun_direction, float)
        d = d / np.linalg.norm(d)
        direct = np.asarray(sun_color, float) * self.transmittance_to_sun(
            index, point_world, d
        )
        cos_sun = float(p @ d) / max(float(np.linalg.norm(p)), 1e-9)
        return direct, np.asarray(sky_light, float) * twilight(cos_sun)


def twilight(cos_sun: float) -> float:
    """How much of the sky's light is left with the sun at *cos_sun* from
    the vertical: terrain.frag's ``smoothstep(-0.2, 0.2, dot(up, L))``."""
    t = min(max((cos_sun + 0.2) / 0.4, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def in_shadow(point: ArrayLike, sun_direction: ArrayLike, radius: float) -> bool:
    """Whether a body of *radius* (centred at the origin) hides the sun from
    *point*: for airless bodies, which :meth:`Atmospheres.light_at` does not
    cover. The shadow is the body's cylinder away from the sun (the sun is
    taken as a point at infinity: no penumbra)."""
    p = np.asarray(point, float)
    d = np.asarray(sun_direction, float)
    d = d / np.linalg.norm(d)
    along = float(p @ d)
    if along >= 0.0:
        return False
    return float(p @ p) - along * along < radius * radius


def _curve(radius: float, params: AtmosphereParams) -> float:
    """atmosphere.glsl's slant curvature scale (uAtmoExtra.y)."""
    if params.sunset_radius is None:
        return 1.0
    return max(1.0, params.sunset_radius / radius)


def _chapman(x: float, cos_chi: float) -> float:
    y = math.sqrt(0.5 * x) * cos_chi
    erfcx = 1.0 / (1.1635781 * y + math.sqrt(0.3707296 * y * y + 1.0))
    return math.sqrt(1.5707963 * x) * erfcx


def depth_toward(
    r: float, cos_chi: float, ground: float, height: float, curve: float = 1.0
) -> float:
    """atmosphere.glsl's atmoDepthToward: optical depth (density-weighted
    metres) from radius *r* toward a direction at *cos_chi* from the vertical,
    through a layer of scale height *height* over a ground of radius
    *ground*; 1e9 if the planet is in the way.

    *curve* (>= 1) takes the slant through the air as over a planet that many
    times larger (:attr:`AtmosphereParams.sunset_radius`): the same overhead,
    longer near the horizon.
    """
    density = math.exp(-(r - ground) / height)
    k = curve / height
    if cos_chi >= 0.0:
        return height * density * _chapman(r * k, cos_chi)
    tangent = r * math.sqrt(max(0.0, 1.0 - cos_chi * cos_chi))
    if tangent < ground:
        return 1e9
    return height * (
        2.0 * math.exp(-(tangent - ground) / height) * _chapman(tangent * k, 0.0)
        - density * _chapman(r * k, -cos_chi)
    )


_DEFAULTS: dict | None = None


def bind_defaults(node: NodePath) -> None:
    """Give a node "no atmosphere" inputs, so shaders including
    atmosphere.glsl run unchanged until :meth:`Atmospheres.bind` overrides
    them (a GLSL uniform the shader reads must always be set)."""
    global _DEFAULTS
    if _DEFAULTS is None:
        count = PTA_int.empty_array(1)
        count[0] = 0
        _DEFAULTS = {"uAtmoCount": count}
        for name in _ARRAYS:
            array = PTA_LVecBase4f.empty_array(MAX_ATMOSPHERES)
            for i in range(MAX_ATMOSPHERES):
                array[i] = LVecBase4f(0.0, 0.0, 1.0, 1.0)
            _DEFAULTS[name] = array
    for name, value in _DEFAULTS.items():
        node.set_shader_input(name, value)
