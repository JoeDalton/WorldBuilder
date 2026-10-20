"""
Atmospheres: single scattering of sunlight by air (Rayleigh) and haze (Mie).

Every shader that draws something seen through air includes
``shaders/atmosphere.glsl`` and calls ``applyAtmospheres`` on its linear
colour; :class:`Atmospheres` feeds it the uniforms of every atmosphere in the
world, once per frame, through arrays shared by all those shaders. A body seen
through another's air (a moon from the ground) is dimmed and tinted by it.

The model, per view ray: march ``ATMO_STEPS`` samples through each atmosphere
crossed; at each, the optical depth toward the sun is analytic (Chapman
function, erfcx form, ~1.5 % from a numerical integral), and samples in the
planet's shadow get no sun. That gives the sky colour, sunsets, the limb seen
from space, aerial perspective and the reddened sunlight on the ground, with
no precomputation. What it leaves out is multiple scattering (twilight comes
out a little dark): see the docs for the precomputed alternative (Bruneton).

Defaults are Earth's air (Bruneton 2017's constants) over a 60 km shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    """

    height: float = 60e3
    rayleigh: tuple = (5.802e-6, 13.558e-6, 33.1e-6)
    rayleigh_height: float = 8000.0
    mie: float = 3.996e-6
    mie_extinction: float = 4.44e-6
    mie_height: float = 1200.0
    mie_g: float = 0.8
    sun_intensity: float = 22.0
    multiple_scattering: float = 0.12


#: The vec4 uniform arrays atmosphere.glsl reads, MAX_ATMOSPHERES each.
_ARRAYS = ("uAtmoCam", "uAtmoShape", "uAtmoScatter", "uAtmoMie", "uAtmoExtra")


class Atmospheres:
    """The atmospheres of a world, as shader inputs shared by every node bound.

    Usage::

        atmospheres = Atmospheres()
        index = atmospheres.add(planet_radius, AtmosphereParams(), position)
        atmospheres.bind(terrain.root)        # and the sky, clouds...
        # every frame, camera in the world frame (float64):
        atmospheres.update(camera_world)
    """

    def __init__(self):
        self._bodies: list[list] = []  # [radius, params, position]
        self._count = PTA_int.empty_array(1)
        self._arrays = {
            name: PTA_LVecBase4f.empty_array(MAX_ATMOSPHERES) for name in _ARRAYS
        }

    def add(self, radius: float, params: AtmosphereParams, position=(0, 0, 0)) -> int:
        """Add a planet's atmosphere.

        :param radius: the planet's ground (sea-level) radius, metres
        :param params: the air
        :param position: the planet centre in the world frame (float64)
        :returns: its index (the GLSL's atmosphere index)
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
            p.multiple_scattering, 0.0, 0.0, 0.0
        )
        self._count[0] = len(self._bodies)
        return index

    def set_position(self, index: int, position) -> None:
        """Move a planet (orbits): its centre in the world frame, float64."""
        self._bodies[index][2] = np.asarray(position, float)

    def params(self, index: int) -> AtmosphereParams:
        return self._bodies[index][1]

    def __len__(self) -> int:
        return len(self._bodies)

    def bind(self, node: NodePath) -> None:
        """Give a node (and its subtree) the atmosphere uniforms."""
        node.set_shader_input("uAtmoCount", self._count)
        for name, array in self._arrays.items():
            node.set_shader_input(name, array)

    def update(self, camera_world) -> None:
        """Per frame: the camera's place in each atmosphere (float64 maths)."""
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
        self, index: int, point_world, sun_direction
    ) -> np.ndarray:
        """Sunlight reaching a point through atmosphere *index* (the GLSL's
        atmoSunlight, for CPU-side lighting): (3,) factors in [0, 1].

        Marched numerically (64 steps); for occasional use.
        """
        radius, params, position = self._bodies[index]
        p = np.asarray(point_world, float) - position
        d = np.asarray(sun_direction, float)
        d = d / np.linalg.norm(d)
        top = radius + params.height
        # Distance to the top of the atmosphere along the sun ray.
        b = float(p @ d)
        c = float(p @ p) - top * top
        if c > 0.0 and b > 0.0:
            return np.ones(3)  # above the air, looking away from it
        disc = b * b - c
        if disc <= 0.0:
            return np.ones(3)
        t1 = -b + np.sqrt(disc)
        s = (np.arange(64) + 0.5) / 64.0 * max(t1, 0.0)
        points = p + s[:, None] * d
        r = np.linalg.norm(points, axis=1)
        if np.any(r < radius):
            return np.zeros(3)
        ds = max(t1, 0.0) / 64.0
        depth_r = np.sum(np.exp(-(r - radius) / params.rayleigh_height)) * ds
        depth_m = np.sum(np.exp(-(r - radius) / params.mie_height)) * ds
        return np.exp(
            -(np.asarray(params.rayleigh) * depth_r + params.mie_extinction * depth_m)
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
