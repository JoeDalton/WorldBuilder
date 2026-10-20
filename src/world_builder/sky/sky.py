"""The sky sphere: stars, the sun's disc, and the atmospheres' scattered light."""

from __future__ import annotations

import math
from pathlib import Path

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LMatrix4f,
    LVecBase3f,
    NodePath,
    OmniBoundingVolume,
    PTA_LVecBase3f,
    Shader,
    Texture,
    Vec3,
)

from world_builder.atmosphere import Atmospheres, bind_defaults

SHADER_PATH = Path(__file__).parent / "shaders"
#: The sun's angular radius as drawn (about twice the real one: a disc that
#: reads at any resolution), radians.
SUN_ANGULAR_RADIUS = math.radians(0.55)


class Sky:
    """A sky sphere around the camera.

    Draw it with the far camera only (``hide`` it from the near one), recentre
    and scale it every frame inside the far camera's range (:meth:`update`).
    """

    def __init__(
        self,
        parent: NodePath,
        stars: Texture,
        atmospheres: Atmospheres | None = None,
        star_intensity: float = 0.6,
        sun_radiance=(4000.0, 3800.0, 3500.0),
        exposure: float = 0.9,
        star_hpr=(0.0, 0.0, 0.0),
        star_adaptation: float = 60.0,
    ):
        """
        :param parent: usually ``render``; the sphere is placed at the camera
        :param stars: a cube map (see :mod:`starfield`)
        :param atmospheres: the world's atmospheres (bound here); None = space
        :param star_intensity: linear radiance of a full-white star texel
        :param sun_radiance: the disc's linear radiance before the air
        :param exposure: as the terrain's (they share the tonemap)
        :param star_hpr: rotation of the star map in the world, degrees
        :param star_adaptation: stars dim as exp(-sky luminance * this):
            the eye adapting to a bright sky (a twilight sky of luminance
            ~0.05 hides all but the brightest)
        """
        self.node = parent.attach_new_node(_sphere(48, 24))
        self.node.node().set_bounds(OmniBoundingVolume())
        self.node.node().set_final(True)
        self.node.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "sky.vert",
                fragment=SHADER_PATH / "sky.frag",
            )
        )
        self.node.set_bin("background", 0)
        self.node.set_depth_write(False)
        self.node.set_depth_test(False)
        self.node.set_light_off()
        self.node.set_two_sided(True)
        bind_defaults(self.node)
        if atmospheres is not None:
            atmospheres.bind(self.node)
        self._sun = PTA_LVecBase3f.empty_array(1)
        self._sun[0] = LVecBase3f(0.0, 0.0, 1.0)
        self.node.set_shader_inputs(
            uStars=stars,
            uStarIntensity=float(star_intensity),
            uSunDir=self._sun,
            uSunRadiance=LVecBase3f(*sun_radiance),
            uSunCos=math.cos(SUN_ANGULAR_RADIUS),
            uExposure=float(exposure),
            uStarAdaptation=float(star_adaptation),
        )
        self.set_star_rotation(star_hpr)

    def set_star_rotation(self, hpr) -> None:
        """Turn the star map in the world (degrees)."""
        rotation = NodePath("star_rotation")
        rotation.set_hpr(*hpr)
        # The shader maps world directions into the map: the inverse rotation.
        # (A mat3 uniform takes an LMatrix4's upper 3x3; Panda uploads it so
        # GLSL's M * v is Panda's v * M.)
        matrix = LMatrix4f(rotation.get_mat())
        matrix.invert_in_place()
        self.node.set_shader_input("uStarRotation", matrix)

    def set_sun(self, direction) -> None:
        """:param direction: toward the sun, world frame"""
        self._sun[0] = LVecBase3f(*Vec3(*direction).normalized())

    def update(self, camera_render, far: float) -> None:
        """Per frame: around the camera, inside the far camera's range.

        :param camera_render: the camera's position in the render frame
        :param far: the far camera's far distance
        """
        self.node.set_pos(*camera_render)
        self.node.set_scale(0.9 * far)

    def remove(self) -> None:
        self.node.remove_node()


def _sphere(segments: int, rings: int) -> GeomNode:
    """A unit UV sphere, seen from inside."""
    vdata = GeomVertexData("sky", GeomVertexFormat.get_v3(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    for ring in range(rings + 1):
        el = -0.5 * math.pi + math.pi * ring / rings
        for seg in range(segments + 1):
            az = 2.0 * math.pi * seg / segments
            vertex.add_data3(
                math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
            )
    tris = GeomTriangles(Geom.UH_static)
    for ring in range(rings):
        for seg in range(segments):
            a = ring * (segments + 1) + seg
            b = a + segments + 1
            tris.add_vertices(a, a + 1, b)
            tris.add_vertices(a + 1, b + 1, b)
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("sky")
    node.add_geom(geom)
    return node
