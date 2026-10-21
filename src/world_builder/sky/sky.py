"""The sky sphere: stars, the sun's disc, and the atmospheres' scattered light."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    ColorBlendAttrib,
    Geom,
    GeomEnums,
    GeomNode,
    GeomPoints,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    InternalName,
    LMatrix4f,
    LVecBase3f,
    NodePath,
    OmniBoundingVolume,
    PTA_LVecBase3f,
    Shader,
    ShaderAttrib,
    TexGenAttrib,
    Texture,
    TextureStage,
    Vec3,
)

from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.sky.starfield import NebulaStyle, StarCatalogue, Starfield

SHADER_PATH = Path(__file__).parent / "shaders"
#: The sun's angular radius as drawn (about twice the real one: a disc that
#: reads at any resolution), radians.
SUN_ANGULAR_RADIUS = math.radians(0.55)


class Sky:
    """A sky sphere around the camera.

    Draw it with the far camera only (``hide(depth.HIDE_FROM_NEAR)``):
    it draws first, without depth, over whatever is there, so the near pass
    would paint it over the far one. Recentre it every frame
    (:meth:`update`). ``sky.vert`` puts it on the far plane whatever its
    scale, so any camera that draws it (the ocean's reflection, with a range
    of its own) sees it whole.

    With a :class:`starfield.Starfield`, the point stars are a child node,
    :attr:`stars`, drawn just after the sphere and added to it.

    :ivar node: the sphere (shader inputs, bins, transform)
    :ivar stars: the point stars' node; None with a painted cube map
    :ivar nebula: the current :class:`starfield.NebulaStyle`
    """

    def __init__(
        self,
        parent: NodePath,
        stars: Starfield | Texture,
        atmospheres: Atmospheres | None = None,
        star_intensity: float = 0.6,
        star_brightness: float = 8.0,
        nebula: NebulaStyle = NebulaStyle(),
        sun_radiance: ArrayLike = (4000.0, 3800.0, 3500.0),
        exposure: float = 0.9,
        star_hpr: ArrayLike = (0.0, 0.0, 0.0),
        star_adaptation: float = 60.0,
    ) -> None:
        """
        :param parent: usually ``render``; the sphere is placed at the camera
        :param stars: a procedural :class:`starfield.Starfield` (point stars
            and the diffuse light), or a painted cube map
            (:func:`starfield.load_cube_map`)
        :param atmospheres: the world's atmospheres (bound here); None = space
        :param star_intensity: a painted map's linear radiance of full white
        :param star_brightness: procedural stars: scales their flux (the
            linear radiance of a pixel a star covers whole)
        :param nebula: procedural glow: its colours (live: :meth:`set_nebula`)
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
        procedural = isinstance(stars, Starfield)
        self.node.set_shader_inputs(
            uStars=stars.glow if procedural else stars,
            uGlowMode=1 if procedural else 0,
            uStarIntensity=float(star_intensity),
            uStarBrightness=float(star_brightness),
            uSunDir=self._sun,
            uSunRadiance=LVecBase3f(*sun_radiance),
            uSunCos=math.cos(SUN_ANGULAR_RADIUS),
            uExposure=float(exposure),
            uStarAdaptation=float(star_adaptation),
        )
        self.set_nebula(nebula)
        self.set_star_rotation(star_hpr)
        self.stars = None
        if procedural:
            self.stars = self.node.attach_new_node(_star_points(stars.catalogue))
            self.stars.set_shader(
                Shader.load(
                    Shader.SL_GLSL,
                    vertex=SHADER_PATH / "stars.vert",
                    fragment=SHADER_PATH / "stars.frag",
                )
            )
            attrib = self.stars.get_attrib(ShaderAttrib)
            self.stars.set_attrib(
                attrib.set_flag(ShaderAttrib.F_shader_point_size, True)
            )
            # Point sprites on, or gl_PointCoord is undefined (Panda's GL
            # context is a compatibility one).
            self.stars.set_tex_gen(
                TextureStage.get_default(), TexGenAttrib.M_point_sprite
            )
            # Just after the sphere, added to it.
            self.stars.set_bin("background", 1)
            self.stars.set_attrib(
                ColorBlendAttrib.make(
                    ColorBlendAttrib.M_add,
                    ColorBlendAttrib.O_one,
                    ColorBlendAttrib.O_one,
                )
            )

    def set_nebula(self, style: NebulaStyle) -> None:
        """Colour the procedural glow (no effect on a painted map)."""
        self.nebula = style
        self.node.set_shader_inputs(
            uMilkyWay=LVecBase3f(*style.milky_way),
            uNebulaIntensity=float(style.intensity),
        )
        palette = PTA_LVecBase3f.empty_array(3)
        for index, colour in enumerate(style.palette):
            palette[index] = LVecBase3f(*colour)
        self.node.set_shader_input("uNebulaPalette", palette)

    def set_star_brightness(self, brightness: float) -> None:
        """Scale the procedural stars' flux (``star_brightness``; no effect on
        a painted map)."""
        self.node.set_shader_input("uStarBrightness", float(brightness))

    def set_star_rotation(self, hpr: ArrayLike) -> None:
        """Turn the star map in the world (``star_hpr``).

        :param hpr: heading, pitch, roll, degrees
        """
        rotation = NodePath("star_rotation")
        rotation.set_hpr(*hpr)
        # The shader maps world directions into the map: the inverse rotation.
        # (A mat3 uniform takes an LMatrix4's upper 3x3; Panda uploads it so
        # GLSL's M * v is Panda's v * M.)
        matrix = LMatrix4f(rotation.get_mat())
        matrix.invert_in_place()
        self.node.set_shader_input("uStarRotation", matrix)

    def set_sun(self, direction: ArrayLike) -> None:
        """Place the sun's disc (and the light the air scatters).

        :param direction: toward the sun, world frame (any length)
        """
        self._sun[0] = LVecBase3f(*Vec3(*direction).normalized())

    def update(self, camera_render: ArrayLike, far: float) -> None:
        """Per frame: around the camera, inside the far camera's range (the
        scale is a margin only: ``sky.vert`` draws it on the far plane).

        :param camera_render: the camera's position in the render frame
        :param far: the far camera's far distance
        """
        self.node.set_pos(*camera_render)
        self.node.set_scale(0.9 * far)

    def remove(self) -> None:
        """Take the sky (and its stars) off the scene."""
        self.node.remove_node()


def _star_points(catalogue: StarCatalogue) -> GeomNode:
    """One point per star: its direction (on the unit sphere, in the star
    map's frame) and (tint, flux)."""
    fmt = GeomVertexArrayFormat()
    fmt.add_column(
        InternalName.get_vertex(), 3, GeomEnums.NT_float32, GeomEnums.C_point
    )
    fmt.add_column(
        InternalName.make("star"), 4, GeomEnums.NT_float32, GeomEnums.C_other
    )
    combined = GeomVertexFormat()
    combined.add_array(fmt)
    n = len(catalogue.flux)
    data = np.empty((n, 7), dtype=np.float32)
    data[:, 0:3] = catalogue.dirs
    data[:, 3:6] = catalogue.colour
    data[:, 6] = catalogue.flux
    vdata = GeomVertexData(
        "stars", GeomVertexFormat.register_format(combined), Geom.UH_static
    )
    vdata.unclean_set_num_rows(n)
    memoryview(vdata.modify_array(0)).cast("B")[:] = data.tobytes()
    points = GeomPoints(Geom.UH_static)
    points.add_next_vertices(n)
    geom = Geom(vdata)
    geom.add_primitive(points)
    node = GeomNode("stars")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    return node


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
