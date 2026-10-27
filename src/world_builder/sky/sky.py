"""The sky sphere: stars, the sun's disc, and the atmospheres' scattered light."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    ColorBlendAttrib,
    CullBinManager,
    DepthTestAttrib,
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
from world_builder.quality import REBUILD, QualityLevel, QualitySettings, knob
from world_builder.shadows import SUN_ANGULAR_RADIUS
from world_builder.shadows import bind_defaults as bind_shadow_defaults
from world_builder.sky.starfield import NebulaStyle, StarCatalogue, Starfield

SHADER_PATH = Path(__file__).parent / "shaders"
#: The sky's cull bin: after the opaque bin (20), so its per-pixel atmosphere
#: is computed only where no terrain was drawn (its vertices sit on the far
#: plane, depth-tested), and before the clouds (``clouds.sphere``, 22-29).
SKY_BIN = "sky"
SKY_BIN_SORT = 21


@dataclass(frozen=True)
class SkySettings(QualitySettings):
    """What the procedural sky costs (:mod:`world_builder.quality`); neither
    does anything with a painted cube map. Its real per-pixel cost is the
    air's (``AtmosphereSettings.steps``).

    :ivar star_fraction: the share of the point stars drawn, the brightest
        (each runs a short march through the air). Live
    :ivar glow_size: the diffuse light's cube map, texels per face (made
        once per size, at a little startup time). Rebuild
    """

    star_fraction: float = knob(
        1.0,
        "Stars",
        unit="x",
        range=(0.05, 1.0),
        help="The share of the stars drawn: the faintest go first.",
    )
    glow_size: int = knob(
        256,
        "Milky Way detail",
        REBUILD,
        unit="px",
        choices={"64": 64, "128": 128, "256": 256, "512": 512},
        advanced=True,
        help="The diffuse light's resolution, per cube face.",
    )

    LABEL: ClassVar = "Sky"
    PRESETS: ClassVar = {
        QualityLevel.LOW: {"star_fraction": 0.4, "glow_size": 128},
        QualityLevel.MID: {"star_fraction": 0.7},
        QualityLevel.ULTRA: {"glow_size": 512},
    }


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

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = SkySettings

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
        settings: SkySettings | None = None,
        sun_angular_radius: float = SUN_ANGULAR_RADIUS,
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
        :param settings: what the sky costs (:class:`SkySettings`;
            :meth:`set_settings` changes it); None = the HIGH preset
        :param sun_angular_radius: the disc's angular radius, radians: the
            sun's size, which every penumbra shares
            (:class:`world_builder.shadows.Shadows`)
        """
        self._starfield = stars if isinstance(stars, Starfield) else None
        glow = None
        if settings is None:
            # The starfield's glow as built (its size, if a settings choice).
            settings = SkySettings()
            if self._starfield is not None:
                glow = self._starfield.glow
                sizes = dict(SkySettings.knobs())["glow_size"].choices.values()
                if self._starfield.glow_size in sizes:
                    settings = replace(settings, glow_size=self._starfield.glow_size)
        elif self._starfield is not None:
            glow = self._starfield.glow_at(settings.glow_size)
        self._settings = settings
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
        _ensure_sky_bin()
        self.node.set_bin(SKY_BIN, 0)
        self.node.set_depth_write(False)
        self.node.set_attrib(DepthTestAttrib.make(DepthTestAttrib.M_less_equal))
        self.node.set_light_off()
        self.node.set_two_sided(True)
        bind_defaults(self.node)
        bind_shadow_defaults(self.node)
        if atmospheres is not None:
            atmospheres.bind(self.node)
        self._sun = PTA_LVecBase3f.empty_array(1)
        self._sun[0] = LVecBase3f(0.0, 0.0, 1.0)
        procedural = isinstance(stars, Starfield)
        self.node.set_shader_inputs(
            uStars=glow if procedural else stars,
            uGlowMode=1 if procedural else 0,
            uStarIntensity=float(star_intensity),
            uStarBrightness=float(star_brightness),
            uSunDir=self._sun,
            uSunRadiance=LVecBase3f(*sun_radiance),
            uSunCos=math.cos(sun_angular_radius),
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
            self.stars.set_bin(SKY_BIN, 1)
            self.stars.set_attrib(
                ColorBlendAttrib.make(
                    ColorBlendAttrib.M_add,
                    ColorBlendAttrib.O_one,
                    ColorBlendAttrib.O_one,
                )
            )
            self._show_stars(self._settings.star_fraction)

    @property
    def settings(self) -> SkySettings:
        """What the sky costs (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: SkySettings) -> None:
        """Change what the sky costs: the star share live, the glow's size
        by a new cube map (made once per size)."""
        changes = self._settings.changes(settings)
        if not changes:
            return
        self._settings = settings
        if self._starfield is None:
            return
        if "glow_size" in changes:
            self.node.set_shader_input(
                "uStars", self._starfield.glow_at(settings.glow_size)
            )
        self._show_stars(settings.star_fraction)

    def _show_stars(self, fraction: float) -> None:
        """Draw the brightest *fraction* of the stars (they are stored
        brightest first)."""
        geom = self.stars.node().modify_geom(0)
        total = geom.get_vertex_data().get_num_rows()
        points = GeomPoints(Geom.UH_static)
        points.add_next_vertices(max(1, round(total * fraction)))
        geom.set_primitive(0, points)

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
    """One point per star, brightest first: its direction (on the unit
    sphere, in the star map's frame) and (tint, flux)."""
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
    # Brightest first, so drawing the first N keeps the same sky's brightest
    # (SkySettings.star_fraction); the order is otherwise irrelevant (added).
    order = np.argsort(-catalogue.flux, kind="stable")
    data = np.empty((n, 7), dtype=np.float32)
    data[:, 0:3] = catalogue.dirs[order]
    data[:, 3:6] = catalogue.colour[order]
    data[:, 6] = catalogue.flux[order]
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


def _ensure_sky_bin() -> None:
    manager = CullBinManager.get_global_ptr()
    if manager.find_bin(SKY_BIN) == -1:
        manager.add_bin(SKY_BIN, CullBinManager.BT_fixed, SKY_BIN_SORT)
