"""
A game's models lit and displayed as the world is (docs/source/object_lighting.md,
part G of the shadows' plan), not by Panda3D's fixed-function lighting.

``model_main.glsl`` takes the model's colour (its texture, vertex colour,
colour scale and material's base colour) in linear light, lights it, as the
terrain lights its ground (``object_light.glsl``) or physically based
(``model_pbr.glsl``, see below), by:

- the sunlight at each pixel (``model_light.glsl``'s ``modelSunlight``):
  through its body's air, in every world shadow (the other bodies'
  eclipses, an airless body's own horizon, its clouds, its mountains);
- the sky light at the model: the node's own Panda3D ambient light, which
  :class:`~world_builder.lights.TrackedLights` sets where it is;

then adds the air in front, and applies the world's exposure and tonemap.
No more channels clipped at 1, no undecoded mid-tones.

A model needs the inputs of the body it is near (its air, its maps):
:class:`ModelLighting` holds them on a node of its own, bound as the body's
terrain root is, and :meth:`ModelLighting.apply` gives a model that state.
``World.track_light(node, shader=True)`` does it every frame for the body
nearest the node.

Usage, by hand (``World`` does all of it)::

    lighting = ModelLighting(exposure=0.9, atmosphere=0, terrain=terrain)
    atmospheres.bind(lighting.inputs)
    shadows.bind(lighting.inputs, body_index)
    shadow_map.bind(lighting.inputs)
    clouds.bind_shadow_inputs(lighting.inputs)
    lighting.set_sun(sun, sun_color)
    tracked.track(ship, shader=True)
    # every frame, after the lights:
    lighting.update(camera_render)
    lighting.apply(ship)

How a model is shaded is a quality knob (:class:`ModelSettings`'s
``shading``): the terrain's light (:data:`SIMPLE`), or physically based,
derived from panda3d-simplepbr's (``model_pbr.glsl``: glTF's metal and
roughness, emission), with the models' normal maps, and the sky reflected
from the sky-view table. One shader per tier (``model_<tier>.frag``).

A glTF model is authored in linear light (its colour textures sRGB, which
the GPU decodes); an older one as displayed. :func:`mark_linear` tells the
shader which, per part (``World.track_light`` does it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    GeomNode,
    LMatrix4f,
    LVecBase3f,
    MaterialAttrib,
    NodePath,
    PTA_float,
    PTA_LMatrix4f,
    PTA_LVecBase3f,
    Shader,
    ShaderAttrib,
    ShaderInput,
    Texture,
    TextureAttrib,
)

from world_builder.atmosphere import bind_defaults
from world_builder.atmosphere.sky_view import bind_sky_view_defaults
from world_builder.clouds.shadow_defaults import cloud_shadow_defaults
from world_builder.quality import QualityLevel, QualitySettings, knob
from world_builder.shadows import bind_defaults as bind_shadow_defaults

SHADER_PATH = Path(__file__).parent / "shaders"

__all__ = [
    "PBR",
    "PBR_NORMAL",
    "PBR_SKY",
    "SIMPLE",
    "ModelLighting",
    "ModelSettings",
    "mark_linear",
    "model_shader",
    "unmark_linear",
]

#: The shading tiers (:class:`ModelSettings`' ``shading``), cheapest first.
#: As the terrain lights its ground (``object_light.glsl``).
SIMPLE = 0
#: Physically based: glTF's metal and roughness, emission
#: (``model_pbr.glsl``, derived from panda3d-simplepbr).
PBR = 1
#: :data:`PBR` and the models' normal maps.
PBR_NORMAL = 2
#: :data:`PBR_NORMAL` and the sky reflected from the sky-view table.
PBR_SKY = 3
_FRAGMENTS = {
    SIMPLE: "model_simple.frag",
    PBR: "model_pbr.frag",
    PBR_NORMAL: "model_pbr_normal.frag",
    PBR_SKY: "model_pbr_sky.frag",
}

_SHADERS: dict[int, Shader] = {}


def model_shader(shading: int = PBR_NORMAL) -> Shader:
    """The models' shader at a shading tier (one per tier, for all).

    :param shading: :data:`SIMPLE`, :data:`PBR`, :data:`PBR_NORMAL` (the
        default, HIGH's) or :data:`PBR_SKY`
    """
    shader = _SHADERS.get(shading)
    if shader is None:
        shader = Shader.load(
            Shader.SL_GLSL,
            vertex=SHADER_PATH / "model.vert",
            fragment=SHADER_PATH / _FRAGMENTS[shading],
        )
        _SHADERS[shading] = shader
    return shader


@dataclass(frozen=True)
class ModelSettings(QualitySettings):
    """How the game's models drawn by the model shader are shaded
    (``World.track_light(node, shader=True)``; see the module). Live.

    :ivar shading: :data:`SIMPLE`, :data:`PBR`, :data:`PBR_NORMAL` or
        :data:`PBR_SKY`
    """

    shading: int = knob(
        PBR_NORMAL,
        "Model shading",
        choices={
            "Simple": SIMPLE,
            "PBR": PBR,
            "PBR + normal maps": PBR_NORMAL,
            "PBR + sky reflections": PBR_SKY,
        },
        help="How ships and other models reflect light. Simple lights them as "
        "the ground; PBR shows their metal, gloss and glowing parts, then "
        "their surface detail (normal maps), then the sky in smooth metal. "
        "Costs per pixel a model covers: most where a cockpit fills the "
        "screen.",
    )

    LABEL: ClassVar = "Models"
    PRESETS: ClassVar = {
        QualityLevel.LOW: {"shading": SIMPLE},
        QualityLevel.MID: {"shading": PBR},
        QualityLevel.ULTRA: {"shading": PBR_SKY},
    }


# The texture formats the GPU decodes from sRGB as it samples them.
_SRGB_FORMATS = frozenset(
    (
        Texture.F_srgb,
        Texture.F_srgb_alpha,
        Texture.F_sluminance,
        Texture.F_sluminance_alpha,
    )
)


def _authored_linear(state: Any) -> bool:
    """:returns: whether a render state is glTF's: an sRGB colour texture
    (decoded by the GPU), or a PBR material (its base colour linear)"""
    if state.has_attrib(TextureAttrib):
        textures = state.get_attrib(TextureAttrib)
        for index in range(textures.get_num_on_stages()):
            texture = textures.get_on_texture(textures.get_on_stage(index))
            if texture is not None and texture.get_format() in _SRGB_FORMATS:
                return True
    if state.has_attrib(MaterialAttrib):
        material = state.get_attrib(MaterialAttrib).get_material()
        if material is not None and material.has_base_color():
            return True
    return False


#: The tag :func:`mark_linear` puts on a model that is itself a part
#: authored in linear light (:meth:`ModelLighting.apply` replaces its shader
#: state, a ``uBaseLinear`` input on it with it).
LINEAR_TAG = "world_builder_linear"


def mark_linear(node: NodePath) -> list[NodePath]:
    """Tell the model shader which parts of a model are authored in linear
    light (glTF: sRGB colour textures, or PBR materials) rather than as
    displayed: a ``uBaseLinear`` input on each such part (its states', or
    any of its geoms'), or :data:`LINEAR_TAG` on *node* itself.

    :returns: the parts marked (for :func:`unmark_linear`)
    """
    marked = []
    parts = list(node.find_all_matches("**/+GeomNode"))  # below it only
    if isinstance(node.node(), GeomNode):
        parts.insert(0, node)
    for part in parts:
        geoms = part.node()
        states = [part.get_net_state()] + [
            geoms.get_geom_state(i) for i in range(geoms.get_num_geoms())
        ]
        if any(_authored_linear(state) for state in states):
            if part == node:
                part.set_tag(LINEAR_TAG, "1")
            else:
                part.set_shader_input("uBaseLinear", 1)
            marked.append(part)
    return marked


def unmark_linear(parts: list[NodePath]) -> None:
    """Undo :func:`mark_linear`."""
    for part in parts:
        if part.is_empty():
            continue
        if part.has_tag(LINEAR_TAG):
            part.clear_tag(LINEAR_TAG)
        else:
            part.clear_shader_input("uBaseLinear")


class ModelLighting:
    """The model shader and its inputs for the models near one body (or in
    open space, with none), on a node of their own (:attr:`inputs`): bind
    the body's parts to it as to its terrain's root, and :meth:`apply` it
    to each model every frame (a part that swaps a texture in rebinds it
    there).

    :ivar inputs: the node holding the shader and its inputs (never drawn)
    :ivar terrain: the body's terrain (its rotation), or None
    :ivar settings: how its models are shaded (:meth:`set_settings`)
    """

    SETTINGS_TYPE: ClassVar = ModelSettings

    def __init__(
        self,
        exposure: float = 0.9,
        atmosphere: int | None = None,
        terrain: Any = None,
        light_scale: float = 1.0,
        settings: ModelSettings | None = None,
    ) -> None:
        """
        :param exposure: the body's exposure (its terrain's)
        :param atmosphere: the body's atmosphere index; None = airless
        :param terrain: the body's terrain (its rotation each frame); None
            = none (open space)
        :param light_scale: ``World.light_scale``: the Panda3D lights'
            colours are the world's times it, taken back off in the shader
        :param settings: how its models are shaded; None = the default
            (HIGH's)
        """
        self.terrain = terrain
        self.settings = settings or ModelSettings()
        self.inputs = NodePath("model_lighting")
        self.inputs.set_shader(model_shader(self.settings.shading))
        bind_defaults(self.inputs)
        bind_shadow_defaults(self.inputs)
        cloud_shadow_defaults(self.inputs)
        # The sky-view table (the PBR_SKY tier): none until the world's
        # sky binds its own (SkyView.bind)
        bind_sky_view_defaults(self.inputs)
        # As displayed, but on the parts mark_linear marks
        self.inputs.set_shader_input("uBaseLinear", 0)
        # The inputs' attrib marked linear (apply), and what it was made from
        self._linear_from: ShaderAttrib | None = None
        self._linear: ShaderAttrib | None = None
        self._frame = {}
        for name in ("uCamRender", "uSunDir", "uSunColor"):
            array = PTA_LVecBase3f.empty_array(1)
            self._frame[name] = array
            self.inputs.set_shader_input(name, array)
        # A mat3 input is given as a 4x4 (its upper-left 3x3 is taken).
        self._axes = PTA_LMatrix4f.empty_array(1)
        self._axes[0] = LMatrix4f.ident_mat()
        # The body's rotation last given to the shader (see update)
        self._rotation: np.ndarray | None = None
        self._scale = PTA_float.empty_array(1)
        self.inputs.set_shader_inputs(
            uPlanetAxes=self._axes,
            uLightScale=self._scale,
            uExposure=float(exposure),
            uOwnAtmosphere=-1 if atmosphere is None else int(atmosphere),
        )
        self.set_light_scale(light_scale)
        self.set_sun((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))

    def set_sun(self, direction: ArrayLike, color: ArrayLike) -> None:
        """:param direction: toward the sun, render frame (any length)
        :param color: sunlight above any air, linear"""
        direction = np.asarray(direction, dtype=np.float64)
        self._frame["uSunDir"][0] = LVecBase3f(*(direction / np.linalg.norm(direction)))
        self._frame["uSunColor"][0] = LVecBase3f(*np.asarray(color, dtype=np.float64))

    def set_settings(self, settings: ModelSettings) -> None:
        """Shade its models at another tier (live: :meth:`apply` gives
        them the new shader at the next frame)."""
        self.settings = settings
        shader = model_shader(settings.shading)
        if self.inputs.get_shader() != shader:
            self.inputs.set_shader(shader)

    def set_light_scale(self, scale: float) -> None:
        """``World.light_scale``, which the models' ambient lights carry."""
        self._scale[0] = float(scale) if scale > 0.0 else 1.0

    def update(self, camera_render: ArrayLike) -> None:
        """Per frame: the main camera (render frame), and the body's turn.

        :param camera_render: the main camera's position in render
        """
        self._frame["uCamRender"][0] = LVecBase3f(
            *np.asarray(camera_render, dtype=np.float64)
        )
        if self.terrain is not None:
            rotation = np.asarray(self.terrain.rotation)
            # Only when it turned: most bodies never do
            if self._rotation is not None and np.array_equal(rotation, self._rotation):
                return
            self._rotation = rotation.copy()
            # Panda's matrices act on row vectors: its rows are GLSL's
            # columns, so the world-from-planet rotation goes in transposed.
            axes = np.eye(4)
            axes[:3, :3] = rotation.T
            self._axes[0] = LMatrix4f(*axes.ravel())

    def apply(self, node: NodePath) -> None:
        """Give a model (and its children) the shader and these inputs, in
        place of any shader state of its own: each frame, as the inputs'
        textures may have changed and its nearest body with them."""
        attrib = self.inputs.get_attrib(ShaderAttrib)
        if node.has_tag(LINEAR_TAG):  # mark_linear's, on the node itself
            if self._linear_from != attrib:
                self._linear_from = attrib
                self._linear = attrib.set_shader_input(ShaderInput("uBaseLinear", 1))
            attrib = self._linear
        # Setting the same attrib again still makes the node a new state,
        # every frame (Panda's state cache churns, and its garbage
        # collection with it)
        if not node.has_attrib(ShaderAttrib) or node.get_attrib(ShaderAttrib) != attrib:
            node.set_attrib(attrib)

    def destroy(self) -> None:
        self.inputs.remove_node()
