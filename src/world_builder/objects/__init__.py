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
the GPU decodes); an older one as displayed. :func:`mark_model` tells the
shader which, per part, with each part's glTF occlusion map
(``World.track_light`` does it).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    Filename,
    GeomNode,
    LMatrix4f,
    LVecBase3f,
    MaterialAttrib,
    ModelRoot,
    NodePath,
    PNMImage,
    PTA_float,
    PTA_LMatrix4f,
    PTA_LVecBase3f,
    SamplerState,
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
LOGGER = logging.getLogger(__name__)

__all__ = [
    "PBR",
    "PBR_FULL",
    "PBR_NORMAL",
    "PBR_SKY",
    "SIMPLE",
    "ModelLighting",
    "ModelSettings",
    "gltf_occlusion",
    "mark_model",
    "model_shader",
    "unmark_model",
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
#: :data:`PBR_SKY` and glTF's occlusion maps, the sky's reflection blurred
#: by the roughness (the table's mipmaps), and the light the ground bounces
#: up.
PBR_FULL = 4
_FRAGMENTS = {
    SIMPLE: "model_simple.frag",
    PBR: "model_pbr.frag",
    PBR_NORMAL: "model_pbr_normal.frag",
    PBR_SKY: "model_pbr_sky.frag",
    PBR_FULL: "model_pbr_full.frag",
}
#: The ground's albedo under a model (its bounced light, :data:`PBR_FULL`)
#: where its body gives none: the land's average, about.
GROUND_ALBEDO = 0.2

_SHADERS: dict[int, Shader] = {}


def model_shader(shading: int = PBR_SKY) -> Shader:
    """The models' shader at a shading tier (one per tier, for all).

    :param shading: :data:`SIMPLE`, :data:`PBR`, :data:`PBR_NORMAL`,
        :data:`PBR_SKY` (the default, LOW to HIGH's) or :data:`PBR_FULL`
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

    :ivar shading: :data:`SIMPLE`, :data:`PBR`, :data:`PBR_NORMAL`,
        :data:`PBR_SKY` or :data:`PBR_FULL`
    """

    shading: int = knob(
        PBR_SKY,
        "Model shading",
        choices={
            "Simple": SIMPLE,
            "PBR": PBR,
            "PBR + normal maps": PBR_NORMAL,
            "PBR + sky reflections": PBR_SKY,
            "Full PBR": PBR_FULL,
        },
        help="How ships and other models reflect light. Simple lights them as "
        "the ground; PBR shows their metal, gloss and glowing parts, then "
        "their surface detail (normal maps), then the sky in smooth metal. "
        "Full adds their occlusion maps, the sky blurred in rough metal and "
        "the light the ground bounces up. Costs per pixel a model covers: "
        "most where a cockpit fills the screen.",
    )

    LABEL: ClassVar = "Models"
    # LOW to HIGH alike: the cheaper tiers stay choices of the knob.
    PRESETS: ClassVar = {QualityLevel.ULTRA: {"shading": PBR_FULL}}
    DISTINCT_PRESETS: ClassVar = False


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


_WHITE: Texture | None = None


def _white() -> Texture:
    """A 1 x 1 white texture: no occlusion."""
    global _WHITE
    if _WHITE is None:
        _WHITE = Texture("no_occlusion")
        _WHITE.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
        _WHITE.set_ram_image(b"\xff")
    return _WHITE


def _material_name(state: Any) -> str | None:
    if state.has_attrib(MaterialAttrib):
        material = state.get_attrib(MaterialAttrib).get_material()
        if material is not None:
            return material.get_name()
    return None


@dataclass
class _Occlusion:
    """A glTF material's occlusion: in its metal-roughness texture's red
    (packed: ``texture`` None) or a texture of its own; and its strength."""

    texture: Texture | None
    strength: float


def _gltf_path(node: NodePath) -> Path | None:
    """:returns: the .gltf file a model was loaded from (its ModelRoot's
    path, at or under *node*); None for another kind of model"""
    roots = [node] if isinstance(node.node(), ModelRoot) else []
    roots += list(node.find_all_matches("**/+ModelRoot"))
    for root in roots:
        path = Path(root.node().get_fullpath().to_os_specific())
        if path.suffix.lower() == ".gltf" and path.is_file():
            return path
    return None


def gltf_occlusion(node: NodePath) -> dict[str, _Occlusion]:
    """A glTF model's occlusion maps, which panda3d-gltf does not load: read
    from its .gltf file, by material name. A map that is the material's
    metal-roughness texture is packed in its red channel (glTF's ORM); any
    other is loaded (linear).

    :returns: material name -> its occlusion; {} for a model with none, or
        not from a .gltf file
    """
    path = _gltf_path(node)
    if path is None:
        return {}
    try:
        gltf = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        LOGGER.warning(f"{path}: {error}; no occlusion maps")
        return {}
    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    found = {}
    for index, material in enumerate(gltf.get("materials", [])):
        occlusion = material.get("occlusionTexture")
        if occlusion is None:
            continue
        name = material.get("name", f"mat{index}")  # panda3d-gltf's names
        strength = float(occlusion.get("strength", 1.0))
        metal = material.get("pbrMetallicRoughness", {})
        packed = metal.get("metallicRoughnessTexture", {}).get("index")
        if occlusion["index"] == packed:
            found[name] = _Occlusion(None, strength)
            continue
        try:
            uri = images[textures[occlusion["index"]]["source"]]["uri"]
        except (IndexError, KeyError):
            continue  # embedded in a buffer: not read
        texture = _load_occlusion(path.parent / uri)
        if texture is not None:
            found[name] = _Occlusion(texture, strength)
    return found


def _load_occlusion(path: Path) -> Texture | None:
    image = PNMImage()
    if not image.read(Filename.from_os_specific(str(path))):
        LOGGER.warning(f"{path}: not read; no occlusion map")
        return None
    texture = Texture(path.name)
    texture.load(image)
    texture.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    texture.set_magfilter(SamplerState.FT_linear)
    return texture


#: What :func:`mark_model` changed: (a GeomNode, a geom's index, its state
#: before).
Marks = list[tuple[NodePath, int, Any]]


def _part_inputs(state: Any, occlusion: dict[str, _Occlusion]) -> dict[str, Any]:
    """The inputs a geom of *state* (its net state) needs: ``uBaseLinear``
    if authored in linear light, its occlusion's if it has one."""
    inputs: dict[str, Any] = {}
    if _authored_linear(state):
        inputs["uBaseLinear"] = 1
    found = occlusion.get(_material_name(state) or "")
    if found is not None:
        inputs["uOcclusionMode"] = 1 if found.texture is None else 2
        inputs["uOcclusionStrength"] = found.strength
        if found.texture is not None:
            inputs["uOcclusion"] = found.texture
    return inputs


def mark_model(node: NodePath) -> Marks:
    """Tell the model shader what each part of a model needs beyond the
    defaults: whether it is authored in linear light (glTF: sRGB colour
    textures, or PBR materials) rather than as displayed
    (``uBaseLinear``), and its glTF occlusion map (:func:`gltf_occlusion`).
    Set on each geom's own state, which composes over the shader state
    :meth:`ModelLighting.apply` gives the model.

    :returns: what changed, for :func:`unmark_model`
    """
    occlusion = gltf_occlusion(node)
    parts = list(node.find_all_matches("**/+GeomNode"))  # below it only
    if isinstance(node.node(), GeomNode):
        parts.insert(0, node)
    marks: Marks = []
    for part in parts:
        geoms = part.node()
        net = part.get_net_state()
        for index in range(geoms.get_num_geoms()):
            state = geoms.get_geom_state(index)
            inputs = _part_inputs(net.compose(state), occlusion)
            if not inputs:
                continue
            attrib = (
                state.get_attrib(ShaderAttrib)
                if state.has_attrib(ShaderAttrib)
                else ShaderAttrib.make()
            )
            for name, value in inputs.items():
                attrib = attrib.set_shader_input(ShaderInput(name, value))
            geoms.set_geom_state(index, state.set_attrib(attrib))
            marks.append((part, index, state))
    return marks


def unmark_model(marks: Marks) -> None:
    """Undo :func:`mark_model`."""
    for part, index, state in marks:
        if not part.is_empty() and index < part.node().get_num_geoms():
            part.node().set_geom_state(index, state)


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
        # The parts' own (mark_model): as displayed, no occlusion map
        self.inputs.set_shader_inputs(
            uBaseLinear=0,
            uOcclusionMode=0,
            uOcclusionStrength=1.0,
            uOcclusion=_white(),
        )
        self.set_ground_albedo(GROUND_ALBEDO)
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

    def set_ground_albedo(self, albedo: float) -> None:
        """The ground's albedo under its models: how much light it bounces
        up to them (:data:`PBR_FULL`)."""
        self.inputs.set_shader_input("uGroundAlbedo", float(albedo))

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
        # Setting the same attrib again still makes the node a new state,
        # every frame (Panda's state cache churns, and its garbage
        # collection with it)
        if not node.has_attrib(ShaderAttrib) or node.get_attrib(ShaderAttrib) != attrib:
            node.set_attrib(attrib)

    def destroy(self) -> None:
        self.inputs.remove_node()
