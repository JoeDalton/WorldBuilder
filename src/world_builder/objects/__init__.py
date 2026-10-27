"""
A game's models lit and displayed as the world is (docs/source/object_lighting.md,
part G of the shadows' plan), not by Panda3D's fixed-function lighting.

``model.frag`` decodes the model's colour (its texture, vertex colour, colour
scale and material's diffuse, authored as displayed) to linear light, lights
it as the terrain lights its ground (``object_light.glsl``) by:

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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    LMatrix4f,
    LVecBase3f,
    NodePath,
    PTA_float,
    PTA_LMatrix4f,
    PTA_LVecBase3f,
    Shader,
    ShaderAttrib,
)

from world_builder.atmosphere import bind_defaults
from world_builder.clouds.field import cloud_shadow_defaults
from world_builder.shadows import bind_defaults as bind_shadow_defaults

SHADER_PATH = Path(__file__).parent / "shaders"

__all__ = ["ModelLighting", "model_shader"]

_SHADER: Shader | None = None


def model_shader() -> Shader:
    """The models' shader (one for all)."""
    global _SHADER
    if _SHADER is None:
        _SHADER = Shader.load(
            Shader.SL_GLSL,
            vertex=SHADER_PATH / "model.vert",
            fragment=SHADER_PATH / "model.frag",
        )
    return _SHADER


class ModelLighting:
    """The model shader and its inputs for the models near one body (or in
    open space, with none), on a node of their own (:attr:`inputs`): bind
    the body's parts to it as to its terrain's root, and :meth:`apply` it
    to each model every frame (a part that swaps a texture in rebinds it
    there).

    :ivar inputs: the node holding the shader and its inputs (never drawn)
    :ivar terrain: the body's terrain (its rotation), or None
    """

    def __init__(
        self,
        exposure: float = 0.9,
        atmosphere: int | None = None,
        terrain: Any = None,
        light_scale: float = 1.0,
    ) -> None:
        """
        :param exposure: the body's exposure (its terrain's)
        :param atmosphere: the body's atmosphere index; None = airless
        :param terrain: the body's terrain (its rotation each frame); None
            = none (open space)
        :param light_scale: ``World.light_scale``: the Panda3D lights'
            colours are the world's times it, taken back off in the shader
        """
        self.terrain = terrain
        self.inputs = NodePath("model_lighting")
        self.inputs.set_shader(model_shader())
        bind_defaults(self.inputs)
        bind_shadow_defaults(self.inputs)
        cloud_shadow_defaults(self.inputs)
        self._frame = {}
        for name in ("uCamRender", "uSunDir", "uSunColor"):
            array = PTA_LVecBase3f.empty_array(1)
            self._frame[name] = array
            self.inputs.set_shader_input(name, array)
        # A mat3 input is given as a 4x4 (its upper-left 3x3 is taken).
        self._axes = PTA_LMatrix4f.empty_array(1)
        self._axes[0] = LMatrix4f.ident_mat()
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
            # Panda's matrices act on row vectors: its rows are GLSL's
            # columns, so the world-from-planet rotation goes in transposed.
            axes = np.eye(4)
            axes[:3, :3] = np.asarray(self.terrain.rotation).T
            self._axes[0] = LMatrix4f(*axes.ravel())

    def apply(self, node: NodePath) -> None:
        """Give a model (and its children) the shader and these inputs, in
        place of any shader state of its own: each frame, as the inputs'
        textures may have changed and its nearest body with them."""
        node.set_attrib(self.inputs.get_attrib(ShaderAttrib))

    def destroy(self) -> None:
        self.inputs.remove_node()
