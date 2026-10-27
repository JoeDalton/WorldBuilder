"""
The cloud shadow maps: the clouds' shadow on the ground, marched once per
texel instead of once per pixel of ground (part E of the shadows' plan).

The ground's shadow from the planet's clouds (cloud_field.glsl's
``cloudShadow``) is a march along the sun's ray through the slab. That ray
is the same for every point below the slab on it, so the march can be
stored by where the ray enters the slab's base: a map on the base's sphere
around the camera, each texel marched once (a render-to-texture pass,
``cloud_shadow_map.frag``), and each pixel of ground then one fetch at its
ray's entry point (analytic). Two maps: a near one, fine (60 m texels), and
a far one, coarse (500 m), out toward the horizon; past them the march per
pixel, and farther the shadow's mean (as before). Ground inside the slab
(mountain tops in the clouds) still marches: its ray does not start on the
base.

A map is drawn again (one frame of its buffer) once the camera has moved a
quarter of it, the sun has turned, the steps or the clouds' layer changed:
with the sun still, the shadow costs a fetch; with the sun swinging, a
march per texel each frame (still not per pixel).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    Camera,
    CardMaker,
    FrameBufferProperties,
    LVecBase4f,
    NodePath,
    OmniBoundingVolume,
    PTA_int,
    PTA_LVecBase4f,
    SamplerState,
    Shader,
    Texture,
)

from world_builder.terrain.chart import TangentFrame, latlon

SHADER_PATH = Path(__file__).parent / "shaders"
#: The near map's and the far map's texel spacing, metres.
NEAR_SPACING = 60.0
FAR_SPACING = 500.0
#: A map is drawn again once its centre is this share of its width away.
RECENTRE = 0.25


class _Map:
    """One cloud shadow map: its buffer, its card and its chart."""

    def __init__(
        self,
        output: Any,
        clouds: Any,
        name: str,
        size: int,
        spacing: float,
        to_ram: bool,
    ) -> None:
        self.size, self.spacing = size, spacing
        self.texture = Texture(name)
        props = FrameBufferProperties()
        props.set_rgba_bits(16, 16, 16, 16)
        props.set_float_color(True)
        self.buffer = output.make_texture_buffer(
            name, size, size, self.texture, to_ram, props
        )
        # Before every camera that reads it, and only when asked.
        self.buffer.set_sort(-300)
        self.buffer.set_clear_color_active(False)
        self.buffer.set_active(False)
        for setter in (self.texture.set_wrap_u, self.texture.set_wrap_v):
            setter(SamplerState.WM_clamp)
        self.texture.set_magfilter(SamplerState.FT_linear)
        self.texture.set_minfilter(SamplerState.FT_linear)
        self.root = NodePath(name)
        camera = self.root.attach_new_node(Camera(name))
        self.buffer.make_display_region().set_camera(camera)
        maker = CardMaker(name)
        maker.set_frame(-1.0, 1.0, -1.0, 1.0)
        card = self.root.attach_new_node(maker.generate())
        card.node().set_bounds(OmniBoundingVolume())
        card.node().set_final(True)
        card.set_two_sided(True)
        card.set_depth_test(False)
        card.set_depth_write(False)
        card.set_shader(_bake_shader())
        clouds.bind_shadow_inputs(card)
        self.chart = PTA_LVecBase4f.empty_array(4)  # the receivers' too
        self.steps = PTA_int.empty_array(1)
        card.set_shader_inputs(uMapChart=self.chart, uMapSteps=self.steps)
        self.drawn: tuple | None = None  # (centre, sun, steps, layer) drawn

    def draw(
        self, up: np.ndarray, sun: np.ndarray, steps: int, layer: int, base: float
    ) -> None:
        """Draw the map this frame, centred under *up* (planet frame)."""
        lat, lon = latlon(up[None])
        frame = TangentFrame.at(float(lat[0]), float(lon[0]))
        self.chart[0] = LVecBase4f(*frame.up, base)
        self.chart[1] = LVecBase4f(*frame.east, self.spacing)
        self.chart[2] = LVecBase4f(*frame.north, float(self.size))
        self.steps[0] = steps
        self.buffer.set_active(True)
        self.buffer.set_one_shot(True)
        self.drawn = (frame.up, sun.copy(), steps, layer)

    def wanted(
        self, up: np.ndarray, sun: np.ndarray, steps: int, layer: int, base: float
    ) -> bool:
        """Whether it must be drawn again for this camera, sun and layer."""
        if self.drawn is None:
            return True
        centre, drawn_sun, drawn_steps, drawn_layer = self.drawn
        moved = float(np.arccos(np.clip(up @ centre, -1.0, 1.0))) * base
        return (
            moved > RECENTRE * self.size * self.spacing
            or not np.array_equal(sun, drawn_sun)
            or steps != drawn_steps
            or layer != drawn_layer
        )

    def destroy(self, output: Any) -> None:
        output.get_engine().remove_window(self.buffer)
        self.root.remove_node()


class CloudShadowMaps:
    """A planet's clouds' shadow maps (see the module), for the nodes whose
    shaders shade by them (the planet's terrain: :meth:`bind`).

    :ivar size: the near map's texels per side (the far one's half, at
        least 256); 0 = none (the march per pixel)
    """

    def __init__(
        self, output: Any, clouds: Any, size: int, to_ram: bool = False
    ) -> None:
        """
        :param output: the window (its GSG draws the maps)
        :param clouds: the planet's clouds
            (:class:`~world_builder.clouds.sphere.SphericalClouds`)
        :param size: ``ShadowSettings.cloud_map_size`` (0: none)
        :param to_ram: copy each map to RAM when drawn (tests)
        """
        self._output = output
        self._clouds = clouds
        self._to_ram = to_ram
        self._on = PTA_int.empty_array(1)
        self._nodes: list[NodePath] = []
        self.near: _Map | None = None
        self.far: _Map | None = None
        self.size = 0
        self.set_size(size)

    def bind(self, node: NodePath) -> None:
        """Give a node's shaders the maps (off until drawn)."""
        self._nodes.append(node)
        self._bind(node)

    def _bind(self, node: NodePath) -> None:
        node.set_shader_input("cloudShadowMapOn", self._on)
        if self.near is not None:
            node.set_shader_inputs(
                cloudShadowNear=self.near.texture,
                cloudShadowFar=self.far.texture,
                cloudNearChart=self.near.chart,
                cloudFarChart=self.far.chart,
            )

    def set_size(self, size: int) -> None:
        """A new size (0: none): the maps made again."""
        if size == self.size:
            return
        self._destroy_maps()
        self.size = int(size)
        if self.size:
            far = max(self.size // 2, 256)
            self.near = _Map(
                self._output,
                self._clouds,
                "cloud_shadow_near",
                self.size,
                NEAR_SPACING,
                self._to_ram,
            )
            self.far = _Map(
                self._output,
                self._clouds,
                "cloud_shadow_far",
                far,
                FAR_SPACING,
                self._to_ram,
            )
        self._nodes = [node for node in self._nodes if not node.is_empty()]
        for node in self._nodes:
            self._bind(node)

    def update(self, camera_world: ArrayLike, sun: ArrayLike, steps: int) -> None:
        """Per frame, before the frame is drawn: each map drawn again if it
        must be (:meth:`_Map.wanted`).

        :param camera_world: the camera, world frame
        :param sun: toward the sun, world frame (unit)
        :param steps: ``ShadowSettings.cloud_shadow_steps``
        """
        clouds = self._clouds
        layer = int(clouds._shadow_layer[0])
        if self.near is None or layer < 0 or steps <= 0:
            self._on[0] = 0
            return
        relative = np.asarray(camera_world, dtype=np.float64) - clouds.position
        up = relative / np.linalg.norm(relative)
        sun = np.asarray(sun, dtype=np.float64)
        base = clouds.radius + clouds._deck_field.slab[0]
        for shadow_map in (self.near, self.far):
            if shadow_map.wanted(up, sun, steps, layer, base):
                shadow_map.draw(up, sun, steps, layer, base)
        self._on[0] = 1

    def _destroy_maps(self) -> None:
        for shadow_map in (self.near, self.far):
            if shadow_map is not None:
                shadow_map.destroy(self._output)
        self.near = self.far = None
        self._on[0] = 0

    def destroy(self) -> None:
        """Remove the maps' buffers; the nodes shade by the march again."""
        self._destroy_maps()
        self.size = 0


_SHADER: Shader | None = None


def _bake_shader() -> Shader:
    global _SHADER
    if _SHADER is None:
        _SHADER = Shader.load(
            Shader.SL_GLSL,
            vertex=SHADER_PATH / "cloud_shadow_map.vert",
            fragment=SHADER_PATH / "cloud_shadow_map.frag",
        )
    return _SHADER
