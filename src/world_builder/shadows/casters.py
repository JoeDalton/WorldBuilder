"""
The shadows objects cast (part H of the shadows' plan): a sun-space depth
map of the game's casting models, read by the terrain, the sea and the
models themselves (``shadows.glsl``'s ``casterShadow``).

- **The casters**: the nodes tracked with ``cast=True``
  (``World.track_light``). ``render`` is hidden from the map's camera bit
  (:data:`~world_builder.depth.CASTER_BIT`), each caster shows through it:
  the map's pass culls and draws them alone (Panda prunes the rest of the
  scene by its draw masks).
- **The map** is an orthographic depth buffer looking down the sun's rays,
  its lens fitted every frame to the casters' bounding spheres, not to the
  view: a ship's shadow lands wherever it falls, on the ground under it or
  kilometres away. Only depth is drawn, with a stage of its own over the
  casters' (their shaders, textures and lights are skipped), and their back
  faces (the faces turned to the sun stay clear of their own shadow; what
  shades the back faces does not show: the sun does not light them).
- **A receiver** maps its position (from the camera, render frame) through
  ``uCasterMatrix`` to the map's texture and depth, pushed out along its
  normal by a texel or so as it turns from the sun, and takes 3x3 of the
  hardware's depth comparisons (bilinear each). Outside the map's square,
  or nearer the sun than it, nothing.

- **Out of sight.** A caster's shadow lies in the cylinder its sphere
  sweeps down the sun's rays: nothing it shades is nearer the camera than
  that. A caster whose cylinder is so far that its shadow would be under
  :data:`MIN_SHADOW_PIXELS` across is left out, and with none left the map
  is not drawn (a ship left far behind costs nothing).

One map for every caster: two ships far apart share its texels (their
bounding square spans both).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BitMask32,
    Camera,
    ColorWriteAttrib,
    CullFaceAttrib,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    LMatrix4f,
    LVecBase4f,
    NodePath,
    OrthographicLens,
    Point3,
    PTA_int,
    PTA_LMatrix4f,
    PTA_LVecBase4f,
    RenderState,
    SamplerState,
    Shader,
    ShaderAttrib,
    Texture,
    TextureAttrib,
    Vec3,
    WindowProperties,
)

from world_builder.depth import CASTER_BIT

#: The map's camera's mask: its bit alone.
CASTER_CAMERA_MASK = BitMask32.bit(CASTER_BIT)
#: A receiver is pushed out along its normal by this many of the map's texels
#: (times 1 - N.L: none facing the sun), and its depth pulled toward the sun
#: by this many texels' worth: no acne, no gap at a caster's foot.
NORMAL_OFFSET_TEXELS = 1.5
DEPTH_BIAS_TEXELS = 1.0
#: The map's square around the casters, and its depth range, are this much
#: wider than their spheres (metres): a sphere's edge stays inside it.
MARGIN = 1.0
#: A caster whose shadow would be narrower than this, in pixels, wherever it
#: falls, is not drawn.
MIN_SHADOW_PIXELS = 2.0

_DEPTH_VERTEX = """#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""
_DEPTH_FRAGMENT = """#version 330
void main() {
}
"""


class CasterMap:
    """The casters' shadow map (see the module). Bind the receivers' nodes
    (:meth:`bind`; they read nothing until it is drawn), then
    :meth:`update` every frame with the casters.

    :ivar size: texels per side (0: none)
    :ivar casters: the nodes drawn into it at the last update
    """

    def __init__(
        self, output: Any, render: NodePath, size: int, to_ram: bool = False
    ) -> None:
        """
        :param output: the window (its GSG draws the map)
        :param render: the scene's root: hidden from the map's camera but
            for the casters
        :param size: ``ShadowSettings.caster_map_size`` (0: none)
        :param to_ram: copy the map to RAM when drawn (tests)
        """
        self._output = output
        self._render = render
        self._to_ram = to_ram
        render.hide(CASTER_CAMERA_MASK)
        self._matrix = PTA_LMatrix4f.empty_array(1)
        self._matrix[0] = LMatrix4f.ident_mat()
        self._on = PTA_int.empty_array(1)
        self._bias = PTA_LVecBase4f.empty_array(1)
        self.texture = Texture("caster_map")
        self.texture.setup_2d_texture(
            1, 1, Texture.T_float, Texture.F_depth_component32
        )
        self.texture.set_ram_image(np.float32(1.0).tobytes())
        self._set_sampling(self.texture)
        lens = OrthographicLens()
        self.camera = render.attach_new_node(Camera("caster_map", lens))
        self.camera.node().set_camera_mask(CASTER_CAMERA_MASK)
        self.camera.node().set_initial_state(
            RenderState.make(
                ShaderAttrib.make(_depth_shader()),
                ColorWriteAttrib.make(ColorWriteAttrib.C_off),
                CullFaceAttrib.make_reverse(),
                TextureAttrib.make_off(),
                1000,  # over the casters' own
            )
        )
        self.buffer = None
        self.casters: list[NodePath] = []
        self._nodes: list[NodePath] = []
        self.size = 0
        self.set_size(size)

    @staticmethod
    def _set_sampling(texture: Texture) -> None:
        """The hardware's depth comparison, bilinear, clamped."""
        texture.set_minfilter(SamplerState.FT_shadow)
        texture.set_magfilter(SamplerState.FT_shadow)
        texture.set_wrap_u(SamplerState.WM_clamp)
        texture.set_wrap_v(SamplerState.WM_clamp)

    def bind(self, node: NodePath) -> None:
        """Give a receiver's shaders the map (shadows.glsl's
        ``casterShadow``; off until drawn)."""
        node.set_shader_inputs(
            uCasterMap=self.texture,
            uCasterOn=self._on,
            uCasterMatrix=self._matrix,
            uCasterBias=self._bias,
        )
        self._nodes.append(node)

    def set_size(self, size: int) -> None:
        """A new size (0: none): the buffer made again."""
        size = int(size)
        if size == self.size:
            return
        self._remove_buffer()
        self.size = size
        if size:
            self._make_buffer(size)

    def _make_buffer(self, size: int) -> None:
        props = FrameBufferProperties()
        props.set_rgb_color(False)
        props.set_depth_bits(32)
        props.set_float_depth(True)
        engine = self._output.get_engine()
        self.buffer = engine.make_output(
            self._output.get_pipe(),
            "caster_map",
            -400,  # before every camera that reads it
            props,
            WindowProperties.size(size, size),
            GraphicsPipe.BF_refuse_window,
            self._output.get_gsg(),
            self._output,
        )
        if self.buffer is None:
            self.size = 0
            return
        # A depth texture of its own (the default's is 1 texel).
        self.texture = Texture("caster_map")
        self._set_sampling(self.texture)
        mode = GraphicsOutput.RTM_bind_or_copy
        if self._to_ram:
            mode = GraphicsOutput.RTM_copy_ram
        self.buffer.add_render_texture(self.texture, mode, GraphicsOutput.RTP_depth)
        self.buffer.set_clear_depth_active(True)
        self.buffer.set_clear_depth(1.0)
        self.buffer.set_clear_color_active(False)
        self.buffer.make_display_region().set_camera(self.camera)
        self.buffer.set_active(False)
        self._nodes = [node for node in self._nodes if not node.is_empty()]
        for node in self._nodes:
            node.set_shader_input("uCasterMap", self.texture)

    def _remove_buffer(self) -> None:
        if self.buffer is not None:
            self._output.get_engine().remove_window(self.buffer)
            self.buffer = None
        self._on[0] = 0

    def update(
        self,
        casters: Sequence[NodePath],
        sun_direction: ArrayLike,
        camera_render: ArrayLike,
        pixel_angle: float | None = None,
    ) -> None:
        """Per frame, before the frame is drawn: fit the map to the casters
        in sight and draw it this frame, or turn it off (none, no map).

        :param casters: the casting nodes
        :param sun_direction: toward the sun, render frame
        :param camera_render: the main camera, render frame (the receivers'
            positions are relative to it)
        :param pixel_angle: the angle a pixel of the view spans, radians: a
            caster whose shadow is under :data:`MIN_SHADOW_PIXELS` wherever
            it falls is left out; None = none is
        """
        sun = np.asarray(sun_direction, dtype=np.float64)
        sun = sun / np.linalg.norm(sun)
        camera = np.asarray(camera_render, dtype=np.float64)
        drawn, spheres = [], []
        for node in casters:
            if node.is_empty():
                continue
            sphere = _sphere(node, self._render)
            if sphere is None or not _in_sight(sphere, sun, camera, pixel_angle):
                continue
            drawn.append(node)
            spheres.append(sphere)
        self._show(drawn)
        if self.buffer is None or not spheres:
            self._on[0] = 0
            if self.buffer is not None:
                self.buffer.set_active(False)
            return
        # Looking down the sun's rays, from the origin for now: its axes.
        hint = Vec3(0.0, 0.0, 1.0) if abs(sun[2]) < 0.9 else Vec3(1.0, 0.0, 0.0)
        self.camera.set_pos(0.0, 0.0, 0.0)
        self.camera.look_at(Point3(*(-sun)), hint)
        quat = self.camera.get_quat(self._render)
        right = np.array(quat.get_right())
        forward = np.array(quat.get_forward())
        up = np.array(quat.get_up())
        centres = np.array([centre for centre, _ in spheres])
        radii = np.array([radius for _, radius in spheres])
        across = centres @ right, centres @ up
        low = [float(np.min(c - radii)) - MARGIN for c in across]
        high = [float(np.max(c + radii)) + MARGIN for c in across]
        depth = centres @ forward
        nearest = float(np.min(depth - radii)) - MARGIN
        farthest = float(np.max(depth + radii)) + MARGIN
        side = max(high[0] - low[0], high[1] - low[1])
        middle = 0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])
        eye = right * middle[0] + up * middle[1] + forward * nearest
        self.camera.set_pos(*eye)
        lens = self.camera.node().get_lens()
        lens.set_film_size(side, side)
        span = farthest - nearest
        lens.set_near_far(0.0, span)
        # From the camera (render frame) to the map: texture u, v and depth.
        offset = camera - eye
        rows = np.array(
            [
                [*(right / side), offset @ right / side + 0.5],
                [*(up / side), offset @ up / side + 0.5],
                [*(forward / span), offset @ forward / span],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Panda's rows are GLSL's columns.
        self._matrix[0] = LMatrix4f(*rows.T.ravel())
        texel = side / self.size
        self._bias[0] = LVecBase4f(
            NORMAL_OFFSET_TEXELS * texel,
            DEPTH_BIAS_TEXELS * texel / span,
            1.0 / self.size,
            0.0,
        )
        self.buffer.set_active(True)
        self._on[0] = 1

    def _show(self, casters: list[NodePath]) -> None:
        """Show the casters through render's hiding, and no others."""
        for node in self.casters:
            if not node.is_empty() and node not in casters:
                node.show(CASTER_CAMERA_MASK)
        for node in casters:
            if node not in self.casters:
                node.show_through(CASTER_CAMERA_MASK)
        self.casters = casters

    def destroy(self) -> None:
        """Remove the buffer and the camera; the receivers read nothing."""
        self._show([])
        self._remove_buffer()
        self.size = 0
        self.camera.remove_node()
        self._render.show(CASTER_CAMERA_MASK)


def _in_sight(
    sphere: tuple[np.ndarray, float],
    sun: np.ndarray,
    camera: np.ndarray,
    pixel_angle: float | None,
) -> bool:
    """Whether a caster's shadow can be MIN_SHADOW_PIXELS across or more
    from the camera: the nearest point of the cylinder its sphere sweeps
    down the sun's rays is near enough."""
    if pixel_angle is None:
        return True
    centre, radius = sphere
    along = max(float((centre - camera) @ sun), 0.0)
    nearest = float(np.linalg.norm(camera - (centre - sun * along))) - radius
    return nearest * pixel_angle * MIN_SHADOW_PIXELS <= 2.0 * radius


def _sphere(node: NodePath, render: NodePath) -> tuple[np.ndarray, float] | None:
    """A node's bounding sphere in render's frame, or None (empty)."""
    bounds = node.get_bounds()
    if bounds.is_empty() or bounds.is_infinite():
        return None
    # In its parent's frame (the node's own transform is in it).
    bounds = bounds.make_copy()
    parent = node.get_parent()
    if not parent.is_empty():
        bounds.xform(parent.get_mat(render))
    if hasattr(bounds, "get_radius"):
        centre = np.array(bounds.get_center(), dtype=np.float64)
        return centre, float(bounds.get_radius())
    # A box: the sphere around it.
    low, high = np.array(bounds.get_min()), np.array(bounds.get_max())
    return (low + high) / 2.0, float(np.linalg.norm(high - low)) / 2.0


_SHADER: Shader | None = None


def _depth_shader() -> Shader:
    global _SHADER
    if _SHADER is None:
        _SHADER = Shader.make(Shader.SL_GLSL, _DEPTH_VERTEX, _DEPTH_FRAGMENT)
    return _SHADER
