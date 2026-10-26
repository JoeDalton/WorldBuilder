"""
Half-resolution clouds: the clouds drawn into buffers of half the view's
size, then laid over the scene (``CloudSettings.half_resolution``).

The clouds' GPU cost is mostly per pixel (2-3 ms of their 3.6-4.6 ms at
1600x900 low over the planet goes with a quarter of the pixels), and it is
what makes the slowest frames in flight: the view holds more cloud. A
quarter of the pixels keeps most of that, for softer cloud edges against
the ground.

How:

- **One layer per depth range** of
  :class:`~world_builder.depth.SplitDepthCameras`, each an offscreen buffer
  half the view's size (``scale``) with its own depth, drawn by a camera
  that shares the split camera's very lens (so it follows its range, fov
  and aspect every frame) and hangs under the app's camera.
- **Occlusion.** Everything but the clouds is drawn there into depth only:
  the layer cameras' initial state turns colour writes off, and the clouds'
  roots turn them back on. The terrain swaps in a depth-only fragment stage
  (:func:`world_builder.terrain.terrain.depth_only_shader`, through the
  ``PASS_TAG`` tag state), so its depth costs little. A game's own models
  occlude the clouds too. The sky is kept out (:meth:`exclude`): it hides
  nothing and shades every pixel.
- **The clouds** are hidden from the main split cameras and drawn by the
  layer cameras instead; the sea's reflection camera still draws them.
- **The composite**: each layer is laid over its own range's pass by a
  fullscreen card in the clouds' slot (after the sky, before the
  transparent bin), premultiplied "over", upsampled bilinearly. The far
  layer goes on in the far camera's pass, so the near pass's ground and
  models cover it as they cover everything far; the near layer goes on in
  the near pass. (One buffer for both ranges would show the far clouds
  through near ground: depth-only near ground cannot hide what the far
  range already put in its colour.)

Limits: cloud edges against the ground's silhouettes are resolved at half
resolution (a soft fringe up to a pixel wide).
"""

from __future__ import annotations

import math
from typing import Any

from panda3d.core import (
    BitMask32,
    Camera,
    CardMaker,
    ColorBlendAttrib,
    ColorWriteAttrib,
    CullBinManager,
    FrameBufferProperties,
    Lens,
    LVecBase2f,
    NodePath,
    OmniBoundingVolume,
    RenderState,
    SamplerState,
    Shader,
    ShaderAttrib,
    Texture,
    TransparencyAttrib,
)

from world_builder.depth import (
    CLOUD_FAR_BIT,
    CLOUD_NEAR_BIT,
    FAR_BIT,
    HIDE_FROM_CLOUD_PASS,
    HIDE_FROM_FAR,
    HIDE_FROM_NEAR,
    NEAR_BIT,
    PASS_TAG,
    SplitDepthCameras,
)
from world_builder.terrain.terrain import depth_only_shader

#: The composites' cull bin: after the sky (21), where the clouds draw
#: without this pass (22-29), before the transparent bin (30).
LAYER_BIN = "cloud_layer"
LAYER_BIN_SORT = 22

_VERTEX = """
#version 330
in vec4 p3d_Vertex;
uniform vec2 uScale;
out vec2 vUv;
void main() {
    // The card's x and z span -1..1: clip space directly, whatever the lens.
    vUv = (p3d_Vertex.xz * 0.5 + 0.5) * uScale;
    gl_Position = vec4(p3d_Vertex.x, p3d_Vertex.z, 0.0, 1.0);
}
"""

_FRAGMENT = """
#version 330
uniform sampler2D uClouds;
in vec2 vUv;
out vec4 fragColor;
void main() {
    fragColor = texture(uClouds, vUv);  // premultiplied
}
"""


class _Layer:
    """One depth range's clouds: its buffer, camera and composite card."""

    def __init__(
        self,
        app: Any,
        output: Any,
        name: str,
        lens: Lens,
        bit: int,
        shown_to: int,
        size: tuple[int, int],
        sort: int,
        initial: RenderState,
        terrain: RenderState,
    ) -> None:
        self.texture = Texture(name)
        self.texture.set_wrap_u(SamplerState.WM_clamp)
        self.texture.set_wrap_v(SamplerState.WM_clamp)
        self.texture.set_minfilter(SamplerState.FT_linear)
        self.texture.set_magfilter(SamplerState.FT_linear)
        props = FrameBufferProperties()
        props.set_rgba_bits(8, 8, 8, 8)
        props.set_depth_bits(24)
        self.buffer = output.make_texture_buffer(
            name, *size, self.texture, False, props
        )
        self.buffer.set_sort(sort)
        self.buffer.set_clear_color_active(True)
        self.buffer.set_clear_color((0.0, 0.0, 0.0, 0.0))
        self.buffer.set_clear_depth_active(True)
        node = Camera(name, lens)
        node.set_camera_mask(BitMask32.bit(bit))
        node.set_initial_state(initial)
        node.set_tag_state_key(PASS_TAG)
        node.set_tag_state("terrain", terrain)
        self.camera = app.camera.attach_new_node(node)
        self.buffer.make_display_region().set_camera(self.camera)

        maker = CardMaker(name)
        maker.set_frame(-1.0, 1.0, -1.0, 1.0)
        self.card = app.camera.attach_new_node(maker.generate())
        self.card.node().set_bounds(OmniBoundingVolume())
        self.card.node().set_final(True)
        self.card.set_shader(_composite_shader())
        self.card.set_shader_input("uClouds", self.texture)
        self.uv_scale = LVecBase2f(1.0, 1.0)
        self.card.set_shader_input("uScale", self.uv_scale)
        self.card.set_bin(LAYER_BIN, 0)
        self.card.set_depth_test(False)
        self.card.set_depth_write(False)
        self.card.set_transparency(TransparencyAttrib.M_none)
        self.card.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_one,
                ColorBlendAttrib.O_one_minus_incoming_alpha,
            )
        )
        self.card.set_two_sided(True)
        # Only its own range's main camera lays it on.
        self.card.hide(BitMask32.all_on() & ~BitMask32.bit(shown_to))

    def update(self, size: tuple[int, int], resized: bool) -> None:
        if resized:
            self.buffer.set_size(*size)
        tx, ty = self.texture.get_x_size(), self.texture.get_y_size()
        if tx and ty:
            scale = LVecBase2f(size[0] / tx, size[1] / ty)
            if scale != self.uv_scale:
                self.uv_scale = scale
                self.card.set_shader_input("uScale", scale)

    def destroy(self, app: Any) -> None:
        self.card.remove_node()
        self.camera.remove_node()
        app.graphicsEngine.remove_window(self.buffer)


class HalfResClouds:
    """The clouds at a fraction of the view's resolution (see the module).

    Build it after the :class:`SplitDepthCameras` (and the window or the
    game's post-process buffer they draw into); :meth:`add` each cloud
    object's root, :meth:`exclude` the sky, call :meth:`update` every frame
    after ``SplitDepthCameras.update``, and feed the clouds a pixel angle
    times :attr:`pixel_scale`.

    :ivar far: the far range's layer (``buffer``, ``texture``, ``camera``,
        ``card``)
    :ivar near: the near range's
    """

    def __init__(
        self, app: Any, depth: SplitDepthCameras, scale: float = 0.5, sort: int = -50
    ) -> None:
        """
        :param app: the ShowBase (``camera``, ``camLens``, ``graphicsEngine``)
        :param depth: the split cameras whose views the layers mirror
        :param scale: the layers' size over the view's (0.5: a quarter of
            the pixels)
        :param sort: the buffers' sort: before the view they are laid over
        """
        self.app = app
        self.depth = depth
        self.scale = scale
        self._nodes: list[NodePath] = []
        self._size = self._view_size()
        # Everything writes depth only, but the clouds (add) and the terrain,
        # which swaps in a depth-only fragment stage.
        initial = RenderState.make(ColorWriteAttrib.make(ColorWriteAttrib.C_off))
        terrain = RenderState.make(
            ShaderAttrib.make(depth_only_shader()),
            ColorWriteAttrib.make(ColorWriteAttrib.C_off),
            1,  # over the sea's own shader below the terrain's root
        )
        _ensure_layer_bin()
        self.far = _Layer(
            app,
            depth.output,
            "cloud_layer_far",
            depth.far_lens,
            CLOUD_FAR_BIT,
            FAR_BIT,
            self._size,
            sort,
            initial,
            terrain,
        )
        self.near = _Layer(
            app,
            depth.output,
            "cloud_layer_near",
            app.camLens,
            CLOUD_NEAR_BIT,
            NEAR_BIT,
            self._size,
            sort + 1,
            initial,
            terrain,
        )

    @property
    def pixel_scale(self) -> float:
        """What a pixel of the layers spans over a pixel of the view: the
        clouds' pixel angle is the view's times this."""
        return 1.0 / self.scale

    def add(self, root: NodePath) -> None:
        """Draw these clouds (a cloud object's root) in the layers instead of
        the main split cameras (the sea's reflection still draws them)."""
        root.hide(HIDE_FROM_NEAR | HIDE_FROM_FAR)
        root.set_attrib(ColorWriteAttrib.make(ColorWriteAttrib.C_all))
        self._nodes.append(root)

    def remove(self, root: NodePath) -> None:
        """Give clouds back to the main split cameras."""
        if root in self._nodes:
            self._nodes.remove(root)
            if not root.is_empty():
                root.show(HIDE_FROM_NEAR | HIDE_FROM_FAR)
                root.clear_attrib(ColorWriteAttrib)

    @staticmethod
    def exclude(node: NodePath) -> None:
        """Keep a node out of the layers (the sky: it hides no cloud)."""
        node.hide(HIDE_FROM_CLOUD_PASS)

    def update(self) -> None:
        """Per frame, after ``SplitDepthCameras.update``: the far camera's
        cull bounds (its true far plane) and the view's size."""
        far_bounds = self.depth.far_camera.node().get_cull_bounds()
        if far_bounds is not None:
            self.far.camera.node().set_cull_bounds(far_bounds)
        size = self._view_size()
        resized = size != self._size
        self._size = size
        self.far.update(size, resized)
        self.near.update(size, resized)

    def destroy(self) -> None:
        """Give every cloud back to the main cameras and remove the layers."""
        for root in list(self._nodes):
            self.remove(root)
        self.far.destroy(self.app)
        self.near.destroy(self.app)

    def _view_size(self) -> tuple[int, int]:
        region = self.depth.near_region
        width = max(region.get_pixel_width(), 1)
        height = max(region.get_pixel_height(), 1)
        return (
            max(1, math.ceil(width * self.scale)),
            max(1, math.ceil(height * self.scale)),
        )


_SHADER: list[Shader] = []


def _composite_shader() -> Shader:
    if not _SHADER:
        _SHADER.append(Shader.make(Shader.SL_GLSL, vertex=_VERTEX, fragment=_FRAGMENT))
    return _SHADER[0]


def _ensure_layer_bin() -> None:
    manager = CullBinManager.get_global_ptr()
    if manager.find_bin(LAYER_BIN) == -1:
        manager.add_bin(LAYER_BIN, CullBinManager.BT_fixed, LAYER_BIN_SORT)
