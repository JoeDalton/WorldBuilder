"""
The sky-view table (Hillaire 2020; RENDERING_PLAN.md B.3): the air between
the camera and space for every direction, baked on the GPU once a frame
while the camera is inside an atmosphere, read by the sky (``sky.frag``, the
sea's reflection with it) and the stars instead of marching every pixel and
vertex.

256 x 128 directions (``shaders/sky_view.glsl``: the camera's vertical and
the sun's azimuth, texels crowded toward the horizon and the sun), each
marched by the sky's own ``atmosphereAlong`` at its most samples (16): the
eclipses, the rings' shadow and the planet's shadow come with it, per texel
instead of per pixel. In-scatter in the lower half of an RGBA16F texture,
transmittance in the upper. From space (the camera in no atmosphere) the
table holds nothing and the readers march, as before.

Re-baked every frame (~65 thousand marches), rather than tracking what it
depends on (the camera's altitude, the sun's elevation, but also eclipses,
rings and the other atmosphere): with its bake it saves 0.15-0.43 ms a frame
against the sky's march at 2560 x 1440 (performance.md §25). A sun turning
in azimuth alone turns the table's frame. Not baked from space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from panda3d.core import (
    Camera,
    CardMaker,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    NodePath,
    OmniBoundingVolume,
    OrthographicLens,
    PTA_int,
    SamplerState,
    Shader,
    Texture,
    WindowProperties,
)

from world_builder.atmosphere import Atmospheres, bind_defaults

SHADER_PATH = Path(__file__).parent / "shaders"
#: The table's directions (sky_view.glsl's SKY_VIEW_SIZE): powers of two,
#: as Panda pads a render target's texture to them (Hillaire's is 192 x 108).
SKY_VIEW_SIZE = (256, 128)


def _fallback() -> Texture:
    tex = Texture("no_sky_view")
    tex.setup_2d_texture(1, 1, Texture.T_float, Texture.F_rgba16)
    tex.set_ram_image(b"\0" * 16)
    return tex


_DEFAULTS: dict | None = None


def bind_sky_view_defaults(node: NodePath) -> None:
    """Give a node including sky_view.glsl "no table" inputs (the readers
    march), until :meth:`SkyView.bind` overrides them."""
    global _DEFAULTS
    if _DEFAULTS is None:
        off = PTA_int.empty_array(1)
        off[0] = 0
        _DEFAULTS = {"uSkyView": _fallback(), "uSkyViewOn": off}
    for name, value in _DEFAULTS.items():
        node.set_shader_input(name, value)


class SkyView:
    """The sky-view table and its bake (see the module).

    :ivar texture: the table, read as ``uSkyView``
    :ivar root: the bake's scene: give it the shadows' inputs
        (``Shadows.bind(view.root)``) as the sky has them
    """

    def __init__(self, output: Any, atmospheres: Atmospheres, sun: Any) -> None:
        """
        :param output: the window (its GSG draws the table)
        :param atmospheres: the world's atmospheres (bound to the bake)
        :param sun: the sky's ``uSunDir`` input (a PTA, shared live)
        """
        self._output = output
        self._atmospheres = atmospheres
        width, height = SKY_VIEW_SIZE
        self.texture = Texture("sky_view")
        self.texture.setup_2d_texture(
            width, 2 * height, Texture.T_half_float, Texture.F_rgba16
        )
        self.texture.set_magfilter(SamplerState.FT_linear)
        self.texture.set_minfilter(SamplerState.FT_linear)
        self.texture.set_wrap_u(SamplerState.WM_repeat)  # around the azimuth
        self.texture.set_wrap_v(SamplerState.WM_clamp)
        props = FrameBufferProperties()
        props.set_rgba_bits(16, 16, 16, 16)
        props.set_float_color(True)
        engine = output.get_engine()
        self.buffer = engine.make_output(
            output.get_pipe(),
            "sky_view_bake",
            -400,  # before every camera that reads it
            props,
            WindowProperties.size(width, 2 * height),
            # Its own framebuffer: a parasite of a smaller window (the
            # tests' 64 x 64) would be clipped to it.
            GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_refuse_parasite,
            output.get_gsg(),
            output,
        )
        if self.buffer is None:
            raise RuntimeError("no offscreen buffer for the sky-view table")
        self.buffer.add_render_texture(
            self.texture, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_color
        )
        self.buffer.set_clear_color_active(False)
        self.root = NodePath("sky_view_bake")
        self.root.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "sky_view_bake.vert",
                fragment=SHADER_PATH / "sky_view_bake.frag",
            )
        )
        bind_defaults(self.root)
        from world_builder.shadows import bind_defaults as bind_shadow_defaults

        bind_shadow_defaults(self.root)
        atmospheres.bind(self.root)
        atmospheres.god_ray_root(self.root)  # the sky's shafts too
        self.root.set_shader_input("uSunDir", sun)
        maker = CardMaker("sky_view")
        maker.set_frame(-1.0, 1.0, -1.0, 1.0)
        card = self.root.attach_new_node(maker.generate())
        card.node().set_bounds(OmniBoundingVolume())
        card.node().set_final(True)
        card.set_two_sided(True)
        card.set_depth_test(False)
        card.set_depth_write(False)
        camera = self.root.attach_new_node(Camera("sky_view", OrthographicLens()))
        region = self.buffer.make_display_region()
        region.set_camera(camera)
        self._on = PTA_int.empty_array(1)
        self._on[0] = 1

    def bind(self, node: NodePath) -> None:
        """Give a reader (the sky) the table."""
        node.set_shader_input("uSkyView", self.texture)
        node.set_shader_input("uSkyViewOn", self._on)

    def set_active(self, active: bool) -> None:
        """Bake (and read) the table, or not (the readers march)."""
        self._on[0] = int(bool(active))
        self.update()

    def update(self) -> None:
        """Per frame, after the atmospheres': bake only with the camera in
        an atmosphere (from space the readers march: no bake)."""
        self.buffer.set_active(
            bool(self._on[0]) and self._atmospheres.camera_inside() >= 0
        )

    @property
    def active(self) -> bool:
        return bool(self._on[0])

    def destroy(self) -> None:
        self._output.get_engine().remove_window(self.buffer)
        self.root.remove_node()
