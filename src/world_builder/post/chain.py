"""
The passes from a drawn view to its window region (:class:`PostChain`):

========================  ==============================================
upscaler                  passes
========================  ==============================================
bilinear                  view -> window region: a copy, or FXAA, with
                          the GPU's bilinear stretch
FSR 1                     view [-> FXAA, view size] -> EASU (output
                          size) -> RCAS -> window region
========================  ==============================================

EASU and RCAS read texels by index (``texelFetch``), so the textures'
padding to powers of two does not touch them; the copy and FXAA sample in
the padded texture's coordinates (``texScale``, refreshed by
:meth:`PostChain.update`).

The view should be in display values (gamma-encoded LDR): FSR 1's
edge detection and FXAA's luma assume them. WorldBuilder's shaders tone map
and encode in the shader, into 8-bit buffers.
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
    OrthographicLens,
    SamplerState,
    Shader,
    Texture,
    WindowProperties,
)

from world_builder.post.settings import RenderSettings

SHADER_PATH = Path(__file__).parent / "shaders"


def _shader(fragment: str) -> Shader:
    return Shader.load(
        Shader.SL_GLSL,
        vertex=str(SHADER_PATH / "composite.vert"),
        fragment=str(SHADER_PATH / fragment),
    )


def target_texture(name: str) -> Texture:
    """:returns: a render target that a pass may read by index or sample:
    no mipmaps (a texture with none is complete for ``texelFetch``),
    bilinear"""
    texture = Texture(name)
    texture.set_minfilter(SamplerState.FT_linear)
    texture.set_magfilter(SamplerState.FT_linear)
    texture.set_wrap_u(SamplerState.WM_clamp)
    texture.set_wrap_v(SamplerState.WM_clamp)
    return texture


class _Quad:
    """A full-screen card under an orthographic camera of its own."""

    def __init__(self, name: str, shader: Shader) -> None:
        self.root = NodePath(name)
        lens = OrthographicLens()
        lens.set_film_size(2.0, 2.0)
        lens.set_near_far(-1.0, 1.0)
        self.camera = self.root.attach_new_node(Camera(name, lens))
        maker = CardMaker(name)
        maker.set_frame_fullscreen_quad()
        self.card = self.root.attach_new_node(maker.generate())
        self.card.set_depth_test(False)
        self.card.set_depth_write(False)
        self.card.set_shader(shader)


class _Pass:
    """A quad drawn into a buffer of its own, its colour into a texture."""

    def __init__(
        self,
        engine: Any,
        window: Any,
        name: str,
        size: tuple[int, int],
        sort: int,
        quad: _Quad,
    ) -> None:
        self.texture = target_texture(name)
        props = FrameBufferProperties()
        props.set_rgba_bits(8, 8, 8, 8)
        props.set_depth_bits(0)
        self.buffer = engine.make_output(
            window.get_pipe(),
            name,
            sort,
            props,
            WindowProperties.size(*size),
            GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_resizeable,
            window.get_gsg(),
            window,
        )
        if self.buffer is None:
            raise RuntimeError(f"no buffer could be made for the {name} pass")
        self.buffer.add_render_texture(
            self.texture, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_color
        )
        self.buffer.set_clear_color_active(False)
        self.quad = quad
        region = self.buffer.make_display_region()
        region.disable_clears()
        region.set_camera(self.quad.camera)
        self.engine = engine

    def resize(self, size: tuple[int, int]) -> None:
        if size != (self.buffer.get_x_size(), self.buffer.get_y_size()):
            self.buffer.set_size(*size)

    def destroy(self) -> None:
        self.buffer.clear_render_textures()
        self.engine.remove_window(self.buffer)
        self.quad.root.remove_node()


class PostChain:
    """The passes from a view's texture to a window region (see the module).

    The view is drawn elsewhere (:class:`~world_builder.depth.ReversedDepthCamera`'s
    scene buffer, or a game's), into *source*, at :meth:`RenderSettings.render_size`
    of the region's size. The chain's last pass draws into a region of
    *window* of its own (:attr:`region`), at *dimensions* and *sort*: where
    the game's 3D region was.

    Or, given an *output_size*, the chain ends in a texture of that size
    instead (:attr:`texture`): a view shown on a card, e.g. a game's
    rear-view mirror drawn small and upscaled to the card's size on screen.
    The card shows it texel for texel (with the texture's
    ``get_tex_scale()``, should the GSG pad it).

    :ivar region: the window's region the chain ends in (None when it ends
        in :attr:`texture`)
    :ivar texture: the texture it ends in (None when it ends in a region)
    :ivar settings: what it was built for (:meth:`set_sharpness` changes
        the one live knob)
    :ivar render_size: the view's size, pixels
    :ivar output_size: the region's (or :attr:`texture`'s) size, pixels
    """

    def __init__(
        self,
        engine: Any,
        window: Any,
        source: Texture,
        settings: RenderSettings,
        render_size: tuple[int, int],
        dimensions: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        sort: int = 0,
        buffer_sort: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        """
        :param engine: the GraphicsEngine
        :param window: the window the chain ends in (its GSG draws the passes)
        :param source: the view's colour
        :param settings: what to build
        :param render_size: the view's size in *source*, pixels
        :param dimensions: the region's (left, right, bottom, top) in the
            window, 0-1
        :param sort: the region's sort in the window
        :param buffer_sort: the passes' first sort (up to 3 buffers, one
            sort each): after the view's buffer, before the window or what
            shows the texture; None = the window's - 2
        :param output_size: end in a texture of this size, pixels, instead of
            a window region (*dimensions* and *sort* unused)
        """
        self.engine = engine
        self.window = window
        self.source = source
        self.settings = settings
        self.render_size = render_size
        self._fixed_size = output_size
        self.region = None
        self.texture: Texture | None = None
        self._out: _Pass | None = None
        if output_size is None:
            self.region = window.make_display_region(*dimensions)
            self.region.set_sort(sort)
            self.region.disable_clears()
        self.output_size = self._target_size()
        first_sort = window.get_sort() - 2 if buffer_sort is None else buffer_sort
        self._passes: list[_Pass] = []
        # Cards sampling a possibly padded texture by coordinates: (card, its
        # texture), their texScale and rcpFrame refreshed by update().
        self._sampled: list[tuple[NodePath, Texture]] = []
        self.easu: NodePath | None = None
        self.fxaa = settings.fxaa_active
        if settings.fsr:
            self._final = self._build_fsr(first_sort)
        else:
            self._final = _Quad(
                "post_present", _shader("fxaa.frag" if self.fxaa else "blit.frag")
            )
            self._sample(self._final.card, source)
        if self.region is not None:
            self.region.set_camera(self._final.camera)
        else:
            self._out = _Pass(
                engine,
                window,
                "post_out",
                self.output_size,
                first_sort + 2,
                self._final,
            )
            self.texture = self._out.texture
        self._set_size_inputs()
        self.update()

    def _build_fsr(self, sort: int) -> _Quad:
        """View [-> FXAA] -> EASU -> RCAS (the region's quad)."""
        easu_input = self.source
        if self.fxaa:
            fxaa = _Pass(
                self.engine,
                self.window,
                "post_fxaa",
                self.render_size,
                sort,
                _Quad("post_fxaa", _shader("fxaa.frag")),
            )
            self._sample(fxaa.quad.card, self.source)
            self._passes.append(fxaa)
            easu_input = fxaa.texture
        easu = _Pass(
            self.engine,
            self.window,
            "post_easu",
            self.output_size,
            sort + 1,
            _Quad("post_easu", _shader("fsr_easu.frag")),
        )
        easu.quad.card.set_shader_input("inputTex", easu_input)
        self._passes.append(easu)
        self.easu = easu.quad.card
        rcas = _Quad("post_rcas", _shader("fsr_rcas.frag"))
        rcas.card.set_shader_input("inputTex", easu.texture)
        rcas.card.set_shader_input("sharpness", self.settings.sharpness)
        return rcas

    def _sample(self, card: NodePath, texture: Texture) -> None:
        card.set_shader_input("sceneTex", texture)
        self._sampled.append((card, texture))

    def _target_size(self) -> tuple[int, int]:
        if self.region is None:
            return self._fixed_size
        return (
            max(1, self.region.get_pixel_width()),
            max(1, self.region.get_pixel_height()),
        )

    def _set_size_inputs(self) -> None:
        if self.easu is not None:
            self.easu.set_shader_input("inputSize", self.render_size)
            self.easu.set_shader_input("outputSize", self.output_size)
        if self.settings.fsr:
            # The region's lower-left pixel, as Panda rounds it.
            left = bottom = 0
            if self.region is not None:
                width, height = self.window.get_x_size(), self.window.get_y_size()
                left = int(self.region.get_left() * width + 0.5)
                bottom = int(self.region.get_bottom() * height + 0.5)
            card = self._final.card
            card.set_shader_input("inputSize", self.output_size)
            card.set_shader_input("outputOrigin", (float(left), float(bottom)))

    @property
    def passes(self) -> int:
        """The intermediate buffers' count (FSR 1: EASU, and FXAA)."""
        return len(self._passes)

    def buffers(self) -> list[Any]:
        """:returns: the intermediate passes' buffers, in drawing order"""
        return [p.buffer for p in self._passes]

    def textures(self) -> list[Texture]:
        """:returns: the intermediate passes' textures, in drawing order"""
        return [p.texture for p in self._passes]

    def resize(
        self, render_size: tuple[int, int], output_size: tuple[int, int] | None = None
    ) -> None:
        """Follow a new view size, and the region's (the window's) size, or
        the texture's new *output_size*."""
        self.render_size = render_size
        if output_size is not None and self.region is None:
            self._fixed_size = output_size
        self.output_size = self._target_size()
        if self._out is not None:
            self._out.resize(self.output_size)
        if self.fxaa and self.settings.fsr:
            self._passes[0].resize(render_size)
        if self.settings.fsr:
            self._passes[-1].resize(self.output_size)
        self._set_size_inputs()

    def update(self) -> None:
        """Per frame: the sampled textures' padding (only known once the GSG
        has made them)."""
        width, height = self.render_size
        for card, texture in self._sampled:
            sx, sy = texture.get_tex_scale()
            card.set_shader_input("texScale", (sx, sy))
            card.set_shader_input("rcpFrame", (sx / width, sy / height))

    def set_active(self, active: bool) -> None:
        """Draw the passes, or not (a hidden mirror's): every buffer, and the
        window's region."""
        for p in self._passes:
            p.buffer.set_active(active)
        if self._out is not None:
            self._out.buffer.set_active(active)
        if self.region is not None:
            self.region.set_active(active)

    def set_sharpness(self, sharpness: float) -> None:
        """FSR 1's sharpening, live (:attr:`RenderSettings.sharpness`)."""
        from dataclasses import replace

        self.settings = replace(self.settings, sharpness=sharpness)
        if self.settings.fsr:
            self._final.card.set_shader_input("sharpness", sharpness)

    def describe(self) -> str:
        """:returns: the chain in a line, for a log or a HUD, e.g.
        ``1072x603 -> 1600x900 | FXAA -> EASU -> RCAS 0.10``"""
        (rw, rh), (ow, oh) = self.render_size, self.output_size
        if self.settings.fsr:
            chain = f"EASU -> RCAS {self.settings.sharpness:.2f}"
            if self.fxaa:
                forced = " (forced)" if not self.settings.fxaa else ""
                chain = f"FXAA{forced} -> {chain}"
        else:
            chain = "copy" if (rw, rh) == (ow, oh) else "bilinear"
            if self.fxaa:
                chain = f"FXAA + {chain}"
        return f"{rw}x{rh} -> {ow}x{oh} | {chain}"

    def destroy(self) -> None:
        """Remove the passes and the window's region (or the texture's
        buffer)."""
        if self.region is not None:
            self.window.remove_display_region(self.region)
        for p in self._passes:
            p.destroy()
        self._passes = []
        if self._out is not None:
            self._out.destroy()
            self._out = None
        self._final.root.remove_node()
        self._sampled = []
