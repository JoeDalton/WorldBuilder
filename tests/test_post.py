"""
The post chain (world_builder.post): its settings, and its passes drawn
offscreen from a known view (skipped without a GL context): a flat view
stays flat through every chain, pure black and white make no NaN, the
passes take the view's and the output's sizes, RCAS's sharpness shows.
"""

import math

import numpy as np
import pytest
from panda3d.core import (
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    Texture,
    WindowProperties,
)

from world_builder.post import (
    BILINEAR,
    FSR1,
    FSR_MODES,
    MIN_SCALE,
    PostChain,
    RenderSettings,
)
from world_builder.quality import QualityLevel
from world_builder.settings import WorldSettings

# ── settings ─────────────────────────────────────────────────────────────


def test_the_default_draws_the_view_as_it_was():
    settings = RenderSettings()
    assert settings.scale == 1.0 and settings.msaa == 0
    assert not settings.fxaa and settings.upscaler == BILINEAR
    assert not settings.needs_chain
    assert WorldSettings().render == settings


def test_fsr_without_msaa_forces_fxaa():
    fsr = RenderSettings(upscaler=FSR1)
    assert fsr.fsr and fsr.fxaa_forced and fsr.fxaa_active
    with_msaa = RenderSettings(upscaler=FSR1, msaa=4)
    assert not with_msaa.fxaa_forced and not with_msaa.fxaa_active
    assert RenderSettings(upscaler=FSR1, msaa=4, fxaa=True).fxaa_active
    assert not RenderSettings(msaa=0).fxaa_forced  # bilinear never forces it


def test_render_size_rounds_each_axis_and_keeps_a_pixel():
    settings = RenderSettings(scale=0.67)
    assert settings.render_size((1600, 900)) == (1072, 603)
    assert RenderSettings(scale=MIN_SCALE).render_size((1, 1)) == (1, 1)
    assert RenderSettings().render_size((1600, 900)) == (1600, 900)


def test_the_fsr_modes_are_within_the_scale_range():
    for scale in FSR_MODES.values():
        assert MIN_SCALE <= scale < 1.0
        RenderSettings(scale=scale)  # strict: in range


def test_the_render_section_is_kept_by_every_preset_and_saved():
    """A choice, as the display's: no level of its own, kept when the world
    moves to a preset, saved and read back by label."""
    assert not RenderSettings.LEVELLED
    assert all(k.kept for _, k in RenderSettings.knobs())
    world = (
        WorldSettings()
        .with_value("render", "scale", 0.59)
        .with_value("render", "upscaler", "FSR 1")
        .with_value("render", "msaa", "4x")
    )
    assert world.render == RenderSettings(scale=0.59, upscaler=FSR1, msaa=4)
    assert world.with_level(QualityLevel.LOW).render == world.render
    assert not world.is_custom()
    saved = world.to_dict()
    assert saved["overrides"]["render"] == {
        "scale": 0.59,
        "msaa": "4x",
        "upscaler": "FSR 1",
    }
    assert WorldSettings.from_dict(saved).render == world.render


def test_out_of_range_render_values_are_clamped_or_refused():
    world = WorldSettings().with_value("render", "scale", 0.1)
    assert world.render.scale == MIN_SCALE
    assert WorldSettings().with_value("render", "msaa", 3).render.msaa == 0
    with pytest.raises(ValueError):
        RenderSettings(scale=1.5)


def test_only_the_sharpness_applies_live():
    changes = RenderSettings().changes(
        RenderSettings(scale=0.5, msaa=2, fxaa=True, upscaler=FSR1, sharpness=1.0)
    )
    assert changes == {
        "scale": "rebuild",
        "msaa": "rebuild",
        "fxaa": "rebuild",
        "upscaler": "rebuild",
        "sharpness": "live",
    }


# ── the passes, drawn ────────────────────────────────────────────────────

#: The output's size: odd sizes, so a scale's rounding shows.
OUT = (61, 37)


@pytest.fixture(scope="module")
def window(showbase):
    """The session's ShowBase with an offscreen window; skipped where none
    can be opened."""
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


def _view(pixels: np.ndarray) -> Texture:
    """A view's texture from (h, w) grey values 0-1, row 0 at the bottom."""
    height, width = pixels.shape
    grey = np.round(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgra = np.empty((height, width, 4), np.uint8)
    bgra[..., :3] = grey[..., None]
    bgra[..., 3] = 255
    texture = Texture("view")
    texture.setup_2d_texture(width, height, Texture.T_unsigned_byte, Texture.F_rgba8)
    texture.set_ram_image(bgra.tobytes())
    from world_builder.post import target_texture

    template = target_texture("filters")
    texture.set_minfilter(template.get_minfilter())
    texture.set_magfilter(template.get_magfilter())
    return texture


class _Target:
    """An offscreen output standing for the game's window, read back."""

    def __init__(self, window, size=OUT):
        props = FrameBufferProperties()
        props.set_rgba_bits(8, 8, 8, 8)
        self.engine = window.graphicsEngine
        self.output = self.engine.make_output(
            window.pipe,
            "post_target",
            0,
            props,
            WindowProperties.size(*size),
            GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_resizeable,
            window.win.get_gsg(),
            window.win,
        )
        assert self.output is not None
        # Red where the chain would leave a pixel undrawn.
        self.output.set_clear_color_active(True)
        self.output.set_clear_color((1, 0, 0, 1))
        self.texture = Texture("post_target")
        self.output.add_render_texture(self.texture, GraphicsOutput.RTM_copy_ram)

    def draw(self) -> np.ndarray:
        """:returns: (h, w, 3) RGB 0-1, row 0 at the bottom"""
        self.engine.render_frame()
        self.engine.render_frame()
        width, height = self.output.get_x_size(), self.output.get_y_size()
        raw = np.frombuffer(self.texture.get_ram_image_as("RGB"), np.uint8)
        return raw.reshape(height, width, 3).astype(np.float64) / 255.0

    def destroy(self):
        self.engine.remove_window(self.output)


@pytest.fixture
def target(window):
    made = _Target(window)
    yield made
    made.destroy()


def _chain(window, target, settings, pixels):
    size = settings.render_size(OUT)
    assert pixels.shape == (size[1], size[0])
    return PostChain(
        window.graphicsEngine,
        target.output,
        _view(pixels),
        settings,
        size,
        buffer_sort=-10,
    )


CHAINS = {
    "copy": RenderSettings(),
    "bilinear": RenderSettings(scale=0.67),
    "fxaa": RenderSettings(scale=0.67, fxaa=True),
    "fsr_forced_fxaa": RenderSettings(scale=0.67, upscaler=FSR1),
    "fsr_over_msaa": RenderSettings(scale=0.5, upscaler=FSR1, msaa=4),
    "fsr_native": RenderSettings(upscaler=FSR1),
}


@pytest.mark.parametrize("name", list(CHAINS))
def test_a_flat_view_stays_flat(window, target, name):
    """Every chain compiles, covers its region and keeps a flat grey (the
    kernels' weights sum to one; RCAS has nothing to sharpen)."""
    settings = CHAINS[name]
    width, height = settings.render_size(OUT)
    chain = _chain(window, target, settings, np.full((height, width), 0.5))
    try:
        image = target.draw()
    finally:
        chain.destroy()
    assert np.all(np.abs(image - 128 / 255) <= 1.5 / 255), name


@pytest.mark.parametrize("name", ["fsr_forced_fxaa", "fsr_over_msaa", "fsr_native"])
def test_pure_black_and_white_make_no_nan(window, target, name):
    """FSR 1's reciprocals over a black or a white neighbourhood (0 / 0
    unguarded): the flat parts stay exactly black and white, the edge
    between them in [0, 1] and monotonic across it."""
    settings = CHAINS[name]
    width, height = settings.render_size(OUT)
    pixels = np.zeros((height, width))
    pixels[:, width // 2 :] = 1.0
    chain = _chain(window, target, settings, pixels)
    try:
        image = target.draw()[..., 0]
    finally:
        chain.destroy()
    assert np.all(image[:, : OUT[0] // 4] == 0.0)
    assert np.all(image[:, -OUT[0] // 4 :] == 1.0)
    row = image[OUT[1] // 2]
    assert np.all(np.diff(row) >= -1.0 / 255.0)  # deringed: no overshoot


def test_the_passes_take_the_view_and_output_sizes(window, target):
    settings = RenderSettings(scale=0.67, upscaler=FSR1)
    size = settings.render_size(OUT)
    chain = _chain(window, target, settings, np.full((size[1], size[0]), 0.5))
    try:
        fxaa, easu = chain.buffers()
        assert (fxaa.get_x_size(), fxaa.get_y_size()) == size
        assert (easu.get_x_size(), easu.get_y_size()) == OUT
        assert chain.output_size == OUT
        assert chain.describe() == (
            f"{size[0]}x{size[1]} -> {OUT[0]}x{OUT[1]} | "
            "FXAA (forced) -> EASU -> RCAS 0.10"
        )
        # A new output size: the EASU pass follows the region, the FXAA one
        # the view.
        target.output.set_size(80, 45)
        smaller = (50, 30)
        chain.resize(smaller)
        assert chain.output_size == (80, 45)
        assert (fxaa.get_x_size(), fxaa.get_y_size()) == smaller
        assert (easu.get_x_size(), easu.get_y_size()) == (80, 45)
    finally:
        chain.destroy()


def test_destroy_removes_the_region_and_the_buffers(window, target):
    engine = window.graphicsEngine
    engine.render_frame()  # new outputs are counted from the next frame
    windows = engine.get_num_windows()
    regions = target.output.get_num_display_regions()
    settings = RenderSettings(scale=0.5, upscaler=FSR1)
    size = settings.render_size(OUT)
    chain = _chain(window, target, settings, np.full((size[1], size[0]), 0.5))
    engine.render_frame()
    assert engine.get_num_windows() == windows + 2
    assert target.output.get_num_display_regions() == regions + 1
    chain.destroy()
    engine.render_frame()
    assert engine.get_num_windows() == windows
    assert target.output.get_num_display_regions() == regions


@pytest.mark.parametrize("name", ["copy", "bilinear", "fxaa"])
def test_bilinear_draws_straight_into_the_region(window, target, name):
    """No intermediate pass: the copy or FXAA samples the view itself."""
    settings = CHAINS[name]
    width, height = settings.render_size(OUT)
    chain = _chain(window, target, settings, np.full((height, width), 0.5))
    try:
        assert chain.passes == 0
    finally:
        chain.destroy()


def _stripes(width: int, height: int) -> np.ndarray:
    """Soft vertical stripes: edges for RCAS to sharpen, no clipping."""
    x = np.arange(width)
    row = 0.5 + 0.3 * np.sin(2.0 * math.pi * x / 6.0)
    return np.tile(row, (height, 1))


def test_rcas_sharpness_shows_and_applies_live(window, target):
    """Sharper (fewer stops) means more contrast; set_sharpness changes the
    drawn image without a rebuild."""
    settings = RenderSettings(scale=0.5, upscaler=FSR1, msaa=4, sharpness=2.0)
    width, height = settings.render_size(OUT)
    chain = _chain(window, target, settings, _stripes(width, height))
    try:
        soft = target.draw()[..., 0]
        chain.set_sharpness(0.0)
        sharp = target.draw()[..., 0]
    finally:
        chain.destroy()
    inner = (slice(4, -4), slice(4, -4))
    assert sharp[inner].std() > soft[inner].std() * 1.02
    assert chain.settings.sharpness == 0.0


def _valid_size(texture) -> tuple[int, int]:
    """A render target's drawn size: its size times its tex scale (the GSG
    may pad it to powers of two)."""
    sx, sy = texture.get_tex_scale()
    return round(texture.get_x_size() * sx), round(texture.get_y_size() * sy)


@pytest.mark.parametrize("name", ["bilinear", "fxaa", "fsr_forced_fxaa"])
def test_a_chain_can_end_in_a_texture(window, name):
    """Given an output size (a mirror's card on screen), the chain ends in a
    texture of that size instead of a region, a flat view still flat in it;
    resize follows a new size; destroy removes every buffer."""
    engine = window.graphicsEngine
    engine.render_frame()
    windows = engine.get_num_windows()
    regions = window.win.get_num_display_regions()
    settings = CHAINS[name]
    width, height = settings.render_size(OUT)
    chain = PostChain(
        engine,
        window.win,
        _view(np.full((height, width), 0.5)),
        settings,
        (width, height),
        buffer_sort=-50,
        output_size=OUT,
    )
    try:
        assert chain.region is None and chain.output_size == OUT
        assert window.win.get_num_display_regions() == regions
        engine.render_frame()
        engine.render_frame()
        assert engine.get_num_windows() == windows + chain.passes + 1
        texture = chain.texture
        assert engine.extract_texture_data(texture, window.win.get_gsg())
        assert _valid_size(texture) == OUT  # the rest padding, maybe
        raw = np.frombuffer(texture.get_ram_image_as("RGB"), np.uint8)
        image = raw.reshape(texture.get_y_size(), texture.get_x_size(), 3)
        valid = image[: OUT[1], : OUT[0]].astype(int)
        assert np.all(np.abs(valid - 128) <= 1), name
        chain.resize((width, height), (40, 20))
        assert chain.output_size == (40, 20)
        engine.render_frame()
        assert _valid_size(chain.texture) == (40, 20)
    finally:
        chain.destroy()
    engine.render_frame()
    assert engine.get_num_windows() == windows


def test_set_active_stops_and_restarts_every_pass(window):
    """A hidden view (a game's mirror): no buffer of the chain draws."""
    engine = window.graphicsEngine
    settings = CHAINS["fsr_forced_fxaa"]
    width, height = settings.render_size(OUT)
    chain = PostChain(
        engine,
        window.win,
        _view(np.full((height, width), 0.5)),
        settings,
        (width, height),
        buffer_sort=-50,
        output_size=OUT,
    )
    try:
        engine.render_frame()
        buffers = [*chain.buffers(), chain._out.buffer]
        assert len(buffers) == 3 and all(b.is_active() for b in buffers)
        chain.set_active(False)
        assert not any(b.is_active() for b in buffers)
        chain.set_active(True)
        assert all(b.is_active() for b in buffers)
    finally:
        chain.destroy()
