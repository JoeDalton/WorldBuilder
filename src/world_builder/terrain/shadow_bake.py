"""
The mountains' shadow maps baked on the GPU: a new sun's map the next
frame, not seconds later in the worker pool.

The pool's bake (:func:`~world_builder.terrain.shadow_map.ceiling_cube`) is
one pass over the rows of a grid aligned with the sun, each row from the
last: sequential, ~2 s at 1024. Marching each texel toward the sun over
the ground instead is as exact (closer to the straight ray, in the tests)
and every texel stands alone: a fragment shader per cube face
(``shadow_bake_cube.frag``), a few hundred steps at most, each a fetch of
the ground's cube map. Its CPU twin is
:func:`~world_builder.terrain.shadow_map.march_ceiling`. The ground itself
is still sampled once per size in the pool (the planet's noise lives on
the CPU), and uploaded once.

:class:`CeilingBake` draws the six faces into a float cube map, read by
the shaders as the pool's was (``shadows.glsl``'s ``terrainShadow``: km;
not mipmapped, see :class:`CeilingBake`); :class:`LocalBake` the local
map. A bake is drawn a band of rows a frame, at most :data:`BAKE_TEXELS`:
:meth:`_Bake.step` each frame. A moving sun gets a new map as soon as the
last is done.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BitMask32,
    Camera,
    CardMaker,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    LVecBase4f,
    NodePath,
    OmniBoundingVolume,
    OrthographicLens,
    PTA_LVecBase4f,
    SamplerState,
    Shader,
    Texture,
    WindowProperties,
)

from world_builder.terrain.shadow_map import (
    STEP_TEXELS,
    TOPS_REACH_TEXELS,
    march_limits,
)

SHADER_PATH = Path(__file__).parent / "shaders"
#: The most texels a bake draws a frame: the global map's (~0.4 ms a frame
#: in the demo's views), the local map's (dearer per node: ~0.5 ms).
BAKE_TEXELS = 1 << 19
LOCAL_BAKE_TEXELS = 1 << 17


def tops_texture(tops: np.ndarray) -> Texture:
    """The highest ground near each block (:func:`shadow_map.local_tops`)
    as a float cube texture, nearest (a block's own)."""
    size = tops.shape[1]
    tex = Texture("terrain_tops")
    tex.setup_cube_map(size, Texture.T_float, Texture.F_r32)
    tex.set_ram_image(np.ascontiguousarray(tops, dtype=np.float32).tobytes())
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_minfilter(SamplerState.FT_nearest)
    return tex


def heights_texture(heights: np.ndarray) -> Texture:
    """The ground's (6, n, n) cube map (:func:`shadow_map.height_cube`) as a
    float cube texture, linearly filtered, for the bake to march over."""
    size = heights.shape[1]
    tex = Texture("terrain_heights")
    tex.setup_cube_map(size, Texture.T_float, Texture.F_r32)
    # Pages in face order, rows bottom-up = t up (sky.starfield's layout).
    tex.set_ram_image(np.ascontiguousarray(heights, dtype=np.float32).tobytes())
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_minfilter(SamplerState.FT_linear)
    return tex


class _Bake:
    """A bake drawn a band of rows a frame (see the module): :meth:`start`
    lays out the bands, :meth:`step` draws the next with this frame."""

    buffer: Any
    size: int
    texture: Texture
    budget: int = BAKE_TEXELS  # texels a frame
    _regions: list
    _frames: list
    _next: int

    def start(self, pages: int) -> None:
        """Lay out the bands over *pages* pages of the size, then group
        them into frames: as many as :attr:`budget` holds, a band a page (a
        region draws one rectangle)."""
        rows = max(1, min(self.size, self.budget // self.size))
        bands = [
            (page, bottom, min(bottom + rows, self.size))
            for page in range(pages)
            for bottom in range(0, self.size, rows)
        ]
        self._frames: list[list] = []
        for band in bands:
            frame = self._frames[-1] if self._frames else None
            texels = (band[2] - band[1]) * self.size
            if (
                frame is None
                or any(page == band[0] for page, _, _ in frame)
                or sum((t - b) * self.size for _, b, t in frame) + texels > self.budget
            ):
                self._frames.append([band])
            else:
                frame.append(band)
        self._next = 0

    def step(self) -> bool:
        """Draw the next frame's bands with this frame.

        :returns: whether the bake is done: every band drawn by the frames
            before (nothing left to draw)
        """
        if self._next >= len(self._frames):
            self.buffer.set_active(False)
            return True
        bands = {page: (bottom, top) for page, bottom, top in self._frames[self._next]}
        self._next += 1
        for page, region in enumerate(self._regions):
            region.set_active(page in bands)
            if page in bands:
                bottom, top = bands[page]
                region.set_dimensions(0.0, 1.0, bottom / self.size, top / self.size)
        self.buffer.set_active(True)
        self.buffer.set_one_shot(True)
        return False


class CeilingBake(_Bake):
    """The global shadow ceiling cube map, baked on the GPU (see the
    module): :meth:`bake` starts one, :meth:`step` draws it over frames.

    :ivar texture: the map, km (ceiling, distance), read as
        ``uTerrainShadow``
    :ivar size: texels per face edge
    """

    def __init__(self, output: Any, size: int, to_ram: bool = False) -> None:
        """
        :param output: the window (its GSG draws the map)
        :param size: texels per face edge
        :param to_ram: copy the map to RAM when drawn (tests)
        """
        self._output = output
        self.size = int(size)
        self.texture = Texture("terrain_shadow")
        # Two half floats a texel (24 MB at 1024), as the pool's upload.
        self.texture.setup_cube_map(self.size, Texture.T_half_float, Texture.F_rg16)
        self.texture.set_magfilter(SamplerState.FT_linear)
        # Not mipmapped: a render target with mipmaps had them made again
        # after every frame its buffer drew (+2.7 ms a frame of a bake, in
        # the views' draws), and without them the views cost the same
        # (orbit, stratosphere, the ground: within noise).
        self.texture.set_minfilter(SamplerState.FT_linear)
        props = FrameBufferProperties()
        props.set_rgba_bits(16, 16, 0, 0)
        props.set_float_color(True)
        engine = output.get_engine()
        self.buffer = engine.make_output(
            output.get_pipe(),
            "terrain_shadow_bake",
            -500,  # before every camera that reads it
            props,
            WindowProperties.size(self.size, self.size),
            GraphicsPipe.BF_refuse_window,
            output.get_gsg(),
            output,
        )
        if self.buffer is None:
            raise RuntimeError("no offscreen buffer for the shadow bake")
        mode = GraphicsOutput.RTM_bind_or_copy
        if to_ram:
            mode = GraphicsOutput.RTM_copy_ram
        self.buffer.add_render_texture(self.texture, mode, GraphicsOutput.RTP_color)
        self.buffer.set_clear_color_active(False)
        self.root = NodePath("terrain_shadow_bake")
        self.root.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "shadow_bake.vert",
                fragment=SHADER_PATH / "shadow_bake_cube.frag",
            )
        )
        maker = CardMaker("face")
        maker.set_frame(-1.0, 1.0, -1.0, 1.0)
        self._regions = []
        self._frames = []
        self._next = 0
        for face in range(6):
            # Each face's camera sees its own card alone (their masks).
            card = self.root.attach_new_node(maker.generate())
            card.node().set_bounds(OmniBoundingVolume())
            card.node().set_final(True)
            card.set_two_sided(True)
            card.set_depth_test(False)
            card.set_depth_write(False)
            card.hide(BitMask32.all_on() & ~BitMask32.bit(face))
            card.set_shader_input("uFace", face)
            camera = self.root.attach_new_node(
                Camera(f"terrain_shadow_face{face}", OrthographicLens())
            )
            camera.node().set_camera_mask(BitMask32.bit(face))
            region = self.buffer.make_display_region()
            region.set_camera(camera)
            region.set_target_tex_page(face)
            self._regions.append(region)
        self.root.set_shader_input("uSize", float(self.size))
        self.buffer.set_active(False)

    def bake(
        self,
        heights: Texture,
        tops: Texture,
        sun: ArrayLike,
        radius: float,
        top: float,
        step_texels: float = STEP_TEXELS,
    ) -> None:
        """Start the map for a sun (its first band with this frame:
        :meth:`step`).

        :param heights: the ground's cube map (:func:`heights_texture`)
        :param tops: the highest ground near each block (:func:`tops_texture`)
        :param sun: toward the sun, the planet's frame
        :param radius: sea-level radius, metres
        :param top: the highest ground, metres
        """
        sun = np.asarray(sun, dtype=np.float64)
        sun = sun / np.linalg.norm(sun)
        step, near, last, steps = march_limits(self.size, radius, top, step_texels)
        self.root.set_shader_inputs(
            uHeights=heights,
            uTops=tops,
            uReach=TOPS_REACH_TEXELS * 2.0 / self.size,
            uSun=tuple(float(v) for v in sun),
            uBake=LVecBase4f(radius, max(top, 0.0), step, last),
            uSteps=(near, steps),
        )
        self.start(6)
        self.step()

    def destroy(self) -> None:
        self._output.get_engine().remove_window(self.buffer)
        self.root.remove_node()


def local_heights_texture(heights: np.ndarray) -> Texture:
    """A local map's (n, n) ground at its chart's nodes as a float texture
    (rows bottom-up = north), linearly filtered, clamped."""
    size = heights.shape[0]
    tex = Texture("local_terrain_heights")
    tex.setup_2d_texture(size, size, Texture.T_float, Texture.F_r32)
    tex.set_ram_image(np.ascontiguousarray(heights, dtype=np.float32).tobytes())
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


class LocalBake(_Bake):
    """The local shadow map (around the camera), baked on the GPU
    (``shadow_bake_local.frag``): :meth:`bake` starts one, :meth:`step`
    draws it over frames (:data:`LOCAL_BAKE_TEXELS` a frame).

    :ivar texture: the map, metres (ceiling, distance), read as
        ``uLocalShadow``
    :ivar size: nodes per side
    """

    budget = LOCAL_BAKE_TEXELS

    def __init__(self, output: Any, size: int, to_ram: bool = False) -> None:
        self._output = output
        self.size = int(size)
        self.texture = Texture("local_terrain_shadow")
        # Two floats a node: metres over a few hundred (half floats would
        # shift a ridge's shadow by its penumbra).
        self.texture.setup_2d_texture(
            self.size, self.size, Texture.T_float, Texture.F_rg32
        )
        self.texture.set_magfilter(SamplerState.FT_linear)
        # Not mipmapped (CeilingBake's reasons).
        self.texture.set_minfilter(SamplerState.FT_linear)
        self.texture.set_wrap_u(SamplerState.WM_clamp)
        self.texture.set_wrap_v(SamplerState.WM_clamp)
        props = FrameBufferProperties()
        props.set_rgba_bits(32, 32, 0, 0)
        props.set_float_color(True)
        engine = output.get_engine()
        self.buffer = engine.make_output(
            output.get_pipe(),
            "local_terrain_shadow_bake",
            -500,
            props,
            WindowProperties.size(self.size, self.size),
            GraphicsPipe.BF_refuse_window,
            output.get_gsg(),
            output,
        )
        if self.buffer is None:
            raise RuntimeError("no offscreen buffer for the local shadow bake")
        mode = GraphicsOutput.RTM_bind_or_copy
        if to_ram:
            mode = GraphicsOutput.RTM_copy_ram
        self.buffer.add_render_texture(self.texture, mode, GraphicsOutput.RTP_color)
        self.buffer.set_clear_color_active(False)
        self.root = NodePath("local_terrain_shadow_bake")
        self.root.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "shadow_bake.vert",
                fragment=SHADER_PATH / "shadow_bake_local.frag",
            )
        )
        maker = CardMaker("chart")
        maker.set_frame(-1.0, 1.0, -1.0, 1.0)
        card = self.root.attach_new_node(maker.generate())
        card.node().set_bounds(OmniBoundingVolume())
        card.node().set_final(True)
        card.set_two_sided(True)
        card.set_depth_test(False)
        card.set_depth_write(False)
        camera = self.root.attach_new_node(Camera("local_shadow", OrthographicLens()))
        region = self.buffer.make_display_region()
        region.set_camera(camera)
        self._regions = [region]
        self._frames = []
        self._next = 0
        self._chart = PTA_LVecBase4f.empty_array(3)
        self.root.set_shader_input("uChart", self._chart)
        self.buffer.set_active(False)

    def bake(
        self,
        heights: Texture,
        chart: Any,
        spacing: float,
        sun: ArrayLike,
        near: float,
        far: float,
        top: float = 1.0e5,
    ) -> None:
        """Start the map for a square of ground and a sun (its first band
        with this frame: :meth:`step`).

        :param heights: its ground (:func:`local_heights_texture`)
        :param chart: its chart (:class:`terrain.chart.AzimuthalEquidistantChart`)
        :param spacing: metres between nodes
        :param sun: toward the sun, the planet's frame
        :param near: the nearest occluder counted, metres
        :param far: the farthest (the global map's nearest), metres
        :param top: the square's highest ground, metres (the march stops
            once nothing can rise above the ceiling found)
        """
        sun = np.asarray(sun, dtype=np.float64)
        sun = sun / np.linalg.norm(sun)
        frame = chart.frame
        self._chart[0] = LVecBase4f(*frame.up, chart.radius)
        self._chart[1] = LVecBase4f(*frame.east, spacing)
        self._chart[2] = LVecBase4f(*frame.north, float(self.size))
        first = max(1, round(near / spacing))
        self.root.set_shader_inputs(
            uHeights=heights,
            uSun=tuple(float(v) for v in sun),
            uSteps=(first, int(math.ceil(far / spacing))),
            uFar=float(far),
            uTop=float(top),
        )
        self.start(1)
        self.step()

    def destroy(self) -> None:
        self._output.get_engine().remove_window(self.buffer)
        self.root.remove_node()
