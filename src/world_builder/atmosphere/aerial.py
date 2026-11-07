"""
The aerial-perspective cells (Hillaire 2020; RENDERING_PLAN.md B.4): the air
between the camera and every cell of a 32 x 32 x 32 grid over the main
camera's view, baked on the GPU each frame, read by ``applyAtmospheres``
(the terrain, the near sea, the models) in place of their per-pixel march.

Each cell is ``atmosphereAlong`` from the camera to its slice's end, at its
most samples (16): the eclipses, the rings' shadow and the planet's shadow
with it. The slices run over the longest path the air allows: from the
camera to past the horizon (``sqrt(|cam|^2 - R^2) + sqrt(top^2 - R^2)``) or
:data:`AERIAL_REACH` (128 km), the nearer, spaced as the square of their
index (finer near). Only with the camera inside an atmosphere: from space the
air lies in a thin shell at the far end of every ray, which slices from the
camera do not resolve (marched there, as before). Atlas: 8 x 4 tiles of 32 x 32 a half,
in-scatter in the lower, transmittance in the upper (RGBA16F, 256 x 256: a
power of two, as Panda pads render targets to one).

A point outside the main camera's view (the sea's reflection's rays) or
beyond the slices (far ground, a moon behind the air) is marched
as before (``aerialAlong`` returns false). The clouds keep their own
treatment (no air in front of them, atmosphere.md).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from panda3d.core import (
    Camera,
    CardMaker,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    LMatrix4f,
    LVecBase4f,
    NodePath,
    OmniBoundingVolume,
    OrthographicLens,
    PTA_LMatrix4f,
    Shader,
    WindowProperties,
)

SHADER_PATH = Path(__file__).parent / "shaders"
#: The atlas's texels (atmosphere.glsl's AERIAL_ATLAS).
AERIAL_ATLAS = 256
#: The farthest the slices reach, metres: past it a point is marched. The
#: slices must resolve the air where it is dense (the last few kilometres
#: over the ground): at 128 km their spacing there is ~8 km; over the
#: longest path the air allows (~300 km, more from space) the in-scatter of
#: a ground seen from high up was up to 38 % off.
AERIAL_REACH = 128e3


class AerialPerspective:
    """The aerial-perspective cells' bake (see the module), into
    :attr:`Atmospheres.aerial_texture`; made by
    :meth:`Atmospheres.attach_aerial`.

    :ivar root: the bake's scene: give it the shadows' inputs
    """

    def __init__(
        self, output: Any, atmospheres: Any, camera: NodePath, lens: Any, sun: Any
    ) -> None:
        """
        :param output: the window (its GSG bakes the cells)
        :param atmospheres: the world's :class:`Atmospheres`
        :param camera: the main camera (its view is the grid's)
        :param lens: its lens
        :param sun: the sky's ``uSunDir`` input (a PTA, shared live)
        """
        self._output = output
        self._atmospheres = atmospheres
        self._camera = camera
        self._lens = lens
        props = FrameBufferProperties()
        props.set_rgba_bits(16, 16, 16, 16)
        props.set_float_color(True)
        engine = output.get_engine()
        self.buffer = engine.make_output(
            output.get_pipe(),
            "aerial_bake",
            -400,  # before every camera that reads it
            props,
            WindowProperties.size(AERIAL_ATLAS, AERIAL_ATLAS),
            GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_refuse_parasite,
            output.get_gsg(),
            output,
        )
        if self.buffer is None:
            raise RuntimeError("no offscreen buffer for the aerial perspective")
        self.buffer.add_render_texture(
            atmospheres.aerial_texture,
            GraphicsOutput.RTM_bind_or_copy,
            GraphicsOutput.RTP_color,
        )
        self.buffer.set_clear_color_active(False)
        self.root = NodePath("aerial_bake")
        self.root.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "aerial_bake.vert",
                fragment=SHADER_PATH / "aerial_bake.frag",
            )
        )
        from world_builder.atmosphere import bind_defaults
        from world_builder.shadows import bind_defaults as bind_shadow_defaults

        bind_defaults(self.root)
        bind_shadow_defaults(self.root)
        atmospheres.bind(self.root)
        self._from_clip = PTA_LMatrix4f.empty_array(1)
        self.root.set_shader_inputs(uSunDir=sun, uAerialFromClip=self._from_clip)
        maker = CardMaker("aerial")
        maker.set_frame(-1.0, 1.0, -1.0, 1.0)
        card = self.root.attach_new_node(maker.generate())
        card.node().set_bounds(OmniBoundingVolume())
        card.node().set_final(True)
        card.set_two_sided(True)
        card.set_depth_test(False)
        card.set_depth_write(False)
        bake_camera = self.root.attach_new_node(Camera("aerial", OrthographicLens()))
        self.buffer.make_display_region().set_camera(bake_camera)

    def update(self, offsets: list, active: bool) -> None:
        """Per frame, after the atmospheres' uniforms: the grid's view and
        reach, and whether to bake.

        :param offsets: per atmosphere, (camera - centre as a float64
            array, ground radius, top radius)
        :param active: bake and read the cells
        """
        inside = [o for o in offsets if float(o[0] @ o[0]) < o[2] * o[2]]
        if not active or not inside:  # from space: the march
            self._atmospheres._aerial_on[0] = 0
            self.buffer.set_active(False)
            return
        offset, ground, top = inside[0]  # the air the camera is in
        rho = math.sqrt(float(offset @ offset))
        reach = math.sqrt(max(rho * rho - ground * ground, 0.0)) + math.sqrt(
            top * top - ground * ground
        )
        start, length = 0.0, min(reach, AERIAL_REACH)
        rotation = LMatrix4f(self._camera.get_net_transform().get_mat())
        rotation.set_row(3, (0.0, 0.0, 0.0))  # directions: no translation
        to_clip = LMatrix4f()
        to_clip.invert_from(rotation)
        to_clip = to_clip * self._lens.get_projection_mat()
        from_clip = LMatrix4f()
        from_clip.invert_from(to_clip)
        self._atmospheres._aerial_to_clip[0] = to_clip
        self._from_clip[0] = from_clip
        self._atmospheres._aerial_range[0] = LVecBase4f(start, length, 0.0, 0.0)
        self._atmospheres._aerial_on[0] = 1
        self.buffer.set_active(True)

    def destroy(self) -> None:
        self._atmospheres._aerial_on[0] = 0
        self._output.get_engine().remove_window(self.buffer)
        self.root.remove_node()
