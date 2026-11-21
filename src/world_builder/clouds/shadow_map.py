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

One pair holds every genus (:mod:`genera`): the maps are on the sky bands'
base's sphere (``march.sky_bands``: the genera's slabs merged), a texel the
genera's summed depth along its ray (cloud_field.glsl's ``skyShadowLevels``:
each band's chord cut at its clouds' own top, every genus in it marched by
its own field and weather), its levels at the quarters of the bands' summed
chord. The ground under them takes ``exp2(-depth)``, the product of the
genera's shades (``planetCloudShadow``). (Two pairs, before, one for the
convective clouds and one for the stratus, made every receiver carry a
second set of inputs and take the two shadows' product itself.)

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
from world_builder.utils import sun_moved

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
        bind: Any,
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
        self.card, self._bind = card, bind
        bind(card)  # the field's inputs: its volume's bind_shadow_inputs
        self.chart = PTA_LVecBase4f.empty_array(4)  # the receivers' too
        self.steps = PTA_int.empty_array(1)
        card.set_shader_inputs(uMapChart=self.chart, uMapSteps=self.steps)
        self.drawn: tuple | None = None  # (centre, sun, steps, layer) drawn

    def rebind(self) -> None:
        """Take the field's inputs again, from the clouds' volume now (a change
        of renderer makes a new one), and be drawn again."""
        self._bind(self.card)
        self.drawn = None

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
            or sun_moved(drawn_sun, sun)
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
    :ivar pair: the (near, far) pair, or None (no maps)
    """

    #: The receivers' inputs: (near map, far map, near chart, far chart, on).
    _INPUTS = (
        "cloudShadowNear",
        "cloudShadowFar",
        "cloudNearChart",
        "cloudFarChart",
        "cloudShadowMapOn",
    )

    def __init__(
        self, output: Any, clouds: Any, size: int, to_ram: bool = False
    ) -> None:
        """
        :param output: the window (its GSG draws the maps)
        :param clouds: the planet's clouds
            (:class:`~world_builder.clouds.planet.PlanetClouds` or a
            :class:`~world_builder.clouds.march.CloudMarch`: their
            ``shadow_caster()``, or a caster itself)
        :param size: ``ShadowSettings.cloud_map_size`` (0: none)
        :param to_ram: copy each map to RAM when drawn (tests)
        """
        self._output = output
        self._clouds = clouds
        self._to_ram = to_ram
        self._on = PTA_int.empty_array(1)
        self._nodes: list[NodePath] = []
        #: The caster the maps were baked from (a new one: rebind)
        self._caster_bound: Any = None
        self.pair: tuple[_Map, _Map] | None = None
        self.size = 0
        self.set_size(size)

    @property
    def near(self) -> _Map | None:
        """The near map, or None."""
        return self.pair[0] if self.pair else None

    @property
    def far(self) -> _Map | None:
        """The far map, or None."""
        return self.pair[1] if self.pair else None

    def bind(self, node: NodePath) -> None:
        """Give a node's shaders the maps (off until drawn)."""
        self._nodes.append(node)
        self._bind(node)

    def _bind(self, node: NodePath) -> None:
        near, far, near_chart, far_chart, on = self._INPUTS
        node.set_shader_input(on, self._on)
        pair = self.pair
        if pair is not None:
            node.set_shader_inputs(
                **{
                    near: pair[0].texture,
                    far: pair[1].texture,
                    near_chart: pair[0].chart,
                    far_chart: pair[1].chart,
                }
            )

    def set_size(self, size: int) -> None:
        """A new size (0: none): the maps made again."""
        if size == self.size:
            return
        self._destroy_maps()
        self._caster_bound = None
        self.size = int(size)
        if self.size:
            far = max(self.size // 2, 256)
            bind = self._clouds.bind_shadow_inputs
            self.pair = (
                _Map(
                    self._output,
                    bind,
                    "cloud_shadow_near",
                    self.size,
                    NEAR_SPACING,
                    self._to_ram,
                ),
                _Map(
                    self._output,
                    bind,
                    "cloud_shadow_far",
                    far,
                    FAR_SPACING,
                    self._to_ram,
                ),
            )
        self._nodes = [node for node in self._nodes if not node.is_empty()]
        for node in self._nodes:
            self._bind(node)

    def _caster(self) -> Any:
        clouds = self._clouds
        if hasattr(clouds, "shadow_caster"):
            return clouds.shadow_caster()
        return clouds

    def update(self, camera_world: ArrayLike, sun: ArrayLike, steps: int) -> None:
        """Per frame, before the frame is drawn: each map drawn again if it
        must be (:meth:`_Map.wanted`).

        :param camera_world: the camera, world frame
        :param sun: toward the sun, world frame (unit)
        :param steps: ``ShadowSettings.cloud_shadow_steps``
        """
        clouds = self._clouds
        relative = np.asarray(camera_world, dtype=np.float64) - clouds.position
        up = relative / np.linalg.norm(relative)
        sun = np.asarray(sun, dtype=np.float64)
        caster = self._caster()
        if self.pair and caster is not self._caster_bound:
            # A new caster: its inputs, not the removed one's, in the bake.
            self._caster_bound = caster
            for shadow_map in self.pair:
                shadow_map.rebind()
        layer = -1 if caster is None else caster.shadow_layer
        if not self.pair or layer < 0 or steps <= 0:
            self._on[0] = 0
            return
        bands = getattr(caster, "sky_bands", None) or [caster.shadow_field.slab]
        base = clouds.radius + bands[0][0]
        for shadow_map in self.pair:
            if shadow_map.wanted(up, sun, steps, layer, base):
                shadow_map.draw(up, sun, steps, layer, base)
        self._on[0] = 1

    def _destroy_maps(self) -> None:
        for shadow_map in self.pair or ():
            shadow_map.destroy(self._output)
        self.pair = None
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
