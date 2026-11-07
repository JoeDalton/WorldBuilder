"""
The clouds' light volume (RENDERING_PLAN.md part C; EVE-Redux's
``LightVolume``): for every voxel of a field's slab around the camera, the
clouds' optical depth toward the sun and straight up, baked on the GPU, for
the raymarch to light its clouds by (in place of the shadow maps' four
levels, whose bands showed in towers) and later the air's god rays.

**Chart** (``shaders/cloud_field.glsl``, ``lightVoxelAt``). The cloud shadow maps'
azimuthal-equidistant chart on the slab base's sphere around the camera,
warped radially (a texel at ``t`` in [-1, 1]^2 of its tile lies
``|t|^2 * reach`` from the centre: fine near the camera, coarse toward the
horizon), and :data:`SLICES` shells through the slab. The reach is how far
a sight line inside the slab can see (the base's horizon from the camera
plus the top's), between :data:`REACH_MIN` and :data:`REACH_MAX`.

**Storage.** An atlas of the slices' tiles (:data:`COLUMNS` a row), RG16F:
the optical depth toward the sun (r) and up (g). Depths, not
transmittances: a depth is linear along a ray, so the depth between two
points of a sun ray is the difference of theirs.

**Bake** (``cloud_light_bake.frag``): each voxel marched toward the sun and
up to the slab's top, the weather and the cell octave taken at every step
(the ground's shadow march takes them once, where its chord starts: two
voxels on one sun ray then disagreed about the clouds between them, and a
tower's face streaked), in steps of half a sun step (:data:`STEP_SCALE`):
:func:`light_depth_on_sphere` on the CPU. A few slices a
frame (:attr:`CloudLightVolumes.per_frame`) into a back atlas, swapped to
the front once whole: the readers never see a half-baked volume, nor one of
two suns. Baked again (change-driven, the shadow maps' rules) when the sun
turns past ``utils.SUN_TOLERANCE``, the camera moves :data:`RECENTRE` of the
reach, the reach changes by as much, or the steps or the clouds' layer
change; one bake at a time, the next one started when it is swapped in (a
sun swinging faster than a bake lags by up to two). With all still, nothing
a frame.

With genera, each family has its own volume (its own slab and field), as it
has its own shadow maps.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    Camera,
    CardMaker,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    LVecBase3f,
    LVecBase4f,
    NodePath,
    OmniBoundingVolume,
    OrthographicLens,
    PTA_int,
    PTA_LVecBase3f,
    PTA_LVecBase4f,
    SamplerState,
    Shader,
    Texture,
    WindowProperties,
)

from world_builder.clouds.noise import (
    CLOUD_SHADOW_STEP_SCALE,
    _slab_density,
    cell_shift,
    fbm,
    shadow_calibration,
)
from world_builder.terrain.chart import TangentFrame, latlon
from world_builder.utils import sun_moved

SHADER_PATH = Path(__file__).parent / "shaders"
#: Shells through the slab.
SLICES = 32
#: Tiles a row of the atlas (SLICES / COLUMNS rows).
COLUMNS = 8
#: The reach's bounds, metres.
REACH_MIN = 40e3
REACH_MAX = 200e3
#: Baked again once the camera has moved this share of the reach (EVE's 3 %),
#: or the reach has changed by it.
RECENTRE = 0.03
#: The most steps of a voxel's march.
STEPS = 64
#: Its steps' length against ``cloudShadowSteps``' (about two sun steps):
#: half a sun step. Each voxel marches its own chord: with longer steps
#: neighbours sampled the field at different phases, grain in the readers'
#: difference of two voxels (a tower's face streaked).
STEP_SCALE = 0.25


def light_reach(camera_radius: float, base: float, top: float) -> float:
    """How far from the camera the volume reaches, metres: the longest sight
    line inside the slab (from the camera to the base's horizon, then on to
    the top's), within :data:`REACH_MIN` and :data:`REACH_MAX`.

    :param camera_radius: the camera's distance from the planet's centre
    :param base: the slab base's radius
    :param top: the slab top's radius
    """
    inside = math.sqrt(max((top - base) * (top + base), 0.0))
    below = math.sqrt(max((camera_radius - base) * (camera_radius + base), 0.0))
    return min(max(inside + below, REACH_MIN), REACH_MAX)


def light_chart(
    up: np.ndarray, base: float, thickness: float, reach: float, tile: int
) -> list[tuple[float, float, float, float]]:
    """The chart's four vec4s (cloud_field.glsl) centred under *up*."""
    lat, lon = latlon(np.asarray(up, dtype=np.float64)[None])
    frame = TangentFrame.at(float(lat[0]), float(lon[0]))
    return [
        (*map(float, frame.up), float(base)),
        (*map(float, frame.east), float(reach)),
        (*map(float, frame.north), float(tile)),
        (float(thickness), float(SLICES), float(COLUMNS), 0.0),
    ]


def voxel_positions(chart: list, tile: int) -> np.ndarray:
    """Every voxel's planet-frame centre, (SLICES, tile, tile, 3) (slice,
    row, column): cloud_field.glsl's ``lightVoxelAt`` on the CPU."""
    up, east, north = (np.asarray(chart[k][:3], dtype=np.float64) for k in range(3))
    base, reach, thickness = chart[0][3], chart[1][3], chart[3][0]
    t = (np.arange(tile) + 0.5) / tile * 2.0 - 1.0
    tx, ty = np.meshgrid(t, t)  # (row, column): y, x
    rho = np.hypot(tx, ty)
    angle = rho * rho * reach / base
    toward = np.stack([tx, ty], -1) / np.maximum(rho, 1e-12)[..., None]
    dirs = np.cos(angle)[..., None] * up + np.sin(angle)[..., None] * (
        toward[..., :1] * east + toward[..., 1:] * north
    )
    heights = (np.arange(SLICES) + 0.5) / SLICES * thickness
    return dirs[None] * (base + heights)[:, None, None, None]


def light_depth_on_sphere(
    volume: np.ndarray,
    spec: Any,
    optics: Any,
    points: np.ndarray,
    dirs: ArrayLike,
    radius: float,
    max_steps: int,
    shift_at: Any = None,
    depth_at: Any = None,
    spread_at: Any = None,
    octave_mean: float | None = None,
) -> np.ndarray:
    """The bake's march (``cloud_light_bake.frag``'s ``lightDepth``) on the
    CPU: the clouds' optical depth from points (in or under the slab) toward
    *dirs*, to where the ray leaves the slab's top, the weather (*shift_at*,
    *depth_at*, *spread_at*: planet-frame positions (m, 3) -> (m,); None
    none) and the cell octave taken at every step (not once at the chord's
    start, as :func:`noise.cloud_shadow_on_sphere`), steps about
    ``sun_step_length * CLOUD_SHADOW_STEP_SCALE`` long (at most
    *max_steps*).

    :returns: (...) optical depth, 0 above the slab
    """
    points = np.asarray(points, dtype=np.float64)
    dirs = np.broadcast_to(np.asarray(dirs, dtype=np.float64), points.shape)
    if octave_mean is None:
        octave_mean = float(volume.mean())
    base, thickness = spec.slab
    inner, outer = radius + base, radius + base + thickness
    r = np.linalg.norm(points, axis=-1)
    b = np.sum(points * dirs, axis=-1)
    leave = -b + np.sqrt(np.maximum((outer - r) * (outer + r) + b * b, 0.0))
    enter = np.where(
        r < inner, -b + np.sqrt(np.maximum((inner - r) * (inner + r) + b * b, 0.0)), 0.0
    )
    chord = leave - enter
    alive = (r < outer) & (chord > 0.0)
    target = max(optics.sun_step_length * CLOUD_SHADOW_STEP_SCALE, 1.0)
    steps = np.clip(np.ceil(chord / target), 1, max_steps).astype(np.int64)
    step_length = np.where(alive, chord, 0.0) / steps
    start = points + dirs * enter[..., None]
    k = np.arange(max_steps)
    positions = (
        start[..., None, :]
        + dirs[..., None, :] * ((k + 0.5) * step_length[..., None])[..., None]
    )
    h = (np.linalg.norm(positions, axis=-1) - radius - base) / max(thickness, 1e-3)
    coords = positions * np.asarray(spec.noise_scale) + np.asarray(spec.offset)
    pair = fbm(volume, coords, spec, octaves=2, octave_mean=0.0)
    flat = positions.reshape(-1, 3)

    def at(sample):
        if sample is None:
            return np.zeros(positions.shape[:-1])
        return np.reshape(sample(flat), positions.shape[:-1])

    convective = at(depth_at)
    cell = cell_shift(volume, spec, coords, convective, octave_mean)
    gain, bias = shadow_calibration(spec, octave_mean)
    raw = pair * gain + at(shift_at) + cell + bias
    shelf = None
    if spec.anvil and spread_at is not None:
        w = spec.octave_weights
        value = pair + octave_mean * (w[2] + w[3]) + cell + spec.shelf_shift
        shelf = (at(spread_at), value)
    density = _slab_density(spec, h, raw, convective, shelf)
    density = np.where(k < steps[..., None], density, 0.0).sum(axis=-1)
    depth = density * step_length * spec.density * optics.shadow_strength
    return np.where(alive, depth, 0.0)


class _Atlas:
    """One atlas of the slices: its texture, buffer, region and chart."""

    def __init__(self, output: Any, name: str, tile: int, to_ram: bool) -> None:
        width, height = COLUMNS * tile, SLICES // COLUMNS * tile
        self.texture = Texture(name)
        self.texture.setup_2d_texture(
            width, height, Texture.T_half_float, Texture.F_rg16
        )
        for setter in (self.texture.set_wrap_u, self.texture.set_wrap_v):
            setter(SamplerState.WM_clamp)
        self.texture.set_magfilter(SamplerState.FT_linear)
        self.texture.set_minfilter(SamplerState.FT_linear)
        props = FrameBufferProperties()
        props.set_rgba_bits(16, 16, 0, 0)
        props.set_float_color(True)
        engine = output.get_engine()
        self.buffer = engine.make_output(
            output.get_pipe(),
            name,
            -300,  # with the shadow maps: before every camera that reads it
            props,
            WindowProperties.size(width, height),
            GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_refuse_parasite,
            output.get_gsg(),
            output,
        )
        if self.buffer is None:
            raise RuntimeError("no offscreen buffer for the clouds' light volume")
        mode = (
            GraphicsOutput.RTM_copy_ram if to_ram else GraphicsOutput.RTM_bind_or_copy
        )
        self.buffer.add_render_texture(self.texture, mode, GraphicsOutput.RTP_color)
        self.buffer.set_clear_color_active(False)  # slices kept between frames
        self.buffer.set_active(False)
        self.region = self.buffer.make_display_region()
        self.chart = PTA_LVecBase4f.empty_array(4)
        #: (centre's up, sun, steps, layer, reach, base) baked; None: not whole
        self.baked: tuple | None = None

    def set_chart(self, chart: list) -> None:
        for k, row in enumerate(chart):
            self.chart[k] = LVecBase4f(*row)

    def destroy(self, output: Any) -> None:
        output.get_engine().remove_window(self.buffer)


class _Volume:
    """A family's light volume: two atlases (front read, back baked) and
    the bake's card."""

    def __init__(
        self, output: Any, bind: Any, name: str, tile: int, to_ram: bool
    ) -> None:
        self.tile = tile
        self.atlases = [
            _Atlas(output, f"{name}_a", tile, to_ram),
            _Atlas(output, f"{name}_b", tile, to_ram),
        ]
        self.front: _Atlas | None = None  # the whole one the readers read
        self.root = NodePath(name)
        # The field's inputs on a holder, the bake's own sun on the card
        # below it (the deeper input wins): a bake keeps the sun it started
        # with, whatever the live one does meanwhile.
        holder = self.root.attach_new_node("inputs")
        self._bind = bind
        self._holder = holder
        bind(holder)
        maker = CardMaker(name)
        maker.set_frame(-1.0, 1.0, -1.0, 1.0)
        card = holder.attach_new_node(maker.generate())
        card.node().set_bounds(OmniBoundingVolume())
        card.node().set_final(True)
        card.set_two_sided(True)
        card.set_depth_test(False)
        card.set_depth_write(False)
        card.set_shader(_bake_shader())
        self._sun = PTA_LVecBase3f.empty_array(1)
        self._steps = PTA_int.empty_array(1)
        self._chart = PTA_LVecBase4f.empty_array(4)
        card.set_shader_inputs(
            sunDir=self._sun,
            uLightSteps=self._steps,
            uLightChart=self._chart,
            uLightStepScale=STEP_SCALE,
        )
        self.camera = self.root.attach_new_node(Camera(name, OrthographicLens()))
        for atlas in self.atlases:
            atlas.region.set_camera(self.camera)
        self._baking: tuple | None = None  # (atlas, wanted tuple, next slice)

    def rebind(self) -> None:
        """Take the field's inputs again (a new volume) and bake again."""
        self._bind(self._holder)
        for atlas in self.atlases:
            atlas.baked = None
        self._baking = None

    @property
    def back(self) -> _Atlas:
        return self.atlases[1] if self.front is self.atlases[0] else self.atlases[0]

    def wanted(self, want: tuple) -> bool:
        """Whether *want* (centre's up, sun, steps, layer, reach, base's
        radius) differs from what the front holds past the tolerances."""
        if self.front is None or self.front.baked is None:
            return True
        up, sun, steps, layer, reach, base = want
        centre, baked_sun, baked_steps, baked_layer, baked_reach, _ = self.front.baked
        moved = float(np.arccos(np.clip(up @ centre, -1.0, 1.0))) * base
        return (
            moved > RECENTRE * baked_reach
            or abs(reach - baked_reach) > RECENTRE * baked_reach
            or sun_moved(baked_sun, sun)
            or steps != baked_steps
            or layer != baked_layer
        )

    def start(self, want: tuple, chart: list) -> None:
        """Start baking *want* into the back atlas."""
        atlas = self.back
        atlas.baked = None
        atlas.set_chart(chart)
        for k, row in enumerate(chart):
            self._chart[k] = LVecBase4f(*row)
        self._sun[0] = LVecBase3f(*map(float, want[1]))
        self._steps[0] = int(want[2])
        self._baking = (atlas, want, 0)

    def step(self, per_frame: int) -> _Atlas | None:
        """Draw the bake's next slices with this frame.

        :returns: the atlas made whole by the frames before (all its slices
            drawn: swap it in), else None
        """
        for atlas in self.atlases:
            atlas.buffer.set_active(False)
        if self._baking is None:
            return None
        atlas, want, first = self._baking
        if first >= SLICES:
            self._baking = None
            atlas.baked = want
            return atlas
        count = min(per_frame, COLUMNS - first % COLUMNS, SLICES - first)
        rows = SLICES // COLUMNS
        column, row = first % COLUMNS, first // COLUMNS
        atlas.region.set_dimensions(
            column / COLUMNS, (column + count) / COLUMNS, row / rows, (row + 1) / rows
        )
        atlas.buffer.set_active(True)
        self._baking = (atlas, want, first + count)
        return None

    @property
    def baking(self) -> bool:
        return self._baking is not None

    def destroy(self, output: Any) -> None:
        for atlas in self.atlases:
            atlas.destroy(output)
        self.root.remove_node()


class CloudLightVolumes:
    """A planet's clouds' light volumes (see the module): one per family
    that casts a shadow, for the nodes whose shaders light the clouds by
    them (:meth:`bind`).

    :ivar tile: a slice's texels per side; 0 = none
    :ivar per_frame: slices baked a frame
    """

    #: The readers' inputs of each family: (atlas, chart, on).
    _INPUTS = (
        ("cloudLight", "cloudLightChart", "cloudLightOn"),
        ("otherLight", "otherLightChart", "otherLightOn"),
    )

    def __init__(
        self,
        output: Any,
        clouds: Any,
        tile: int = 256,
        per_frame: int = 1,
        steps: int = STEPS,
        to_ram: bool = False,
    ) -> None:
        """
        :param output: the window (its GSG bakes the volumes)
        :param clouds: the planet's clouds (a
            :class:`~world_builder.clouds.planet.PlanetClouds`: a volume a
            family, or a :class:`~world_builder.clouds.sphere.CloudVolume`)
        :param tile: a slice's texels per side (0: none)
        :param per_frame: slices baked a frame
        :param steps: the most steps of a voxel's march
        :param to_ram: copy each atlas to RAM when drawn (tests)
        """
        self._output = output
        self._clouds = clouds
        self._to_ram = to_ram
        self._families = int(getattr(clouds, "shadow_families", 1))
        self._on = [PTA_int.empty_array(1) for _ in range(self._families)]
        self._nodes: list[NodePath] = []
        self._casters_bound: list[Any] = [None] * self._families
        self.volumes: list[_Volume] = []
        self.per_frame = max(1, int(per_frame))
        self.steps = int(steps)
        self.tile = 0
        self.set_tile(tile)

    def bind(self, node: NodePath) -> None:
        """Give a node's shaders the volumes (off until one is whole)."""
        self._nodes.append(node)
        self._bind(node)

    def _bind(self, node: NodePath) -> None:
        for family in range(self._families):
            atlas, chart, on = self._INPUTS[family]
            node.set_shader_input(on, self._on[family])
            if self.volumes and self.volumes[family].front is not None:
                front = self.volumes[family].front
                node.set_shader_inputs(**{atlas: front.texture, chart: front.chart})

    def _rebind_all(self) -> None:
        self._nodes = [node for node in self._nodes if not node.is_empty()]
        for node in self._nodes:
            self._bind(node)

    def _binder(self, family: int):
        clouds = self._clouds
        if family == 0:
            return clouds.bind_shadow_inputs
        return lambda node: clouds.bind_shadow_inputs(node, family=family)

    def _caster(self, family: int):
        clouds = self._clouds
        if hasattr(clouds, "shadow_caster"):
            return clouds.shadow_caster(family)
        return clouds

    def set_tile(self, tile: int) -> None:
        """A new slice size (0: none): the volumes made again."""
        if tile == self.tile:
            return
        self._destroy_volumes()
        self._casters_bound = [None] * self._families
        self.tile = int(tile)
        if self.tile:
            for family in range(self._families):
                suffix = "" if family == 0 else f"_{family}"
                self.volumes.append(
                    _Volume(
                        self._output,
                        self._binder(family),
                        f"cloud_light{suffix}",
                        self.tile,
                        self._to_ram,
                    )
                )
        self._rebind_all()

    def update(self, camera_world: ArrayLike, sun: ArrayLike) -> None:
        """Per frame, before the frame is drawn: each volume's bake
        stepped, swapped in once whole, and started again if the camera,
        the sun or the clouds have moved past what its front holds.

        :param camera_world: the camera, world frame
        :param sun: toward the sun, world frame (unit)
        """
        clouds = self._clouds
        relative = np.asarray(camera_world, dtype=np.float64) - clouds.position
        rho = float(np.linalg.norm(relative))
        up = relative / rho
        sun = np.asarray(sun, dtype=np.float64)
        swapped = False
        for family, volume in enumerate(self.volumes):
            caster = self._caster(family)
            if caster is not self._casters_bound[family]:
                self._casters_bound[family] = caster
                volume.rebind()
            layer = -1 if caster is None else caster.shadow_layer
            if layer < 0 or self.steps <= 0:
                volume.step(self.per_frame)  # nothing more to draw
                volume._baking = None
                self._on[family][0] = 0
                continue
            done = volume.step(self.per_frame)
            if done is not None:
                volume.front = done
                swapped = True
            slab = caster.shadow_field.slab
            base = clouds.radius + slab[0]
            reach = light_reach(rho, base, base + slab[1])
            want = (up, sun.copy(), self.steps, layer, reach, base)
            if not volume.baking and volume.wanted(want):
                volume.start(want, light_chart(up, base, slab[1], reach, self.tile))
                volume.step(self.per_frame)
            self._on[family][0] = int(volume.front is not None)
        if swapped:
            self._rebind_all()

    def _destroy_volumes(self) -> None:
        for volume in self.volumes:
            volume.destroy(self._output)
        self.volumes = []
        for on in self._on:
            on[0] = 0

    def destroy(self) -> None:
        """Remove the volumes' buffers; the readers light by the maps again."""
        self._destroy_volumes()
        self.tile = 0


_SHADER: Shader | None = None


def _bake_shader() -> Shader:
    global _SHADER
    if _SHADER is None:
        _SHADER = Shader.load(
            Shader.SL_GLSL,
            vertex=SHADER_PATH / "cloud_shadow_map.vert",
            fragment=SHADER_PATH / "cloud_light_bake.frag",
        )
    return _SHADER
