"""
The clouds' light volume (RENDERING_PLAN.md part C; EVE-Redux's
``LightVolume``): for every voxel of the clouds' sky bands around the camera,
the optical depth of all their genera together toward the sun and straight
up, baked on the GPU, for the raymarch to light its clouds by (in place of
the shadow maps' four levels, whose bands showed in towers) and the air's
god rays.

**Chart** (``shaders/cloud_field.glsl``, ``lightVoxelAt``). The cloud shadow maps'
azimuthal-equidistant chart on the slab base's sphere around the camera,
warped radially (a texel at ``t`` in [-1, 1]^2 of its tile lies
``|t|^2 * reach`` from the centre: fine near the camera, coarse toward the
horizon), and :data:`SLICES` shells through each sky band (``march.sky_bands``:
the genera's slabs merged), none through a gap between two: a 500 m stratus
deck keeps the slices of a thin slab of its own, a 9 km convective band
those of its own (``lightSliceHeight``, :func:`slice_heights`). The reach is
how far a sight line inside the bands can see (the base's horizon from the
camera plus the top's), between :data:`REACH_MIN` and :data:`REACH_MAX`.

**Storage.** An atlas of the slices' tiles (:data:`COLUMNS` a row), RG16F:
the optical depth toward the sun (r) and up (g). Depths, not
transmittances: a depth is linear along a ray, so the depth between two
points of a sun ray is the difference of theirs.

**Bake** (``cloud_light_bake.frag``): each voxel marched toward the sun and
up to the highest band's top, every band ahead in its own steps (half a sun
step of its genera's shortest, :data:`STEP_SCALE`), every genus in it at
those samples by its own field and weather, the depths summed (a depth is
linear: the sum of the genera's is the depth of their summed density); the
weather and the cell octave taken at every step (the ground's shadow march
takes them once, where its chord starts: two voxels on one sun ray then
disagreed about the clouds between them, and a tower's face streaked):
:func:`light_depth_union` on the CPU. A few slices of each band a frame
(:attr:`CloudLightVolumes.per_frame`) into a raw atlas; once whole, smoothed
in one frame (``cloud_light_smooth.frag``: each voxel the [1, 2, 1] tent of
its neighbours along the three axes, inside its sky band,
:func:`smooth_volume` on the CPU) into a third atlas, swapped to the front:
the readers never see a half-baked volume, nor one of two suns. The
smoothing: far out a feature a texel across was one deep voxel among shallow
ones, and the readers' reads across it (jittered then) came out as dark
specks in motion (performance.md §28); it keeps such steps soft. Baked again
(change-driven, the shadow maps' rules) when the sun turns past
``utils.SUN_TOLERANCE``, the camera moves :data:`RECENTRE` of the reach, the
reach changes by as much, or the steps or the clouds' layer change; one bake
at a time, the next one started when it is swapped in (a sun swinging faster
than a bake lags by up to two). With all still, nothing a frame.

One volume holds every genus. Each genus's readers take its own sun window
from it and its ``far_shadow`` of what lies beyond (cloud_field.glsl's
``cloudLightSunDepth``): the stratus under a cumulonimbus is in its shadow.
(Two volumes, before, one for the convective clouds and one for the
stratus, read the stratus's shadow from the convective clouds' maps, a
second set of inputs every reader carried.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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

from world_builder.clouds.march import band_masks, sky_bands
from world_builder.clouds.noise import (
    CLOUD_SHADOW_STEP_SCALE,
    _slab_density,
    cell_shift,
    march_pairs,
    mix_pairs,
    recipe_density_from,
    recipe_terms,
    shadow_calibration,
)
from world_builder.terrain.chart import TangentFrame, latlon
from world_builder.utils import sun_moved

SHADER_PATH = Path(__file__).parent / "shaders"
#: Shells through each sky band (a volume of one band: through its slab).
SLICES = 32
#: Tiles a row of the atlas (slices / COLUMNS rows).
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
    up: np.ndarray,
    base: float,
    thickness: float,
    reach: float,
    tile: int,
    bands: int = 1,
) -> list[tuple[float, float, float, float]]:
    """The chart's four vec4s (cloud_field.glsl) centred under *up*: *bands*
    sky bands of :data:`SLICES` slices each."""
    lat, lon = latlon(np.asarray(up, dtype=np.float64)[None])
    frame = TangentFrame.at(float(lat[0]), float(lon[0]))
    return [
        (*map(float, frame.up), float(base)),
        (*map(float, frame.east), float(reach)),
        (*map(float, frame.north), float(tile)),
        (float(thickness), float(SLICES * bands), float(COLUMNS), float(bands)),
    ]


def slice_heights(
    bands: list[tuple[float, float]] | None, thickness: float
) -> np.ndarray:
    """Each slice's middle above the lowest band's base, metres:
    :data:`SLICES` through each band (cloud_field.glsl's
    ``lightSliceHeight``).

    :param bands: the sky bands' (base, thickness) above sea level; None or
        one: the slices through *thickness*
    """
    middles = (np.arange(SLICES) + 0.5) / SLICES
    if not bands or len(bands) == 1:
        return middles * thickness
    lowest = bands[0][0]
    return np.concatenate([b - lowest + middles * t for b, t in bands])


def voxel_positions(
    chart: list, tile: int, bands: list[tuple[float, float]] | None = None
) -> np.ndarray:
    """Every voxel's planet-frame centre, (slices, tile, tile, 3) (slice,
    row, column): cloud_field.glsl's ``lightVoxelAt`` on the CPU.

    :param bands: the sky bands the chart's slices go through (as
        :func:`slice_heights`)
    """
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
    heights = slice_heights(bands, thickness)
    return dirs[None] * (base + heights)[:, None, None, None]


@dataclass(frozen=True)
class GenusShade:
    """One genus as the light volume's CPU twin marches it: its shadow's
    field and optics, and its weather (planet-frame positions (m, 3) ->
    (m,); None none): the fbm's shift, the convective depth, the anvil's
    spread. A recipe genus has its :class:`noise.GenusField` (its field the
    shadow's) and the Worley volume: its recipe's density
    (``genusRecipeDensity``, the footprint one ``sun_step_length``), its
    octaves *octaves*."""

    spec: Any
    optics: Any
    shift_at: Any = None
    depth_at: Any = None
    spread_at: Any = None
    recipe: Any = None
    cells: Any = None
    octaves: int = 4


def light_depth_union(
    volume: np.ndarray,
    genera: list[GenusShade],
    points: np.ndarray,
    dirs: ArrayLike,
    radius: float,
    max_steps: int,
    octave_mean: float | None = None,
) -> np.ndarray:
    """The bake's march (``cloud_light_bake.frag``, cloud_field.glsl's
    ``skyShadowLevels``) on the CPU: the genera's summed optical depth from
    points toward *dirs*, to where the ray leaves the highest sky band's top
    (:func:`march.sky_bands` of their slabs). Each band ahead of a point
    is marched along its own chord in steps about ``sun_step_length *
    CLOUD_SHADOW_STEP_SCALE`` long (its genera's shortest; at most
    *max_steps*), every genus in it at those samples, its weather and cell
    octave taken at every step (not once at the chord's start, as
    :func:`noise.cloud_shadow_on_sphere`).

    :returns: (...) optical depth, 0 above the bands
    """
    points = np.asarray(points, dtype=np.float64)
    dirs = np.broadcast_to(np.asarray(dirs, dtype=np.float64), points.shape)
    if octave_mean is None:
        octave_mean = float(volume.mean())
    slabs = [tuple(genus.spec.slab) for genus in genera]
    bands = sky_bands(slabs)
    r = np.linalg.norm(points, axis=-1)
    b = np.sum(points * dirs, axis=-1)
    k = np.arange(max_steps)
    total = np.zeros(points.shape[:-1])
    for (base, thickness), mask in zip(bands, band_masks(slabs, bands)):
        members = [genus for g, genus in enumerate(genera) if mask >> g & 1]
        inner, outer = radius + base, radius + base + thickness
        leave = -b + np.sqrt(np.maximum((outer - r) * (outer + r) + b * b, 0.0))
        enter = np.where(
            r < inner,
            -b + np.sqrt(np.maximum((inner - r) * (inner + r) + b * b, 0.0)),
            0.0,
        )
        chord = leave - enter
        alive = (r < outer) & (chord > 0.0)
        shortest = min(genus.optics.sun_step_length for genus in members)
        target = max(shortest * CLOUD_SHADOW_STEP_SCALE, 1.0)
        steps = np.clip(np.ceil(chord / target), 1, max_steps).astype(np.int64)
        step_length = np.where(alive, chord, 0.0) / steps
        start = points + dirs * enter[..., None]
        positions = (
            start[..., None, :]
            + dirs[..., None, :] * ((k + 0.5) * step_length[..., None])[..., None]
        )
        counted = k < steps[..., None]
        for genus in members:
            density = _genus_density(volume, genus, positions, radius, octave_mean)
            density = np.where(counted, density, 0.0).sum(axis=-1)
            depth = density * step_length * genus.spec.density
            total += np.where(alive, depth * genus.optics.shadow_strength, 0.0)
    return total


def _genus_density(
    volume: np.ndarray,
    genus: GenusShade,
    positions: np.ndarray,
    radius: float,
    octave_mean: float,
) -> np.ndarray:
    """A genus's density at the march's samples (..., steps, 3), its weather
    and cell octave taken at each (``slabSampleDensity``)."""
    spec = genus.spec
    flat = positions.reshape(-1, 3)

    def at(sample):
        if sample is None:
            return np.zeros(positions.shape[:-1])
        return np.reshape(sample(flat), positions.shape[:-1])

    if genus.recipe is not None:
        recipe = genus.recipe
        terms = recipe_terms(
            volume, genus.cells, recipe, positions, radius, genus.octaves,
            recipe.optics.sun_step_length,
        )  # fmt: skip
        return recipe_density_from(recipe, *terms, at(genus.shift_at))
    base, thickness = spec.slab
    h = (np.linalg.norm(positions, axis=-1) - radius - base) / max(thickness, 1e-3)
    coords = positions * np.asarray(spec.noise_scale) + np.asarray(spec.offset)
    pairs = march_pairs(volume, coords, spec)

    convective = at(genus.depth_at)
    cell = cell_shift(volume, spec, coords, convective, octave_mean)
    gain, bias = shadow_calibration(spec, octave_mean)
    pair, low = mix_pairs(spec, h, convective, pairs)
    raw = pair * gain + at(genus.shift_at) + cell + bias
    shelf = None
    if spec.anvil and genus.spread_at is not None:
        w = spec.octave_weights
        value = low + octave_mean * (w[2] + w[3]) + cell + spec.shelf_shift
        shelf = (at(genus.spread_at), value)
    return _slab_density(spec, h, raw, convective, shelf)


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
    """:func:`light_depth_union` of one genus: the clouds' optical depth
    from points (in or under its slab) toward *dirs*, to where the ray
    leaves the slab's top.

    :returns: (...) optical depth, 0 above the slab
    """
    genus = GenusShade(spec, optics, shift_at, depth_at, spread_at)
    return light_depth_union(
        volume, [genus], points, dirs, radius, max_steps, octave_mean
    )


def smooth_volume(volume: np.ndarray, band_slices: int | None = None) -> np.ndarray:
    """``cloud_light_smooth.frag`` on the CPU: a volume (slice, row, column,
    ...) with each voxel the [1, 2, 1] / 4 tent of its neighbours along each
    of the three axes, the edges clamped (an edge voxel weighs itself in for
    the missing side), across the slices inside each sky band's
    *band_slices* (None: all one band).
    """
    out = np.asarray(volume, dtype=np.float64)
    band = band_slices or out.shape[0]

    def tent(a: np.ndarray, axis: int) -> np.ndarray:
        pad = [(0, 0)] * a.ndim
        pad[axis] = (1, 1)
        padded = np.pad(a, pad, mode="edge")
        n = a.shape[axis]
        take = lambda k: np.take(padded, np.arange(k, k + n), axis=axis)  # noqa: E731
        return 0.25 * take(0) + 0.5 * take(1) + 0.25 * take(2)

    out = np.concatenate(
        [tent(out[k : k + band], 0) for k in range(0, out.shape[0], band)]
    )
    for axis in (1, 2):
        out = tent(out, axis)
    return out


class _Atlas:
    """One atlas of the slices: its texture, buffer, region and chart."""

    def __init__(
        self, output: Any, name: str, tile: int, slices: int, to_ram: bool
    ) -> None:
        width, height = COLUMNS * tile, slices // COLUMNS * tile
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
    """The light volume: three atlases (the front read; the raw bake; the
    smoothed bake, the next front) and the bake's and the smoothing's
    cards.

    :ivar slices: its slices (:data:`SLICES` a sky band)
    """

    def __init__(
        self, output: Any, bind: Any, name: str, tile: int, slices: int, to_ram: bool
    ) -> None:
        self.tile = tile
        self.slices = slices
        self.atlases = [
            _Atlas(output, f"{name}_a", tile, slices, to_ram),
            _Atlas(output, f"{name}_b", tile, slices, to_ram),
            _Atlas(output, f"{name}_c", tile, slices, to_ram),
        ]
        self.front: _Atlas | None = None  # the whole one the readers read
        #: the last bake before its smoothing (tests compare it to the CPU's)
        self.raw: _Atlas | None = None
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
        # The smoothing: a card of its own, seen by its own camera, drawn
        # into the next front from the raw bake.
        self._smooth_root = NodePath(f"{name}_smooth")
        smooth = self._smooth_root.attach_new_node(maker.generate())
        smooth.node().set_bounds(OmniBoundingVolume())
        smooth.node().set_final(True)
        smooth.set_two_sided(True)
        smooth.set_depth_test(False)
        smooth.set_depth_write(False)
        smooth.set_shader(_smooth_shader())
        smooth.set_shader_input("uLayout", LVecBase4f(tile, slices, COLUMNS, SLICES))
        self._smooth_card = smooth
        self._smooth_camera = self._smooth_root.attach_new_node(
            Camera(f"{name}_smooth", OrthographicLens())
        )
        # (raw atlas, target atlas, wanted tuple, next slice: slices = the
        # smoothing's frame)
        self._baking: tuple | None = None

    def rebind(self) -> None:
        """Take the field's inputs again (a new volume) and bake again."""
        self._bind(self._holder)
        for atlas in self.atlases:
            atlas.baked = None
        self._baking = None

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
        """Start baking *want* into the atlases the front is not."""
        raw, target = [atlas for atlas in self.atlases if atlas is not self.front][:2]
        for atlas in (raw, target):
            atlas.baked = None
            atlas.set_chart(chart)
        raw.region.set_camera(self.camera)
        target.region.set_camera(self._smooth_camera)
        target.region.set_dimensions(0.0, 1.0, 0.0, 1.0)
        self._smooth_card.set_shader_input("uRaw", raw.texture)
        self.raw = raw
        for k, row in enumerate(chart):
            self._chart[k] = LVecBase4f(*row)
        self._sun[0] = LVecBase3f(*map(float, want[1]))
        self._steps[0] = int(want[2])
        self._baking = (raw, target, want, 0)

    def step(self, per_frame: int) -> _Atlas | None:
        """Draw the bake's next slices with this frame.

        :returns: the atlas made whole by the frames before (all its slices
            drawn: swap it in), else None
        """
        for atlas in self.atlases:
            atlas.buffer.set_active(False)
        if self._baking is None:
            return None
        atlas, target, want, first = self._baking
        slices = self.slices
        if first > slices:  # smoothed by the frame before
            self._baking = None
            target.baked = want
            return target
        if first == slices:  # all baked: smoothed into the target this frame
            target.buffer.set_active(True)
            self._baking = (atlas, target, want, first + 1)
            return None
        count = min(per_frame, COLUMNS - first % COLUMNS, slices - first)
        rows = slices // COLUMNS
        column, row = first % COLUMNS, first // COLUMNS
        atlas.region.set_dimensions(
            column / COLUMNS, (column + count) / COLUMNS, row / rows, (row + 1) / rows
        )
        atlas.buffer.set_active(True)
        self._baking = (atlas, target, want, first + count)
        return None

    @property
    def baking(self) -> bool:
        return self._baking is not None

    def destroy(self, output: Any) -> None:
        for atlas in self.atlases:
            atlas.destroy(output)
        self.root.remove_node()
        self._smooth_root.remove_node()


class CloudLightVolumes:
    """A planet's clouds' light volume (see the module): one, of all their
    genera, for the nodes whose shaders light the clouds by it
    (:meth:`bind`).

    :ivar volume: the volume, or None (no tile)
    :ivar tile: a slice's texels per side; 0 = none
    :ivar per_frame: slices of each sky band baked a frame
    """

    #: The readers' inputs: (atlas, chart, on).
    _INPUTS = ("cloudLight", "cloudLightChart", "cloudLightOn")

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
        :param output: the window (its GSG bakes the volume)
        :param clouds: the planet's clouds (a
            :class:`~world_builder.clouds.planet.PlanetClouds` or a
            :class:`~world_builder.clouds.march.CloudMarch`: their
            ``shadow_caster()``, or a caster itself)
        :param tile: a slice's texels per side (0: none)
        :param per_frame: slices of each sky band baked a frame
        :param steps: the most steps of a voxel's march in a band
        :param to_ram: copy each atlas to RAM when drawn (tests)
        """
        self._output = output
        self._clouds = clouds
        self._to_ram = to_ram
        self._on = PTA_int.empty_array(1)
        self._nodes: list[NodePath] = []
        self._caster_bound: Any = None
        self.volume: _Volume | None = None
        self.per_frame = max(1, int(per_frame))
        self.steps = int(steps)
        self.tile = 0
        self.set_tile(tile)

    def bind(self, node: NodePath) -> None:
        """Give a node's shaders the volume (off until whole)."""
        self._nodes.append(node)
        self._bind(node)

    def _bind(self, node: NodePath) -> None:
        atlas, chart, on = self._INPUTS
        node.set_shader_input(on, self._on)
        if self.volume is not None and self.volume.front is not None:
            front = self.volume.front
            node.set_shader_inputs(**{atlas: front.texture, chart: front.chart})

    def _rebind_all(self) -> None:
        self._nodes = [node for node in self._nodes if not node.is_empty()]
        for node in self._nodes:
            self._bind(node)

    def _caster(self) -> Any:
        clouds = self._clouds
        if hasattr(clouds, "shadow_caster"):
            return clouds.shadow_caster()
        return clouds

    @staticmethod
    def _bands(caster: Any) -> list[tuple[float, float]]:
        """The caster's sky bands (one: its shadow field's slab)."""
        bands = getattr(caster, "sky_bands", None)
        return list(bands) if bands else [tuple(caster.shadow_field.slab)]

    def set_tile(self, tile: int) -> None:
        """A new slice size (0: none): the volume made again."""
        if tile == self.tile:
            return
        self._destroy_volume()
        self.tile = int(tile)
        if self.tile:
            self._make_volume(self._caster())
        self._rebind_all()

    def _make_volume(self, caster: Any) -> None:
        bands = 1 if caster is None else len(self._bands(caster))
        self._caster_bound = None
        self.volume = _Volume(
            self._output,
            self._clouds.bind_shadow_inputs,
            "cloud_light",
            self.tile,
            SLICES * bands,
            self._to_ram,
        )

    def update(self, camera_world: ArrayLike, sun: ArrayLike) -> None:
        """Per frame, before the frame is drawn: the volume's bake stepped,
        swapped in once whole, and started again if the camera, the sun or
        the clouds have moved past what its front holds.

        :param camera_world: the camera, world frame
        :param sun: toward the sun, world frame (unit)
        """
        if self.volume is None:
            return
        clouds = self._clouds
        relative = np.asarray(camera_world, dtype=np.float64) - clouds.position
        rho = float(np.linalg.norm(relative))
        up = relative / rho
        sun = np.asarray(sun, dtype=np.float64)
        caster = self._caster()
        volume = self.volume
        if caster is not None:
            bands = self._bands(caster)
            if volume.slices != SLICES * len(bands):  # other bands: made again
                self._destroy_volume()
                self._make_volume(caster)
                volume = self.volume
                self._rebind_all()
        if caster is not self._caster_bound:
            self._caster_bound = caster
            volume.rebind()
        layer = -1 if caster is None else caster.shadow_layer
        if layer < 0 or self.steps <= 0:
            volume.step(self.per_frame)  # nothing more to draw
            volume._baking = None
            self._on[0] = 0
            return
        # per_frame slices a band a frame: a bake takes as many frames
        # however many bands.
        per_frame = self.per_frame * len(bands)
        done = volume.step(per_frame)
        if done is not None:
            volume.front = done
        base = clouds.radius + bands[0][0]
        top = clouds.radius + bands[-1][0] + bands[-1][1]
        reach = light_reach(rho, base, top)
        want = (up, sun.copy(), self.steps, layer, reach, base)
        if not volume.baking and volume.wanted(want):
            chart = light_chart(up, base, top - base, reach, self.tile, len(bands))
            volume.start(want, chart)
            volume.step(per_frame)
        self._on[0] = int(volume.front is not None)
        if done is not None:
            self._rebind_all()

    def _destroy_volume(self) -> None:
        if self.volume is not None:
            self.volume.destroy(self._output)
        self.volume = None
        self._on[0] = 0

    def destroy(self) -> None:
        """Remove the volume's buffers; the readers light by the maps again."""
        self._destroy_volume()
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


_SMOOTH_SHADER: Shader | None = None


def _smooth_shader() -> Shader:
    global _SMOOTH_SHADER
    if _SMOOTH_SHADER is None:
        _SMOOTH_SHADER = Shader.load(
            Shader.SL_GLSL,
            vertex=SHADER_PATH / "cloud_shadow_map.vert",
            fragment=SHADER_PATH / "cloud_light_smooth.frag",
        )
    return _SMOOTH_SHADER
