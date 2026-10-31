"""
A planet's low clouds, raymarched (:class:`CloudMarch`): the same density
field, weather, lighting and far field as the billboards and the deck
(:mod:`sphere`), drawn without their overdraw.

Billboards are a raymarch whose steps are quads: every one of them is drawn,
shaded and blended, so their cost grows with the cloud's volume on screen
(15 ms inside the slab on a laptop's GPU). Here each pixel steps along its
own ray instead (``shaders/cloud_march.frag``):

* through the **slab** between two spheres, less the planet: up to two
  segments, from the range's near plane to the scene's depth, so the ground
  and a game's models inside a cloud are hidden by just the cloud in front
  of them, and the camera inside a cloud sees it all around;
* in **steps** :data:`MIN_STEP_FEATURE` of the finest feature long near the
  camera, a share of the distance (:data:`STEP_GROWTH`) farther, all
  lengthened alike when a ray would need more than the budget
  (``CloudSettings.march_steps``), each jittered within its length;
* sampling the **field** with the octaves the billboard shells would have
  there (``CloudSettings.shell_octaves`` over the shells' distance bands),
  and stopping once the cloud is opaque;
* lit per sample as the billboards are (``shaders/cloud_shading.glsl``), the
  air applied once per pixel at the cloud's mean distance;
* **far** away, where a pixel spans more than the features, the deck's
  expected coverage as a homogeneous slab, lit by the deck's averages: the
  march reaches orbit with no deck.

**The passes.** One per depth range of
:class:`~world_builder.depth.SplitDepthCameras`, each at a fraction of the
view's resolution (``CloudSettings.march_resolution``): a camera sharing the
range's lens draws the scene's depth into a texture (everything but the
clouds and the sky, the terrain through its depth-only stage, as
:mod:`half_res` does); a fullscreen card marches into a second buffer; a
card lays that over the range's own pass in the clouds' slot. The far
range's layer goes on in the far pass, the near range's in the near pass:
along a ray the near march's cloud lies in front of the far one's, which is
what premultiplied "over" composes.

The sea's reflection sees the deck alone (:func:`sphere.deck_mesh`, the
billboards' far field), the far layer packed for it.

Design notes: docs/source/clouds.md, "The raymarch".
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    Camera,
    CardMaker,
    ColorBlendAttrib,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    Lens,
    LMatrix4f,
    LVecBase2f,
    LVecBase2i,
    LVecBase3f,
    LVecBase4f,
    NodePath,
    OmniBoundingVolume,
    OrthographicLens,
    PTA_float,
    PTA_int,
    PTA_LVecBase3f,
    PTA_LVecBase4f,
    RenderState,
    SamplerState,
    Shader,
    Texture,
    TransparencyAttrib,
    Vec3,
    WindowProperties,
)

from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.clouds.cloud import (
    CloudSettings,
    CloudType,
    load_cloud_atlas,
    scale_march,
)
from world_builder.clouds.half_res import composite_card, pass_camera, pass_states
from world_builder.clouds.noise import (
    DECK_LIGHT_COS_MIN,
    LAYER_VEC4S,
    MAX_LAYERS,
    NOISE_FEATURE,
    NOISE_SIZE,
    DensityField,
    build_deck_light_texture,
    build_noise_texture,
    pack_layer_params,
)
from world_builder.clouds.sphere import (
    CLOUD_BIN,
    DECK_MAX_OPTICAL_DEPTH,
    ShellSpec,
    _deck_light,
    _ensure_bin,
    _noise,
    _peaks,
    _shaders,
    _weather,
    deck_mesh,
    octave_offset,
    planet_field,
    planet_weather,
    shell_bands,
)
from world_builder.clouds.weather import WeatherSpec, build_weather_texture, shift_table
from world_builder.depth import (
    CLOUD_FAR_BIT,
    CLOUD_NEAR_BIT,
    FAR_BIT,
    HIDE_FROM_CLOUD_PASS,
    HIDE_FROM_FAR,
    HIDE_FROM_NEAR,
    NEAR_BIT,
    REFLECTION_CAMERA_MASK,
    SplitDepthCameras,
)
from world_builder.terrain.cubesphere import grid_for
from world_builder.terrain.planet import PlanetSpec

if TYPE_CHECKING:
    from world_builder.terrain.workers import ChunkExecutor

_SHADERS = Path(__file__).parent / "shaders"
#: The shortest step, near the camera, as a share of the finest octave's
#: feature (~125 m for cumulus: 50 m steps).
MIN_STEP_FEATURE = 0.4
#: Farther, a step's length per metre of distance from the camera.
STEP_GROWTH = 0.01
#: The view ray's extinction over the field's (the sun march keeps the
#: field's own). The billboards' look: each of their quads carries its
#: sample's density over a whole chord, which keeps a cloud's edge crisp;
#: the field's own soft edge, integrated, reads as a blur.
VIEW_EXTINCTION_SCALE = 2.0
#: The edges' billows, which the billboards' sprites carry (their alpha is a
#: lumpy thickness): a billowy noise this many times finer than the field's
#: first octave (~85 m for cumulus)...
DETAIL_SCALE = 24.0
#: ...moving the threshold by up to this share of its window about its mean
#: (the coverage holds). 1 breaks the edges into rags.
DETAIL_EROSION = 0.5
#: The march's buffers' sorts: before the view they are laid over (the
#: half-resolution pass's are -50 and -49).
_FIRST_SORT = -60


# ── The march's geometry (cloud_march.frag's, for the tests) ─────────────────


def sphere_interval(
    origin: ArrayLike, direction: ArrayLike, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Where rays ``origin + t * direction`` are inside a sphere of *radius*
    around the planet's centre (the shader's ``sphereInterval``).

    :param origin: (..., 3) planet frame, metres
    :param direction: (..., 3) unit
    :returns: (t0, t1), each (...): empty where t0 >= t1
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    b = np.einsum("...i,...i->...", origin, direction)
    q = np.linalg.norm(origin - direction * b[..., None], axis=-1)
    inside = q < radius
    h = np.sqrt(np.where(inside, (radius - q) * (radius + q), 0.0))
    return np.where(inside, -b - h, 1.0), np.where(inside, -b + h, -1.0)


def slab_segments(
    origin: ArrayLike,
    direction: ArrayLike,
    radius: float,
    slab: tuple[float, float],
    t_start: float = 0.0,
    t_end: float = math.inf,
) -> np.ndarray:
    """The stretches of a ray in a slab of sky: between the spheres at its
    base and top, before the ray meets the planet, within [*t_start*,
    *t_end*] (the shader's segments).

    :param origin: (3,) planet frame, metres
    :param direction: (3,) unit
    :param radius: the planet's sea-level radius, metres
    :param slab: (base, thickness) above sea level, metres
    :returns: (2, 2) [segment, (t0, t1)]: the second is empty (t0 >= t1)
        unless the ray leaves the slab downward and comes back
    """
    base, thickness = slab
    top = np.array(sphere_interval(origin, direction, radius + base + thickness))
    inner = np.array(sphere_interval(origin, direction, radius + base))
    ground = np.array(sphere_interval(origin, direction, radius))
    if ground[0] < ground[1] and ground[1] > 0.0:
        t_end = min(t_end, max(ground[0], 0.0))
    if inner[0] < inner[1]:
        segments = np.array([[top[0], inner[0]], [inner[1], top[1]]])
    else:
        segments = np.array([top, [1.0, -1.0]])
    segments[:, 0] = np.maximum(segments[:, 0], t_start)
    segments[:, 1] = np.minimum(segments[:, 1], t_end)
    return segments


def steps_over(segment: ArrayLike, min_step: float, growth: float) -> float:
    """About how many steps cover a segment: *min_step* long up to the knee
    (``min_step / growth``), ``growth * t`` past it (``stepsOver``)."""
    t0, t1 = segment
    if t1 <= t0:
        return 0.0
    knee = min_step / growth
    n = 1.0
    if t0 < knee:
        n += (min(t1, knee) - t0) / min_step
    if t1 > knee:
        n += math.log(t1 / max(t0, knee)) / growth
    return n


def step_scale(
    segments: ArrayLike, min_step: float, growth: float, budget: int
) -> float:
    """What every step is lengthened by so a ray's steps fit *budget*.

    :returns: 1 when they fit
    """
    needed = sum(steps_over(s, min_step, growth) for s in segments)
    return max(1.0, needed / budget)


def march_steps(
    segments: ArrayLike,
    min_step: float,
    growth: float,
    budget: int,
    jitter: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """The samples the march takes along a ray, as the shader takes them
    (before it stops at an opaque cloud).

    :param segments: (n, 2) the ray's stretches in the slab
    :param jitter: where in its step each sample lies, [0, 1)
    :returns: (t, dt): each sample's distance and the length it stands for
    """
    k = step_scale(segments, min_step, growth, budget)
    min_step, growth = min_step * k, growth * k
    ts, dts = [], []
    for t0, t1 in segments:
        if t1 <= t0:
            continue
        least = 0.5 * (t1 - t0)
        a = t0
        while a < t1 and len(ts) < budget + 8:
            dt = min(max(min_step, a * growth), least, t1 - a)
            ts.append(a + dt * jitter)
            dts.append(dt)
            a += dt
    return np.array(ts), np.array(dts)


def march_transmittance(
    extinction: np.ndarray, dt: np.ndarray
) -> tuple[float, np.ndarray]:
    """Composite a march's samples front to back, as the shader does.

    :param extinction: (n,) optical depth per metre at each sample (the
        field's extinction times its density)
    :param dt: (n,) the length each stands for
    :returns: the transmittance left, and each sample's weight (its share
        of the pixel: transmittance in front times its own alpha)
    """
    alpha = 1.0 - np.exp2(-extinction * dt)
    transmittance = np.concatenate([[1.0], np.cumprod(1.0 - alpha)])
    return float(transmittance[-1]), transmittance[:-1] * alpha


def min_step(field: DensityField) -> float:
    """:returns: the march's shortest step for a field, metres
    (:data:`MIN_STEP_FEATURE` of its finest octave's feature)"""
    finest = max(field.noise_scale) * field.octave_scales[-1]
    return MIN_STEP_FEATURE * NOISE_FEATURE / finest


# ── Drawing ────────────────────────────────────────────────────────────────


#: Which pixel of each block the march traces, frame after frame, by block
#: size: Bayer orders, so neighbours come far apart in time.
_BAYER = {
    1: ((0, 0),),
    2: ((0, 0), (1, 1), (1, 0), (0, 1)),
    4: tuple(
        divmod(int(v), 4)[::-1]
        for v in np.argsort(
            [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5], kind="stable"
        )
    ),
}
#: A traced pixel's share over its history, by block size: at full
#: resolution every pixel is traced every frame, so it averages longer.
_FRESH_WEIGHT = {1: 0.2, 2: 1.0 / 3.0, 4: 0.5}
#: The camera may move this far between frames and keep its history, metres
#: (more is a jump: a new pose, a teleport).
HISTORY_JUMP = 2000.0


def block_of(resolution: float) -> int:
    """:returns: the view's pixels per march pixel, per axis, for
    ``CloudSettings.march_resolution``"""
    return max(1, round(1.0 / resolution))


def bayer_offset(block: int, frame: int) -> tuple[int, int]:
    """:returns: the pixel of each *block* x *block* block the march traces
    on *frame*"""
    order = _BAYER[block]
    return order[frame % len(order)]


@functools.cache
def _shader(vertex: str, fragment: str) -> Shader:
    return Shader.load(
        Shader.SL_GLSL, vertex=_SHADERS / vertex, fragment=_SHADERS / fragment
    )


def _target(name: str, linear: bool = False) -> Texture:
    texture = Texture(name)
    texture.set_wrap_u(SamplerState.WM_clamp)
    texture.set_wrap_v(SamplerState.WM_clamp)
    filtering = SamplerState.FT_linear if linear else SamplerState.FT_nearest
    texture.set_minfilter(filtering)
    texture.set_magfilter(filtering)
    return texture


def _buffer(
    app: Any,
    output: Any,
    name: str,
    size: tuple[int, int],
    sort: int,
    targets: dict[int, Texture],
) -> GraphicsOutput:
    """An offscreen buffer drawing into *targets* (render texture plane ->
    texture): half-float colour and a half-float aux plane (the cloud's
    distance, in kilometres), or a depth texture."""
    props = FrameBufferProperties()
    if GraphicsOutput.RTP_color in targets:
        props.set_rgba_bits(16, 16, 16, 16)
        props.set_float_color(True)
    else:
        props.set_rgba_bits(8, 8, 8, 8)
    if GraphicsOutput.RTP_aux_hrgba_0 in targets:
        props.set_aux_hrgba(1)
    props.set_depth_bits(24 if GraphicsOutput.RTP_depth in targets else 0)
    buffer = app.graphicsEngine.make_output(
        app.pipe,
        name,
        sort,
        props,
        WindowProperties.size(*size),
        GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_resizeable,
        output.get_gsg(),
        output,
    )
    if buffer is None:
        raise RuntimeError(f"the clouds' raymarch could not make its {name} buffer")
    for plane, texture in targets.items():
        buffer.add_render_texture(texture, GraphicsOutput.RTM_bind_or_copy, plane)
    return buffer


def _fullscreen(buffer: GraphicsOutput, name: str, shader: Shader) -> NodePath:
    """A fullscreen card drawn into *buffer*, in a scene of its own.

    :returns: the card
    """
    scene = NodePath(f"{name}_scene")
    camera = scene.attach_new_node(Camera(f"{name}_camera", OrthographicLens()))
    buffer.make_display_region().set_camera(camera)
    maker = CardMaker(name)
    maker.set_frame(-1.0, 1.0, -1.0, 1.0)
    card = scene.attach_new_node(maker.generate())
    card.node().set_bounds(OmniBoundingVolume())
    card.node().set_final(True)
    card.set_shader(shader)
    card.set_depth_test(False)
    card.set_depth_write(False)
    card.set_two_sided(True)
    return card


class _Resolve:
    """One of a range's two full-resolution results (they alternate: each
    frame's reads the other's as its history)."""

    def __init__(self, app: Any, output: Any, name: str, size, sort: int) -> None:
        self.colour = _target(f"{name}_colour", linear=True)
        self.distance = _target(f"{name}_distance", linear=True)
        self.buffer = _buffer(
            app,
            output,
            name,
            size,
            sort,
            {
                GraphicsOutput.RTP_color: self.colour,
                GraphicsOutput.RTP_aux_hrgba_0: self.distance,
            },
        )
        self.buffer.set_clear_color_active(False)
        self.card = _fullscreen(
            self.buffer, name, _shader("cloud_march.vert", "cloud_resolve.frag")
        )


class _MarchLayer:
    """One depth range's march: its depth pass, its march, its two
    alternating results and the card that lays them over the range's pass.

    A *gathering* layer traces a different pixel of each block every frame
    and gathers them into the view's resolution; the other traces the same
    one and only averages its jitter over the frames, at the march's own
    resolution, upsampled by the composite (the near range: its clouds are
    close, large and soft on screen, and it costs a full-resolution pass
    less).
    """

    def __init__(
        self,
        app: Any,
        output: Any,
        name: str,
        lens: Lens,
        bit: int,
        shown_to: int,
        view: tuple[int, int],
        block: int,
        sort: int,
        states: tuple[RenderState, RenderState],
        gather: bool = True,
    ) -> None:
        self.lens = lens
        self.block = block
        self.gather = gather
        # The scene's depth, at the view's resolution, for the march to stop at.
        self.depth = _target(f"{name}_depth")
        self.depth_buffer = _buffer(
            app,
            output,
            f"{name}_depth",
            view,
            sort,
            {GraphicsOutput.RTP_depth: self.depth},
        )
        self.depth_buffer.set_clear_color_active(False)
        self.depth_buffer.set_clear_depth_active(True)
        self.camera = pass_camera(app, f"{name}_depth", lens, bit, *states)
        self.depth_buffer.make_display_region().set_camera(self.camera)

        # The march: one pixel of every block.
        self.colour = _target(f"{name}_colour")
        self.distance = _target(f"{name}_distance")
        self.buffer = _buffer(
            app,
            output,
            name,
            self._march_size(view),
            sort + 1,
            {
                GraphicsOutput.RTP_color: self.colour,
                GraphicsOutput.RTP_aux_hrgba_0: self.distance,
            },
        )
        self.buffer.set_clear_color_active(True)
        self.buffer.set_clear_color((0.0, 0.0, 0.0, 0.0))
        self.buffer.set_clear_value(
            GraphicsOutput.RTP_aux_hrgba_0, (0.0, 0.0, 0.0, 0.0)
        )
        self.buffer.set_clear_active(GraphicsOutput.RTP_aux_hrgba_0, True)
        self.card = _fullscreen(
            self.buffer, name, _shader("cloud_march.vert", "cloud_march.frag")
        )
        self.card.set_shader_input("uSceneDepth", self.depth)

        # The result, gathered over the frames.
        self.resolves = [
            _Resolve(
                app, output, f"{name}_resolve{i}", self._resolve_size(view), sort + 2
            )
            for i in (0, 1)
        ]
        for resolve, history in zip(self.resolves, self.resolves[::-1]):
            resolve.card.set_shader_inputs(
                uMarch=self.colour,
                uMarchDistance=self.distance,
                uHistory=history.colour,
                uHistoryDistance=history.distance,
                uFreshWeight=_FRESH_WEIGHT[block if gather else 1],
            )

        # Laid over the range's pass at the cloud's depth.
        self.composite = composite_card(app, f"{name}_composite", self.colour, shown_to)
        self.composite.set_shader(
            _shader("cloud_composite.vert", "cloud_composite.frag")
        )
        self.composite.set_depth_test(True)
        # Until the first update: inputs every shader reads must be set.
        identity = LMatrix4f.ident_mat()
        self.card.set_shader_inputs(uClipToRender=identity, uOffset=LVecBase2i(0, 0))
        for resolve in self.resolves:
            resolve.card.set_shader_inputs(
                uClipToRel=identity,
                uPrevRelToClip=identity,
                uOffset=LVecBase2i(0, 0),
                uHistoryValid=0,
            )
        self.composite.set_shader_input("uDistance", self.distance)
        self._view = view
        self.active = True
        self._previous: tuple[Any, np.ndarray] | None = None
        self._set_sizes(view)

    def _march_size(self, view: tuple[int, int]) -> tuple[int, int]:
        return tuple(max(1, math.ceil(v / self.block)) for v in view)

    def _resolve_size(self, view: tuple[int, int]) -> tuple[int, int]:
        return view if self.gather else self._march_size(view)

    def _offset(self, frame: int) -> tuple[int, int]:
        """:returns: the pixel of each block traced on *frame*: in turn when
        gathering, else its middle"""
        if self.gather:
            return bayer_offset(self.block, frame)
        return (self.block // 2, self.block // 2)

    def _set_sizes(self, view: tuple[int, int]) -> None:
        self.card.set_shader_inputs(uBlock=self.block, uViewSize=LVecBase2f(*view))
        size = self._resolve_size(view)
        for resolve in self.resolves:
            resolve.card.set_shader_inputs(
                uBlock=self.block if self.gather else 1, uViewSize=LVecBase2f(*size)
            )
        self.composite.set_shader_input(
            "uDownscale", 1.0 if self.gather else float(self.block)
        )

    def resize(self, view: tuple[int, int]) -> None:
        self._view = view
        self.depth_buffer.set_size(*view)
        self.buffer.set_size(*self._march_size(view))
        for resolve in self.resolves:
            resolve.buffer.set_size(*self._resolve_size(view))
        self._set_sizes(view)
        self._previous = None

    def set_active(self, active: bool) -> None:
        """Run the passes or stop them (and hide the composite); the history
        starts over."""
        self.active = active
        for buffer in (self.depth_buffer, self.buffer):
            buffer.set_active(active)
        for resolve in self.resolves:
            resolve.buffer.set_active(active)
        if active:
            self.composite.show()
        else:
            self.composite.hide()
        self._previous = None

    def update(self, app: Any, frame: int, camera: np.ndarray) -> None:
        """Per frame: this frame's pixel of each block, the range's clip
        space to the render frame (the march) and to the camera's
        surroundings now and a frame ago (the reprojection), and which
        result is written.

        :param frame: the march's frame count
        :param camera: the camera in the planet frame, float64
        """
        offset = self._offset(frame)
        projection = self.lens.get_projection_mat()
        unproject = self.lens.get_projection_mat_inv()
        placed = self.camera.get_mat(app.render)
        turned = LMatrix4f(placed)
        turned.set_row(3, LVecBase3f(0.0, 0.0, 0.0))
        rel_to_clip = LMatrix4f(turned)
        rel_to_clip.invert_in_place()
        rel_to_clip = rel_to_clip * projection
        self.card.set_shader_inputs(
            uClipToRender=unproject * placed, uOffset=LVecBase2i(*offset)
        )

        valid = self._previous is not None
        previous = rel_to_clip
        if valid:
            last_rel_to_clip, last_camera = self._previous
            moved = camera - last_camera
            valid = float(np.linalg.norm(moved)) <= HISTORY_JUMP
            # A point relative to the camera now, as it was relative to the
            # camera then (float64 for the move, float32 only near the camera).
            previous = LMatrix4f.translate_mat(LVecBase3f(*moved)) * last_rel_to_clip
        self._previous = (rel_to_clip, camera.copy())

        write = self.resolves[frame % 2]
        read = self.resolves[1 - frame % 2]
        write.buffer.set_active(True)
        read.buffer.set_active(False)
        write.card.set_shader_inputs(
            uClipToRel=unproject * turned,
            uPrevRelToClip=previous,
            uOffset=LVecBase2i(*offset) if self.gather else LVecBase2i(0, 0),
            uHistoryValid=int(valid),
        )
        self.composite.set_shader_inputs(uClouds=write.colour, uDistance=write.distance)

    def destroy(self, app: Any) -> None:
        self.composite.remove_node()
        self.camera.remove_node()
        for card in [self.card] + [r.card for r in self.resolves]:
            card.get_top().remove_node()
        for buffer in [self.depth_buffer, self.buffer] + [
            r.buffer for r in self.resolves
        ]:
            app.graphicsEngine.remove_window(buffer)


class CloudMarch:
    """A planet's low clouds, raymarched (see the module): the
    ``CloudSettings.renderer == "march"`` volume layer of
    :class:`planet.PlanetClouds`, with :class:`sphere.CloudVolume`'s
    interface and arguments.

    It draws nothing until :meth:`attach` gives it the app and its split
    cameras (``PlanetClouds.use_split_depth``, which a ``World`` calls).

    :ivar root: on the scene: the sea's reflection's deck (hide it to hide
        the reflection; :meth:`set_visible` hides everything)
    :ivar field: the density field, threshold resolved for the planet
    :ivar weather: (6, n, n, 2) cube map of coverage and fbm shift, once set
        up
    :ivar stats: as :class:`sphere.CloudVolume`'s (no tiles nor billboards:
        ``pending`` is 1 until the set-up is done)
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = CloudSettings

    def __init__(
        self,
        parent: NodePath,
        radius: float,
        executor: ChunkExecutor,
        position: ArrayLike = (0.0, 0.0, 0.0),
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        cloud_type: CloudType = CloudType.CUMULUS,
        altitude: float | None = None,
        planet: PlanetSpec | None = None,
        weather: WeatherSpec | None = None,
        seed: int = 7,
        weather_seed: int = 11,
        settings: CloudSettings | None = None,
        sun_brightness: float = 2.8,
        sky_strength: float = 0.35,
        sky_occlusion: float = 0.55,
        exposure: float = 0.9,
        near_fade_radii: float = 2.5,
        channel: str | None = None,
        game: Any = None,
    ):
        """As :class:`sphere.CloudVolume`'s (*near_fade_radii* and
        *channel* are unused: nothing fades near the camera, nothing is
        generated per tile)."""
        settings = settings or CloudSettings()
        _ensure_bin()
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=np.float64)
        self.executor = executor
        self.grid = grid_for(self.radius)
        self.stats = {"tiles": 0, "billboards": 0, "pending": 1, "gen_ms": 0.0}
        self._atmospheres = atmospheres

        preset, field, peaks = planet_field(cloud_type, altitude, seed, self.radius)
        volume = _noise(seed)
        self.field = field
        self._preset = preset
        self._seed = seed
        self._peaks = peaks
        self._octave_mean = float(volume.mean())
        weather = planet_weather(weather, planet, self.radius, weather_seed)
        self._table = tuple(float(v) for v in shift_table(peaks, field.threshold[0]))
        self._weather_job = executor.submit_once(_weather, weather, self._table)
        self.weather: np.ndarray | None = None
        self._settings = settings
        self._layer_far = PTA_float.empty_array(MAX_LAYERS)
        self._layer_params = PTA_LVecBase4f.empty_array(MAX_LAYERS * LAYER_VEC4S)
        params, self._deck_job = self._configure(settings)
        self._write_params(params)

        # The sea's reflection: the deck alone.
        self.root = parent.attach_new_node("cloud_march")
        self.root.stash()  # out of every camera until the set-up is done
        self.root.hide(HIDE_FROM_NEAR | HIDE_FROM_FAR | HIDE_FROM_CLOUD_PASS)
        self.deck = self.root.attach_new_node(
            deck_mesh(self.radius + field.slab[0] + 0.5 * field.slab[1])
        )
        self.deck.set_shader(_shaders()[1])
        self.deck.set_two_sided(True)
        self.deck.set_texture(load_cloud_atlas(game)[0])
        self.root.set_transparency(TransparencyAttrib.M_none)
        self.root.set_depth_test(True)
        self.root.set_depth_write(False)
        self.root.set_bin(CLOUD_BIN, 0)
        self.root.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_one,
                ColorBlendAttrib.O_one_minus_incoming_alpha,
            )
        )

        self._camera_planet = PTA_LVecBase3f.empty_array(1)
        self._camera_render = PTA_LVecBase3f.empty_array(1)
        self._planet_render = PTA_LVecBase3f.empty_array(1)
        self._sun = PTA_LVecBase3f.empty_array(1)
        self._sun[0] = LVecBase3f(0.0, 0.0, 1.0)
        self._steps = PTA_int.empty_array(1)
        self._frame = PTA_int.empty_array(1)
        self._layer_count = PTA_int.empty_array(1)
        self._layer_count[0] = self._layers_used
        self._inputs: dict[str, Any] = {
            "layerParams": self._layer_params,
            "cloudNoise": build_noise_texture(volume),
            "noiseSize": float(NOISE_SIZE),
            "noiseOffset": Vec3(0, 0, 0),
            "deckLightCosMin": DECK_LIGHT_COS_MIN,
            "planetRadius": self.radius,
            "uOwnAtmosphere": -1 if atmosphere is None else int(atmosphere),
            "camPos": self._camera_planet,
            "uCamRender": self._camera_render,
            "uPlanetCentre": self._planet_render,
            "sunDir": self._sun,
            # White: the planet's air colours it (planetCloudLight).
            "sunColor": Vec3(1.0, 1.0, 1.0),
            "skyColor": Vec3(0.45, 0.6, 0.85),
            "sunBrightness": float(sun_brightness),
            "skyStrength": float(sky_strength),
            "skyOcclusion": float(sky_occlusion),
            "exposure": float(exposure),
            "uPixelAngle": 0.001,
            "uCloudDebug": 0,
            "uMarchSteps": self._steps,
            "uMinStep": min_step(field),
            "uStepGrowth": STEP_GROWTH,
            "uLayerFar": self._layer_far,
            "uLayerCount": self._layer_count,
            "uFrame": self._frame,
            "uDetailScale": DETAIL_SCALE,
            "uDetailErosion": DETAIL_EROSION,
            "uBillowMean": float(np.mean(1.0 - np.abs(2.0 * volume - 1.0))),
            # The deck's (cloud_deck.vert, cloud.frag)
            "uDeck": 1,
            "uDeckLayer": self._layers_used - 1,
        }
        self._nodes: list[NodePath] = []
        self._bind(self.root)
        self._app: Any = None
        self._depth: SplitDepthCameras | None = None
        self._layers: list[_MarchLayer] = []
        self._size = (0, 0)
        self._visible = True
        self._pixel_angle = 0.001
        self._sky_light: float | None = None
        self._apply_live(settings)

    # ── wiring ────────────────────────────────────────────────────────────

    def _bind(self, node: NodePath) -> None:
        """Give a node every input the march and the deck read (kept in step
        by :meth:`_set_inputs`)."""
        bind_defaults(node)
        if self._atmospheres is not None:
            self._atmospheres.bind(node)
        node.set_shader_inputs(**self._inputs)
        if self.weather is not None:
            node.set_shader_inputs(**self._textures)
        self._nodes.append(node)

    def _set_inputs(self, **inputs: Any) -> None:
        self._inputs.update(inputs)
        for node in self._nodes:
            node.set_shader_inputs(**inputs)

    def attach(self, app: Any, depth: SplitDepthCameras) -> None:
        """Build the march's passes, one per depth range of *depth* (see the
        module). Build the split cameras first; call again after a new set.

        :param app: the ShowBase (``camera``, ``render``, ``graphicsEngine``,
            ``pipe``)
        :param depth: the game's split cameras
        """
        self.detach()
        self._app, self._depth = app, depth
        self._build_layers()

    def detach(self) -> None:
        """Remove the march's passes (it draws nothing until :meth:`attach`)."""
        for layer in self._layers:
            self._nodes.remove(layer.card)
            layer.destroy(self._app)
        self._layers = []

    def _build_layers(self) -> None:
        app, depth = self._app, self._depth
        self._size = self._view_size()
        block = block_of(self._settings.march_resolution)
        states = pass_states()
        sort = _FIRST_SORT
        for name, lens, bit, shown_to in (
            ("cloud_march_far", depth.far_lens, CLOUD_FAR_BIT, FAR_BIT),
            ("cloud_march_near", app.camLens, CLOUD_NEAR_BIT, NEAR_BIT),
        ):
            layer = _MarchLayer(
                app,
                depth.output,
                name,
                lens,
                bit,
                shown_to,
                self._size,
                block,
                sort,
                states,
                gather=bit == CLOUD_FAR_BIT,
            )
            self._bind(layer.card)
            # Nothing to march until the weather is in (_finish).
            layer.set_active(self._visible and self.weather is not None)
            self._layers.append(layer)
            sort += 4

    def _view_size(self) -> tuple[int, int]:
        region = self._depth.near_region
        return (max(region.get_pixel_width(), 1), max(region.get_pixel_height(), 1))

    # ── settings ──────────────────────────────────────────────────────────

    @property
    def settings(self) -> CloudSettings:
        """What the clouds cost (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: CloudSettings) -> None:
        """Change what the clouds cost: the steps apply at once; the
        octaves per distance and the sun march repack the layers (the
        deck's light is remade for its octaves, ~0.4 s); the resolution
        resizes the passes. A change of renderer is
        :class:`planet.PlanetClouds`' to make."""
        if not self._settings.changes(settings):
            return
        self._finish()
        old = self._settings
        self._settings = settings
        params, deck_job = self._configure(settings)
        self._write_params(params)
        self._set_inputs(uDeckLayer=self._layers_used - 1)
        if old.shell_octaves[-1:] != settings.shell_octaves[-1:]:
            self._set_inputs(deckLight=build_deck_light_texture(deck_job.result()))
        if old.march_resolution != settings.march_resolution:
            self.set_pixel_angle(self._pixel_angle)
            if self._layers:
                self.attach(self._app, self._depth)
        self._apply_live(settings)

    def _apply_live(self, settings: CloudSettings) -> None:
        self._steps[0] = int(settings.march_steps)
        if settings.reflected:
            self.root.show(REFLECTION_CAMERA_MASK)
        else:
            self.root.hide(REFLECTION_CAMERA_MASK)

    def _configure(self, settings: CloudSettings) -> tuple[np.ndarray, Any]:
        """Pack the march's layers: one per billboard shell's distance band
        (its octaves), then the far one, which the reflection's deck shares.

        :returns: the packed layer parameters, and the far layer's light
            table being made in the pool (a future)
        """
        preset, field, peaks, seed = self._preset, self.field, self._peaks, self._seed
        shells = min(settings.shells, MAX_LAYERS - 1)
        schedule = settings.shell_octaves
        _, bands = shell_bands(self.grid, shells)
        optics = scale_march(preset, settings.march_scale).optics
        params = np.zeros((MAX_LAYERS, LAYER_VEC4S, 4), dtype=np.float32)
        far = np.zeros(MAX_LAYERS, dtype=np.float32)
        for k in range(shells + 1):
            octaves = schedule[min(k, len(schedule) - 1)]
            layer = ShellSpec(
                0,
                (0.0, 0.0),
                1.0,
                octaves,
                octave_offset(peaks, _peaks(seed, field, self.radius, octaves)),
            )
            # The far layer is the reflection's deck too: its cap and no fade.
            layer_optics = (
                replace(optics, max_optical_depth=DECK_MAX_OPTICAL_DEPTH)
                if k == shells
                else optics
            )
            params[k] = pack_layer_params(
                layer.field_of(field),
                # Per metre of the view's path
                extinction=field.density * VIEW_EXTINCTION_SCALE,
                octave_mean=self._octave_mean,
                optics=layer_optics,
                octaves=octaves,
            )
            if k < shells:
                far[k] = 0.5 * sum(bands[k])  # mid-band: where the shells cross
        self._layers_used = shells + 1
        self._far_values = far
        deck = layer.field_of(field)
        deck_job = self.executor.submit_once(
            _deck_light,
            seed,
            deck,
            layer_optics,
            self.radius,
            self._table,
            schedule[-1],
        )
        return params, deck_job

    def _write_params(self, params: np.ndarray) -> None:
        for i, row in enumerate(params.reshape(-1, 4)):
            self._layer_params[i] = LVecBase4f(*(float(v) for v in row))
        for i, value in enumerate(self._far_values):
            self._layer_far[i] = float(value)
        if hasattr(self, "_layer_count"):
            self._layer_count[0] = self._layers_used

    # ── public ────────────────────────────────────────────────────────────

    def set_sun(
        self,
        direction: ArrayLike,
        sun_color: ArrayLike | None = None,
        sky_color: ArrayLike | None = None,
    ) -> None:
        """As :meth:`sphere.CloudVolume.set_sun`."""
        d = np.asarray(direction, dtype=float)
        self._sun[0] = LVecBase3f(*(d / np.linalg.norm(d)))
        if sun_color is not None:
            self._set_inputs(sunColor=Vec3(*sun_color))
        if sky_color is not None:
            self._set_inputs(skyColor=Vec3(*sky_color))

    def set_sky_light(self, strength: float) -> None:
        """As :meth:`sphere.CloudVolume.set_sky_light`."""
        if strength != self._sky_light:
            self._sky_light = strength
            self._set_inputs(skyStrength=float(strength))

    def set_pixel_angle(self, radians: float) -> None:
        """The angle one pixel of the VIEW spans (vertical fov / its height):
        the march's own pixels are larger by its resolution.

        :param radians: the angle per pixel of the view
        """
        self._pixel_angle = radians
        value = float(radians / self._settings.march_resolution)
        if value != self._inputs["uPixelAngle"]:
            self._set_inputs(uPixelAngle=value)

    def set_debug(self, flags: int) -> None:
        """Profiling switches (cloud_field.glsl's ``uCloudDebug`` bits)."""
        self._set_inputs(uCloudDebug=int(flags))

    def set_visible(self, visible: bool) -> None:
        """Show or hide everything: the march (its passes stop) and the
        reflection's deck."""
        self._visible = visible
        if visible:
            self.root.show()
        else:
            self.root.hide()
        self.root.hide(HIDE_FROM_NEAR | HIDE_FROM_FAR | HIDE_FROM_CLOUD_PASS)
        for layer in self._layers:
            layer.set_active(visible and self.weather is not None)

    def update(
        self, camera_world: ArrayLike, camera_render: ArrayLike = (0.0, 0.0, 0.0)
    ) -> None:
        """Per frame, once the camera and the split cameras are placed.

        :param camera_world: camera in the world frame, metres (float64)
        :param camera_render: camera in the render frame
        """
        if self._weather_job is not None:
            if not (self._weather_job.done() and self._deck_job.done()):
                self.stats["pending"] = 1
                return
            self._finish()
        cam = np.asarray(camera_world, dtype=np.float64) - self.position
        render_cam = np.asarray(camera_render, dtype=np.float64)
        self._camera_planet[0] = LVecBase3f(*cam)
        self._camera_render[0] = LVecBase3f(*render_cam)
        self._planet_render[0] = LVecBase3f(*(render_cam - cam))
        self.deck.set_pos(*(render_cam - cam))
        if self._layers and self._visible:
            size = self._view_size()
            if size != self._size:
                self._size = size
                for layer in self._layers:
                    layer.resize(size)
            self._frame[0] += 1
            far, near = self._layers
            # The near range marches only when the slab reaches into it.
            near_wanted = self._slab_distance(cam) <= self._depth.split * 1.01
            if near_wanted != near.active:
                near.set_active(near_wanted)
            for layer in self._layers:
                if layer.active:
                    layer.update(self._app, self._frame[0], cam)

    def _slab_distance(self, camera: np.ndarray) -> float:
        """:returns: how far the slab's nearest point is from *camera*
        (planet frame), metres"""
        altitude = float(np.linalg.norm(camera)) - self.radius
        base, thickness = self.field.slab
        return max(base - altitude, altitude - base - thickness, 0.0)

    def remove(self) -> None:
        """Take the clouds off the scene."""
        self.detach()
        self.root.remove_node()

    def _finish(self) -> None:
        """Finish the set-up begun in the pool (waiting for it if need be)."""
        if self._weather_job is None:
            return
        self.weather = self._weather_job.result()
        self._textures = {
            "weatherMap": build_weather_texture(self.weather),
            "deckLight": build_deck_light_texture(self._deck_job.result()),
        }
        self._weather_job = self._deck_job = None
        for node in self._nodes:
            node.set_shader_inputs(**self._textures)
        self._inputs.update(self._textures)
        self.stats["pending"] = 0
        self.root.unstash()
        for layer in self._layers:
            layer.set_active(self._visible)
