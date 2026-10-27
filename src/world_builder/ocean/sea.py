"""
A planet's sea, drawn by its terrain.

The terrain's chunks are the water: their seabed vertices are lifted to the
sea surface (terrain.vert) and shaded as water (terrain.frag). From high up
that is a flat water lit per pixel. :class:`Sea` gives it its near look below
an altitude band (``near_band``, crossfaded through it): the waves of
sea.glsl, filtered by their size on screen, the planar reflection, colour by
depth and shoreline foam, all on the shader inputs of the terrain's root.

The reflection is a camera mirrored across the tangent plane at the camera's
nadir, rendering into an offscreen buffer that sea.glsl samples. The waves
live on that plane, in the coordinates of a chart anchored near the nadir and
re-anchored (a jump in the pattern) once the nadir is ``anchor_range`` away;
their fine octaves' phases are computed here in float64 (:func:`wave_phases`).

Usage (``World`` does this for a body with an ``OceanLook``)::

    sea = Sea(app, terrain, clock=clock)
    # every frame, after the terrain's update:
    sea.update(camera_world)
    sea.set_light(light)  # the light on the water relative to its tuning
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    Camera,
    ClipPlaneAttrib,
    LMatrix3f,
    LMatrix4f,
    LPlane,
    LQuaternionf,
    LVecBase2f,
    LVecBase3f,
    LVecBase4f,
    PlaneNode,
    PTA_float,
    PTA_LMatrix4f,
    PTA_LVecBase2f,
    PTA_LVecBase3f,
    SamplerState,
    Texture,
    Vec3,
)

from world_builder import SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M
from world_builder.depth import REFLECTION_CAMERA_MASK
from world_builder.ocean.ocean import (
    _MAX_WAVE_ITER,
    DEFAULT_EXPOSURE,
    OceanSettings,
    SwellDrift,
    aces_tonemap,
    compute_wave_dirs,
    make_fresnel_texture,
    wave_directions,
    wave_phases,
    wave_tail_variance,
    wave_weight_sum,
)
from world_builder.ocean.swell import Swell
from world_builder.terrain.chart import AzimuthalEquidistantChart, TangentFrame, latlon
from world_builder.utils import compute_next_power_of_2

if TYPE_CHECKING:
    from world_builder.terrain.terrain import Terrain


class Sea:
    """A planet's sea: its terrain's water, with its near look.

    :ivar terrain: the terrain that draws it
    :ivar frame: the tangent plane at the camera's nadir (sea level), under
        render: the mirror's plane and the waves' (``place`` moves it)
    :ivar refl_buffer: the offscreen reflection buffer
    :ivar refl_tex: its texture
    :ivar refl_cam: the mirrored reflection camera's NodePath
    :ivar near: the near look's share this frame (1 below ``near_band``, 0
        above it)
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = OceanSettings

    # What the per-frame work was last done for, so it is skipped when nothing
    # changed: the clock the swell was handed over at, the light, and what
    # the reflection's lens follows (fov, margin, render size).
    _swell_time: float | None = None
    _light: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    _lens_key: tuple | None = None
    _anchor_east: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __init__(
        self,
        app: Any,
        terrain: Terrain,
        clock: Callable[[], float],
        settings: OceanSettings | None = None,
        water_color: ArrayLike = (0.02, 0.06, 0.14),
        ripple_strength: float = 0.02,
        wind_dir: ArrayLike = (1.0, 0.0),
        wind_strength: float = 0.5,
        swell_amplitude: float = 20.0,
        swell_scale: float = 0.004,
        swell_drift: float = 0.1,
        fm_depth: float = 0.2,
        shallow_color: ArrayLike = SHALLOW_WATER_COLOR,
        shallow_depth: float = SHALLOW_WATER_DEPTH_M,
        foam_color: ArrayLike = (0.8, 0.8, 0.8),
        foam_depth: float = 2.5,
        foam_spacing: float = 0.6,
        foam_speed: float = 1.2,
        foam_strength: float = 0.85,
        swell_shallow: float = 5.0,
        swell_deep: float = 60.0,
        near_band: tuple[float, float] = (12000.0, 18000.0),
        anchor_range: float = 50000.0,
        name: str = "sea",
    ) -> None:
        """
        :param app: the ShowBase (``camera``, ``cam``, ``camLens``, ``win``,
            ``graphicsEngine``, ``render``)
        :param terrain: the planet's terrain (its spec must have an ocean)
        :param clock: the time the waves move with, seconds
        :param settings: what the sea costs (:class:`OceanSettings`;
            :meth:`set_settings` changes it); None = HIGH
        :param water_color: deep-water scatter colour (linear)
        :param ripple_strength: how far the waves move the reflection
        :param wind_dir: wind direction on the chart (east, north), unit
        :param wind_strength: how strongly the waves follow the wind; 0 =
            random, 1 = all of them
        :param swell_amplitude: the swell's height in deep water, metres
            (:meth:`set_swell_amplitude`)
        :param swell_scale: the swell's spatial frequency, 1 / metres
        :param swell_drift: its drift speed
        :param fm_depth: how far the swell moves the small waves' phase,
            radians (0 = off)
        :param shallow_color: the water's colour over shallow seabed (linear)
        :param shallow_depth: metres of water over which the deep colour
            takes over (e-folding depth)
        :param foam_color: shoreline foam's colour in daylight (linear)
        :param foam_depth: foam shows under this many metres of water
        :param foam_spacing: metres of depth between two foam lines
        :param foam_speed: how fast they run shoreward, radians a second
        :param foam_strength: foam's opacity at its thickest
        :param swell_shallow: depth, metres, where the swell starts...
        :param swell_deep: ...and where it reaches its amplitude
        :param near_band: altitudes, metres, over which the near look (and
            its reflection pass) fades into the far one
        :param anchor_range: the waves' chart is re-anchored at the nadir
            once it is this far, metres
        :param name: the frame node's name
        """
        self.app = app
        self.terrain = terrain
        self._clock = clock
        self._settings = settings or OceanSettings()
        self.near_band = (float(near_band[0]), float(near_band[1]))
        self.anchor_range = float(anchor_range)
        self.near = 1.0
        self._showing: bool | None = None  # the reflection pass is running
        self._water_color = LVecBase3f(*water_color)
        self._wind_dir = np.asarray(wind_dir, dtype=np.float64)
        self._directions = wave_directions(self._wind_dir, wind_strength)
        self._anchor: AzimuthalEquidistantChart | None = None
        self._uv_scale_set = False

        render = app.render
        self.frame = render.attach_new_node(f"{name}_frame")
        self._make_reflection_camera()
        self.refl_tex, uv_scale = self._make_reflection_buffer(
            self._settings.reflection_scale, self._clear_color()
        )

        root = terrain.root
        terrain.set_shallow_water(shallow_color, shallow_depth)
        # Rewritten every frame: arrays bound once and written in place, so
        # the terrain root's render state is not rebuilt each frame.
        self._frame_inputs = {
            "iTime": PTA_float.empty_array(1),
            "uSeaNear": PTA_float.empty_array(1),
            "uSeaPixelAngle": PTA_float.empty_array(1),
            "uSeaChart": PTA_LVecBase2f.empty_array(1),
            "uSeaReflMVP": PTA_LMatrix4f.empty_array(1),
            "uWavePhase": PTA_float.empty_array(_MAX_WAVE_ITER),
            "iWaterLight": PTA_LVecBase3f.empty_array(1),
            "iWaterSun": PTA_LVecBase3f.empty_array(1),
            "uSwellRotA": PTA_LMatrix4f.empty_array(1),  # mat3s, as Panda takes them
            "uSwellRotB": PTA_LMatrix4f.empty_array(1),
        }
        for axis in ("uSeaOrigin", "uSeaEast", "uSeaNorth", "uSeaUp", "uSeaCamForward"):
            self._frame_inputs[axis] = PTA_LVecBase3f.empty_array(1)
        frame = self._frame_inputs
        frame["iWaterLight"][0] = LVecBase3f(1.0, 1.0, 1.0)
        frame["iWaterSun"][0] = LVecBase3f(0.0, 0.0, 0.0)
        frame["uSeaPixelAngle"][0] = 0.001
        frame["uSeaReflMVP"][0] = LMatrix4f.ident_mat()
        frame["uSwellRotA"][0] = LMatrix4f.ident_mat()
        frame["uSwellRotB"][0] = LMatrix4f.ident_mat()
        # The octaves' phases, written in one go through a view of the array.
        self._phase_view = np.frombuffer(memoryview(frame["uWavePhase"]), np.float32)
        for name_, array in frame.items():
            root.set_shader_input(name_, array)
        root.set_shader_input("iWaterColor", self._water_color)
        root.set_shader_input("iReflectionTex", self.refl_tex)
        root.set_shader_input("uReflUVScale", uv_scale)
        root.set_shader_input("iRippleStrength", float(ripple_strength))
        root.set_shader_input("iWindDir", LVecBase2f(*self._wind_dir))
        root.set_shader_input(
            "iWaveDirs", compute_wave_dirs(LVecBase2f(*self._wind_dir), wind_strength)
        )
        root.set_shader_input("uFresnelLUT", make_fresnel_texture())
        root.set_shader_input("iSwellScale", float(swell_scale))
        self._swell_scale = float(swell_scale)
        self._swell_drift = float(swell_drift)
        self._swell_shallow = float(swell_shallow)
        self._swell_deep = float(swell_deep)
        # The layers drift as the wind blows at the planet's reference point:
        # one axis for the whole planet, whatever the camera.
        spec = terrain.spec
        self._swell_frame = TangentFrame.at(spec.reference_lat, spec.reference_lon)
        frame = self._swell_frame
        self._drift = SwellDrift(
            frame.up,
            frame.east,
            frame.north,
            self._wind_dir,
            self._swell_scale,
            self._swell_drift,
            terrain.radius,
        )
        root.set_shader_input("uFmDepth", float(fm_depth))
        root.set_shader_input("uFoamColor", LVecBase3f(*foam_color))
        root.set_shader_input("uFoamDepth", float(foam_depth))
        root.set_shader_input("uFoamSpacing", float(foam_spacing))
        root.set_shader_input("uFoamSpeed", float(foam_speed))
        root.set_shader_input("uFoamStrength", float(foam_strength))
        root.set_shader_input("uSwellShallow", float(swell_shallow))
        root.set_shader_input("uSwellDeep", float(swell_deep))
        root.set_shader_input("uSeaDebug", 0)
        root.set_shader_input("uWaveOff", 0)
        self.set_swell_amplitude(swell_amplitude)
        self._apply_live(self._settings)

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, camera_world: ArrayLike) -> None:
        """Per frame, after the terrain's update: the near look's share, and
        while it shows, the nadir frame, the mirror, the waves' phases.

        :param camera_world: the camera in the world frame, metres (float64)
        """
        time = float(self._clock())
        if time != self._swell_time:
            self._drift_swell(time)
        cam = np.asarray(camera_world, dtype=np.float64) - self.terrain.position
        distance = float(np.linalg.norm(cam))
        up = cam / distance
        low, high = self.near_band
        altitude = distance - self.terrain.radius
        t = min(max((altitude - low) / max(high - low, 1e-6), 0.0), 1.0)
        self.near = 1.0 - t * t * (3.0 - 2.0 * t)
        if not self.terrain.spec.has_ocean:  # a dry planet (a spec change)
            self.near = 0.0
        frame = self._frame_inputs
        frame["uSeaNear"][0] = self.near
        showing = self.near > 0.0
        if showing != self._showing:
            self._showing = showing
            self.refl_buffer.set_active(showing)
            self.terrain.set_sea_near(showing)
        if not showing:
            return
        self._place(up)
        frame["iTime"][0] = time
        self._phase_view[:] = wave_phases(self._directions, self._chart_xy, time)
        self._mirror()

    def set_light(self, light: ArrayLike, sun: ArrayLike = (0.0, 0.0, 0.0)) -> None:
        """How lit the water is near the camera, relative to the daylight its
        colour was chosen in: (r, g, b) multiplying the water's own colour
        and the colour the reflection is cleared to (seen where it renders
        nothing; 1 = as tuned, 0 = night).

        Given as the sky's share and the sun's, the world's shadows on the
        water (the clouds') dim the sun's share per pixel (``iWaterSun``);
        all in *light*, nothing is shadowed.

        :param light: linear (r, g, b) factors: all the light, or the sky's
            share of it
        :param sun: the sun's share, (r, g, b)
        """
        light = tuple(float(c) for c in light)
        sun = tuple(float(c) for c in sun)
        if (light, sun) == self._light:
            return
        self._light = (light, sun)
        self._frame_inputs["iWaterLight"][0] = LVecBase3f(*light)
        self._frame_inputs["iWaterSun"][0] = LVecBase3f(*sun)
        total = np.asarray(light) + np.asarray(sun)
        self.refl_buffer.set_clear_color(self._clear_color(total))

    @property
    def settings(self) -> OceanSettings:
        """What the sea costs (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: OceanSettings) -> None:
        """Change what the sea costs (e.g. from a settings menu): live, but
        ``reflection_scale`` (the reflection buffer is made again)."""
        changes = self._settings.changes(settings)
        if not changes:
            return
        self._settings = settings
        if "reflection_scale" in changes:
            self._rebuild_reflection()
        self._apply_live(settings)
        self._swell_time = None  # swell_by_depth: hand the swell over again
        self._lens_key = None  # the reflection's fov margin, or a new lens

    def set_swell_amplitude(self, amplitude: float) -> None:
        """The swell's height in deep water, metres, live. The field mostly
        spans 0.1 to 0.5 of it, so crests stand some 0.4 times this above the
        troughs.

        :param amplitude: metres; the swell's wavelength (about
            1 / ``swell_scale``, 250 m) should stay well above it
        """
        self.swell_amplitude = float(amplitude)
        self._swell_time = None  # hand the swell over again
        self.terrain.root.set_shader_input("uSwellAmplitude", self.swell_amplitude)
        # The field is at most 1: the water rises at most this much (chunks
        # uploaded from now on are culled by bounds this much wider).
        self.terrain.sea_swell = self.swell_amplitude

    def set_debug_mode(self, mode: int) -> None:
        """Debug view of the near look (``uSeaDebug`` in sea.glsl): 0 shaded,
        1 normal, 2 reflection UV, 3 fresnel, 4 world grid, 5 reflection UV
        clamp, 6 raw reflection, 7 pre-tonemap clipping, 8 FM phase, 9 share
        of the wave octaves resolved, 10 unresolved slope spread, 11 wave
        octaves evaluated (the cost), 12 depth (black at the waterline,
        white at 50 m)."""
        self.terrain.root.set_shader_input("uSeaDebug", int(mode))

    def destroy(self) -> None:
        """Remove the reflection buffer and the frame; the terrain's water
        keeps its far look."""
        self._frame_inputs["uSeaNear"][0] = 0.0
        self.terrain.set_sea_near(False)
        self.terrain.root.set_shader_input("uSwellAmplitude", 0.0)  # calm water
        self.terrain.sea_swell = 0.0
        self.terrain.set_swell(None)
        self.app.graphicsEngine.remove_window(self.refl_buffer)
        self.frame.remove_node()

    # ── internals ─────────────────────────────────────────────────────────────

    def _drift_swell(self, time: float) -> None:
        """The swell layers' rotations at *time* (:func:`swell_rotations`,
        its constant part computed once: :class:`SwellDrift`), to the shader
        (a Panda matrix multiplies a row: the transposes)."""
        self._swell_time = time
        a, b = self._drift.at(time)
        self.swell_rotations = (a, b)
        settings = self._settings
        self.terrain.set_swell(
            Swell(
                self.swell_amplitude,
                self._swell_scale,
                a,
                b,
                settings.swell_by_depth,
                self._swell_shallow,
                self._swell_deep,
            )
        )
        self._frame_inputs["uSwellRotA"][0] = _mat4(a.T)
        self._frame_inputs["uSwellRotB"][0] = _mat4(b.T)

    def _place(self, up: np.ndarray) -> None:
        """Put the frame on the tangent plane at the nadir's sea level, its
        axes the chart's east and north there."""
        if self._anchor is not None:
            gx, gy = self._anchor.from_dirs(up[None, :])
            gx, gy = float(gx[0]), float(gy[0])
        if self._anchor is None or math.hypot(gx, gy) > self.anchor_range:
            lat, lon = latlon(up[None, :])
            self._anchor = AzimuthalEquidistantChart(
                float(lat[0]), float(lon[0]), self.terrain.radius
            )
            self._anchor_east = tuple(float(c) for c in self._anchor.frame.east)
            gx = gy = 0.0
        self._chart_xy = (gx, gy)
        # Plain floats: numpy's per-call cost dominates on 3-vectors.
        ux, uy, uz = (float(c) for c in up)
        ex, ey, ez = self._anchor_east
        along = ex * ux + ey * uy + ez * uz
        ex, ey, ez = ex - ux * along, ey - uy * along, ez - uz * along
        norm = math.sqrt(ex * ex + ey * ey + ez * ez)
        ex, ey, ez = ex / norm, ey / norm, ez / norm
        n = (uy * ez - uz * ey, uz * ex - ux * ez, ux * ey - uy * ex)
        radius = self.terrain.radius
        origin = self.terrain.to_render((ux * radius, uy * radius, uz * radius))
        render = self.app.render
        quat = LQuaternionf()
        quat.set_from_matrix(LMatrix3f(Vec3(ex, ey, ez), Vec3(*n), Vec3(ux, uy, uz)))
        self.frame.set_quat(render, quat)
        self.frame.set_pos(render, *origin)
        frame = self._frame_inputs
        frame["uSeaOrigin"][0] = LVecBase3f(*origin)
        frame["uSeaEast"][0] = LVecBase3f(ex, ey, ez)
        frame["uSeaNorth"][0] = LVecBase3f(*n)
        frame["uSeaUp"][0] = LVecBase3f(ux, uy, uz)
        frame["uSeaChart"][0] = LVecBase2f(gx, gy)

    def _mirror(self) -> None:
        """Mirror the reflection camera across the frame's plane, follow the
        main lens's field of view, and hand the shader its projection."""
        app, frame = self.app, self.frame
        pos = app.camera.get_pos(frame)
        hpr = app.camera.get_hpr(frame)
        self.refl_cam.set_pos(frame, pos.x, pos.y, -pos.z)
        self.refl_cam.set_hpr(frame, hpr.x, -hpr.y, -hpr.z)
        fov = app.camLens.get_fov()
        scale = self._settings.reflection_fov_scale
        lens = self.refl_cam_node.get_lens()
        size = self._render_size()
        if (fov.x, fov.y, scale, size) != self._lens_key:
            self._lens_key = (fov.x, fov.y, scale, size)
            lens.set_fov(fov.x * scale, fov.y * scale)
            self._frame_inputs["uSeaPixelAngle"][0] = self._pixel_angle(fov.y)
        view = LMatrix4f()
        view.invert_from(self.refl_cam.get_mat(app.render))
        inputs = self._frame_inputs
        inputs["uSeaReflMVP"][0] = view * lens.get_projection_mat()
        forward = app.camera.get_quat(app.render).get_forward()
        inputs["uSeaCamForward"][0] = LVecBase3f(forward)
        # The texture's valid region (buffer / texture size): its real size is
        # known once the buffer is realized.
        if not self._uv_scale_set:
            tex_w, tex_h = self.refl_tex.get_x_size(), self.refl_tex.get_y_size()
            if tex_w > 0 and tex_h > 0:
                self._uv_scale_set = True
                self.terrain.root.set_shader_input(
                    "uReflUVScale",
                    LVecBase2f(
                        self.refl_buffer.get_x_size() / tex_w,
                        self.refl_buffer.get_y_size() / tex_h,
                    ),
                )

    def _apply_live(self, settings: OceanSettings) -> None:
        """The settings that are shader inputs."""
        iterations = settings.wave_iterations
        root = self.terrain.root
        root.set_shader_input("iIterationsNormal", iterations)
        root.set_shader_input("uWaveWeightSum", wave_weight_sum(iterations))
        tail = PTA_float()
        for v in wave_tail_variance(iterations):
            tail.push_back(float(v))
        root.set_shader_input("uTailVar", tail)
        # Pixels per wavelength to phase per pixel; the cut-off is never
        # sharper than the full-strength limit.
        root.set_shader_input("uAAStart", 2.0 * math.pi / settings.aa_full_px)
        root.set_shader_input(
            "uAAEnd", 2.0 * math.pi / min(settings.aa_gone_px, settings.aa_full_px)
        )
        root.set_shader_input("uRoughFresnel", int(settings.rough_fresnel))
        root.set_shader_input("uDepthColourOn", int(settings.depth_colour))
        root.set_shader_input("uFoamOn", int(settings.foam))
        root.set_shader_input("uFoamNoiseOn", int(settings.foam_noise))
        root.set_shader_input("uSwellByDepth", int(settings.swell_by_depth))
        # The reflection's mipmaps cost a regeneration every frame: only kept
        # when the blur reads them.
        blur = float(settings.reflection_blur)
        root.set_shader_input("uReflBlur", blur)
        self.refl_tex.set_minfilter(
            SamplerState.FT_linear_mipmap_linear
            if blur > 0.0
            else SamplerState.FT_linear
        )

    def _make_reflection_camera(self) -> None:
        """The mirrored camera, its lens and its clip plane (made once)."""
        self.refl_cam_node = Camera("sea_reflection")
        # A copy of the main lens, its field of view widened by
        # reflection_fov_scale (a margin against culling at the edges when
        # turning; _mirror follows the main lens's). Near and far are taken
        # here, before a split-depth setup shortens the main lens; the far
        # plane is pushed out, as the mirrored camera sits farther from what
        # it sees.
        lens = self.app.camLens.make_copy()
        fov = self.app.camLens.get_fov()
        scale = self._settings.reflection_fov_scale
        lens.set_fov(fov.x * scale, fov.y * scale)
        lens.set_far(self.app.camLens.get_far() * 4.0)
        self.refl_cam_node.set_lens(lens)
        self.refl_cam_node.set_camera_mask(REFLECTION_CAMERA_MASK)
        self.refl_cam = self.frame.attach_new_node(self.refl_cam_node)
        # Only what is above the water: a clip plane on the camera's initial
        # state (the shaders apply it, through p3d_ClipPlane).
        clip = PlaneNode("sea_clip")
        clip.set_plane(LPlane(0, 0, 1, 0))
        self.clip_np = self.frame.attach_new_node(clip)
        self.refl_cam_node.set_initial_state(
            self.refl_cam_node.get_initial_state().add_attrib(
                ClipPlaneAttrib.make_all_off().add_on_plane(self.clip_np)
            )
        )

    def _make_reflection_buffer(
        self, refl_scale: float, clear_color: LVecBase4f
    ) -> tuple[Texture, LVecBase2f]:
        """The offscreen buffer the reflection camera renders into, sized off
        the 3D render (not the window).

        :returns: its texture, and the provisional UV scale (buffer over
            padded texture size) until the real one is known
        """
        render_w, render_h = self._render_size()
        buf_w = max(1, int(render_w * refl_scale))
        buf_h = max(1, int(render_h * refl_scale))
        refl_tex = Texture("sea_reflection")
        refl_tex.set_wrap_u(Texture.WM_clamp)
        refl_tex.set_wrap_v(Texture.WM_clamp)
        self.refl_buffer = self.app.win.make_texture_buffer(
            "sea_reflection", buf_w, buf_h, refl_tex
        )
        self.refl_buffer.set_sort(-100)
        # Cleared to the water's colour: where the camera renders nothing,
        # the reflection blends into the water.
        self.refl_buffer.set_clear_color(clear_color)
        self.refl_buffer.make_display_region(0, 1, 0, 1).set_camera(self.refl_cam)
        uv_scale = LVecBase2f(
            buf_w / compute_next_power_of_2(buf_w),
            buf_h / compute_next_power_of_2(buf_h),
        )
        return refl_tex, uv_scale

    def _rebuild_reflection(self) -> None:
        """Make the buffer again at the settings' scale, on the same camera,
        keeping whether it is active and its clear colour."""
        old = self.refl_buffer
        active, clear = old.is_active(), old.get_clear_color()
        self.app.graphicsEngine.remove_window(old)
        self.refl_tex, uv_scale = self._make_reflection_buffer(
            self._settings.reflection_scale, clear
        )
        self.refl_buffer.set_active(active)
        self.terrain.root.set_shader_input("iReflectionTex", self.refl_tex)
        self.terrain.root.set_shader_input("uReflUVScale", uv_scale)
        self._uv_scale_set = False

    def _render_size(self) -> tuple[int, int]:
        """:returns: the 3D render's size in pixels: the main camera's display
        region's, else the window's"""
        camera = getattr(self.app, "cam", None)
        if camera is not None and camera.node().get_num_display_regions():
            region = camera.node().get_display_region(0)
            if region.get_pixel_width() > 0:
                return region.get_pixel_width(), region.get_pixel_height()
        return self.app.win.get_x_size(), self.app.win.get_y_size()

    def _pixel_angle(self, fov_y: float) -> float:
        """:returns: a pixel's angle at the screen's centre, radians"""
        height = max(1, self._render_size()[1])
        return 2.0 * math.tan(math.radians(fov_y) / 2.0) / height

    def _clear_color(self, light: ArrayLike = (1.0, 1.0, 1.0)) -> LVecBase4f:
        """The reflection's clear colour: the lit water colour, as the display
        colour the shader decodes the reflection from."""
        lit = np.asarray(self._water_color, dtype=np.float64) * np.asarray(light)
        return LVecBase4f(*aces_tonemap(lit, DEFAULT_EXPOSURE), 1.0)


def _mat4(m: np.ndarray) -> LMatrix4f:
    """:returns: a 3 x 3 matrix in a 4 x 4 one (how Panda passes a mat3)"""
    out = np.eye(4)
    out[:3, :3] = m
    return LMatrix4f(*out.ravel())
