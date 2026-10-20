"""
Standalone flythrough demo for the procedural planet terrain.

Flies a free camera around a small planetary system: a Kerbin-sized planet
with hand-made stamps (world_builder/terrain/levels/demo.yaml), the
WorldBuilder ocean and haze, and an airless sand moon 10,000 km away, from
orbit down to the ground of either. Run it from an environment where
world_builder is importable::

    python scripts/demo_terrain.py [--level FILE] [--radius METRES]
        [--moon-distance METRES] [--no-moon] [--workers N]
        [--backend process|thread|inline]

The default backend generates terrain in spawned worker processes, which
re-import this script: keep the ``if __name__ == "__main__"`` guard. All
bodies share one pool.

Frames: positions are float64 in the world frame: the planet's centre at
the origin, the moon elsewhere, no rotation. Each body's terrain works in its
own planet frame (its centre at the origin, same axes). The camera stays at
the render origin and the world is moved around it (see Terrain.to_render),
so float32 never sees a large number near the camera. The camera's
orientation is absolute in the world frame: flying straight ahead from level
flight climbs away from the curving surface, tangentially. "Up", the ground
and the speed follow the nearest body. Yaw turns about the local vertical,
so the horizon stays level while turning; Y re-levels it after a long flight.

Controls (AZERTY):
    Z / S           pitch down / up
    Q / D           roll left / right
    A / E           yaw left / right (about the local vertical)
    Y               level the roll against the local horizon
    UP / DOWN       fly forward / backward
    LEFT / RIGHT    slide left / right
    W / X           slide up / down
    SHIFT           hold to move 5x faster (speed also grows with altitude)
    PAGE UP / DOWN  double / halve the base speed
    HOME            jump to orbit of the nearest body: 3 radii up, looking down
    TAB             jump to the next body: 3 radii up on the side facing you
    END / R         back to the level's start pose

    F1 - F4         planet preset: earthlike / desert / forest / ice
    N               next seed (the planet)
    L               reload the level file (edit stamps live)
    O               toggle the ocean
    C               toggle the clouds
    F5              cycle the cloud quality: high / mid / low / off
    F6              cycle the nebula's palette
    F7 / F8         dimmer / brighter nebula
    M               toggle the terrain in the ocean's reflection
    , / ;           sharper / softer biome transitions
    T               toggle textures (off: flat biome colours)
    B               toggle wireframe
    V               cycle the debug view: shaded / LOD / biomes / slope /
                    normals / skirts
    P               cycle the detail: full, then dropping the finest LOD levels
                    one at a time (vertex spacing doubling each time)
    G               toggle the collision probe: a 20 m sphere ahead of the
                    camera showing its contact point and normal
    SPACE           fire a probe laser along the view; marks the hit
    J / K           swing the sun around the sky (azimuth, here)
    I / U           raise / lower the sun (here)
    F               print the camera pose and terrain stats
    ESC             quit

Terrain is generated in background processes, coarse first: the world shows
blurry for a second, then sharpens.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ClockObject,
    GraphicsWindow,
    LineSegs,
    LMatrix3f,
    LQuaternionf,
    LVecBase3f,
    TextNode,
    TransparencyAttrib,
    Vec3,
    loadPrcFileData,
)

from world_builder.atmosphere import AtmosphereParams, Atmospheres
from world_builder.clouds.cloud import CloudQuality
from world_builder.clouds.sphere import SphericalClouds
from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCameras
from world_builder.sky import NEBULA_PALETTES, Sky, Starfield
from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chart import (
    AzimuthalEquidistantChart,
    TangentFrame,
    latlon,
)
from world_builder.terrain.cubesphere import FACE_NAMES, dirs_to_face
from world_builder.terrain.level import DEMO_LEVEL, load_level
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.quadtree import UNKNOWN_BOUNDS
from world_builder.terrain.terrain import REFLECTION_CAMERA_BITS

INITIAL_FAR_M = 600000.0  # the lens the ocean's reflection camera copies
# The ocean's own haze (it blends to this by distance where no atmosphere
# is bound; the demo binds one).
HAZE_COLOR = (0.62, 0.71, 0.84)
HAZE_DISTANCE = 150000.0
SPEED = 300.0  # metres/second at low altitude
BOOST = 5.0
MIN_CLEARANCE = 3.0  # metres the camera stays above the ground
ORBIT_RADII = 3.0  # HOME and TAB jump this high
MAX_DROPPED_LEVELS = 5  # P cycles the finest drawn level from 0 to this
# The moon: centre-to-centre distance, and where it hangs in the start view
# (this far from the start point's vertical, toward its north).
MOON_DISTANCE_M = 10_000e3
MOON_SEED = 7
MOON_ZENITH_ANGLE = 75.0
# The camera stays within this many moon distances of the planet.
MAX_RANGE_FACTOR = 3.0
# The swell ocean is built for eyes up to 20 km; above this band the terrain
# draws the sea itself, as flat water (hysteresis, so it does not flicker).
OCEAN_HANDOVER = (14000.0, 16000.0)
# The ocean's wave pattern is anchored to the ground through a chart around
# this point, re-anchored (a jump in the pattern) past this distance.
OCEAN_ANCHOR_RANGE = 50000.0
# Sunlight above the air (the atmosphere reddens it), the planet's sky light,
# and an airless body's next to no ambient light (a black sky).
SUN_COLOR = (2.6, 2.5, 2.35)
SKY_AMBIENT = (0.34, 0.44, 0.64)
SPACE_AMBIENT = (0.025, 0.025, 0.03)
CLOUD_BASE = 2200.0  # metres above sea level: above the start pose
HUD_PERIOD = 0.2  # seconds between HUD refreshes
PROBE_RADIUS = 20.0
PROBE_DISTANCE = 60.0
LASER_RANGE = 30000.0

_ROTATIONS = (
    ("s", "z", "pitch", 40.0),
    ("d", "q", "roll", 60.0),
    ("a", "e", "yaw", 40.0),
)
#: Camera-local axes of pitch and roll (Panda's P and R).
_LOCAL_AXES = {"pitch": Vec3(1, 0, 0), "roll": Vec3(0, 1, 0)}
_TRANSLATIONS = (
    ("arrow_up", "arrow_down", 1),  # forward (+Y)
    ("arrow_right", "arrow_left", 0),  # right (+X)
    ("w", "x", 2),  # up (+Z)
)
_SUN_KEYS = (("j", "k", 0, 45.0), ("i", "u", 1, 25.0))
_CONTROLS = (
    "shift",
    *(key for row in (*_ROTATIONS, *_TRANSLATIONS, *_SUN_KEYS) for key in row[:2]),
)
#: F5 cycles the clouds through these.
CLOUD_QUALITIES = ("high", "mid", "low", "off")
_DEBUG_VIEWS = ("shaded", "LOD", "biomes", "slope", "normals", "skirts")


def window_config(
    title: str, size: str = "1600 900", offscreen: bool = False, vsync: bool = True
) -> None:
    """Load the window's PRC settings (before ShowBase opens it).

    No multisampling: the driver's choice of sample count is not ours to
    make (Panda picks the config with the most samples, x16 on a GTX 1070,
    whatever count is asked for), and it was most of the clouds' cost.

    :param size: "width height"
    :param offscreen: an offscreen buffer (no display refresh cap or
        throttling: timings are the GPU's)
    """
    lines = [
        f"window-title {title}",
        f"win-size {size}",
        "audio-library-name null",
        "framebuffer-multisample 0",
        "multisamples 0",
        "depth-bits 24",
    ]
    if not vsync:
        lines.append("sync-video false")
    if offscreen:
        lines.append("window-type offscreen")
    loadPrcFileData("", "\n".join(lines))


def describe_window(demo) -> str:
    """:returns: the window's kind, size and samples per pixel"""
    win = demo.win
    kind = "window" if isinstance(win, GraphicsWindow) else "offscreen"
    samples = max(1, win.get_fb_properties().get_multisamples())
    return f"{kind} {win.get_x_size()}x{win.get_y_size()}, MSAA x{samples}"


def sun_lighting(azimuth: float, elevation: float):
    """Sun direction (local east/north/up) and colours; a hand-made ramp.

    :param azimuth: degrees, 0 = north, increasing toward east
    :param elevation: degrees above the horizon
    :returns: (direction (3,) in east, north, up, sun colour, ambient colour),
        linear
    """
    az, el = math.radians(azimuth), math.radians(elevation)
    direction = np.array(
        [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)]
    )
    height = max(0.0, min(1.0, elevation / 40.0))
    ramp = height * height * (3.0 - 2.0 * height)
    low_sun, high_sun = np.array([2.2, 1.3, 0.6]), np.array([2.6, 2.45, 2.2])
    low_sky, high_sky = np.array([0.30, 0.30, 0.42]), np.array([0.38, 0.48, 0.70])
    sun = low_sun + (high_sun - low_sun) * ramp
    sky = low_sky + (high_sky - low_sky) * ramp
    if elevation < 0.0:
        dim = max(0.05, 1.0 + elevation / 10.0)
        sun, sky = sun * dim, sky * dim
    return direction, tuple(sun), tuple(sky)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _quat_from_axes(x, y, z) -> LQuaternionf:
    """:returns: the rotation taking +X, +Y, +Z to the given unit vectors"""
    q = LQuaternionf()
    q.set_from_matrix(LMatrix3f(Vec3(*x), Vec3(*y), Vec3(*z)))
    return q


@dataclass
class Body:
    """One planet or moon of the demo.

    :ivar name: shown in the HUD
    :ivar terrain: its terrain (which carries its world position)
    :ivar atmosphere: whether it has air (sky, haze, coloured sunlight)
    """

    name: str
    terrain: object
    atmosphere: bool

    @property
    def position(self) -> np.ndarray:
        return self.terrain.position

    @property
    def radius(self) -> float:
        return self.terrain.radius


class TerrainDemo(ShowBase):
    """A free-flying camera around and onto a planet and its moon."""

    def __init__(
        self,
        level_path,
        backend: str,
        radius: float | None,
        moon_distance: float | None = MOON_DISTANCE_M,
        workers: int | None = None,
        cloud_quality: str = "high",
        cloud_debug: int = 0,
    ):
        """
        :param level_path: the planet's level file
        :param backend: chunk generation backend, see ChunkExecutor
        :param radius: overrides the level's planet radius, metres
        :param moon_distance: centre-to-centre, metres; None = no moon
        :param workers: generation processes; None = the machine's default
        :param cloud_quality: one of :data:`CLOUD_QUALITIES`
        :param cloud_debug: the clouds' profiling switches
            (:meth:`SphericalClouds.set_debug`)
        """
        super().__init__()
        self.setFrameRateMeter(True)
        self.disable_mouse()
        self._clock = ClockObject.get_global_clock()
        self.camLens.set_near_far(1.0, INITIAL_FAR_M)
        self.camLens.set_fov(70)
        self.set_background_color(0.0, 0.0, 0.0, 1.0)

        # World frame, float64.
        self.cam_world = np.array([0.0, 0.0, 1.0])
        # Absolute, in the world frame (= the render frame's axes): moving
        # does not turn the camera.
        self._orientation = LQuaternionf()
        self.speed = SPEED

        self.level_path = level_path
        self.radius_override = radius
        self.preset = None
        spec, self.start_pose = load_level(level_path, radius=radius)
        self.seed = spec.seed

        from world_builder.terrain.terrain import Terrain
        from world_builder.terrain.workers import ChunkExecutor

        print("starting terrain workers ...")
        self.executor = ChunkExecutor(backend=backend, workers=workers)
        # The planet's air: everything is seen through it, the moon included.
        self.atmospheres = Atmospheres()
        self.atmospheres.add(spec.radius, AtmosphereParams())
        self.terrain = Terrain(
            parent=self.render,
            spec=spec,
            executor=self.executor,
            atmospheres=self.atmospheres,
            atmosphere=0,
            sun_color=SUN_COLOR,
            ambient=SKY_AMBIENT,
        )
        self.home = Body("planet", self.terrain, atmosphere=True)
        self.bodies = [self.home]
        if moon_distance is not None:
            moon = Terrain(
                parent=self.render,
                spec=replace(PRESETS["moon"], seed=MOON_SEED),
                executor=self.executor,
                atmospheres=self.atmospheres,  # seen through the planet's
                sun_color=SUN_COLOR,
                ambient=SPACE_AMBIENT,
                position=self._moon_position(spec, moon_distance),
                materials=self.terrain.materials,
            )
            self.bodies.append(Body("moon", moon, atmosphere=False))
        self.max_range = MAX_RANGE_FACTOR * (moon_distance or 10.0 * spec.radius)
        self.body = self.home  # the nearest body, updated every frame

        self._sun = np.array([0.0, 0.0, 1.0])  # world frame, toward the sun
        self.sky = Sky(self.render, Starfield(), self.atmospheres)
        self._palette = 0
        self.sky.node.hide(HIDE_FROM_NEAR)  # the far camera draws it, first
        self.clouds = None
        self._clouds_shown = True
        self._cloud_debug = cloud_debug
        self.set_cloud_quality(cloud_quality)
        # Built before the depth split changes the lens: the ocean's
        # reflection camera copies it once.
        self.ocean = self._make_ocean()
        self.depth = SplitDepthCameras(self, near=1.0)
        self._ocean_wanted = True
        self._ocean_low = True  # below the handover band
        self._apply_ocean()
        self._reflected = True
        self._textured = True
        self._wireframe = False
        self._debug = 0
        self._probe = self._make_probe()
        self._laser_marks: list[tuple[object, Body, np.ndarray]] = []
        self._ground = 0.0

        self._keys = dict.fromkeys(_CONTROLS, False)
        for key in _CONTROLS:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])
        presses = (
            ("f1", self._set_preset, ["earthlike"]),
            ("f2", self._set_preset, ["desert"]),
            ("f3", self._set_preset, ["forest"]),
            ("f4", self._set_preset, ["ice"]),
            ("n", self._next_seed, []),
            ("l", self._reload, []),
            ("o", self._toggle_ocean, []),
            ("c", self._toggle_clouds, []),
            ("f5", self._cycle_cloud_quality, []),
            ("f6", self._cycle_nebula_palette, []),
            ("f7", self._scale_nebula, [1 / 1.5]),
            ("f8", self._scale_nebula, [1.5]),
            ("m", self._toggle_reflection, []),
            ("t", self._toggle_textures, []),
            ("b", self._toggle_wireframe, []),
            ("v", self._cycle_debug, []),
            ("p", self._cycle_detail, []),
            ("g", self._toggle_probe, []),
            ("y", self._level_roll, []),
            ("space", self._fire_laser, []),
            (",", self._scale_blend, [0.8]),
            (";", self._scale_blend, [1.25]),
            ("page_up", self._scale_speed, [2.0]),
            ("page_down", self._scale_speed, [0.5]),
            ("home", self._jump_to_orbit, []),
            ("tab", self._jump_to_next_body, []),
            ("end", self._reset_camera, []),
            ("r", self._reset_camera, []),
            ("f", self._print_pose, []),
            ("escape", self._quit, []),
        )
        for key, method, args in presses:
            self.accept(key, method, extraArgs=args)
        if isinstance(self.win, GraphicsWindow):
            self.win.set_close_request_event("window-close")
            self.accept("window-close", self._quit)

        self._hud = OnscreenText(
            text="",
            pos=(-1.32, 0.92),
            scale=0.04,
            fg=(1, 1, 1, 0.9),
            shadow=(0, 0, 0, 0.7),
            align=TextNode.ALeft,
            mayChange=True,
        )
        self._hud_time = -1.0
        self._reset_camera()
        self.taskMgr.add(self._update, "terrain_demo_update")
        print(__doc__)

    # ── frames ─────────────────────────────────────────────────────────────────

    @property
    def radius(self) -> float:
        """The planet's radius."""
        return self.home.radius

    @staticmethod
    def _moon_position(spec, distance: float) -> np.ndarray:
        """Where the moon hangs in the start view: above the northern horizon
        of the level's reference point (the sun, south-west, lights it)."""
        frame = TangentFrame.at(spec.reference_lat, spec.reference_lon)
        angle = math.radians(MOON_ZENITH_ANGLE)
        return distance * (math.cos(angle) * frame.up + math.sin(angle) * frame.north)

    def _relative(self, body: Body | None = None) -> np.ndarray:
        """:returns: the camera in a body's planet frame (the nearest's by default)"""
        return self.cam_world - (body or self.body).position

    def _nearest_body(self) -> Body:
        return min(
            self.bodies,
            key=lambda b: float(np.linalg.norm(self.cam_world - b.position)) - b.radius,
        )

    def _true_frame(self, body: Body | None = None):
        """:returns: (east, north, up) of the geographic frame at the nadir"""
        p = self._relative(body)
        lat, lon = latlon(p[None, :] / np.linalg.norm(p))
        frame = TangentFrame.at(float(lat[0]), float(lon[0]))
        return frame.east, frame.north, frame.up

    def _up(self) -> np.ndarray:
        """:returns: the local vertical of the nearest body at the camera"""
        p = self._relative()
        return p / np.linalg.norm(p)

    def _forward(self) -> np.ndarray:
        f = self._orientation.get_forward()
        return np.array([f.x, f.y, f.z])

    def _terrains(self):
        return [body.terrain for body in self.bodies]

    # ── scene pieces ───────────────────────────────────────────────────────────

    def _make_ocean(self):
        """The game's real curved ocean, in a frame that follows the nadir.

        The ocean is a flat-world object (sea level at z = 0, droop from the
        camera). Its frame is the tangent plane at the sea-level point under
        the camera, so it is exact where it matters; its xy follow the ground
        (a chart around an anchor), so the waves stay put.
        """
        from world_builder.ocean import Ocean

        self._sea_frame = self.render.attach_new_node("sea_frame")
        self._ocean_anchor = None
        self._game = SimpleNamespace(
            app=self,
            root_node=self._sea_frame,
            method_lists={},
            game_time=SimpleNamespace(get_current_time=self._clock.get_frame_time),
        )
        ocean = Ocean(
            game=self._game,
            geometric_swell=True,
            # The game's 20 m swell lifts the water up to 20 m near the camera,
            # drowning the beach band; a gentler swell keeps the coast readable.
            swell_amplitude=3.0,
            haze_color=LVecBase3f(*HAZE_COLOR),
            haze_distance=HAZE_DISTANCE,
            # The terrain is a true sphere: the paraboloid would stand 58 m
            # above its sea level 100 km out.
            planet_radius=self.radius,
            exact_sphere=True,
        )
        self.atmospheres.bind(ocean.ocean_node)
        return ocean

    def _make_probe(self):
        # Under render, re-placed every frame from planet-frame positions.
        probe = self.render.attach_new_node("probe")
        ball = self.loader.load_model("models/misc/sphere")
        ball.set_scale(PROBE_RADIUS)
        ball.set_color(0.3, 0.7, 1.0, 0.35)
        ball.set_transparency(TransparencyAttrib.M_alpha)
        ball.reparent_to(probe)
        self._probe_ball = ball
        self._probe_hit = self.loader.load_model("models/misc/sphere")
        self._probe_hit.set_scale(1.5)
        self._probe_hit.set_color(1.0, 0.2, 0.1, 1.0)
        self._probe_hit.reparent_to(probe)
        self._probe_normal = probe.attach_new_node("normal")
        probe.hide()
        return probe

    # ── per frame ──────────────────────────────────────────────────────────────

    def _update(self, task):
        dt = min(self._clock.get_dt(), 0.1)
        self.body = self._nearest_body()
        self._fly(dt)
        self.camera.set_pos(0.0, 0.0, 0.0)
        self.camera.set_quat(self.render, self._orientation)

        for terrain in self._terrains():
            terrain.update(self.cam_world)
        ranges = []
        for body in self.bodies:
            spec = body.terrain.spec
            ranges.append(
                (
                    body.radius,
                    float(np.linalg.norm(self._relative(body))),
                    UNKNOWN_BOUNDS[1],
                    0.0 if spec.has_ocean else UNKNOWN_BOUNDS[0],
                )
            )
        split, far = SplitDepthCameras.system_ranges(ranges)
        self.depth.update(split, far)

        home_up = self._relative(self.home) / np.linalg.norm(self._relative(self.home))
        home_altitude = float(np.linalg.norm(self._relative(self.home))) - self.radius
        self.atmospheres.update(self.cam_world)
        self.sky.update((0.0, 0.0, 0.0), far)
        if self.clouds is not None and self._clouds_shown:
            fov = math.radians(self.camLens.get_fov()[1])
            self.clouds.set_pixel_angle(fov / max(self.win.get_y_size(), 1))
            self.clouds.update(self.cam_world)
        self._update_sun()
        self._update_ocean(home_altitude, home_up)
        if not self._probe.is_hidden():
            self._update_probe()
        for mark, body, point in self._laser_marks:
            mark.set_pos(*body.terrain.to_render(point))
        # Text geometry is rebuilt on every change: a few times a second is
        # plenty to read, and saves ~1 ms a frame.
        now = self._clock.get_frame_time()
        if now - self._hud_time >= HUD_PERIOD:
            self._hud_time = now
            self._update_hud()
        return task.cont

    def _fly(self, dt: float):
        for plus, minus, kind, rate in _ROTATIONS:
            sign = self._keys[plus] - self._keys[minus]
            if sign:
                self._rotate(kind, sign * rate * dt)
        for plus, minus, axis, rate in _SUN_KEYS:
            sign = self._keys[plus] - self._keys[minus]
            if sign:
                self._turn_sun(axis, sign * rate * dt)

        body = self.body
        relative = self._relative(body)
        ground = float(body.terrain.query.height_at(relative, generate=False))
        above = max(float(np.linalg.norm(relative)) - body.radius - ground, 0.0)
        # Speed grows with height above the nearest ground: a descent from
        # orbit is exponential, a few seconds per factor of e, and the trip
        # to the moon is quick.
        speed = (
            self.speed
            * max(1.0, above / 300.0)
            * (BOOST if self._keys["shift"] else 1.0)
        )
        move = np.zeros(3)
        for plus, minus, axis in _TRANSLATIONS:
            move[axis] = self._keys[plus] - self._keys[minus]
        if move.any():
            world = self._orientation.xform(Vec3(*move))
            self.cam_world = self.cam_world + np.array(world) * speed * dt
        # Stay above the nearest ground and within range of the planet.
        relative = self._relative(body)
        rho = float(np.linalg.norm(relative))
        low = body.radius + ground + MIN_CLEARANCE
        if rho < low:
            self.cam_world = body.position + relative * (low / rho)
        distance = float(np.linalg.norm(self.cam_world))
        if distance > self.max_range:
            self.cam_world *= self.max_range / distance
        self._ground = ground

    def _update_sun(self):
        # Sunlight above the air: the atmosphere reddens it where it must.
        self.sky.set_sun(self._sun)
        if self.clouds is not None:
            self.clouds.set_sun(self._sun)
        for body in self.bodies:
            ambient = SKY_AMBIENT if body.atmosphere else SPACE_AMBIENT
            body.terrain.set_sun(tuple(self._sun), SUN_COLOR, ambient)

    def set_local_sun(self, azimuth: float, elevation: float):
        """Put the sun at an azimuth/elevation (degrees) as seen from the
        nadir on the nearest body."""
        east, north, up = self._true_frame()
        d, _, _ = sun_lighting(azimuth, elevation)
        self._sun = d[0] * east + d[1] * north + d[2] * up

    def _turn_sun(self, axis: int, degrees: float):
        """Swing the sun in the sky as seen from the nadir (axis 0 azimuth,
        1 elevation)."""
        east, north, up = self._true_frame()
        s = self._sun
        el = math.degrees(math.asin(np.clip(s @ up, -1.0, 1.0)))
        az = math.degrees(math.atan2(s @ east, s @ north))
        if axis == 0:
            az += degrees
        else:
            el = max(-10.0, min(89.0, el + degrees))
        d, _, _ = sun_lighting(az, el)
        self._sun = d[0] * east + d[1] * north + d[2] * up

    def _update_ocean(self, altitude: float, up: np.ndarray):
        if altitude > OCEAN_HANDOVER[1]:
            low = False
        elif altitude < OCEAN_HANDOVER[0]:
            low = True
        else:
            low = self._ocean_low
        if low != self._ocean_low:
            self._ocean_low = low
            self._apply_ocean()
        if self.ocean.base_node.is_hidden():
            return
        # Ground-anchored xy: a chart around an anchor near the nadir.
        if self._ocean_anchor is not None:
            gx, gy = self._ocean_anchor.from_dirs(up[None, :])
            gx, gy = float(gx[0]), float(gy[0])
        if self._ocean_anchor is None or math.hypot(gx, gy) > OCEAN_ANCHOR_RANGE:
            lat, lon = latlon(up[None, :])
            self._ocean_anchor = AzimuthalEquidistantChart(
                float(lat[0]), float(lon[0]), self.radius
            )
            gx = gy = 0.0
        frame = self._ocean_anchor.frame
        e = frame.east - up * (frame.east @ up)
        e /= np.linalg.norm(e)
        n = np.cross(up, e)
        sea_point = self.terrain.to_render(up * self.radius)
        self._sea_frame.set_quat(self.render, _quat_from_axes(e, n, up))
        self.ocean.set_atmosphere_frame(
            self.ocean.base_node.get_mat(self.render), self._sun
        )
        self._sea_frame.set_pos(*(sea_point - gx * e - gy * n))
        for methods in list(self._game.method_lists.values()):
            for method in methods:
                method()

    def _update_probe(self):
        terrain = self.body.terrain
        center = self._relative() + PROBE_DISTANCE * self._forward()
        self._probe_ball.set_pos(*terrain.to_render(center))
        contact = terrain.query.sphere_contact(center, PROBE_RADIUS)
        self._probe_normal.remove_node()
        if contact is None:
            self._probe_hit.hide()
            self._probe_ball.set_color(0.3, 0.7, 1.0, 0.35)
            self._probe_normal = self._probe.attach_new_node("normal")
            return
        hit, normal = contact
        hit_render = terrain.to_render(hit)
        self._probe_hit.show()
        self._probe_hit.set_pos(*hit_render)
        self._probe_ball.set_color(1.0, 0.5, 0.2, 0.35)
        lines = LineSegs()
        lines.set_thickness(3)
        lines.set_color(1.0, 1.0, 0.2, 1.0)
        lines.move_to(*hit_render)
        lines.draw_to(*(hit_render + normal * 25.0))
        self._probe_normal = self._probe.attach_new_node(lines.create())

    def _update_hud(self):
        body = self.body
        terrain = body.terrain
        relative = self._relative()
        altitude = float(np.linalg.norm(relative)) - body.radius
        up = relative / np.linalg.norm(relative)
        lat, lon = latlon(up[None, :])
        face, _, _ = dirs_to_face(up[None, :])
        weights = terrain.query.biome_weights_at(relative, generate=False)
        biome_text = "  ".join(
            f"{name} {w:.0%}" for name, w in zip(BIOMES, weights) if w >= 0.05
        )
        s = terrain.stats
        spec = terrain.spec
        heading, pitch = self._true_heading_pitch()
        above = altitude - self._ground
        face_name = FACE_NAMES[int(face[0])]
        speed = self.speed / SPEED
        split_km, far_km = self.depth.split / 1e3, self.depth.far / 1e3
        cs = self.clouds.stats if self.clouds is not None else None
        clouds = (
            f"clouds {self._cloud_quality}: {cs['tiles']} tiles, "
            f"{cs['billboards']:,} billboards, {cs['pending']} generating "
            f"({cs['gen_ms']:.0f} ms/tile)"
            if cs is not None
            else "clouds off"
        )
        others = "   ".join(
            f"{b.name} {np.linalg.norm(self._relative(b)) / 1e3:,.0f} km "
            f"({b.terrain.stats['shown']} chunks)"
            for b in self.bodies
            if b is not body
        )
        lines = [
            f"{body.name}: {spec.name}  seed {spec.seed}  radius "
            f"{spec.radius / 1e3:,.0f} km   view: {_DEBUG_VIEWS[self._debug]}"
            f"   blend {self.terrain.blend_depth:.2f}"
            f"   detail: {self._detail_text()}",
            f"altitude {altitude:,.0f} m   above ground {above:,.0f} m"
            f"   lat {lat[0]:.4f}  lon {lon[0]:.4f}  face {face_name}"
            f"   heading {heading:.0f}  pitch {pitch:.0f}   speed x{speed:g}",
            f"under camera: {biome_text or '(not generated yet)'}"
            + (f"   |   {others}" if others else ""),
            f"chunks: {s['shown']} shown (finest level {s['finest']}), "
            f"{s['uploaded']} on GPU, {s['stored']} stored, "
            f"{s['pending']} generating, {s['wanted']} wanted",
            f"gen {s['gen_ms']:.0f} ms/chunk "
            f"({self.executor.workers} workers)   "
            f"upload {s['upload_ms']:.2f} ms/chunk   LOD pass {s['select_ms']:.1f} ms"
            f"   depth split {split_km:,.1f} km, far {far_km:,.0f} km",
            clouds + ("" if self._clouds_shown else " (hidden)"),
        ]
        self._hud.setText("\n".join(lines))

    def _true_heading_pitch(self) -> tuple[float, float]:
        """:returns: the view's heading (degrees counter-clockwise from north,
        as Panda's H) and pitch, in the nearest body's geographic frame"""
        east, north, up = self._true_frame()
        f = self._forward()
        heading = math.degrees(math.atan2(-(f @ east), f @ north))
        pitch = math.degrees(math.asin(np.clip(f @ up, -1.0, 1.0)))
        return heading, pitch

    # ── key handlers ───────────────────────────────────────────────────────────

    def _set_key(self, key, pressed):
        self._keys[key] = pressed

    def _load(self, preset=None):
        return load_level(
            self.level_path, preset=preset, seed=self.seed, radius=self.radius_override
        )

    def _set_spec(self, spec):
        """Replace the planet (the moon stays) and its weather."""
        self.terrain.set_spec(spec)
        self.set_cloud_quality(self._cloud_quality)
        for mark, body, _ in list(self._laser_marks):
            if body is self.home:
                mark.remove_node()
                self._laser_marks.remove((mark, body, _))
        self._apply_ocean()

    def _set_preset(self, name):
        self.preset = name
        spec, _ = self._load(name)
        self._set_spec(spec)

    def _next_seed(self):
        self.seed += 1
        spec, _ = self._load(self.preset)
        self._set_spec(spec)

    def _reload(self):
        try:
            spec, self.start_pose = self._load(self.preset)
        except Exception as error:  # a typo in the YAML must not kill the demo
            print(f"level reload failed: {error}")
            return
        self._set_spec(spec)
        print(f"reloaded {self.level_path}")

    def _apply_ocean(self):
        has_sea = self._ocean_wanted and self.terrain.spec.has_ocean
        swell = has_sea and self._ocean_low
        if swell:
            self.ocean.base_node.show()
            self.ocean.refl_buffer.set_active(True)
        else:
            self.ocean.base_node.hide()
            self.ocean.refl_buffer.set_active(False)
        self.terrain.set_sea_surface(has_sea and not self._ocean_low)

    def _toggle_ocean(self):
        self._ocean_wanted = not self._ocean_wanted
        self._apply_ocean()

    def _toggle_clouds(self):
        self._clouds_shown = not self._clouds_shown
        self._apply_clouds()

    def _apply_clouds(self):
        if self.clouds is None:
            return
        self.clouds.root.show()
        # Not in the ocean's mirror: it would draw every billboard twice.
        self.clouds.root.hide(REFLECTION_CAMERA_BITS)
        if not self._clouds_shown:
            self.clouds.root.hide()

    def set_cloud_quality(self, name: str):
        """Rebuild the planet's clouds at a quality level (or none: "off").
        The new clouds take over the old ones' worker channel, which drops
        their tiles; the new tiles arrive in a few seconds."""
        if self.clouds is not None:
            self.clouds.remove()
            self.clouds = None
        self._cloud_quality = name
        if name != "off":
            self.clouds = SphericalClouds(
                self.render,
                self.radius,
                self.executor,
                atmospheres=self.atmospheres,
                atmosphere=0,
                altitude=CLOUD_BASE,
                planet=self.terrain.spec,
                quality=CloudQuality(name),
                channel="clouds",
            )
            self.clouds.set_debug(self._cloud_debug)
            self.clouds.set_sun(self._sun)
            self._apply_clouds()

    def _cycle_nebula_palette(self):
        self._palette = (self._palette + 1) % len(NEBULA_PALETTES)
        name = list(NEBULA_PALETTES)[self._palette]
        self.sky.set_nebula(replace(self.sky.nebula, palette=NEBULA_PALETTES[name]))
        print(f"nebula: {name}, intensity {self.sky.nebula.intensity:.3g}")

    def _scale_nebula(self, factor):
        intensity = self.sky.nebula.intensity * factor
        self.sky.set_nebula(replace(self.sky.nebula, intensity=intensity))
        print(f"nebula intensity {intensity:.3g}")

    def _cycle_cloud_quality(self):
        index = CLOUD_QUALITIES.index(self._cloud_quality)
        self.set_cloud_quality(CLOUD_QUALITIES[(index + 1) % len(CLOUD_QUALITIES)])

    def _toggle_reflection(self):
        self._reflected = not self._reflected
        for terrain in self._terrains():
            terrain.set_reflected(self._reflected)

    def _toggle_textures(self):
        self._textured = not self._textured
        for terrain in self._terrains():
            terrain.set_textured(self._textured)

    def _toggle_wireframe(self):
        self._wireframe = not self._wireframe
        for terrain in self._terrains():
            terrain.set_wireframe(self._wireframe)

    def _cycle_debug(self):
        self.set_debug_view((self._debug + 1) % len(_DEBUG_VIEWS))

    def set_debug_view(self, view: int):
        """0 shaded, 1 LOD, 2 biomes, 3 slope, 4 normals, 5 skirts (all bodies)."""
        self._debug = view
        for terrain in self._terrains():
            terrain.set_debug_mode(view)

    def _cycle_detail(self):
        level = (self.terrain.min_level + 1) % (MAX_DROPPED_LEVELS + 1)
        for terrain in self._terrains():
            terrain.set_min_level(level)

    def _detail_text(self) -> str:
        terrain = self.body.terrain
        level = terrain.min_level
        text = f"{terrain.grid.spacing(level):.1f} m"
        return f"full ({text})" if level == 0 else f"-{level} levels ({text})"

    def _toggle_probe(self):
        if self._probe.is_hidden():
            self._probe.show()
        else:
            self._probe.hide()

    def _rotate(self, kind: str, degrees: float):
        """Turn the camera: yaw about the local vertical, pitch and roll about
        the camera's own right and forward axes."""
        q = LQuaternionf()
        if kind == "yaw":
            q.set_from_axis_angle(degrees, Vec3(*self._up()))
            self._orientation = self._orientation * q  # a world-frame turn
        else:
            q.set_from_axis_angle(degrees, _LOCAL_AXES[kind])
            self._orientation = q * self._orientation  # a camera-frame turn
        self._orientation.normalize()

    def _level_roll(self):
        """Keep the view direction, bring the camera's up into the plane of
        it and the local vertical (no roll)."""
        forward = self._forward()
        right = np.cross(forward, self._up())
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            return  # looking straight up or down: any roll is level
        right /= norm
        self._orientation = _quat_from_axes(right, forward, np.cross(right, forward))

    def _fire_laser(self):
        body = self.body
        direction = self._forward()
        start = self._relative(body)
        hit = body.terrain.query.segment_intersect(
            start, start + LASER_RANGE * direction
        )
        if hit is None:
            print("laser: no hit within range")
            return
        t, point, normal = hit
        distance = t * LASER_RANGE
        material = body.terrain.query.material_at(point, generate=False)
        height = float(np.linalg.norm(point)) - body.radius
        up = point / np.linalg.norm(point)
        print(
            f"laser: hit the {body.name} at {distance:,.0f} m, {material}, "
            f"height {height:.1f} m, slope "
            f"{math.degrees(math.acos(np.clip(normal @ up, -1.0, 1.0))):.0f} deg"
        )
        mark = self.loader.load_model("models/misc/sphere")
        mark.set_scale(max(2.0, distance * 0.004))
        mark.set_color(1.0, 0.9, 0.1, 1.0)
        mark.set_light_off()
        mark.reparent_to(self.render)
        self._laser_marks.append((mark, body, point))

    def _scale_blend(self, factor):
        depth = max(0.05, min(3.0, self.terrain.blend_depth * factor))
        for terrain in self._terrains():
            terrain.set_blend_depth(depth)

    def _scale_speed(self, factor):
        self.speed = max(10.0, min(SPEED * 64, self.speed * factor))

    def _place_camera(
        self, lat, lon, altitude, heading, pitch, body: Body | None = None
    ):
        """Put the camera over (lat, lon) of a body (the planet by default),
        with heading and pitch in the local geographic frame (degrees; heading
        counter-clockwise from north, as Panda's H) and no roll."""
        body = body or self.home
        frame = TangentFrame.at(lat, lon)
        self.cam_world = body.position + frame.up * (body.radius + altitude)
        local = LQuaternionf()
        local.set_hpr((heading, pitch, 0.0))
        self._orientation = local * _quat_from_axes(frame.east, frame.north, frame.up)
        self.body = body
        self._ground = 0.0

    def _reset_camera(self):
        pose = self.start_pose or {}
        spec = self.terrain.spec
        self._place_camera(
            pose.get("lat", spec.reference_lat),
            pose.get("lon", spec.reference_lon),
            pose.get("alt", 800.0),
            pose.get("heading", 0.0),
            pose.get("pitch", 0.0),
        )
        # The sun stands where the original flat demo had it, at the start.
        frame = TangentFrame.at(spec.reference_lat, spec.reference_lon)
        d, _, _ = sun_lighting(210.0, 35.0)
        self._sun = d[0] * frame.east + d[1] * frame.north + d[2] * frame.up

    def _jump_over(self, body: Body, direction: np.ndarray):
        """3 radii above a body, along *direction* from its centre, looking down."""
        lat, lon = latlon(direction[None, :] / np.linalg.norm(direction))
        self._place_camera(
            float(lat[0]), float(lon[0]), ORBIT_RADII * body.radius, 0.0, -89.9, body
        )

    def _jump_to_orbit(self):
        self._jump_over(self.body, self._relative())

    def _jump_to_next_body(self):
        """To the next body, over its side facing the camera."""
        index = self.bodies.index(self.body)
        target = self.bodies[(index + 1) % len(self.bodies)]
        self._jump_over(target, self.cam_world - target.position)

    def _print_pose(self):
        relative = self._relative()
        up = relative / np.linalg.norm(relative)
        lat, lon = latlon(up[None, :])
        heading, pitch = self._true_heading_pitch()
        altitude = float(np.linalg.norm(relative)) - self.body.radius
        right = self._orientation.get_right()
        roll = -math.degrees(math.asin(np.clip(np.array(right) @ up, -1.0, 1.0)))
        print(
            f"camera over the {self.body.name}: {{lat: {lat[0]:.5f}, "
            f"lon: {lon[0]:.5f}, alt: {altitude:.0f}, heading: {heading:.1f}, "
            f"pitch: {pitch:.1f}}}  roll {roll:.1f}"
        )
        for body in self.bodies:
            print(f"{body.name} terrain: {body.terrain.stats}")

    def destroy_terrain(self):
        """Remove every body's terrain and stop the shared workers."""
        for terrain in self._terrains():
            terrain.destroy()
        self.executor.shutdown()

    def _quit(self):
        self.destroy_terrain()
        sys.exit()


def main():
    parser = argparse.ArgumentParser(description="Procedural terrain demo")
    parser.add_argument("--level", default=str(DEMO_LEVEL))
    parser.add_argument(
        "--radius", type=float, default=None, help="planet radius, metres"
    )
    parser.add_argument(
        "--moon-distance",
        type=float,
        default=MOON_DISTANCE_M,
        help="moon distance, centre to centre, metres",
    )
    parser.add_argument("--no-moon", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=None, help="generation processes"
    )
    parser.add_argument(
        "--backend", default="process", choices=("process", "thread", "inline")
    )
    args = parser.parse_args()
    window_config("Procedural terrain demo")
    moon = None if args.no_moon else args.moon_distance
    TerrainDemo(args.level, args.backend, args.radius, moon, args.workers).run()


if __name__ == "__main__":
    main()
