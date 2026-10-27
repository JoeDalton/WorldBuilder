"""
Standalone flythrough demo for the procedural planet terrain.

Flies a free camera around a small planetary system: a Kerbin-sized planet
(600 km) with hand-made stamps (world_builder/terrain/levels/demo.yaml), its
atmosphere, its clouds (cumulus over the whole planet, following its
weather) and its sea (and, with --ring, a small ring of dust and rocks in
its equator's plane, 1.5 to 2.1 radii out), an airless sand moon 21,000 km
away (as wide in the sky as the sun: it can eclipse it), and a procedural
star field, from orbit down to the ground of either. Run it from an
environment where world_builder is importable::

    python scripts/demo_terrain.py [--level FILE] [--radius METRES]
        [--moon-distance METRES] [--no-moon] [--workers N]
        [--backend process|thread|inline] [--quality low|mid|high|ultra]
        [--terrain-quality L] [--atmosphere-quality L] [--sky-quality L]
        [--clouds-quality L] [--ocean-quality L] [--asteroids-quality L]
        [--no-clouds] [--ring]

    --level          the planet's level file (default: the demo level)
    --radius         overrides the level's planet radius
    --moon-distance  centre to centre (default 21,000 km)
    --no-moon        the planet alone
    --workers        generation processes (the terrain's ``workers``
                     setting; default: physical cores - 1)
    --backend        where chunks and cloud tiles are generated (default
                     process)
    --quality        every part's quality preset at start (default high;
                     world_builder.settings.WorldSettings)
    --<part>-quality one part's preset, over --quality
    --no-clouds      no clouds at all
    --ring           a small ring around the planet (off by default, so
                     runs compare with earlier ones)

The default backend generates terrain in spawned worker processes, which
re-import this script: keep the ``if __name__ == "__main__"`` guard. All
bodies, and the clouds, share one pool.

The window has no multisampling and no vsync (:func:`window_config`): the
frame-rate meter shows what a frame costs, not the display's refresh rate.

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

    J / K           swing the sun around the sky (azimuth, here)
    I / U           raise / lower the sun (here)
    F1 - F5         cycle one part's quality preset, low / mid / high / ultra:
                    terrain, air, sky, clouds, ocean (each part's own
                    settings; a cloud or ocean change may refill a moment)
    F6              cycle every part's preset at once
    F7              cycle the shadows' preset (the clouds' shadow on the
                    ground: off at low, more samples above)
    M               the test balls: the world's model shader / Panda3D's
                    lighting
    F               print the camera pose and terrain stats
    ESC             quit

The HUD (top left, refreshed 5 times a second) shows the nearest body (its
preset, seed and radius) and the finest detail drawn; altitude, height above
the ground, latitude/longitude, cube face, heading, pitch and speed factor;
the biomes under the camera and the other bodies' distances; chunk counts
and generation, upload and LOD timings; the depth split and far plane; the
clouds' tiles and billboards; each part's quality preset ("custom" when it
differs from every preset).

Lighting: the sun is one direction for the whole system, a disc as wide as
the sky draws it; the atmosphere reddens it, the planets' shadows hide it,
and each body eclipses it for the other, with a penumbra (swing the sun
behind the moon with J / K and I / U). Panda3D's own lights (a
DirectionalLight and an AmbientLight on render, for anything drawn without
the world's shaders) follow the light at the camera, and the sea's near look
is lit as the sea under the camera (Sea.set_light), so it darkens at night.
A white ball (Panda3D's smiley, untextured, 20 m across) floats ahead of the
start pose, at its height, the textured smiley beside it, to judge how an
object is lit: each has its own Panda3D lights, set to the light where it
is, not at the camera (world_builder.lights.TrackedLights: it darkens in an
eclipse or past the terminator whatever the camera's light), and is drawn
by the world's model shader (world_builder.objects: lit and tonemapped as
the terrain, every world shadow per pixel); M switches them to Panda3D's
own lighting, to compare.

Terrain is generated in background processes, coarse first: the world shows
blurry for a second, then sharpens.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    ClockObject,
    DirectionalLight,
    GraphicsWindow,
    LMatrix3f,
    LQuaternionf,
    TextNode,
    Vec3,
    loadPrcFileData,
)

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.field import AsteroidField
from world_builder.asteroids.spec import PRESETS as FIELD_PRESETS
from world_builder.asteroids.spec import Annulus, Clumping, Density, FieldSpec
from world_builder.atmosphere import AtmosphereParams, Atmospheres, in_shadow
from world_builder.clouds.half_res import HalfResClouds
from world_builder.clouds.shadow_map import CloudShadowMaps
from world_builder.clouds.sphere import SphericalClouds
from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCameras
from world_builder.lights import TrackedLights
from world_builder.objects import ModelLighting
from world_builder.ocean import Sea
from world_builder.quality import QualityLevel
from world_builder.settings import SECTIONS, WorldSettings
from world_builder.shadows import ECLIPSE_SKY_FLOOR, Shadows
from world_builder.sky import Sky, Starfield
from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chart import (
    TangentFrame,
    latlon,
)
from world_builder.terrain.cubesphere import FACE_NAMES, dirs_to_face
from world_builder.terrain.level import DEMO_LEVEL, load_level
from world_builder.terrain.planet import PLANET_HEIGHT_BOUNDS, PRESETS
from world_builder.terrain.quadtree import UNKNOWN_BOUNDS
from world_builder.terrain.shadow_map import ShadowMap

INITIAL_FAR_M = 600000.0  # the lens the sea's reflection camera copies
SPEED = 300.0  # metres/second at low altitude
BOOST = 5.0
MIN_CLEARANCE = 3.0  # metres the camera stays above the ground
ORBIT_RADII = 3.0  # HOME and TAB jump this high
# The moon: centre-to-centre distance (as wide in the sky as the sun, seen
# from the planet's ground: a total eclipse is possible), and where it hangs
# in the start view (this far from the start point's vertical, toward its
# north).
MOON_DISTANCE_M = 21_000e3
MOON_SEED = 7
MOON_ZENITH_ANGLE = 75.0
# The camera stays within this many moon distances of the planet.
MAX_RANGE_FACTOR = 3.0
# The sea's near look (waves, reflection, foam) fades into its far one (flat
# water lit per pixel) over this band of altitudes, metres.
OCEAN_NEAR_BAND = (12000.0, 18000.0)
# The sea's wave pattern is anchored to the ground through a chart around
# this point, re-anchored (a jump in the pattern) past this distance.
OCEAN_ANCHOR_RANGE = 50000.0
# Sunlight above the air (the atmosphere reddens it), the planet's sky light,
# and an airless body's next to no ambient light (a black sky).
SUN_COLOR = (2.6, 2.5, 2.35)
SKY_AMBIENT = (0.34, 0.44, 0.64)
SPACE_AMBIENT = (0.025, 0.025, 0.03)
# The clouds' share of the sky light (SphericalClouds' sky_strength).
CLOUD_SKY_LIGHT = 0.35
# The sun's elevation the ocean's water colour was tuned in (the start pose).
WATER_TUNED_ELEVATION = 35.0
CLOUD_BASE = 2200.0  # metres above sea level: above the start pose
# The planet's ring: from and to this many planet radii from its centre, in
# its equator's plane (the field's xy plane is the world's: the poles are on
# z), this thick, metres. Seen from the start area, 10-26 degrees over the
# southern horizon.
RING_RADII = (1.5, 2.1)
RING_THICKNESS = 1000.0
HUD_PERIOD = 0.2  # seconds between HUD refreshes
# The white ball ahead of the start pose (judging the lights on objects).
TEST_BALL_DIAMETER = 20.0
TEST_BALL_DISTANCE = 150.0  # metres ahead of the start pose, level with it
TEST_BALL_SPACING = 30.0  # the textured ball, this far right of the white one

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
#: F1 - F5 and F7 cycle these parts' presets (WorldSettings sections), F6
#: all.
QUALITY_KEYS = (
    ("f1", "terrain"),
    ("f2", "atmosphere"),
    ("f3", "sky"),
    ("f4", "clouds"),
    ("f5", "ocean"),
    ("f7", "shadows"),
)
QUALITY_ALL_KEY = "f6"
_DEBUG_VIEWS = ("shaded", "LOD", "biomes", "slope", "normals", "skirts")


def window_config(
    title: str, size: str = "1600 900", offscreen: bool = False, vsync: bool = True
) -> None:
    """Load the window's PRC settings (before ShowBase opens it).

    No multisampling: the driver's choice of sample count is not ours to
    make (Panda picks the config with the most samples, x16 on a GTX 1070,
    whatever count is asked for), and it was most of the clouds' cost.

    No transform cache: the world moves around the camera, so every drawn
    node's net transform is new every frame and Panda's cache of composed
    transforms only filled up for its garbage collection to empty (1 ms a
    frame in low flight). A game drawing WorldBuilder's world wants the same.

    :param title: the window's title
    :param size: "width height"
    :param offscreen: an offscreen buffer (no display refresh cap or
        throttling: timings are the GPU's)
    :param vsync: False turns the display's sync off (the demo and the
        scripts all do)
    """
    lines = [
        f"window-title {title}",
        f"win-size {size}",
        "audio-library-name null",
        "framebuffer-multisample 0",
        "multisamples 0",
        "depth-bits 24",
        "transform-cache false",
    ]
    if not vsync:
        lines.append("sync-video false")
    if offscreen:
        lines.append("window-type offscreen")
    loadPrcFileData("", "\n".join(lines))


def describe_window(demo: ShowBase) -> str:
    """:returns: the window's kind and size, and its samples per pixel if
    more than one"""
    win = demo.win
    kind = "window" if isinstance(win, GraphicsWindow) else "offscreen"
    samples = win.get_fb_properties().get_multisamples()
    msaa = f", MSAA x{samples}" if samples > 1 else ""
    return f"{kind} {win.get_x_size()}x{win.get_y_size()}{msaa}"


def sun_lighting(azimuth: float, elevation: float) -> tuple[np.ndarray, tuple, tuple]:
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


def ring_spec(radius: float) -> FieldSpec:
    """The planet's small ring: the ring preset's ringlets, spins and icy
    rocks, scaled down around a planet of *radius*. Mostly dust (an optical
    depth of ~0.2 across it), so it reads as a band from the ground, with
    sparse rocks (20-400 m) to fly through."""
    inner, outer = (radius * r for r in RING_RADII)
    return replace(
        FIELD_PRESETS["ring"],
        shape=Annulus(
            inner,
            outer,
            RING_THICKNESS,
            edge=250.0,
            radial_edge=0.02 * radius,
            radial_profile=FIELD_PRESETS["ring"].shape.radial_profile,
        ),
        density=Density(
            1.0, Clumping(scale=8000.0, coverage=0.7, softness=0.15), dust=0.3
        ),
        sizes=Distribution.power_law(3.0, 20.0, 400.0),
        seed=5,
    )


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
    :ivar shadow_map: its mountains' shadow
    :ivar models: the model shader's inputs near it
    """

    name: str
    terrain: object
    atmosphere: bool
    shadow_map: ShadowMap | None = None
    models: ModelLighting | None = None

    @property
    def position(self) -> np.ndarray:
        return self.terrain.position

    @property
    def radius(self) -> float:
        return self.terrain.radius


def _level_name(world: WorldSettings, section: str) -> str:
    """:returns: the preset a section is, or "custom" """
    level = world.section_level(section)
    return "custom" if level is None else level.value


def _next_level(level: QualityLevel | None) -> QualityLevel:
    """:returns: the next preset, cheapest after dearest (HIGH after custom)"""
    if level is None:
        return QualityLevel.HIGH
    levels = list(QualityLevel)
    return levels[(levels.index(level) + 1) % len(levels)]


class TerrainDemo(ShowBase):
    """A free-flying camera around and onto a planet and its moon."""

    def __init__(
        self,
        level_path: str,
        backend: str,
        radius: float | None,
        moon_distance: float | None = MOON_DISTANCE_M,
        settings: WorldSettings | None = None,
        clouds: bool = True,
        cloud_debug: int = 0,
        ocean_debug: int = 0,
        ring: bool = False,
    ) -> None:
        """
        :param level_path: the planet's level file
        :param backend: chunk generation backend, see ChunkExecutor
        :param radius: overrides the level's planet radius, metres
        :param moon_distance: centre-to-centre, metres; None = no moon
        :param settings: what every part costs (:class:`WorldSettings`;
            F1 - F6 change it); None = HIGH
        :param clouds: draw the planet's clouds
        :param cloud_debug: the clouds' profiling switches
            (:meth:`SphericalClouds.set_debug`)
        :param ocean_debug: the sea's debug view (:meth:`Sea.set_debug_mode`)
        :param ring: draw the planet's ring (:func:`ring_spec`)
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

        spec, self.start_pose = load_level(level_path, radius=radius)

        from world_builder.terrain.terrain import Terrain
        from world_builder.terrain.workers import ChunkExecutor

        world = settings or WorldSettings()
        self.world = world
        self._clouds_on = clouds
        print("starting terrain workers ...")
        self.executor = ChunkExecutor(
            backend=backend, workers=world.terrain.workers or None
        )
        # The sky's glow is made in the pool while the terrains are built.
        stars = Starfield()
        stars.prefetch_glow(self.executor, world.sky.glow_size)
        # The planet's air: everything is seen through it, the moon included.
        self.atmospheres = Atmospheres(world.atmosphere)
        self.atmospheres.add(spec.radius, AtmosphereParams())
        # Each body eclipses the sun for the other (body 0 owns the air).
        self.shadows = Shadows(settings=world.shadows)
        self.shadows.add(spec.radius, reach=spec.radius + AtmosphereParams().height)
        self.shadows.set_atmosphere(0, 0)
        self.terrain = Terrain(
            parent=self.render,
            spec=spec,
            executor=self.executor,
            atmospheres=self.atmospheres,
            atmosphere=0,
            sun_color=SUN_COLOR,
            ambient=SKY_AMBIENT,
            settings=world.terrain,
        )
        self.shadows.bind(self.terrain.root, 0)
        self.home = Body("planet", self.terrain, atmosphere=True)
        self.bodies = [self.home]
        if moon_distance is not None:
            moon_position = self._moon_position(spec, moon_distance)
            moon = Terrain(
                parent=self.render,
                spec=replace(PRESETS["moon"], seed=MOON_SEED),
                executor=self.executor,
                atmospheres=self.atmospheres,  # seen through the planet's
                sun_color=SUN_COLOR,
                ambient=SPACE_AMBIENT,
                position=moon_position,
                materials=self.terrain.materials,
                settings=world.terrain,
            )
            reach = moon.radius + PLANET_HEIGHT_BOUNDS[1]
            index = self.shadows.add(moon.radius, moon_position, reach=reach)
            self.shadows.bind(moon.root, index)
            self.bodies.append(Body("moon", moon, atmosphere=False))
        self.max_range = MAX_RANGE_FACTOR * (moon_distance or 10.0 * spec.radius)
        self.body = self.home  # the nearest body, updated every frame

        self._sun = np.array([0.0, 0.0, 1.0])  # world frame, toward the sun
        # Each body's mountains' shadow, baked in the pool for each sun.
        for body in self.bodies:
            body.shadow_map = ShadowMap(
                body.terrain,
                self.executor,
                world.shadows.terrain_map_size,
                self._sun,
                world.shadows.local_map_size,
                world.shadows.local_map_spacing,
            )
            body.shadow_map.bind(body.terrain.root)
        # The models' shader's inputs near each (the clouds' are bound with
        # them).
        for index, body in enumerate(self.bodies):
            body.models = ModelLighting(
                atmosphere=0 if body.atmosphere else None, terrain=body.terrain
            )
            self.atmospheres.bind(body.models.inputs)
            self.shadows.bind(body.models.inputs, index)
            body.shadow_map.bind(body.models.inputs)
        self.ring = None
        if ring:
            self.ring = AsteroidField(
                self,
                ring_spec(spec.radius),
                clock=self._clock.get_frame_time,
                settings=world.asteroids,
                executor=self.executor,
                sun_color=SUN_COLOR,
                ambient=SPACE_AMBIENT,
                materials=self.terrain.materials,
                name="ring",
            )
        self.sky = Sky(self.render, stars, self.atmospheres, settings=world.sky)
        self.shadows.bind(self.sky.node)
        self.sky.node.hide(HIDE_FROM_NEAR)  # the far camera draws it, first
        self.clouds = None
        self.cloud_maps = None  # the clouds' shadow maps
        self._cloud_debug = cloud_debug
        self._rebuild_clouds()
        # Built before the depth split changes the lens: the sea's reflection
        # camera copies it once.
        self.ocean = Sea(
            self,
            self.terrain,
            self._clock.get_frame_time,
            settings=self.world.ocean,
            # The game's 20 m swell, in deep water: it dies out over the
            # shallows, so it does not drown the beaches.
            swell_amplitude=20.0,
            near_band=OCEAN_NEAR_BAND,
            anchor_range=OCEAN_ANCHOR_RANGE,
        )
        self.ocean.set_debug_mode(ocean_debug)
        self.depth = SplitDepthCameras(self, near=1.0)
        self.half_res = None  # the clouds' half-resolution pass, if asked for
        self._sync_half_res()
        self._debug = 0
        # Panda3D lights for anything not drawn by the world's own shaders
        # (models, ships): the sun and the sky as they are at the camera.
        self.sun_light = self.render.attach_new_node(DirectionalLight("sun"))
        self.ambient_light = self.render.attach_new_node(AmbientLight("ambient"))
        self.render.set_light(self.sun_light)
        self.render.set_light(self.ambient_light)
        self._water_reference = None
        self._ground = 0.0

        self._keys = dict.fromkeys(_CONTROLS, False)
        for key in _CONTROLS:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])
        presses = [
            ("y", self._level_roll, []),
            ("page_up", self._scale_speed, [2.0]),
            ("page_down", self._scale_speed, [0.5]),
            ("home", self._jump_to_orbit, []),
            ("tab", self._jump_to_next_body, []),
            ("end", self._reset_camera, []),
            ("r", self._reset_camera, []),
            ("f", self._print_pose, []),
            ("m", self._toggle_model_shader, []),
            ("escape", self._quit, []),
            (QUALITY_ALL_KEY, self._cycle_all_quality, []),
        ]
        presses += [(key, self._cycle_quality, [name]) for key, name in QUALITY_KEYS]
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
        self._test_balls = self._make_test_balls()
        # Lit where they are, by their own lights (_update_lights), drawn by
        # the model shader (_shade_models).
        self.tracked = TrackedLights(self.render)
        self._model_shader = True
        self._track_test_balls()
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

    def _nearest_body(self, point: np.ndarray | None = None) -> Body:
        """:returns: the body whose sea level is nearest a point (the camera
        by default)"""
        point = self.cam_world if point is None else point
        return min(
            self.bodies,
            key=lambda b: float(np.linalg.norm(point - b.position)) - b.radius,
        )

    def _true_frame(
        self, body: Body | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _terrains(self) -> list:
        return [body.terrain for body in self.bodies]

    # ── scene pieces ───────────────────────────────────────────────────────────

    def _make_test_balls(self) -> list:
        """A white ball and the textured smiley, hung ahead of the camera
        (the start pose: :meth:`_test_ball_points`), each lit where it is:
        for judging how objects are lit.

        :returns: [(node, its position in the planet frame)]
        """
        nodes = []
        for textured in (False, True):
            ball = self.loader.load_model("models/smiley")
            if not textured:
                ball.set_texture_off(1)
                ball.set_color(1.0, 1.0, 1.0, 1.0)
            ball.set_scale(TEST_BALL_DIAMETER / 2.0)  # the model is a unit sphere
            ball.reparent_to(self.render)
            nodes.append(ball)
        return list(zip(nodes, self._test_ball_points()))

    def _test_ball_points(self) -> list[np.ndarray]:
        """Where the balls hang for the camera as it is (planet frame): the
        white one TEST_BALL_DISTANCE ahead, level with it, the textured one
        TEST_BALL_SPACING to its right."""
        start = self._relative(self.home)
        up = start / np.linalg.norm(start)
        ahead = self._forward() - up * float(self._forward() @ up)
        ahead /= max(np.linalg.norm(ahead), 1e-9)
        white = start + ahead * TEST_BALL_DISTANCE
        return [white, white + np.cross(ahead, up) * TEST_BALL_SPACING]

    def _place_test_balls(self) -> None:
        """The balls where they hang, this frame (before their light is set:
        the camera is render's origin, and the planet does not turn)."""
        for ball, point in self._test_balls:
            ball.set_pos(*(self.home.position + point - self.cam_world))

    def _track_test_balls(self) -> None:
        """Light the balls where they are, by the model shader or not."""
        for ball, _ in self._test_balls:
            self.tracked.track(ball, shader=self._model_shader)

    def _toggle_model_shader(self) -> None:
        """M: the test balls by the world's model shader, or by Panda3D's
        lighting (as any model without it)."""
        self._model_shader = not self._model_shader
        self._track_test_balls()
        print("test balls:", "model shader" if self._model_shader else "Panda3D lights")

    # ── per frame ──────────────────────────────────────────────────────────────

    def _update(self, task) -> int:
        dt = min(self._clock.get_dt(), 0.1)
        self.body = self._nearest_body()
        self._fly(dt)
        self.camera.set_pos(0.0, 0.0, 0.0)
        self.camera.set_quat(self.render, self._orientation)

        forward = self._forward()
        for terrain in self._terrains():
            terrain.update(self.cam_world, camera_forward=forward)
        if self.ring is not None:
            self.ring.update(self.cam_world)
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
        if self.half_res is not None:
            self.half_res.update()

        self.atmospheres.update(self.cam_world)
        self.shadows.update(self.cam_world)
        for body in self.bodies:
            body.shadow_map.update(self.cam_world)  # the local map follows it
        self.sky.update((0.0, 0.0, 0.0), far)
        if self.clouds is not None:
            fov = math.radians(self.camLens.get_fov()[1])
            scale = self.half_res.pixel_scale if self.half_res is not None else 1.0
            self.clouds.set_pixel_angle(scale * fov / max(self.win.get_y_size(), 1))
            self.clouds.update(self.cam_world)
            if self.cloud_maps is not None:
                steps = self.shadows.settings.cloud_shadow_steps
                self.cloud_maps.update(self.cam_world, self._sun, steps)
        self._place_test_balls()
        self._update_sun()
        self._shade_models()
        self.ocean.update(self.cam_world)
        # Text geometry is rebuilt on every change: a few times a second is
        # plenty to read, and saves ~1 ms a frame.
        now = self._clock.get_frame_time()
        if now - self._hud_time >= HUD_PERIOD:
            self._hud_time = now
            self._update_hud()
        return task.cont

    def _fly(self, dt: float) -> None:
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

    def _update_sun(self) -> None:
        # Sunlight above the air: the atmosphere reddens it where it must.
        self.shadows.set_sun(self._sun)
        self.sky.set_sun(self._sun)
        if self.clouds is not None:
            self.clouds.set_sun(self._sun)
            self.clouds.set_sky_light(CLOUD_SKY_LIGHT)
        for body in self.bodies:
            ambient = SKY_AMBIENT if body.atmosphere else SPACE_AMBIENT
            body.terrain.set_sun(tuple(self._sun), SUN_COLOR, ambient)
            body.shadow_map.set_sun(self._sun)  # baked again if it moved
            body.models.set_sun(self._sun, SUN_COLOR)
        if self.ring is not None:
            self.ring.set_sun(self._sun, SUN_COLOR, SPACE_AMBIENT)
        self._update_lights()

    def _shade_models(self) -> None:
        """The tracked nodes drawn by the model shader: each given the inputs
        of the body nearest it, this frame's (the camera is render's
        origin)."""
        for body in self.bodies:
            body.models.update((0.0, 0.0, 0.0))
        points = self.cam_world + self.tracked.positions()
        for node, shaded, point in zip(self.tracked.nodes, self.tracked.shaded, points):
            if shaded:
                self._nearest_body(point).models.apply(node)

    def _light_at(
        self, body: Body, point_world, shade: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """:returns: (direct, ambient) light at a point near a body, eclipsed
        by the other, its sunlight times *shade* (the planet's clouds' shadow
        there: :meth:`_local_shades`)"""
        if body.atmosphere:
            direct, ambient = self.atmospheres.light_at(
                0,
                point_world,
                self._sun,
                SUN_COLOR,
                np.asarray(SKY_AMBIENT),
            )
        else:
            hidden = in_shadow(point_world - body.position, self._sun, body.radius)
            direct = np.zeros(3) if hidden else np.asarray(SUN_COLOR)
            ambient = np.asarray(SPACE_AMBIENT)
        own = next(i for i, other in enumerate(self.bodies) if other is body)
        eclipse = float(self.shadows.eclipse(point_world, self._sun, own))
        return direct * (eclipse * shade), ambient * (
            ECLIPSE_SKY_FLOOR + (1.0 - ECLIPSE_SKY_FLOOR) * eclipse
        )

    def _local_shades(self, points: np.ndarray) -> np.ndarray:
        """:returns: (n,) the local shadows at points: the planet's clouds'
        near it (one call for them all: each costs ~0.2 ms), and the nearest
        body's mountains'"""
        shades = np.ones(len(points))
        steps = self.shadows.settings.cloud_shadow_steps
        nearest = [self._nearest_body(point) for point in points]
        near = np.array([body is self.home for body in nearest])
        if self.clouds is not None and near.any():
            shades[near] = self.clouds.sun_visibility(points[near], self._sun, steps)
        radius = self.shadows.sun_angular_radius
        for i, body in enumerate(nearest):
            shades[i] *= float(body.shadow_map.visibility(points[i], radius))
        return shades

    def _update_lights(self) -> None:
        """Panda3D's sun and ambient lights as they are at the camera, the
        tracked nodes' (the ball's) as they are at each, and the sea's near
        look lit as the sea under the camera."""
        points = np.concatenate(
            [self.cam_world[None, :], self.cam_world + self.tracked.positions()]
        )
        shades = self._local_shades(points)
        direct, ambient = self._light_at(self.body, self.cam_world, shades[0])
        self.sun_light.node().set_color((*direct, 1.0))
        self.sun_light.node().set_direction(Vec3(*(-self._sun)))
        self.ambient_light.node().set_color((*ambient, 1.0))
        lights = [
            self._light_at(self._nearest_body(point), point, shade)
            for point, shade in zip(points[1:], shades[1:])
        ]
        if lights:
            self.tracked.apply(*zip(*lights), self._sun)
        # The water's colour was tuned in daylight: scale it by the light on
        # the sea now (sun on a level surface plus sky) over that day's.
        up = self._relative(self.home) / np.linalg.norm(self._relative(self.home))
        sea = self.home.position + up * self.radius
        # The clouds' shadow is the shader's, per pixel, on the sun's share.
        direct, ambient = self._light_at(self.home, sea)
        sun_light = direct * max(float(self._sun @ up), 0.0)
        light = sun_light + ambient
        if self._water_reference is None:
            elevation = math.radians(WATER_TUNED_ELEVATION)
            east = np.cross([0.0, 0.0, 1.0], up)
            east /= max(np.linalg.norm(east), 1e-9)
            tuned = math.sin(elevation) * up + math.cos(elevation) * east
            t = self.atmospheres.transmittance_to_sun(0, sea, tuned)
            self._water_reference = np.asarray(SUN_COLOR) * t * math.sin(
                elevation
            ) + np.asarray(SKY_AMBIENT)
        total = np.clip(light / self._water_reference, 0.0, 1.5)
        sun_share = np.divide(sun_light, light, out=np.zeros(3), where=light > 0.0)
        self.ocean.set_light(total * (1.0 - sun_share), sun=total * sun_share)

    def set_local_sun(self, azimuth: float, elevation: float) -> None:
        """Put the sun at an azimuth/elevation (degrees) as seen from the
        nadir on the nearest body."""
        east, north, up = self._true_frame()
        d, _, _ = sun_lighting(azimuth, elevation)
        self._sun = d[0] * east + d[1] * north + d[2] * up

    def _turn_sun(self, axis: int, degrees: float) -> None:
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

    def _update_hud(self) -> None:
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
            f"clouds: {cs['tiles']} tiles, "
            f"{cs['billboards']:,} billboards, {cs['pending']} generating "
            f"({cs['gen_ms']:.0f} ms/tile)"
            if cs is not None
            else "clouds off"
        )
        quality = "   ".join(
            f"{SECTIONS[name].LABEL.lower()} {_level_name(self.world, name)}"
            for _, name in QUALITY_KEYS
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
            clouds,
            f"quality: {quality}   test balls: "
            + ("model shader" if self._model_shader else "Panda3D lights"),
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

    def _set_key(self, key: str, pressed: bool) -> None:
        self._keys[key] = pressed

    def _rebuild_clouds(self) -> None:
        """New clouds for the planet (its weather changes with it). They take
        over the old ones' worker channel, which drops their tiles."""
        if self.clouds is not None:
            self.clouds.remove()
            if self.cloud_maps is not None:
                self.cloud_maps.destroy()
                self.cloud_maps = None
            self.clouds = None
        if not self._clouds_on:
            return
        self.clouds = SphericalClouds(
            self.render,
            self.radius,
            self.executor,
            atmospheres=self.atmospheres,
            atmosphere=0,
            altitude=CLOUD_BASE,
            planet=self.terrain.spec,
            settings=self.world.clouds,
            channel="clouds",
        )
        self.shadows.bind(self.clouds.root, 0)
        self.clouds.bind_shadow_inputs(self.terrain.root)  # their shadow on it
        self.clouds.bind_shadow_inputs(self.home.models.inputs)  # on the models
        if self.home.shadow_map is not None:  # the peaks' on them
            self.home.shadow_map.bind(self.clouds.root)
        # Their shadow marched once per texel of two maps (Shadows, part E).
        self.cloud_maps = CloudShadowMaps(
            self.win, self.clouds, self.world.shadows.cloud_map_size
        )
        self.cloud_maps.bind(self.terrain.root)
        self.cloud_maps.bind(self.home.models.inputs)
        self.clouds.set_debug(self._cloud_debug)
        self.clouds.set_sun(self._sun)
        if getattr(self, "half_res", None) is not None:
            self.half_res.add(self.clouds.root)

    def _sync_half_res(self) -> None:
        """Build or remove the clouds' half-resolution pass as the settings
        ask (CloudSettings.half_resolution)."""
        wanted = self.world.clouds.half_resolution
        if wanted and self.half_res is None:
            self.half_res = HalfResClouds(self, self.depth)
            self.half_res.exclude(self.sky.node)
            if self.clouds is not None:
                self.half_res.add(self.clouds.root)
        elif not wanted and self.half_res is not None:
            self.half_res.destroy()
            self.half_res = None

    def _cycle_quality(self, section: str) -> None:
        """One part (a WorldSettings section) to its next preset."""
        level = _next_level(self.world.section_level(section))
        self._set_world(self.world.with_section_level(section, level))
        print(f"{SECTIONS[section].LABEL} quality: {level.value}")

    def _cycle_all_quality(self) -> None:
        """Every part to the next preset."""
        level = _next_level(self.world.level)
        self._set_world(WorldSettings.preset(level))
        print(f"quality: {level.value}")

    def _set_world(self, world: WorldSettings) -> None:
        """Hand every part its section. What only a restart changes (the
        terrain's texture size and workers) is logged and left as it is."""
        self.world = world
        world.apply(
            *self._terrains(), self.atmospheres, self.shadows, self.sky, self.ocean
        )
        if self.cloud_maps is not None:
            self.cloud_maps.set_size(world.shadows.cloud_map_size)
        for body in self.bodies:
            body.shadow_map.set_size(world.shadows.terrain_map_size)
            body.shadow_map.set_local(
                world.shadows.local_map_size, world.shadows.local_map_spacing
            )
        if self.clouds is not None:
            world.apply(self.clouds)
        if self.ring is not None:
            world.apply(self.ring)
        self._sync_half_res()

    def set_debug_view(self, view: int) -> None:
        """0 shaded, 1 LOD, 2 biomes, 3 slope, 4 normals, 5 skirts (all bodies)."""
        self._debug = view
        for terrain in self._terrains():
            terrain.set_debug_mode(view)

    def _detail_text(self) -> str:
        terrain = self.body.terrain
        level = terrain.min_level
        text = f"{terrain.grid.spacing(level):.1f} m"
        return f"full ({text})" if level == 0 else f"-{level} levels ({text})"

    def _rotate(self, kind: str, degrees: float) -> None:
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

    def _level_roll(self) -> None:
        """Keep the view direction, bring the camera's up into the plane of
        it and the local vertical (no roll)."""
        forward = self._forward()
        right = np.cross(forward, self._up())
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            return  # looking straight up or down: any roll is level
        right /= norm
        self._orientation = _quat_from_axes(right, forward, np.cross(right, forward))

    def _scale_speed(self, factor: float) -> None:
        self.speed = max(10.0, min(SPEED * 64, self.speed * factor))

    def _place_camera(
        self,
        lat: float,
        lon: float,
        altitude: float,
        heading: float,
        pitch: float,
        body: Body | None = None,
    ) -> None:
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

    def _reset_camera(self) -> None:
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

    def _jump_over(self, body: Body, direction: np.ndarray) -> None:
        """3 radii above a body, along *direction* from its centre, looking down."""
        lat, lon = latlon(direction[None, :] / np.linalg.norm(direction))
        self._place_camera(
            float(lat[0]), float(lon[0]), ORBIT_RADII * body.radius, 0.0, -89.9, body
        )

    def _jump_to_orbit(self) -> None:
        self._jump_over(self.body, self._relative())

    def _jump_to_next_body(self) -> None:
        """To the next body, over its side facing the camera."""
        index = self.bodies.index(self.body)
        target = self.bodies[(index + 1) % len(self.bodies)]
        self._jump_over(target, self.cam_world - target.position)

    def _print_pose(self) -> None:
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

    def destroy_terrain(self) -> None:
        """Remove every body's terrain and the ring, and stop the shared
        workers."""
        if self.ring is not None:
            self.ring.destroy()
        self.tracked.clear()
        if self.cloud_maps is not None:
            self.cloud_maps.destroy()
        for terrain in self._terrains():
            terrain.destroy()
        self.executor.shutdown()

    def _quit(self) -> None:
        self.destroy_terrain()
        sys.exit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Procedural terrain demo")
    parser.add_argument("--level", default=str(DEMO_LEVEL), help="level file")
    parser.add_argument(
        "--radius", type=float, default=None, help="planet radius, metres"
    )
    parser.add_argument(
        "--moon-distance",
        type=float,
        default=MOON_DISTANCE_M,
        help="moon distance, centre to centre, metres",
    )
    parser.add_argument("--no-moon", action="store_true", help="the planet alone")
    parser.add_argument(
        "--backend",
        default="process",
        choices=("process", "thread", "inline"),
        help="where chunks and cloud tiles are generated",
    )
    add_world_arguments(parser)
    args = parser.parse_args()
    # No vsync: the frame rate shows what a frame costs, not the display.
    window_config("Procedural terrain demo", vsync=False)
    moon = None if args.no_moon else args.moon_distance
    TerrainDemo(
        args.level,
        args.backend,
        args.radius,
        moon,
        settings=settings_from_args(args),
        clouds=not args.no_clouds,
        ring=args.ring,
    ).run()


def add_world_arguments(parser: argparse.ArgumentParser) -> None:
    """The quality arguments every script built on the demo takes:
    ``--quality``, a ``--<part>-quality`` per WorldSettings section,
    ``--workers`` (the terrain's setting), ``--no-clouds`` and
    ``--ring``."""
    levels = [level.value for level in QualityLevel]
    parser.add_argument(
        "--quality", default="high", choices=levels, help="every part's preset"
    )
    for section in SECTIONS:
        parser.add_argument(
            f"--{section}-quality",
            default=None,
            choices=levels,
            help=f"the {section}'s preset, over --quality",
        )
    parser.add_argument(
        "--workers", type=int, default=None, help="generation processes"
    )
    parser.add_argument("--no-clouds", action="store_true", help="no clouds")
    parser.add_argument(
        "--ring", action="store_true", help="a small ring around the planet"
    )
    parser.add_argument(
        "--half-res-clouds",
        action="store_true",
        help="clouds at half resolution (CloudSettings.half_resolution)",
    )


def settings_from_args(args: argparse.Namespace) -> WorldSettings:
    """:returns: the WorldSettings :func:`add_world_arguments`' arguments ask
    for"""
    settings = WorldSettings.preset(args.quality)
    for section in SECTIONS:
        level = getattr(args, f"{section}_quality")
        if level is not None:
            settings = settings.with_section_level(section, level)
    if args.workers is not None:
        settings = settings.with_value("terrain", "workers", args.workers)
    if args.half_res_clouds:
        settings = settings.with_value("clouds", "half_resolution", True)
    return settings


if __name__ == "__main__":
    main()
