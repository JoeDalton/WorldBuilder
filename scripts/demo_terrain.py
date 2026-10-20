"""
Standalone flythrough demo for the procedural planet terrain.

Flies a free camera from orbit down to the ground of a procedurally generated
planet with hand-made stamps (world_builder/terrain/levels/demo.yaml), the
WorldBuilder ocean, and haze. Run it from an environment where world_builder
is importable::

    python scripts/demo_terrain.py [--level FILE] [--radius METRES]
        [--backend process|thread|inline]

The default backend generates terrain in spawned worker processes, which
re-import this script: keep the ``if __name__ == "__main__"`` guard.

Frames: the camera position is float64 in the planet frame (planet centre at
the origin). The camera stays at the render origin and the world is moved
around it (see Terrain.to_render), so float32 never sees a large number near
the camera. The camera's orientation is absolute in the planet frame: flying
straight ahead from level flight climbs away from the curving surface,
tangentially. Yaw turns about the local vertical, so the horizon stays level
while turning; Y re-levels it after a long flight.

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
    HOME            jump to orbit: 3 planet radii up, looking down
    END / R         back to the level's start pose

    F1 - F4         planet preset: earthlike / desert / forest / ice
    N               next seed
    L               reload the level file (edit stamps live)
    O               toggle the ocean
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
from types import SimpleNamespace

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ClockObject,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
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

from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCameras
from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chart import (
    AzimuthalEquidistantChart,
    TangentFrame,
    latlon,
)
from world_builder.terrain.cubesphere import FACE_NAMES, dirs_to_face
from world_builder.terrain.level import DEMO_LEVEL, load_level
from world_builder.terrain.quadtree import UNKNOWN_BOUNDS

INITIAL_FAR_M = 600000.0  # the lens the ocean's reflection camera copies
HAZE_COLOR = (0.62, 0.71, 0.84)  # display space; sky at the horizon
ZENITH_COLOR = (0.28, 0.45, 0.76)
HAZE_DISTANCE = 150000.0
SPEED = 150.0  # metres/second at low altitude
BOOST = 5.0
MIN_CLEARANCE = 3.0  # metres the camera stays above the ground
MAX_ALTITUDE_RADII = 5.0  # the camera stays within this many radii above sea
ORBIT_RADII = 3.0  # HOME jumps this high
MAX_DROPPED_LEVELS = 5  # P cycles the finest drawn level from 0 to this
# The swell ocean is built for eyes up to 20 km; above this band the terrain
# draws the sea itself, as flat water (hysteresis, so it does not flicker).
OCEAN_HANDOVER = (14000.0, 16000.0)
# The ocean's wave pattern is anchored to the ground through a chart around
# this point, re-anchored (a jump in the pattern) past this distance.
OCEAN_ANCHOR_RANGE = 50000.0
SKY_FADE = (20000.0, 80000.0)  # altitudes over which the sky dome fades to black
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
_DEBUG_VIEWS = ("shaded", "LOD", "biomes", "slope", "normals", "skirts")


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


class TerrainDemo(ShowBase):
    """A free-flying camera around and onto the procedural planet."""

    def __init__(self, level_path, backend: str, radius: float | None):
        super().__init__()
        self.setFrameRateMeter(True)
        self.disable_mouse()
        self._clock = ClockObject.get_global_clock()
        self.camLens.set_near_far(1.0, INITIAL_FAR_M)
        self.camLens.set_fov(70)
        self.set_background_color(0.0, 0.0, 0.0, 1.0)

        self.cam_planet = np.array([0.0, 0.0, 1.0])
        # Absolute, in the planet frame (= the render frame's axes): moving
        # does not turn the camera.
        self._orientation = LQuaternionf()
        self.speed = SPEED

        self.level_path = level_path
        self.radius_override = radius
        self.preset = None
        spec, self.start_pose = load_level(level_path, radius=radius)
        self.seed = spec.seed

        from world_builder.terrain.terrain import Terrain

        print("starting terrain workers ...")
        self.terrain = Terrain(
            parent=self.render,
            spec=spec,
            backend=backend,
            haze_color=HAZE_COLOR,
            haze_distance=HAZE_DISTANCE,
        )
        self._sun_planet = np.array([0.0, 0.0, 1.0])
        self._sky = self._make_sky()
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
        self._laser_marks: list[tuple[object, np.ndarray]] = []
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
        self._reset_camera()
        self.taskMgr.add(self._update, "terrain_demo_update")
        print(__doc__)

    # ── frames ─────────────────────────────────────────────────────────────────

    @property
    def radius(self) -> float:
        return self.terrain.radius

    def _true_frame(self):
        """:returns: (east, north, up) of the geographic frame at the nadir"""
        lat, lon = latlon(self.cam_planet[None, :] / np.linalg.norm(self.cam_planet))
        frame = TangentFrame.at(float(lat[0]), float(lon[0]))
        return frame.east, frame.north, frame.up

    def _up(self) -> np.ndarray:
        """:returns: the local vertical at the camera"""
        return self.cam_planet / np.linalg.norm(self.cam_planet)

    def _forward(self) -> np.ndarray:
        f = self._orientation.get_forward()
        return np.array([f.x, f.y, f.z])

    # ── scene pieces ───────────────────────────────────────────────────────────

    def _make_ocean(self):
        """The game's real curved ocean, in a frame that follows the nadir.

        The ocean is a flat-world object (sea level at z = 0, droop d^2 / 2R
        from the camera). Its frame is the tangent plane at the sea-level
        point under the camera, so it is exact where it matters; its xy
        follow the ground (a chart around an anchor), so the waves stay put.
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
        return Ocean(
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

    def _make_sky(self):
        """A unit gradient dome: haze at and below the horizon, blue overhead.

        Recentred on the camera, aligned with the local vertical and scaled
        inside the far camera's range every frame; drawn by the far camera
        only, first, without depth writes.
        """
        rings, segments = 24, 48
        vdata = GeomVertexData("sky", GeomVertexFormat.get_v3c4(), Geom.UH_static)
        vertex = GeomVertexWriter(vdata, "vertex")
        color = GeomVertexWriter(vdata, "color")
        for ring in range(rings + 1):
            el = -0.25 * math.pi + 0.75 * math.pi * ring / rings
            t = max(0.0, math.sin(el)) ** 0.6
            c = [h + (z - h) * t for h, z in zip(HAZE_COLOR, ZENITH_COLOR)]
            for seg in range(segments + 1):
                az = 2.0 * math.pi * seg / segments
                vertex.add_data3(
                    math.cos(el) * math.cos(az),
                    math.cos(el) * math.sin(az),
                    math.sin(el),
                )
                color.add_data4(*c, 1.0)
        tris = GeomTriangles(Geom.UH_static)
        for ring in range(rings):
            for seg in range(segments):
                a = ring * (segments + 1) + seg
                b = a + segments + 1
                tris.add_vertices(a, a + 1, b)  # seen from inside
                tris.add_vertices(a + 1, b + 1, b)
        geom = Geom(vdata)
        geom.add_primitive(tris)
        node = GeomNode("sky")
        node.add_geom(geom)
        sky = self.render.attach_new_node(node)
        sky.set_bin("background", 0)
        sky.set_depth_write(False)
        sky.set_light_off()
        sky.set_two_sided(True)
        sky.hide(HIDE_FROM_NEAR)
        return sky

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
        self._fly(dt)
        self.camera.set_pos(0.0, 0.0, 0.0)
        self.camera.set_quat(self.render, self._orientation)
        east, north, up = self._true_frame()

        self.terrain.update(self.cam_planet)
        rho = float(np.linalg.norm(self.cam_planet))
        altitude = rho - self.radius
        floor = 0.0 if self.terrain.spec.has_ocean else UNKNOWN_BOUNDS[0]
        split, far = SplitDepthCameras.planet_ranges(
            self.radius, rho, max_height=UNKNOWN_BOUNDS[1], min_height=floor
        )
        self.depth.update(split, far)
        self._update_sky(east, north, up, altitude, far)
        self._update_sun(up)
        self._update_ocean(altitude, up)
        if not self._probe.is_hidden():
            self._update_probe()
        for mark, point in self._laser_marks:
            mark.set_pos(*self.terrain.to_render(point))
        self._update_hud(altitude)
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

        ground = float(self.terrain.query.height_at(self.cam_planet, generate=False))
        rho = float(np.linalg.norm(self.cam_planet))
        above = max(rho - self.radius - ground, 0.0)
        # Speed grows with height above ground: a descent from orbit is
        # exponential, a few seconds per factor of e.
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
            self.cam_planet = self.cam_planet + np.array(world) * speed * dt
        rho = float(np.linalg.norm(self.cam_planet))
        low = self.radius + ground + MIN_CLEARANCE
        high = self.radius * (1.0 + MAX_ALTITUDE_RADII)
        self.cam_planet *= min(max(rho, low), high) / rho
        self._ground = ground

    def _update_sky(self, east, north, up, altitude, far):
        self._sky.set_pos(0.0, 0.0, 0.0)
        self._sky.set_quat(self.render, _quat_from_axes(east, north, up))
        self._sky.set_scale(0.9 * far)
        bright = 1.0 - _smoothstep(*SKY_FADE, altitude)
        day = _smoothstep(-0.15, 0.1, float(self._sun_planet @ up))
        k = bright * (0.05 + 0.95 * day)
        self._sky.set_color_scale(k, k, k, 1.0)

    def _update_sun(self, up):
        # Colours follow the sun's height above the horizon under the camera.
        elevation = math.degrees(math.asin(np.clip(self._sun_planet @ up, -1.0, 1.0)))
        _, sun, sky = sun_lighting(0.0, elevation)
        self.terrain.set_sun(tuple(self._sun_planet), sun, sky)

    def set_local_sun(self, azimuth: float, elevation: float):
        """Put the sun at an azimuth/elevation (degrees) as seen from the nadir."""
        east, north, up = self._true_frame()
        d, _, _ = sun_lighting(azimuth, elevation)
        self._sun_planet = d[0] * east + d[1] * north + d[2] * up

    def _turn_sun(self, axis: int, degrees: float):
        """Swing the sun in the sky as seen from the nadir (axis 0 azimuth,
        1 elevation)."""
        east, north, up = self._true_frame()
        s = self._sun_planet
        el = math.degrees(math.asin(np.clip(s @ up, -1.0, 1.0)))
        az = math.degrees(math.atan2(s @ east, s @ north))
        if axis == 0:
            az += degrees
        else:
            el = max(-10.0, min(89.0, el + degrees))
        d, _, _ = sun_lighting(az, el)
        self._sun_planet = d[0] * east + d[1] * north + d[2] * up

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
        self._sea_frame.set_pos(*(sea_point - gx * e - gy * n))
        for methods in list(self._game.method_lists.values()):
            for method in methods:
                method()

    def _update_probe(self):
        center = self.cam_planet + PROBE_DISTANCE * self._forward()
        self._probe_ball.set_pos(*self.terrain.to_render(center))
        contact = self.terrain.query.sphere_contact(center, PROBE_RADIUS)
        self._probe_normal.remove_node()
        if contact is None:
            self._probe_hit.hide()
            self._probe_ball.set_color(0.3, 0.7, 1.0, 0.35)
            self._probe_normal = self._probe.attach_new_node("normal")
            return
        hit, normal = contact
        hit_render = self.terrain.to_render(hit)
        self._probe_hit.show()
        self._probe_hit.set_pos(*hit_render)
        self._probe_ball.set_color(1.0, 0.5, 0.2, 0.35)
        lines = LineSegs()
        lines.set_thickness(3)
        lines.set_color(1.0, 1.0, 0.2, 1.0)
        lines.move_to(*hit_render)
        lines.draw_to(*(hit_render + normal * 25.0))
        self._probe_normal = self._probe.attach_new_node(lines.create())

    def _update_hud(self, altitude: float):
        up = self.cam_planet / np.linalg.norm(self.cam_planet)
        lat, lon = latlon(up[None, :])
        face, _, _ = dirs_to_face(up[None, :])
        weights = self.terrain.query.biome_weights_at(self.cam_planet, generate=False)
        biome_text = "  ".join(
            f"{name} {w:.0%}" for name, w in zip(BIOMES, weights) if w >= 0.05
        )
        s = self.terrain.stats
        spec = self.terrain.spec
        heading, pitch = self._true_heading_pitch()
        above = altitude - self._ground
        face_name = FACE_NAMES[int(face[0])]
        speed = self.speed / SPEED
        split_km, far_km = self.depth.split / 1e3, self.depth.far / 1e3
        lines = [
            f"{spec.name}  seed {spec.seed}  radius {spec.radius / 1e3:,.0f} km"
            f"   view: {_DEBUG_VIEWS[self._debug]}"
            f"   blend {self.terrain.blend_depth:.2f}"
            f"   detail: {self._detail_text()}",
            f"altitude {altitude:,.0f} m   above ground {above:,.0f} m"
            f"   lat {lat[0]:.4f}  lon {lon[0]:.4f}  face {face_name}"
            f"   heading {heading:.0f}  pitch {pitch:.0f}   speed x{speed:g}",
            f"under camera: {biome_text or '(not generated yet)'}",
            f"chunks: {s['shown']} shown (finest level {s['finest']}), "
            f"{s['uploaded']} on GPU, {s['stored']} stored, "
            f"{s['pending']} generating, {s['wanted']} wanted",
            f"gen {s['gen_ms']:.0f} ms/chunk "
            f"({self.terrain.executor.workers} workers)   "
            f"upload {s['upload_ms']:.2f} ms/chunk   LOD pass {s['select_ms']:.1f} ms"
            f"   depth split {split_km:,.1f} km, far {far_km:,.0f} km",
        ]
        self._hud.setText("\n".join(lines))

    def _true_heading_pitch(self) -> tuple[float, float]:
        """:returns: the view's heading (degrees counter-clockwise from north,
        as Panda's H) and pitch, in the geographic frame"""
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
        self.terrain.set_spec(spec)
        for mark, _ in self._laser_marks:
            mark.remove_node()
        self._laser_marks.clear()
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

    def _toggle_reflection(self):
        self._reflected = not self._reflected
        self.terrain.set_reflected(self._reflected)

    def _toggle_textures(self):
        self._textured = not self._textured
        self.terrain.set_textured(self._textured)

    def _toggle_wireframe(self):
        self._wireframe = not self._wireframe
        self.terrain.set_wireframe(self._wireframe)

    def _cycle_debug(self):
        self._debug = (self._debug + 1) % len(_DEBUG_VIEWS)
        self.terrain.set_debug_mode(self._debug)

    def _cycle_detail(self):
        self.terrain.set_min_level(
            (self.terrain.min_level + 1) % (MAX_DROPPED_LEVELS + 1)
        )

    def _detail_text(self) -> str:
        level = self.terrain.min_level
        spacing = self.terrain.grid.spacing(level)
        text = f"{spacing:.1f} m"
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
        direction = self._forward()
        start = self.cam_planet.copy()
        hit = self.terrain.query.segment_intersect(
            start, start + LASER_RANGE * direction
        )
        if hit is None:
            print("laser: no hit within range")
            return
        t, point, normal = hit
        distance = t * LASER_RANGE
        material = self.terrain.query.material_at(point, generate=False)
        height = float(np.linalg.norm(point)) - self.radius
        up = point / np.linalg.norm(point)
        print(
            f"laser: hit at {distance:,.0f} m, {material}, height {height:.1f} m, "
            f"slope {math.degrees(math.acos(np.clip(normal @ up, -1.0, 1.0))):.0f} deg"
        )
        mark = self.loader.load_model("models/misc/sphere")
        mark.set_scale(max(2.0, distance * 0.004))
        mark.set_color(1.0, 0.9, 0.1, 1.0)
        mark.set_light_off()
        mark.reparent_to(self.render)
        self._laser_marks.append((mark, point))

    def _scale_blend(self, factor):
        depth = max(0.05, min(3.0, self.terrain.blend_depth * factor))
        self.terrain.set_blend_depth(depth)

    def _scale_speed(self, factor):
        self.speed = max(10.0, min(SPEED * 64, self.speed * factor))

    def _place_camera(self, lat, lon, altitude, heading, pitch):
        """Put the camera over (lat, lon), with heading and pitch in the local
        geographic frame (degrees; heading counter-clockwise from north, as
        Panda's H) and no roll."""
        frame = TangentFrame.at(lat, lon)
        self.cam_planet = frame.up * (self.radius + altitude)
        local = LQuaternionf()
        local.set_hpr((heading, pitch, 0.0))
        self._orientation = local * _quat_from_axes(frame.east, frame.north, frame.up)
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
        self._sun_planet = d[0] * frame.east + d[1] * frame.north + d[2] * frame.up

    def _jump_to_orbit(self):
        lat, lon = latlon(self.cam_planet[None, :] / np.linalg.norm(self.cam_planet))
        self._place_camera(
            float(lat[0]), float(lon[0]), ORBIT_RADII * self.radius, 0.0, -89.9
        )

    def _print_pose(self):
        up = self.cam_planet / np.linalg.norm(self.cam_planet)
        lat, lon = latlon(up[None, :])
        heading, pitch = self._true_heading_pitch()
        altitude = float(np.linalg.norm(self.cam_planet)) - self.radius
        right = self._orientation.get_right()
        roll = -math.degrees(math.asin(np.clip(np.array(right) @ up, -1.0, 1.0)))
        print(
            f"camera: {{lat: {lat[0]:.5f}, lon: {lon[0]:.5f}, alt: {altitude:.0f}, "
            f"heading: {heading:.1f}, pitch: {pitch:.1f}}}  roll {roll:.1f}"
        )
        print(f"terrain: {self.terrain.stats}")

    def _quit(self):
        self.terrain.destroy()
        sys.exit()


def main():
    parser = argparse.ArgumentParser(description="Procedural terrain demo")
    parser.add_argument("--level", default=str(DEMO_LEVEL))
    parser.add_argument(
        "--radius", type=float, default=None, help="planet radius, metres"
    )
    parser.add_argument(
        "--backend", default="process", choices=("process", "thread", "inline")
    )
    args = parser.parse_args()
    loadPrcFileData(
        "",
        "\n".join(
            [
                "window-title Procedural terrain demo",
                "win-size 1600 900",
                "audio-library-name null",
                "framebuffer-multisample 1",
                "multisamples 4",
                "depth-bits 24",
            ]
        ),
    )
    TerrainDemo(args.level, args.backend, args.radius).run()


if __name__ == "__main__":
    main()
