"""
Standalone flythrough demo for the procedural planet terrain.

Flies a free camera over a procedurally generated planet with hand-made
stamps (world_builder/terrain/levels/demo.yaml), the WorldBuilder ocean, and
haze. Run it from an environment where world_builder is importable::

    python scripts/demo_terrain.py [--level FILE]
        [--backend process|thread|inline]

The default backend generates terrain in spawned worker processes, which
re-import this script: keep the ``if __name__ == "__main__"`` guard.

Controls (AZERTY):
    Z / S           pitch down / up
    Q / D           roll left / right
    A / E           yaw left / right
    UP / DOWN       fly forward / backward
    LEFT / RIGHT    slide left / right
    W / X           slide up / down
    SHIFT           hold to move 5x faster (speed also grows with altitude)
    PAGE UP / DOWN  double / halve the base speed

    F1 - F4         planet preset: earthlike / desert / forest / ice
    N               next seed
    L               reload the level file (edit stamps live)
    O               toggle the ocean
    M               toggle the terrain in the ocean's reflection
    , / ;           sharper / softer biome transitions
    T               toggle textures (off: flat biome colours)
    C               toggle the curvature drop
    B               toggle wireframe
    V               cycle the debug view: shaded / LOD / biomes / slope /
                    normals / skirts
    H               toggle the draw distance: haze limit / full horizon
    P               cycle the detail: full, then dropping the finest LOD levels
                    one at a time (vertex spacing 2 m, 4 m, 8 m, ... 64 m)
    G               toggle the collision probe: a 20 m sphere ahead of the
                    camera showing its contact point and normal
    SPACE           fire a probe laser along the view; marks the hit
    J / K           swing the sun around the sky (azimuth)
    I / U           raise / lower the sun
    R               reset the camera to the level's start pose
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
    LVecBase3f,
    TextNode,
    TransparencyAttrib,
    Vec3,
    loadPrcFileData,
)

from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.chart import latlon
from world_builder.terrain.chunkgen import chunk_spacing
from world_builder.terrain.level import DEMO_LEVEL, load_level

CAMERA_FAR_M = 600000.0  # the game's own far plane (actors.player)
HAZE_COLOR = (0.62, 0.71, 0.84)  # display space; sky at the horizon
ZENITH_COLOR = (0.28, 0.45, 0.76)
HAZE_DISTANCE = 150000.0
SPEED = 150.0  # metres/second at low altitude
BOOST = 5.0
MIN_CLEARANCE = 3.0  # metres the camera stays above the ground
# Speed grows with height above ground, up to this factor; unbounded, holding
# climb with boost compounds into an exponential escape.
MAX_SPEED_FACTOR = 40.0
MAX_DROPPED_LEVELS = 5  # P cycles the finest drawn level from 0 to this
MAX_ALTITUDE = 40000.0  # the terrain and ocean are built for eyes up to ~20 km
PROBE_RADIUS = 20.0
PROBE_DISTANCE = 60.0
LASER_RANGE = 30000.0

_ROTATIONS = (
    ("s", "z", "p", 40.0),
    ("d", "q", "r", 60.0),
    ("a", "e", "h", 40.0),
)
_TRANSLATIONS = (
    ("arrow_up", "arrow_down", 1),  # forward (+Y)
    ("arrow_right", "arrow_left", 0),  # right (+X)
    ("w", "x", 2),  # up (+Z)
)
_SUN_KEYS = (("j", "k", "_sun_azimuth", 45.0), ("i", "u", "_sun_elevation", 25.0))
_CONTROLS = (
    "shift",
    *(key for row in (*_ROTATIONS, *_TRANSLATIONS, *_SUN_KEYS) for key in row[:2]),
)
_DEBUG_VIEWS = ("shaded", "LOD", "biomes", "slope", "normals", "skirts")


def sun_lighting(azimuth: float, elevation: float):
    """Sun direction and colours from its position; a hand-made ramp.

    :param azimuth: degrees, 0 = +Y (north), increasing toward +X
    :param elevation: degrees above the horizon
    :returns: (direction, sun colour, ambient colour), linear
    """
    az, el = math.radians(azimuth), math.radians(elevation)
    direction = Vec3(
        math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)
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


class TerrainDemo(ShowBase):
    """A free-flying camera over the procedural planet."""

    def __init__(self, level_path, backend: str):
        super().__init__()
        self.setFrameRateMeter(True)
        self.disable_mouse()
        self._clock = ClockObject.get_global_clock()
        self.camLens.set_near_far(1.0, CAMERA_FAR_M)
        self.camLens.set_fov(70)
        self.set_background_color(*HAZE_COLOR, 1.0)

        # The game's frame layout: the world hangs under a root offset by
        # -camera every frame, so the camera sits at the render origin.
        self.root = self.render.attach_new_node("game_root")
        self.cam_world = np.zeros(3)
        self.speed = SPEED
        self._sun_azimuth = 210.0
        self._sun_elevation = 35.0

        self.level_path = level_path
        self.preset = None
        spec, self.start_pose = load_level(level_path)
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
        self._make_sky()
        self.ocean = self._make_ocean()
        self._ocean_wanted = True
        self._apply_ocean()
        self._reflected = True
        self._textured = True
        self._curved = True
        self._wireframe = False
        self._debug = 0
        self._full_horizon = False
        self._probe = self._make_probe()
        self._laser_marks = self.root.attach_new_node("laser_marks")
        self._apply_sun()

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
            ("c", self._toggle_curvature, []),
            ("b", self._toggle_wireframe, []),
            ("v", self._cycle_debug, []),
            ("h", self._toggle_horizon, []),
            ("p", self._cycle_detail, []),
            ("g", self._toggle_probe, []),
            ("space", self._fire_laser, []),
            (",", self._scale_blend, [0.8]),
            (";", self._scale_blend, [1.25]),
            ("page_up", self._scale_speed, [2.0]),
            ("page_down", self._scale_speed, [0.5]),
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

    # ── scene pieces ───────────────────────────────────────────────────────────

    def _make_ocean(self):
        """The game's real curved ocean, behind a minimal stand-in game object."""
        from world_builder.ocean import Ocean

        self._game = SimpleNamespace(
            app=self,
            root_node=self.root,
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
        )

    def _make_sky(self):
        """A gradient dome: haze colour at and below the horizon, blue overhead."""
        rings, segments, radius = 24, 48, 400000.0
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
                    radius * math.cos(el) * math.cos(az),
                    radius * math.cos(el) * math.sin(az),
                    radius * math.sin(el),
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
        return sky

    def _make_probe(self):
        probe = self.root.attach_new_node("probe")
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
        self.root.set_pos(*(-self.cam_world))
        self.terrain.update(self.cam_world)
        if self.ocean is not None and not self.ocean.base_node.is_hidden():
            for methods in list(self._game.method_lists.values()):
                for method in methods:
                    method()
        if not self._probe.is_hidden():
            self._update_probe()
        self._update_hud()
        return task.cont

    def _fly(self, dt: float):
        for plus, minus, axis, rate in _ROTATIONS:
            sign = self._keys[plus] - self._keys[minus]
            if sign:
                setter = getattr(self.camera, f"set_{axis}")
                getter = getattr(self.camera, f"get_{axis}")
                setter(getter() + sign * rate * dt)
        for plus, minus, attr, rate in _SUN_KEYS:
            sign = self._keys[plus] - self._keys[minus]
            if sign:
                setattr(self, attr, getattr(self, attr) + sign * rate * dt)
                self._sun_elevation = max(-10.0, min(89.0, self._sun_elevation))
                self._apply_sun()

        ground = float(
            self.terrain.query.height_at(
                self.cam_world[0], self.cam_world[1], generate=False
            )
        )
        altitude = max(self.cam_world[2] - ground, 0.0)
        speed = (
            self.speed
            * min(MAX_SPEED_FACTOR, max(1.0, altitude / 300.0))
            * (BOOST if self._keys["shift"] else 1.0)
        )
        move = np.zeros(3)
        for plus, minus, axis in _TRANSLATIONS:
            move[axis] = self._keys[plus] - self._keys[minus]
        if move.any():
            quat = self.camera.get_quat(self.render)
            world = quat.xform(Vec3(*move))
            self.cam_world += np.array([world.x, world.y, world.z]) * speed * dt
        self.cam_world[2] = min(
            max(self.cam_world[2], ground + MIN_CLEARANCE), MAX_ALTITUDE
        )
        self._ground = ground

    def _update_probe(self):
        forward = self.camera.get_quat(self.render).get_forward()
        center = self.cam_world + PROBE_DISTANCE * np.array(
            [forward.x, forward.y, forward.z]
        )
        self._probe_ball.set_pos(*center)
        contact = self.terrain.query.sphere_contact(center, PROBE_RADIUS)
        self._probe_normal.remove_node()
        if contact is None:
            self._probe_hit.hide()
            self._probe_ball.set_color(0.3, 0.7, 1.0, 0.35)
            self._probe_normal = self._probe.attach_new_node("normal")
            return
        hit, normal = contact
        self._probe_hit.show()
        self._probe_hit.set_pos(*hit)
        self._probe_ball.set_color(1.0, 0.5, 0.2, 0.35)
        lines = LineSegs()
        lines.set_thickness(3)
        lines.set_color(1.0, 1.0, 0.2, 1.0)
        lines.move_to(*hit)
        lines.draw_to(*(hit + normal * 25.0))
        self._probe_normal = self._probe.attach_new_node(lines.create())

    def _update_hud(self):
        x, y, z = self.cam_world
        lat, lon = latlon(
            self.terrain.planet.chart.to_dirs(np.array([x]), np.array([y]))
        )
        weights = self.terrain.query.biome_weights_at(x, y, generate=False)
        biome_text = "  ".join(
            f"{name} {w:.0%}" for name, w in zip(BIOMES, weights) if w >= 0.05
        )
        s = self.terrain.stats
        spec = self.terrain.spec
        lines = [
            f"{spec.name}  seed {spec.seed}   view: {_DEBUG_VIEWS[self._debug]}"
            f"   draw: {'horizon' if self._full_horizon else 'haze limit'}"
            f"   blend {self.terrain.blend_depth:.2f}"
            f"   detail: {self._detail_text()}",
            f"altitude {z:,.0f} m   above ground {z - self._ground:,.0f} m   "
            f"lat {lat[0]:.4f}  lon {lon[0]:.4f}   speed x{self.speed / SPEED:g}",
            f"under camera: {biome_text or '(not generated yet)'}",
            f"chunks: {s['shown']} shown (finest level {s['finest']}), "
            f"{s['uploaded']} on GPU, {s['stored']} stored, "
            f"{s['pending']} generating, {s['wanted']} wanted",
            f"gen {s['gen_ms']:.0f} ms/chunk "
            f"({self.terrain.executor.workers} workers)   "
            f"upload {s['upload_ms']:.2f} ms/chunk   LOD pass {s['select_ms']:.1f} ms",
        ]
        self._hud.setText("\n".join(lines))

    # ── key handlers ───────────────────────────────────────────────────────────

    def _set_key(self, key, pressed):
        self._keys[key] = pressed

    def _set_spec(self, spec):
        self.terrain.set_spec(spec)
        self._laser_marks.node().remove_all_children()
        self._apply_ocean()

    def _set_preset(self, name):
        self.preset = name
        spec, _ = load_level(self.level_path, preset=name, seed=self.seed)
        self._set_spec(spec)

    def _next_seed(self):
        self.seed += 1
        spec, _ = load_level(self.level_path, preset=self.preset, seed=self.seed)
        self._set_spec(spec)

    def _reload(self):
        try:
            spec, self.start_pose = load_level(
                self.level_path, preset=self.preset, seed=self.seed
            )
        except Exception as error:  # a typo in the YAML must not kill the demo
            print(f"level reload failed: {error}")
            return
        self._set_spec(spec)
        print(f"reloaded {self.level_path}")

    def _apply_ocean(self):
        visible = self._ocean_wanted and self.terrain.spec.has_ocean
        if visible:
            self.ocean.base_node.show()
            self.ocean.refl_buffer.set_active(True)
        else:
            self.ocean.base_node.hide()
            self.ocean.refl_buffer.set_active(False)

    def _toggle_ocean(self):
        self._ocean_wanted = not self._ocean_wanted
        self._apply_ocean()

    def _toggle_reflection(self):
        self._reflected = not self._reflected
        self.terrain.set_reflected(self._reflected)

    def _toggle_textures(self):
        self._textured = not self._textured
        self.terrain.set_textured(self._textured)

    def _toggle_curvature(self):
        self._curved = not self._curved
        self.terrain.set_curvature(self._curved)

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
        spacing = chunk_spacing(level)
        return "full (2 m)" if level == 0 else f"-{level} levels ({spacing:g} m)"

    def _toggle_horizon(self):
        self._full_horizon = not self._full_horizon
        self.terrain.set_max_distance(
            None if self._full_horizon else HAZE_DISTANCE * 1.05
        )

    def _toggle_probe(self):
        if self._probe.is_hidden():
            self._probe.show()
        else:
            self._probe.hide()

    def _fire_laser(self):
        forward = self.camera.get_quat(self.render).get_forward()
        direction = np.array([forward.x, forward.y, forward.z])
        start = self.cam_world.copy()
        hit = self.terrain.query.segment_intersect(
            start, start + LASER_RANGE * direction
        )
        if hit is None:
            print("laser: no hit within range")
            return
        t, point, normal = hit
        distance = t * LASER_RANGE
        material = self.terrain.query.material_at(point[0], point[1], generate=False)
        print(
            f"laser: hit at {distance:,.0f} m, {material}, point "
            f"({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}), normal "
            f"({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
        )
        mark = self.loader.load_model("models/misc/sphere")
        mark.set_scale(max(2.0, distance * 0.004))
        mark.set_color(1.0, 0.9, 0.1, 1.0)
        mark.set_light_off()
        mark.reparent_to(self._laser_marks)
        mark.set_pos(*point)

    def _scale_blend(self, factor):
        depth = max(0.05, min(3.0, self.terrain.blend_depth * factor))
        self.terrain.set_blend_depth(depth)

    def _scale_speed(self, factor):
        self.speed = max(10.0, min(SPEED * 64, self.speed * factor))

    def _apply_sun(self):
        direction, sun, sky = sun_lighting(self._sun_azimuth, self._sun_elevation)
        self.terrain.set_sun(direction, sun, sky)

    def _reset_camera(self):
        pose = self.start_pose or {}
        self.cam_world = np.array(
            [
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("z", 800.0)),
            ]
        )
        self.camera.set_pos(0.0, 0.0, 0.0)
        self.camera.set_hpr(
            float(pose.get("heading", 0.0)), float(pose.get("pitch", 0.0)), 0.0
        )
        self._ground = 0.0

    def _print_pose(self):
        h, p, r = self.camera.get_hpr()
        x, y, z = self.cam_world
        print(
            f"camera: {{x: {x:.0f}, y: {y:.0f}, z: {z:.0f}, heading: {h:.1f}, "
            f"pitch: {p:.1f}}}  roll {r:.1f}"
        )
        print(f"terrain: {self.terrain.stats}")

    def _quit(self):
        self.terrain.destroy()
        sys.exit()


def main():
    parser = argparse.ArgumentParser(description="Procedural terrain demo")
    parser.add_argument("--level", default=str(DEMO_LEVEL))
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
    TerrainDemo(args.level, args.backend).run()


if __name__ == "__main__":
    main()
