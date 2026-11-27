"""
Fly through a world file.

    python scripts/demo_world.py [WORLD.yaml] [options]

The default world is src/world_builder/worlds/three_planets.yaml (a home
planet and its moon, an ice planet and its ring); deep_space.yaml is a field
of asteroids in open space. Any world file works: write your own (see
docs/source/world_file.md).

The whole of loading it is two lines (see :class:`WorldDemo`)::

    self.world = load_world(self, path, settings=settings)    # once
    self.world.update(self.cam_world)                         # every frame

and the rest of this script is a camera, a HUD and the keys.

Options: ``--quality`` (and ``--terrain-quality`` etc.: a preset per part),
``--backend``, ``--size "W H"``, ``--frame-cap``. The camera starts above the
first body's reference point (or at the first asteroid field's clear corridor
or edge); place it elsewhere with ``--body NAME --lat LAT --lon LON --alt M
[--heading DEG --pitch DEG]`` or ``--position X Y Z [--heading --pitch]`` (world
frame, metres). ``--frames N [--shot FILE.png]`` quits after N frames, saving
a picture.

Controls (AZERTY; the keys are where they are for the left hand)::

    arrows     forward / back, slide right / left     W / X   up / down
    S / Z      pitch up / down     D / Q   roll     A / E   yaw
    SHIFT      x5 speed (speed grows with the height above the ground)
    PAGE UP/DN double / halve the speed
    Y          level the roll
    HOME       3 radii above the nearest body, looking down
    TAB        to the next body, over the side facing the camera
    R / END    back to the start
    F1 - F5, F7, F8   cycle terrain, atmosphere, sky, clouds, ocean, shadows,
               models quality;  F6 all
    F          print the camera pose in the --lat/--lon/--alt form
    ESC        quit

The HUD (top left) shows the nearest body, the height above it and above
its ground, where the camera looks and what the terrain is doing.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ClockObject,
    Filename,
    GraphicsWindow,
    LQuaternionf,
    TextNode,
    Vec3,
)

sys.path.insert(0, str(Path(__file__).parent))

import tools  # noqa: E402
from world_builder import load_world  # noqa: E402
from world_builder.asteroids.spec import Annulus, Box, Sphere  # noqa: E402
from world_builder.quality import QualityLevel  # noqa: E402
from world_builder.settings import SECTIONS, WorldSettings  # noqa: E402
from world_builder.terrain.chart import TangentFrame, latlon  # noqa: E402

DEFAULT_WORLD = tools.DEFAULT_WORLD
SPEED = 300.0  # metres/second at low altitude
BOOST = 5.0
MIN_CLEARANCE = 3.0  # metres the camera stays above the ground
ORBIT_RADII = 3.0  # HOME and TAB jump this high
HUD_PERIOD = 0.2  # seconds between HUD refreshes
HUD_LINE = 0.048  # the HUD's line spacing (its text's scale 0.04, as one text's)
#: Where a world with no body starts: this many times a field's extent from
#: its centre, when it has no corridor to start in.
START_EXTENT = 1.2

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
_CONTROLS = ("shift", *(k for row in (*_ROTATIONS, *_TRANSLATIONS) for k in row[:2]))
#: F1 - F5, F7 and F8 cycle these parts' presets, F6 all.
QUALITY_KEYS = (
    ("f1", "terrain"),
    ("f2", "atmosphere"),
    ("f3", "sky"),
    ("f4", "clouds"),
    ("f5", "ocean"),
    ("f7", "shadows"),
    ("f8", "models"),
)
QUALITY_ALL_KEY = "f6"


def _level_name(settings: WorldSettings, section: str) -> str:
    level = settings.section_level(section)
    return "custom" if level is None else level.value


def _next_level(level: QualityLevel | None) -> QualityLevel:
    """:returns: the next preset, cheapest after dearest (HIGH after custom)"""
    if level is None:
        return QualityLevel.HIGH
    levels = list(QualityLevel)
    return levels[(levels.index(level) + 1) % len(levels)]


class WorldDemo(ShowBase):
    """A free-flying camera in a world file's world.

    :ivar world: the loaded :class:`GameWorld` (``world.world`` is the
        :class:`World`)
    :ivar settings: what every part costs
    :ivar cam_world: the camera in the world frame, metres, float64 (the
        camera itself stays at render's origin)
    :ivar speed: metres/second at low altitude
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_WORLD,
        settings: WorldSettings | None = None,
        backend: str = "process",
        start: dict | None = None,
    ) -> None:
        """
        :param path: the world file
        :param settings: what every part costs; None = HIGH
        :param backend: where chunks and cloud tiles are generated
        :param start: where the camera starts: ``{body, lat, lon, alt,
            heading, pitch}`` or ``{position: (x, y, z), heading, pitch}``;
            None = :meth:`default_start`
        """
        super().__init__()
        self.setFrameRateMeter(True)
        self.disable_mouse()
        self._clock = ClockObject.get_global_clock()
        self.camLens.set_fov(70)
        self.set_background_color(0.0, 0.0, 0.0, 1.0)
        self.settings = settings or WorldSettings()
        self.speed = SPEED
        self.cam_world = np.zeros(3)
        self._orientation = LQuaternionf()
        self._ground = 0.0
        print("starting terrain workers ...")
        # The one line: lens, bodies, fields, sky, sun, lights, depth camera.
        self.world = load_world(self, path, settings=self.settings, backend=backend)
        self.bodies = list(self.world.world.bodies)
        self.body = self.bodies[0] if self.bodies else None
        self.start = start or self.default_start()
        self._keys = dict.fromkeys(_CONTROLS, False)
        self._bind_keys()
        # One text per line, set only when its text changed.
        self._hud: list[OnscreenText] = []
        self._hud_hidden = False
        self._hud_time = -1.0
        self.reset_camera()
        self.taskMgr.add(self._update, "world_demo_update")

    # ── start ──────────────────────────────────────────────────────────────

    def default_start(self) -> dict:
        """:returns: a start: over the first body's reference point; else at
        the first field's first keep-out (a corridor to fly along), else at
        the field's edge looking at it; else the origin"""
        if self.bodies:
            spec = self.bodies[0].look.spec
            return {
                "body": self.bodies[0].name,
                "lat": spec.reference_lat,
                "lon": spec.reference_lon,
                "alt": tools.ground_height(spec, spec.reference_lat, spec.reference_lon)
                + tools.START_CLEARANCE,
                "heading": 0.0,
                "pitch": -8.0,
            }
        if not self.world.spec.fields:
            return {"position": (0.0, 0.0, 0.0)}
        look = self.world.spec.fields[0]
        centre = np.array(look.position)
        keep_outs = look.spec.keep_outs
        if keep_outs and keep_outs[0].end is not None:
            a, b = np.array(keep_outs[0].centre), np.array(keep_outs[0].end)
            return {"position": tuple(centre + a), **_look_along(b - a)}
        shape = look.spec.shape
        extent = {
            Box: lambda s: max(s.half_size),
            Sphere: lambda s: s.radius,
            Annulus: lambda s: s.outer,
        }.get(type(shape), lambda s: 1000.0)(shape)
        return {
            "position": tuple(centre + (0.0, -START_EXTENT * extent, 0.0)),
            **_look_along(np.array([0.0, 1.0, 0.0])),
        }

    def reset_camera(self) -> None:
        """Back to the start."""
        s = self.start
        if "position" in s:
            self.place_at(s["position"], s.get("heading", 0.0), s.get("pitch", 0.0))
            return
        body = next((b for b in self.bodies if b.name == s["body"]), self.bodies[0])
        self.place_camera(
            s["lat"],
            s["lon"],
            s["alt"],
            s.get("heading", 0.0),
            s.get("pitch", 0.0),
            body,
        )

    def place_at(self, position, heading: float = 0.0, pitch: float = 0.0) -> None:
        """Put the camera at a world position, heading and pitch (degrees) in
        the world frame, no roll."""
        self.cam_world = np.array(position, dtype=np.float64)
        q = LQuaternionf()
        q.set_hpr((heading, pitch, 0.0))
        self._orientation = q
        self._ground = 0.0

    def place_camera(
        self, lat: float, lon: float, altitude: float, heading, pitch, body=None
    ) -> None:
        """Put the camera over (lat, lon) of a body (the first by default),
        heading and pitch in the local geographic frame (degrees; heading
        counter-clockwise from north, as Panda's H), no roll."""
        body = body or self.bodies[0]
        frame = TangentFrame.at(lat, lon)
        self.cam_world = body.position + frame.up * (body.radius + altitude)
        local = LQuaternionf()
        local.set_hpr((heading, pitch, 0.0))
        self._orientation = local * tools.quat_from_axes(
            frame.east, frame.north, frame.up
        )
        self.body = body
        self._ground = 0.0

    # ── per frame ──────────────────────────────────────────────────────────

    def _update(self, task) -> int:
        dt = min(self._clock.get_dt(), 0.1)
        if self.bodies:
            self.body = self.world.world.nearest(self.cam_world)
        self._fly(dt)
        self.camera.set_pos(0.0, 0.0, 0.0)
        self.camera.set_quat(self.render, self._orientation)
        self.world.update(self.cam_world)  # the other line
        now = self._clock.get_frame_time()
        if now - self._hud_time >= HUD_PERIOD:
            self._hud_time = now
            self._update_hud()
        return task.cont

    def _forward(self) -> np.ndarray:
        f = self._orientation.get_forward()
        return np.array([f.x, f.y, f.z])

    def _relative(self, body=None) -> np.ndarray:
        """:returns: the camera in a body's frame (the nearest's by default)"""
        return self.cam_world - (body or self.body).position

    def _up(self) -> np.ndarray:
        if self.body is None:
            return np.array([0.0, 0.0, 1.0])
        p = self._relative()
        return p / np.linalg.norm(p)

    def _fly(self, dt: float) -> None:
        for plus, minus, kind, rate in _ROTATIONS:
            sign = self._keys[plus] - self._keys[minus]
            if sign:
                self._rotate(kind, sign * rate * dt)
        body = self.body
        above = 1.0e3
        ground = 0.0
        if body is not None:
            relative = self._relative(body)
            ground = float(body.terrain.query.height_at(relative, generate=False))
            above = max(float(np.linalg.norm(relative)) - body.radius - ground, 0.0)
        # Speed grows with the height above the nearest ground: a descent
        # from orbit is exponential, a few seconds per factor of e.
        speed = (
            self.speed
            * max(1.0, above / 300.0)
            * (BOOST if self._keys["shift"] else 1.0)
        )
        move = np.zeros(3)
        for plus, minus, axis in _TRANSLATIONS:
            move[axis] = self._keys[plus] - self._keys[minus]
        if move.any():
            moved = self._orientation.xform(Vec3(*move))
            self.cam_world = self.cam_world + np.array(moved) * speed * dt
        if body is not None:  # stay above the nearest ground
            relative = self._relative(body)
            rho = float(np.linalg.norm(relative))
            low = body.radius + ground + MIN_CLEARANCE
            if rho < low:
                self.cam_world = body.position + relative * (low / rho)
        self._ground = ground

    def _rotate(self, kind: str, degrees: float) -> None:
        """Yaw about the local vertical, pitch and roll about the camera's
        own right and forward axes."""
        q = LQuaternionf()
        if kind == "yaw":
            q.set_from_axis_angle(degrees, Vec3(*self._up()))
            self._orientation = self._orientation * q  # a world-frame turn
        else:
            q.set_from_axis_angle(degrees, _LOCAL_AXES[kind])
            self._orientation = q * self._orientation  # a camera-frame turn
        self._orientation.normalize()

    # ── HUD ────────────────────────────────────────────────────────────────

    def _true_heading_pitch(self) -> tuple[float, float]:
        """:returns: the view's heading (degrees counter-clockwise from north,
        as Panda's H) and pitch in the nearest body's geographic frame"""
        f = self._forward()
        if self.body is None:
            return _heading_pitch(f)
        up = self._up()
        lat, lon = latlon(up[None, :])
        frame = TangentFrame.at(float(lat[0]), float(lon[0]))
        heading = math.degrees(math.atan2(-(f @ frame.east), f @ frame.north))
        pitch = math.degrees(math.asin(np.clip(f @ up, -1.0, 1.0)))
        return heading, pitch

    def _update_hud(self) -> None:
        quality = "   ".join(
            f"{SECTIONS[name].LABEL.lower()} {_level_name(self.settings, name)}"
            for _, name in QUALITY_KEYS
        )
        depth = self.world.depth
        heading, pitch = self._true_heading_pitch()
        lines = [
            f"camera {self.cam_world[0]:,.0f} {self.cam_world[1]:,.0f} "
            f"{self.cam_world[2]:,.0f}   heading {heading:.0f}  pitch {pitch:.0f}"
            f"   speed x{self.speed / SPEED:g}"
            f"   depth near {depth.near:,.0f} m, far {depth.far / 1e3:,.0f} km"
        ]
        body = self.body
        if body is not None:
            relative = self._relative()
            altitude = float(np.linalg.norm(relative)) - body.radius
            lat, lon = latlon(self._up()[None, :])
            s = body.terrain.stats
            lines.insert(
                0,
                f"{body.name}: {body.look.spec.name}  seed {body.look.spec.seed}  "
                f"radius {body.radius / 1e3:,.0f} km   altitude {altitude:,.0f} m   "
                f"above ground {altitude - self._ground:,.0f} m   "
                f"lat {lat[0]:.4f}  lon {lon[0]:.4f}",
            )
            lines.append(
                f"chunks: {s['shown']} shown (finest level {s['finest']}), "
                f"{s['uploaded']} on GPU, {s['pending']} generating, "
                f"{s['wanted']} wanted"
            )
            others = "   ".join(
                f"{b.name} {np.linalg.norm(self._relative(b)) / 1e3:,.0f} km"
                for b in self.bodies
                if b is not body
            )
            if others:
                lines.append(others)
        lines.append(f"quality: {quality}")
        self._show_hud(lines)

    def hide_hud(self) -> None:
        """No HUD from now on (shots and benchmarks)."""
        self._hud_hidden = True
        for text in self._hud:
            text.hide()

    def _show_hud(self, lines: list[str]) -> None:
        """The HUD's lines, each its own text, set only when it changed."""
        if self._hud_hidden:
            return
        while len(self._hud) < len(lines):
            self._hud.append(
                OnscreenText(
                    text="",
                    pos=(-1.32, 0.92 - HUD_LINE * len(self._hud)),
                    scale=0.04,
                    fg=(1, 1, 1, 0.9),
                    shadow=(0, 0, 0, 0.7),
                    align=TextNode.ALeft,
                    mayChange=True,
                )
            )
        for k, text in enumerate(self._hud):
            line = lines[k] if k < len(lines) else ""
            if text.getText() != line:
                text.setText(line)

    # ── keys ───────────────────────────────────────────────────────────────

    def _bind_keys(self) -> None:
        for key in _CONTROLS:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])
        presses = [
            ("y", self._level_roll, []),
            ("page_up", self._scale_speed, [2.0]),
            ("page_down", self._scale_speed, [0.5]),
            ("home", self._jump_to_orbit, []),
            ("tab", self._jump_to_next_body, []),
            ("end", self.reset_camera, []),
            ("r", self.reset_camera, []),
            ("f", self._print_pose, []),
            ("escape", self._quit, []),
            (QUALITY_ALL_KEY, self._cycle_all_quality, []),
        ]
        presses += [(k, self._cycle_quality, [n]) for k, n in QUALITY_KEYS]
        for key, method, args in presses:
            self.accept(key, method, extraArgs=args)
        if isinstance(self.win, GraphicsWindow):
            self.win.set_close_request_event("window-close")
            self.accept("window-close", self._quit)

    def _set_key(self, key: str, pressed: bool) -> None:
        self._keys[key] = pressed

    def _scale_speed(self, factor: float) -> None:
        self.speed = max(10.0, min(SPEED * 64, self.speed * factor))

    def _level_roll(self) -> None:
        """Keep the view direction, bring the camera's up into the plane of
        it and the local vertical (no roll)."""
        forward = self._forward()
        right = np.cross(forward, self._up())
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            return  # looking straight up or down: any roll is level
        right /= norm
        self._orientation = tools.quat_from_axes(
            right, forward, np.cross(right, forward)
        )

    def _jump_over(self, body, direction: np.ndarray) -> None:
        """3 radii above a body, along *direction* from its centre, looking
        down."""
        lat, lon = latlon(direction[None, :] / np.linalg.norm(direction))
        self.place_camera(
            float(lat[0]), float(lon[0]), ORBIT_RADII * body.radius, 0.0, -89.9, body
        )

    def _jump_to_orbit(self) -> None:
        if self.body is not None:
            self._jump_over(self.body, self._relative())

    def _jump_to_next_body(self) -> None:
        """To the next body, over its side facing the camera."""
        if not self.bodies:
            return
        index = self.bodies.index(self.body)
        target = self.bodies[(index + 1) % len(self.bodies)]
        self._jump_over(target, self.cam_world - target.position)

    def _print_pose(self) -> None:
        heading, pitch = self._true_heading_pitch()
        if self.body is None:
            p = self.cam_world
            print(
                f"--position {p[0]:.1f} {p[1]:.1f} {p[2]:.1f} "
                f"--heading {heading:.1f} --pitch {pitch:.1f}"
            )
            return
        relative = self._relative()
        lat, lon = latlon(self._up()[None, :])
        altitude = float(np.linalg.norm(relative)) - self.body.radius
        print(
            f"--body {self.body.name} --lat {lat[0]:.5f} --lon {lon[0]:.5f} "
            f"--alt {altitude:.0f} --heading {heading:.1f} --pitch {pitch:.1f}"
        )
        for body in self.bodies:
            print(f"{body.name} terrain: {body.terrain.stats}")

    def _cycle_quality(self, section: str) -> None:
        """One part (a WorldSettings section) to its next preset."""
        level = _next_level(self.settings.section_level(section))
        self._set_settings(self.settings.with_section_level(section, level))
        print(f"{SECTIONS[section].LABEL} quality: {level.value}")

    def _cycle_all_quality(self) -> None:
        level = _next_level(self.settings.level)
        self._set_settings(self.settings.with_level(level))
        print(f"quality: {level.value}")

    def _set_settings(self, settings: WorldSettings) -> None:
        self.settings = settings
        self.world.world.set_settings(settings)

    def destroy_world(self) -> None:
        """Remove the world and stop its workers."""
        self.world.destroy()

    def _quit(self) -> None:
        self.destroy_world()
        sys.exit()


def _heading_pitch(forward: np.ndarray) -> tuple[float, float]:
    """:returns: heading (degrees counter-clockwise from +Y) and pitch of a
    direction in the world frame, as Panda's H and P"""
    heading = math.degrees(math.atan2(-forward[0], forward[1]))
    pitch = math.degrees(
        math.asin(np.clip(forward[2] / np.linalg.norm(forward), -1, 1))
    )
    return heading, pitch


def _look_along(direction: np.ndarray) -> dict:
    heading, pitch = _heading_pitch(np.asarray(direction, dtype=np.float64))
    return {"heading": heading, "pitch": pitch}


def start_from_args(args: argparse.Namespace) -> dict | None:
    """:returns: the start the camera arguments ask for, or None"""
    if args.position is not None:
        return {
            "position": tuple(args.position),
            "heading": args.heading or 0.0,
            "pitch": args.pitch or 0.0,
        }
    if args.lat is not None or args.lon is not None or args.body is not None:
        return {
            "body": args.body,
            "lat": args.lat or 0.0,
            "lon": args.lon or 0.0,
            "alt": args.alt if args.alt is not None else 800.0,
            "heading": args.heading or 0.0,
            "pitch": args.pitch or 0.0,
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fly through a world file")
    parser.add_argument("world", nargs="?", default=str(DEFAULT_WORLD))
    parser.add_argument(
        "--backend", default="process", choices=("process", "thread", "inline")
    )
    parser.add_argument("--size", default="1600 900", help='"width height"')
    parser.add_argument("--body", default=None, help="start over this body")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--alt", type=float, default=None, help="metres above sea")
    parser.add_argument(
        "--position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z")
    )
    parser.add_argument("--heading", type=float, default=None)
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="quit after N frames and what is still loading",
    )
    parser.add_argument("--shot", default=None, help="save a picture at the end")
    tools.add_quality_arguments(parser)
    args = parser.parse_args()
    tools.window_config(
        "WorldBuilder: " + Path(args.world).stem, size=args.size, vsync=False
    )
    demo = WorldDemo(
        args.world,
        settings=tools.settings_from_args(args),
        backend=args.backend,
        start=start_from_args(args),
    )
    print(tools.describe_window(demo))
    if args.frames is None:
        demo.run()
        return
    try:
        for _ in range(args.frames):
            demo.taskMgr.step()
        # What streams in (ground, rocks, cloud tiles) gets time to arrive.
        deadline, quiet = time.time() + 180.0, 0
        while quiet < 30 and time.time() < deadline:
            demo.taskMgr.step()
            time.sleep(0.02)
            quiet = 0 if demo.world.world.loading else quiet + 1
        if args.shot:
            demo.win.save_screenshot(Filename.from_os_specific(args.shot))
    finally:
        demo.destroy_world()


if __name__ == "__main__":
    main()
