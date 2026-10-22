"""
Watch terrain collision live (issue #16): the load of collision_bench.py over
mountains, in a window, through Terrain.enable_collision and one
CollisionTraverser per frame::

    python scripts/collision_viewer.py [--stress] [--seed S] [--screenshot PATH]

Needs the native library (scripts/build_native.sh).

- Ships: 64 spheres (5-20 m), cyan, red while touching the ground.
- Sensors: 3 look-ahead spheres per ship (30/50/100 m), each ship's active on
  one frame in six as on a bot's think frames; yellow, orange on contact.
- Lasers: 300 segments flying at 2000 m/s (one frame of travel long), green;
  a hit leaves a red mark with the surface normal in white, and the laser
  respawns.
- Readout: traverse time (all colliders, terrain included), contacts.

Keys: left drag orbit, wheel zoom, W/A/S/D move the view, Q/E down/up,
X normal/stress load, R reshuffle, T terrain collision on/off (compare the
traverse time), Space pause the lasers, Escape quit.

--screenshot PATH renders offscreen, saves one frame and exits.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionSegment,
    CollisionSphere,
    CollisionTraverser,
    Filename,
    LineSegs,
    NodePath,
    TextNode,
    loadPrcFileData,
)

sys.path.insert(0, str(Path(__file__).parent))

from collision_bench import (  # noqa: E402
    LASER_LENGTH,
    LASERS,
    SENSOR_RADII,
    SHIP_MASK,
    SHIPS,
    mountain_camera,
    scatter,
    tangent_frame,
)

from world_builder.terrain.planet import PRESETS  # noqa: E402
from world_builder.terrain.terrain import Terrain  # noqa: E402

LASER_SPEED = 2000.0
LASER_LIFE = 2.0
SHIP = (0.2, 0.9, 1.0, 1.0)
SENSOR = (0.9, 0.9, 0.2, 0.35)
LASER = (0.3, 1.0, 0.3, 1.0)
HIT = (1.0, 0.15, 0.1, 1.0)
SENSOR_HIT = (1.0, 0.55, 0.1, 0.8)


def circle_sphere() -> NodePath:
    """A unit wire sphere: three great circles."""
    lines = LineSegs("sphere")
    lines.set_thickness(1.5)
    steps = 32
    for axes in ((0, 1), (1, 2), (0, 2)):
        for k in range(steps + 1):
            a = 2.0 * math.pi * k / steps
            p = [0.0, 0.0, 0.0]
            p[axes[0]], p[axes[1]] = math.cos(a), math.sin(a)
            (lines.draw_to if k else lines.move_to)(*p)
    return NodePath(lines.create())


def unit_line() -> NodePath:
    """A line from the origin to (0, 1, 0)."""
    lines = LineSegs("line")
    lines.set_thickness(2.0)
    lines.move_to(0, 0, 0)
    lines.draw_to(0, 1, 0)
    return NodePath(lines.create())


class Collider:
    """A collider's holder (placed, coloured), CollisionNode path and visual."""

    def __init__(self, holder: NodePath, collision: NodePath, visual: NodePath):
        self.holder = holder
        self.collision = collision
        self.visual = visual
        self.age = 0.0

    def paint(self, color) -> None:
        # Priority 1: the LineSegs under the visual carry their own
        # vertex-colour state, which would otherwise win.
        self.holder.set_color(*color, 1)


class Viewer:
    def __init__(self, base, args):
        self.base = base
        self.render = base.render
        self.rng = np.random.default_rng(args.seed)
        self.stress = args.stress
        self.paused = False
        self.frame = 0
        self.times: list[float] = []

        # Window set up first: the terrain takes seconds to load, shown live.
        lens = base.cam.node().get_lens()
        lens.set_near_far(1.0, 300_000.0)
        lens.set_fov(60.0)
        base.disableMouse()
        base.set_background_color(0.55, 0.7, 0.9, 1.0)
        self.text = base.aspect2d.attach_new_node(TextNode("readout"))
        self.text.node().set_text_color(1, 1, 1, 1)
        self.text.node().set_shadow(0.05, 0.05)
        self.text.set_scale(0.045)
        self.text.set_pos(-1.3 * base.getAspectRatio() / 1.333, 0, 0.9)

        self.terrain = Terrain(self.render, PRESETS["earthlike"])
        self.anchor = mountain_camera(self.terrain, self.rng)
        self.up = self.anchor / np.linalg.norm(self.anchor)
        self.east, self.north = tangent_frame(self.up)
        self._settle()
        self.collision = self.terrain.enable_collision(SHIP_MASK)

        self.sphere = circle_sphere()
        self.line = unit_line()
        self.marks = self.render.attach_new_node("contact-marks")
        self.colliders = None
        self.traverser = CollisionTraverser()
        self.queue = CollisionHandlerQueue()
        self._build()

        ground = self.terrain.query.surface_point(self.anchor, generate=False)
        self.target = np.array(self.terrain.to_render(ground))
        self.heading, self.pitch, self.distance = 30.0, 40.0, args.distance
        self.last_mouse = None
        self._keys()
        base.taskMgr.add(self._update, "collision-viewer")

    # ── scene ──────────────────────────────────────────────────────────────────

    def _settle(self) -> None:
        """Let the ground around the start load before placing anything:
        until the terrain has not been loading for half a second."""
        clock = self.base.clock
        quiet_since = None
        start = clock.get_real_time()
        deadline = start + 120.0
        while clock.get_real_time() < deadline:
            self.terrain.update(self.anchor)
            stats = self.terrain.stats
            self.text.node().set_text(
                f"loading terrain... {clock.get_real_time() - start:.0f} s"
                f"   {stats['shown']} chunks drawn, {stats['wanted']} wanted"
            )
            self.base.graphicsEngine.render_frame()
            now = clock.get_real_time()
            if self.terrain.loading:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since > 0.5:
                break

    def _collider(self, name, solid, shape: NodePath, scale) -> Collider:
        """A collision node and its visual under one holder. The visual is not
        under the CollisionNode: those are hidden, with everything below them
        (show() would also draw Panda's preview of the solid)."""
        holder = self.colliders.attach_new_node(f"{name}-holder")
        node = CollisionNode(name)
        node.add_solid(solid)
        node.set_from_collide_mask(SHIP_MASK)
        node.set_into_collide_mask(BitMask32.all_off())
        collision = holder.attach_new_node(node)
        # Its own node, so scaling or hiding it leaves the shared shape alone.
        visual = holder.attach_new_node("visual")
        visual.set_scale(scale)
        shape.instance_to(visual)
        self.traverser.add_collider(collision, self.queue)
        collider = Collider(holder, collision, visual)
        self.by_path[collision] = collider
        return collider

    def _build(self) -> None:
        if self.colliders is not None:
            self.colliders.remove_node()
        self.traverser.clear_colliders()
        self.colliders = self.render.attach_new_node("colliders")
        self.by_path = {}
        terrain = self.terrain
        low = SHIPS if self.stress else SHIPS // 2
        near, ups = scatter(terrain, self.anchor, self.rng, low, -5.0, 50.0)
        far, far_ups = scatter(terrain, self.anchor, self.rng, SHIPS - low, 50.0, 500.0)
        self.ships, self.sensors = [], []
        for centre, up in zip(
            np.concatenate([near, far]), np.concatenate([ups, far_ups])
        ):
            radius = float(self.rng.uniform(5.0, 20.0))
            ship = self._collider(
                "ship", CollisionSphere(0, 0, 0, radius), self.sphere, radius
            )
            ship.holder.set_pos(*terrain.to_render(centre))
            ship.paint(SHIP)
            self.ships.append(ship)
            east, north = tangent_frame(up)
            angle = self.rng.uniform(0, 2 * np.pi)
            forward = np.cos(angle) * east + np.sin(angle) * north
            group = []
            for ahead, sensor_radius in SENSOR_RADII:
                sensor = self._collider(
                    "sensor",
                    CollisionSphere(0, 0, 0, sensor_radius),
                    self.sphere,
                    sensor_radius,
                )
                sensor.holder.set_pos(*terrain.to_render(centre + ahead * forward))
                sensor.holder.set_transparency(True)
                sensor.paint(SENSOR)
                group.append(sensor)
            self.sensors.append(group)
        self.lasers = []
        for _ in range(LASERS):
            laser = self._collider(
                "laser", CollisionSegment(0, 0, 0, 0, LASER_LENGTH, 0), self.line, 1.0
            )
            self.lasers.append(laser)
            self._respawn(laser)

    def _respawn(self, laser: Collider) -> None:
        top = 60.0 if self.stress else 500.0
        start, up = scatter(self.terrain, self.anchor, self.rng, 1, 0.0, top)
        direction = self.rng.normal(size=3)
        direction -= 0.6 * abs(direction @ up[0]) * up[0]  # mostly level or down
        direction /= np.linalg.norm(direction)
        holder = laser.holder
        holder.set_pos(*self.terrain.to_render(start[0]))
        holder.look_at(*(np.array(holder.get_pos()) + direction))
        laser.paint(LASER)
        laser.age = 0.0

    # ── frame ──────────────────────────────────────────────────────────────────

    def _update(self, task):
        dt = min(self.base.clock.get_dt(), 0.1)
        self._camera(dt)
        self.frame += 1
        # Think frames: each ship's sensors join one traverse in six.
        for index, group in enumerate(self.sensors):
            active = (index + self.frame) % 6 == 0
            for sensor in group:
                mask = SHIP_MASK if active else BitMask32.all_off()
                sensor.collision.node().set_from_collide_mask(mask)
                sensor.paint(SENSOR)
                (sensor.visual.show if active else sensor.visual.hide)()
        step = 0.0 if self.paused else LASER_SPEED * dt
        length = max(LASER_LENGTH, step)
        for laser in self.lasers:
            if step:
                laser.holder.set_pos(laser.holder, 0, step, 0)
                laser.age += dt
            laser.collision.node().modify_solid(0).set_point_b(0, length, 0)
            laser.visual.set_scale(1, length, 1)
        for ship in self.ships:
            ship.paint(SHIP)

        self.queue.clear_entries()
        start = self.base.clock.get_real_time()
        self.traverser.traverse(self.render)
        self.times.append((self.base.clock.get_real_time() - start) * 1e3)
        self.times = self.times[-60:]
        self._show_contacts()
        return task.cont

    def _show_contacts(self) -> None:
        self.marks.node().remove_all_children()
        lines = LineSegs("marks")
        lines.set_thickness(2.0)
        counts = {"ship": 0, "sensor": 0, "laser": 0}
        hit_lasers = set()
        for entry in self.queue.entries:
            if entry.into_node.name != "terrain":
                continue
            kind = entry.from_node.name
            counts[kind] += 1
            point = entry.get_surface_point(self.render)
            normal = entry.get_surface_normal(self.render)
            collider = self.by_path[entry.from_node_path]
            color = SENSOR_HIT if kind == "sensor" else HIT
            collider.paint(color)
            lines.set_color(*color)
            if kind == "laser":
                hit_lasers.add(id(collider))
            lines.move_to(point)
            lines.draw_to(point + normal * 0.5)
            lines.set_color(1, 1, 1, 1)
            lines.move_to(point)
            lines.draw_to(point + normal * 25.0)
        self.marks.attach_new_node(lines.create())
        for laser in self.lasers:
            if id(laser) in hit_lasers or laser.age > LASER_LIFE:
                self._respawn(laser)
        active = sum(
            not s.collision.node().get_from_collide_mask().is_zero()
            for g in self.sensors
            for s in g
        )
        collision = "on" if self.terrain.collision is not None else "OFF"
        self.text.node().set_text(
            f"{'STRESS' if self.stress else 'normal'} load"
            f"   terrain collision {collision}"
            f"   {len(self.collision.registered)} chunks\n"
            f"traverse {statistics.median(self.times):.2f} ms"
            " (all colliders, median of 60 frames)\n"
            f"terrain contacts: ships {counts['ship']}/{len(self.ships)}"
            f"   sensors {counts['sensor']}/{active} active"
            f"   lasers {counts['laser']}/{len(self.lasers)}\n"
            "drag orbit, wheel zoom, WASD/QE move, X stress, R reshuffle,"
            " T collision, Space pause"
        )

    # ── camera and keys ────────────────────────────────────────────────────────

    def _camera(self, dt: float) -> None:
        watcher = self.base.mouseWatcherNode  # None offscreen
        if (
            watcher is not None
            and watcher.has_mouse()
            and watcher.is_button_down("mouse1")
        ):
            mouse = (watcher.get_mouse_x(), watcher.get_mouse_y())
            if self.last_mouse is not None:
                self.heading -= (mouse[0] - self.last_mouse[0]) * 120.0
                self.pitch = min(
                    89.0,
                    max(-10.0, self.pitch - (mouse[1] - self.last_mouse[1]) * 80.0),
                )
            self.last_mouse = mouse
        else:
            self.last_mouse = None
        speed = self.distance * 0.6 * dt
        h = math.radians(self.heading)
        forward = math.cos(h) * self.north + math.sin(h) * self.east
        right = -math.sin(h) * self.north + math.cos(h) * self.east
        for key, move in (
            ("w", forward),
            ("s", -forward),
            ("d", right),
            ("a", -right),
            ("e", self.up),
            ("q", -self.up),
        ):
            if watcher is not None and watcher.is_button_down(key):
                self.target = self.target + move * speed
        p = math.radians(self.pitch)
        offset = self.distance * (math.cos(p) * -forward + math.sin(p) * self.up)
        position = self.target + offset
        self.base.camera.set_pos(*position)
        self.base.camera.look_at(tuple(self.target), tuple(self.up))
        # The terrain's LOD follows the camera; its frame stays put.
        self.terrain.update(self.anchor + position, camera_render=position)

    def _keys(self) -> None:
        base = self.base
        base.accept("escape", sys.exit)
        base.accept("wheel_up", self._zoom, [0.85])
        base.accept("wheel_down", self._zoom, [1.0 / 0.85])
        base.accept("x", self._toggle_stress)
        base.accept("r", self._build)
        base.accept("t", self._toggle_collision)
        base.accept("space", self._pause)

    def _zoom(self, factor: float) -> None:
        self.distance = min(60_000.0, max(30.0, self.distance * factor))

    def _toggle_stress(self) -> None:
        self.stress = not self.stress
        self._build()

    def _toggle_collision(self) -> None:
        if self.terrain.collision is None:
            self.collision = self.terrain.enable_collision(SHIP_MASK)
        else:
            self.terrain.disable_collision()
        self.times = []

    def _pause(self) -> None:
        self.paused = not self.paused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--screenshot", help="render offscreen, save a frame, exit")
    parser.add_argument(
        "--distance", type=float, default=4000.0, help="start view distance, m"
    )
    args = parser.parse_args()
    loadPrcFileData("", "win-size 1600 900")
    loadPrcFileData("", "window-title WorldBuilder terrain collision")
    loadPrcFileData("", "sync-video false")
    if args.screenshot:
        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "audio-library-name null")
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    viewer = Viewer(base, args)
    if args.screenshot:
        for _ in range(30):
            base.taskMgr.step()
        base.win.save_screenshot(Filename.from_os_specific(args.screenshot))
        print(viewer.text.node().get_text())
        return
    base.run()


if __name__ == "__main__":
    main()
