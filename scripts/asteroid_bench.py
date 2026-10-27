"""
Frame-time benchmark of asteroid fields, flown along fixed paths (issue #14).

``--field PRESET`` flies a field of ``world_builder.asteroids`` (a
``spec.PRESETS`` name: ``cube`` is the same 15 km cube) drawn by instancing,
lit by the same sun, its rocks turning with the flight's time. ``--legacy
PATH`` instead builds SpaceFlight's field as its missions 1 and 2 have it
(``space_flight.scenes.asteroid_field``, scene "asteroids"): PATH is the
rock's glTF (SpaceFlight's ``models/asteroids/54509_asteroid/scene.gltf``),
instanced on one node per rock with a collision sphere each, lit by Panda's
fixed-function lights as the game does::

    python scripts/asteroid_bench.py (--field PRESET | --legacy PATH)
        [--flight cruise sweep outside] [--quality LEVEL]
        [--seconds S] [--size "1600 900"] [--offscreen] [--fov DEG] [--profile]
        [--shots DIR] [--collision] [--terrain-rocks N] [--setting NAME=VALUE ...]

    --field     the field preset drawn
    --legacy    the rock model of SpaceFlight's field
    --quality   the field's AsteroidSettings preset (default high)
    --terrain-rocks  override the preset's max_terrain_rocks (0: no rock
                drawn as terrain)
    --setting   override any AsteroidSettings knob (repeatable), e.g.
                --setting instance_budget=40000
    --flight    the flights to run, in order (default: all three)
    --seconds   each flight's length (default 10)
    --size      window or buffer size (default "1600 900")
    --offscreen render to an offscreen buffer: no display refresh cap
    --fov       the lens's horizontal field of view (default: Panda's, as
                SpaceFlight leaves it)
    --profile   cProfile the frames; prints the top 20 by cumulative time
    --shots     save each flight's first frame to DIR/<flight>.png
    --collision instead of flying, time one CollisionTraverser.traverse with
                SpaceFlight's load within 5 km of the camera: 64 ship
                spheres (5-20 m), 3 sensor spheres each (30/50/100 m, a
                sixth active), 300 laser segments (35 m); median of 30

The layout is the scene's: three fields in the same 15 km cube with the same
seeds and draws (2,000 still rocks, 20 big and 500 small rotating ones). The
rotating ones are turned every frame from a closed form (numpy, then one
``setQuat`` each): the game's own Python integration of them costs far more,
but that is a CPU cost its headless profile already shows
(``SpaceFlight/scripts/profile_dev_level.py``); this measures drawing.

Flights:
    cruise   along the field's middle at 250 m/s, 500 m up, heading +Y
    sweep    still at the field's centre, turning 24 degrees a second
    outside  20 km out, the whole field in view, turning slowly
    approach (--field only) at the field's biggest rock, from 6 radii to 1.3
             from its centre over the flight: it turns into terrain on the way

A field too big for these flights (``--field ring``) is placed so that a
point of it (:data:`ORIGINS`: in the ring's dense band) is at the world's
origin: the flights happen there. After each flight of a field, a line of
its counts: rocks held, rocks in range against the instance budget, the
smallest size drawn, the field's update time.

Each frame waits for the GPU (a one-pixel read back), so a frame's time is
the CPU's and the GPU's work in series, as in ``terrain_shots.py``. Per
flight: the frame count and frame time (mean, median, p95, max).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import math
import pstats
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from panda3d.core import (
    AmbientLight,
    BitMask32,
    CollisionNode,
    CollisionSphere,
    DirectionalLight,
    Filename,
    NodePath,
    Quat,
)

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import describe_window, window_config  # noqa: E402

#: SpaceFlight's scene "asteroids" (scenes.py SceneAsteroids.build_upfront):
#: (asteroids, field size m, scale factor, rotating, seed).
LEGACY_FIELDS = (
    (2000, 15000.0, 1.0, False, 0),
    (20, 15000.0, 10.0, True, 1),
    (500, 15000.0, 1.0, True, 2),
)
#: The legacy rock's collision sphere, in its own (scaled) frame.
LEGACY_HIT_RADIUS = 1.6
#: SpaceFlight's far plane (scenes.py CAMERA_FAR_M).
CAMERA_FAR_M = 600_000.0

CRUISE_SPEED = 250.0
SWEEP_RATE = 24.0
OUTSIDE_DISTANCE = 20_000.0
OUTSIDE_RATE = 3.0
FLIGHTS = ("cruise", "sweep", "outside", "approach")
APPROACH_RADII = (6.0, 1.3)


class LegacyField:
    """SpaceFlight's asteroid field, rebuilt without the game: the same draws
    from the same seeds, one node per rock instancing the model."""

    def __init__(self, parent: NodePath, model: NodePath):
        self.root = parent.attach_new_node("legacy_asteroids")
        self.spinning: list[NodePath] = []
        q0, omegas = [], []
        for n, size, scale_factor, rotating, seed in LEGACY_FIELDS:
            rng = np.random.default_rng(seed)
            for _ in range(n):
                rng.choice([0])  # the game picks among its model paths
                instance = self.root.attach_new_node("asteroid_instance")
                model.instance_to(instance)
                instance.set_pos(*(rng.random(3) * size - 0.5 * size))
                scale = (rng.random() * 100 + 1) * scale_factor
                instance.set_scale(scale)
                temp = rng.random(4)
                quat = temp / np.linalg.norm(temp) + 0.2
                instance.set_quat(Quat(*quat))
                if rotating:
                    self.spinning.append(instance)
                    q0.append(quat / np.linalg.norm(quat))
                    omegas.append(5000 * np.deg2rad(rng.random(3) - 0.5) / scale**1.5)
                node = CollisionNode("terrain")
                node.add_solid(CollisionSphere(0, 0, 0, LEGACY_HIT_RADIUS))
                node.set_from_collide_mask(BitMask32.all_off())
                node.set_into_collide_mask(BitMask32.bit(2))
                instance.attach_new_node(node)
        self._q0 = np.array(q0)
        self._omega = np.array(omegas)

    @property
    def count(self) -> int:
        return self.root.get_num_children()

    def update(self, t: float) -> None:
        """Turn the rotating rocks to time *t*: q(t) = exp(ω t / 2) q0, the
        game's constant world-axis rotation."""
        rate = np.linalg.norm(self._omega, axis=1)
        half = 0.5 * rate * t
        axis = self._omega / np.maximum(rate, 1e-12)[:, None]
        s = np.sin(half)
        w1, x1, y1, z1 = np.cos(half), axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s
        w2, x2, y2, z2 = self._q0.T
        q = np.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            axis=1,
        )
        for node, row in zip(self.spinning, q.tolist()):
            node.set_quat(Quat(*row))


#: Where a preset's flights happen, field frame (the field is placed so that
#: this point is at the world's origin); the field's centre otherwise.
ORIGINS = {"ring": (0.0, 1.01e8, 0.0)}


class InstancedField:
    """A ``world_builder.asteroids`` field, flown like the legacy one."""

    def __init__(
        self, base, preset: str, quality: str, terrain_rocks=None, overrides=()
    ):
        from world_builder.asteroids.field import AsteroidField
        from world_builder.asteroids.settings import AsteroidSettings
        from world_builder.asteroids.spec import PRESETS

        self.base = base
        self.t = 0.0
        settings = AsteroidSettings.preset(quality)
        if terrain_rocks is not None:
            settings = replace(settings, max_terrain_rocks=terrain_rocks)
        for item in overrides:
            name, value = item.split("=", 1)
            kind = type(getattr(settings, name))
            settings = replace(settings, **{name: kind(float(value))})
        self.field = AsteroidField(
            base,
            PRESETS[preset],
            clock=lambda: self.t,
            settings=settings,
            sun_direction=-np.array(_SUN_TOWARD),
            far=CAMERA_FAR_M,
            position=-np.array(ORIGINS.get(preset, (0.0, 0.0, 0.0))),
        )
        self.updates: list[float] = []
        self.in_range: list[int] = []

    @property
    def count(self) -> int:
        return int(self.field.spec.summary()["expected_count"])

    def wait(self, timeout: float = 60.0) -> float:
        """Step until the prototypes are built. :returns: seconds taken"""
        start = time.perf_counter()
        while self.field.loading and time.perf_counter() - start < timeout:
            self.update(0.0)
            self.base.taskMgr.step()
        return time.perf_counter() - start

    def update(self, t: float) -> None:
        self.t = t
        pos = self.base.camera.get_pos(self.base.render)
        self.field.update(np.array(pos), np.array(pos))
        self.updates.append(self.field.stats.get("update_ms", 0.0))
        self.in_range.append(self.field.stats.get("in_range", 0))

    def report(self) -> str:
        """:returns: the counts since the last report, and resets them"""
        s, ms = self.field.stats, np.array(self.updates or [0.0])
        in_range = max(self.in_range or [0])
        line = (
            f"{'':8} rocks {s.get('rocks', 0)}  in range max {in_range}"
            f" (budget {self.field.settings.instance_budget})  smallest "
            f"{s.get('min_pixels', 0):.1f} px  update ms median {np.median(ms):.2f}"
            f" p95 {np.percentile(ms, 95):.2f}"
        )
        self.updates, self.in_range = [], []
        return line

    def biggest_rock(self) -> tuple[np.ndarray, float]:
        """:returns: the centre (world) and radius of the biggest rock held
        (the field is not turned here)"""
        pop = self.field.population
        live = np.flatnonzero(pop.live)
        rock = pop.rocks[live[np.argmax(pop.rocks["radius"][live])]]
        return rock["centre"] + self.field.position, float(rock["radius"])

    def destroy(self) -> None:
        self.field.destroy()


#: Toward where SpaceFlight's asteroid-scene light points (scenes/lighting.py).
_SUN_TOWARD = (30.0 / 67.08, 60.0 / 67.08, 0.0)


def _collision_load(render: NodePath, centre: np.ndarray, field_kind: str):
    """SpaceFlight's colliders around *centre* (scripts/collision_bench.py's
    load). :returns: their NodePaths"""
    from panda3d.core import CollisionNode, CollisionSegment, CollisionSphere

    rng = np.random.default_rng(3)

    def somewhere() -> np.ndarray:
        d = rng.normal(size=3)
        return centre + d / np.linalg.norm(d) * 5000.0 * rng.random() ** (1 / 3)

    def collider(solid) -> NodePath:
        node = CollisionNode("probe")
        node.add_solid(solid)
        node.set_from_collide_mask(BitMask32.bit(2))
        node.set_into_collide_mask(BitMask32.all_off())
        return render.attach_new_node(node)

    paths = []
    for ship in range(64):
        p = somewhere()
        paths.append(collider(CollisionSphere(*p, rng.uniform(5.0, 20.0))))
        if ship % 6 == 0:  # a sixth of the sensors active
            for ahead, r in ((5.0, 30.0), (50.0, 50.0), (125.0, 100.0)):
                paths.append(collider(CollisionSphere(*(p + [0.0, ahead, 0.0]), r)))
    for _ in range(300):
        p = somewhere()
        d = rng.normal(size=3)
        q = p + d / np.linalg.norm(d) * 35.0
        paths.append(collider(CollisionSegment(*p, *q)))
    return paths


def _time_collision(base, field) -> None:
    """One traverse with SpaceFlight's load: median of 30."""
    from panda3d.core import CollisionHandlerQueue, CollisionTraverser

    pos = (0.0, -2000.0, 500.0)
    base.camera.set_pos(*pos)
    field.update(0.0)
    paths = _collision_load(base.render, np.array(pos), type(field).__name__)
    traverser = CollisionTraverser()
    queue = CollisionHandlerQueue()
    for path in paths:
        traverser.add_collider(path, queue)
    times = []
    for _ in range(30):
        start = time.perf_counter()
        traverser.traverse(base.render)
        times.append(time.perf_counter() - start)
    print(
        f"traverse   {len(paths)} colliders  median {np.median(times) * 1e3:6.2f} ms"
        f"  ({queue.get_num_entries()} contacts)"
    )
    for path in paths:
        path.remove_node()


def _light(render: NodePath) -> None:
    """SpaceFlight's asteroid scene lighting (scenes/lighting.py)."""
    sun = DirectionalLight("sun")
    sun.set_color((0.5, 0.5, 0.45, 1))
    sun_np = render.attach_new_node(sun)
    sun_np.look_at(30, 60, 0)
    ambient = AmbientLight("ambient")
    ambient.set_color((0.2, 0.4, 0.6, 1))
    render.set_light(sun_np)
    render.set_light(render.attach_new_node(ambient))


def _place(camera, field, flight: str, t: float, seconds: float) -> None:
    """Put the camera where the flight is at *t* seconds."""
    if flight == "approach":
        centre, radius = field.biggest_rock()
        near, far = APPROACH_RADII[1], APPROACH_RADII[0]
        d = radius * far * (near / far) ** min(t / seconds, 1.0)
        side = np.array([0.3, -1.0, 0.25]) / np.linalg.norm([0.3, -1.0, 0.25])
        camera.set_pos(*(centre + side * d))
        camera.look_at(*centre)
        return
    pos, h, p = _pose(flight, t)
    camera.set_pos_hpr(*pos, h, p, 0)


def _pose(flight: str, t: float) -> tuple[tuple[float, float, float], float, float]:
    """:returns: the camera's (position, heading, pitch) at *t* seconds"""
    if flight == "cruise":
        return (0.0, -7400.0 + CRUISE_SPEED * t, 500.0), 0.0, 0.0
    if flight == "sweep":
        return (0.0, 0.0, 0.0), SWEEP_RATE * t, 0.0
    heading = OUTSIDE_RATE * t
    a = math.radians(heading)
    pos = (OUTSIDE_DISTANCE * math.sin(a), -OUTSIDE_DISTANCE * math.cos(a), 0.0)
    return pos, heading, 0.0


def _fly(
    base,
    field: LegacyField | InstancedField,
    probe,
    flight: str,
    seconds: float,
    profile: bool,
    shots: Path | None,
):
    camera = base.camera
    frames = []
    profiler = cProfile.Profile() if profile else None
    t = 0.0
    dt = 1.0 / 60.0  # the flight's own time: the same path whatever the frame rate
    for warm in range(30):
        _place(camera, field, flight, 0.0, seconds)
        field.update(0.0)
        base.taskMgr.step()
        probe.get_screenshot()
    if isinstance(field, InstancedField):
        field.report()  # drop the warm-up's counts
    if shots is not None:
        path = shots / f"{flight}.png"
        base.win.save_screenshot(Filename.from_os_specific(str(path)))
    if profiler:
        profiler.enable()
    while t < seconds:
        start = time.perf_counter()
        _place(camera, field, flight, t, seconds)
        field.update(t)
        base.taskMgr.step()
        probe.get_screenshot()  # waits for the GPU
        frames.append(time.perf_counter() - start)
        t += dt
    if profiler:
        profiler.disable()
    f = np.array(frames) * 1e3
    print(
        f"{flight:<8} {len(f):5d} frames  frame ms mean {np.mean(f):6.2f} "
        f"median {np.median(f):6.2f} p95 {np.percentile(f, 95):6.2f} "
        f"max {f.max():6.2f}"
    )
    if isinstance(field, InstancedField):
        print(field.report())
    if profiler:
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(20)
        print(out.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--field", default=None)
    kind.add_argument("--legacy", type=Path, default=None)
    parser.add_argument("--quality", default="high")
    parser.add_argument("--flight", nargs="+", choices=FLIGHTS, default=list(FLIGHTS))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--size", default="1600 900")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--shots", type=Path, default=None)
    parser.add_argument("--collision", action="store_true")
    parser.add_argument("--terrain-rocks", type=int, default=None)
    parser.add_argument("--setting", action="append", default=[])
    args = parser.parse_args()

    window_config("asteroid bench", args.size, args.offscreen, vsync=False)
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    base.disable_mouse()
    base.setFrameRateMeter(False)
    base.set_background_color(0.02, 0.01, 0.04, 1)
    lens = base.cam.node().get_lens()
    lens.set_near_far(1.0, CAMERA_FAR_M)
    if args.fov is not None:
        lens.set_fov(args.fov)
    win = base.win
    probe = win.make_display_region(0, 1 / win.get_x_size(), 0, 1 / win.get_y_size())
    if args.legacy is not None:
        _light(base.render)
        path = Filename.from_os_specific(str(args.legacy.resolve()))
        field = LegacyField(base.render, base.loader.load_model(path))
        print(describe_window(base), f"legacy field: {field.count} rocks")
    else:
        field = InstancedField(
            base, args.field, args.quality, args.terrain_rocks, args.setting
        )
    try:
        if args.legacy is None:
            waited = field.wait()
            print(
                describe_window(base),
                f"field {args.field}: ~{field.count} rocks,",
                f"prototypes in {waited:.1f} s",
            )
        if args.collision:
            if args.legacy is None:
                field.field.enable_collision(BitMask32.bit(2), max_distance=20_000.0)
            _time_collision(base, field)
            return
        if args.shots is not None:
            args.shots.mkdir(parents=True, exist_ok=True)
        if args.legacy is not None and "approach" in args.flight:
            args.flight.remove("approach")  # the instanced field's flight
        for flight in args.flight:
            _fly(base, field, probe, flight, args.seconds, args.profile, args.shots)
            stats = getattr(getattr(field, "field", None), "stats", None)
            if stats:
                print("         ", {k: round(v, 2) for k, v in stats.items()})
    finally:
        if args.legacy is None:
            field.destroy()  # its worker processes
        base.destroy()


if __name__ == "__main__":
    main()
