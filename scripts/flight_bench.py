"""
Frame-time benchmark of the terrain demo in fast, scripted flights.

Runs the real demo (window, workers, both cameras, atmosphere, clouds,
ocean, sky) with no multisampling and vsync off, and flies fixed paths,
reporting frame times and where the main thread spends them::

    python scripts/flight_bench.py [--demo terrain|world]
        [--flight low cruise descent] [--seconds S]
        [--profile] [--trace] [--no-hud] [--size "1600 900"] [--workers N]
        [--offscreen] [--quality low|mid|high|ultra] [--<part>-quality LEVEL]
        [--no-clouds] [--cloud-debug BITS]

    --demo           which demo flies: demo_terrain.py's hand-built world
                     (default) or demo_world.py's, built by World
    --flight         the flights to run, in order (default: all three)
    --seconds        each flight's length, at most (default 15)
    --profile        cProfile the main thread; prints the top 28 by
                     cumulative time
    --trace          time the main thread's parts in every frame and print
                     what the slowest 5 % of frames spend beyond the median
                     ones (the slow frames' causes)
    --no-hud         skip the demo's HUD refresh (its text costs ~5 ms every
                     0.2 s), so the frames measure the world only
    --size           window or buffer size (default "1600 900")
    --workers        generation processes (the terrain's setting; default:
                     physical cores - 1)
    --offscreen      render to an offscreen buffer: no display refresh cap
                     or throttling
    --quality        every part's preset (WorldSettings), default high
    --<part>-quality one part's preset over it: terrain, atmosphere, sky,
                     clouds, ocean
    --no-clouds      no clouds at all
    --cloud-debug    PlanetClouds.set_debug bits (default 0)

Flights (boosted, along the coast of the demo level from its reference
point, heading west):
    low      camera level, 300 m above the ground, ~1,500 m/s
    cruise   camera level, 3 km up, ~15 km/s
    descent  nose straight down from 3 planet radii, until 400 m above the
             ground

Each starts from a settled view, so it measures flying, not the initial fill.
Per flight it prints the frame count, mean fps, frame time (mean, median,
p95, max), the demo's update (mean, median, p95), chunks generated per second, the
mean backlog (chunks wanted), the refinement lag (from a chunk first
wanted to it drawn: median and p95, and the median of the three finest
levels, the ground nearest the camera) and the finest level drawn, on
average over the frames (lower is sharper).
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    TerrainDemo,
    add_world_arguments,
    describe_window,
    settings_from_args,
    window_config,
)
from demo_world import WorldDemo  # noqa: E402

from world_builder.terrain.level import DEMO_LEVEL  # noqa: E402

DEMOS = {"terrain": TerrainDemo, "world": WorldDemo}

#: The main thread's parts that --trace times: (module, class, method).
TRACED = (
    ("world_builder.terrain.terrain", "Terrain", "_select"),
    ("world_builder.terrain.terrain", "Terrain", "_upload"),
    ("world_builder.terrain.terrain", "Terrain", "_rebase"),
    ("world_builder.terrain.terrain", "Terrain", "_store"),
    ("world_builder.terrain.terrain", "Terrain", "_evict_store"),
    ("world_builder.terrain.terrain", "Terrain", "_evict_nodes"),
    ("world_builder.terrain.workers", "ChunkExecutor", "collect"),
    ("world_builder.clouds.sphere", "CloudVolume", "_select"),
    ("world_builder.clouds.sphere", "CloudVolume", "_upload"),
    ("world_builder.clouds.sphere", "CloudVolume", "_resort"),
    ("world_builder.clouds.sphere", "CloudVolume", "_rebase"),
    ("world_builder.ocean.sea", "Sea", "update"),
    ("demo_terrain", "TerrainDemo", "_fly"),
    ("direct.showbase.ShowBase", "ShowBase", "_ShowBase__garbageCollectStates"),
    ("direct.showbase.ShowBase", "ShowBase", "_ShowBase__igLoop"),
)
#: Shorter names for Panda's own tasks.
_TRACE_NAMES = {
    "ShowBase._ShowBase__garbageCollectStates": "panda: state GC",
    "ShowBase._ShowBase__igLoop": "panda: renderFrame",
}


class Trace:
    """Times the :data:`TRACED` parts of every frame, and Python's garbage
    collections. Install before the demo is built: Panda's tasks keep the
    methods they were given."""

    def __init__(self):
        self.frame: dict[str, float] = defaultdict(float)
        self.frames: list[tuple[float, dict[str, float]]] = []
        self.active = False
        for module, cls, method in TRACED:
            owner = getattr(
                sys.modules.get(module) or __import__(module, fromlist=[cls]), cls
            )
            self._wrap(
                owner, method, _TRACE_NAMES.get(f"{cls}.{method}", f"{cls}.{method}")
            )
        gc.callbacks.append(self._gc)
        self._gc_start = 0.0

    def _wrap(self, owner, method: str, label: str) -> None:
        original = getattr(owner, method)
        frame = self.frame

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                frame[label] += (time.perf_counter() - start) * 1e3

        setattr(owner, method, timed)

    def _gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._gc_start = time.perf_counter()
        else:
            label = f"python: gc gen{info['generation']}"
            self.frame[label] += (time.perf_counter() - self._gc_start) * 1e3

    def begin(self) -> None:
        self.frame.clear()

    def end(self, total_ms: float) -> None:
        if self.active:
            self.frames.append((total_ms, dict(self.frame)))

    def report(self) -> None:
        """Print what the slowest 5 % of frames spend beyond median ones."""
        if not self.frames:
            return
        totals = np.array([t for t, _ in self.frames])
        median, p95 = np.median(totals), np.percentile(totals, 95)
        slow = [r for t, r in self.frames if t >= p95]
        typical = [r for t, r in self.frames if abs(t - median) < 1.5] or [{}]
        print(
            f"  trace: {len(slow)} frames at or over p95 ({p95:.1f} ms) against "
            f"{len(typical)} within 1.5 ms of the median ({median:.1f} ms)"
        )
        print(
            f"  {'part':34s} {'slow':>6s} {'median':>7s} {'excess':>7s}  in slow frames"
        )
        rows = []
        for label in {k for _, r in self.frames for k in r}:
            s = float(np.mean([r.get(label, 0.0) for r in slow]))
            m = float(np.mean([r.get(label, 0.0) for r in typical]))
            share = float(np.mean([r.get(label, 0.0) > 0.5 for r in slow]))
            rows.append((s - m, label, s, m, share))
        for excess, label, s, m, share in sorted(rows, reverse=True):
            print(
                f"  {label:34s} {s:6.2f} {m:7.2f} {excess:7.2f}  {share * 100:5.0f} %"
            )
        self.frames.clear()


FLIGHTS = ("low", "cruise", "descent")


def _settle(demo: TerrainDemo, timeout: float = 60.0) -> None:
    """Step frames until nothing is wanted or generating for 10 frames, or
    *timeout* seconds."""
    start = time.perf_counter()
    quiet = 0
    while time.perf_counter() - start < timeout and quiet < 10:
        demo.taskMgr.step()
        busy = demo.executor.pending or any(
            body.terrain.stats["wanted"] for body in demo.bodies
        )
        quiet = 0 if busy else quiet + 1


def _start(demo: TerrainDemo, flight: str) -> None:
    """Put the camera at a flight's start, boosted, its keys held."""
    spec = demo.terrain.spec
    lat, lon = spec.reference_lat, spec.reference_lon
    if flight == "descent":
        # Nose straight down, flying forward: the camera's own "down" would
        # point backwards at any other pitch, and climb away.
        demo._place_camera(lat, lon, 3.0 * demo.radius, 0.0, -89.9)
        demo._keys["arrow_up"] = True
    else:
        altitude = 300.0 if flight == "low" else 3000.0
        # Over the start first: the ground is sampled under the camera.
        demo._place_camera(lat, lon, altitude, 90.0, 0.0)
        ground = float(demo.terrain.planet.sample_dirs(demo._up()[None, :])[0][0])
        demo._place_camera(lat, lon, max(ground, 0.0) + altitude, 90.0, 0.0)
        demo._keys["arrow_up"] = True
    demo._keys["shift"] = True
    demo.set_local_sun(210.0, 35.0)


def _release(demo: TerrainDemo) -> None:
    """Let go of every key."""
    for key in demo._keys:
        demo._keys[key] = False


def run(
    demo: TerrainDemo,
    flight: str,
    seconds: float,
    profile: bool,
    trace: Trace | None = None,
) -> None:
    """Fly one flight from a settled view and print its timings.

    :param seconds: the flight's length, at most (the descent stops 400 m
        above the ground)
    :param profile: also print a cProfile of the main thread
    :param trace: also print what the slowest frames spend their time on
    """
    _release(demo)
    _start(demo, flight)
    _release(demo)
    _settle(demo)
    _start(demo, flight)
    frames, updates, backlog, finest = [], [], [], []
    generated = sum(body.terrain.stats["generated"] for body in demo.bodies)
    for body in demo.bodies:
        body.terrain.refine_lags.clear()
    original = demo._update

    def timed(task) -> int:
        start = time.perf_counter()
        result = original(task)
        updates.append(time.perf_counter() - start)
        return result

    demo.taskMgr.remove("terrain_demo_update")
    demo.taskMgr.add(timed, "terrain_demo_update")
    profiler = cProfile.Profile() if profile else None
    start = last = time.perf_counter()
    if profiler:
        profiler.enable()
    if trace:
        trace.active = True
    while time.perf_counter() - start < seconds:
        if trace:
            trace.begin()
        demo.taskMgr.step()
        now = time.perf_counter()
        frames.append(now - last)
        if trace:
            trace.end((now - last) * 1e3)
        backlog.append(sum(body.terrain.stats["wanted"] for body in demo.bodies))
        finest.append(demo.body.terrain.stats["finest"])
        last = now
        if flight == "descent" and (
            float(np.linalg.norm(demo._relative())) - demo.body.radius - demo._ground
            < 400.0
        ):
            break
    if profiler:
        profiler.disable()
    if trace:
        trace.active = False
    demo.taskMgr.remove("terrain_demo_update")
    demo.taskMgr.add(original, "terrain_demo_update")
    _release(demo)

    elapsed = time.perf_counter() - start
    generated = sum(body.terrain.stats["generated"] for body in demo.bodies) - generated
    f = np.array(frames[3:]) * 1e3
    u = np.array(updates[3:]) * 1e3
    lagged = [lag for b in demo.bodies for lag in b.terrain.refine_lags] or [(0, 0.0)]
    lags = np.array([t for _, t in lagged]) * 1e3
    near = np.array([t for level, t in lagged if level <= 2] or [0.0]) * 1e3
    print(
        f"{flight:<8} {len(f):5d} frames  fps {1e3 / np.mean(f):5.1f}   "
        f"frame ms mean {np.mean(f):5.1f} median {np.median(f):5.1f} "
        f"p95 {np.percentile(f, 95):5.1f} max {f.max():6.1f}   demo update ms "
        f"mean {np.mean(u):5.1f} median {np.median(u):5.1f} "
        f"p95 {np.percentile(u, 95):5.1f}   chunks/s {generated / elapsed:4.0f}"
        f"  backlog mean {np.mean(backlog):4.0f}"
        f"   lag ms median {np.median(lags):5.0f} p95 {np.percentile(lags, 95):5.0f}"
        f" levels 0-2 {np.median(near):5.0f}   finest level {np.mean(finest):4.1f}"
    )
    if trace:
        trace.report()
    if profiler:
        out = io.StringIO()
        stats = pstats.Stats(profiler, stream=out).sort_stats("cumulative")
        stats.print_stats(28)
        print(out.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--demo", default="terrain", choices=tuple(DEMOS), help="which demo flies"
    )
    parser.add_argument(
        "--flight", nargs="*", default=list(FLIGHTS), help="low, cruise, descent"
    )
    parser.add_argument(
        "--seconds", type=float, default=15.0, help="per flight, at most"
    )
    parser.add_argument(
        "--profile", action="store_true", help="cProfile the main thread"
    )
    parser.add_argument(
        "--trace", action="store_true", help="what the slowest frames spend"
    )
    parser.add_argument(
        "--no-hud", action="store_true", help="skip the demo's HUD refresh"
    )
    parser.add_argument("--size", default="1600 900", help='"width height"')
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="render to an offscreen buffer: no display refresh cap or throttling",
    )
    add_world_arguments(parser)
    parser.add_argument(
        "--cloud-debug", type=int, default=0, help="PlanetClouds.set_debug bits"
    )
    args = parser.parse_args()
    window_config("Flight benchmark", args.size, args.offscreen, vsync=False)
    settings = settings_from_args(args)
    trace = Trace() if args.trace else None  # before the demo: see Trace
    demo = DEMOS[args.demo](
        str(DEMO_LEVEL),
        "process",
        None,
        settings=settings,
        clouds=not args.no_clouds,
        cloud_debug=args.cloud_debug,
    )
    demo.setFrameRateMeter(False)
    if args.no_hud:
        demo._update_hud = lambda: None
    print(f"{args.demo} demo,", describe_window(demo), f"settings {settings.to_dict()}")
    try:
        for flight in args.flight:
            run(demo, flight, args.seconds, args.profile, trace)
    finally:
        demo.destroy_terrain()


if __name__ == "__main__":
    main()
