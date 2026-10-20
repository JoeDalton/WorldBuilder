"""
Frame-time benchmark of the terrain demo in fast, scripted flights.

Runs the real demo (window, workers, both cameras, ocean) with vsync off and
flies fixed paths, reporting frame times and where the main thread spends
them::

    python scripts/flight_bench.py [--flight low cruise descent] [--seconds S]
        [--profile] [--size "1600 900"] [--workers N] [--offscreen]
        [--cloud-quality high|mid|low|off] [--cloud-debug BITS]

Flights (camera level, boosted, along the coast of the demo level):
    low      300 m above the ground, ~750 m/s
    cruise   3 km up, ~7.5 km/s
    descent  straight down from 3 planet radii to 300 m

Each starts from a settled view, so it measures flying, not the initial fill.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    CLOUD_QUALITIES,
    TerrainDemo,
    describe_window,
    window_config,
)

from world_builder.terrain.level import DEMO_LEVEL  # noqa: E402

FLIGHTS = ("low", "cruise", "descent")


def _settle(demo: TerrainDemo, timeout: float = 60.0) -> None:
    start = time.perf_counter()
    quiet = 0
    while time.perf_counter() - start < timeout and quiet < 10:
        demo.taskMgr.step()
        busy = demo.executor.pending or any(
            body.terrain.stats["wanted"] for body in demo.bodies
        )
        quiet = 0 if busy else quiet + 1


def _start(demo: TerrainDemo, flight: str) -> None:
    spec = demo.terrain.spec
    lat, lon = spec.reference_lat, spec.reference_lon
    if flight == "descent":
        # Nose straight down, flying forward: the camera's own "down" would
        # point backwards at any other pitch, and climb away.
        demo._place_camera(lat, lon, 3.0 * demo.radius, 0.0, -89.9)
        demo._keys["arrow_up"] = True
    else:
        altitude = 300.0 if flight == "low" else 3000.0
        ground = float(demo.terrain.planet.sample_dirs(demo._up()[None, :])[0][0])
        demo._place_camera(lat, lon, max(ground, 0.0) + altitude, 90.0, 0.0)
        demo._keys["arrow_up"] = True
    demo._keys["shift"] = True
    demo.set_local_sun(210.0, 35.0)


def _release(demo: TerrainDemo) -> None:
    for key in demo._keys:
        demo._keys[key] = False


def run(demo: TerrainDemo, flight: str, seconds: float, profile: bool):
    _release(demo)
    _start(demo, flight)
    _release(demo)
    _settle(demo)
    _start(demo, flight)
    frames, updates, backlog = [], [], []
    generated = sum(body.terrain.stats["generated"] for body in demo.bodies)
    original = demo._update

    def timed(task):
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
    while time.perf_counter() - start < seconds:
        demo.taskMgr.step()
        now = time.perf_counter()
        frames.append(now - last)
        backlog.append(sum(body.terrain.stats["wanted"] for body in demo.bodies))
        last = now
        if (
            flight == "descent"
            and demo._ground
            and (
                float(np.linalg.norm(demo._relative()))
                - demo.body.radius
                - demo._ground
                < 400.0
            )
        ):
            break
    if profiler:
        profiler.disable()
    demo.taskMgr.remove("terrain_demo_update")
    demo.taskMgr.add(original, "terrain_demo_update")
    _release(demo)

    elapsed = time.perf_counter() - start
    generated = sum(body.terrain.stats["generated"] for body in demo.bodies) - generated
    f = np.array(frames[3:]) * 1e3
    u = np.array(updates[3:]) * 1e3
    print(
        f"{flight:<8} {len(f):5d} frames  fps {1e3 / np.mean(f):5.1f}   "
        f"frame ms median {np.median(f):5.1f} p95 {np.percentile(f, 95):5.1f} "
        f"max {f.max():6.1f}   demo update ms median {np.median(u):5.1f} "
        f"p95 {np.percentile(u, 95):5.1f}   chunks/s {generated / elapsed:4.0f}"
        f"  backlog mean {np.mean(backlog):4.0f}"
    )
    if profiler:
        out = io.StringIO()
        stats = pstats.Stats(profiler, stream=out).sort_stats("cumulative")
        stats.print_stats(28)
        print(out.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--flight", nargs="*", default=list(FLIGHTS))
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--size", default="1600 900")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="render to an offscreen buffer: no display refresh cap or throttling",
    )
    parser.add_argument("--cloud-quality", default="high", choices=CLOUD_QUALITIES)
    parser.add_argument(
        "--cloud-debug", type=int, default=0, help="SphericalClouds.set_debug bits"
    )
    args = parser.parse_args()
    window_config("Flight benchmark", args.size, args.offscreen, vsync=False)
    demo = TerrainDemo(
        str(DEMO_LEVEL),
        "process",
        None,
        workers=args.workers,
        cloud_quality=args.cloud_quality,
        cloud_debug=args.cloud_debug,
    )
    demo.setFrameRateMeter(False)
    print(describe_window(demo), f"clouds {args.cloud_quality}")
    try:
        for flight in args.flight:
            run(demo, flight, args.seconds, args.profile)
    finally:
        demo.destroy_terrain()


if __name__ == "__main__":
    main()
