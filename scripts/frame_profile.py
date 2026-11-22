"""
Where a frame's time goes, per part, at fixed poses of a world file (the
reference profile, RENDERING_PLAN.md part A; results in
docs/source/frame_profile.md)::

    python scripts/frame_profile.py [WORLD.yaml] [--pose NAME ...] [--size "W H"]
        [--quality LEVEL] [--frames N] [--offscreen]
        [--at LAT LON ALT HEADING PITCH ...] [--scene bench-sea|bench-dry]

The poses are over the world file's first body (default: the home planet of
three_planets.yaml), at its reference point.

At each pose (POSES below), settled, GPU-synced frames (a one-pixel
read back after each step, as terrain_shots.py times them):

* **frame** the median frame, everything shown, and its p95;
* **step** the median of ``taskMgr.step()`` alone, before the GPU sync: the
  CPU's part (cull, draw and submission, the demo's update). Close to the
  frame: CPU-bound there;
* each part's cost: the median frame with it shown less with it off:
  **terrain** (the terrain's roots, the near sea with them), **refl** (the
  sea's reflection pass), **sky**, **clouds** (the low clouds and their
  passes), **cirrus** (the march's, its high layer), **skyview** (the sky-view table and
  its bake against the sky's march: negative where it saves), **aerial** (the
  aerial-perspective cells and their bake against the terrain's, sea's and
  models' march: the same), **godrays** (the
  clouds' and the mountains' shadows in that bake,
  ``AtmosphereSettings.god_rays``);
* **atmo** the per-pixel atmosphere's share: the frame at the preset's
  samples less at 2 (``AtmosphereSettings.steps``), scaled to all of them
  (``* steps / (steps - 2)``): a linear estimate of the whole march.

With --scene, a bench scene (BENCH_SCENES): the level's planet simple (no
vegetation, mountains 600 m), with its reflective sea or without, at the
ground, 1.5 km and 10 km up over the start.

A part's cost includes its CPU's when the frame is CPU-bound. Every number
is a median of --frames frames; two runs differ by 0.3-0.5 ms. Frames faster
than the display's refresh are held to it in a window even with vsync off
(5.53 ms at 180 Hz, every part 0): use --offscreen there (LOW).
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import tools  # noqa: E402
from demo_world import DEFAULT_WORLD, WorldDemo  # noqa: E402
from world_builder.terrain.chart import TangentFrame, latlon  # noqa: E402
from world_builder.world_file import WorldSpec, parse_world, world_to_yaml  # noqa: E402

#: The bench scenes (--scene): the first body's planet, simple: no vegetation,
#: its mountains 600 m (not 4.8 km), with its reflective sea (bench-sea) or
#: none (bench-dry). Their poses: the ground, 1.5 km and 10 km up at the
#: start, the default clouds in view.
BENCH_SCENES = {"bench-sea": True, "bench-dry": False}
BENCH_POSES = ("ground", "bench_1500", "bench_10000")

#: The poses profiled by default.
POSES = (
    "ground",
    "coast",
    "ground_up",
    "in_clouds",
    "clouds_low",
    "sunset",
    "high",
    "orbit",
)

#: Sun per pose: (azimuth, elevation) as seen there; default (210, 35).
SUNS = {
    "sunset": (270.0, 3.0),
    "clouds_noon": (210.0, 80.0),
    "clouds_low": (210.0, 20.0),
}
DEFAULT_SUN = (210.0, 35.0)


def _poses(demo: WorldDemo) -> dict:
    """:returns: name -> (lat, lon, altitude, heading, pitch, body index), over
    the first body's reference point"""
    spec = demo.bodies[0].look.spec
    lat, lon = spec.reference_lat, spec.reference_lon
    radius = demo.bodies[0].radius
    return {
        "ground": (lat, lon, 800.0, 0.0, 0.0, 0),
        "ground_up": (lat, lon, 800.0, 0.0, 11.0, 0),
        "coast": (lat, lon, 2000.0, 0.0, -10.0, 0),
        "high": (lat, lon, 20000.0, 0.0, -25.0, 0),
        "orbit": (lat, lon, 3.0 * radius, 0.0, -89.9, 0),
        # Inside the cloud deck (2.2-3.2 km above sea level).
        "in_clouds": (lat, lon, 2700.0, 0.0, 0.0, 0),
        # The sun low in the west, looking at it (heading 90 = west).
        "sunset": (lat, lon, 600.0, 90.0, 2.0, 0),
        # The clouds from above, looking across the sun's azimuth (210).
        "clouds_noon": (lat, lon, 6000.0, -120.0, -35.0, 0),
        "clouds_low": (lat, lon, 6000.0, -120.0, -35.0, 0),
    }


def _set_local_sun(demo: WorldDemo, azimuth: float, elevation: float) -> None:
    """Put the sun at an azimuth/elevation (degrees; 0 = north, toward east) as
    seen from the point under the camera."""
    up = demo._up()
    lat, lon = latlon(up[None, :])
    frame = TangentFrame.at(float(lat[0]), float(lon[0]))
    az, el = math.radians(azimuth), math.radians(elevation)
    demo.world.world.set_sun(
        math.cos(el) * math.sin(az) * frame.east
        + math.cos(el) * math.cos(az) * frame.north
        + math.sin(el) * up
    )


def _bench_world(path: str, ocean: bool) -> str:
    """:returns: a temporary world file: *path*'s with the first body simple
    (no vegetation, mountains 600 m, a sea or none)"""
    spec = parse_world(path)
    body = spec.bodies[0]
    planet = replace(
        body.spec,
        has_vegetation=False,
        has_ocean=ocean,
        elevation=replace(body.spec.elevation, mountain_height=600.0),
    )
    if not ocean:
        body = replace(body, ocean=None)
    body = replace(body, spec=planet)
    simple = WorldSpec(
        bodies=(body, *spec.bodies[1:]),
        fields=spec.fields,
        sun=spec.sun,
        sky=spec.sky,
        space_ambient=spec.space_ambient,
    )
    out = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    out.write(world_to_yaml(simple))
    out.close()
    return out.name


def _settle(demo: WorldDemo, timeout: float) -> bool:
    """Step frames until nothing is loading for 10 frames.

    :returns: whether that happened before *timeout* seconds
    """
    start = time.perf_counter()
    quiet = 0
    world = demo.world.world
    while time.perf_counter() - start < timeout:
        demo.taskMgr.step()
        busy = (
            world.executor.pending
            or world.loading
            or not all(body.shadow_map.current for body in demo.bodies)
        )
        quiet = 0 if busy else quiet + 1
        if quiet >= 10:
            return True
    return False


def _time(demo: WorldDemo, probe, frames: int) -> tuple[float, float, float]:
    """:returns: the median frame, its p95 and the median step alone, ms"""
    for _ in range(5):
        demo.taskMgr.step()
    whole, steps = [], []
    for _ in range(frames):
        start = time.perf_counter()
        demo.taskMgr.step()
        stepped = time.perf_counter()
        probe.get_screenshot()  # waits for the GPU
        end = time.perf_counter()
        whole.append(end - start)
        steps.append(stepped - start)
    return (
        float(np.median(whole)) * 1e3,
        float(np.percentile(whole, 95)) * 1e3,
        float(np.median(steps)) * 1e3,
    )


def _parts(demo: WorldDemo) -> dict:
    """Each part as (off, on) callables."""
    world = demo.world.world
    terrains = [body.terrain.root for body in demo.bodies]
    parts = {
        "terrain": (
            lambda: [t.hide() for t in terrains],
            lambda: [t.show() for t in terrains],
        ),
    }
    if world.sky is not None:
        parts["sky"] = (world.sky.node.hide, world.sky.node.show)
    atmospheres = world.atmospheres
    if getattr(atmospheres, "aerial", None) is not None:
        # the aerial-perspective cells against the per-pixel march
        on = atmospheres.settings
        parts["aerial"] = (
            lambda: atmospheres.set_settings(replace(on, aerial=False)),
            lambda: atmospheres.set_settings(on),
        )
        if on.god_rays:
            # the clouds' and the mountains' shadows in the air's bake
            parts["godrays"] = (
                lambda: atmospheres.set_settings(replace(on, god_rays=False)),
                lambda: atmospheres.set_settings(on),
            )
    view = getattr(world.sky, "view", None)
    if view is not None:
        # the sky-view table against the sky's march (negative: it saves)
        parts["skyview"] = (
            lambda: view.set_active(False),
            lambda: view.set_active(True),
        )
    oceans = [body.ocean for body in demo.bodies if body.ocean is not None]
    if oceans:
        parts["refl"] = (
            lambda: [o.refl_buffer.set_active(False) for o in oceans],
            lambda: [o.refl_buffer.set_active(bool(o._showing)) for o in oceans],
        )
    clouds = [body.clouds for body in demo.bodies if body.clouds is not None]
    if clouds:
        parts["clouds"] = (
            lambda: [c.set_visible(False) for c in clouds],
            lambda: [c.set_visible(True) for c in clouds],
        )
        cirrus = [c for c in clouds if "cirrus" in c.volume.high_genera]
        if cirrus:
            # the march's cirrus (its high layer's channel)
            parts["cirrus"] = (
                lambda: [c.set_genus_visible("cirrus", False) for c in cirrus],
                lambda: [c.set_genus_visible("cirrus", True) for c in cirrus],
            )
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "world", nargs="?", default=str(DEFAULT_WORLD), help="the world file"
    )
    parser.add_argument("--pose", nargs="*", default=None, help="default: POSES")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--size", default="2560 1440", help='"width height"')
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument(
        "--at",
        nargs=5,
        type=float,
        action="append",
        default=[],
        metavar=("LAT", "LON", "ALT", "HEADING", "PITCH"),
        help='more poses, "at1", "at2"...: the demo\'s printed camera line',
    )
    parser.add_argument(
        "--scene",
        choices=sorted(BENCH_SCENES),
        default=None,
        help="a bench scene (BENCH_SCENES) and its poses instead of the default",
    )
    tools.add_quality_arguments(parser)
    parser.set_defaults(frame_cap=0)  # timings: never capped (--frame-cap N)
    args = parser.parse_args()
    tools.window_config("Frame profile", args.size, args.offscreen, vsync=False)
    settings = tools.settings_from_args(args)
    path = args.world
    if args.scene:
        path = _bench_world(path, BENCH_SCENES[args.scene])
    demo = WorldDemo(path, settings=settings)
    demo.setFrameRateMeter(False)
    print(tools.describe_window(demo))
    win = demo.win
    probe = win.make_display_region(0, 1 / win.get_x_size(), 0, 1 / win.get_y_size())
    poses = _poses(demo)
    extra = [f"at{i + 1}" for i in range(len(args.at))]
    for name, at in zip(extra, args.at):
        poses[name] = (*at, 0)
    if args.scene:
        lat, lon = poses["ground"][:2]
        for alt in (1500.0, 10000.0):
            poses[f"bench_{alt:.0f}"] = (lat, lon, alt, 0.0, 3.0, 0)
    atmo = settings.atmosphere
    columns = ["frame", "p95", "step", "terrain", "refl", "sky", "clouds", "cirrus"]
    columns += ["skyview", "aerial", "godrays", "atmo"]
    print(f"{'pose':<12} " + " ".join(f"{c:>7}" for c in columns))
    default = list(BENCH_POSES if args.scene else POSES)
    try:
        for name in (args.pose or default) + extra:
            lat, lon, alt, heading, pitch, body = poses[name]
            demo.place_camera(lat, lon, alt, heading, pitch, demo.bodies[body])
            _set_local_sun(demo, *SUNS.get(name, DEFAULT_SUN))
            settled = _settle(demo, args.timeout)
            frame, p95, step = _time(demo, probe, args.frames)
            row = {"frame": frame, "p95": p95, "step": step}
            for part, (off, on) in _parts(demo).items():
                off()
                row[part] = frame - _time(demo, probe, args.frames)[0]
                on()
            if atmo.steps > 2:
                demo.world.world.atmospheres.set_settings(replace(atmo, steps=2))
                fewer = _time(demo, probe, args.frames)[0]
                demo.world.world.atmospheres.set_settings(atmo)
                row["atmo"] = (frame - fewer) * atmo.steps / (atmo.steps - 2)
            cells = " ".join(
                f"{row[c]:>7.2f}" if c in row else f"{'-':>7}" for c in columns
            )
            print(f"{name:<12} {cells}{'' if settled else '  (not settled)'}")
            sys.stdout.flush()
    finally:
        demo.destroy_world()
        if args.scene:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
