"""
Render the terrain demo from fixed poses to PNG: a visual regression check.

The shaders cannot be validated headless (see tests/test_terrain_mesh.py), so
this drives the real demo (planet, moon, atmosphere, clouds, ocean, sky): for
each pose it waits until the terrain has finished refining (or a timeout),
times a few frames, then saves a screenshot::

    python scripts/terrain_shots.py [--out DIR] [--level FILE]
        [--planet PRESET] [--seed N] [--start LAT LON] [--radius METRES]
        [--pose NAME ...]
        [--timeout SECONDS]
        [--size "1600 900"] [--offscreen]
        [--quality low|mid|high|ultra] [--<part>-quality LEVEL] [--no-clouds]
        [--workers N] [--cloud-debug BITS] [--frames N] [--ocean-debug MODE]

    --out            where the PNGs go (default ./terrain_shots)
    --level          the planet's level file (default: the demo level)
    --planet         overrides the level's planet preset (earthlike, desert,
                     forest, moon, ice)
    --seed           overrides the level's seed
    --start          the ground pose over this latitude and longitude
                     (degrees), 400 m above the ground
    --radius         overrides the level's planet radius
    --pose           the poses to shoot (default: all, in the order below)
    --timeout        seconds to wait for a pose to settle (default 90)
    --size           window or buffer size (default "1600 900")
    --offscreen      render to an offscreen buffer, not a window
    --quality        every part's preset (WorldSettings), default high
    --<part>-quality one part's preset over it: terrain, atmosphere, sky,
                     clouds, ocean
    --no-clouds      no clouds at all
    --workers        generation processes (the terrain's setting)
    --cloud-debug    PlanetClouds.set_debug bits (cloud.frag's
                     uCloudDebug: 1 no sun march, 2 no atmosphere, 4 two
                     octaves at most, 8 flat colour; default 0)
    --frames         frames timed once settled (default 60)
    --ocean-debug    Sea.set_debug_mode view (sea.glsl's uSeaDebug, on water
                     in its near look: 9 octaves resolved, 10 unresolved
                     slope spread, 12 depth, ...; default 0)

No multisampling, vsync off. Once a pose has settled, N frames are timed,
each waiting for the GPU to finish (a one-pixel read back): CPU and GPU time
in series, so a difference between two runs is the difference in work.
Without the wait, an offscreen frame is not held back by the GPU at all. Each
pose prints whether it settled, the body's chunk count and finest level, the
cloud billboards, the LOD pass and the median frame time.

Poses: ground (the level's start), coast (2 km up over the reference point),
high (20 km up), swell / sea_surface (13 and 17 km up, inside the sea's 12-18 km
fade from its near look to its flat far one), stratosphere (200 km up),
orbit (3 radii up, looking down), corner (over a cube corner, where three faces
meet), lod (the LOD debug view from 20 km), in_clouds (inside the cloud deck,
2.7 km up), sunset (600 m up, looking west at the sun 3 degrees up), moonrise
(the moon over the northern horizon, from the start area), moon_orbit (3 moon
radii up, looking down), moon_ground (2 km over the moon's sand), cirrus_up
(from the ground, looking 35 degrees up: the cumulus in front of the cirrus),
cirrus_between (5 km up, between the cumulus and the cirrus, looking at the
horizon), cirrus_above (15 km up, above both, looking down at the horizon: the
cirrus in front of the cumulus), cirrus_dusk (the cirrus still lit after the
cumulus has gone grey: the sun 2 degrees down), night_orbit (1.5 radii up,
looking down at the terminator: the sun 60 degrees below the nadir's horizon),
night_sea (3 m over the sea, the sun 8 degrees down: the sky's reflection must
be as dark as the sky), beach (40 m up, 300 m off the coast nearest the
reference point, looking at it: the sea's foam, colour by depth and swell),
open_sea (150 m over open water, looking along it: the swell and how it
flattens with distance). Every other pose has the sun at azimuth 210, elevation
35, as seen there.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from panda3d.core import Filename

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    TerrainDemo,
    add_planet_arguments,
    add_world_arguments,
    describe_window,
    settings_from_args,
    window_config,
)

from world_builder.terrain.chart import AzimuthalEquidistantChart, latlon  # noqa: E402
from world_builder.terrain.level import DEMO_LEVEL  # noqa: E402

CORNER = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
#: Sun (azimuth, elevation) per pose, as seen there; default (210, 35).
SUNS = {
    "sunset": (270.0, 3.0),
    "night_orbit": (90.0, -60.0),
    "night_sea": (90.0, -8.0),
    "cirrus_dusk": (270.0, -2.0),
}


def _poses(demo: TerrainDemo) -> dict:
    """:returns: name -> (lat, lon, altitude, heading, pitch, debug view,
    body index)"""
    start = demo.start_pose or {}
    lat0 = start.get("lat", demo.terrain.spec.reference_lat)
    lon0 = start.get("lon", demo.terrain.spec.reference_lon)
    ref_lat, ref_lon = demo.terrain.spec.reference_lat, demo.terrain.spec.reference_lon
    corner_lat, corner_lon = (float(v[0]) for v in latlon(CORNER[None, :]))
    radius = demo.radius
    poses = {
        "ground": (
            lat0,
            lon0,
            start.get("alt", 800.0),
            start.get("heading", 0.0),
            start.get("pitch", 0.0),
            0,
            0,
        ),
        "coast": (ref_lat, ref_lon, 2000.0, 0.0, -10.0, 0, 0),
        "high": (lat0, lon0, 20000.0, 0.0, -25.0, 0, 0),
        "swell": (ref_lat, ref_lon, 13000.0, 0.0, -20.0, 0, 0),
        "sea_surface": (ref_lat, ref_lon, 17000.0, 0.0, -20.0, 0, 0),
        "stratosphere": (lat0, lon0, 200e3, 0.0, -45.0, 0, 0),
        "orbit": (lat0, lon0, 3.0 * radius, 0.0, -89.9, 0, 0),
        "corner": (corner_lat, corner_lon, 3000.0, 30.0, -30.0, 0, 0),
        "lod": (lat0, lon0, 20000.0, 0.0, -25.0, 1, 0),
    }
    # Inside the cloud deck (2.2-3.2 km above sea level).
    poses["in_clouds"] = (ref_lat, ref_lon, 2700.0, 0.0, 0.0, 0, 0)
    # The sun low in the west, looking at it (heading 90 = west).
    poses["sunset"] = (ref_lat, ref_lon, 600.0, 90.0, 2.0, 0, 0)
    if len(demo.bodies) > 1:
        moon = demo.bodies[1]
        poses["moonrise"] = (ref_lat, ref_lon, 1500.0, 0.0, 6.0, 0, 0)
        # Over the moon's side facing the planet.
        facing = -moon.position / np.linalg.norm(moon.position)
        lat, lon = (float(v[0]) for v in latlon(facing[None, :]))
        poses["moon_orbit"] = (lat, lon, 3.0 * moon.radius, 0.0, -89.9, 0, 1)
        poses["moon_ground"] = (lat + 1.0, lon, 2000.0, 0.0, -15.0, 0, 1)
    # The cirrus (8-9 km) against the cumulus (2.2-3.2 km): under both, between,
    # above both, and at dusk (looking west, toward the set sun).
    poses["cirrus_up"] = (lat0, lon0, start.get("alt", 800.0), 0.0, 35.0, 0, 0)
    poses["cirrus_between"] = (ref_lat, ref_lon, 5000.0, 0.0, 3.0, 0, 0)
    poses["cirrus_above"] = (ref_lat, ref_lon, 15000.0, 0.0, -8.0, 0, 0)
    poses["cirrus_dusk"] = (ref_lat, ref_lon, 600.0, 90.0, 15.0, 0, 0)
    # The terminator from orbit: the sun 60 degrees below the nadir's horizon.
    poses["night_orbit"] = (lat0, lon0, 1.5 * radius, 0.0, -89.9, 0, 0)
    # At sea level by night: the sky's reflection must be as dark as the sky.
    poses["night_sea"] = (-6.272, -52.0456, 3.0, -115.0, 2.0, 0, 0)
    # The open sea by day, 150 m up, looking out over it: the swell, and
    # where it fades out with the terrain's LOD (4-7 km away).
    poses["open_sea"] = (-6.272, -52.0456, 150.0, -115.0, -4.0, 0, 0)
    beach = _beach_pose(demo)
    if beach is not None:
        poses["beach"] = beach
    return poses


def _beach_pose(demo: TerrainDemo) -> tuple | None:
    """The ocean's shoreline up close: 40 m up, 300 m off the coast nearest
    the reference point, looking at it (its foam, colour by depth and swell).

    :returns: a pose (see :func:`_poses`), or None without a coast in 10 km
    """
    spec = demo.terrain.spec
    chart = AzimuthalEquidistantChart(
        spec.reference_lat, spec.reference_lon, demo.radius
    )
    angles = np.radians(np.arange(0.0, 360.0, 10.0))
    reach = np.arange(0.0, 10000.0, 20.0)
    x = np.outer(np.sin(angles), reach)  # east
    y = np.outer(np.cos(angles), reach)  # north
    heights = demo.terrain.planet.sample_dirs(chart.to_dirs(x, y), apply_stamps=True)[
        0
    ].reshape(x.shape)
    best = None
    for ray, h in enumerate(heights):
        # The steps from sea onto land along this ray, with 300 m of sea
        # before them.
        for step in np.nonzero((h[:-1] < 0.0) & (h[1:] >= 0.0))[0]:
            if step >= 15 and np.all(h[step - 15 : step + 1] < 0.0):
                if best is None or step < best[1]:
                    best = (ray, int(step))
                break
    if best is None:
        return None
    ray, step = best
    back = step - 15  # 300 m out to sea
    dirs = chart.to_dirs(x[ray, back], y[ray, back])
    lat, lon = (float(v[0]) for v in latlon(dirs))
    # Heading toward the shore, counter-clockwise from north.
    heading = -math.degrees(angles[ray])
    return (lat, lon, 40.0, heading, -8.0, 0, 0)


def _settle(demo: TerrainDemo, timeout: float) -> bool:
    """Step frames until nothing is wanted or generating for 10 frames.

    :returns: whether that happened before *timeout* seconds
    """
    start = time.perf_counter()
    quiet = 0
    while time.perf_counter() - start < timeout:
        demo.taskMgr.step()
        busy = demo.executor.pending or any(
            body.terrain.stats["wanted"] for body in demo.bodies
        )
        quiet = 0 if busy else quiet + 1
        if quiet >= 10:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=Path("terrain_shots"), help="PNG directory"
    )
    parser.add_argument("--level", default=str(DEMO_LEVEL), help="level file")
    parser.add_argument(
        "--radius", type=float, default=None, help="planet radius, metres"
    )
    add_planet_arguments(parser)
    parser.add_argument("--pose", nargs="*", default=None, help="default: all")
    parser.add_argument(
        "--timeout", type=float, default=90.0, help="seconds to settle a pose"
    )
    parser.add_argument("--size", default="1600 900", help='"width height"')
    parser.add_argument(
        "--offscreen", action="store_true", help="an offscreen buffer, no window"
    )
    add_world_arguments(parser)
    parser.add_argument(
        "--cloud-debug", type=int, default=0, help="PlanetClouds.set_debug bits"
    )
    parser.add_argument(
        "--frames", type=int, default=60, help="frames timed once settled"
    )
    parser.add_argument(
        "--ocean-debug", type=int, default=0, help="Sea.set_debug_mode view"
    )
    args = parser.parse_args()
    window_config("Terrain shots", args.size, args.offscreen, vsync=False)
    args.out.mkdir(parents=True, exist_ok=True)
    settings = settings_from_args(args)
    demo = TerrainDemo(
        args.level,
        "process",
        args.radius,
        settings=settings,
        clouds=not args.no_clouds,
        cloud_debug=args.cloud_debug,
        ocean_debug=args.ocean_debug,
        preset=args.planet,
        seed=args.seed,
        start=args.start,
    )
    demo.setFrameRateMeter(False)
    print(describe_window(demo), f"settings {settings.to_dict()}")
    win = demo.win
    probe = win.make_display_region(0, 1 / win.get_x_size(), 0, 1 / win.get_y_size())
    poses = _poses(demo)
    for name in args.pose or list(poses):
        lat, lon, alt, heading, pitch, view, body = poses[name]
        demo.set_debug_view(view)
        demo._place_camera(lat, lon, alt, heading, pitch, demo.bodies[body])
        demo.set_local_sun(*SUNS.get(name, (210.0, 35.0)))
        settled = _settle(demo, args.timeout)
        for _ in range(3):  # let the last uploads reach the screen
            demo.taskMgr.step()
        times = []
        for _ in range(args.frames):  # the settled view's frame time
            start = time.perf_counter()
            demo.taskMgr.step()
            probe.get_screenshot()  # waits for the GPU
            times.append(time.perf_counter() - start)
        median = float(np.median(times)) * 1e3
        path = args.out / f"{name}.png"
        demo.win.save_screenshot(Filename.from_os_specific(str(path)))
        s = demo.bodies[body].terrain.stats
        clouds = demo.clouds.stats if demo.clouds is not None else None
        billboards = f"{clouds['billboards'] / 1e3:.0f}k bb, " if clouds else ""
        print(
            f"{name:<13} {'settled' if settled else 'TIMEOUT'}  {s['shown']} chunks, "
            f"finest {s['finest']}, {billboards}LOD {s['select_ms']:.1f} ms, "
            f"{median:.2f} ms ({1e3 / median:.0f} fps) -> {path}"
        )
    demo.destroy_terrain()


if __name__ == "__main__":
    main()
