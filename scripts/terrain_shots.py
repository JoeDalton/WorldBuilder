"""
Render the terrain demo from fixed poses to PNG: a visual regression check.

The shaders cannot be validated headless (see tests/test_terrain_mesh.py), so
this drives the real demo (planet, moon, atmosphere, clouds, ocean, sky): for
each pose it waits until the terrain has finished refining (or a timeout),
times a few frames, then saves a screenshot::

    python scripts/terrain_shots.py [--out DIR] [--level FILE]
        [--radius METRES] [--pose NAME ...] [--timeout SECONDS]
        [--size "1600 900"] [--offscreen]
        [--quality low|mid|high|ultra] [--<part>-quality LEVEL] [--no-clouds]
        [--workers N] [--cloud-debug BITS] [--frames N] [--ocean-debug MODE]
        [--no-ring]

    --out            where the PNGs go (default ./terrain_shots)
    --level          the planet's level file (default: the demo level)
    --radius         overrides the level's planet radius
    --pose           the poses to shoot (default: all, in the order below)
    --timeout        seconds to wait for a pose to settle (default 90)
    --size           window or buffer size (default "1600 900")
    --offscreen      render to an offscreen buffer, not a window
    --quality        every part's preset (WorldSettings), default high
    --<part>-quality one part's preset over it: terrain, atmosphere, sky,
                     clouds, ocean
    --no-clouds      no clouds at all
    --no-ring        no ring around the planet
    --workers        generation processes (the terrain's setting)
    --cloud-debug    SphericalClouds.set_debug bits (cloud.frag's
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
high (20 km up), swell / sea_surface (13 and 17 km up, inside the sea's
12-18 km fade from its near look to its flat far one), stratosphere (200 km
up), orbit (3 radii up, looking down), corner (over a cube corner, where three
faces meet), lod (the LOD debug view from 20 km), in_clouds (inside the cloud
deck, 2.7 km up), sunset (600 m up, looking west at the sun 3 degrees up),
moonrise (the moon over the northern horizon, from the start area),
moon_orbit (3 moon radii up, looking down), moon_ground (2 km over the
moon's sand), night_orbit (1.5 radii up, looking down at the terminator: the
sun 60 degrees below the nadir's horizon), night_sea (3 m over the sea, the
sun 8 degrees down: the sky's reflection must be as dark as the sky), beach
(40 m up, 300 m off the coast nearest the reference point, looking at it: the
sea's foam, colour by depth and swell), open_sea (150 m over open water,
looking along it: the swell and how it flattens with distance).

Eclipses, the sun aimed at a body from the point under the camera (plus an
offset): eclipse_ground (the moonrise view, looking at the moon, the sun
0.6 degrees off its centre: a partial eclipse), eclipse_total (the same, the
sun behind the moon's centre: the camera in the umbra's tip),
eclipse_orbit (3 radii over the point under the moon, looking down: the
moon's shadow on the planet, noon there), lunar_eclipse (the moon_orbit view,
the sun behind the planet, 0.8 degrees off: the planet's umbra edge across
the moon). The ring: ring_ground (the start area, 1.5 km up, looking south
at its arc), ring_orbit (6 radii over latitude 35, looking down: the planet
and its whole ring). Every other pose has the sun at azimuth 210, elevation
35, as seen there.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from panda3d.core import Filename

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    TerrainDemo,
    add_world_arguments,
    describe_window,
    settings_from_args,
    window_config,
)

from world_builder.terrain.chart import (  # noqa: E402
    AzimuthalEquidistantChart,
    TangentFrame,
    latlon,
)
from world_builder.terrain.level import DEMO_LEVEL  # noqa: E402

CORNER = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)


@dataclass(frozen=True)
class AimedSun:
    """A sun aimed at a body: the direction from the point under the camera
    (at sea level) to the body's centre, turned *offset* degrees toward that
    point's north (an eclipse's depth).

    :ivar body: the body's index in ``demo.bodies``
    :ivar offset: degrees
    """

    body: int
    offset: float = 0.0


#: Sun per pose: (azimuth, elevation) as seen there, or an :class:`AimedSun`;
#: default (210, 35).
SUNS = {
    "sunset": (270.0, 3.0),
    "night_orbit": (90.0, -60.0),
    "night_sea": (90.0, -8.0),
    "eclipse_ground": AimedSun(1, 0.6),
    "eclipse_total": AimedSun(1),
    "eclipse_orbit": AimedSun(1),
    "lunar_eclipse": AimedSun(0, 0.8),
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
        # Eclipses (SUNS): looking at the moon (a little above the middle
        # of the view, the horizon in it), and their shadows from orbit.
        heading, pitch = _look_at(demo, ref_lat, ref_lon, 1500.0, moon.position)
        eclipse = (ref_lat, ref_lon, 1500.0, heading, pitch - 8.0, 0, 0)
        poses["eclipse_ground"] = poses["eclipse_total"] = eclipse
        under = moon.position / np.linalg.norm(moon.position)
        lat, lon = (float(v[0]) for v in latlon(under[None, :]))
        poses["eclipse_orbit"] = (lat, lon, 3.0 * radius, 0.0, -89.9, 0, 0)
        poses["lunar_eclipse"] = poses["moon_orbit"]
    if demo.ring is not None:
        poses["ring_ground"] = (ref_lat, ref_lon, 1500.0, 180.0, 12.0, 0, 0)
        poses["ring_orbit"] = (35.0, lon0, 6.0 * radius, 0.0, -89.9, 0, 0)
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


def _look_at(
    demo: TerrainDemo, lat: float, lon: float, altitude: float, target: np.ndarray
) -> tuple[float, float]:
    """:returns: (heading, pitch) from altitude over (lat, lon) of the planet
    toward a point of the world frame (degrees, as :func:`_poses`')"""
    frame = TangentFrame.at(lat, lon)
    eye = demo.home.position + frame.up * (demo.radius + altitude)
    forward = (target - eye) / np.linalg.norm(target - eye)
    heading = math.degrees(math.atan2(-(forward @ frame.east), forward @ frame.north))
    return heading, math.degrees(math.asin(float(forward @ frame.up)))


def _aim_sun(demo: TerrainDemo, sun: AimedSun) -> None:
    """Point the demo's sun as *sun* says, from the point under the camera
    (once it is placed)."""
    relative = demo.cam_world - demo.body.position
    up = relative / np.linalg.norm(relative)
    ground = demo.body.position + up * demo.body.radius
    toward = demo.bodies[sun.body].position - ground
    toward /= np.linalg.norm(toward)
    lat, lon = latlon(up[None, :])
    side = TangentFrame.at(float(lat[0]), float(lon[0])).north
    side = side - toward * float(side @ toward)
    side /= np.linalg.norm(side)
    angle = math.radians(sun.offset)
    demo._sun = math.cos(angle) * toward + math.sin(angle) * side


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
        busy = (
            demo.executor.pending
            or any(body.terrain.stats["wanted"] for body in demo.bodies)
            or (demo.ring is not None and demo.ring.loading)
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
        "--cloud-debug", type=int, default=0, help="SphericalClouds.set_debug bits"
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
        ring=not args.no_ring,
        cloud_debug=args.cloud_debug,
        ocean_debug=args.ocean_debug,
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
        sun = SUNS.get(name, (210.0, 35.0))
        if isinstance(sun, AimedSun):
            _aim_sun(demo, sun)
        else:
            demo.set_local_sun(*sun)
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
