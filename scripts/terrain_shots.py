"""
Render the terrain demo from fixed poses to PNG: a visual regression check.

The shaders cannot be validated headless (see tests/test_terrain_mesh.py), so
this drives the real demo: for each pose it waits until the terrain has
finished refining (or a timeout), then saves a screenshot::

    python scripts/terrain_shots.py [--out DIR] [--radius METRES]
        [--pose NAME ...] [--timeout SECONDS]

Every pose is lit by the same local sun (azimuth 210, elevation 35).
Poses: ground (the level's start), coast (2 km up over the reference point),
high (20 km up), swell / sea_surface (either side of the ocean handover at
15 km: the swell ocean, then the terrain's own flat sea), stratosphere (200 km
up), orbit (3 radii up, looking down),
corner (over a cube corner, where three faces meet), lod (the LOD debug view
from 20 km).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from panda3d.core import Filename, loadPrcFileData

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import TerrainDemo  # noqa: E402

from world_builder.terrain.chart import latlon  # noqa: E402
from world_builder.terrain.level import DEMO_LEVEL  # noqa: E402

CORNER = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)


def _poses(demo: TerrainDemo) -> dict:
    """:returns: name -> (lat, lon, altitude, heading, pitch, debug view)"""
    start = demo.start_pose or {}
    lat0 = start.get("lat", demo.terrain.spec.reference_lat)
    lon0 = start.get("lon", demo.terrain.spec.reference_lon)
    ref_lat, ref_lon = demo.terrain.spec.reference_lat, demo.terrain.spec.reference_lon
    corner_lat, corner_lon = (float(v[0]) for v in latlon(CORNER[None, :]))
    radius = demo.radius
    return {
        "ground": (
            lat0,
            lon0,
            start.get("alt", 800.0),
            start.get("heading", 0.0),
            start.get("pitch", 0.0),
            0,
        ),
        "coast": (ref_lat, ref_lon, 2000.0, 0.0, -10.0, 0),
        "high": (lat0, lon0, 20000.0, 0.0, -25.0, 0),
        "swell": (ref_lat, ref_lon, 13000.0, 0.0, -20.0, 0),
        "sea_surface": (ref_lat, ref_lon, 17000.0, 0.0, -20.0, 0),
        "stratosphere": (lat0, lon0, 200e3, 0.0, -45.0, 0),
        "orbit": (lat0, lon0, 3.0 * radius, 0.0, -89.9, 0),
        "corner": (corner_lat, corner_lon, 3000.0, 30.0, -30.0, 0),
        "lod": (lat0, lon0, 20000.0, 0.0, -25.0, 1),
    }


def _settle(demo: TerrainDemo, timeout: float) -> bool:
    """Step frames until nothing is wanted or generating for a while."""
    start = time.perf_counter()
    quiet = 0
    while time.perf_counter() - start < timeout:
        demo.taskMgr.step()
        stats = demo.terrain.stats
        quiet = quiet + 1 if stats["wanted"] == 0 and stats["pending"] == 0 else 0
        if quiet >= 10:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("terrain_shots"))
    parser.add_argument("--level", default=str(DEMO_LEVEL))
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--pose", nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--size", default="1280 720")
    args = parser.parse_args()
    loadPrcFileData(
        "",
        "\n".join(
            [
                "window-title Terrain shots",
                f"win-size {args.size}",
                "audio-library-name null",
                "framebuffer-multisample 1",
                "multisamples 4",
                "depth-bits 24",
                "show-frame-rate-meter false",
            ]
        ),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    demo = TerrainDemo(args.level, "process", args.radius)
    demo.setFrameRateMeter(False)
    poses = _poses(demo)
    for name in args.pose or list(poses):
        lat, lon, alt, heading, pitch, view = poses[name]
        demo._debug = view
        demo.terrain.set_debug_mode(view)
        demo._place_camera(lat, lon, alt, heading, pitch)
        demo.set_local_sun(210.0, 35.0)  # the same light everywhere
        settled = _settle(demo, args.timeout)
        for _ in range(3):  # let the last uploads reach the screen
            demo.taskMgr.step()
        path = args.out / f"{name}.png"
        demo.win.save_screenshot(Filename.from_os_specific(str(path)))
        s = demo.terrain.stats
        print(
            f"{name:<13} {'settled' if settled else 'TIMEOUT'}  {s['shown']} chunks, "
            f"finest {s['finest']}, LOD {s['select_ms']:.1f} ms -> {path}"
        )
    demo.terrain.destroy()


if __name__ == "__main__":
    main()
