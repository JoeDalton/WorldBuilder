"""
Where a frame's time goes, per part, at fixed poses of the terrain demo (the
reference profile, RENDERING_PLAN.md part A; results in
docs/source/frame_profile.md)::

    python scripts/frame_profile.py [--pose NAME ...] [--size "W H"]
        [--quality LEVEL] [--genera] [--ring] [--frames N] [--offscreen]

At each pose (terrain_shots.py's), settled, GPU-synced frames (a one-pixel
read back after each step, as terrain_shots.py times them):

* **frame** the median frame, everything shown, and its p95;
* **step** the median of ``taskMgr.step()`` alone, before the GPU sync: the
  CPU's part (cull, draw and submission, the demo's update). Close to the
  frame: CPU-bound there;
* each part's cost: the median frame with it shown less with it off:
  **terrain** (the terrain's roots, the near sea with them), **refl** (the
  sea's reflection pass), **sky**, **clouds** (the low clouds and their
  passes), **cirrus**, **models** (the test balls), **skyview** (the
  sky-view table and its bake against the sky's march: negative where it
  saves), **aerial** (the aerial-perspective cells and their bake against
  the terrain's, sea's and models' march: the same), **godrays** (the
  clouds' and the mountains' shadows in that bake,
  ``AtmosphereSettings.god_rays``);
* **atmo** the per-pixel atmosphere's share: the frame at the preset's
  samples less at 2 (``AtmosphereSettings.steps``), scaled to all of them
  (``* steps / (steps - 2)``): a linear estimate of the whole march.

A part's cost includes its CPU's when the frame is CPU-bound. Every number
is a median of --frames frames; two runs differ by 0.3-0.5 ms. Frames faster
than the display's refresh are held to it in a window even with vsync off
(5.53 ms at 180 Hz, every part 0): use --offscreen there (LOW).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    TerrainDemo,
    add_planet_arguments,
    add_world_arguments,
    describe_window,
    settings_from_args,
    window_config,
)

from terrain_shots import SUNS, AimedSun, _aim_sun, _poses, _settle  # noqa: E402
from world_builder.terrain.level import DEMO_LEVEL  # noqa: E402

#: The poses profiled by default.
POSES = (
    "ground",
    "coast",
    "ground_up",
    "in_clouds",
    "clouds_low",
    "sunset",
    "high",
    "ball_noon",
    "orbit",
)


def _time(demo: TerrainDemo, probe, frames: int) -> tuple[float, float, float]:
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


def _parts(demo: TerrainDemo) -> dict:
    """Each part as (off, on) callables."""
    terrains = [body.terrain.root for body in demo.bodies]
    parts = {
        "terrain": (
            lambda: [t.hide() for t in terrains],
            lambda: [t.show() for t in terrains],
        ),
        "sky": (demo.sky.node.hide, demo.sky.node.show),
    }
    atmospheres = demo.atmospheres
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
    view = getattr(demo.sky, "view", None)
    if view is not None:
        # the sky-view table against the sky's march (negative: it saves)
        parts["skyview"] = (
            lambda: view.set_active(False),
            lambda: view.set_active(True),
        )
    ocean = getattr(demo, "ocean", None)
    if ocean is not None:
        parts["refl"] = (
            lambda: ocean.refl_buffer.set_active(False),
            lambda: ocean.refl_buffer.set_active(bool(ocean._showing)),
        )
    if demo.clouds is not None:
        parts["clouds"] = (
            lambda: demo.clouds.set_visible(False),
            lambda: demo.clouds.set_visible(True),
        )
        if demo.clouds.sheet is not None:
            sheet = demo.clouds.sheet.root
            parts["cirrus"] = (sheet.hide, sheet.show)
    balls = [node for node, _ in demo._test_balls]
    parts["models"] = (
        lambda: [b.hide() for b in balls],
        lambda: [b.show() for b in balls],
    )
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", default=str(DEMO_LEVEL), help="level file")
    add_planet_arguments(parser)
    parser.add_argument("--pose", nargs="*", default=None, help="default: POSES")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--size", default="2560 1440", help='"width height"')
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--frames", type=int, default=80)
    add_world_arguments(parser)
    args = parser.parse_args()
    window_config("Frame profile", args.size, args.offscreen, vsync=False)
    settings = settings_from_args(args)
    demo = TerrainDemo(
        args.level,
        "process",
        None,
        settings=settings,
        clouds=not args.no_clouds,
        genera=args.genera,
        ring=args.ring,
        preset=args.planet,
        seed=args.seed,
    )
    demo.setFrameRateMeter(False)
    print(describe_window(demo))
    win = demo.win
    probe = win.make_display_region(0, 1 / win.get_x_size(), 0, 1 / win.get_y_size())
    poses = _poses(demo)
    atmo = settings.atmosphere
    columns = ["frame", "p95", "step", "terrain", "refl", "sky", "clouds", "cirrus"]
    columns += ["models", "skyview", "aerial", "godrays", "atmo"]
    print(f"{'pose':<12} " + " ".join(f"{c:>7}" for c in columns))
    for name in args.pose or list(POSES):
        lat, lon, alt, heading, pitch, view, body = poses[name]
        demo.set_debug_view(view)
        demo._place_camera(lat, lon, alt, heading, pitch, demo.bodies[body])
        sun = SUNS.get(name, (210.0, 35.0))
        if isinstance(sun, AimedSun):
            _aim_sun(demo, sun)
        else:
            demo.set_local_sun(*sun)
        settled = _settle(demo, args.timeout)
        frame, p95, step = _time(demo, probe, args.frames)
        row = {"frame": frame, "p95": p95, "step": step}
        for part, (off, on) in _parts(demo).items():
            off()
            row[part] = frame - _time(demo, probe, args.frames)[0]
            on()
        if atmo.steps > 2:
            demo.atmospheres.set_settings(replace(atmo, steps=2))
            fewer = _time(demo, probe, args.frames)[0]
            demo.atmospheres.set_settings(atmo)
            row["atmo"] = (frame - fewer) * atmo.steps / (atmo.steps - 2)
        cells = " ".join(
            f"{row[c]:>7.2f}" if c in row else f"{'-':>7}" for c in columns
        )
        print(f"{name:<12} {cells}{'' if settled else '  (not settled)'}")
        sys.stdout.flush()
    demo.destroy_terrain()


if __name__ == "__main__":
    main()
