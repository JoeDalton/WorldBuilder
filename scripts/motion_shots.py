"""
The clouds in motion against the clouds converged: the demo flown from a
pose, forward or backward, with its own keys and speed rule (so a frame's
move is the user's), frames saved along the way; then each saved position
revisited, held still until the clouds' history converges, and shot again
(with the clouds and without). What the motion costs the clouds is the moving
frame against its converged twin, over the clouds (where the converged shot
with clouds differs from the one without) and over the whole image::

    python scripts/motion_shots.py --at LAT LON ALT HEADING PITCH
        [--backward] [--boost] [--frames N] [--every K] [--settle F]
        [--out DIR] [--size "W H"] [--offscreen] [--genera] [--cloud-debug B]

Prints per sample frame: the altitude, the mean error over the clouds and
over the image (8-bit levels), the share of cloud pixels lost (in the
converged shot, missing from the moving one) and found (the reverse); then
the means. Saves each sample's moving / converged / no-cloud shots side by
side in --out.

Made for the clouds' refresh (RENDERING_PLAN.md, "clouds in motion"): the
stratus/mountain edge sliding and the clouds vanishing in a fast descent,
both seen moving forward only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from panda3d.core import LQuaternionf as LQuat

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    TerrainDemo,
    add_planet_arguments,
    add_world_arguments,
    describe_window,
    settings_from_args,
    start_sun,
    window_config,
)

from terrain_shots import _settle  # noqa: E402
from world_builder.terrain.level import DEMO_LEVEL  # noqa: E402


class _FixedClock:
    """The demo's clock during the flight: a fixed frame time, so each
    frame's move is the same whatever reading frames back costs."""

    def __init__(self, fps: float) -> None:
        self.dt = 1.0 / fps
        self.time = 0.0

    def get_dt(self) -> float:
        return self.dt

    def get_frame_time(self) -> float:
        self.time += self.dt
        return self.time


#: A cloud pixel this far off its converged twin (8-bit levels) is a speck.
SPECK_LEVELS = 10.0
#: A pixel counts as cloud where the converged shots with and without the
#: clouds differ by more than this (8-bit levels, the channels' mean).
CLOUD_LEVELS = 6.0


def _grab(demo: TerrainDemo) -> np.ndarray:
    tex = demo.win.get_screenshot()
    w, h = tex.get_x_size(), tex.get_y_size()
    ram = np.frombuffer(bytes(tex.get_ram_image_as("RGB")), np.uint8)
    return ram.reshape(h, w, 3)[::-1].astype(np.float32)


def _detail(image: np.ndarray) -> np.ndarray:
    """The fine detail per pixel: luminance against its 5 x 5 mean (grain)."""
    lum = image.mean(-1)
    pad = np.pad(lum, 2, mode="edge")
    box = (
        sum(
            pad[2 + dy : 2 + dy + lum.shape[0], 2 + dx : 2 + dx + lum.shape[1]]
            for dy in range(-2, 3)
            for dx in range(-2, 3)
        )
        / 25.0
    )
    return np.abs(lum - box)


def _save(path: Path, images: list[np.ndarray]) -> None:
    """Side by side, as a binary PPM (no image library in the venv)."""
    row = np.concatenate([np.clip(i, 0, 255).astype(np.uint8) for i in images], 1)
    h, w = row.shape[:2]
    path.write_bytes(b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(row).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_planet_arguments(parser)
    add_world_arguments(parser)
    parser.set_defaults(frame_cap=0)
    parser.add_argument(
        "--at",
        nargs=5,
        type=float,
        required=True,
        metavar=("LAT", "LON", "ALT", "HEADING", "PITCH"),
    )
    parser.add_argument("--backward", action="store_true", help="fly backward")
    parser.add_argument(
        "--vertical",
        choices=("up", "down"),
        default=None,
        help="fly straight up or down (the demo's W / X) instead",
    )
    parser.add_argument(
        "--flicker-map",
        default=None,
        help="save the flight's mean flicker per pixel as this PPM (bright: blinking)",
    )
    parser.add_argument("--boost", action="store_true", help="shift held (x5)")
    parser.add_argument("--frames", type=int, default=240, help="frames flown")
    parser.add_argument("--every", type=int, default=24, help="a sample each K")
    parser.add_argument(
        "--settle", type=int, default=90, help="still frames to converge a sample"
    )
    parser.add_argument("--out", default="motion_shots", help="image directory")
    parser.add_argument("--size", default="2560 1440", help='"width height"')
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--cloud-debug", type=int, default=0)
    parser.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="the flight's frame rate: each frame moves as one of this many "
        "a second does (the demo's speed rule, a fixed frame time)",
    )
    parser.add_argument(
        "--flicker",
        action="store_true",
        help="read every frame of the flight back: its flicker (the second "
        "difference in time, which smooth motion cancels and blinking does not)",
    )
    args = parser.parse_args()
    window_config("Motion shots", args.size, args.offscreen, vsync=False)
    demo = TerrainDemo(
        str(DEMO_LEVEL),
        "process",
        None,
        settings=settings_from_args(args),
        clouds=not args.no_clouds,
        genera=args.genera,
        ring=args.ring,
        preset=args.planet,
        seed=args.seed,
        start=args.start,
        cloud_debug=args.cloud_debug,
    )
    demo.setFrameRateMeter(False)
    demo._update_hud = lambda: None
    print(describe_window(demo))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    demo._place_camera(*args.at)
    demo._sun = start_sun(demo.terrain.spec, demo.start_pose)
    _settle(demo, 90.0)
    for _ in range(args.settle):
        demo.taskMgr.step()

    # The flight: the demo's own keys (its speed rule, its dt).
    key = "arrow_down" if args.backward else "arrow_up"
    if args.vertical:
        key = "w" if args.vertical == "up" else "x"
    demo._keys[key] = True
    demo._keys["shift"] = bool(args.boost)
    clock = demo._clock
    demo._clock = _FixedClock(args.fps)
    samples = []
    flicker, previous = [], []
    for frame in range(args.frames):
        demo.taskMgr.step()
        if args.flicker:
            now = _grab(demo)
            if (
                len(previous) == 2
                and previous[0].shape == now.shape == previous[1].shape
            ):
                flicker.append(np.abs(now - 2 * previous[1] + previous[0]).mean(-1))
            previous = [previous[-1], now] if previous else [now]
        if frame % args.every == args.every - 1:
            samples.append(
                (demo.cam_world.copy(), LQuat(demo._orientation), _grab(demo))
            )
    for k in list(demo._keys):
        demo._keys[k] = False
    demo._clock = clock
    if flicker:
        stack = np.stack(flicker)
        print(
            f"flicker over {len(flicker)} frames: mean {stack.mean():.2f}, "
            f"p99 {np.percentile(stack.mean(0), 99):.1f} (per pixel, levels)"
        )
        if args.flicker_map:
            level = np.clip(stack.mean(0) * 8.0, 0, 255)
            _save(Path(args.flicker_map), [np.repeat(level[..., None], 3, -1)])
        np.save(out / "flicker_stack_last.npy", np.stack(previous).astype(np.uint8))

    # Each sample again, held still: the converged clouds, then none.
    rows = []
    for index, (position, orientation, moving) in enumerate(samples):
        demo.cam_world = position
        demo._orientation = orientation
        for _ in range(args.settle):
            demo.taskMgr.step()
        converged = _grab(demo)
        demo.clouds.set_visible(False)
        demo.taskMgr.step()
        demo.taskMgr.step()
        bare = _grab(demo)
        demo.clouds.set_visible(True)
        # A window may change by a pixel between shots (its manager).
        h = min(i.shape[0] for i in (moving, converged, bare))
        w = min(i.shape[1] for i in (moving, converged, bare))
        moving, converged, bare = (i[:h, :w] for i in (moving, converged, bare))
        cloud = np.abs(converged - bare).mean(-1) > CLOUD_LEVELS
        moved = np.abs(moving - bare).mean(-1) > CLOUD_LEVELS
        error = np.abs(moving - converged).mean(-1)
        # Grain: the moving frame's fine detail where the converged clouds
        # are smooth (detail under a level), against the converged's there.
        detail_moving, detail_converged = _detail(moving), _detail(converged)
        smooth = cloud & (detail_converged < 1.0)
        grain = 1.0
        if smooth.any():
            smooth_converged = max(float(detail_converged[smooth].mean()), 1e-6)
            grain = float(detail_moving[smooth].mean()) / smooth_converged
        altitude = float(np.linalg.norm(demo._relative())) - demo.body.radius
        rows.append(
            (
                altitude,
                float(error[cloud].mean()) if cloud.any() else 0.0,
                float(error.mean()),
                float((cloud & ~moved).sum() / max(cloud.sum(), 1)),
                float((moved & ~cloud).sum() / cloud.sum()) if cloud.any() else 0.0,
                float(cloud.mean()),
                grain,
                float((error[cloud] > SPECK_LEVELS).mean()) if cloud.any() else 0.0,
            )
        )
        _save(out / f"sample_{index:02d}.ppm", [moving, converged, bare])
    print(
        f"{'sample':>6} {'alt m':>9} {'err cloud':>9} {'err all':>8} "
        f"{'lost %':>7} {'found %':>8} {'cloud %':>8} {'grain':>6} {'speck %':>8}"
    )
    for index, (alt, ec, ea, lost, found, share, grain, speck) in enumerate(rows):
        print(
            f"{index:6d} {alt:9.0f} {ec:9.2f} {ea:8.2f} {lost * 100:7.1f} "
            f"{found * 100:8.1f} {share * 100:8.1f} {grain:6.2f} {speck * 100:8.2f}"
        )
    r = np.array(rows)
    print(
        f"{'mean':>6} {'':9} {r[:, 1].mean():9.2f} {r[:, 2].mean():8.2f} "
        f"{r[:, 3].mean() * 100:7.1f} {r[:, 4].mean() * 100:8.1f} {'':8} "
        f"{r[:, 6].mean():6.2f} {r[:, 7].mean() * 100:8.2f}"
    )
    demo.destroy_terrain()


if __name__ == "__main__":
    main()
