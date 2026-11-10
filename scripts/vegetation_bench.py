"""
The vegetation's benchmark and visual check, on the tiny test planet.

Renders the tiny planet (``PRESETS["tiny"]``, 20 km, no clouds) from fixed
poses over its forest, grassland and desert, and times each settled view
as ``terrain_shots.py`` does (every timed frame waits for the GPU, the
median is reported). Saves a PNG per pose, and optionally the timings as
JSON, or compares them with a saved run::

    python scripts/vegetation_bench.py [--out DIR] [--size W H] [--quality L]
        [--<part>-quality L] [--set SECTION.FIELD=VALUE ...] [--pose NAME ...]
        [--frames N] [--save FILE] [--baseline FILE] [--offscreen]

    --set            a setting over the preset, e.g. terrain.min_level=0
                     (repeatable; the value is a Python literal)
    --save           write {pose: median ms} as JSON
    --baseline       a --save file: print each pose's difference to it
    --ab             each pose timed with the vegetation off and on in turn
                     (--rounds of each, --frames in all): the difference

Poses (altitudes above the ground under the camera; the sun at azimuth 210,
35 degrees up unless said): forest_edge (2 m up in the grassland, looking
east into the forest 230 m away: the camera inside the canopy's shell),
forest_low (150 m over the forest, 10 degrees down: grazing rays through the
canopy), mixed_mid (800 m up, 30 degrees down, looking south over desert,
grassland and forest), high (4 km up, 60 degrees down: the far look only),
desert (20 m over the desert's sparse bushes), forest_dusk (forest_low with
the sun 6 degrees up behind the camera).

Measure with the machine idle: other load moves a frame by 1-2 ms (see
docs/source/performance_backlog.md).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import sys
import time
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
from terrain_shots import _settle  # noqa: E402
from world_builder.terrain.chart import dirs_from_latlon  # noqa: E402
from world_builder.terrain.level import TINY_LEVEL  # noqa: E402
from world_builder.terrain.planet import Planet  # noqa: E402

#: name -> (lat, lon, metres above the ground, heading, pitch, (sun azimuth,
#: elevation)); heading counter-clockwise from north (-90 = east)
POSES = {
    "forest_edge": (33.31, -147.0, 2.0, -90.0, 2.0, (210.0, 35.0)),
    "forest_low": (32.0, -147.6, 150.0, -90.0, -10.0, (210.0, 35.0)),
    "mixed_mid": (34.6, -147.4, 800.0, 160.0, -30.0, (210.0, 35.0)),
    "high": (33.3, -147.0, 4000.0, 180.0, -60.0, (210.0, 35.0)),
    "desert": (36.2, -149.6, 20.0, 0.0, -5.0, (210.0, 35.0)),
    "forest_dusk": (32.0, -147.6, 150.0, -90.0, -10.0, (90.0, 6.0)),
}


def _parse_set(text: str) -> tuple[str, str, object]:
    key, _, value = text.partition("=")
    section, _, name = key.partition(".")
    if not name or not value:
        raise argparse.ArgumentTypeError(f"expected SECTION.FIELD=VALUE: {text}")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    return section, name, parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("vegetation_shots"))
    parser.add_argument("--size", default="2560 1440", help='"W H"')
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--pose", nargs="+", choices=list(POSES), default=None)
    parser.add_argument("--set", type=_parse_set, action="append", default=[])
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument(
        "--ab",
        action="store_true",
        help="time each pose with the vegetation off and on in turn, and "
        "print the difference (robust to the machine's drift)",
    )
    parser.add_argument("--rounds", type=int, default=4, help="--ab's rounds")
    add_world_arguments(parser)
    args = parser.parse_args()

    window_config("Vegetation bench", args.size, args.offscreen, vsync=False)
    args.out.mkdir(parents=True, exist_ok=True)
    settings = settings_from_args(args)
    for section, name, value in args.set:
        settings = settings.with_value(section, name, value)
    demo = TerrainDemo(
        str(TINY_LEVEL), "process", None, settings=settings, clouds=False
    )
    demo.setFrameRateMeter(False)
    print(describe_window(demo), f"settings {settings.to_dict()}")
    win = demo.win
    probe = win.make_display_region(0, 1 / win.get_x_size(), 0, 1 / win.get_y_size())
    planet = Planet(demo.terrain.spec)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else {}
    results: dict[str, float] = {}
    for name in args.pose or list(POSES):
        lat, lon, above, heading, pitch, sun = POSES[name]
        ground, _, _ = planet.sample_dirs(dirs_from_latlon(lat, lon), None, True)
        alt = max(float(ground[0]), 0.0) + above
        demo._place_camera(lat, lon, alt, heading, pitch)
        demo.set_local_sun(*sun)
        settled = _settle(demo, args.timeout)
        canopy = demo.terrain.canopy
        on = canopy.settings
        off = dataclasses.replace(on, enabled=False)

        def timed(veg) -> np.ndarray:
            canopy.set_settings(veg)
            demo.terrain._select(demo.terrain._camera)  # the shells now
            while canopy.waiting:
                demo.terrain._select(demo.terrain._camera)
            for _ in range(3):
                demo.taskMgr.step()
            times = []
            for _ in range(
                args.frames // (2 * args.rounds) if args.ab else args.frames
            ):
                start = time.perf_counter()
                demo.taskMgr.step()
                probe.get_screenshot()  # waits for the GPU
                times.append(time.perf_counter() - start)
            return np.array(times)

        if args.ab:
            # Off and on in turn, in rounds: the machine's drift cancels.
            with_veg, without = [], []
            for _ in range(args.rounds):
                without.append(timed(off))
                with_veg.append(timed(on))
            times = np.concatenate(with_veg)
            reference = float(np.median(np.concatenate(without))) * 1e3
        else:
            times = timed(on)
        median = float(np.median(times)) * 1e3
        p90 = float(np.percentile(times, 90)) * 1e3
        results[name] = median
        path = args.out / f"{name}.png"
        demo.win.save_screenshot(Filename.from_os_specific(str(path)))
        s = demo.terrain.stats
        c = canopy.stats
        diff = ""
        if args.ab:
            diff = f"  {median - reference:+.2f} ms vs off ({reference:.2f})"
        elif name in baseline:
            diff = f"  {median - baseline[name]:+.2f} ms vs baseline"
        print(
            f"{name:<12} {'settled' if settled else 'TIMEOUT'}  {s['shown']} chunks, "
            f"finest {s['finest']}, {c['shells']} shells ({c['inside']} inside, "
            f"{c['update_ms']:.1f} ms), median {median:.2f} ms"
            f"({1e3 / median:.0f} fps), "
            f"p90 {p90:.2f} ms{diff} -> {path}"
        )
    if args.save is not None:
        args.save.write_text(json.dumps(results, indent=1))
    demo.destroy_terrain()


if __name__ == "__main__":
    main()
