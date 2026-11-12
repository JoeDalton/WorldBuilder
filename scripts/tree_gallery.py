"""
The tree assets (world_builder.datafiles) on the ground of the tiny test
planet, for a look at them: the terrain demo (its controls, its light, the
world's model shader) with every asset set out in rows near the start, each
labelled with its name, kind and triangles.

Run it from an environment where world_builder is importable::

    python scripts/tree_gallery.py [--quality L] [--<part>-quality L]
        [--vegetation] [--spacing M] [--shot PNG]

    --vegetation     the terrain's own vegetation too (off by
                     default: the assets alone)
    --spacing        metres between the trees (default 22)
    --shot           render offscreen, save a PNG of the start view, quit

The rows, from the camera's side: rocks, plants, birches, the near conifers
(trunk and branch cards), the background conifers (crossed cards). The
camera starts south of them, looking north. END or R comes back there; the
demo's controls otherwise (I / U and J / K move the sun).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from panda3d.core import TextNode

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    TerrainDemo,
    _quat_from_axes,
    add_world_arguments,
    settings_from_args,
    window_config,
)
from world_builder.datafiles import tree_assets  # noqa: E402
from world_builder.terrain.chart import TangentFrame, dirs_from_latlon  # noqa: E402
from world_builder.terrain.level import TINY_LEVEL  # noqa: E402
from world_builder.terrain.planet import Planet  # noqa: E402

#: The rows, nearest the camera first: (kinds, metres north of the first).
ROWS = (
    (("rock",), 0.0),
    (("plant",), 12.0),
    (("tree",), 30.0),
    (("tree_near",), 55.0),
    (("tree_far",), 80.0),
)
#: Where the gallery stands: on the tiny planet's grassland, flat and open.
SITE = (33.6, -146.2)
#: The camera: this far south of the first row and above the ground, metres.
CAMERA_BACK = 45.0
CAMERA_UP = 6.0
#: The models sink this far into the ground (the drawn ground is the level's
#: grid, a little off the exact one), metres.
SINK = 0.3


class TreeGallery:
    """The assets set out on the ground, placed and lit with the demo's test
    balls (their positions each frame, their lights where they stand), and
    their labels above them."""

    def __init__(self, demo: TerrainDemo, spacing: float):
        self.demo = demo
        planet = Planet(demo.terrain.spec)
        frame = TangentFrame.at(*SITE)
        self.frame = frame
        radius = demo.terrain.radius
        centre = dirs_from_latlon(*SITE)[0]
        self.labels: list[tuple] = []
        assets = tree_assets()
        for kinds, north in ROWS:
            row = [a for a in assets if a.kind in kinds]
            gap = spacing if kinds[0].startswith("tree") else spacing / 3.0
            for k, asset in enumerate(row):
                east = (k - 0.5 * (len(row) - 1)) * gap
                # The point on the sphere, then its ground.
                d = centre + (frame.east * east + frame.north * north) / radius
                d /= np.linalg.norm(d)
                ground, _, _ = planet.sample_dirs(d[None], None, True)
                ground = max(float(ground[0]), 0.0) - SINK
                up = d
                e = np.cross(frame.north, up)
                e /= np.linalg.norm(e)
                n = np.cross(up, e)
                node = asset.load()
                node.reparent_to(demo.render)
                node.set_quat(_quat_from_axes(e, n, up))
                point = d * (radius + ground)
                demo._test_balls.append((node, point))
                demo.tracked.track(node, cast=True, shader=demo._model_shader)
                text = TextNode(asset.name)
                text.set_text(f"{asset.name}\n{asset.kind}, {asset.triangles} tris")
                text.set_align(TextNode.ACenter)
                text.set_text_color(1.0, 1.0, 1.0, 1.0)
                text.set_shadow(0.05, 0.05)
                label = demo.render.attach_new_node(text)
                label.set_scale(0.8 if kinds[0].startswith("tree") else 0.35)
                label.set_billboard_point_eye()
                label.set_light_off(1)
                label.set_shader_off(1)
                top = point + up * (asset.size[2] + 1.0)
                self.labels.append((label, top))
        # The start: south of the rows, looking north at them.
        start = centre - frame.north * CAMERA_BACK / radius
        start /= np.linalg.norm(start)
        ground, _, _ = planet.sample_dirs(start[None], None, True)
        lat = math.degrees(math.asin(start[2]))
        lon = math.degrees(math.atan2(start[1], start[0]))
        demo.start_pose = {
            "lat": lat,
            "lon": lon,
            "alt": max(float(ground[0]), 0.0) + CAMERA_UP,
            "heading": 0.0,
            "pitch": 8.0,
        }
        demo._reset_camera()
        demo.taskMgr.add(self._place_labels, "tree_gallery_labels", sort=1)

    def _place_labels(self, task):
        demo = self.demo
        for label, point in self.labels:
            label.set_pos(*(demo.home.position + point - demo.cam_world))
        return task.cont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", default="1600 900", help='"W H"')
    parser.add_argument("--vegetation", action="store_true")
    parser.add_argument("--spacing", type=float, default=22.0)
    parser.add_argument(
        "--shot",
        type=Path,
        default=None,
        help="render offscreen, save this PNG once the ground is loaded, quit",
    )
    add_world_arguments(parser)
    args = parser.parse_args()
    window_config(
        "Tree gallery",
        args.size,
        offscreen=args.shot is not None,
        vsync=args.shot is None,
    )
    settings = settings_from_args(args)
    if not args.vegetation:
        settings = settings.with_value("vegetation", "enabled", False)
    demo = TerrainDemo(
        str(TINY_LEVEL), "process", None, settings=settings, clouds=False
    )
    TreeGallery(demo, args.spacing)
    if args.shot is not None:
        from panda3d.core import Filename

        from terrain_shots import _settle

        _settle(demo, 120.0)
        for _ in range(5):
            demo.taskMgr.step()
        demo.win.save_screenshot(Filename.from_os_specific(str(args.shot)))
        demo.destroy_terrain()
        return
    demo.run()


if __name__ == "__main__":
    main()
