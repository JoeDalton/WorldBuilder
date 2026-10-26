"""
Fly through an asteroid field or a ring (``world_builder.asteroids``), drawn
by a :class:`~world_builder.world.World` of open space as a game would have
it: split depth cameras, the sky, the render origin kept at the camera::

    python scripts/demo_asteroids.py [--preset NAME] [--quality LEVEL]
        [--backend process|thread|inline] [--size "1600 900"]

    --preset   the field (spec.PRESETS; default cube, missions 1-2's 15 km;
               ring: a planet ring 220,000 km across)
    --quality  the world's settings preset at start (default high)
    --backend  where the rock prototypes are built (default process)

Controls (AZERTY, as demo_terrain.py): arrows forward / back and slide, W / X
up / down, Z / S pitch, Q / D roll, A / E yaw, Shift 5x, PgUp / PgDn speed.
P pauses the world's clock (the rocks stop turning, the camera still flies),
F1 cycles the quality preset, End returns to the start, Home goes to the
preset's far view (a ring seen from afar) and back.

The rocks turn with the demo's own clock, never Panda's: what a game's
pausable clock does. The default backend builds prototypes in spawned worker
processes, which re-import this script: keep the ``__main__`` guard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import ClockObject, TextNode, Vec3

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import describe_window, window_config  # noqa: E402

from world_builder.asteroids.spec import PRESETS  # noqa: E402
from world_builder.depth import SplitDepthCameras  # noqa: E402
from world_builder.quality import QualityLevel  # noqa: E402
from world_builder.settings import WorldSettings  # noqa: E402
from world_builder.world import FieldLook, SkyLook, SunLook, World  # noqa: E402

#: Per preset: where the camera starts (position, heading, pitch), the far
#: view (Home), and the speed, m/s
VIEWS = {
    "cube": (
        ((0.0, -9000.0, 500.0), 0.0, 0.0),
        ((0.0, -60_000.0, 8000.0), 0.0, -7.0),
        250.0,
    ),
    "icy_ball": (
        ((0.0, -16_000.0, 0.0), 0.0, 0.0),
        ((0.0, -80_000.0, 0.0), 0.0, 0.0),
        250.0,
    ),
    # inside the ring's dense band, looking along it; from afar, its whole
    "ring": (
        ((0.0, 1.01e8, 0.0), -90.0, 0.0),
        ((0.0, -2.6e8, 7.0e7), 0.0, -15.0),
        2000.0,
    ),
}
BOOST = 5.0
SUN = SunLook(direction=(-0.45, -0.85, 0.3), color=(2.6, 2.5, 2.35))
AMBIENT = (0.03, 0.03, 0.04)
HUD_PERIOD = 0.25

_ROTATIONS = (("s", "z", "p", 40.0), ("d", "q", "r", 60.0), ("a", "e", "h", 40.0))
_TRANSLATIONS = (
    ("arrow_up", "arrow_down", 1),
    ("arrow_right", "arrow_left", 0),
    ("w", "x", 2),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", choices=list(PRESETS), default="cube")
    parser.add_argument("--quality", default="high")
    parser.add_argument("--backend", default="process")
    parser.add_argument("--size", default="1600 900")
    args = parser.parse_args()
    window_config("asteroid field", args.size, vsync=False)
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    base.disable_mouse()
    base.set_background_color(0.0, 0.0, 0.0, 1)
    base.setFrameRateMeter(True)
    level = QualityLevel.parse(args.quality)
    start, far_view, speed = VIEWS.get(args.preset, VIEWS["cube"])
    state = {"t": 0.0, "paused": False, "speed": speed, "level": level, "far": False}
    world = World(
        base,
        bodies=[],
        sun=SUN,
        sky=SkyLook(),
        settings=WorldSettings.preset(level),
        clock=lambda: state["t"],
        backend=args.backend,
        space_ambient=AMBIENT,
        fields=[FieldLook(args.preset, PRESETS[args.preset])],
    )
    field = world.field(args.preset)
    depth = SplitDepthCameras(base)
    # The camera stays at render's origin; its world position is float64.
    position = np.array(start[0], dtype=np.float64)

    def go(view):
        position[:] = view[0]
        base.camera.set_pos_hpr(0, 0, 0, view[1], view[2], 0)

    go(start)
    keys: set[str] = set()
    for key in {k for row in (*_ROTATIONS, *_TRANSLATIONS) for k in row[:2]} | {
        "shift"
    }:
        base.accept(key, keys.add, [key])
        base.accept(f"{key}-up", keys.discard, [key])

    def toggle_pause():
        state["paused"] = not state["paused"]

    def cycle_quality():
        levels = list(QualityLevel)
        state["level"] = levels[(levels.index(state["level"]) + 1) % len(levels)]
        world.set_settings(WorldSettings.preset(state["level"]))

    def toggle_far():
        state["far"] = not state["far"]
        go(far_view if state["far"] else start)

    base.accept("p", toggle_pause)
    base.accept("f1", cycle_quality)
    base.accept("page_up", lambda: state.update(speed=state["speed"] * 2.0))
    base.accept("page_down", lambda: state.update(speed=state["speed"] / 2.0))
    base.accept("end", lambda: go(start))
    base.accept("home", toggle_far)
    hud = OnscreenText(
        text="",
        pos=(-1.3, 0.9),
        scale=0.04,
        fg=(1, 1, 1, 1),
        align=TextNode.A_left,
        mayChange=True,
    )
    hud_timer = [0.0]

    def fly(task):
        dt = min(ClockObject.get_global_clock().get_dt(), 0.1)
        if not state["paused"]:
            state["t"] += dt
        cam = base.camera
        for plus, minus, axis, rate in _ROTATIONS:
            turn = (plus in keys) - (minus in keys)
            if turn:
                delta = Vec3(0, 0, 0)
                delta["hpr".index(axis)] = turn * rate * dt
                cam.set_hpr(cam, delta)
        move = Vec3(0, 0, 0)
        for plus, minus, axis in _TRANSLATIONS:
            move[axis] = (plus in keys) - (minus in keys)
        speed = state["speed"] * (BOOST if "shift" in keys else 1.0)
        step = cam.get_quat(base.render).xform(move * speed * dt)
        position[:] += (step.x, step.y, step.z)
        depth.update(*world.depth_ranges(position))
        world.update(position, far=depth.far)
        hud_timer[0] -= dt
        if hud_timer[0] <= 0.0:
            hud_timer[0] = HUD_PERIOD
            s = field.stats
            local = field.to_field(position)
            hud.setText(
                f"{args.preset}  {state['level'].label}  "
                f"{'loading' if field.loading else ''}"
                f"{'PAUSED' if state['paused'] else ''}\n"
                f"rocks {s.get('rocks', 0)}  drawn {s.get('drawn', 0)}  "
                f"smallest {s.get('min_pixels', 0):.1f} px  "
                f"update {s.get('update_ms', 0):.2f} ms  "
                f"speed {speed:.0f} m/s\n"
                f"axis {np.hypot(local[0], local[1]) / 1e3:,.0f} km  "
                f"height {local[2] / 1e3:,.1f} km"
            )
        return task.cont

    base.taskMgr.add(fly, "asteroid_demo")
    print(describe_window(base), args.preset, args.quality)
    try:
        base.run()
    finally:
        depth.destroy()
        world.destroy()


if __name__ == "__main__":
    main()
