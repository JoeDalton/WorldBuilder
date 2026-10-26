"""
Fly through an asteroid field (``world_builder.asteroids``)::

    python scripts/demo_asteroids.py [--preset NAME] [--quality LEVEL]
        [--backend process|thread|inline] [--size "1600 900"]

    --preset   the field (spec.PRESETS; default cube, missions 1-2's 15 km)
    --quality  its AsteroidSettings preset at start (default high)
    --backend  where the rock prototypes are built (default process)

Controls (AZERTY, as demo_terrain.py): arrows forward / back and slide, W / X
up / down, Z / S pitch, Q / D roll, A / E yaw, Shift 5x, PgUp / PgDn speed.
P pauses the field's clock (the rocks stop turning, the camera still flies),
F1 cycles the quality preset, End returns to the start.

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

from world_builder.asteroids.field import AsteroidField  # noqa: E402
from world_builder.asteroids.settings import AsteroidSettings  # noqa: E402
from world_builder.asteroids.spec import PRESETS  # noqa: E402
from world_builder.quality import QualityLevel  # noqa: E402

START = (0.0, -9000.0, 500.0)
SPEED = 250.0
BOOST = 5.0
FAR_M = 600_000.0
SUN_TOWARD = (-0.45, -0.85, 0.3)
SUN_COLOR = (2.6, 2.5, 2.35)
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
    base.set_background_color(0.01, 0.01, 0.02, 1)
    base.setFrameRateMeter(True)
    base.cam.node().get_lens().set_near_far(1.0, FAR_M)
    level = QualityLevel.parse(args.quality)
    state = {"t": 0.0, "paused": False, "speed": SPEED, "level": level}
    field = AsteroidField(
        base,
        PRESETS[args.preset],
        clock=lambda: state["t"],
        settings=AsteroidSettings.preset(level),
        backend=args.backend,
        sun_direction=SUN_TOWARD,
        sun_color=SUN_COLOR,
        ambient=AMBIENT,
        far=FAR_M,
    )
    base.camera.set_pos(*START)
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
        field.set_settings(AsteroidSettings.preset(state["level"]))

    base.accept("p", toggle_pause)
    base.accept("f1", cycle_quality)
    base.accept("page_up", lambda: state.update(speed=state["speed"] * 2.0))
    base.accept("page_down", lambda: state.update(speed=state["speed"] / 2.0))
    base.accept("end", lambda: base.camera.set_pos_hpr(*START, 0, 0, 0))
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
        cam.set_pos(cam, move * speed * dt)
        pos = np.array(cam.get_pos(base.render))
        field.update(pos, pos)
        hud_timer[0] -= dt
        if hud_timer[0] <= 0.0:
            hud_timer[0] = HUD_PERIOD
            s = field.stats
            hud.setText(
                f"{args.preset}  {state['level'].label}  "
                f"{'loading' if field.loading else ''}"
                f"{'PAUSED' if state['paused'] else ''}\n"
                f"rocks {s.get('rocks', 0)}  drawn {s.get('drawn', 0)}  "
                f"smallest {s.get('min_pixels', 0):.1f} px  "
                f"update {s.get('update_ms', 0):.2f} ms  "
                f"speed {speed:.0f} m/s"
            )
        return task.cont

    base.taskMgr.add(fly, "asteroid_demo")
    print(describe_window(base), args.preset, args.quality)
    try:
        base.run()
    finally:
        field.destroy()


if __name__ == "__main__":
    main()
