"""
What the scripts share: the window's settings, the quality arguments, a few
camera maths. Nothing here draws a world: that is
``world_builder.load_world`` (see ``scripts/demo_world.py``).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    GraphicsWindow,
    LMatrix3f,
    LQuaternionf,
    Vec3,
    loadPrcFileData,
)

from world_builder.post import FSR1
from world_builder.quality import QualityLevel
from world_builder.settings import SECTIONS, WorldSettings
from world_builder.terrain.chart import TangentFrame, dirs_from_latlon, latlon
from world_builder.terrain.planet import Planet

#: The world the scripts fly by default.
WORLDS_PATH = Path(__file__).resolve().parents[1] / "src/world_builder/worlds"
DEFAULT_WORLD = WORLDS_PATH / "three_planets.yaml"
#: Height of a start above the ground there, metres.
START_CLEARANCE = 400.0
#: Shortest wavelength of the coarse ground a start also clears, metres:
#: about what the first chunks drawn there resolve.
START_COARSE_M = 20e3


def window_config(
    title: str, size: str = "1600 900", offscreen: bool = False, vsync: bool = True
) -> None:
    """Load the window's PRC settings (before ShowBase opens it).

    No multisampling: the driver's choice of sample count is not ours to
    make (Panda picks the config with the most samples, x16 on a GTX 1070,
    whatever count is asked for), and it was most of the clouds' cost.

    Clip control (``gl-depth-zero-to-one``) but on macOS: the world's
    reversed camera keeps a float depth buffer's precision with it.

    No transform cache: the world moves around the camera, so every drawn
    node's net transform is new every frame and Panda's cache of composed
    transforms only filled up for its garbage collection to empty (1 ms a
    frame in low flight). A game drawing WorldBuilder's world wants the same.

    :param title: the window's title
    :param size: "width height"
    :param offscreen: an offscreen buffer (no display refresh cap or
        throttling: timings are the GPU's)
    :param vsync: False turns the display's sync off (the demo and the
        scripts all do)
    """
    lines = [
        f"window-title {title}",
        f"win-size {size}",
        "audio-library-name null",
        "framebuffer-multisample 0",
        "multisamples 0",
        "depth-bits 24",
        "transform-cache false",
    ]
    # The reversed camera's 0-1 depth (clip control): read when the GL
    # context is made; macOS's OpenGL has none (depth.DEPTH_REVERSED_11).
    if sys.platform != "darwin":
        lines.append("gl-depth-zero-to-one #t")
    if not vsync:
        lines.append("sync-video false")
    if offscreen:
        lines.append("window-type offscreen")
    loadPrcFileData("", "\n".join(lines))


def describe_window(demo: ShowBase) -> str:
    """:returns: the window's kind and size, and its samples per pixel if
    more than one"""
    win = demo.win
    kind = "window" if isinstance(win, GraphicsWindow) else "offscreen"
    samples = win.get_fb_properties().get_multisamples()
    msaa = f", MSAA x{samples}" if samples > 1 else ""
    return f"{kind} {win.get_x_size()}x{win.get_y_size()}{msaa}"


def add_quality_arguments(parser: argparse.ArgumentParser) -> None:
    """The run-time arguments every script takes: ``--quality``, a
    ``--<part>-quality`` per WorldSettings section, ``--workers`` (the
    terrain's setting), ``--frame-cap`` and the view's way to the screen
    (the ``render`` section: ``--render-scale``, ``--msaa``, ``--fxaa``,
    ``--fsr``, ``--sharpness``)."""
    levels = [level.value for level in QualityLevel]
    parser.add_argument(
        "--quality", default="high", choices=levels, help="every part's preset"
    )
    for section in SECTIONS:
        parser.add_argument(
            f"--{section}-quality",
            default=None,
            choices=levels,
            help=f"the {section}'s preset, over --quality",
        )
    parser.add_argument(
        "--workers", type=int, default=None, help="generation processes"
    )
    parser.add_argument(
        "--frame-cap",
        type=int,
        default=None,
        help="the most frames a second, 0 none (default: the settings', none)",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=None,
        help="the view's resolution over the window's, 0.5-1 (default 1)",
    )
    parser.add_argument(
        "--msaa", type=int, default=None, choices=(0, 2, 4, 8, 16), help="samples"
    )
    parser.add_argument("--fxaa", action="store_true", help="FXAA over the view")
    parser.add_argument(
        "--fsr", action="store_true", help="upscale with FSR 1 (else bilinear)"
    )
    parser.add_argument(
        "--sharpness",
        type=float,
        default=None,
        help="FSR 1's sharpening in stops: 0 the most (default 0.1)",
    )


def settings_from_args(args: argparse.Namespace) -> WorldSettings:
    """:returns: the WorldSettings :func:`add_quality_arguments`' arguments
    ask for"""
    settings = WorldSettings.preset(args.quality)
    for section in SECTIONS:
        level = getattr(args, f"{section}_quality")
        if level is not None:
            settings = settings.with_section_level(section, level)
    if args.workers is not None:
        settings = settings.with_value("terrain", "workers", args.workers)
    if args.frame_cap is not None:
        settings = settings.with_value("display", "frame_cap", args.frame_cap)
    render = {
        "scale": args.render_scale,
        "msaa": args.msaa,
        "fxaa": True if args.fxaa else None,
        "upscaler": FSR1 if args.fsr else None,
        "sharpness": args.sharpness,
    }
    for name, value in render.items():
        if value is not None:
            settings = settings.with_value("render", name, value)
    return settings


def quat_from_axes(x, y, z) -> LQuaternionf:
    """:returns: the rotation taking +X, +Y, +Z to the given unit vectors"""
    q = LQuaternionf()
    q.set_from_matrix(LMatrix3f(Vec3(*x), Vec3(*y), Vec3(*z)))
    return q


def ground_height(spec, lat: float, lon: float) -> float:
    """:returns: the height above sea level (never below 0) of the ground at
    (lat, lon), degrees, over the full detail and over the coarse ground
    first drawn there (a start above it is not lifted while the detail
    loads)"""
    planet = Planet(spec)
    dirs = dirs_from_latlon(lat, lon)
    fine, _, _ = planet.sample_dirs(dirs, None, apply_stamps=True)
    coarse, _, _ = planet.sample_dirs(dirs, START_COARSE_M, apply_stamps=True)
    return max(float(fine[0]), float(coarse[0]), 0.0)


def local_sun(azimuth: float, elevation: float, frame: TangentFrame) -> np.ndarray:
    """:returns: the world-frame direction of a sun at an azimuth (degrees,
    0 = north, increasing toward east) and elevation, in a local frame"""
    az, el = math.radians(azimuth), math.radians(elevation)
    return (
        math.cos(el) * math.sin(az) * frame.east
        + math.cos(el) * math.cos(az) * frame.north
        + math.sin(el) * frame.up
    )


def set_local_sun(demo, azimuth: float, elevation: float) -> None:
    """Put the world's sun at an azimuth/elevation (degrees; 0 = north, toward
    east) as seen from the point under the camera.

    :param demo: a ``demo_world.WorldDemo``
    """
    lat, lon = latlon(demo._up()[None, :])
    frame = TangentFrame.at(float(lat[0]), float(lon[0]))
    demo.world.world.set_sun(local_sun(azimuth, elevation, frame))


def settle(demo, timeout: float) -> bool:
    """Step frames until the world is loaded and nothing is wanted or
    generating for 10 frames.

    :param demo: a ``demo_world.WorldDemo``
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
            or any(body.terrain.stats["wanted"] for body in demo.bodies)
            or not all(body.shadow_map.current for body in demo.bodies)
        )
        quiet = 0 if busy else quiet + 1
        if quiet >= 10:
            return True
    return False
