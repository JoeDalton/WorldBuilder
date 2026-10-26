"""
The terrain demo built on :class:`world_builder.world.World`.

Flies the same small planetary system as scripts/demo_terrain.py (the demo
level's 600 km planet with its air, clouds and sea, an airless moon
10,000 km away, the star field), with the same controls, keys, HUD and
options. The difference is the assembly: here the world's look is a handful
of looks (:class:`BodyLook`, :class:`CloudLook`, :class:`OceanLook`,
:class:`SunLook`, :class:`SkyLook`), and each frame is::

    self.depth.update(*world.depth_ranges(cam_world))
    world.update(cam_world, far=self.depth.far)

Run it from an environment where world_builder is importable::

    python scripts/demo_world.py [the options of demo_terrain.py]

Run the two side by side to compare them. The controls, the HUD and the
flight are demo_terrain.py's (this demo is a subclass of its TerrainDemo);
see that script's docstring.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ClockObject,
    GraphicsWindow,
    LQuaternionf,
    TextNode,
)

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    _CONTROLS,
    CLOUD_BASE,
    CLOUD_SKY_LIGHT,
    HUD_PERIOD,
    INITIAL_FAR_M,
    MAX_RANGE_FACTOR,
    MOON_DISTANCE_M,
    MOON_SEED,
    OCEAN_ANCHOR_RANGE,
    OCEAN_NEAR_BAND,
    QUALITY_ALL_KEY,
    QUALITY_KEYS,
    SKY_AMBIENT,
    SPACE_AMBIENT,
    SPEED,
    SUN_COLOR,
    WATER_TUNED_ELEVATION,
    Body,
    TerrainDemo,
    add_world_arguments,
    settings_from_args,
    sun_lighting,
    window_config,
)

from world_builder.atmosphere import AtmosphereParams  # noqa: E402
from world_builder.depth import SplitDepthCameras  # noqa: E402
from world_builder.settings import WorldSettings  # noqa: E402
from world_builder.terrain.chart import TangentFrame  # noqa: E402
from world_builder.terrain.level import DEMO_LEVEL, load_level  # noqa: E402
from world_builder.terrain.planet import PRESETS  # noqa: E402
from world_builder.world import (  # noqa: E402
    BodyLook,
    CloudLook,
    OceanLook,
    SkyLook,
    SunLook,
    World,
)

# ── the look: the same as demo_terrain.py's, as looks ───────────────────────


def planet_look(spec, clouds: bool) -> BodyLook:
    """The demo's planet: air, cumulus at CLOUD_BASE, the 20 m swell."""
    return BodyLook(
        "planet",
        spec,
        atmosphere=AtmosphereParams(),
        clouds=(
            CloudLook(altitude=CLOUD_BASE, sky_strength=CLOUD_SKY_LIGHT)
            if clouds
            else None
        ),
        ocean=OceanLook(
            # The game's 20 m swell, in deep water: it dies out over the
            # shallows, so it does not drown the beaches.
            swell_amplitude=20.0,
            near_band=OCEAN_NEAR_BAND,
            anchor_range=OCEAN_ANCHOR_RANGE,
            tuned_elevation=WATER_TUNED_ELEVATION,
        ),
        ambient=SKY_AMBIENT,
        mass=5.97e24,
    )


def moon_look(spec, distance: float) -> BodyLook:
    """The demo's airless sand moon, in its start view's northern sky."""
    return BodyLook(
        "moon",
        replace(PRESETS["moon"], seed=MOON_SEED),
        position=tuple(TerrainDemo._moon_position(spec, distance)),
        ambient=SPACE_AMBIENT,
        mass=7.35e22,
    )


def start_sun(spec) -> np.ndarray:
    """Where the original flat demo had the sun, at the level's start."""
    frame = TangentFrame.at(spec.reference_lat, spec.reference_lon)
    d, _, _ = sun_lighting(210.0, 35.0)
    return d[0] * frame.east + d[1] * frame.north + d[2] * frame.up


# ── the demo ────────────────────────────────────────────────────────────────


class WorldDemo(TerrainDemo):
    """demo_terrain.py's demo, its world built and updated by :class:`World`.

    The attributes TerrainDemo's controls, HUD and flight read (``bodies``,
    ``terrain``, ``clouds``, ``ocean``, ``executor``, ``atmospheres``,
    ``sky``, ``depth``, ``world``: the settings) point into :attr:`scene`.

    :ivar scene: the :class:`World`
    """

    def __init__(
        self,
        level_path: str,
        backend: str,
        radius: float | None,
        moon_distance: float | None = MOON_DISTANCE_M,
        settings: WorldSettings | None = None,
        clouds: bool = True,
        cloud_debug: int = 0,
        ocean_debug: int = 0,
    ) -> None:
        ShowBase.__init__(self)
        self.setFrameRateMeter(True)
        self.disable_mouse()
        self._clock = ClockObject.get_global_clock()
        # Before the world: the ocean's reflection camera copies this lens.
        self.camLens.set_near_far(1.0, INITIAL_FAR_M)
        self.camLens.set_fov(70)
        self.set_background_color(0.0, 0.0, 0.0, 1.0)
        self.cam_world = np.array([0.0, 0.0, 1.0])
        self._orientation = LQuaternionf()
        self.speed = SPEED

        spec, self.start_pose = load_level(level_path, radius=radius)
        self.world = settings or WorldSettings()
        self._clouds_on = clouds
        self._cloud_debug = cloud_debug
        self._sun = start_sun(spec)
        looks = [planet_look(spec, clouds)]
        if moon_distance is not None:
            looks.append(moon_look(spec, moon_distance))
        print("starting terrain workers ...")
        self.scene = World(
            self,
            looks,
            sun=SunLook(direction=tuple(self._sun), color=SUN_COLOR),
            sky=SkyLook(),
            settings=self.world,
            backend=backend,
        )
        self._scene_sun = self._sun.copy()

        planet = self.scene.body("planet")
        self.executor = self.scene.executor
        self.atmospheres = self.scene.atmospheres
        self.sky = self.scene.sky
        self.terrain = planet.terrain
        self.clouds = planet.clouds
        self.ocean = planet.ocean
        self.bodies = [
            Body(b.name, b.terrain, atmosphere=b.atmosphere is not None)
            for b in self.scene.bodies
        ]
        self.home = self.bodies[0]
        self.body = self.home
        self.max_range = MAX_RANGE_FACTOR * (moon_distance or 10.0 * spec.radius)
        if self.clouds is not None:
            self.clouds.set_debug(cloud_debug)
        self.ocean.set_debug_mode(ocean_debug)
        # After the world, whose sea copied the lens.
        self.depth = SplitDepthCameras(self, near=1.0)
        self.scene.use_split_depth(self.depth)  # the clouds' half-res layers
        self._debug = 0
        self._ground = 0.0

        self._keys = dict.fromkeys(_CONTROLS, False)
        for key in _CONTROLS:
            self.accept(key, self._set_key, [key, True])
            self.accept(f"{key}-up", self._set_key, [key, False])
        presses = [
            ("y", self._level_roll, []),
            ("page_up", self._scale_speed, [2.0]),
            ("page_down", self._scale_speed, [0.5]),
            ("home", self._jump_to_orbit, []),
            ("tab", self._jump_to_next_body, []),
            ("end", self._reset_camera, []),
            ("r", self._reset_camera, []),
            ("f", self._print_pose, []),
            ("escape", self._quit, []),
            (QUALITY_ALL_KEY, self._cycle_all_quality, []),
        ]
        presses += [(key, self._cycle_quality, [name]) for key, name in QUALITY_KEYS]
        for key, method, args in presses:
            self.accept(key, method, extraArgs=args)
        if isinstance(self.win, GraphicsWindow):
            self.win.set_close_request_event("window-close")
            self.accept("window-close", self._quit)

        self._hud = OnscreenText(
            text="",
            pos=(-1.32, 0.92),
            scale=0.04,
            fg=(1, 1, 1, 0.9),
            shadow=(0, 0, 0, 0.7),
            align=TextNode.ALeft,
            mayChange=True,
        )
        self._hud_time = -1.0
        self._reset_camera()
        self._test_ball, self._test_ball_point = self._make_test_ball()
        self.taskMgr.add(self._update, "terrain_demo_update")
        print(__doc__)

    def _update(self, task) -> int:
        dt = min(self._clock.get_dt(), 0.1)
        self.body = self._nearest_body()
        self._fly(dt)
        self.camera.set_pos(0.0, 0.0, 0.0)
        self.camera.set_quat(self.render, self._orientation)
        # The keys (and the start pose) move the sun: hand it over.
        if not np.array_equal(self._sun, self._scene_sun):
            self.scene.set_sun(self._sun)
            self._scene_sun = self._sun.copy()

        self.depth.update(*self.scene.depth_ranges(self.cam_world))
        self.scene.update(self.cam_world, far=self.depth.far)

        self._test_ball.set_pos(*self.home.terrain.to_render(self._test_ball_point))
        now = self._clock.get_frame_time()
        if now - self._hud_time >= HUD_PERIOD:
            self._hud_time = now
            self._update_hud()
        return task.cont

    def _set_world(self, world: WorldSettings) -> None:
        """Hand every part its section (F1 - F6)."""
        self.world = world
        self.scene.set_settings(world)

    def destroy_terrain(self) -> None:
        """Remove the world and stop its workers."""
        self.depth.destroy()
        self.scene.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="The terrain demo, on World")
    parser.add_argument("--level", default=str(DEMO_LEVEL), help="level file")
    parser.add_argument(
        "--radius", type=float, default=None, help="planet radius, metres"
    )
    parser.add_argument(
        "--moon-distance",
        type=float,
        default=MOON_DISTANCE_M,
        help="moon distance, centre to centre, metres",
    )
    parser.add_argument("--no-moon", action="store_true", help="the planet alone")
    parser.add_argument(
        "--backend",
        default="process",
        choices=("process", "thread", "inline"),
        help="where chunks and cloud tiles are generated",
    )
    add_world_arguments(parser)
    args = parser.parse_args()
    window_config("Procedural terrain demo (World)", vsync=False)
    moon = None if args.no_moon else args.moon_distance
    WorldDemo(
        args.level,
        args.backend,
        args.radius,
        moon,
        settings=settings_from_args(args),
        clouds=not args.no_clouds,
    ).run()


if __name__ == "__main__":
    main()
