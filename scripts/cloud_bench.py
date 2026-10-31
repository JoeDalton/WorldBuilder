"""
Time and shoot a planet's clouds alone: no terrain, no sea, only the air, the
sky and the clouds, so the clouds are most of the frame.

    python scripts/cloud_bench.py [--out DIR] [--pose NAME ...]
        [--size "1600 900"] [--offscreen] [--frames N] [--timeout SECONDS]
        [--quality low|mid|high|ultra] [--renderer billboards|march]
        [--half-res | --no-half-res]
        [--uniform-coverage C] [--no-cirrus] [--cloud-debug BITS]
        [--move M] [--genera] [--interior-growth | --no-interior-growth]
        [--interactive]

    --out              where the PNGs go (default ./cloud_bench)
    --pose             the poses to time (default: all, in the order below)
    --size             window or buffer size (default "1600 900")
    --offscreen        render to an offscreen buffer, not a window
    --frames           frames timed per pose, clouds on and off (default 60)
    --timeout          seconds to wait for a pose's clouds to load (default 60)
    --quality          the clouds' preset (CloudSettings), default high
    --renderer         billboards (the default) or march (the raymarch)
    --half-res         the clouds at half resolution (CloudSettings
                       .half_resolution), over the preset's choice
    --uniform-coverage the same coverage everywhere (no weather systems,
                       climate or terrain): comparisons that do not depend
                       on where the pose is
    --no-cirrus        no cirrus sheet
    --cloud-debug      PlanetClouds.set_debug bits (cloud.frag's uCloudDebug)
    --genera           the low clouds by genus (clouds.genera's placeholder
                       maps around the reference point: cumulonimbus,
                       congestus, stratus) and their poses (below)
    --interior-growth  larger billboards inside clouds
                       (CloudSettings.interior_growth), over the preset's
    --move             fly the camera (and the probe) along the view at M
                       metres a frame while timing: reprojection under motion
    --interactive      fly the probe in and out of the clouds in a window
                       (keys below); no timing

The planet is the demo level's (its radius and climate drive the weather),
with no terrain drawn: below the horizon is the sky's own ground. As
terrain_shots.py does, each timed frame waits for the GPU (a one-pixel read
back), so a frame is CPU and GPU in series. Each pose is timed with the
clouds shown, then hidden; the difference is the clouds' cost.

On WSL, Mesa picks its CPU rasteriser (llvmpipe) unless told to use the
GPU: run with ``GALLIUM_DRIVER=d3d12``. The renderer is printed first.

Poses, over the level's reference point, with the clouds' base at 2.2 km and
the sun at azimuth 210, elevation 35: below (300 m, the horizon), start
(1.3 km, looking a little up), inside (2.7 km, in the slab), above (4.5 km,
looking down at the deck), high (20 km), orbit (3 radii up, looking down),
sunset (600 m, looking west at the sun 3 degrees up), model_in_cloud (the
probe, a black ball standing in for a game's model, half into the nearest
cloud ahead at 2.7 km, seen from 250 m).

With --genera (the features: a cumulonimbus 40 km north of the reference
point, another 70 km north-east, congestus 30 km west, a stratus deck 30 km
south): cb_far (300 m up at the reference, looking north at the towers),
cb_high (20 km up, 50 km south of the cumulonimbus), between_towers (3.5 km
up, 25 km north, in the congestus around the cumulonimbus), under_anvil
(2 km up under the cumulonimbus's edge, looking up at its anvil),
in_congestus (4 km up, inside the congestus to the west), stratus_above
(2 km up over the stratus deck, looking down), stratus_below (300 m up under
it), genera_orbit (3 radii up).

Interactive (``--interactive``): the probe flies, the camera follows it.

    W/S A/D R/F   the probe forward/back, left/right, up/down (along the view)
    shift         five times faster
    arrows        turn the camera around the probe (free: turn the camera)
    +/-           the camera nearer/farther (free: slower/faster)
    tab           free camera: the keys fly the camera, the probe stays
    p             the next pose (model_in_cloud puts the probe at a cloud)
    m             billboards <-> raymarch
    escape        quit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import ClockObject, Filename, LQuaternionf, TextNode

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import (  # noqa: E402
    CLOUD_BASE,
    CLOUD_SKY_LIGHT,
    INITIAL_FAR_M,
    _quat_from_axes,
    describe_window,
    sun_lighting,
    window_config,
)

from world_builder.atmosphere import AtmosphereParams, Atmospheres  # noqa: E402
from world_builder.clouds.cirrus import CirrusSpec  # noqa: E402
from world_builder.clouds.cloud import CloudSettings  # noqa: E402
from world_builder.clouds.genera import GeneraSpec  # noqa: E402
from world_builder.clouds.half_res import HalfResClouds  # noqa: E402
from world_builder.clouds.noise import density_on_sphere  # noqa: E402
from world_builder.clouds.planet import PlanetClouds  # noqa: E402
from world_builder.clouds.sphere import _noise  # noqa: E402
from world_builder.clouds.weather import WeatherSpec  # noqa: E402
from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCameras  # noqa: E402
from world_builder.quality import QualityLevel  # noqa: E402
from world_builder.settings import WorldSettings  # noqa: E402
from world_builder.sky import Sky, Starfield  # noqa: E402
from world_builder.terrain.chart import TangentFrame, latlon  # noqa: E402
from world_builder.terrain.level import DEMO_LEVEL, load_level  # noqa: E402
from world_builder.terrain.workers import ChunkExecutor  # noqa: E402

#: name -> (altitude, heading, pitch), over the level's reference point
POSES = {
    "below": (300.0, 0.0, 5.0),
    "start": (1300.0, 0.0, 10.0),
    "inside": (2700.0, 0.0, 0.0),
    "above": (4500.0, 0.0, -20.0),
    "high": (20000.0, 0.0, -25.0),
    "orbit": (None, 0.0, -89.9),  # 3 radii up
    "sunset": (600.0, 90.0, 2.0),
    "model_in_cloud": (2700.0, 0.0, 0.0),  # the probe placed by probe_at_cloud
}
#: name -> (altitude, heading, pitch, east, north): over a point *east* and
#: *north* metres from the reference (--genera)
GENERA_POSES = {
    "cb_far": (300.0, 0.0, 6.0, 0.0, 0.0),
    "cb_high": (20000.0, 0.0, -12.0, 0.0, -10e3),
    "between_towers": (3500.0, 0.0, 2.0, 0.0, 25e3),
    "under_anvil": (2000.0, 0.0, 35.0, 0.0, 27e3),
    "in_congestus": (4000.0, 90.0, 0.0, -30e3, 30e3),
    "stratus_above": (2000.0, 180.0, -20.0, 0.0, -15e3),
    "stratus_below": (300.0, 180.0, 10.0, 0.0, -25e3),
    "genera_orbit": (None, 0.0, -89.9, 0.0, 0.0),
}
#: Sun (azimuth, elevation) per pose, as seen there; default (210, 35).
SUNS = {"sunset": (270.0, 3.0)}
#: The probe's radius, metres: a small aircraft's size.
PROBE_RADIUS = 25.0
#: How far behind the probe the model_in_cloud pose looks from, metres.
PROBE_VIEW = 250.0
#: The density the model_in_cloud probe looks for (the field's, 0 to 1).
PROBE_CLOUD = 0.3
#: The probe's speed, m/s, and the free camera's to start with.
PROBE_SPEED = 60.0


def uniform_weather(planet, coverage: float) -> WeatherSpec:
    """A weather of *coverage* everywhere: no systems, no climate, no
    terrain, no mesoscale texture."""
    return WeatherSpec(
        planet,
        zonal=((0.0, coverage), (90.0, coverage)),
        cyclones=0,
        tropical=0,
        cyclone_strength=0.0,
        front_strength=0.0,
        tropical_strength=0.0,
        meso_strength=0.0,
        dry_floor=1.0,
        desert_clearing=0.0,
        maritime=0.0,
        orographic=0.0,
    )


def local_orientation(position: np.ndarray, heading: float, pitch: float):
    """:returns: the camera's quaternion for a heading and pitch (degrees,
    heading counter-clockwise from north) in the local frame at *position*"""
    up = np.asarray(position, dtype=np.float64)
    lat, lon = latlon((up / np.linalg.norm(up))[None, :])
    frame = TangentFrame.at(float(lat[0]), float(lon[0]))
    local = LQuaternionf()
    local.set_hpr((heading, pitch, 0.0))
    return local * _quat_from_axes(frame.east, frame.north, frame.up)


class CloudBench(ShowBase):
    """The air, the sky and one planet's clouds, around a camera kept at
    render's origin (the scene drawn camera-relative, as the demo does)."""

    def __init__(
        self,
        settings: CloudSettings,
        cirrus: bool = True,
        coverage: float | None = None,
        cloud_debug: int = 0,
        genera: bool = False,
    ) -> None:
        super().__init__()
        self.disable_mouse()
        self.camLens.set_near_far(1.0, INITIAL_FAR_M)
        self.camLens.set_fov(70)
        self.set_background_color(0.0, 0.0, 0.0, 1.0)
        self.settings = settings

        self.spec, _ = load_level(str(DEMO_LEVEL))
        self.radius = float(self.spec.radius)
        self.cam_world = np.array([0.0, 0.0, self.radius + 1000.0])
        self._orientation = LQuaternionf()
        self.sun = np.array([0.0, 0.0, 1.0])

        world = WorldSettings()
        self.executor = ChunkExecutor(backend="process")
        stars = Starfield()
        stars.prefetch_glow(self.executor, world.sky.glow_size)
        self.atmospheres = Atmospheres(world.atmosphere)
        self.atmospheres.add(self.radius, AtmosphereParams())
        self.sky = Sky(self.render, stars, self.atmospheres, settings=world.sky)
        self.sky.node.hide(HIDE_FROM_NEAR)  # the far camera draws it
        weather = None
        if coverage is not None:
            weather = uniform_weather(self.spec, coverage)
        spec = None
        if genera:
            spec = GeneraSpec(
                weather or WeatherSpec(self.spec),
                focus=(self.spec.reference_lat, self.spec.reference_lon),
            )
        self.clouds = PlanetClouds(
            self.render,
            self.radius,
            self.executor,
            atmospheres=self.atmospheres,
            atmosphere=0,
            altitude=CLOUD_BASE,
            cirrus=CirrusSpec() if cirrus else None,
            planet=self.spec,
            weather=weather,
            settings=settings,
            channel="clouds",
            genera=spec,
        )
        self.clouds.set_debug(cloud_debug)
        self.depth = SplitDepthCameras(self, near=1.0)
        HalfResClouds.exclude(self.sky.node)  # hides no cloud
        self.clouds.use_split_depth(self, self.depth)
        self.half_res = None
        if settings.half_resolution and settings.renderer == "billboards":
            self.half_res = HalfResClouds(self, self.depth)
            self.clouds.use_half_res(self.half_res)
        self._clock = ClockObject.get_global_clock()
        # A game's model: opaque, black, writing depth like any.
        self.probe = self.loader.load_model("models/smiley")
        self.probe.set_texture_off(1)
        self.probe.set_light_off(1)
        self.probe.set_color(0.0, 0.0, 0.0, 1.0)
        self.probe.set_scale(PROBE_RADIUS)  # a unit sphere
        self.probe.reparent_to(self.render)
        #: The probe in the planet frame, None = not shown
        self.probe_world: np.ndarray | None = None
        #: Metres the camera and the probe move along the view each frame
        self.move = 0.0
        self.taskMgr.add(self._update, "cloud_bench_update")

    def place(
        self,
        altitude: float,
        heading: float,
        pitch: float,
        sun: tuple[float, float] = (210.0, 35.0),
        east: float = 0.0,
        north: float = 0.0,
    ) -> None:
        """Over the level's reference point (moved *east* and *north*
        metres along the ground): *altitude* metres up, heading and pitch in
        the local frame (degrees), the sun at (azimuth, elevation) as seen
        there."""
        frame = TangentFrame.at(self.spec.reference_lat, self.spec.reference_lon)
        up = frame.up
        distance = math.hypot(east, north)
        if distance > 0.0:
            along = (east * frame.east + north * frame.north) / distance
            angle = distance / self.radius
            up = math.cos(angle) * frame.up + math.sin(angle) * along
            lat, lon = latlon(up[None, :])
            frame = TangentFrame.at(float(lat[0]), float(lon[0]))
        self.cam_world = up * (self.radius + altitude)
        local = LQuaternionf()
        local.set_hpr((heading, pitch, 0.0))
        self._orientation = local * _quat_from_axes(frame.east, frame.north, frame.up)
        d, _, _ = sun_lighting(*sun)
        self.sun = d[0] * frame.east + d[1] * frame.north + d[2] * frame.up

    def place_pose(self, name: str) -> tuple[float, float]:
        """A pose of :data:`POSES` or :data:`GENERA_POSES` by name.

        :returns: its heading and pitch
        """
        altitude, heading, pitch, *offset = {**POSES, **GENERA_POSES}[name]
        if altitude is None:
            altitude = 2.0 * self.radius
        self.place(altitude, heading, pitch, SUNS.get(name, (210.0, 35.0)), *offset)
        return heading, pitch

    def look(self, position: np.ndarray, heading: float, pitch: float) -> None:
        """The camera at *position* (planet frame), heading and pitch in the
        local frame there (degrees)."""
        self.cam_world = np.asarray(position, dtype=np.float64)
        self._orientation = local_orientation(position, heading, pitch)

    def density(self, points: np.ndarray) -> np.ndarray:
        """The low clouds' density at planet-frame *points* (the field the
        shaders draw, without the raymarch's billows), once set up."""
        volume = self.clouds.volume
        return density_on_sphere(
            _noise(7), volume.weather, volume.field, points, self.radius
        )

    def probe_at_cloud(self) -> bool:
        """The model_in_cloud pose: from 2.7 km over the reference point,
        the probe half into the nearest dense cloud ahead (turning until
        one is in reach), the camera :data:`PROBE_VIEW` behind it.

        :returns: whether a cloud was found
        """
        frame = TangentFrame.at(self.spec.reference_lat, self.spec.reference_lon)
        start = frame.up * (self.radius + POSES["model_in_cloud"][0])
        reach = np.arange(PROBE_VIEW + 50.0, 4000.0, 10.0)
        for heading in range(0, 360, 15):
            h = math.radians(heading)
            ahead = math.cos(h) * frame.north - math.sin(h) * frame.east
            # Substantial cloud, not a wisp
            cloudy = self.density(start + reach[:, None] * ahead) > PROBE_CLOUD
            if cloudy.any():
                edge = start + reach[int(np.argmax(cloudy))] * ahead
                self.probe_world = edge + ahead * PROBE_RADIUS
                self.look(self.probe_world - ahead * PROBE_VIEW, float(heading), 0.0)
                return True
        return False

    def _update(self, task) -> int:
        if self.move:
            step = np.array(self._orientation.get_forward()) * self.move
            self.cam_world = self.cam_world + step
            if self.probe_world is not None:
                self.probe_world = self.probe_world + step
        self.camera.set_pos(0.0, 0.0, 0.0)
        self.camera.set_quat(self.render, self._orientation)
        if self.probe_world is None:
            self.probe.hide()
        else:
            self.probe.show()
            self.probe.set_pos(*(self.probe_world - self.cam_world))
        distance = float(np.linalg.norm(self.cam_world))
        split, far = SplitDepthCameras.system_ranges(
            [(self.radius, distance, 0.0, 0.0)]
        )
        self.depth.update(split, far)
        if self.half_res is not None:
            self.half_res.update()
        self.atmospheres.update(self.cam_world)
        self.sky.set_sun(self.sun)
        self.sky.update((0.0, 0.0, 0.0), far)
        fov = math.radians(self.camLens.get_fov()[1])
        scale = self.half_res.pixel_scale if self.half_res is not None else 1.0
        self.clouds.set_pixel_angle(
            fov / max(self.win.get_y_size(), 1), volume_scale=scale
        )
        self.clouds.set_sun(self.sun)
        self.clouds.set_sky_light(CLOUD_SKY_LIGHT)
        self.clouds.update(self.cam_world)
        return task.cont

    def settle(self, timeout: float) -> bool:
        """Step frames until the clouds have nothing pending for 10 frames.

        :returns: whether that happened before *timeout* seconds
        """
        start = time.perf_counter()
        quiet = 0
        while time.perf_counter() - start < timeout:
            self.taskMgr.step()
            busy = self.executor.pending or self.clouds.stats["pending"]
            quiet = 0 if busy else quiet + 1
            if quiet >= 10:
                return True
        return False

    def time_frames(self, frames: int) -> float:
        """:returns: the median of *frames* GPU-synced frames, milliseconds"""
        win = self.win
        probe = win.make_display_region(
            0, 1 / win.get_x_size(), 0, 1 / win.get_y_size()
        )
        times = []
        for _ in range(frames):
            start = time.perf_counter()
            self.taskMgr.step()
            probe.get_screenshot()  # waits for the GPU
            times.append(time.perf_counter() - start)
        win.remove_display_region(probe)
        return float(np.median(times)) * 1e3


class Flight:
    """The interactive bench: the probe flies, the camera follows it (or
    flies free); see the module for the keys."""

    MOVES = {"w": (0, 1), "s": (0, -1), "d": (1, 1), "a": (1, -1)}
    MOVES_UP = {"r": 1, "f": -1}
    TURNS = {"arrow_left": (0, 1), "arrow_right": (0, -1)}
    TILTS = {"arrow_up": 1, "arrow_down": -1}

    def __init__(self, bench: CloudBench, poses: list[str]) -> None:
        self.bench = bench
        self.poses = poses
        self.pose = -1
        self.heading, self.pitch = 0.0, -5.0
        self.distance = PROBE_VIEW
        self.free = False
        self._to_cloud = False
        self.speed = PROBE_SPEED
        self.keys: dict[str, bool] = {}
        for key in [*self.MOVES, *self.MOVES_UP, *self.TURNS, *self.TILTS, "shift"]:
            self.keys[key] = False
            bench.accept(key, self.keys.__setitem__, [key, True])
            bench.accept(f"{key}-up", self.keys.__setitem__, [key, False])
        for key, method in (
            ("tab", self.toggle_free),
            ("p", self.next_pose),
            ("m", self.toggle_renderer),
            ("+", lambda: self.zoom(0.8)),
            ("=", lambda: self.zoom(0.8)),
            ("-", lambda: self.zoom(1.25)),
            ("escape", sys.exit),
        ):
            bench.accept(key, method)
        bench.setFrameRateMeter(True)
        self.hud = OnscreenText(
            text="",
            pos=(-1.32, 0.92),
            scale=0.04,
            fg=(1, 1, 1, 0.9),
            shadow=(0, 0, 0, 0.7),
            align=TextNode.ALeft,
            mayChange=True,
        )
        bench.taskMgr.add(self.fly, "cloud_bench_flight", sort=-1)

    def toggle_free(self) -> None:
        self.free = not self.free

    def zoom(self, factor: float) -> None:
        if self.free:
            self.speed = min(max(self.speed / factor, 5.0), 5000.0)
        else:
            self.distance = min(max(self.distance * factor, 2 * PROBE_RADIUS), 20e3)

    def toggle_renderer(self) -> None:
        clouds = self.bench.clouds
        renderer = "march" if clouds.settings.renderer == "billboards" else "billboards"
        clouds.set_settings(replace(clouds.settings, renderer=renderer))

    def next_pose(self) -> None:
        bench = self.bench
        self.pose = (self.pose + 1) % len(self.poses)
        name = self.poses[self.pose]
        self.heading, self.pitch = bench.place_pose(name)
        # The probe waits ahead of the camera; at model_in_cloud it goes to
        # a cloud's edge once the weather is in (fly).
        forward = np.array(bench._orientation.get_forward())
        bench.probe_world = bench.cam_world + forward * self.distance
        self.free = name != "model_in_cloud"
        self._to_cloud = name == "model_in_cloud"

    def _place_at_cloud(self) -> None:
        bench = self.bench
        self._to_cloud = False
        if bench.probe_at_cloud():
            self.heading = self._heading_to(bench.probe_world)
            self.distance, self.pitch = PROBE_VIEW, 0.0

    def _heading_to(self, target: np.ndarray) -> float:
        """:returns: the heading (degrees) from the camera toward *target*"""
        up = self.bench.cam_world / np.linalg.norm(self.bench.cam_world)
        lat, lon = latlon(up[None, :])
        frame = TangentFrame.at(float(lat[0]), float(lon[0]))
        d = target - self.bench.cam_world
        return math.degrees(math.atan2(-(d @ frame.east), d @ frame.north))

    def fly(self, task) -> int:
        bench = self.bench
        if self._to_cloud and bench.clouds.volume.weather is not None:
            self._place_at_cloud()
        dt = min(ClockObject.get_global_clock().get_dt(), 0.1)
        keys = self.keys
        self.heading += (
            60.0 * dt * sum(sign for key, (_, sign) in self.TURNS.items() if keys[key])
        )
        self.pitch += (
            40.0 * dt * sum(sign for key, sign in self.TILTS.items() if keys[key])
        )
        self.pitch = min(max(self.pitch, -89.0), 89.0)
        speed = self.speed * (5.0 if keys["shift"] else 1.0) * dt
        mover = bench.cam_world if self.free else bench.probe_world
        if mover is None:
            return task.cont
        orientation = local_orientation(mover, self.heading, 0.0)
        axes = (
            np.array(orientation.get_forward()),
            np.array(orientation.get_right()),
        )
        up = mover / np.linalg.norm(mover)
        step = np.zeros(3)
        for key, (axis, sign) in self.MOVES.items():
            if keys[key]:
                step += sign * axes[axis]
        for key, sign in self.MOVES_UP.items():
            if keys[key]:
                step += sign * up
        mover = mover + step * speed
        if self.free:
            bench.look(mover, self.heading, self.pitch)
        else:
            bench.probe_world = mover
            view = local_orientation(mover, self.heading, self.pitch)
            forward = np.array(view.get_forward())
            bench.look(mover - forward * self.distance, self.heading, self.pitch)
        if task.frame % 10 == 0:
            self._hud()
        return task.cont

    def _hud(self) -> None:
        bench = self.bench
        probe = bench.probe_world
        inside = ""
        if probe is not None and bench.clouds.volume.weather is not None:
            density = float(bench.density(probe[None, :])[0])
            inside = f"  density at the probe {density:.2f}"
        altitude = 0.0 if probe is None else float(np.linalg.norm(probe)) - bench.radius
        self.hud.setText(
            f"{bench.clouds.settings.renderer} (m)   "
            f"{'free camera' if self.free else 'following'} (tab)\n"
            f"probe {altitude:.0f} m up{inside}\n"
            f"camera {self.distance:.0f} m back, speed {self.speed:.0f} m/s (+/-)\n"
            "WASD RF move, arrows turn, shift faster, p next pose"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=Path("cloud_bench"), help="PNG directory"
    )
    parser.add_argument(
        "--pose",
        nargs="*",
        default=None,
        choices=[*POSES, *GENERA_POSES],
        help="default: all (the genera's with --genera)",
    )
    parser.add_argument("--size", default="1600 900", help='"width height"')
    parser.add_argument(
        "--offscreen", action="store_true", help="an offscreen buffer, no window"
    )
    parser.add_argument("--frames", type=int, default=60, help="frames timed")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="seconds to load a pose"
    )
    parser.add_argument(
        "--quality",
        default="high",
        choices=[level.value for level in QualityLevel],
        help="the clouds' preset",
    )
    parser.add_argument(
        "--renderer",
        default="billboards",
        choices=["billboards", "march"],
        help="the low clouds' renderer",
    )
    parser.add_argument(
        "--half-res",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="half-resolution clouds, over the preset's choice",
    )
    parser.add_argument(
        "--uniform-coverage",
        type=float,
        default=None,
        help="the same coverage everywhere",
    )
    parser.add_argument("--no-cirrus", action="store_true", help="no cirrus sheet")
    parser.add_argument(
        "--cloud-debug", type=int, default=0, help="PlanetClouds.set_debug bits"
    )
    parser.add_argument(
        "--move", type=float, default=0.0, help="metres a frame along the view"
    )
    parser.add_argument(
        "--genera", action="store_true", help="the low clouds by genus, their poses"
    )
    parser.add_argument(
        "--interior-growth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="larger billboards inside clouds, over the preset's choice",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="fly the probe, no timing"
    )
    args = parser.parse_args()

    window_config("Cloud bench", args.size, args.offscreen, vsync=False)
    settings = CloudSettings.preset(QualityLevel(args.quality))
    settings = replace(settings, renderer=args.renderer)
    if args.interior_growth is not None:
        settings = replace(settings, interior_growth=args.interior_growth)
    if args.half_res is not None:
        settings = replace(settings, half_resolution=args.half_res)
    bench = CloudBench(
        settings,
        cirrus=not args.no_cirrus,
        coverage=args.uniform_coverage,
        cloud_debug=args.cloud_debug,
        genera=args.genera,
    )
    gsg = bench.win.get_gsg()
    renderer = gsg.get_driver_renderer()
    coverage = args.uniform_coverage
    print(
        f"{describe_window(bench)}, {gsg.get_driver_vendor()} {renderer}, "
        f"clouds {args.quality} {args.renderer}"
        f"{' half-res' if bench.half_res is not None else ''}"
        f"{'' if coverage is None else f', coverage {coverage}'}"
    )
    if "llvmpipe" in renderer:
        print("warning: CPU rasteriser; on WSL set GALLIUM_DRIVER=d3d12 for the GPU")
    if args.interactive:
        poses = args.pose or (
            list(GENERA_POSES)
            if args.genera
            else ["model_in_cloud", *[p for p in POSES if p != "model_in_cloud"]]
        )
        flight = Flight(bench, poses)
        flight.next_pose()
        print(__doc__.split("Interactive")[1])
        bench.run()
        return
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"{'pose':<14} {'frame':>9} {'no clouds':>10} {'clouds':>9} {'billboards':>11}"
    )
    for name in args.pose or list(GENERA_POSES if args.genera else POSES):
        bench.place_pose(name)
        bench.probe_world = None
        settled = bench.settle(args.timeout)
        if name == "model_in_cloud" and not bench.probe_at_cloud():
            print(f"{name:<8} no cloud in reach")
            continue
        for _ in range(3):  # let the last uploads reach the screen
            bench.taskMgr.step()
        bench.move = args.move
        with_clouds = bench.time_frames(args.frames)
        path = args.out / f"{name}.png"
        bench.win.save_screenshot(Filename.from_os_specific(str(path)))
        bench.move = 0.0
        billboards = bench.clouds.stats["billboards"]
        bench.clouds.set_visible(False)
        bench.taskMgr.step()
        without = bench.time_frames(args.frames)
        bench.clouds.set_visible(True)
        print(
            f"{name:<14} {with_clouds:>7.2f}ms {without:>8.2f}ms "
            f"{with_clouds - without:>7.2f}ms {billboards / 1e3:>9.0f}k"
            f"{'' if settled else '  TIMEOUT'}"
        )
    bench.executor.shutdown()


if __name__ == "__main__":
    main()
