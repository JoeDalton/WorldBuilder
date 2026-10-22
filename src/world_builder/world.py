"""
A world: every WorldBuilder part a game draws around its camera, built and
updated as one.

A game describes its world (what bodies there are, where, and how they look)
and :class:`World` builds and wires the parts: each body's terrain, its air,
its planet clouds and its swell ocean, the sky, the sun, and Panda3D lights
for everything else (the game's own models). Every frame,
:meth:`World.update` does what scripts/demo_terrain.py does by hand: the
terrain, the air, the sky, the clouds, the lights and the ocean's frame.

Usage::

    from world_builder.world import BodyLook, CloudLook, OceanLook, SunLook, World

    world = World(
        app,
        bodies=[
            BodyLook(
                "planet",
                spec,
                atmosphere=AtmosphereParams(),
                clouds=CloudLook(altitude=2200.0),
                ocean=OceanLook(),
            ),
        ],
        sun=SunLook(direction=(0.4, -0.5, 0.75)),
        sky=SkyLook(),
        settings=world_settings,
    )
    # Every frame, once the camera is placed:
    split_depth.update(*world.depth_ranges(origin_world))
    world.update(origin_world, far=split_depth.far)
    ...
    world.destroy()

Frames:

- **World frame**: float64 metres, with render's axes (nothing is rotated).
  Each body's centre (:attr:`BodyLook.position`) and the sun's direction are
  in it.
- **Render frame**: what Panda3D draws in (float32). The game keeps its
  origin near the camera, so its numbers stay small: the parts are placed
  relative to the camera in float64, then handed to render.
- **origin_world**: where render's origin is in the world frame, which the
  game knows (its camera, or the object it keeps at the origin). The camera's
  own render position is read from ``app.camera``; the two give the camera's
  world position.

Build a world before any :class:`world_builder.depth.SplitDepthCameras`: the
ocean's reflection camera copies the app's lens once, and the split cameras
shorten it to the near range.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    AmbientLight,
    ClockObject,
    DirectionalLight,
    LMatrix3f,
    LQuaternionf,
    LVecBase2f,
    LVecBase3f,
    Texture,
    Vec3,
)

from world_builder import SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M
from world_builder.atmosphere import AtmosphereParams, Atmospheres, in_shadow
from world_builder.clouds.cloud import CloudType
from world_builder.clouds.sphere import SphericalClouds
from world_builder.clouds.weather import WeatherSpec
from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCameras
from world_builder.ocean import Ocean
from world_builder.settings import WorldSettings
from world_builder.sky import Sky
from world_builder.sky.starfield import NebulaStyle, Starfield
from world_builder.terrain.chart import AzimuthalEquidistantChart, latlon
from world_builder.terrain.collision import TerrainCollision
from world_builder.terrain.planet import PlanetSpec
from world_builder.terrain.quadtree import UNKNOWN_BOUNDS
from world_builder.terrain.terrain import Terrain
from world_builder.terrain.workers import ChunkExecutor

# ── looks: what the game decides ─────────────────────────────────────────────


@dataclass(frozen=True)
class SunLook:
    """The sun: one direction for the whole world.

    :ivar direction: toward the sun, world frame (any length)
    :ivar color: sunlight above any air, linear (may exceed 1); the air
        reddens it where it must
    """

    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    color: tuple[float, float, float] = (2.6, 2.5, 2.35)


@dataclass(frozen=True)
class SkyLook:
    """The sky's look: :class:`world_builder.sky.Sky`'s arguments of the same
    names.

    :ivar stars: a procedural :class:`Starfield` or a painted cube map; None
        = a default Starfield
    """

    stars: Starfield | Texture | None = None
    star_intensity: float = 0.6
    star_brightness: float = 8.0
    nebula: NebulaStyle = NebulaStyle()
    sun_radiance: tuple[float, float, float] = (4000.0, 3800.0, 3500.0)
    exposure: float = 0.9
    star_hpr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    star_adaptation: float = 60.0


@dataclass(frozen=True)
class CloudLook:
    """A body's planet clouds: :class:`SphericalClouds`' arguments of the
    same names.

    :ivar altitude: the cloud base above sea level, metres; None = the cloud
        type's
    """

    altitude: float | None = None
    cloud_type: CloudType = CloudType.CUMULUS
    weather: WeatherSpec | None = None
    seed: int = 7
    weather_seed: int = 11
    sun_brightness: float = 2.8
    sky_strength: float = 0.35
    sky_occlusion: float = 0.55
    exposure: float = 1.15
    near_fade_radii: float = 2.5


@dataclass(frozen=True)
class OceanLook:
    """A body's swell ocean: :class:`Ocean`'s arguments of the same names
    (always with its geometric swell, on the body's exact sphere), plus how
    the world places it.

    :ivar handover: altitudes (metres) between which the swell ocean hands
        the sea over to the terrain's own flat water (above) and back
        (below): a hysteresis band. The swell is built for eyes up to 20 km.
    :ivar anchor_range: the wave pattern is anchored to the ground through a
        chart around a point near the camera's nadir, re-anchored (a jump in
        the pattern) once the nadir is this far from it, metres
    :ivar tuned_elevation: the sun's elevation (degrees) the water's colour
        was tuned in: the water is lit relative to that daylight
    """

    water_color: tuple[float, float, float] = (0.02, 0.06, 0.14)
    ripple_strength: float = 0.02
    wind_dir: tuple[float, float] = (1.0, 0.0)
    wind_strength: float = 0.5
    swell_grid_half: float = 4000.0
    swell_amplitude: float = 20.0
    swell_scale: float = 0.004
    swell_drift: float = 0.1
    fm_depth: float = 0.2
    haze_color: tuple[float, float, float] = (0.2, 0.2, 0.4)
    haze_distance: float = 150000.0
    shallow_color: tuple[float, float, float] = SHALLOW_WATER_COLOR
    shallow_depth: float = SHALLOW_WATER_DEPTH_M
    foam_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    foam_depth: float = 2.5
    foam_spacing: float = 0.6
    foam_speed: float = 1.2
    foam_strength: float = 0.85
    swell_shallow: float = 5.0
    swell_deep: float = 60.0
    handover: tuple[float, float] = (14000.0, 16000.0)
    anchor_range: float = 50000.0
    tuned_elevation: float = 35.0

    #: The fields that are the world's, not Ocean's arguments
    WORLD_FIELDS = ("handover", "anchor_range", "tuned_elevation")


@dataclass(frozen=True)
class BodyLook:
    """One planet or moon.

    :ivar name: what :meth:`World.body` finds it by
    :ivar spec: its planet (a preset, or a level file's, see
        :func:`world_builder.terrain.level.load_level`)
    :ivar position: its centre, world frame, metres
    :ivar atmosphere: its air; None = airless
    :ivar clouds: its planet clouds; None = none (needs air)
    :ivar ocean: its swell ocean; None = the terrain draws its sea as flat
        water (needs ``spec.has_ocean``)
    :ivar ambient: the sky light on its ground, linear: the sky's with air,
        next to nothing without
    :ivar exposure: its terrain's exposure (shared tonemap)
    """

    name: str
    spec: PlanetSpec
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    atmosphere: AtmosphereParams | None = None
    clouds: CloudLook | None = None
    ocean: OceanLook | None = None
    ambient: tuple[float, float, float] = (0.34, 0.44, 0.64)
    exposure: float = 0.9


# ── the parts of one body ────────────────────────────────────────────────────


def _quat_from_axes(x, y, z) -> LQuaternionf:
    """:returns: the rotation taking +X, +Y, +Z to the given unit vectors"""
    q = LQuaternionf()
    q.set_from_matrix(LMatrix3f(Vec3(*x), Vec3(*y), Vec3(*z)))
    return q


class BodyOcean:
    """A body's swell ocean, in a frame that follows the camera's nadir.

    :class:`Ocean` is a flat-world object (sea level at z = 0, droop from the
    camera). Its frame here is the tangent plane at the sea-level point under
    the camera, so it is exact where it matters; its xy follow the ground (a
    chart around an anchor), so the waves stay put. Above
    :attr:`OceanLook.handover` the swell is hidden and the terrain draws the
    sea as flat water.

    :ivar ocean: the :class:`Ocean`
    :ivar frame: its parent node, under render, moved every frame
    :ivar swell: whether the swell is shown (below the handover band)
    """

    def __init__(
        self,
        app: Any,
        terrain: Terrain,
        look: OceanLook,
        atmospheres: Atmospheres,
        settings: Any,
        clock: Callable[[], float],
        name: str,
    ) -> None:
        self.app = app
        self.terrain = terrain
        self.look = look
        self.frame = app.render.attach_new_node(f"{name}_sea_frame")
        self._game = SimpleNamespace(
            app=app,
            root_node=self.frame,
            method_lists={},
            game_time=SimpleNamespace(get_current_time=clock),
        )
        kwargs = {
            name: getattr(look, name)
            for name in look.__dataclass_fields__
            if name not in OceanLook.WORLD_FIELDS
        }
        for vector in ("water_color", "haze_color"):
            kwargs[vector] = LVecBase3f(*kwargs[vector])
        kwargs["wind_dir"] = LVecBase2f(*kwargs["wind_dir"])
        self.ocean = Ocean(
            game=self._game,
            settings=settings,
            geometric_swell=True,
            # The terrain is a true sphere: the paraboloid would stand 58 m
            # above its sea level 100 km out.
            planet_radius=terrain.radius,
            exact_sphere=True,
            **kwargs,
        )
        atmospheres.bind(self.ocean.ocean_node)
        self.ocean.set_seabed(self._seabed_heights)
        self._anchor: AzimuthalEquidistantChart | None = None
        self.swell = True
        self._apply()
        #: The light on the sea the water's colour was tuned in (see
        #: World._light_ocean); set on the first update
        self.water_reference: np.ndarray | None = None

    def _seabed_heights(self, points: np.ndarray, texel: float) -> np.ndarray:
        """The ocean's seabed sampler (on its worker thread): the drawn
        terrain under points of the sea's plane, read at the grid level
        matching the texel and only from stored chunks (generating one would
        take the workers' time from the terrain).

        :param points: (N, 3) planet frame, metres
        :param texel: the seabed texel, metres
        :returns: (N,) ground heights, metres (negative under the sea)
        """
        query = self.terrain.query
        grid = query.grid
        level = int(math.floor(math.log2(max(texel / grid.spacing0, 1.0))))
        level = min(max(level, query.level), grid.max_level)
        return query.ground_height_at(points, level=level, generate=False)

    def _apply(self) -> None:
        """Show the swell, or hand the sea to the terrain's flat water."""
        if self.swell:
            self.ocean.base_node.show()
            self.ocean.refl_buffer.set_active(True)
        else:
            self.ocean.base_node.hide()
            self.ocean.refl_buffer.set_active(False)
        self.terrain.set_sea_surface(not self.swell)

    def handover(self, altitude: float) -> None:
        """Swell below the handover band, flat water above, unchanged within.

        :param altitude: the camera's, above sea level, metres
        """
        low, high = self.look.handover
        if altitude > high:
            swell = False
        elif altitude < low:
            swell = True
        else:
            swell = self.swell
        if swell != self.swell:
            self.swell = swell
            self._apply()

    def anchor(self, up: np.ndarray) -> tuple[float, float]:
        """The chart the waves are anchored to, re-anchored at the nadir once
        it is :attr:`OceanLook.anchor_range` away.

        :param up: the nadir's direction from the body's centre (unit)
        :returns: the nadir's chart coordinates (east, north), metres
        """
        if self._anchor is not None:
            gx, gy = self._anchor.from_dirs(up[None, :])
            gx, gy = float(gx[0]), float(gy[0])
        if self._anchor is None or math.hypot(gx, gy) > self.look.anchor_range:
            lat, lon = latlon(up[None, :])
            self._anchor = AzimuthalEquidistantChart(
                float(lat[0]), float(lon[0]), self.terrain.radius
            )
            gx = gy = 0.0
        return gx, gy

    def place(self, up: np.ndarray, sun: np.ndarray) -> None:
        """Put the sea's plane on the tangent plane at the nadir, and run the
        ocean's own updates.

        :param up: the nadir's direction from the body's centre (unit)
        :param sun: toward the sun, world frame (unit)
        """
        if not self.swell:
            return
        gx, gy = self.anchor(up)
        chart_east = self._anchor.frame.east
        e = chart_east - up * (chart_east @ up)
        e /= np.linalg.norm(e)
        n = np.cross(up, e)
        radius = self.terrain.radius
        sea_point = self.terrain.to_render(up * radius)
        render = self.app.render
        self.frame.set_quat(render, _quat_from_axes(e, n, up))
        # The sea's plane in the planet frame, for its seabed: its (0, 0) is
        # gx, gy west and south of the nadir's sea-level point.
        self.ocean.set_seabed_frame(up * radius - gx * e - gy * n, e, n)
        self.ocean.set_atmosphere_frame(self.ocean.base_node.get_mat(render), sun)
        self.frame.set_pos(*(sea_point - gx * e - gy * n))
        for methods in list(self._game.method_lists.values()):
            for method in methods:
                method()

    def destroy(self) -> None:
        """Remove the ocean (its reflection buffer included) and its frame."""
        self.ocean.clean()
        self.frame.remove_node()


@dataclass
class Body:
    """A built body: its look and its parts.

    :ivar look: what it was built from
    :ivar terrain: its terrain
    :ivar atmosphere: its index in the world's :class:`Atmospheres`; None =
        airless
    :ivar clouds: its planet clouds, or None
    :ivar ocean: its swell ocean, or None
    """

    look: BodyLook
    terrain: Terrain
    atmosphere: int | None = None
    clouds: SphericalClouds | None = None
    ocean: BodyOcean | None = None

    @property
    def name(self) -> str:
        return self.look.name

    @property
    def position(self) -> np.ndarray:
        return self.terrain.position

    @property
    def radius(self) -> float:
        return self.terrain.radius

    def parts(self) -> list:
        """:returns: its parts that take world settings"""
        parts = [self.terrain]
        if self.clouds is not None:
            parts.append(self.clouds)
        if self.ocean is not None:
            parts.append(self.ocean.ocean)
        return parts


# ── the world ────────────────────────────────────────────────────────────────


class World:
    """Every WorldBuilder part around a game's camera: bodies, sky, sun and
    lights (see the module).

    :ivar bodies: the built bodies, in the order given
    :ivar atmospheres: the bodies' air, one :class:`Atmospheres` for all
    :ivar sky: the sky, or None
    :ivar executor: the worker pool every terrain and cloud layer shares
    :ivar settings: what every part costs
    :ivar sun_light: Panda3D's sun on render (None without lights)
    :ivar ambient_light: Panda3D's ambient light on render (None without)
    """

    def __init__(
        self,
        app: Any,
        bodies: Sequence[BodyLook],
        sun: SunLook = SunLook(),
        sky: SkyLook | None = None,
        settings: WorldSettings | None = None,
        lights: bool = True,
        light_scale: float = 1.0,
        clock: Callable[[], float] | None = None,
        backend: str = "process",
    ) -> None:
        """
        :param app: the ShowBase: its render, camera, lens and window
        :param bodies: the planets and moons (at least one)
        :param sun: the sun
        :param sky: the sky; None = none (the game draws its own)
        :param settings: what every part costs; None = HIGH
        :param lights: put Panda3D's sun and ambient lights on render, as the
            world's light at the camera, for the game's own models
        :param light_scale: multiplies those lights' colours. The world's
            light is physical (noon sun about 2.5), and models lit by
            Panda3D's fixed-function lighting, with no tone map, clip every
            channel at 1 (see docs/source/object_lighting.md): a game whose
            models are lit that way scales it down. 1 = as the world is lit
        :param clock: game time in seconds, for the swell; None = Panda's
            frame time
        :param backend: where chunks and cloud tiles are generated
            (:class:`ChunkExecutor`)
        :raises ValueError: no body, two bodies of one name, clouds without
            air, or an ocean on a dry planet
        """
        if not bodies:
            raise ValueError("a world needs a body")
        names = [look.name for look in bodies]
        if len(set(names)) != len(names):
            raise ValueError(f"body names must differ: {names}")
        for look in bodies:
            if look.clouds is not None and look.atmosphere is None:
                raise ValueError(f"{look.name}: clouds need an atmosphere")
            if look.ocean is not None and not look.spec.has_ocean:
                raise ValueError(f"{look.name}: an ocean needs spec.has_ocean")

        self.app = app
        self.settings = settings or WorldSettings()
        self._sun = self._unit(sun.direction)
        self._sun_color = np.asarray(sun.color, dtype=np.float64)
        if clock is None:
            frame_clock = ClockObject.get_global_clock()
            clock = frame_clock.get_frame_time
        self.executor = ChunkExecutor(
            backend=backend, workers=self.settings.terrain.workers or None
        )

        # Every body's air first: the terrains and clouds bind them all.
        self.atmospheres = Atmospheres(self.settings.atmosphere)
        indices = [
            None
            if look.atmosphere is None
            else self.atmospheres.add(look.spec.radius, look.atmosphere, look.position)
            for look in bodies
        ]

        render = app.render
        self.bodies: list[Body] = []
        materials = None
        for look, index in zip(bodies, indices):
            terrain = Terrain(
                parent=render,
                spec=look.spec,
                executor=self.executor,
                atmospheres=self.atmospheres,
                atmosphere=index,
                sun_direction=tuple(self._sun),
                sun_color=tuple(self._sun_color),
                ambient=look.ambient,
                exposure=look.exposure,
                materials=materials,
                position=look.position,
                settings=self.settings.terrain,
            )
            # One texture set for every body (they load once)
            materials = materials or terrain.materials
            body = Body(look=look, terrain=terrain, atmosphere=index)
            if look.clouds is not None:
                body.clouds = self._make_clouds(look, index)
            if look.ocean is not None:
                body.ocean = BodyOcean(
                    app,
                    terrain,
                    look.ocean,
                    self.atmospheres,
                    self.settings.ocean,
                    clock,
                    look.name,
                )
            else:
                terrain.set_sea_surface(look.spec.has_ocean)
            self.bodies.append(body)

        self.sky = None
        if sky is not None:
            self.sky = Sky(
                render,
                sky.stars if sky.stars is not None else Starfield(),
                self.atmospheres,
                star_intensity=sky.star_intensity,
                star_brightness=sky.star_brightness,
                nebula=sky.nebula,
                sun_radiance=sky.sun_radiance,
                exposure=sky.exposure,
                star_hpr=sky.star_hpr,
                star_adaptation=sky.star_adaptation,
                settings=self.settings.sky,
            )
            self.sky.node.hide(HIDE_FROM_NEAR)  # the far camera draws it, first
            self.sky.set_sun(self._sun)

        self.light_scale = float(light_scale)
        self.sun_light = self.ambient_light = None
        if lights:
            self.sun_light = render.attach_new_node(DirectionalLight("world_sun"))
            self.ambient_light = render.attach_new_node(AmbientLight("world_ambient"))
            render.set_light(self.sun_light)
            render.set_light(self.ambient_light)

        #: The camera's world position at the last update
        self.camera_world = np.asarray(bodies[0].position, dtype=np.float64)

    def _make_clouds(self, look: BodyLook, atmosphere: int) -> SphericalClouds:
        cl = look.clouds
        clouds = SphericalClouds(
            self.app.render,
            look.spec.radius,
            self.executor,
            position=look.position,
            atmospheres=self.atmospheres,
            atmosphere=atmosphere,
            cloud_type=cl.cloud_type,
            altitude=cl.altitude,
            planet=look.spec,
            weather=cl.weather,
            seed=cl.seed,
            weather_seed=cl.weather_seed,
            settings=self.settings.clouds,
            sun_brightness=cl.sun_brightness,
            sky_strength=cl.sky_strength,
            sky_occlusion=cl.sky_occlusion,
            exposure=cl.exposure,
            near_fade_radii=cl.near_fade_radii,
            channel=f"clouds:{look.name}",
        )
        clouds.set_sun(self._sun)
        return clouds

    @staticmethod
    def _unit(vector: ArrayLike) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float64)
        return v / np.linalg.norm(v)

    # ── queries ─────────────────────────────────────────────────────────────

    def body(self, name: str) -> Body:
        """:returns: the body of that name
        :raises KeyError: for an unknown name"""
        for body in self.bodies:
            if body.name == name:
                return body
        raise KeyError(name)

    @property
    def sun(self) -> np.ndarray:
        """Toward the sun, world frame (unit)."""
        return self._sun.copy()

    @property
    def loading(self) -> bool:
        """Whether any terrain still waits for ground, or any clouds for
        tiles: for a game's loading phase. Wait for it to stay False for a
        moment rather than for one frame (see :attr:`Terrain.loading`)."""
        return any(
            body.terrain.loading
            or (body.clouds is not None and body.clouds.stats["pending"] > 0)
            for body in self.bodies
        )

    def camera_render(self) -> np.ndarray:
        """:returns: the camera's position in render, float64 (the origin
        without a camera)"""
        camera = getattr(self.app, "camera", None)
        if camera is None:
            return np.zeros(3)
        return np.array(camera.get_pos(self.app.render), dtype=np.float64)

    def depth_ranges(self, origin_world: ArrayLike) -> tuple[float, float]:
        """Split and far distances for :class:`SplitDepthCameras` this
        frame: the split follows the nearest ground, the far plane reaches
        the farthest visible part of any body.

        :param origin_world: render's origin in the world frame (float64)
        :returns: (split, far), metres
        """
        camera = np.asarray(origin_world, dtype=np.float64) + self.camera_render()
        return SplitDepthCameras.system_ranges(
            (
                body.radius,
                float(np.linalg.norm(camera - body.position)),
                UNKNOWN_BOUNDS[1],
                0.0 if body.look.spec.has_ocean else UNKNOWN_BOUNDS[0],
            )
            for body in self.bodies
        )

    def nearest(self, camera_world: ArrayLike) -> Body:
        """:returns: the body whose sea level is nearest a point"""
        point = np.asarray(camera_world, dtype=np.float64)
        return min(
            self.bodies,
            key=lambda b: float(np.linalg.norm(point - b.position)) - b.radius,
        )

    def light_at(self, point_world: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """The light at a point, as the terrain's shader lights the ground
        there: through the nearest body's air, or in its shadow if airless.

        :param point_world: world frame, metres
        :returns: (direct, ambient), linear rgb
        """
        return self._light_at(self.nearest(point_world), point_world)

    def _light_at(self, body: Body, point_world) -> tuple[np.ndarray, np.ndarray]:
        ambient = np.asarray(body.look.ambient, dtype=np.float64)
        if body.atmosphere is not None:
            return self.atmospheres.light_at(
                body.atmosphere, point_world, self._sun, self._sun_color, ambient
            )
        hidden = in_shadow(
            np.asarray(point_world, dtype=np.float64) - body.position,
            self._sun,
            body.radius,
        )
        return (np.zeros(3) if hidden else self._sun_color.copy()), ambient

    # ── changes ─────────────────────────────────────────────────────────────

    def set_sun(self, direction: ArrayLike) -> None:
        """Move the sun (toward it, world frame, any length): every part
        follows at once, the lights at the next :meth:`update`."""
        self._sun = self._unit(direction)
        if self.sky is not None:
            self.sky.set_sun(self._sun)
        for body in self.bodies:
            body.terrain.set_sun(
                tuple(self._sun), tuple(self._sun_color), body.look.ambient
            )
            if body.clouds is not None:
                body.clouds.set_sun(self._sun)

    def set_settings(self, settings: WorldSettings) -> None:
        """Change what every part costs (see :meth:`WorldSettings.apply`)."""
        self.settings = settings
        parts: list = [self.atmospheres]
        if self.sky is not None:
            parts.append(self.sky)
        for body in self.bodies:
            parts += body.parts()
        settings.apply(*parts)

    def enable_collision(
        self, into_mask: Any, bodies: Iterable[str] | None = None, **kwargs: Any
    ) -> list[TerrainCollision]:
        """Make bodies' drawn terrain solid to the game's collision traverser
        (:meth:`Terrain.enable_collision`).

        :param into_mask: the into mask of their collision nodes
        :param bodies: their names; None = every body
        :param kwargs: :meth:`Terrain.enable_collision`'s other arguments
        :returns: their collisions, in the order of :attr:`bodies`
        """
        chosen = None if bodies is None else set(bodies)
        return [
            body.terrain.enable_collision(into_mask, **kwargs)
            for body in self.bodies
            if chosen is None or body.name in chosen
        ]

    def disable_collision(self) -> None:
        """Remove every body's collision node."""
        for body in self.bodies:
            body.terrain.disable_collision()

    # ── per frame ───────────────────────────────────────────────────────────

    def update(self, origin_world: ArrayLike, far: float | None = None) -> None:
        """Per frame, once the camera is placed: everything around it.

        :param origin_world: render's origin in the world frame (float64);
            the camera's render position is read from ``app.camera``
        :param far: the far camera's far distance this frame
            (:attr:`SplitDepthCameras.far`), metres: the sky is drawn just
            inside it. None = :meth:`depth_ranges`'
        """
        camera_render = self.camera_render()
        camera = np.asarray(origin_world, dtype=np.float64) + camera_render
        self.camera_world = camera

        for body in self.bodies:
            body.terrain.update(camera, camera_render)
        self.atmospheres.update(camera)
        if self.sky is not None:
            if far is None:
                far = self.depth_ranges(origin_world)[1]
            self.sky.update(tuple(camera_render), far)
        pixel_angle = self._pixel_angle()
        for body in self.bodies:
            if body.clouds is not None:
                if pixel_angle is not None:
                    body.clouds.set_pixel_angle(pixel_angle)
                body.clouds.update(camera, camera_render)
        self._light(camera)
        for body in self.bodies:
            if body.ocean is not None:
                relative = camera - body.position
                distance = float(np.linalg.norm(relative))
                up = relative / distance
                body.ocean.handover(distance - body.radius)
                self._light_ocean(body, up)
                body.ocean.place(up, self._sun)

    def _pixel_angle(self) -> float | None:
        """:returns: the angle one pixel of the camera's output spans
        (vertical fov / its height), or None without a camera"""
        lens = getattr(self.app, "camLens", None)
        cam = getattr(self.app, "cam", None)
        if lens is None or cam is None or cam.node().get_num_display_regions() == 0:
            return None
        height = cam.node().get_display_region(0).get_pixel_height()
        return math.radians(lens.get_fov()[1]) / max(height, 1)

    def _light(self, camera: np.ndarray) -> None:
        """Panda3D's sun and ambient lights as they are at the camera."""
        if self.sun_light is None:
            return
        direct, ambient = self.light_at(camera)
        direct, ambient = direct * self.light_scale, ambient * self.light_scale
        self.sun_light.node().set_color((*direct, 1.0))
        self.sun_light.node().set_direction(Vec3(*(-self._sun)))
        self.ambient_light.node().set_color((*ambient, 1.0))

    def _light_ocean(self, body: Body, up: np.ndarray) -> None:
        """Light the swell as the sea under the camera, relative to the
        daylight its colour was tuned in (dark at night)."""
        ocean = body.ocean
        sea = body.position + up * body.radius
        direct, ambient = self._light_at(body, sea)
        light = direct * max(float(self._sun @ up), 0.0) + ambient
        if ocean.water_reference is None:
            elevation = math.radians(ocean.look.tuned_elevation)
            east = np.cross([0.0, 0.0, 1.0], up)
            if np.linalg.norm(east) < 1e-9:  # at a pole: any east
                east = np.array([1.0, 0.0, 0.0])
            east /= np.linalg.norm(east)
            tuned = math.sin(elevation) * up + math.cos(elevation) * east
            sun = self._sun_color
            if body.atmosphere is not None:
                sun = sun * self.atmospheres.transmittance_to_sun(
                    body.atmosphere, sea, tuned
                )
            ocean.water_reference = sun * math.sin(elevation) + np.asarray(
                body.look.ambient
            )
        ocean.ocean.set_light(np.clip(light / ocean.water_reference, 0.0, 1.5))

    # ── teardown ────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Remove every part and the lights, and stop the workers."""
        render = self.app.render
        for light in (self.sun_light, self.ambient_light):
            if light is not None:
                render.clear_light(light)
                light.remove_node()
        self.sun_light = self.ambient_light = None
        if self.sky is not None:
            self.sky.remove()
            self.sky = None
        for body in self.bodies:
            if body.ocean is not None:
                body.ocean.destroy()
            if body.clouds is not None:
                body.clouds.remove()
            body.terrain.destroy()
        self.bodies = []
        self.executor.shutdown()
