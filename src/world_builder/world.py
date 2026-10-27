"""
A world: every WorldBuilder part a game draws around its camera, built and
updated as one.

A game describes its world (what bodies there are, where, and how they look)
and :class:`World` builds and wires the parts: each body's terrain, its air,
its planet clouds and its sea, the asteroid fields and rings, the sky, the
sun, and Panda3D lights for everything else (the game's own models). Every
frame, :meth:`World.update` does what scripts/demo_terrain.py does by hand:
the terrain, the fields, the air, the sky, the clouds, the lights and the
sea's frame.

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
                mass=5.97e24,
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
sea's reflection camera copies the app's lens once, and the split cameras
shorten it to the near range.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    AmbientLight,
    ClockObject,
    DirectionalLight,
    NodePath,
    Texture,
    Vec3,
)

from world_builder import SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M
from world_builder.asteroids.field import AsteroidField
from world_builder.asteroids.spec import FieldSpec, KeepOut
from world_builder.atmosphere import AtmosphereParams, Atmospheres, in_shadow
from world_builder.clouds.cloud import CloudType
from world_builder.clouds.half_res import HalfResClouds
from world_builder.clouds.sphere import SphericalClouds
from world_builder.clouds.weather import WeatherSpec
from world_builder.depth import HIDE_FROM_NEAR, SplitDepthCameras
from world_builder.gravity import Gravity
from world_builder.lights import TrackedLights
from world_builder.ocean import Sea
from world_builder.settings import WorldSettings
from world_builder.shadows import ECLIPSE_SKY_FLOOR, SUN_ANGULAR_RADIUS, Shadows
from world_builder.sky import Sky
from world_builder.sky.starfield import NebulaStyle, Starfield
from world_builder.terrain.collision import TerrainCollision
from world_builder.terrain.planet import PLANET_HEIGHT_BOUNDS, PlanetSpec
from world_builder.terrain.shadow_map import ShadowMap
from world_builder.terrain.terrain import Terrain
from world_builder.terrain.workers import ChunkExecutor

#: Open space (a world with no body): the depth split and the far plane,
#: metres (the sky is drawn just inside the far plane).
SPACE_SPLIT_M = 2000.0
SPACE_FAR_M = 1.0e6

# ── looks: what the game decides ─────────────────────────────────────────────


@dataclass(frozen=True)
class SunLook:
    """The sun: one direction for the whole world.

    :ivar direction: toward the sun, world frame (any length)
    :ivar color: sunlight above any air, linear (may exceed 1); the air
        reddens it where it must
    :ivar angular_radius: its disc's angular radius, radians: as the sky
        draws it, and every eclipse's penumbra
    """

    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    color: tuple[float, float, float] = (2.6, 2.5, 2.35)
    angular_radius: float = SUN_ANGULAR_RADIUS


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
    exposure: float = 0.9
    near_fade_radii: float = 2.5


@dataclass(frozen=True)
class OceanLook:
    """A body's sea, as :class:`world_builder.ocean.Sea` draws it near the
    camera: its arguments of the same names, plus how the world lights it.

    :ivar tuned_elevation: the sun's elevation (degrees) the water's colour
        was tuned in: the water is lit relative to that daylight
    :ivar haze_color: ignored (the atmospheres haze the sea); kept so older
        looks still build
    :ivar haze_distance: ignored, likewise
    """

    water_color: tuple[float, float, float] = (0.02, 0.06, 0.14)
    ripple_strength: float = 0.02
    wind_dir: tuple[float, float] = (1.0, 0.0)
    wind_strength: float = 0.5
    swell_amplitude: float = 20.0
    swell_scale: float = 0.004
    swell_drift: float = 0.1
    fm_depth: float = 0.2
    shallow_color: tuple[float, float, float] = SHALLOW_WATER_COLOR
    shallow_depth: float = SHALLOW_WATER_DEPTH_M
    foam_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    foam_depth: float = 2.5
    foam_spacing: float = 0.6
    foam_speed: float = 1.2
    foam_strength: float = 0.85
    swell_shallow: float = 5.0
    swell_deep: float = 60.0
    near_band: tuple[float, float] = (12000.0, 18000.0)
    anchor_range: float = 50000.0
    tuned_elevation: float = 35.0
    haze_color: tuple[float, float, float] = (0.2, 0.2, 0.4)
    haze_distance: float = 150000.0

    #: The fields that are not Sea's arguments: the world's own, and those
    #: kept for older looks
    WORLD_FIELDS = ("tuned_elevation", "haze_color", "haze_distance")


@dataclass(frozen=True)
class BodyLook:
    """One planet or moon.

    :ivar name: what :meth:`World.body` finds it by
    :ivar spec: its planet (a preset, or a level file's, see
        :func:`world_builder.terrain.level.load_level`)
    :ivar position: its centre, world frame, metres
    :ivar mass: its mass, in the units of the gravity constant given to
        :meth:`World.gravity_field` (keyword only: it has no default)
    :ivar atmosphere: its air; None = airless
    :ivar clouds: its planet clouds; None = none (needs air)
    :ivar ocean: its sea's near look; None = the terrain draws its sea as
        flat water only (either needs ``spec.has_ocean``)
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
    mass: float = field(kw_only=True)


@dataclass(frozen=True)
class FieldLook:
    """An asteroid field or a planet's ring
    (:class:`~world_builder.asteroids.field.AsteroidField`'s arguments of the
    same names). Its rocks are massless; they turn with the world's clock.

    :ivar name: what :meth:`World.field` finds it by
    :ivar spec: the field
    :ivar position: its centre, world frame, metres (a ring's planet's)
    :ivar orientation: world from field, unit quaternion (w, x, y, z): a
        ring lies in the field's xy plane
    :ivar keep_outs: room no rock enters, added to the spec's
    :ivar ambient: the sky light on its rocks, linear; None = the world's
        ``space_ambient``
    :ivar exposure: its exposure (shared tonemap)
    :ivar far: no rock farther is drawn (they fade into its haze); None =
        as far as each is drawn by its size
    :ivar haze: draw the rocks not drawn as a haze
    """

    name: str
    spec: FieldSpec
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    keep_outs: tuple[KeepOut, ...] = ()
    ambient: tuple[float, float, float] | None = None
    exposure: float = 0.9
    far: float | None = None
    haze: bool = True


# ── the parts of one body ────────────────────────────────────────────────────


@dataclass
class Body:
    """A built body: its look and its parts.

    :ivar look: what it was built from
    :ivar terrain: its terrain
    :ivar atmosphere: its index in the world's :class:`Atmospheres`; None =
        airless
    :ivar clouds: its planet clouds, or None
    :ivar ocean: its sea's near look, or None
    :ivar ocean_reference: the light its water's colour was tuned in (see
        :meth:`World._light_ocean`); set on the first update
    :ivar shadow_map: its mountains' shadow (one per body)
    """

    look: BodyLook
    terrain: Terrain
    atmosphere: int | None = None
    clouds: SphericalClouds | None = None
    ocean: Sea | None = None
    ocean_reference: np.ndarray | None = None
    shadow_map: ShadowMap | None = None

    @property
    def name(self) -> str:
        return self.look.name

    @property
    def position(self) -> np.ndarray:
        return self.terrain.position

    @property
    def radius(self) -> float:
        return self.terrain.radius

    @property
    def mass(self) -> float:
        return self.look.mass

    def parts(self) -> list:
        """:returns: its parts that take world settings"""
        parts = [self.terrain]
        if self.clouds is not None:
            parts.append(self.clouds)
        if self.ocean is not None:
            parts.append(self.ocean)
        return parts


# ── the world ────────────────────────────────────────────────────────────────


class World:
    """Every WorldBuilder part around a game's camera: bodies, sky, sun and
    lights (see the module).

    :ivar bodies: the built bodies, in the order given
    :ivar fields: the asteroid fields and rings, in the order given
    :ivar atmospheres: the bodies' air, one :class:`Atmospheres` for all
    :ivar shadows: the bodies as sun occluders (eclipses), one
        :class:`Shadows` for all, in the order of :attr:`bodies`
    :ivar sky: the sky, or None
    :ivar executor: the worker pool every terrain and cloud layer shares
    :ivar settings: what every part costs
    :ivar sun_light: Panda3D's sun on render (None without lights)
    :ivar ambient_light: Panda3D's ambient light on render (None without)
    """

    #: The game's split cameras (:meth:`use_split_depth`), and the clouds'
    #: half-resolution layers when the settings ask for them
    depth: SplitDepthCameras | None = None
    half_res: HalfResClouds | None = None

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
        space_ambient: ArrayLike = (0.025, 0.025, 0.03),
        fields: Sequence[FieldLook] = (),
    ) -> None:
        """
        :param app: the ShowBase: its render, camera, lens and window
        :param bodies: the planets and moons; none for open space (the sky,
            the sun and the lights alone)
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
        :param clock: game time in seconds, for the swell and the rocks'
            spin; None = Panda's frame time
        :param backend: where chunks and cloud tiles are generated
            (:class:`ChunkExecutor`)
        :param space_ambient: the sky light where there is no body (open
            space), linear
        :param fields: the asteroid fields and rings
        :raises ValueError: two bodies (or fields) of one name, clouds
            without air, or an ocean on a dry planet
        """
        names = [look.name for look in bodies]
        if len(set(names)) != len(names):
            raise ValueError(f"body names must differ: {names}")
        field_names = [look.name for look in fields]
        if len(set(field_names)) != len(field_names):
            raise ValueError(f"field names must differ: {field_names}")
        for look in bodies:
            if look.clouds is not None and look.atmosphere is None:
                raise ValueError(f"{look.name}: clouds need an atmosphere")
            if look.ocean is not None and not look.spec.has_ocean:
                raise ValueError(f"{look.name}: an ocean needs spec.has_ocean")

        self.app = app
        self.settings = settings or WorldSettings()
        self._sun = self._unit(sun.direction)
        self._sun_color = np.asarray(sun.color, dtype=np.float64)
        self._space_ambient = np.asarray(space_ambient, dtype=np.float64)
        if clock is None:
            frame_clock = ClockObject.get_global_clock()
            clock = frame_clock.get_frame_time
        self.executor = ChunkExecutor(
            backend=backend, workers=self.settings.terrain.workers or None
        )
        # The sky's glow is made in the pool while the bodies are built.
        stars = None
        if sky is not None:
            stars = sky.stars if sky.stars is not None else Starfield()
            if isinstance(stars, Starfield):
                stars.prefetch_glow(self.executor, self.settings.sky.glow_size)

        # Every body's air and shadow first: the terrains and clouds bind
        # them all.
        self.atmospheres = Atmospheres(self.settings.atmosphere)
        self.shadows = Shadows(sun.angular_radius, self.settings.shadows)
        indices = []
        for look in bodies:
            # What the body holds reaches its highest ground, or its air's top.
            top = PLANET_HEIGHT_BOUNDS[1]
            if look.atmosphere is not None:
                top = max(top, look.atmosphere.height)
            body_index = self.shadows.add(
                look.spec.radius, look.position, reach=look.spec.radius + top
            )
            index = None
            if look.atmosphere is not None:
                index = self.atmospheres.add(
                    look.spec.radius, look.atmosphere, look.position
                )
                self.shadows.set_atmosphere(index, body_index)
            indices.append(index)
        self.shadows.set_sun(self._sun)

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
            self.shadows.bind(terrain.root, len(self.bodies))
            # One texture set for every body (they load once)
            materials = materials or terrain.materials
            body = Body(look=look, terrain=terrain, atmosphere=index)
            body.shadow_map = ShadowMap(
                terrain,
                self.executor,
                self.settings.shadows.terrain_map_size,
                self._sun,
            )
            body.shadow_map.bind(terrain.root)
            if look.clouds is not None:
                body.clouds = self._make_clouds(look, index)
                self.shadows.bind(body.clouds.root, len(self.bodies))
                body.clouds.bind_shadow_inputs(terrain.root)  # their shadow
                body.shadow_map.bind(body.clouds.root)  # the peaks' on them
            if look.ocean is not None:
                kwargs = {
                    name: getattr(look.ocean, name)
                    for name in look.ocean.__dataclass_fields__
                    if name not in OceanLook.WORLD_FIELDS
                }
                body.ocean = Sea(
                    app,
                    terrain,
                    clock,
                    settings=self.settings.ocean,
                    name=f"{look.name}_sea",
                    **kwargs,
                )
            self.bodies.append(body)

        self.fields: list[AsteroidField] = []
        self._field_looks = list(fields)
        for look in fields:
            ambient = self._space_ambient if look.ambient is None else look.ambient
            self.fields.append(
                AsteroidField(
                    app,
                    look.spec,
                    clock=clock,
                    settings=self.settings.asteroids,
                    keep_outs=look.keep_outs,
                    executor=self.executor,
                    position=look.position,
                    orientation=look.orientation,
                    sun_direction=tuple(self._sun),
                    sun_color=tuple(self._sun_color),
                    ambient=tuple(ambient),
                    exposure=look.exposure,
                    materials=materials,
                    far=look.far,
                    name=look.name,
                    haze=look.haze,
                )
            )

        self.sky = None
        if sky is not None:
            self.sky = Sky(
                render,
                stars,
                self.atmospheres,
                star_intensity=sky.star_intensity,
                star_brightness=sky.star_brightness,
                nebula=sky.nebula,
                sun_radiance=sky.sun_radiance,
                exposure=sky.exposure,
                star_hpr=sky.star_hpr,
                star_adaptation=sky.star_adaptation,
                settings=self.settings.sky,
                sun_angular_radius=sun.angular_radius,
            )
            self.shadows.bind(self.sky.node)
            self.sky.node.hide(HIDE_FROM_NEAR)  # the far camera draws it
            self.sky.set_sun(self._sun)

        self.light_scale = float(light_scale)
        #: The game's models lit where they are (:meth:`track_light`)
        self.tracked = TrackedLights(render)
        self.sun_light = self.ambient_light = None
        if lights:
            self.sun_light = render.attach_new_node(DirectionalLight("world_sun"))
            self.ambient_light = render.attach_new_node(AmbientLight("world_ambient"))
            render.set_light(self.sun_light)
            render.set_light(self.ambient_light)

        #: The camera's world position at the last update
        self.camera_world = np.asarray(
            bodies[0].position if bodies else (0.0, 0.0, 0.0), dtype=np.float64
        )

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

    def field(self, name: str) -> AsteroidField:
        """:returns: the asteroid field of that name
        :raises KeyError: for an unknown name"""
        for look, built in zip(self._field_looks, self.fields):
            if look.name == name:
                return built
        raise KeyError(name)

    @property
    def sun(self) -> np.ndarray:
        """Toward the sun, world frame (unit)."""
        return self._sun.copy()

    @property
    def loading(self) -> bool:
        """Whether any terrain still waits for ground, any clouds for
        tiles, or any field for its rocks: for a game's loading phase. Wait
        for it to stay False for a moment rather than for one frame (see
        :attr:`Terrain.loading`)."""
        return any(
            body.terrain.loading
            or (body.clouds is not None and body.clouds.stats["pending"] > 0)
            or (body.shadow_map is not None and body.shadow_map.loading)
            for body in self.bodies
        ) or any(asteroids.loading for asteroids in self.fields)

    def camera_render(self) -> np.ndarray:
        """:returns: the camera's position in render, float64 (the origin
        without a camera)"""
        camera = getattr(self.app, "camera", None)
        if camera is None:
            return np.zeros(3)
        return np.array(camera.get_pos(self.app.render), dtype=np.float64)

    def camera_forward(self) -> np.ndarray | None:
        """:returns: the camera's view direction (render's axes are the
        world's), or None without a camera"""
        camera = getattr(self.app, "camera", None)
        if camera is None:
            return None
        f = camera.get_quat(self.app.render).get_forward()
        return np.array([f.x, f.y, f.z])

    def depth_ranges(self, origin_world: ArrayLike) -> tuple[float, float]:
        """Split and far distances for :class:`SplitDepthCameras` this
        frame: the split follows the nearest ground, the far plane reaches
        the farthest visible part of any body.

        :param origin_world: render's origin in the world frame (float64)
        :returns: (split, far), metres; in open space (no body) a fixed
            split and :data:`SPACE_FAR_M`
        """
        if not self.bodies:
            return SPACE_SPLIT_M, SPACE_FAR_M
        camera = np.asarray(origin_world, dtype=np.float64) + self.camera_render()
        return SplitDepthCameras.system_ranges(
            (
                body.radius,
                float(np.linalg.norm(camera - body.position)),
                body.terrain.planet.height_bounds[1],
                0.0
                if body.look.spec.has_ocean
                else body.terrain.planet.height_bounds[0],
            )
            for body in self.bodies
        )

    def gravity_field(self, g_constant: float = 1.0) -> Gravity:
        """The bodies' gravity as point masses (see :mod:`world_builder.gravity`).
        Build it once and evaluate it for many points: the bodies do not move.

        :param g_constant: G, in the units of the bodies' masses
        :returns: the field; ``acceleration(points)`` is the raw gravity vector
        """
        return Gravity.from_bodies(self.bodies, g_constant)

    def nearest(self, camera_world: ArrayLike) -> Body:
        """:returns: the body whose sea level is nearest a point"""
        point = np.asarray(camera_world, dtype=np.float64)
        return min(
            self.bodies,
            key=lambda b: float(np.linalg.norm(point - b.position)) - b.radius,
        )

    def light_at(self, point_world: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """The light at a point, as the terrain's shader lights the ground
        there: through the nearest body's air, or in its shadow if airless,
        and in the other world shadows (:meth:`sun_visibility`).

        :param point_world: world frame, metres
        :returns: (direct, ambient), linear rgb; in open space (no body) the
            bare sun and ``space_ambient``
        """
        direct, ambient = self.lights_at(np.asarray(point_world, float)[None, :])
        return direct[0], ambient[0]

    def lights_at(
        self, points_world: ArrayLike, local: bool | ArrayLike = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`light_at` for many points at once, each by its nearest
        body.

        :param points_world: (n, 3) world frame, metres
        :param local: whether the local shadows (the planet's clouds', its
            mountains') shade each point (one flag, or one per point): False
            for what the shaders shade per pixel (the near sea)
        :returns: (direct, ambient), (n, 3) each, linear rgb
        """
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if not self.bodies:
            shape = (len(points), 3)
            return (
                np.broadcast_to(self._sun_color, shape).copy(),
                np.broadcast_to(self._space_ambient, shape).copy(),
            )
        shaded = np.broadcast_to(np.asarray(local, dtype=bool), (len(points),))
        direct = np.empty((len(points), 3))
        ambient = np.empty((len(points), 3))
        nearest = self._nearest_indices(points)
        for index in np.unique(nearest):
            mask = nearest == index
            direct[mask], ambient[mask] = self._lights_at(
                int(index), points[mask], shaded[mask]
            )
        return direct, ambient

    def sun_visibility(self, points_world: ArrayLike) -> np.ndarray:
        """The share of the sun the world's shadows leave each point, each
        by its nearest body: the product of the shadow terms the shaders
        draw (:mod:`world_builder.shadows`: the other bodies' eclipses and
        the planet's clouds). The nearest body's own shadow is not in it:
        that is its air's or its day and night (:meth:`light_at` has both).

        :param points_world: (n, 3) world frame, metres
        :returns: (n,) in [0, 1]
        """
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        visible = np.ones(len(points))
        if not self.bodies:
            return visible
        nearest = self._nearest_indices(points)
        for index in np.unique(nearest):
            mask = nearest == index
            eclipse, clouds = self._sun_visibility(int(index), points[mask])
            visible[mask] = eclipse * clouds
        return visible

    def _nearest_indices(self, points: np.ndarray) -> np.ndarray:
        """:returns: (n,) each point's nearest body (by sea level), an index
        into :attr:`bodies`"""
        centres = np.array([body.position for body in self.bodies])
        radii = np.array([body.radius for body in self.bodies])
        gaps = np.linalg.norm(points[:, None, :] - centres[None], axis=-1) - radii
        return np.argmin(gaps, axis=1)

    def _sun_visibility(
        self, index: int, points: np.ndarray, local: ArrayLike = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`sun_visibility` for points of body *index*, as its terms:
        (the eclipses, the local shadows where *local*: the clouds', the
        mountains'), (n,) each."""
        eclipse = self.shadows.eclipse(points, self._sun, index)
        shade = np.ones(len(points))
        body = self.bodies[index]
        shaded = np.broadcast_to(np.asarray(local, dtype=bool), (len(points),))
        if not shaded.any():
            return eclipse, shade
        steps = self.shadows.settings.cloud_shadow_steps
        if body.clouds is not None and steps > 0:
            shade[shaded] = body.clouds.sun_visibility(points[shaded], self._sun, steps)
        if body.shadow_map is not None:
            shade[shaded] *= body.shadow_map.visibility(
                points[shaded], self.shadows.sun_angular_radius
            )
        return eclipse, shade

    def _lights_at(
        self, index: int, points: np.ndarray, local: ArrayLike = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`lights_at` for (n, 3) points of body *index*."""
        body = self.bodies[index]
        ambient = np.asarray(body.look.ambient, dtype=np.float64)
        if body.atmosphere is not None:
            direct, ambient = self.atmospheres.light_at(
                body.atmosphere, points, self._sun, self._sun_color, ambient
            )
        else:
            hidden = in_shadow(points - body.position, self._sun, body.radius)
            direct = np.where(np.asarray(hidden)[..., None], 0.0, self._sun_color)
        eclipse, shade = self._sun_visibility(index, points, local)
        # The sky's light dims in an eclipse, not under a cloud (lit air
        # all round).
        sky = ECLIPSE_SKY_FLOOR + (1.0 - ECLIPSE_SKY_FLOOR) * eclipse[:, None]
        return (
            np.broadcast_to(direct * (eclipse * shade)[:, None], (len(points), 3)),
            np.broadcast_to(ambient * sky, (len(points), 3)),
        )

    # ── objects ─────────────────────────────────────────────────────────────

    def track_light(self, node: NodePath, cast: bool = False) -> None:
        """Light a game's model by the light at its own position, not the
        camera's: its own Panda3D sun and ambient lights (the world's are
        off on it), set at every :meth:`update` from :meth:`lights_at`, in
        the world's shadows, times ``light_scale``. Its position is read
        there: place it before. See :class:`world_builder.lights.TrackedLights`.

        :param node: the model (its children are lit alike)
        :param cast: whether it casts shadows on the world (kept for later:
            not drawn yet)
        """
        self.tracked.track(node, cast)

    def untrack_light(self, node: NodePath) -> None:
        """Back to the world's lights (the light at the camera)."""
        self.tracked.untrack(node)

    # ── changes ─────────────────────────────────────────────────────────────

    def set_sun(self, direction: ArrayLike) -> None:
        """Move the sun (toward it, world frame, any length): every part
        follows at once, the lights at the next :meth:`update`."""
        self._sun = self._unit(direction)
        self.shadows.set_sun(self._sun)
        if self.sky is not None:
            self.sky.set_sun(self._sun)
        for body in self.bodies:
            body.terrain.set_sun(
                tuple(self._sun), tuple(self._sun_color), body.look.ambient
            )
            if body.clouds is not None:
                body.clouds.set_sun(self._sun)
            if body.shadow_map is not None:
                body.shadow_map.set_sun(self._sun)  # baked again, in the pool
        for look, built in zip(self._field_looks, self.fields):
            ambient = self._space_ambient if look.ambient is None else look.ambient
            built.set_sun(self._sun, self._sun_color, ambient)

    def set_settings(self, settings: WorldSettings) -> None:
        """Change what every part costs (see :meth:`WorldSettings.apply`)."""
        self.settings = settings
        parts: list = [self.atmospheres, self.shadows]
        if self.sky is not None:
            parts.append(self.sky)
        for body in self.bodies:
            parts += body.parts()
        parts += self.fields
        settings.apply(*parts)
        for body in self.bodies:
            if body.shadow_map is not None:
                body.shadow_map.set_size(settings.shadows.terrain_map_size)
        self._sync_half_res()

    def use_split_depth(self, depth: SplitDepthCameras) -> None:
        """The split cameras the game built after the world: the clouds'
        half-resolution layers (``CloudSettings.half_resolution``) mirror
        them, so the world needs them for that setting. Call
        ``depth.update`` before :meth:`update` every frame, as always."""
        self.depth = depth
        self._sync_half_res()

    def _sync_half_res(self) -> None:
        """Build or remove the clouds' half-resolution layers as the
        settings ask (they need the split cameras)."""
        clouds = [body.clouds for body in self.bodies if body.clouds is not None]
        wanted = (
            self.settings.clouds.half_resolution
            and self.depth is not None
            and bool(clouds)
        )
        if wanted and self.half_res is None:
            self.half_res = HalfResClouds(self.app, self.depth)
            if self.sky is not None:
                self.half_res.exclude(self.sky.node)
            for cloud in clouds:
                self.half_res.add(cloud.root)
        elif not wanted and self.half_res is not None:
            self.half_res.destroy()
            self.half_res = None

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
        forward = self.camera_forward()

        for body in self.bodies:
            body.terrain.update(camera, camera_render, camera_forward=forward)
            if body.shadow_map is not None:
                body.shadow_map.update()
        for asteroids in self.fields:
            asteroids.update(camera, camera_render)
        self.atmospheres.update(camera)
        self.shadows.update(camera)
        if self.sky is not None:
            if far is None:
                far = self.depth_ranges(origin_world)[1]
            self.sky.update(tuple(camera_render), far)
        pixel_angle = self._pixel_angle()
        if self.half_res is not None:
            self.half_res.update()
            if pixel_angle is not None:
                pixel_angle *= self.half_res.pixel_scale
        for body in self.bodies:
            if body.clouds is not None:
                if pixel_angle is not None:
                    body.clouds.set_pixel_angle(pixel_angle)
                body.clouds.update(camera, camera_render)
        for body in self.bodies:
            if body.ocean is not None:
                body.ocean.update(camera)
        self._light(camera, np.asarray(origin_world, dtype=np.float64))

    def _pixel_angle(self) -> float | None:
        """:returns: the angle one pixel of the camera's output spans
        (vertical fov / its height), or None without a camera"""
        lens = getattr(self.app, "camLens", None)
        cam = getattr(self.app, "cam", None)
        if lens is None or cam is None or cam.node().get_num_display_regions() == 0:
            return None
        height = cam.node().get_display_region(0).get_pixel_height()
        return math.radians(lens.get_fov()[1]) / max(height, 1)

    def _light(self, camera: np.ndarray, origin_world: np.ndarray) -> None:
        """Panda3D's sun and ambient lights as they are at the camera, each
        tracked node's as they are where it is, and each near sea's look as
        the sea under the camera: one :meth:`lights_at` for them all (each
        call costs ~0.2 ms, whatever its points)."""
        seas = [
            body
            for body in self.bodies
            if body.ocean is not None and body.ocean.near > 0.0
        ]
        ups = [
            (camera - b.position) / np.linalg.norm(camera - b.position) for b in seas
        ]
        tracked = origin_world + self.tracked.positions()
        if self.sun_light is None and not seas and not len(tracked):
            return
        points = [camera[None, :], tracked]
        points += [
            (body.position + up * body.radius)[None, :] for body, up in zip(seas, ups)
        ]
        # The near seas take the local shadows per pixel (sea.glsl).
        local = np.arange(1 + len(tracked) + len(seas)) < 1 + len(tracked)
        direct, ambient = self.lights_at(np.concatenate(points), local)
        scale = self.light_scale
        if self.sun_light is not None:
            self.sun_light.node().set_color((*(direct[0] * scale), 1.0))
            self.sun_light.node().set_direction(Vec3(*(-self._sun)))
            self.ambient_light.node().set_color((*(ambient[0] * scale), 1.0))
        count = len(tracked)
        if count:
            self.tracked.apply(
                direct[1 : 1 + count] * scale, ambient[1 : 1 + count] * scale, self._sun
            )
        for i, (body, up) in enumerate(zip(seas, ups), start=1 + count):
            self._light_ocean(body, up, direct[i], ambient[i])

    def _light_ocean(
        self, body: Body, up: np.ndarray, direct: np.ndarray, ambient: np.ndarray
    ) -> None:
        """Light the sea's near look as the sea under the camera (*direct*
        and *ambient* there, the clouds' shadow left out), relative to the
        daylight its colour was tuned in (dark at night): the sun's share
        apart, which the clouds' shadow dims per pixel."""
        sea = body.position + up * body.radius
        sun_light = direct * max(float(self._sun @ up), 0.0)
        light = sun_light + ambient
        if body.ocean_reference is None:
            elevation = math.radians(body.look.ocean.tuned_elevation)
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
            body.ocean_reference = sun * math.sin(elevation) + np.asarray(
                body.look.ambient
            )
        # As lit as before the split (the sum is clipped), shared as the
        # sun and the sky share it.
        total = np.clip(light / body.ocean_reference, 0.0, 1.5)
        sun_share = np.divide(sun_light, light, out=np.zeros(3), where=light > 0.0)
        body.ocean.set_light(total * (1.0 - sun_share), sun=total * sun_share)

    # ── teardown ────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Remove every part and the lights, and stop the workers."""
        if self.half_res is not None:
            self.half_res.destroy()
            self.half_res = None
        self.tracked.clear()
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
        for asteroids in self.fields:
            asteroids.destroy()
        self.fields = []
        self._field_looks = []
        self.executor.shutdown()
