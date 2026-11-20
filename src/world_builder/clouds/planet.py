"""
A planet's clouds as one part: the low clouds and the cirrus above them,
one raymarch.

:class:`PlanetClouds` is what a game (or :class:`world_builder.world.World`)
builds for a planet. It owns the **volume**: the low clouds, cumulus by
default (any :class:`cloud.CloudType`, or the genera), raymarched
(:class:`march.CloudMarch`; it needs the split cameras,
:meth:`PlanetClouds.use_split_depth`), with a deck sphere for the sea's
reflection, and, with ``cirrus=True``, the weather's cirrus in the march's
high layer (:func:`weather.cirrus_cover_map`; clouds.md, "The high layer").

The layers follow the planet's weather (the same cyclones carry the cumulus
and bend the cirrus), the sun, the settings and the camera. They have rules
of their own (:func:`check_layers`): the high layer's slab must lie above
the volume's slabs (all the genera's) and inside the air; the genera must
fit the march (how many, and their bands of sky).

Design notes: docs/source/clouds.md and docs/source/atmosphere.md, "Planet
clouds".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import NodePath

from world_builder.atmosphere import Atmospheres
from world_builder.clouds.cloud import PRESETS, CloudSettings, CloudType
from world_builder.clouds.genera import HIGH_SLAB, GeneraSpec, genus_specs
from world_builder.clouds.march import MAX_SKY_BANDS, CloudMarch, sky_bands
from world_builder.clouds.noise import MAX_GENERA, DensityField
from world_builder.clouds.weather import WeatherSpec
from world_builder.terrain.planet import PlanetSpec

if TYPE_CHECKING:
    from world_builder.depth import SplitDepthCameras
    from world_builder.terrain.workers import ChunkExecutor


def layer_slabs(
    cloud_type: CloudType, altitude: float | None, genera: GeneraSpec | None = None
) -> list[tuple[float, float]]:
    """The volume's slabs, as the march builds them: one per genus
    (:func:`genera.genus_specs`), or the one type's.

    :param cloud_type: the volume's type (ignored with genera)
    :param altitude: the volume's (the convective genus's) base, metres;
        None = the type's
    :param genera: the genera (:mod:`genera`); None = one type
    :returns: each slab's (base, thickness) above sea level, metres
    """
    specs = (
        genus_specs(genera, altitude)
        if genera is not None
        else [(None, cloud_type, altitude, None, None)]
    )
    slabs = []
    for _, kind, base, thickness, _ in specs:
        preset = PRESETS[kind].field.slab
        slabs.append(
            (
                float(preset[0] if base is None else base),
                float(preset[1] if thickness is None else thickness),
            )
        )
    return slabs


def check_layers(
    cloud_type: CloudType,
    altitude: float | None,
    cirrus: bool = False,
    air_top: float | None = None,
    genera: GeneraSpec | None = None,
) -> None:
    """Refuse layers the march or the air cannot hold: with genera, no more
    than :data:`noise.MAX_GENERA` of them, their slabs in no more than
    :data:`march.MAX_SKY_BANDS` bands of sky (:func:`march.sky_bands`; the
    genera's slabs may overlap: the march takes them as one); the high
    layer's slab (:attr:`genera.GeneraSpec.high_slab`, without genera
    :data:`genera.HIGH_SLAB`), where it has a genus or the weather's cirrus,
    must clear the union of the slabs, its top stay under the air's.

    :param cloud_type: the volume's type (the convective one with genera)
    :param altitude: the volume's base, metres; None = the type's
    :param cirrus: the march draws the weather's cirrus (its high layer)
    :param air_top: the air's top above sea level, metres; None = unchecked
    :param genera: the genera (:mod:`genera`); None = one type
    :raises ValueError: for layers that do not fit
    """
    if genera is not None and len(genera.genera) > MAX_GENERA:
        raise ValueError(
            f"{len(genera.genera)} genera: the march takes at most {MAX_GENERA}"
        )
    slabs = layer_slabs(cloud_type, altitude, genera)
    try:
        bands = sky_bands(slabs)
    except ValueError:
        raise ValueError(
            f"the genera's slabs make more than {MAX_SKY_BANDS} bands of sky "
            f"(slabs {slabs}): merge or overlap some"
        ) from None
    top = bands[-1][0] + bands[-1][1]
    if cirrus or (genera is not None and genera.high):
        # The high layer: a band of its own above the union (its crossings
        # come before and after the bands'), inside the air
        base, thickness = genera.high_slab if genera is not None else HIGH_SLAB
        if base <= top:
            raise ValueError(
                f"the high layer's base ({base:g} m) must be above the "
                f"clouds' top ({top:g} m)"
            )
        if air_top is not None and base + thickness >= air_top:
            raise ValueError(
                f"the high layer's top ({base + thickness:g} m) must be inside "
                f"the air ({air_top:g} m)"
            )


class PlanetClouds:
    """One planet's clouds: the low clouds' volume, and the weather's
    cirrus in its high layer when asked for (see the module).

    Usage::

        clouds = PlanetClouds(render, planet_radius, executor,
                              atmospheres=atmospheres, atmosphere=0,
                              cirrus=True)
        clouds.set_sun(sun_direction)
        # every frame, camera in the world frame (float64):
        clouds.update(camera_world)

    :ivar root: the node the volume hangs under (:meth:`set_visible` hides
        the clouds: a march draws in passes of its own)
    :ivar volume: the low clouds and the cirrus (:class:`march.CloudMarch`)
    :ivar weather: the weather they follow
    :ivar radius: the planet's sea-level radius, metres
    :ivar position: its centre, world frame
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = CloudSettings

    def __init__(
        self,
        parent: NodePath,
        radius: float,
        executor: ChunkExecutor,
        position: ArrayLike = (0.0, 0.0, 0.0),
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        cloud_type: CloudType = CloudType.CUMULUS,
        altitude: float | None = None,
        cirrus: bool = False,
        planet: PlanetSpec | None = None,
        weather: WeatherSpec | None = None,
        seed: int = 7,
        weather_seed: int = 11,
        settings: CloudSettings | None = None,
        sun_brightness: float = 2.8,
        sky_strength: float = 0.35,
        sky_occlusion: float = 0.55,
        exposure: float = 0.9,
        genera: GeneraSpec | None = None,
    ):
        """
        :param parent: usually ``render``
        :param radius: the planet's sea-level radius, metres
        :param executor: the world's :class:`terrain.workers.ChunkExecutor`
        :param position: the planet centre in the world frame (float64)
        :param atmospheres: the world's atmospheres (the clouds are seen
            through them)
        :param atmosphere: the planet's own atmosphere index (reddens the
            clouds' sunlight, and its top bounds the high layer); None = none
        :param cloud_type: the volume's cloud type (its preset's field and
            optics)
        :param altitude: the volume's base above sea level, metres; None
            keeps the type's
        :param cirrus: the march's high layer draws the weather's cirrus
            (:func:`weather.cirrus_cover_map`, streaming along the upper
            wind; :attr:`genera.GeneraSpec.cirrus_weather`); False = only the
            genera's cirrus features, if any
        :param planet: the planet the weather follows (climate, terrain);
            None = an earthlike one of this radius
        :param weather: the weather in full; None = the defaults for
            *planet* with *weather_seed*
        :param seed: the volume's noise seed (field and placement)
        :param weather_seed: the weather systems' seed
        :param settings: what the clouds cost (:class:`cloud.CloudSettings`;
            :meth:`set_settings` changes it); None = HIGH
        :param sun_brightness: direct-term gain
        :param sky_strength: ambient sky weight (live:
            :meth:`set_sky_light`)
        :param sky_occlusion: the volume's share of sky light reaching its
            slab's underside
        :param exposure: radiance scale before the sky's tonemap
        :param genera: the low clouds by genus (:mod:`genera`): the march
            marches every genus as one (*cloud_type* is
            :attr:`cloud.CloudType.CONVECTIVE` for the primary genus,
            *weather* the genera's); None = one type
        :raises ValueError: a high layer not above the volume or not inside
            the air, too many genera or bands of sky (:func:`check_layers`)
        """
        air_top = None
        if atmospheres is not None and atmosphere is not None:
            air_top = atmospheres.params(atmosphere).height
        check_layers(cloud_type, altitude, cirrus, air_top, genera)
        settings = settings or CloudSettings()
        self.genera = genera
        if genera is not None:
            cloud_type = CloudType.CONVECTIVE
            weather = genera.weather
        if weather is None:
            planet = planet or PlanetSpec(name="clouds", radius=float(radius))
            weather = WeatherSpec(planet, seed=weather_seed)
        self.weather = weather
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=np.float64)
        self._settings = settings
        self.root = parent.attach_new_node("planet_clouds")
        # What the volume is built from.
        self._volume_args = dict(
            radius=radius,
            executor=executor,
            position=position,
            atmospheres=atmospheres,
            atmosphere=atmosphere,
            cloud_type=cloud_type,
            altitude=altitude,
            weather=weather,
            seed=seed,
            sun_brightness=sun_brightness,
            sky_strength=sky_strength,
            sky_occlusion=sky_occlusion,
            exposure=exposure,
        )
        if genera is not None:
            self._volume_args.update(genera=genera)
        if cirrus:
            self._volume_args.update(cirrus=True)
        self._split: tuple[Any, SplitDepthCameras] | None = None
        # The live state a new volume is given (set_sun, set_sky_light,
        # set_debug).
        self._sun: tuple = ((0.0, 0.0, 1.0), None, None)
        self._sky_light: float | None = None
        self._debug = 0
        # The light resources' binders (bind_maps), again for a new volume.
        self._map_binders: list[Callable[[NodePath], None]] = []
        # The ground shadow pass's readers (the terrain) and its card's binders
        self._ground_nodes: list[NodePath] = []
        self._ground_binders: list[Callable[[NodePath], None]] = []
        # The nodes the clouds shade (bind_shadow_inputs), again for a new
        # volume.
        self._shadow_nodes: list[NodePath] = []
        self.volume = self._make_volume(settings)

    def _make_volume(self, settings: CloudSettings) -> CloudMarch:
        volume = CloudMarch(self.root, settings=settings, **self._volume_args)
        if self._split is not None:
            volume.attach(*self._split)
        return volume

    # ── settings ──────────────────────────────────────────────────────────────

    @property
    def settings(self) -> CloudSettings:
        """What the clouds cost (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: CloudSettings) -> None:
        """Change what the clouds cost (see
        :meth:`march.CloudMarch.set_settings`).

        :param settings: the new settings
        """
        self._settings = settings
        self.volume.set_settings(settings)

    def bind_maps(self, bind: Callable[[NodePath], None]) -> None:
        """Have a raymarched volume read a body's cloud shadow maps for its own
        light (:meth:`march.CloudMarch.bind_maps`).

        :param bind: sets the maps' inputs on a node
            (:meth:`shadow_map.CloudShadowMaps.bind`)
        """
        self._map_binders.append(bind)
        self.volume.bind_maps(bind)

    def bind_ground_shadow(
        self, node: NodePath, binders: list[Callable[[NodePath], None]]
    ) -> None:
        """Let a node's shaders (the planet's terrain: terrain_frag.glsl's
        ``uGroundShadow``) read the clouds' shadow on the view's scene from a
        raymarch's pass (:meth:`march.CloudMarch.bind_ground_shadow`) instead of
        shading it per pixel.

        :param node: the node that reads it
        :param binders: set the pass's own inputs beyond the clouds' (the
            world's ``Shadows.bind``: the shadow's steps)
        """
        self._ground_nodes.append(node)
        self._ground_binders.extend(binders)
        self._bind_ground()

    def _bind_ground(self) -> None:
        self._ground_nodes = [n for n in self._ground_nodes if not n.is_empty()]
        for node in self._ground_nodes:
            self.volume.bind_ground_shadow(node, self._ground_binders)

    def shadow_caster(self) -> Any:
        """The caster of the clouds' shadow, all their genera's together
        (:meth:`march.CloudMarch.shadow_caster`)."""
        return self.volume.shadow_caster()

    def bind_shadow_inputs(self, node: NodePath) -> None:
        """Let the shaders on a node (the planet's terrain, the light
        resources' bakes) darken the sun under the clouds: all their genera's
        shadow (:meth:`march.CloudMarch.bind_shadow_inputs`).

        :param node: the node whose shaders include cloud_field.glsl's
            ``planetCloudShadow``
        """
        self._shadow_nodes.append(node)
        self._shade([node])

    def _shade(self, nodes: list[NodePath]) -> None:
        for node in nodes:
            self.volume.bind_shadow_inputs(node)

    @property
    def top_radius(self) -> float:
        """How far from the planet's centre the clouds reach, metres: the
        highest layer's top (the high layer's, when it draws). What a ray
        meets past where it enters this sphere is behind them (a ring's
        haze, :meth:`asteroids.haze.FieldHaze.set_clouds_top`)."""
        band = self.volume.sky_bands[-1]  # the genera's union's top
        tops = [self.radius + band[0] + band[1]]
        if self.volume.high_slab is not None:  # the high layer's, drawn
            tops.append(self.radius + sum(self.volume.high_slab))
        return max(tops)

    @property
    def high_slab(self) -> tuple[float, float] | None:
        """The high layer's slab while it draws (the cirrus's;
        :attr:`march.CloudMarch.high_slab`), (base, thickness) above sea
        level, metres; None = no high layer."""
        return self.volume.high_slab

    @property
    def shadow_layer(self) -> int:
        """The volume's layer its shadow is marched in
        (:attr:`march.CloudMarch.shadow_layer`); -1 = none (not ready)."""
        return self.volume.shadow_layer

    @property
    def shadow_field(self) -> DensityField:
        """The field the volume's shadow is marched in."""
        return self.volume.shadow_field

    def sun_visibility(
        self, points_world: ArrayLike, sun_direction: ArrayLike, max_steps: int
    ) -> np.ndarray:
        """The share of the sun the clouds leave points under them (or in
        them), as their shaders draw it on the ground: all the genera's
        (:func:`field_setup.sun_visibility`).

        :param points_world: (..., 3) world frame, metres
        :param sun_direction: toward the sun, world frame (any length)
        :param max_steps: the most samples a chord takes
            (``ShadowSettings.cloud_shadow_steps``; 0 = none)
        :returns: (...) in [0, 1]
        """
        return self.volume.sun_visibility(points_world, sun_direction, max_steps)

    # ── live ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """``pending``: the volume's set-up still to come."""
        return dict(self.volume.stats)

    def set_sun(
        self,
        direction: ArrayLike,
        sun_color: ArrayLike | None = None,
        sky_color: ArrayLike | None = None,
    ) -> None:
        """Move the sun and/or restate its colours (uniform writes only).

        :param direction: toward the sun, world frame (need not be unit)
        :param sun_color: RGB of direct sunlight, or None to keep
        :param sky_color: RGB of the ambient sky light, or None to keep
        """
        self._sun = (
            direction,
            sun_color if sun_color is not None else self._sun[1],
            sky_color if sky_color is not None else self._sun[2],
        )
        self.volume.set_sun(direction, sun_color=sun_color, sky_color=sky_color)

    def set_sky_light(self, strength: float) -> None:
        """The sky's (ambient) light on the clouds, as ``sky_strength``."""
        self._sky_light = strength
        self.volume.set_sky_light(strength)

    def set_pixel_angle(self, radians: float, volume_scale: float = 1.0) -> None:
        """The angle one pixel of the view spans (vertical fov / its height):
        how far the volume draws its own detail before averaging it.

        :param radians: the angle per pixel of the view
        :param volume_scale: the volume's pixels over the view's
        """
        self.volume.set_pixel_angle(radians * volume_scale)

    def set_debug(self, flags: int) -> None:
        """Profiling switches (cloud_field.glsl's ``uCloudDebug`` bits).

        :param flags: the bits, 0 = normal
        """
        self._debug = flags
        self.volume.set_debug(flags)

    @property
    def genus_names(self) -> list[str]:
        """The low clouds' genera, by index (:attr:`march.CloudMarch.genus_names`)."""
        return self.volume.genus_names

    def set_genus_visible(self, genus: int | str, visible: bool) -> None:
        """Draw a genus of the low clouds, or a high genus (``"cirrus"``),
        or not, live (:meth:`march.CloudMarch.set_genus_visible`)."""
        self.volume.set_genus_visible(genus, visible)

    def genus_visible(self, genus: int | str) -> bool:
        """:returns: whether a genus is drawn"""
        return self.volume.genus_visible(genus)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the clouds (a march's passes included)."""
        if visible:
            self.root.show()
        else:
            self.root.hide()
        self.volume.set_visible(visible)

    def use_split_depth(self, app: Any, depth: SplitDepthCameras) -> None:
        """The game's split cameras, which a raymarched volume draws its
        passes with (:meth:`march.CloudMarch.attach`).

        :param app: the ShowBase
        :param depth: the split cameras
        """
        self._split = (app, depth)
        self.volume.attach(app, depth)

    def update(
        self, camera_world: ArrayLike, camera_render: ArrayLike = (0.0, 0.0, 0.0)
    ) -> None:
        """Per frame.

        :param camera_world: camera in the world frame, metres (float64)
        :param camera_render: camera in the render frame (the origin when
            the scene is drawn camera-relative)
        """
        self.volume.update(camera_world, camera_render)

    def remove(self) -> None:
        """Take the clouds off the scene and drop the volume's queued tiles."""
        self.volume.remove()
        self.root.remove_node()
