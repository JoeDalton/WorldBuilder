"""
A planet's clouds as one part: the low clouds' volume, the cirrus sheet above.

:class:`PlanetClouds` is what a game (or :class:`world_builder.world.World`)
builds for a planet. It owns two layers, one per technique:

* the **volume**: the low clouds, cumulus by default (any
  :class:`cloud.CloudType`), raymarched (:class:`march.CloudMarch`; it needs
  the split cameras, :meth:`PlanetClouds.use_split_depth`), with a deck
  sphere for the sea's reflection;
* the **sheet** (:class:`cirrus.CloudSheet`, optional): high cirrus as one
  surface above them.

They share the planet's weather (the same cyclones carry the cumulus and bend
the cirrus), the sun, the settings and the camera, so a game updates, lights
and removes them as one. The pair also has a rule of its own
(:func:`check_layers`): the sheet must lie above the volume's slab, which is
what makes its draw order hold without sorting (see :mod:`cirrus`), and
inside the air.

Design notes: docs/source/atmosphere.md, "Planet clouds" and "Cirrus".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import NodePath

from world_builder.atmosphere import Atmospheres
from world_builder.clouds.cirrus import CirrusSpec, CloudSheet
from world_builder.clouds.cloud import PRESETS, CloudSettings, CloudType
from world_builder.clouds.genera import GeneraSpec
from world_builder.clouds.march import CloudMarch
from world_builder.clouds.noise import DensityField
from world_builder.clouds.weather import WeatherSpec
from world_builder.depth import HIDE_FROM_CLOUD_PASS
from world_builder.terrain.planet import PlanetSpec

if TYPE_CHECKING:
    from world_builder.depth import SplitDepthCameras
    from world_builder.terrain.workers import ChunkExecutor


def check_layers(
    cloud_type: CloudType,
    altitude: float | None,
    cirrus: CirrusSpec | None,
    air_top: float | None = None,
    genera: GeneraSpec | None = None,
) -> None:
    """Refuse layers the draw order or the air cannot hold: the sheet's base
    must clear the volume's slab, its top stay under the air's; with genera,
    the stratus's top must stay under the convective clouds' base (the march
    takes each family's slab as up to two stretches of a ray, sorted along
    it, which needs them apart).

    :param cloud_type: the volume's type (the convective one with genera)
    :param altitude: the volume's base, metres; None = the type's
    :param cirrus: the sheet; None = none
    :param air_top: the air's top above sea level, metres; None = unchecked
    :param genera: the genera (:mod:`genera`); None = one type
    :raises ValueError: for layers that do not fit
    """
    if genera is not None:
        cloud_type = CloudType.CONVECTIVE
        stratus = PRESETS[CloudType.STRATUS].field.slab
        stratus_top = genera.stratus_altitude + stratus[1]
        base = PRESETS[cloud_type].field.slab[0] if altitude is None else altitude
        if stratus_top >= base:
            raise ValueError(
                f"the stratus's top ({stratus_top:g} m) must be under the "
                f"convective clouds' base ({base:g} m)"
            )
    if cirrus is None:
        return
    slab = PRESETS[cloud_type].field.slab
    top = (slab[0] if altitude is None else altitude) + slab[1]
    if cirrus.base <= top:
        raise ValueError(
            f"the cirrus's base ({cirrus.base:g} m) must be above the "
            f"clouds' top ({top:g} m)"
        )
    if air_top is not None and cirrus.top >= air_top:
        raise ValueError(
            f"the cirrus's top ({cirrus.top:g} m) must be inside the air "
            f"({air_top:g} m)"
        )


class PlanetClouds:
    """One planet's clouds: the low clouds' volume, and the cirrus sheet
    above it when asked for (see the module).

    Usage::

        clouds = PlanetClouds(render, planet_radius, executor,
                              atmospheres=atmospheres, atmosphere=0,
                              cirrus=CirrusSpec())
        clouds.set_sun(sun_direction)
        # every frame, camera in the world frame (float64):
        clouds.update(camera_world)

    :ivar root: the node both layers hang under (:meth:`set_visible` hides
        the clouds: a march draws in passes of its own)
    :ivar volume: the low clouds (:class:`march.CloudMarch`)
    :ivar sheet: the cirrus (:class:`cirrus.CloudSheet`), or None
    :ivar weather: the weather both follow
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
        cirrus: CirrusSpec | None = None,
        planet: PlanetSpec | None = None,
        weather: WeatherSpec | None = None,
        seed: int = 7,
        weather_seed: int = 11,
        settings: CloudSettings | None = None,
        sun_brightness: float = 2.8,
        sky_strength: float = 0.35,
        sky_occlusion: float = 0.55,
        exposure: float = 0.9,
        game: Any = None,
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
            clouds' sunlight, and its top bounds the cirrus); None = none
        :param cloud_type: the volume's cloud type (its preset's field and
            optics)
        :param altitude: the volume's base above sea level, metres; None
            keeps the type's
        :param cirrus: the sheet (:class:`cirrus.CirrusSpec`); None = no
            cirrus
        :param planet: the planet the weather follows (climate, terrain);
            None = an earthlike one of this radius
        :param weather: the weather in full; None = the defaults for
            *planet* with *weather_seed*
        :param seed: the volume's noise seed (field and placement)
        :param weather_seed: the weather systems' seed
        :param settings: what the clouds cost (:class:`cloud.CloudSettings`;
            :meth:`set_settings` changes it); None = HIGH
        :param sun_brightness: direct-term gain, both layers
        :param sky_strength: ambient sky weight, both layers (live:
            :meth:`set_sky_light`)
        :param sky_occlusion: the volume's share of sky light reaching its
            slab's underside
        :param exposure: radiance scale before the sky's tonemap, both layers
        :param game: the game object, for its asset manager (the cirrus's
            atlas; None: from disk)
        :param genera: the low clouds by genus (:mod:`genera`): the march
            marches the convective family (*cloud_type* is
            :attr:`cloud.CloudType.CONVECTIVE`, *weather* the genera's) and
            the stratus under it; None = one type
        :raises ValueError: a sheet not above the volume or not inside the
            air, a stratus not under the convective clouds
            (:func:`check_layers`)
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
        self._split: tuple[Any, SplitDepthCameras] | None = None
        # The live state a new volume is given (set_sun, set_sky_light,
        # set_debug).
        self._sun: tuple = ((0.0, 0.0, 1.0), None, None)
        self._sky_light: float | None = None
        self._debug = 0
        # What every layer's shader also takes (bind_inputs), again for a
        # new volume.
        self._binders: list[Callable[[NodePath], None]] = []
        self._map_binders: list[Callable[[NodePath], None]] = []
        # The ground shadow pass's readers (the terrain) and its card's binders
        self._ground_nodes: list[NodePath] = []
        self._ground_binders: list[Callable[[NodePath], None]] = []
        # The nodes the clouds shade (bind_shadow_inputs), again for a new
        # volume.
        self._shadow_nodes: list[NodePath] = []
        self.volume = self._make_volume(settings)
        self.sheet: CloudSheet | None = None
        if cirrus is not None:
            self.sheet = CloudSheet(
                self.root,
                radius,
                executor,
                position=position,
                atmospheres=atmospheres,
                atmosphere=atmosphere,
                spec=cirrus,
                weather=weather,
                settings=settings,
                sun_brightness=sun_brightness,
                sky_strength=sky_strength,
                exposure=exposure,
                game=game,
            )

    def _layers(self) -> list:
        return [layer for layer in (self.volume, self.sheet) if layer is not None]

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
        """Change what the clouds cost, both layers (see
        :meth:`march.CloudMarch.set_settings`; the sheet takes ``reflected``
        alone).

        :param settings: the new settings
        """
        self._settings = settings
        for layer in self._layers():
            layer.set_settings(settings)

    def bind_inputs(self, bind: Callable[[NodePath], None]) -> None:
        """Have every layer's shader take more inputs (what shades the
        clouds: the eclipses, a body's shadow maps): *bind* is called with
        each layer's node now, and with a new volume's on a change of
        volume. Each layer binds its own defaults on its own node, which a
        binding on :attr:`root` would not override.

        :param bind: sets inputs on a node (e.g. ``Shadows.bind``)
        """
        self._binders.append(bind)
        for layer in self._layers():
            self._bind_layer(layer, (bind,))

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

    def _bind_layer(self, layer, binders=None) -> None:
        if isinstance(layer, CloudMarch):
            return  # its passes' nodes: not yet
        for bind in self._binders if binders is None else binders:
            bind(layer.root)

    @property
    def shadow_families(self) -> int:
        """How many families cast a shadow: two with genera (the convective
        one, then the stratus), else one."""
        return self.volume.families

    def shadow_caster(self, family: int = 0) -> Any:
        """The caster of *family*'s shadow (0: the convective clouds, or the
        one type; 1: the stratus): :meth:`march.CloudMarch.shadow_caster`."""
        return self.volume.shadow_caster(family)

    def bind_shadow_inputs(self, node: NodePath, family: int = 0) -> None:
        """Let the shaders on a node (the planet's terrain) darken the sun
        under the clouds: the volume's shadow, and with genera the stratus's
        too (:meth:`march.CloudMarch.bind_shadow_inputs`,
        ``uCloudShadowOther``).

        :param node: the node whose shaders include cloud_field.glsl's
            ``planetCloudShadow``
        :param family: 1 binds the stratus's alone, as the first (its shadow
            maps' bake: :class:`shadow_map.CloudShadowMaps`)
        """
        if family == 1:
            self.shadow_caster(1).bind_shadow_inputs(node)
            return
        self._shadow_nodes.append(node)
        self._shade([node])

    def _shade(self, nodes: list[NodePath]) -> None:
        for node in nodes:
            self.volume.bind_shadow_inputs(node)

    @property
    def top_radius(self) -> float:
        """How far from the planet's centre the clouds reach, metres: the
        highest layer's top (the cirrus's, when there is one). What a ray
        meets past where it enters this sphere is behind them (a ring's
        haze, :meth:`asteroids.haze.FieldHaze.set_clouds_top`)."""
        tops = [self.radius + sum(self.volume.field.slab)]
        if self.sheet is not None:
            tops.append(self.radius + self.sheet.spec.top)
        return max(tops)

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
        them), as their shaders draw it on the ground: both families' with
        genera (:func:`field_setup.sun_visibility`).

        :param points_world: (..., 3) world frame, metres
        :param sun_direction: toward the sun, world frame (any length)
        :param max_steps: the most samples a chord takes
            (``ShadowSettings.cloud_shadow_steps``; 0 = none)
        :returns: (...) in [0, 1]
        """
        shade = np.ones(np.shape(points_world)[:-1])
        for family in range(self.shadow_families):
            caster = self.shadow_caster(family)
            if caster is not None:
                shade = shade * caster.sun_visibility(
                    points_world, sun_direction, max_steps
                )
        return shade

    # ── live ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """``pending``: the volume's set-up and the sheet's still to come."""
        stats = dict(self.volume.stats)
        if self.sheet is not None:
            stats["pending"] += self.sheet.stats["pending"]
        return stats

    def set_sun(
        self,
        direction: ArrayLike,
        sun_color: ArrayLike | None = None,
        sky_color: ArrayLike | None = None,
    ) -> None:
        """Move the sun and/or restate its colours, both layers (uniform
        writes only).

        :param direction: toward the sun, world frame (need not be unit)
        :param sun_color: RGB of direct sunlight, or None to keep
        :param sky_color: RGB of the ambient sky light, or None to keep
        """
        self._sun = (
            direction,
            sun_color if sun_color is not None else self._sun[1],
            sky_color if sky_color is not None else self._sun[2],
        )
        for layer in self._layers():
            layer.set_sun(direction, sun_color=sun_color, sky_color=sky_color)

    def set_sky_light(self, strength: float) -> None:
        """The sky's (ambient) light on both layers, as ``sky_strength``."""
        self._sky_light = strength
        for layer in self._layers():
            layer.set_sky_light(strength)

    def set_pixel_angle(self, radians: float, volume_scale: float = 1.0) -> None:
        """The angle one pixel of the view spans (vertical fov / its height):
        how far the volume draws its own detail before averaging it. The
        sheet needs none: its atlas is mipmapped.

        :param radians: the angle per pixel of the view
        :param volume_scale: the volume's pixels over the view's
        """
        self.volume.set_pixel_angle(radians * volume_scale)

    def set_debug(self, flags: int) -> None:
        """Profiling switches, both layers (cloud_field.glsl's
        ``uCloudDebug`` bits).

        :param flags: the bits, 0 = normal
        """
        self._debug = flags
        for layer in self._layers():
            layer.set_debug(flags)

    def set_visible(self, visible: bool) -> None:
        """Show or hide both layers (a march's passes included)."""
        if visible:
            self.root.show()
        else:
            self.root.hide()
        self.volume.set_visible(visible)

    def use_split_depth(self, app: Any, depth: SplitDepthCameras) -> None:
        """The game's split cameras, which a raymarched volume draws its
        passes with (:meth:`march.CloudMarch.attach`). The cirrus stays out
        of those passes' depth.

        :param app: the ShowBase
        :param depth: the split cameras
        """
        self._split = (app, depth)
        if self.sheet is not None:
            self.sheet.root.hide(HIDE_FROM_CLOUD_PASS)
        self.volume.attach(app, depth)

    def update(
        self, camera_world: ArrayLike, camera_render: ArrayLike = (0.0, 0.0, 0.0)
    ) -> None:
        """Per frame: both layers.

        :param camera_world: camera in the world frame, metres (float64)
        :param camera_render: camera in the render frame (the origin when
            the scene is drawn camera-relative)
        """
        for layer in self._layers():
            layer.update(camera_world, camera_render)

    def remove(self) -> None:
        """Take the clouds off the scene and drop the volume's queued tiles."""
        for layer in self._layers():
            layer.remove()
        self.root.remove_node()
