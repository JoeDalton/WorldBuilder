"""
A planet's clouds as one part: the low clouds' volume, the cirrus sheet above.

:class:`PlanetClouds` is what a game (or :class:`world_builder.world.World`)
builds for a planet. It owns two layers, one per technique:

* the **volume**: the low clouds, cumulus by default (any
  :class:`cloud.CloudType`), as billboard shells near the camera and a deck
  beyond (:class:`sphere.CloudVolume`), or raymarched
  (:class:`march.CloudMarch`, ``CloudSettings.renderer``; it needs the
  split cameras, :meth:`PlanetClouds.use_split_depth`);
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

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import NodePath

from world_builder.atmosphere import Atmospheres
from world_builder.clouds.cirrus import CirrusSpec, CloudSheet
from world_builder.clouds.cloud import PRESETS, CloudSettings, CloudType
from world_builder.clouds.genera import CONVECTIVE, STRATIFORM, GeneraSpec
from world_builder.clouds.march import CloudMarch
from world_builder.clouds.sphere import CLOUD_BIN, CloudVolume
from world_builder.clouds.weather import WeatherSpec
from world_builder.depth import HIDE_FROM_CLOUD_PASS
from world_builder.terrain.planet import PlanetSpec

if TYPE_CHECKING:
    from world_builder.clouds.half_res import HalfResClouds
    from world_builder.depth import SplitDepthCameras
    from world_builder.terrain.workers import ChunkExecutor

#: The volume's class per ``CloudSettings.renderer``.
RENDERERS = {"billboards": CloudVolume, "march": CloudMarch}


def check_layers(
    cloud_type: CloudType,
    altitude: float | None,
    cirrus: CirrusSpec | None,
    air_top: float | None = None,
    genera: GeneraSpec | None = None,
) -> None:
    """Refuse layers the draw order or the air cannot hold: the sheet's base
    must clear the volume's slab, its top stay under the air's; with genera,
    the stratus's top must stay under the convective clouds' base (their
    billboards then sort together as one cloud's).

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
    :ivar volume: the low clouds (:class:`sphere.CloudVolume` or
        :class:`march.CloudMarch`, by ``settings.renderer``)
    :ivar sheet: the cirrus (:class:`cirrus.CloudSheet`), or None
    :ivar weather: the weather both follow
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
        near_fade_radii: float = 2.5,
        channel: str | None = None,
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
        :param near_fade_radii: fade a billboard within this many of its radii
            of the camera (0 = off)
        :param channel: the executor channel for the volume's tiles; None =
            one of its own
        :param game: the game object, for its asset manager (None: from disk)
        :param genera: the low clouds by genus (:mod:`genera`, billboards
            only): the convective volume (*cloud_type* is
            :attr:`cloud.CloudType.CONVECTIVE`, *weather* the genera's) and a
            stratus volume under it (:attr:`stratus`); None = one type
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
        self._settings = settings
        self.root = parent.attach_new_node("planet_clouds")
        # What the volume is built from, again on a change of renderer.
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
            near_fade_radii=near_fade_radii,
            channel=channel,
            game=game,
        )
        if genera is not None:
            self._volume_args.update(genera=genera, family=CONVECTIVE)
        self._split: tuple[Any, SplitDepthCameras] | None = None
        # The live state a new volume is given (set_sun, set_sky_light,
        # set_debug).
        self._sun: tuple = ((0.0, 0.0, 1.0), None, None)
        self._sky_light: float | None = None
        self._debug = 0
        self._half_res: HalfResClouds | None = None
        self.volume = self._make_volume(settings)
        #: The genera's stratus, under the convective volume; None = none
        self.stratus: CloudVolume | None = None
        if genera is not None:
            self.stratus = CloudVolume(
                self.root,
                settings=settings,
                **{
                    **self._volume_args,
                    "cloud_type": CloudType.STRATUS,
                    "altitude": genera.stratus_altitude,
                    "family": STRATIFORM,
                    "channel": None if channel is None else f"{channel}:stratus",
                },
            )
        self._deck_order: tuple | None = None
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
        return [
            layer
            for layer in (self.volume, self.stratus, self.sheet)
            if layer is not None
        ]

    def _volumes(self) -> list:
        """:returns: the low clouds' volumes (the stratus's with genera)"""
        return [v for v in (self.volume, self.stratus) if v is not None]

    def _make_volume(self, settings: CloudSettings) -> CloudVolume | CloudMarch:
        # The genera are drawn by billboards (the raymarch has no depth).
        kind = CloudVolume if self.genera is not None else RENDERERS[settings.renderer]
        volume = kind(self.root, settings=settings, **self._volume_args)
        if isinstance(volume, CloudMarch) and self._split is not None:
            volume.attach(*self._split)
        return volume

    # ── settings ──────────────────────────────────────────────────────────────

    @property
    def settings(self) -> CloudSettings:
        """What the clouds cost (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: CloudSettings) -> None:
        """Change what the clouds cost, both layers (see
        :meth:`sphere.CloudVolume.set_settings`; the sheet takes
        ``reflected`` alone).

        :param settings: the new settings
        """
        renderer = settings.renderer != self._settings.renderer and self.genera is None
        self._settings = settings
        if renderer:
            if self._half_res is not None and isinstance(self.volume, CloudVolume):
                self._half_res.remove(self.volume.root)
            self.volume.remove()
            self.volume = self._make_volume(settings)
            self.volume.set_sun(*self._sun)
            if self._sky_light is not None:
                self.volume.set_sky_light(self._sky_light)
            self.volume.set_debug(self._debug)
            if self._half_res is not None and isinstance(self.volume, CloudVolume):
                self._half_res.add(self.volume.root)
            if self.sheet is not None:
                self.sheet.set_settings(settings)
            return
        for layer in self._layers():
            layer.set_settings(settings)

    # ── live ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """The volume's (tiles, billboards, pending, gen_ms), the stratus's
        added in (with genera), the sheet's set-up counted in ``pending``."""
        stats = dict(self.volume.stats)
        if self.stratus is not None:
            for key in ("tiles", "billboards", "pending"):
                stats[key] += self.stratus.stats[key]
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
        :param volume_scale: the volume's pixels over the view's (the half-
            resolution pass's ``pixel_scale``; the sheet stays at full
            resolution, :meth:`use_half_res`)
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
        if isinstance(self.volume, CloudMarch):
            self.volume.set_visible(visible)

    def use_split_depth(self, app: Any, depth: SplitDepthCameras) -> None:
        """The game's split cameras, which a raymarched volume draws its
        passes with (:meth:`march.CloudMarch.attach`); kept for a later
        change of renderer. The cirrus stays out of those passes' depth.

        :param app: the ShowBase
        :param depth: the split cameras
        """
        self._split = (app, depth)
        if self.sheet is not None:
            self.sheet.root.hide(HIDE_FROM_CLOUD_PASS)
        if isinstance(self.volume, CloudMarch):
            self.volume.attach(app, depth)

    def use_half_res(self, half_res: HalfResClouds | None) -> None:
        """Draw the billboards in a half-resolution pass; the sheet stays at
        full resolution (its fibres are its look, and it is one cheap
        surface), and a raymarch has passes of its own. Feed
        :meth:`set_pixel_angle` the pass's ``pixel_scale``.

        :param half_res: the pass (kept for a later change of renderer);
            None = none any more (call it before destroying the pass)
        """
        if half_res is None:
            if self._half_res is not None:
                for volume in self._volumes():
                    if isinstance(volume, CloudVolume):
                        self._half_res.remove(volume.root)
            self._half_res = None
            return
        self._half_res = half_res
        for volume in self._volumes():
            if isinstance(volume, CloudVolume):
                half_res.add(volume.root)
        if self.sheet is not None:
            half_res.exclude(self.sheet.root)

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
        if self.stratus is not None:
            self._order_decks(camera_world)

    def _order_decks(self, camera_world: ArrayLike) -> None:
        """The two decks are spheres, one inside the other: the farther draws
        first. Seen from outside both (above the convective deck, from orbit)
        that is the stratus's; from under both, the convective one's; in
        between they meet only at the horizon."""
        position = np.asarray(self.volume.position, dtype=np.float64)
        r = float(np.linalg.norm(np.asarray(camera_world, dtype=np.float64) - position))
        under = r < self.stratus.deck_radius
        order = (self.volume, self.stratus) if under else (self.stratus, self.volume)
        if order != self._deck_order:
            self._deck_order = order
            for sort, volume in enumerate(order):
                volume.deck.set_bin(CLOUD_BIN, sort)

    def remove(self) -> None:
        """Take the clouds off the scene and drop the volume's queued tiles."""
        for layer in self._layers():
            layer.remove()
        self.root.remove_node()
