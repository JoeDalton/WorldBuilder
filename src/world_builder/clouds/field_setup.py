"""
What a planet's cloud field is built from, shared by the raymarch
(:mod:`march`) and its tables: the noise volume, the resolved field and its
weather, the coverage calibration (column peaks, shift tables), the deck's
light table, and the octave bands the march samples the field in
(:class:`OctaveBand`). Moved out of :mod:`sphere` (the billboards).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from world_builder.clouds.cloud import (
    PRESETS,
    CloudSpec,
    CloudType,
)
from world_builder.clouds.genera import GeneraSpec, depth_shift_tables, genus_cube
from world_builder.clouds.noise import (
    CloudOptics,
    DensityField,
    cloud_shadow_on_sphere,
    deck_light_table,
    fbm_sigma,
    ground_shadow_table,
    quantise,
    sphere_column_peaks,
    value_noise_volume,
    with_shelf,
    with_threshold,
)
from world_builder.clouds.weather import (
    WeatherSpec,
    sample_cube,
    weather_cube,
)
from world_builder.terrain.cubesphere import CubeGrid
from world_builder.terrain.planet import PlanetSpec

if TYPE_CHECKING:
    pass


#: Side of a first-shell tile, metres, rounded to the nearest level of the
#: planet's grid.
BASE_TILE_SIZE = 8000.0


#: The deck's optical depth cap (the billboards' is ~1.5 per quad).
DECK_MAX_OPTICAL_DEPTH = 6.0


@dataclass(frozen=True)
class OctaveBand:
    """One billboard LOD shell.

    :ivar level: cube-sphere level of its tiles
    :ivar radius: (min, max) billboard radius, metres (at the settings' count)
    :ivar volume_fraction: billboard sphere volume per unit cloud volume
    :ivar octaves: noise octaves evaluated (placement and drawing agree)
    :ivar threshold_offset: added to the field's threshold, so the coverage
        holds with fewer octaves (:func:`octave_offset`)
    """

    level: int
    radius: tuple[float, float]
    volume_fraction: float
    octaves: int = 4
    threshold_offset: float = 0.0

    def field_of(self, field: DensityField) -> DensityField:
        """:returns: *field* as this shell evaluates it"""
        lo, hi = field.threshold
        offset = self.threshold_offset
        return replace(field, threshold=(lo + offset, hi + offset))


@functools.cache
def _noise(seed: int) -> np.ndarray:
    return quantise(value_noise_volume(seed=seed))


@functools.cache
def _weather(weather: WeatherSpec, table: tuple) -> np.ndarray:
    return weather_cube(weather, np.asarray(table))


@functools.cache
def _genus_weather(genera: GeneraSpec, family: str, tables: tuple) -> np.ndarray:
    return genus_cube(genera, family, tables)


def _peaks(seed: int, field: DensityField, radius: float, octaves: int) -> np.ndarray:
    """The planet's column peaks (the threshold does not change them)."""
    return _cached_peaks(seed, replace(field, threshold=None), radius, octaves)


@functools.cache
def _cached_peaks(seed, field, radius, octaves) -> np.ndarray:
    return sphere_column_peaks(_noise(seed), field, radius, octaves)


@functools.cache
def _depth_tables(seed: int, field: DensityField, radius: float) -> np.ndarray:
    """A convective field's shift tables by depth (genera.depth_shift_tables)."""
    return depth_shift_tables(_noise(seed), field, radius)


@functools.cache
def _deck_light(
    seed: int,
    field: DensityField,
    optics: CloudOptics,
    radius: float,
    table: tuple,
    octaves: int,
) -> np.ndarray:
    """The far clouds' averages (~0.6 s), RGBA: the deck's lighting
    (:func:`noise.deck_light_table`: its sun term and visibility), the
    clouds' shadow on the ground (:func:`noise.ground_shadow_table`) third,
    and the deck's second order of scattering fourth."""
    volume = _noise(seed)
    mean = float(volume.mean())
    deck = deck_light_table(
        volume, field, optics, radius, np.asarray(table), octaves, mean
    )
    ground = ground_shadow_table(
        volume, field, optics, radius, np.asarray(table), octave_mean=mean
    )
    return np.dstack([deck[..., :2], ground, deck[..., 2:]])


def octave_offset(full: np.ndarray, fewer: np.ndarray) -> float:
    """How far to move the threshold of a field evaluated with fewer octaves
    so its coverage matches the full field's.

    The dropped octaves count as their mean, which keeps the fbm's mean but
    not its spread: the columns' peaks sit lower, and the coverage drops a
    few percent. Matching the peaks' quantiles over the useful range of
    coverage gives it back (within 2 %, tested).

    :param full: the full field's sorted column peaks
        (:func:`noise.sphere_column_peaks`)
    :param fewer: the same columns' with fewer octaves
    :returns: the offset to add to both ends of the threshold
        (:attr:`OctaveBand.threshold_offset`)
    """
    q = np.linspace(0.05, 0.95, 19)
    return float(np.mean(np.quantile(fewer, q) - np.quantile(full, q)))


def planet_field(
    cloud_type: CloudType, altitude: float | None, seed: int, radius: float
) -> tuple[CloudSpec, DensityField, np.ndarray]:
    """A cloud type's field on a planet, the threshold resolved from its
    coverage: what every renderer of a planet's low clouds starts from.

    :param cloud_type: the type (its preset)
    :param altitude: the slab's base above sea level, metres; None keeps
        the type's
    :param seed: the noise seed
    :param radius: the planet's sea-level radius, metres
    :returns: the preset, the resolved field (its anvil's shelf too:
        :func:`noise.with_shelf`) and the planet's sorted column peaks
        (:func:`noise.sphere_column_peaks`, all four octaves)
    """
    preset = PRESETS[cloud_type]
    field = preset.field
    if altitude is not None:
        field = replace(field, slab=(float(altitude), field.slab[1]))
    sigma = fbm_sigma(_noise(seed), field)
    peaks = _peaks(seed, field, radius, 4)
    resolved = with_shelf(with_threshold(field, peaks, sigma), _noise(seed), radius)
    return preset, resolved, peaks


def planet_weather(
    weather: WeatherSpec | None,
    planet: PlanetSpec | None,
    radius: float,
    weather_seed: int,
) -> WeatherSpec:
    """:returns: *weather*, or the defaults for *planet* (an earthlike one
    of *radius* if None) with *weather_seed*"""
    if weather is not None:
        return weather
    planet = planet or PlanetSpec(name="clouds", radius=radius)
    return WeatherSpec(planet, seed=weather_seed)


def octave_bands(grid: CubeGrid, shells: int) -> tuple[int, list[tuple[float, float]]]:
    """The march's detail bands: a tile level (the base band's scale) and their
    distance bands, each twice as far as the last.

    :param grid: the planet's cube-sphere grid
    :param shells: how many shells
    :returns: the first shell's tile level (tiles of about
        :data:`BASE_TILE_SIZE`) and each shell's crossfade band (lo, hi),
        metres: shell *k* fades out over band *k* as shell *k + 1* (or the
        deck) fades in. Each band is twice the last, for a tile four times
        smaller than the shell's domain.
    """
    base_level = max(0, round(math.log2(BASE_TILE_SIZE / grid.chunk_size(0))))
    domain = 4.0 * grid.chunk_size(base_level)
    bands = [
        (0.62 * domain * 2**k * 0.5, 0.85 * domain * 2**k * 0.5) for k in range(shells)
    ]
    return base_level, bands


#: :func:`sun_visibility` keeps a point's shade per cube of this side,
#: metres, and at most this many of them.
SHADE_CELL = 25.0
SHADE_CACHE_SIZE = 4096


def sun_visibility(
    self: Any, points_world: ArrayLike, sun_direction: ArrayLike, max_steps: int
) -> np.ndarray:
    """The share of the sun a family's clouds leave points under them (or in
    them), as a method of its caster (:class:`march._FamilyCaster`: its
    ``weather``, ``position``, ``radius``, ``_seed``, ``_deck_field``,
    ``_deck_optics``, ``_octave_mean``, ``_shade_key`` and ``_shades``): the
    shader's cloud shadow on the CPU (:func:`noise.cloud_shadow_on_sphere`),
    for what the CPU lights.

    :param points_world: (..., 3) world frame, metres
    :param sun_direction: toward the sun, world frame (any length)
    :param max_steps: the most samples a chord takes
        (``ShadowSettings.cloud_shadow_steps``; 0 = none)
    :returns: (...) in [0, 1]; 1 everywhere until the set-up is done
    """
    points = np.asarray(points_world, dtype=np.float64)
    if self.weather is None or max_steps <= 0:
        return np.ones(points.shape[:-1])
    sun = np.asarray(sun_direction, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    # Each call costs ~0.2 ms (numpy on a few points), every frame: a
    # point's shade is kept per SHADE_CELL of ground (a shadow's edge is
    # hundreds of metres wide), until the sun or the steps change.
    key = (tuple(sun), max_steps)
    if key != self._shade_key or len(self._shades) > SHADE_CACHE_SIZE:
        # Dropped before the lookup, not after: the cells found in it
        # must stay for this call.
        self._shade_key, self._shades = key, {}
    flat = (points - self.position).reshape(-1, 3)
    cells = [tuple(c) for c in np.round(flat / SHADE_CELL).astype(np.int64)]
    missing = [i for i, cell in enumerate(cells) if cell not in self._shades]
    if missing:
        weather = self.weather

        def channel(k):
            if weather.shape[-1] <= k:
                return None
            return lambda p: sample_cube(weather[..., k], p)

        depth = cloud_shadow_on_sphere(
            _noise(self._seed),
            self._deck_field,
            self._deck_optics,
            flat[missing],
            sun,
            self.radius,
            max_steps,
            shift_at=channel(1),
            octave_mean=self._octave_mean,
            depth_at=channel(2),
            spread_at=channel(3),
        )
        for i, value in zip(missing, np.exp2(-depth)):
            self._shades[cells[i]] = float(value)
    shades = np.array([self._shades[cell] for cell in cells])
    return shades.reshape(points.shape[:-1])
