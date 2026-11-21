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
from world_builder.clouds.genera import (
    GeneraSpec,
    coverage_cube,
    depth_shift_tables,
    genus_cube,
    genus_weather_at,
)
from world_builder.clouds.noise import (
    CloudOptics,
    DensityField,
    GenusField,
    cloud_shadow_on_sphere,
    deck_light_table,
    fbm_sigma,
    ground_shadow_table,
    quantise,
    recipe_column_peaks,
    recipe_deck_light_table,
    recipe_shadow_on_sphere,
    recipe_sigma,
    sphere_column_peaks,
    value_noise_volume,
    with_shelf,
    with_threshold,
    worley_volume,
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
def _genus_weather(genera: GeneraSpec, kind: str, tables: tuple) -> np.ndarray:
    return genus_cube(genera, kind, tables)


@functools.cache
def _coverage_cube(genera: GeneraSpec, cube: str) -> np.ndarray:
    return coverage_cube(genera, cube)


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


@functools.cache
def _recipe_deck_light(
    seed: int, genus: GenusField, radius: float, table: tuple, octaves: int
) -> np.ndarray:
    """A recipe genus's far averages (:func:`noise.recipe_deck_light_table`:
    from its recipe's own density), RGBA as :func:`_deck_light`'s.

    :param genus: its far level's field (resolved) and the deck's optics
    """
    return recipe_deck_light_table(
        _noise(seed), _cells(), genus, radius, np.asarray(table), octaves
    )


@functools.cache
def _cells() -> np.ndarray:
    """The recipe genera's Worley volume (:func:`noise.worley_volume`)."""
    return worley_volume()


def recipe_peaks(
    seed: int, genus: GenusField, radius: float, octaves: int = 4
) -> np.ndarray:
    """A recipe genus's sorted column peaks (:func:`noise.recipe_column_peaks`),
    cached: its threshold and optics do not change them."""
    key = replace(
        genus, field=replace(genus.field, threshold=None), optics=CloudOptics()
    )
    return _cached_recipe_peaks(seed, key, radius, octaves)


@functools.cache
def _cached_recipe_peaks(seed, genus, radius, octaves) -> np.ndarray:
    return recipe_column_peaks(_noise(seed), _cells(), genus, radius, octaves)


def recipe_field(
    genus: GenusField, seed: int, radius: float
) -> tuple[GenusField, np.ndarray]:
    """A recipe genus with its field's threshold resolved from its coverage
    by the recipe's own column peaks and spread (:func:`noise.recipe_sigma`),
    as :func:`planet_field` resolves a field's from its fbm's.

    :returns: the genus, resolved, and its sorted column peaks (all four
        octaves): its shift table's (:func:`weather.shift_table`)
    """
    peaks = recipe_peaks(seed, genus, radius, 4)
    sigma = recipe_sigma(_noise(seed), _cells(), genus)
    return replace(genus, field=with_threshold(genus.field, peaks, sigma)), peaks


def genus_field(
    name: str,
    base: float | None,
    thickness: float | None,
    seed: int,
    radius: float,
    east: tuple = (0.0, 1.0, 0.0),
) -> tuple[CloudSpec, DensityField, np.ndarray, GenusField | None]:
    """A genus of :data:`genera.GENUS_PRESETS` on a planet, as the union
    march builds it: its type's field (:func:`planet_field`) with the
    preset's overrides (:attr:`genera.GenusPreset.field`), its threshold
    resolved anew; a recipe genus's from its recipe's own column peaks
    (:func:`recipe_field`); its type's optics with the preset's overrides
    (:attr:`genera.GenusPreset.optics`).

    :param base: its slab's base, metres (None: the type's;
        :func:`genera.genus_specs` gives it)
    :param thickness: its slab's thickness (None: the type's)
    :param east: the recipe genera's stretch axis (the genera's focus's
        east, planet frame)
    :returns: its type's preset, its resolved field, its sorted column peaks
        (its shift table's) and, for a recipe genus, its resolved
        :class:`noise.GenusField` (its optics the type's, with the preset's
        :attr:`genera.GenusPreset.optics`), else None
    """
    from world_builder.clouds.genera import GENUS_PRESETS

    own = GENUS_PRESETS[name]
    preset, field, peaks = planet_field(own.cloud_type, base, seed, radius, thickness)
    if own.optics:
        preset = replace(preset, optics=replace(preset.optics, **own.optics))
    if own.recipe is not None:
        # Its threshold from the recipe's peaks alone (the fbm's not needed)
        field = replace(field, threshold=None, **(own.field or {}))
        recipe, peaks = recipe_field(
            GenusField(name, field, preset.optics, own.coverage, own.far_shadow,
                       recipe=True, east=tuple(east), **own.recipe),
            seed, radius,
        )  # fmt: skip
        return preset, recipe.field, peaks, recipe
    if own.field:
        field = replace(field, threshold=None, **own.field)
        peaks = _peaks(seed, field, radius, 4)
        field = with_threshold(field, peaks, fbm_sigma(_noise(seed), field))
    return preset, field, peaks, None


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
    cloud_type: CloudType,
    altitude: float | None,
    seed: int,
    radius: float,
    thickness: float | None = None,
) -> tuple[CloudSpec, DensityField, np.ndarray]:
    """A cloud type's field on a planet, the threshold resolved from its
    coverage: what every renderer of a planet's low clouds starts from.

    :param cloud_type: the type (its preset)
    :param altitude: the slab's base above sea level, metres; None keeps
        the type's
    :param seed: the noise seed
    :param radius: the planet's sea-level radius, metres
    :param thickness: the slab's thickness, metres; None keeps the type's
    :returns: the preset, the resolved field (its anvil's shelf too:
        :func:`noise.with_shelf`) and the planet's sorted column peaks
        (:func:`noise.sphere_column_peaks`, all four octaves)
    """
    preset = PRESETS[cloud_type]
    field = preset.field
    if altitude is not None or thickness is not None:
        base = field.slab[0] if altitude is None else float(altitude)
        depth = field.slab[1] if thickness is None else float(thickness)
        field = replace(field, slab=(base, depth))
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


def genus_sun_depth(
    genus: Any, points: np.ndarray, sun: np.ndarray, max_steps: int
) -> np.ndarray:
    """A genus's optical depth toward the sun from planet-frame points
    (:func:`noise.cloud_shadow_on_sphere` through its deck field and weather):
    its caster's (:class:`march._GenusCaster`: its ``weather``, ``radius``,
    ``_seed``, ``_deck_field``, ``_deck_optics`` and ``_octave_mean``). A
    genus whose coverage is a coverage cube's channel (its
    ``coverage_source``: the cube, the channel, its shift table) takes its
    shift as the shader's ``genusWeatherAt`` does: the coverage filtered,
    then turned into a shift (:func:`genera.genus_weather_at`). A recipe
    genus (its ``recipe`` not None) marches its recipe's own density as the
    bakes do (:func:`noise.recipe_shadow_on_sphere`: the shadow maps' and
    the light volume's march, its far level's field and octaves,
    ``_deck_octaves``), so the CPU's shade is the maps'; the terrain's
    per-pixel march leaves a recipe genus to the maps (clouds.md, "Where
    the recipe is evaluated").

    :param points: (m, 3) planet frame, metres
    :param sun: toward the sun (unit)
    :returns: (m,)
    """
    weather = genus.weather
    source = getattr(genus, "coverage_source", None)
    recipe = getattr(genus, "recipe", None)
    if recipe is not None:
        shift_at = None
        if source is not None:
            cube, index, table = source
            shift_at = lambda p: genus_weather_at(cube, index, table, p)[1]  # noqa: E731
        return recipe_shadow_on_sphere(
            _noise(genus._seed),
            _cells(),
            replace(recipe, field=genus._deck_field, optics=genus._deck_optics),
            points,
            sun,
            genus.radius,
            max_steps,
            shift_at=shift_at,
            octaves=genus._deck_octaves,
        )

    def channel(k):
        if source is not None:
            if k != 1:
                return None  # no depth, no spread
            cube, index, table = source
            return lambda p: genus_weather_at(cube, index, table, p)[1]
        if weather.shape[-1] <= k:
            return None
        return lambda p: sample_cube(weather[..., k], p)

    return cloud_shadow_on_sphere(
        _noise(genus._seed),
        genus._deck_field,
        genus._deck_optics,
        points,
        sun,
        genus.radius,
        max_steps,
        shift_at=channel(1),
        octave_mean=genus._octave_mean,
        depth_at=channel(2),
        spread_at=channel(3),
    )


def sun_visibility(
    self: Any, points_world: ArrayLike, sun_direction: ArrayLike, max_steps: int
) -> np.ndarray:
    """The share of the sun the clouds leave points under them (or in them),
    as a method of a caster (its ``position``, ``_shade_key`` and
    ``_shades``, and its genera: ``shadow_genera``, or itself as one): the
    shader's cloud shadow on the CPU, ``exp2(-sum of the genera's depths)``
    (:func:`genus_sun_depth`; the product of their shades), for what the CPU
    lights.

    :param points_world: (..., 3) world frame, metres
    :param sun_direction: toward the sun, world frame (any length)
    :param max_steps: the most samples a chord takes
        (``ShadowSettings.cloud_shadow_steps``; 0 = none)
    :returns: (...) in [0, 1]; 1 everywhere until the set-up is done
    """
    points = np.asarray(points_world, dtype=np.float64)
    genera = getattr(self, "shadow_genera", None) or [self]
    if any(genus.weather is None for genus in genera) or max_steps <= 0:
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
        depth = sum(
            genus_sun_depth(genus, flat[missing], sun, max_steps) for genus in genera
        )
        for i, value in zip(missing, np.exp2(-depth)):
            self._shades[cells[i]] = float(value)
    shades = np.array([self._shades[cell] for cell in cells])
    return shades.reshape(points.shape[:-1])
