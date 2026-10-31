"""
A planet's low clouds by genus: stratus, and the convective continuum from
fair-weather cumulus through congestus to cumulonimbus. A PLACEHOLDER for the
weather (issue #8): its maps are features laid out around one point.

Two families, each one :class:`sphere.CloudVolume` (one density field), in
bands of sky that do not overlap, so their billboards sort together in the
shells' bins as one cloud would:

* **stratiform** (:attr:`cloud.CloudType.STRATUS`): a flat sheet low down,
  its coverage a map of its own;
* **convective** (:attr:`cloud.CloudType.CONVECTIVE`): the cumulus field,
  whose clouds grow with the convective **depth** (0 to 1, a third map):
  their top rises from the cumulus's 1 km to the slab's cap
  (:attr:`noise.DensityField.min_top`), a coarse cell octave gathers them
  into towers kilometres wide (``cell_weight``), the threshold falls under
  the cap of the deepest into an anvil (``anvil``), and their billboards grow
  with it (``CloudSpec.deep_radius``).

The maps (:func:`genus_planes`, cube maps as the weather's) are the
weather's own coverage for the fair-weather cumulus everywhere, plus, around
:attr:`GeneraSpec.focus`, the features of :attr:`GeneraSpec.features`.

**Coverage to the field.** The share of columns holding cloud depends on
how tall they may grow, so the convective field's fbm shift is tabulated by
depth as well as coverage (:func:`depth_shift_tables`, one
:func:`weather.shift_table` per depth of :data:`DEPTHS`), and the weather
cube carries the depth beside the coverage and the shift
(:func:`genus_cube`).

Design notes: docs/source/clouds.md, "Genera".
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from world_builder.clouds.noise import (
    DensityField,
    _smoothstep,
    sphere_column_peaks,
)
from world_builder.clouds.weather import (
    WEATHER_SIZE,
    WeatherSpec,
    coverage_map,
    coverage_to_shift,
    sample_cube,
    shift_table,
)
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.chart import TangentFrame

#: The genus maps' cube size: twice the weather's (1.85 km texels on a
#: 600 km planet), for features a few kilometres across.
GENUS_SIZE = 2 * WEATHER_SIZE
#: The convective depths the fbm shift is tabulated at (and interpolated
#: between).
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
#: The coverage around a cumulonimbus's core: its anvil's overhang, then
#: the congestus ring.
_CB_RING_COVERAGE = 0.45
#: The families.
CONVECTIVE = "convective"
STRATIFORM = "stratiform"


@dataclass(frozen=True)
class Feature:
    """One feature of the placeholder maps: a disc of a genus.

    :ivar east: its centre east of the focus, metres
    :ivar north: north of it, metres
    :ivar radius: its radius, metres; it fades out over its outer third
    :ivar genus: ``"cumulonimbus"`` (depth 1 to 1.4 radii: its core, then
        its anvil's overhang; a ring of congestus to 2.2 radii),
        ``"congestus"`` (depth ~0.5) or ``"stratus"`` (the stratiform
        family)
    :ivar coverage: the share of its columns holding cloud at its core (a
        cumulonimbus's near 1: a solid tower, the noise carving its edges;
        less, and a deep column holds scattered blobs at every height)
    """

    east: float
    north: float
    radius: float
    genus: str
    coverage: float = 0.6


#: The demo's: within view of the demo's start, a cumulonimbus 40 km north
#: ringed by congestus, another 70 km north-east, a line of congestus to the
#: west, and a stratus deck to the south.
DEFAULT_FEATURES = (
    Feature(0.0, 40e3, 6e3, "cumulonimbus", 1.0),
    Feature(45e3, 55e3, 5e3, "cumulonimbus", 1.0),
    Feature(-30e3, 20e3, 12e3, "congestus", 0.45),
    Feature(-30e3, 40e3, 10e3, "congestus", 0.45),
    Feature(0.0, -30e3, 25e3, "stratus", 0.9),
)


@dataclass(frozen=True)
class GeneraSpec:
    """A planet's low clouds by genus (see the module); frozen and picklable.

    :ivar weather: the weather whose coverage the fair-weather cumulus keep
    :ivar focus: (lat, lon) degrees the features are placed around
    :ivar features: the features
    :ivar stratus_altitude: the stratiform slab's base, metres (its top must
        stay under the convective clouds' base)
    """

    weather: WeatherSpec
    focus: tuple[float, float] = (0.0, 0.0)
    features: tuple[Feature, ...] = field(default=DEFAULT_FEATURES)
    stratus_altitude: float = 900.0


def _feature_dirs(spec: GeneraSpec) -> list[tuple[Feature, np.ndarray, float]]:
    """:returns: each feature, its centre's direction and angular radius"""
    frame = TangentFrame.at(*spec.focus)
    radius = spec.weather.planet.radius
    out = []
    for f in spec.features:
        offset = math.hypot(f.east, f.north) / radius
        heading = (f.east * frame.east + f.north * frame.north) / max(
            math.hypot(f.east, f.north), 1e-9
        )
        centre = math.cos(offset) * frame.up + math.sin(offset) * heading
        out.append((f, centre, f.radius / radius))
    return out


@functools.cache
def genus_planes(spec: GeneraSpec, size: int = GENUS_SIZE) -> np.ndarray:
    """The genera's maps as cube maps (cached per process).

    :returns: (6, size, size, 3) float32: the convective coverage (the
        weather's, smooth, upsampled from its own map), the convective depth
        and the stratiform coverage, the layout of
        :func:`sky.starfield.cube_directions`
    """
    dirs = cube_directions(size).reshape(-1, 3)
    convective = sample_cube(coverage_map(spec.weather), dirs).astype(np.float64)
    depth = np.zeros(len(dirs))
    stratiform = np.zeros(len(dirs))
    for f, centre, angle in _feature_dirs(spec):
        distance = np.arccos(np.clip(dirs @ centre, -1.0, 1.0)) / angle
        core = 1.0 - _smoothstep(2.0 / 3.0, 1.0, distance)
        if f.genus == "stratus":
            stratiform = np.maximum(stratiform, f.coverage * core)
            continue
        if f.genus == "cumulonimbus":
            # Depth 1 over the core and around it, where the sparser columns
            # hold cloud mostly in the anvil's band (an overhang); congestus
            # beyond, gone by 2.2 radii.
            deep = 1.0 - _smoothstep(1.2, 1.6, distance)
            ring = 1.0 - _smoothstep(1.6, 2.2, distance)
            depth = np.maximum(depth, np.maximum(deep, 0.5 * ring))
            cover = np.maximum(f.coverage * core, _CB_RING_COVERAGE * ring)
        else:
            depth = np.maximum(depth, 0.5 * core)
            cover = f.coverage * core
        convective = np.maximum(convective, cover)
    planes = np.stack([convective, depth, stratiform], axis=-1)
    return planes.reshape(6, size, size, 3).astype(np.float32)


def depth_shift_tables(
    volume: np.ndarray, field_: DensityField, radius: float, octaves: int = 4
) -> np.ndarray:
    """The convective field's fbm shift by coverage, one table per depth of
    :data:`DEPTHS` (a deep column may grow taller, so it needs less shift
    for the same coverage).

    :param field_: the resolved field (its threshold set at depth 0)
    :returns: (len(DEPTHS), SHIFT_TABLE_SIZE) float64
    """
    return np.array(
        [
            shift_table(
                sphere_column_peaks(volume, field_, radius, octaves, depth=d),
                field_.threshold[0],
            )
            for d in DEPTHS
        ]
    )


def depth_shift(
    coverage: np.ndarray, depth: np.ndarray, tables: np.ndarray
) -> np.ndarray:
    """The fbm shift giving *coverage* at convective *depth*: the tables'
    shifts at the two depths around it, interpolated.

    :param tables: :func:`depth_shift_tables`
    """
    coverage = np.asarray(coverage, dtype=np.float64)
    depth = np.clip(np.asarray(depth, dtype=np.float64), 0.0, 1.0)
    at = np.stack([coverage_to_shift(coverage, t) for t in tables], axis=-1)
    position = depth * (len(DEPTHS) - 1)
    low = np.minimum(np.floor(position).astype(int), len(DEPTHS) - 2)
    frac = position - low
    pick = np.take_along_axis
    lo = pick(at, low[..., None], axis=-1)[..., 0]
    hi = pick(at, low[..., None] + 1, axis=-1)[..., 0]
    return lo + (hi - lo) * frac


def genus_cube(
    spec: GeneraSpec, family: str, tables: tuple, size: int = GENUS_SIZE
) -> np.ndarray:
    """A family's weather map (:func:`weather.build_weather_texture`
    uploads it), as :func:`weather.weather_cube` is the single field's.

    :param family: :data:`CONVECTIVE` or :data:`STRATIFORM`
    :param tables: the family's shift tables: one per depth of
        :data:`DEPTHS` (:func:`depth_shift_tables`) for the convective one,
        one (:func:`weather.shift_table`) for the stratiform one
    :returns: (6, size, size, 3) float32: coverage, the fbm shift giving
        it, the convective depth (0 for the stratiform family)
    """
    planes = genus_planes(spec, size)
    tables = np.asarray(tables)
    if family == CONVECTIVE:
        cover, depth = planes[..., 0], planes[..., 1]
        shift = depth_shift(cover, depth, tables)
    elif family == STRATIFORM:
        cover, depth = planes[..., 2], np.zeros(planes.shape[:-1], np.float32)
        shift = coverage_to_shift(cover, tables.reshape(-1))
    else:
        raise ValueError(f"unknown family {family!r}")
    return np.stack([cover, shift.astype(np.float32), depth], axis=-1).astype(
        np.float32
    )
