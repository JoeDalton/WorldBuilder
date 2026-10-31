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
  with it (``CloudSpec.deep_radius``). The anvil spreads past its tower as
  a smooth lens, over a fourth map (its spread: :func:`noise.shelf_lift`),
  streaming downwind on the upper wind (:func:`anvil_spread`).

The maps (:func:`genus_planes`, cube maps as the weather's) are the
weather's own coverage for the fair-weather cumulus everywhere, plus, around
:attr:`GeneraSpec.focus`, the features of :attr:`GeneraSpec.features`.

**Coverage to the field.** The share of columns holding cloud depends on
how tall they may grow, so the convective field's fbm shift is tabulated by
depth as well as coverage (:func:`depth_shift_tables`, one
:func:`weather.shift_table` per depth of :data:`DEPTHS`), and the weather
cube carries the depth beside the coverage and the shift
(:func:`genus_cube`), and the anvil's spread (its shift is the field's:
:func:`noise.with_shelf`).

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
    stream_function,
    stream_table,
    stream_vortices,
)
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.chart import TangentFrame, latlon

#: The genus maps' cube size: twice the weather's (1.85 km texels on a
#: 600 km planet), for features a few kilometres across.
GENUS_SIZE = 2 * WEATHER_SIZE
#: The convective depths the fbm shift is tabulated at (and interpolated
#: between).
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
#: The coverage around a cumulonimbus's core: its anvil's overhang, then
#: the congestus ring.
_CB_RING_COVERAGE = 0.45
#: A cumulonimbus's anvil shelf, in radii of it: its spread 1 out to the
#: first, 0 by the second (its underside rising, a lens), with no wind.
_ANVIL_REACH = (1.0, 3.0)
#: The anvil on the upper wind (:func:`anvil_spread`), at the jets' fastest
#: (1): its reach downwind this many times longer, upwind and across this
#: many times shorter (it streams downwind, its upwind edge sharp).
ANVIL_DOWNWIND = 2.5
ANVIL_UPWIND = 0.5
ANVIL_ACROSS = 0.25
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
        its anvil's overhang; a ring of congestus to 2.2 radii; its anvil's
        shelf to 3 radii),
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
    :ivar wind: the upper wind blowing the anvils, (east, north), the jets'
        fastest 1, the same for all; None = the weather's at each
        cumulonimbus (:func:`upper_wind_at`)
    """

    weather: WeatherSpec
    focus: tuple[float, float] = (0.0, 0.0)
    features: tuple[Feature, ...] = field(default=DEFAULT_FEATURES)
    stratus_altitude: float = 900.0
    wind: tuple[float, float] | None = None


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


#: The step the upper wind's stream function is differenced over, metres.
_WIND_STEP = 1000.0


def _frame(centre: np.ndarray) -> TangentFrame:
    lat, lon = latlon(np.asarray(centre)[None])
    return TangentFrame.at(float(lat[0]), float(lon[0]))


def upper_wind_at(spec: WeatherSpec, centre: np.ndarray) -> np.ndarray:
    """The upper wind at a direction: the cirrus's (the jets, swirled by
    the weather systems: :func:`weather.stream_function`), which blows
    along its stream function's contours, ``up x grad psi``.

    :param centre: (3,) unit direction
    :returns: (east, north), the jets' fastest 1
    """
    frame = _frame(centre)
    step = _WIND_STEP / spec.planet.radius
    probes = np.array(
        [
            centre + step * frame.east,
            centre - step * frame.east,
            centre + step * frame.north,
            centre - step * frame.north,
        ]
    )
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    psi = stream_function(stream_table(spec), stream_vortices(spec), probes)
    grad_east = (psi[0] - psi[1]) / (2.0 * _WIND_STEP)
    grad_north = (psi[2] - psi[3]) / (2.0 * _WIND_STEP)
    return np.array([-grad_north, grad_east])


def anvil_spread(
    dirs: np.ndarray, centre: np.ndarray, angle: float, wind: np.ndarray
) -> np.ndarray:
    """A cumulonimbus's anvil spread at *dirs*: 1 to :data:`_ANVIL_REACH`'s
    first radius, 0 by its second, the reach stretched downwind and
    shortened upwind and across on the upper wind (:data:`ANVIL_DOWNWIND`,
    ``ANVIL_UPWIND``, ``ANVIL_ACROSS`` at the jets' fastest): the anvil
    streams downwind, as a plume.

    :param dirs: (N, 3) unit directions
    :param centre: (3,) the tower's centre
    :param angle: its radius, radians
    :param wind: (east, north) at it, the jets' fastest 1
    """
    frame = _frame(centre)
    speed = float(np.hypot(*wind))
    along = frame.east
    if speed > 1e-9:
        along = (wind[0] * frame.east + wind[1] * frame.north) / speed
    across = np.cross(centre, along)
    # Polar coordinates around the tower: its distance (radians, in its
    # radii), along the great circle's heading.
    cosine = np.clip(dirs @ centre, -1.0, 1.0)
    tangent = dirs - cosine[:, None] * centre
    norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)
    distance = np.arccos(cosine) / angle
    x = distance * (tangent @ along) / norm
    y = distance * (tangent @ across) / norm
    speed = min(speed, 1.0)
    x = np.where(
        x > 0.0, x / (1.0 + ANVIL_DOWNWIND * speed), x / (1.0 - ANVIL_UPWIND * speed)
    )
    y = y / (1.0 - ANVIL_ACROSS * speed)
    return 1.0 - _smoothstep(*_ANVIL_REACH, np.hypot(x, y))


@functools.cache
def genus_planes(spec: GeneraSpec, size: int = GENUS_SIZE) -> np.ndarray:
    """The genera's maps as cube maps (cached per process).

    :returns: (6, size, size, 4) float32: the convective coverage (the
        weather's, smooth, upsampled from its own map), the convective depth,
        the stratiform coverage and the anvil shelf's spread, the layout of
        :func:`sky.starfield.cube_directions`
    """
    dirs = cube_directions(size).reshape(-1, 3)
    convective = sample_cube(coverage_map(spec.weather), dirs).astype(np.float64)
    depth = np.zeros(len(dirs))
    stratiform = np.zeros(len(dirs))
    shelf = np.zeros(len(dirs))
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
            wind = (
                np.asarray(spec.wind, dtype=np.float64)
                if spec.wind is not None
                else upper_wind_at(spec.weather, centre)
            )
            shelf = np.maximum(shelf, anvil_spread(dirs, centre, angle, wind))
        else:
            depth = np.maximum(depth, 0.5 * core)
            cover = f.coverage * core
        convective = np.maximum(convective, cover)
    planes = np.stack([convective, depth, stratiform, shelf], axis=-1)
    return planes.reshape(6, size, size, 4).astype(np.float32)


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
    :returns: (6, size, size, 4) float32 for the convective family:
        coverage, the fbm shift giving it, the convective depth, the anvil
        shelf's spread; (6, size, size, 3) for the stratiform one, its
        depth 0
    """
    planes = genus_planes(spec, size)
    tables = np.asarray(tables)
    if family == CONVECTIVE:
        cover, depth = planes[..., 0], planes[..., 1]
        shift = depth_shift(cover, depth, tables)
        channels = [cover, shift, depth, planes[..., 3]]
    elif family == STRATIFORM:
        cover, depth = planes[..., 2], np.zeros(planes.shape[:-1], np.float32)
        shift = coverage_to_shift(cover, tables.reshape(-1))
        channels = [cover, shift, depth]
    else:
        raise ValueError(f"unknown family {family!r}")
    return np.stack([c.astype(np.float32) for c in channels], axis=-1)
