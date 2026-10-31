"""
The weather over a planet: how much of the sky is cloud, by direction.

A cube map of target **coverage** (the share of the slab's columns holding
cloud) built with numpy from:

1. **zonal circulation**: a wet equatorial band, dry subtropical highs,
   mid-latitude storm tracks, polar caps (:attr:`WeatherSpec.zonal`);
2. **extratropical cyclones** in the storm tracks: a vortex that winds the
   mesoscale texture into spiral arms, and a comma of cloud along one wound
   arm (the head poleward of the low, the cold front trailing equatorward and
   west), with clear air in the cold sector behind it;
3. **tropical cyclones** over warm sea: a tight spiral with a clear eye;
4. **mesoscale texture**: fbm warped by the same vortices, for ragged edges;
5. **the planet's climate and terrain** (:class:`terrain.planet.Planet`):
   humidity scales the cover, hot dry land is cleared, the sea adds a little,
   and slopes facing the prevailing wind lift cloud (their lee loses it).

Lengths follow the planet's feature scale, like the terrain's.

The cirrus above (:mod:`cirrus`) has a weather of its own, from the same
systems: an **upper wind** (the jet streams, deflected around the cyclones),
whose stream function sets the direction of its fibres
(:func:`stream_table`, :func:`stream_vortices`), and a cube map of its
**cover** (:func:`cirrus_cover_map`): along the jets, over the wet tropics
(anvils), in a shield ahead of each cyclone.

The clouds never see coverage directly: :func:`shift_table` turns it into the
fbm shift that gives that coverage, exactly, from the field's sorted column
peaks over the planet (:func:`noise.sphere_column_peaks` is the inverse of
the coverage function).
The GPU texture holds both (r = coverage for the far deck, g = the shift for
the density), so the near field and the far deck agree by construction.

Pure numpy, deterministic and picklable: the cloud workers rebuild the same
map from a :class:`WeatherSpec` (cached per process).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
from panda3d.core import SamplerState, Texture

from world_builder.sky.starfield import cube_directions, directions_to_texels
from world_builder.terrain.chart import latlon
from world_builder.terrain.noise import fbm
from world_builder.terrain.planet import (
    REFERENCE_RADIUS_M,
    Planet,
    PlanetSpec,
    scaled_params,
)

#: Texels per face edge of the weather cube map (~3.7 km on a 600 km planet).
WEATHER_SIZE = 256
#: Texels per face edge the planet's climate is sampled at, then upsampled.
CLIMATE_SIZE = 64
#: Coverages the shift table is tabulated at, evenly over [0, 1].
SHIFT_TABLE_SIZE = 256
#: Texels per face edge of the cirrus cover map (~7 km on a 600 km planet):
#: the cover follows the weather systems; the fibres come from the noise.
CIRRUS_SIZE = 128
#: Latitudes the upper wind's stream function is tabulated at, pole to pole.
STREAM_TABLE_SIZE = 1024

_SEED_MESO = 401
_SEED_PLACEMENT = 409
_SEED_CIRRUS = 419


@dataclass(frozen=True)
class WeatherSpec:
    """Everything that defines a planet's weather; frozen and picklable.

    Lengths are for a :data:`terrain.planet.REFERENCE_RADIUS_M` planet and
    follow its feature scale. Coverages are shares of the sky in [0, 1].

    :ivar planet: the planet whose climate and terrain the weather follows
    :ivar seed: placement of the weather systems and the mesoscale texture
    :ivar zonal: (|latitude| degrees, coverage) control points of the zonal
        mean: wet tropics, dry subtropics, storm tracks, polar caps
    :ivar season: degrees north the zonal pattern is moved (the equatorial
        band follows the sun)
    :ivar storm_tracks: latitude band (degrees) of extratropical cyclones
    :ivar cyclones: how many on a reference planet (scaled by area)
    :ivar cyclone_radius: radius of one, metres
    :ivar cyclone_swirl: radians the vortex turns the texture at its centre
    :ivar cyclone_strength: cover added at the core
    :ivar front_strength: cover added along the comma's arm (its head, body
        and the cold front trailing from it)
    :ivar tropical: how many tropical cyclones (over sea only)
    :ivar tropical_radius: radius of one, metres
    :ivar tropical_swirl: radians the vortex turns the texture at its centre
    :ivar tropical_strength: cover added by its spiral arms and eyewall
    :ivar meso_wavelength: the mesoscale texture's largest feature, metres
    :ivar meso_strength: its amplitude, in coverage
    :ivar dry_floor: what the zonal cover is multiplied by with no humidity
    :ivar desert_clearing: share of the cover hot, dry land loses
    :ivar maritime: cover added over the sea
    :ivar orographic: cover added on a steep windward slope of humid land
        (taken in the lee)
    :ivar orographic_slope: the slope along the wind that gives 76 % of it
    :ivar climate_wavelength: the shortest relief and climate features the
        weather follows, metres (the mountain ranges, not single peaks)
    :ivar jets: (|latitude| degrees, eastward wind) control points of the
        upper wind, the jet streams' 1 the fastest: weak tropical easterlies,
        the subtropical jet, the polar-front jet. The cirrus fibres follow it
    :ivar jet_vortex: the cyclones' upper wind at its fastest, relative to
        the jets' (the fibres bend around them)
    :ivar cirrus_zonal: (|latitude| degrees, cover) control points of the
        cirrus: anvils over the wet tropics, streaks along the jets
    :ivar cirrus_shield: cirrus cover added ahead of a cyclone (poleward and
        east of its low), and over a tropical one
    :ivar cirrus_meso: amplitude of the cirrus cover's mesoscale texture
    """

    planet: PlanetSpec
    seed: int = 11
    zonal: tuple = (
        (0.0, 0.72),
        (8.0, 0.66),
        (16.0, 0.40),
        (24.0, 0.15),
        (33.0, 0.16),
        (42.0, 0.36),
        (52.0, 0.55),
        (62.0, 0.55),
        (72.0, 0.44),
        (90.0, 0.40),
    )
    season: float = 0.0
    storm_tracks: tuple = (42.0, 62.0)
    cyclones: int = 6
    cyclone_radius: float = 80e3
    cyclone_swirl: float = 3.4
    cyclone_strength: float = 0.4
    front_strength: float = 0.5
    tropical: int = 2
    tropical_radius: float = 45e3
    tropical_swirl: float = 5.0
    tropical_strength: float = 0.8
    meso_wavelength: float = 160e3
    meso_strength: float = 0.4
    dry_floor: float = 0.12
    desert_clearing: float = 0.6
    maritime: float = 0.08
    orographic: float = 0.3
    orographic_slope: float = 0.03
    climate_wavelength: float = 60e3
    jets: tuple = (
        (0.0, -0.25),
        (12.0, -0.2),
        (22.0, 0.45),
        (30.0, 1.0),
        (38.0, 0.55),
        (48.0, 0.7),
        (55.0, 0.9),
        (64.0, 0.4),
        (75.0, -0.05),
        (90.0, 0.0),
    )
    jet_vortex: float = 0.8
    cirrus_zonal: tuple = (
        (0.0, 0.55),
        (10.0, 0.42),
        (20.0, 0.2),
        (30.0, 0.38),
        (40.0, 0.28),
        (52.0, 0.45),
        (64.0, 0.32),
        (78.0, 0.12),
        (90.0, 0.08),
    )
    cirrus_shield: float = 0.45
    cirrus_meso: float = 0.3


@dataclass(frozen=True)
class Vortex:
    """One weather system.

    :ivar centre: (3,) unit direction of its centre
    :ivar radius: angular radius, radians
    :ivar swirl: radians it turns the texture at its centre (signed:
        counter-clockwise seen from above in the north)
    :ivar tropical: a tropical cyclone (eye, spiral arms) rather than an
        extratropical one (comma, fronts)
    """

    centre: tuple
    radius: float
    swirl: float
    tropical: bool


# ── Cube maps ──────────────────────────────────────────────────────────────────


def sample_cube(cube: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Bilinear lookup in a cube map, as the GPU filters within a face.

    Texel centres read their texel; at a face's edge it clamps (the GPU may
    blend across the seam there: only within half a texel of it).

    :param cube: (6, n, n, ...) [face, row (t), column (s)], the layout of
        :func:`sky.starfield.cube_directions`
    :param dirs: (..., 3) directions (any length)
    :returns: (...) or (..., channels)
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    shape = dirs.shape[:-1]
    size = cube.shape[1]
    face, row, column = directions_to_texels(dirs.reshape(-1, 3), size)
    y = np.clip(row - 0.5, 0.0, size - 1.0)
    x = np.clip(column - 0.5, 0.0, size - 1.0)
    y0 = np.minimum(np.floor(y).astype(np.int64), size - 2)
    x0 = np.minimum(np.floor(x).astype(np.int64), size - 2)
    ty, tx = y - y0, x - x0
    if cube.ndim == 4:
        ty, tx = ty[:, None], tx[:, None]
    c00 = cube[face, y0, x0]
    c01 = cube[face, y0, x0 + 1]
    c10 = cube[face, y0 + 1, x0]
    c11 = cube[face, y0 + 1, x0 + 1]
    top = c00 + (c01 - c00) * tx
    bottom = c10 + (c11 - c10) * tx
    value = top + (bottom - top) * ty
    return value.reshape(shape + cube.shape[3:])


def build_weather_texture(cube: np.ndarray) -> Texture:
    """The (6, n, n, 2) weather (coverage, shift) as an RG32F cube map,
    linearly filtered (:func:`sample_cube` is its CPU copy)."""
    size = cube.shape[1]
    tex = Texture("cloud_weather")
    tex.setup_cube_map(size, Texture.T_float, Texture.F_rg32)
    # Pages in face order, rows bottom-up = t up, as the star field's.
    tex.set_ram_image(np.ascontiguousarray(cube, dtype=np.float32).tobytes())
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


# ── Coverage to the field's shift ──────────────────────────────────────────────


def shift_table(
    peaks: np.ndarray, threshold_lo: float, size: int = SHIFT_TABLE_SIZE
) -> np.ndarray:
    """The fbm shift giving each coverage, ``c = i / (size - 1)``.

    A column holds cloud when its peak plus the shift clears the threshold,
    so the shift for coverage c puts the (1 - c) quantile of the peaks on it:
    ``shift(c) = threshold_lo - quantile(peaks, 1 - c)``. The ends go a hair
    past the extreme peaks, so 0 is clear and 1 is overcast.

    :param peaks: :func:`noise.sphere_column_peaks`, sorted
    :param threshold_lo: the field's resolved threshold (lower edge)
    :returns: (size,) float64, increasing
    """
    coverage = np.linspace(0.0, 1.0, size)
    quantile = np.quantile(peaks, 1.0 - coverage)
    shift = threshold_lo - quantile
    margin = 1e-4
    shift[0] -= margin
    shift[-1] += margin
    return shift


def coverage_to_shift(coverage: np.ndarray, table: np.ndarray) -> np.ndarray:
    """:returns: the shift for each *coverage*, linear between table entries"""
    return np.interp(coverage, np.linspace(0.0, 1.0, len(table)), table)


# ── The weather systems ───────────────────────────────────────────────────────


def _feature_scale(spec: WeatherSpec) -> float:
    return scaled_params(spec.planet)[0].feature_scale


def _random_dirs(rng, lat_lo: float, lat_hi: float, count: int) -> np.ndarray:
    """Uniform on the sphere between two |latitudes|, either hemisphere."""
    s_lo, s_hi = math.sin(math.radians(lat_lo)), math.sin(math.radians(lat_hi))
    z = rng.uniform(s_lo, s_hi, count) * rng.choice([-1.0, 1.0], count)
    lon = rng.uniform(-math.pi, math.pi, count)
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(lon), r * np.sin(lon), z], axis=1)


def vortices(spec: WeatherSpec, planet: Planet | None = None) -> list[Vortex]:
    """The planet's weather systems, placed deterministically from the seed.

    Extratropical cyclones go in the storm tracks, apart from each other;
    tropical ones over open sea at 10-25 degrees (none on a dry planet).

    :param spec: the weather
    :param planet: the evaluated planet (built from the spec if None)
    :returns: the extratropical cyclones, then the tropical ones
    """
    planet = planet or Planet(spec.planet)
    radius = spec.planet.radius
    scale = _feature_scale(spec)
    rng = np.random.default_rng([spec.seed, _SEED_PLACEMENT])
    # The count follows the area in features: a constant while the features
    # follow the radius, more where the feature scale is clamped.
    count = round(spec.cyclones * (radius / (scale * REFERENCE_RADIUS_M)) ** 2)
    found: list[Vortex] = []
    extratropical = spec.cyclone_radius * scale / radius
    candidates = _random_dirs(rng, *spec.storm_tracks, 40 * max(count, 1))
    for centre in candidates:
        if sum(not v.tropical for v in found) >= count:
            break
        if all(
            math.acos(np.clip(centre @ np.array(v.centre), -1, 1)) > 3.5 * extratropical
            for v in found
        ):
            north = 1.0 if centre[2] >= 0.0 else -1.0
            swirl = spec.cyclone_swirl * north * rng.uniform(0.8, 1.2)
            size = extratropical * rng.uniform(0.75, 1.3)
            found.append(Vortex(tuple(centre), size, swirl, False))
    if spec.planet.has_ocean and spec.tropical > 0:
        tropical = spec.tropical_radius * scale / radius
        candidates = _random_dirs(rng, 10.0, 25.0, 64 * spec.tropical)
        elevation, _, _ = _sampler(spec, planet)[0](candidates)
        placed = 0
        for centre, height in zip(candidates, elevation):
            if placed >= spec.tropical:
                break
            if height > -200.0:  # over open sea only
                continue
            if all(
                math.acos(np.clip(centre @ np.array(v.centre), -1, 1))
                > 3.0 * max(v.radius, tropical)
                for v in found
            ):
                north = 1.0 if centre[2] >= 0.0 else -1.0
                swirl = spec.tropical_swirl * north
                found.append(Vortex(tuple(centre), tropical, swirl, True))
                placed += 1
    return found


def _rotate(points: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues: *points* (N, 3) turned about the unit *axis* by *angle* (N,)."""
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    along = (points @ axis)[:, None] * axis
    return points * cos + np.cross(axis, points) * sin + along * (1.0 - cos)


def _local(points: np.ndarray, vortex: Vortex):
    """Gnomonic (east, poleward) coordinates around a vortex, in its radii.

    Poleward is north in the north and south in the south, so one layout
    (comma head poleward, cold front trailing equatorward and west) serves
    both hemispheres.
    """
    c = np.asarray(vortex.centre)
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, c)
    east /= max(np.linalg.norm(east), 1e-9)
    north = np.cross(c, east)
    poleward = north if c[2] >= 0.0 else -north
    depth = np.maximum(points @ c, 1e-6)
    x = (points @ east) / depth / vortex.radius
    y = (points @ poleward) / depth / vortex.radius
    return x, y


def _reach(v: Vortex) -> float:
    """How far a vortex turns the texture, in its radii (a Gaussian's width)."""
    return 0.8 if v.tropical else 1.6


def _near(dirs: np.ndarray, v: Vortex) -> np.ndarray:
    """:returns: the mask of the directions a vortex reaches"""
    return dirs @ np.asarray(v.centre) > math.cos(min(6.0 * v.radius, 1.2))


def _wind_up(dirs: np.ndarray, warped: np.ndarray, v: Vortex) -> np.ndarray:
    """*warped* (the texture's coordinates at *dirs*) wound by the vortex's
    swirl, most at its centre: a line of constant wound angle is a spiral."""
    c = np.asarray(v.centre)
    angle = np.arccos(np.clip(dirs @ c, -1.0, 1.0))
    turn = v.swirl * np.exp(-((angle / (_reach(v) * v.radius)) ** 2))
    return _rotate(warped, c, -turn)


def _gauss(x, y, cx, cy, sx, sy):
    return np.exp(-(((x - cx) / sx) ** 2) - ((y - cy) / sy) ** 2)


#: Where an extratropical cyclone's cloud arm points far from its centre,
#: radians in (east, poleward) coordinates: equatorward and a little west,
#: the cold front. Wound inward by the swirl it curls round the east side
#: into the comma head, poleward of the low.
ARM_ANGLE = math.radians(-105.0)


def _vortex_cover(points: np.ndarray, warped: np.ndarray, v: Vortex, spec) -> tuple:
    """Cover a vortex adds around its centre (zero far away), and the factor
    clearing a tropical cyclone's eye.

    :param points: (N, 3) unit directions near the vortex
    :param warped: the same, wound by its swirl: a line of constant wound
        angle is a spiral arm
    """
    x, y = _local(points, v)
    r = np.hypot(x, y)
    wx, wy = _local(warped, v)
    wound = np.arctan2(wy, wx)
    if v.tropical:
        # Two arms, the eyewall, a clear eye.
        arms = np.cos(2.0 * wound) * 0.5 + 0.5
        cover = spec.tropical_strength * (
            np.exp(-((r / 0.7) ** 2)) + 0.7 * arms**3 * np.exp(-((r / 2.4) ** 2))
        )
        eye = np.clip((r - 0.06) / 0.1, 0.0, 1.0)
        return cover, eye
    off = np.angle(np.exp(1j * (wound - ARM_ANGLE)))
    arm = (
        np.exp(-((off / 0.5) ** 2))
        * np.clip((r - 0.1) / 0.5, 0.0, 1.0)
        * np.exp(-((r / 3.2) ** 2))
    )
    cover = (
        spec.cyclone_strength * np.exp(-((r / 0.6) ** 2))  # the core
        + spec.front_strength * arm  # comma head, body and cold front
        - 0.35 * _gauss(x, y, -1.5, -0.9, 1.1, 1.1)  # the cold sector behind
    )
    return cover, np.ones_like(r)


def _wind_east(lat: np.ndarray) -> np.ndarray:
    """Prevailing wind toward the east (+1) or west (-1) by latitude: the
    easterly trades, the westerlies, the polar easterlies."""
    return np.interp(np.abs(lat), [0, 25, 35, 55, 65, 90], [-1, -1, 1, 1, -1, -1])


def orographic(
    dirs: np.ndarray, elevation_at, spec: WeatherSpec, step: float
) -> np.ndarray:
    """Cover from the terrain lifting the prevailing wind: more on windward
    slopes, less in the lee.

    :param dirs: (N, 3) unit directions
    :param elevation_at: ``(N, 3) dirs -> (N,)`` metres above sea level
    :param spec: the weather (its orographic gain and slope)
    :param step: metres upwind the slope is measured over
    :returns: (N,) coverage to add (negative in the lee)
    """
    lat, _ = latlon(dirs)
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, dirs)
    east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-9)
    wind = east * _wind_east(lat)[:, None]
    upwind = dirs - wind * (step / spec.planet.radius)
    upwind /= np.linalg.norm(upwind, axis=1, keepdims=True)
    # Positive where the ground rises along the wind: the air is lifted.
    slope = (elevation_at(dirs) - elevation_at(upwind)) / step
    return spec.orographic * np.tanh(slope / spec.orographic_slope)


def _wetness(humidity) -> np.ndarray:
    wet = np.clip((humidity - 0.05) / 0.5, 0.0, 1.0)
    return wet * wet * (3.0 - 2.0 * wet)


def climate_factor(
    spec: WeatherSpec, temperature: np.ndarray, humidity: np.ndarray
) -> np.ndarray:
    """What the zonal cover is multiplied by: humidity, and hot dry land
    cleared (deserts stay clear).

    :param temperature: degrees Celsius, as :meth:`terrain.planet.Planet.sample_dirs`
    :param humidity: in [0, 1], likewise
    """
    moist = spec.dry_floor + (1.0 - spec.dry_floor) * _wetness(humidity)
    hot = np.clip((temperature - 22.0) / 12.0, 0.0, 1.0)
    dry = 1.0 - np.clip((humidity - 0.2) / 0.35, 0.0, 1.0)
    return moist * (1.0 - spec.desert_clearing * hot * dry)


def coverage_at(
    spec: WeatherSpec, dirs: np.ndarray, planet: Planet | None = None
) -> np.ndarray:
    """The weather's target coverage at arbitrary directions (no cube map).

    The climate is sampled directly here; :func:`coverage_map` samples it on
    a coarse grid and upsamples, which is what makes the map cheap.

    :param dirs: (N, 3) unit directions
    :param planet: the evaluated planet (built from the spec if None)
    :returns: (N,) in [0, 1]
    """
    planet = planet or Planet(spec.planet)
    return _compose(spec, dirs, planet, _climate_direct(spec, planet))


def _sampler(spec, planet):
    """:returns: ``dirs -> (elevation, temperature, humidity)`` band-limited
    to the climate wavelength, and the orographic slope step, metres"""
    scale = _feature_scale(spec)
    wavelength = spec.climate_wavelength * scale

    def sample(dirs):
        return planet.sample_dirs(dirs, min_wavelength=wavelength)

    return sample, 0.4 * wavelength


def _climate(spec, planet, dirs):
    """:returns: elevation, temperature, humidity and orographic cover"""
    sample, step = _sampler(spec, planet)
    elevation, temperature, humidity = sample(dirs)
    lift = orographic(dirs, lambda d: sample(d)[0], spec, step)
    return elevation, temperature, humidity, lift * _wetness(humidity)


def _climate_direct(spec, planet):
    return lambda dirs: _climate(spec, planet, dirs)


def _climate_gridded(spec, planet):
    """The climate on a coarse cube grid, bilinear in between."""
    grid = cube_directions(CLIMATE_SIZE).reshape(-1, 3)
    elevation, temperature, humidity, lift = _climate(spec, planet, grid)
    fields = np.stack([elevation, temperature, humidity, lift], axis=-1).reshape(
        6, CLIMATE_SIZE, CLIMATE_SIZE, 4
    )

    def climate(dirs):
        return tuple(sample_cube(fields, dirs).T)

    return climate


def _compose(spec: WeatherSpec, dirs: np.ndarray, planet: Planet, climate):
    radius = spec.planet.radius
    scale = _feature_scale(spec)
    lat, _ = latlon(dirs)
    points = np.array(spec.zonal, dtype=np.float64)
    zonal = np.interp(np.abs(lat - spec.season), points[:, 0], points[:, 1])

    elevation, temperature, humidity, lift = climate(dirs)
    cover = zonal * climate_factor(spec, temperature, humidity) + lift
    if spec.planet.has_ocean:
        sea = np.clip(-elevation / 200.0, 0.0, 1.0)
        cover = cover + spec.maritime * sea

    # Vortices: wind the texture's coordinates, add each system's cover.
    warped = dirs.copy()
    eye = np.ones(len(dirs))
    for v in vortices(spec, planet):
        near = _near(dirs, v)
        if not near.any():
            continue
        warped[near] = _wind_up(dirs[near], warped[near], v)
        add, clear = _vortex_cover(dirs[near], warped[near], v, spec)
        cover[near] += add
        eye[near] *= clear
    meso = fbm(
        warped * radius, spec.meso_wavelength * scale, 3, spec.seed * 1000 + _SEED_MESO
    )
    cover = cover + spec.meso_strength * meso
    return np.clip(cover * eye, 0.0, 1.0)


@functools.cache
def coverage_map(spec: WeatherSpec, size: int = WEATHER_SIZE) -> np.ndarray:
    """The weather's target coverage as a cube map (cached per process).

    :returns: (6, size, size) float32 in [0, 1], the layout of
        :func:`sky.starfield.cube_directions`
    """
    planet = Planet(spec.planet)
    dirs = cube_directions(size).reshape(-1, 3)
    cover = _compose(spec, dirs, planet, _climate_gridded(spec, planet))
    return cover.reshape(6, size, size).astype(np.float32)


def weather_cube(
    spec: WeatherSpec, table: np.ndarray, size: int = WEATHER_SIZE
) -> np.ndarray:
    """The weather map the clouds draw from (:func:`build_weather_texture`
    uploads it).

    :param table: the field's shift table (:func:`shift_table`)
    :returns: (6, size, size, 2) float32: coverage, and the fbm shift
        giving it
    """
    cover = coverage_map(spec, size)
    shift = coverage_to_shift(cover, np.asarray(table))
    return np.stack([cover, shift.astype(np.float32)], axis=-1)


# ── The cirrus: the upper wind and the cover ─────────────────────────────────

#: The steepest slope of exp(-x^2) (at x = 1/sqrt(2)): a Gaussian stream
#: function of amplitude A and width s blows at most this * A / s.
_GAUSS_SLOPE = math.sqrt(2.0) * math.exp(-0.5)


def upper_wind(spec: WeatherSpec, lat: np.ndarray) -> np.ndarray:
    """The upper wind toward the east by latitude, the jets' 1 the fastest
    (:attr:`WeatherSpec.jets`, moved north by the season).

    :param lat: latitudes, degrees
    """
    points = np.array(spec.jets, dtype=np.float64)
    return np.interp(np.abs(np.asarray(lat) - spec.season), points[:, 0], points[:, 1])


def stream_table(spec: WeatherSpec, size: int = STREAM_TABLE_SIZE) -> np.ndarray:
    """The zonal upper wind's stream function at *size* latitudes evenly
    from -90 to 90 degrees, metres: ``psi = -R * integral(u dlat)``, so
    the wind blows along its contours and ``|grad psi|`` is its speed. The
    cirrus fibres are those contours (:mod:`cirrus`), one every fibre
    spacing where the jet blows its fastest.

    :returns: (size,) float64
    """
    lat = np.linspace(-90.0, 90.0, size)
    u = upper_wind(spec, lat)
    step = math.radians(180.0 / (size - 1))
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * step)])
    return -spec.planet.radius * (integral - integral[size // 2])


def stream_vortices(spec: WeatherSpec, planet: Planet | None = None) -> np.ndarray:
    """The weather systems' share of the upper wind's stream function: one
    Gaussian each, ``a * exp(-|d - c|^2 * k)`` over unit directions d, so
    the wind turns around each low (cyclonic) and over each tropical
    cyclone's outflow (anticyclonic), at most :attr:`WeatherSpec.jet_vortex`
    of the jets' speed.

    :param planet: the evaluated planet (built from the spec if None)
    :returns: (N, 5) float64: centre (3), k, a (metres)
    """
    radius = spec.planet.radius
    rows = []
    for v in vortices(spec, planet):
        width = _reach(v) * v.radius
        amplitude = spec.jet_vortex * width * radius / _GAUSS_SLOPE
        # A low's stream function is a minimum where it turns counter-
        # clockwise (swirl > 0); the outflow aloft a tropical cyclone turns
        # the other way.
        sign = -math.copysign(1.0, v.swirl) * (-1.0 if v.tropical else 1.0)
        rows.append((*v.centre, 1.0 / width**2, sign * amplitude))
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def stream_function(
    table: np.ndarray, vortex_rows: np.ndarray, dirs: np.ndarray
) -> np.ndarray:
    """The upper wind's stream function at *dirs* (without the cirrus's
    meanders), as the cirrus shader evaluates it.

    :param table: :func:`stream_table`
    :param vortex_rows: :func:`stream_vortices`
    :param dirs: (N, 3) unit directions
    :returns: (N,) metres
    """
    lat, _ = latlon(dirs)
    psi = np.interp(lat, np.linspace(-90.0, 90.0, len(table)), table)
    for cx, cy, cz, k, a in vortex_rows:
        offset = dirs - np.array([cx, cy, cz])
        psi = psi + a * np.exp(-np.einsum("ij,ij->i", offset, offset) * k)
    return psi


def _cirrus_compose(spec: WeatherSpec, dirs: np.ndarray, planet: Planet, climate):
    radius = spec.planet.radius
    scale = _feature_scale(spec)
    lat, _ = latlon(dirs)
    points = np.array(spec.cirrus_zonal, dtype=np.float64)
    cover = np.interp(np.abs(lat - spec.season), points[:, 0], points[:, 1])
    # Ice cloud high above the ground cares less for its humidity than the
    # cumulus: jet cirrus streams over deserts too.
    _, _, humidity, _ = climate(dirs)
    cover = cover * (0.4 + 0.6 * _wetness(humidity))

    warped = dirs.copy()
    for v in vortices(spec, planet):
        near = _near(dirs, v)
        if not near.any():
            continue
        warped[near] = _wind_up(dirs[near], warped[near], v)
        x, y = _local(dirs[near], v)
        if v.tropical:
            shield = _gauss(x, y, 0.0, 0.0, 1.6, 1.6)  # the outflow's canopy
        else:
            shield = _gauss(x, y, 0.9, 1.0, 1.3, 1.0)  # ahead of the warm front
        cover[near] += spec.cirrus_shield * shield
    meso = fbm(
        warped * radius,
        spec.meso_wavelength * scale,
        3,
        spec.seed * 1000 + _SEED_CIRRUS,
    )
    return np.clip(cover + spec.cirrus_meso * meso, 0.0, 1.0)


def cirrus_cover_at(
    spec: WeatherSpec, dirs: np.ndarray, planet: Planet | None = None
) -> np.ndarray:
    """The cirrus's cover at arbitrary directions (no cube map; the climate
    sampled directly).

    :param dirs: (N, 3) unit directions
    :returns: (N,) in [0, 1]
    """
    planet = planet or Planet(spec.planet)
    return _cirrus_compose(spec, dirs, planet, _climate_direct(spec, planet))


@functools.cache
def cirrus_cover_map(spec: WeatherSpec, size: int = CIRRUS_SIZE) -> np.ndarray:
    """The cirrus's cover as a cube map (cached per process): the share of
    the sheet that holds cirrus.

    :returns: (6, size, size) float32 in [0, 1], the layout of
        :func:`sky.starfield.cube_directions`
    """
    planet = Planet(spec.planet)
    dirs = cube_directions(size).reshape(-1, 3)
    cover = _cirrus_compose(spec, dirs, planet, _climate_gridded(spec, planet))
    return cover.reshape(6, size, size).astype(np.float32)
