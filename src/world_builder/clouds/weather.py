"""
The weather over a planet: how much of the sky is cloud, by direction and
by genus, at a time.

The weather is synoptic (:mod:`synoptic`): jets meandering as Rossby waves,
lows with their fronts, highs, cut-off lows and tropical cyclones, each an
analytic object on a closed-form track, so the weather at any
:attr:`WeatherSpec.time_s` is a pure function of the spec. Its cube map
(:func:`synoptic_cube`) holds, per direction:

1. the **convective** continuum's coverage (the share of the slab's columns
   holding cloud), its **depth** (humilis to cumulonimbus) and the anvils'
   **shelf**: a fair-weather background from the climate (the wet tropics,
   the storm tracks, the planet's humidity, deserts cleared, the sea, slopes
   facing the wind lifting cloud, a mesoscale texture curled round the
   systems), with the cold fronts' cumulonimbus lines, the cold air's open
   cells, the tropical cyclones' eyewalls and bands, air-mass convection;
2. every other genus's coverage (:data:`synoptic.rules.LAYERS`): the warm
   fronts' cirrus, cirrostratus, altostratus and nimbostratus, the warm
   sectors' stratus and stratocumulus, the highs' stratocumulus decks and
   fog (:mod:`synoptic.rules`).

Highs, eyes and dry slots clear what lies under them.

The cirrus above (the march's high layer, :mod:`genera`) has a weather of
its own, from the same systems: an **upper wind** (the meandering jets,
swirled round the systems: :func:`upper_stream`), whose stream function
sets the direction of its fibres, and a cube map of its **cover**
(:func:`cirrus_cover_map`): along the jets, over the wet tropics (anvils),
and the systems' own cirrus.

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

from world_builder.clouds.synoptic.fields import wind as surface_wind
from world_builder.clouds.synoptic.jets import across_axis, system_rows
from world_builder.clouds.synoptic.jets import upper_stream as jets_upper_stream
from world_builder.clouds.synoptic.rules import (
    LAYERS,
    GenusFields,
    apply_systems,
    instability,
    swirled,
    zonal_cover,
)
from world_builder.clouds.synoptic.state import synoptic_state
from world_builder.sky.starfield import cube_directions, directions_to_texels
from world_builder.terrain.chart import latlon
from world_builder.terrain.noise import fbm
from world_builder.terrain.planet import (
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
_SEED_CIRRUS = 419


@dataclass(frozen=True)
class WeatherSpec:
    """Everything that defines a planet's weather; frozen and picklable.

    The weather's scales (the systems' sizes, the belts' latitudes, the
    jets) follow the planet's radius and rotation (:class:`synoptic.Scales`).
    Lengths here are for a :data:`terrain.planet.REFERENCE_RADIUS_M` planet
    and follow its feature scale. Coverages are shares of the sky in [0, 1].

    :ivar planet: the planet whose climate and terrain the weather follows
    :ivar seed: the jets, the weather systems and the mesoscale texture
    :ivar time_s: the time the weather is evaluated at, seconds: the
        systems move and live on closed-form tracks (:mod:`synoptic`), so
        any time is as cheap as any other
    :ivar systems: False for a weather without systems (no lows, highs or
        tropical cyclones: the climate's background alone)
    :ivar zonal: (|latitude| degrees, coverage) control points of the
        fair-weather background, or None for the planet's own
        (:func:`synoptic.rules.zonal_cover`)
    :ivar season: degrees north the equatorial band is moved (it follows
        the sun)
    :ivar meso_wavelength: the mesoscale texture's largest feature, metres
    :ivar meso_strength: its amplitude, in coverage
    :ivar dry_floor: what the background is multiplied by with no humidity
    :ivar desert_clearing: share of the cover hot, dry land loses
    :ivar maritime: cover added over the sea
    :ivar orographic: cover added on a steep windward slope of humid land
        (taken in the lee)
    :ivar orographic_slope: the slope along the wind that gives 76 % of it
    :ivar climate_wavelength: the shortest relief and climate features the
        weather follows, metres (the mountain ranges, not single peaks)
    :ivar jets: (|latitude| degrees, eastward wind) control points of a
        zonal upper wind replacing the meandering jets, the fastest 1, or
        None for the planet's own (:mod:`synoptic.jets`)
    :ivar jet_vortex: the systems' upper wind at its fastest, relative to
        the fastest jet's (the cirrus fibres bend around them)
    :ivar cirrus_zonal: (|latitude| degrees, cover) control points of the
        cirrus's background, or None for the planet's own: anvils over the
        wet tropics, streaks along the jets
    :ivar cirrus_shield: the share of the systems' own cirrus (the warm
        fronts' shields, the tropical cyclones' outflow) the cirrus keep
    :ivar cirrus_meso: amplitude of the cirrus cover's mesoscale texture
    """

    planet: PlanetSpec
    seed: int = 11
    time_s: float = 0.0
    systems: bool = True
    zonal: tuple | None = None
    season: float = 0.0
    meso_wavelength: float = 160e3
    meso_strength: float = 0.4
    dry_floor: float = 0.12
    desert_clearing: float = 0.6
    maritime: float = 0.08
    orographic: float = 0.3
    orographic_slope: float = 0.03
    climate_wavelength: float = 60e3
    jets: tuple | None = None
    jet_vortex: float = 0.8
    cirrus_zonal: tuple | None = None
    cirrus_shield: float = 1.0
    cirrus_meso: float = 0.3


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
    """The (6, n, n, 2) weather (coverage, shift) as an RG32F cube map, or a
    (6, n, n, 3) one with the convective depth too, or (6, n, n, 4) with the
    anvil shelf's shift too (:mod:`genera`) as an RGBA32F one, linearly
    filtered (:func:`sample_cube` is its CPU copy)."""
    size = cube.shape[1]
    tex = Texture("cloud_weather")
    if cube.shape[-1] == 2:
        tex.setup_cube_map(size, Texture.T_float, Texture.F_rg32)
    else:
        tex.setup_cube_map(size, Texture.T_float, Texture.F_rgba32)
        pad = np.zeros(cube.shape[:-1] + (4 - cube.shape[-1],), dtype=np.float32)
        # Panda keeps an RGBA texture's RAM image as BGRA.
        cube = np.concatenate([cube, pad], axis=-1)[..., [2, 1, 0, 3]]
    # Pages in face order, rows bottom-up = t up, as the star field's.
    tex.set_ram_image(np.ascontiguousarray(cube, dtype=np.float32).tobytes())
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


def build_coverage_texture(cube: np.ndarray) -> Texture:
    """A (6, n, n, 4) coverage cube (:func:`genera.coverage_cube`) as an
    RGBA16F cube map, linearly filtered: each channel one genus's coverage."""
    size = cube.shape[1]
    tex = Texture("cloud_genus_coverage")
    tex.setup_cube_map(size, Texture.T_float, Texture.F_rgba16)
    # Panda keeps an RGBA texture's RAM image as BGRA.
    data = np.ascontiguousarray(cube[..., [2, 1, 0, 3]], dtype=np.float32)
    tex.set_ram_image(data.tobytes())
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


def build_high_wind_texture(cube: np.ndarray) -> Texture:
    """The upper wind (:func:`genera.high_wind_cube`: the wind, its stream
    function) as an RGBA32F 2D texture array, a layer per cube face, rows
    t up (``highWind``). Not a cube map: its hardware filter weighs the
    texels in 8 bits, which stepped the stream function by about 30 m; the
    shader blends four texels in float (``highWindAt``, as
    :func:`sample_cube`). 32 bits: the stream function spans hundreds of
    kilometres, and the fibres need it to metres."""
    size = cube.shape[1]
    tex = Texture("cloud_high_wind")
    tex.setup_2d_texture_array(size, size, 6, Texture.T_float, Texture.F_rgba32)
    # Panda keeps an RGBA texture's RAM image as BGRA.
    data = np.ascontiguousarray(cube[..., [2, 1, 0, 3]], dtype=np.float32)
    tex.set_ram_image(data.tobytes())
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_minfilter(SamplerState.FT_nearest)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


def build_shift_texture(tables: list) -> Texture:
    """The genera's coverage-to-shift tables (:func:`shift_table`) as one
    R32F texture, a row each (``genusShift``: x the coverage, linearly
    filtered as :func:`coverage_to_shift` interpolates; y the genus's row,
    read at its centre), :data:`noise.MAX_GENERA` rows, unused ones 0."""
    from world_builder.clouds.noise import MAX_GENERA

    rows = np.zeros((MAX_GENERA, SHIFT_TABLE_SIZE), dtype=np.float32)
    for row, table in enumerate(tables):
        rows[row] = np.asarray(table, dtype=np.float32)
    tex = Texture("cloud_genus_shift")
    tex.setup_2d_texture(SHIFT_TABLE_SIZE, MAX_GENERA, Texture.T_float, Texture.F_r32)
    tex.set_ram_image(rows.tobytes())
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


# ── The climate ──────────────────────────────────────────────────────────────


def _feature_scale(spec: WeatherSpec) -> float:
    return scaled_params(spec.planet)[0].feature_scale


def _wind_east(lat: np.ndarray) -> np.ndarray:
    """Prevailing wind toward the east (+1) or west (-1) by latitude: the
    easterly trades, the westerlies, the polar easterlies (when no wind is
    given)."""
    return np.interp(np.abs(lat), [0, 25, 35, 55, 65, 90], [-1, -1, 1, 1, -1, -1])


def orographic(
    dirs: np.ndarray,
    elevation_at,
    spec: WeatherSpec,
    step: float,
    wind: np.ndarray | None = None,
) -> np.ndarray:
    """Cover from the terrain lifting the wind: more on windward slopes,
    less in the lee.

    :param dirs: (N, 3) unit directions
    :param elevation_at: ``(N, 3) dirs -> (N,)`` metres above sea level
    :param spec: the weather (its orographic gain and slope)
    :param step: metres upwind the slope is measured over
    :param wind: (N, 2) the surface wind (east, north), any speed; None for
        the prevailing wind by latitude
    :returns: (N,) coverage to add (negative in the lee)
    """
    lat, _ = latlon(dirs)
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, dirs)
    east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-9)
    if wind is None:
        along = east * _wind_east(lat)[:, None]
    else:
        north = np.cross(dirs, east)
        along = east * wind[:, :1] + north * wind[:, 1:]
        along /= np.maximum(np.linalg.norm(along, axis=1, keepdims=True), 1e-9)
    upwind = dirs - along * (step / spec.planet.radius)
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
    """What the background cover is multiplied by: humidity, and hot dry
    land cleared (deserts stay clear).

    :param temperature: degrees Celsius, as :meth:`terrain.planet.Planet.sample_dirs`
    :param humidity: in [0, 1], likewise
    """
    moist = spec.dry_floor + (1.0 - spec.dry_floor) * _wetness(humidity)
    hot = np.clip((temperature - 22.0) / 12.0, 0.0, 1.0)
    dry = 1.0 - np.clip((humidity - 0.2) / 0.35, 0.0, 1.0)
    return moist * (1.0 - spec.desert_clearing * hot * dry)


def _sampler(spec, planet):
    """:returns: ``dirs -> (elevation, temperature, humidity)`` band-limited
    to the climate wavelength, and the orographic slope step, metres"""
    scale = _feature_scale(spec)
    wavelength = spec.climate_wavelength * scale

    def sample(dirs):
        return planet.sample_dirs(dirs, min_wavelength=wavelength)

    return sample, 0.4 * wavelength


def _climate(spec, planet, dirs):
    """:returns: elevation, temperature, humidity and orographic cover (the
    terrain lifting the systems' surface wind)"""
    sample, step = _sampler(spec, planet)
    elevation, temperature, humidity = sample(dirs)
    wind = None
    if spec.systems:
        state = synoptic_state(spec)
        wind = surface_wind(dirs, state.systems, state.scales, state.t)
    lift = orographic(dirs, lambda d: sample(d)[0], spec, step, wind)
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


# ── The genera ────────────────────────────────────────────────────────────────

#: How much the mesoscale texture modulates the systems' layers, relative to
#: its share of the cumulus's coverage.
LAYER_TEXTURE = 2.0
#: The channels of :func:`synoptic_cube`.
SYNOPTIC_CHANNELS = ("convective", "depth", "shelf") + LAYERS


def _background_zonal(spec: WeatherSpec, lat: np.ndarray) -> np.ndarray:
    if spec.zonal is not None:
        points = np.array(spec.zonal, dtype=np.float64)
        return np.interp(np.abs(lat - spec.season), points[:, 0], points[:, 1])
    return zonal_cover(lat, synoptic_state(spec).scales, itcz=spec.season)


def genus_fields(
    spec: WeatherSpec, dirs: np.ndarray, planet: Planet, climate
) -> GenusFields:
    """The genera at *dirs* (:mod:`synoptic.rules`): the climate's
    background of fair-weather cumulus and air-mass convection, with the
    systems drawn over it.

    :param climate: ``dirs -> (elevation, temperature, humidity, lift)``
    """
    state = synoptic_state(spec)
    radius = spec.planet.radius
    lat, _ = latlon(dirs)
    elevation, temperature, humidity, lift = climate(dirs)
    sea = np.zeros(len(dirs))
    if spec.planet.has_ocean:
        sea = np.clip(-elevation / 200.0, 0.0, 1.0)
    base = _background_zonal(spec, lat) * climate_factor(spec, temperature, humidity)
    base = base + lift + spec.maritime * sea
    warped = swirled(dirs, state.systems, state.scales, state.t)
    meso = fbm(
        warped * radius,
        spec.meso_wavelength * _feature_scale(spec),
        3,
        spec.seed * 1000 + _SEED_MESO,
    )
    base = np.clip(base + spec.meso_strength * meso, 0.0, 1.0)
    unstable = instability(humidity, temperature)
    tropics = 0.5 + 0.5 * np.exp(-(((lat - spec.season) / 6.0) ** 2))
    base = 1.0 - (1.0 - base) * (1.0 - 0.4 * unstable * tropics)
    out = apply_systems(
        dirs,
        state.systems,
        state.scales,
        state.t,
        base=base,
        sea=sea,
        temperature=temperature,
        elevation=elevation,
        depth=unstable,
    )
    # The systems' layers carry the mesoscale texture too: decks and shields
    # break up as the cumulus do, with no stencilled edges.
    texture = np.clip(1.0 + LAYER_TEXTURE * spec.meso_strength * meso, 0.0, 2.0)
    for genus, cover in out.layers.items():
        out.layers[genus] = np.clip(cover * texture, 0.0, 1.0)
    return out


def coverage_at(
    spec: WeatherSpec, dirs: np.ndarray, planet: Planet | None = None
) -> np.ndarray:
    """The weather's convective coverage at arbitrary directions (no cube
    map). The climate is sampled directly here; :func:`synoptic_cube`
    samples it on a coarse grid and upsamples, which is what makes the map
    cheap.

    :param dirs: (N, 3) unit directions
    :param planet: the evaluated planet (built from the spec if None)
    :returns: (N,) in [0, 1]
    """
    planet = planet or Planet(spec.planet)
    return genus_fields(spec, dirs, planet, _climate_direct(spec, planet)).convective


def synoptic_cube(spec: WeatherSpec, size: int = WEATHER_SIZE) -> np.ndarray:
    """The weather's genera as a cube map (cached per process).

    :returns: (6, size, size, len(:data:`SYNOPTIC_CHANNELS`)) float32, the
        layout of :func:`sky.starfield.cube_directions`
    """
    return _synoptic_cube(spec, int(size))


@functools.cache
def _synoptic_cube(spec: WeatherSpec, size: int) -> np.ndarray:
    planet = Planet(spec.planet)
    dirs = cube_directions(size).reshape(-1, 3)
    out = genus_fields(spec, dirs, planet, _climate_gridded(spec, planet))
    channels = [out.convective, out.depth, out.shelf] + [out.layers[g] for g in LAYERS]
    cube = np.stack(channels, axis=-1).reshape(6, size, size, len(channels))
    return cube.astype(np.float32)


@functools.cache
def coverage_map(spec: WeatherSpec, size: int = WEATHER_SIZE) -> np.ndarray:
    """The weather's convective coverage as a cube map (cached per process).

    :returns: (6, size, size) float32 in [0, 1], the layout of
        :func:`sky.starfield.cube_directions`
    """
    return np.ascontiguousarray(synoptic_cube(spec, size)[..., 0])


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


def upper_wind(spec: WeatherSpec, lat: np.ndarray) -> np.ndarray:
    """The zonal-mean upper wind toward the east by latitude, the fastest
    jet's 1: :attr:`WeatherSpec.jets`, or the planet's jets' profiles at
    their mean latitudes (moved north by the season).

    :param lat: latitudes, degrees
    """
    lat = np.asarray(lat, dtype=np.float64)
    if spec.jets is not None:
        points = np.array(spec.jets, dtype=np.float64)
        return np.interp(np.abs(lat - spec.season), points[:, 0], points[:, 1])
    state = synoptic_state(spec)
    top = max(j.speed for j in state.jets)
    out = np.zeros_like(lat)
    for jet in state.jets:
        y = spec.planet.radius * np.radians(
            lat - spec.season - jet.hemisphere * jet.lat0
        )
        out = np.maximum(out, jet.speed / top * np.exp(-((y / jet.width) ** 2)))
    return out


def stream_table(spec: WeatherSpec, size: int = STREAM_TABLE_SIZE) -> np.ndarray:
    """The zonal-mean upper wind's stream function at *size* latitudes
    evenly from -90 to 90 degrees, metres: ``psi = -R * integral(u dlat)``,
    so the wind blows along its contours and ``|grad psi|`` is its speed.

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
    the wind turns around each low (cyclonic), each high and each tropical
    cyclone's outflow (anticyclonic), at most :attr:`WeatherSpec.jet_vortex`
    of the fastest jet's speed (:func:`synoptic.jets.system_rows`).

    :param planet: unused (the systems are the spec's)
    :returns: (N, 5) float64: centre (3), k, a (metres)
    """
    state = synoptic_state(spec)
    return system_rows(state.systems, state.t, spec.planet.radius, spec.jet_vortex)


def stream_function(
    table: np.ndarray, vortex_rows: np.ndarray, dirs: np.ndarray
) -> np.ndarray:
    """The zonal-mean upper wind's stream function at *dirs*, with the
    systems' bumps (the meandering jets are :func:`upper_stream`'s).

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


def upper_stream(spec: WeatherSpec, dirs: np.ndarray) -> np.ndarray:
    """The upper wind's stream function at *dirs*, metres (the fastest jet's
    speed is 1): the meandering jets swirled round the systems
    (:func:`synoptic.jets.upper_stream`), or the zonal one of
    :attr:`WeatherSpec.jets`. The wind is ``up x grad psi``.

    :param dirs: (N, 3) unit directions
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    if spec.jets is not None:
        return stream_function(stream_table(spec), stream_vortices(spec), dirs)
    state = synoptic_state(spec)
    return jets_upper_stream(state.jets, state.systems, state.t, dirs, spec.jet_vortex)


def _cirrus_zonal(spec: WeatherSpec, dirs: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """The cirrus's background: anvils over the wet tropics, streaks along
    the meandering jets."""
    if spec.cirrus_zonal is not None:
        points = np.array(spec.cirrus_zonal, dtype=np.float64)
        return np.interp(np.abs(lat - spec.season), points[:, 0], points[:, 1])
    state = synoptic_state(spec)
    _, lon = latlon(dirs)
    tropics = np.exp(-(((lat - spec.season) / (0.5 * state.scales.hadley_edge)) ** 2))
    streaks = np.zeros_like(lat)
    for jet in state.jets:
        y = across_axis(jet, lat, lon, state.t)
        streaks = np.maximum(streaks, np.exp(-((y / jet.width) ** 2)))
    return 0.08 + 0.45 * tropics + 0.3 * streaks


def _cirrus_compose(spec: WeatherSpec, dirs: np.ndarray, planet: Planet, climate):
    radius = spec.planet.radius
    scale = _feature_scale(spec)
    lat, _ = latlon(dirs)
    cover = _cirrus_zonal(spec, dirs, lat)
    # Ice cloud high above the ground cares less for its humidity than the
    # cumulus: jet cirrus streams over deserts too.
    _, _, humidity, _ = climate(dirs)
    cover = cover * (0.4 + 0.6 * _wetness(humidity))
    # The systems' own: the warm fronts' shields, the tropical cyclones'
    # outflow.
    cube = synoptic_cube(spec)
    pick = [
        SYNOPTIC_CHANNELS.index(g) for g in ("cirrus", "cirrostratus", "cirrocumulus")
    ]
    own = spec.cirrus_shield * sample_cube(cube[..., pick], dirs).max(axis=-1)
    cover = 1.0 - (1.0 - np.clip(cover, 0.0, 1.0)) * (1.0 - np.clip(own, 0.0, 1.0))
    state = synoptic_state(spec)
    warped = swirled(dirs, state.systems, state.scales, state.t)
    meso = fbm(
        warped * radius,
        spec.meso_wavelength * scale,
        3,
        spec.seed * 1000 + _SEED_CIRRUS,
    )
    return np.clip(cover + spec.cirrus_meso * meso, 0.0, 1.0)


@functools.cache
def cirrus_cover_map(spec: WeatherSpec, size: int = CIRRUS_SIZE) -> np.ndarray:
    """The cirrus's cover as a cube map (cached per process): the share of
    the high layer that holds cirrus (:attr:`genera.GeneraSpec.cirrus_weather`).

    :returns: (6, size, size) float32 in [0, 1], the layout of
        :func:`sky.starfield.cube_directions`
    """
    planet = Planet(spec.planet)
    dirs = cube_directions(size).reshape(-1, 3)
    cover = _cirrus_compose(spec, dirs, planet, _climate_gridded(spec, planet))
    return cover.reshape(6, size, size).astype(np.float32)
