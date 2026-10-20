"""
Planet definition: elevation and climate as pure functions of a sphere direction.

A :class:`PlanetSpec` is a small frozen, picklable description (seed, noise
parameters, climate bias, hand-made stamps). :class:`Planet` evaluates it.

Elevation, in metres above sea level, sums these layers at the sample point
``p = dir * R``:

1. **warp**: displace p by a smooth vector field, which bends coastlines and
   ranges into meanders
2. **continents**: low-frequency fBm through a smooth curve (abyss, shelf,
   coast, plains, highlands). The sea bias is set so ``ocean_fraction`` of the
   sphere lies below sea level
3. **mountains**: ridged multifractal, masked into ranges, on land only
4. **hills**, then **detail**: fBm down to the sampling resolution (band
   limited, so coarse samples never alias detail they cannot represent)
5. **stamps**: hand-made heightmaps, in list order

Planet scale: the radius is the one knob. :class:`ElevationParams` and
:class:`ClimateParams` hold the planetary-scale lengths (warp, continents,
mountains, climate) for a :data:`REFERENCE_RADIUS_M` planet; :class:`Planet`
multiplies them by ``feature_scale`` (radius / reference unless set), so a
bigger planet gets bigger continents. Ground-level character (hills, detail,
local climate patches) stays in absolute metres at every radius.

Fields with no wavelength below ~10 km (warp, continents, the mountain mask,
the base climate) are the "low band". :meth:`Planet.sample_face_grid` evaluates
them on a fixed lattice of each cube face (see :mod:`cubesphere`) and
interpolates (Catmull-Rom), which halves the cost of a fine chunk. The lattice
is global, so every LOD level interpolates the same nodes and the levels agree.
Everything else is evaluated per sample.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.interpolate import PchipInterpolator

from world_builder.terrain.chart import dirs_from_latlon, latlon
from world_builder.terrain.cubesphere import face_dirs, grid_for
from world_builder.terrain.noise import fbm, ridged, smoothstep, warp

#: Default terrain planet radius, metres (a small, Kerbin-like world).
TERRAIN_RADIUS_M = 600e3
#: Radius the planetary-scale lengths in the params are given for, metres.
REFERENCE_RADIUS_M = 600e3
#: Planetary features stop shrinking below this feature scale (a 150 km
#: planet), or they would reach the hills' wavelength.
MIN_FEATURE_SCALE = 0.25
#: Aimed-for spacing of the low-band lattice at feature scale 1, metres. Its
#: shortest field (the warp's top octave, 10 km) gets ~40 nodes per wavelength.
LOW_BAND_TARGET = 300.0
#: Mountain height scales with sqrt(feature scale), within these factors.
MOUNTAIN_SCALE_RANGE = (2000.0 / 4800.0, 7000.0 / 4800.0)
#: Directions sampled to set the sea bias from ``ocean_fraction``.
_SEA_BIAS_SAMPLES = 40000

# Seed offsets, so each field is independent for one planet seed.
_SEED_WARP = 11
_SEED_CONTINENT = 23
_SEED_MOUNTAIN = 37
_SEED_MASK = 41
_SEED_HILLS = 53
_SEED_DETAIL = 67
_SEED_TEMPERATURE = 79
_SEED_HUMIDITY = 83
_SEED_JITTER = 97


@dataclass(frozen=True)
class ElevationParams:
    """Shape of the relief. Lengths in metres.

    The planetary-scale lengths (warp, continents, mountains and their mask)
    and the mountain height are for a :data:`REFERENCE_RADIUS_M` planet and get
    scaled with the radius (see the module docstring). Hills and detail are
    absolute.

    :ivar continent_curve: (c, metres) control points mapping the continental
        field (sea bias removed, about [-1, 1]) to a base height; must pass
        through (0, 0), which is the coastline
    :ivar feature_scale: multiplier of the planetary-scale lengths; None =
        radius / REFERENCE_RADIUS_M. Set 1.0 to keep the lengths as given
    """

    warp_wavelength: float = 70e3
    #: Keep amplitude / wavelength near 0.06: a steeper warp folds space over
    #: itself and every warped field gets a crease line along the fold
    warp_amplitude: float = 4.2e3
    continent_wavelength: float = 380e3
    continent_octaves: int = 5
    ocean_fraction: float = 0.55
    continent_curve: tuple[tuple[float, float], ...] = (
        (-1.0, -5500.0),
        (-0.35, -4200.0),
        (-0.12, -2000.0),
        (-0.04, -120.0),
        (0.0, 0.0),
        (0.04, 30.0),
        (0.15, 180.0),
        (0.35, 700.0),
        (1.0, 1800.0),
    )
    mountain_wavelength: float = 40e3
    mountain_octaves: int = 8
    #: Height of the tallest crests (the ridged field cubed, so most of a
    #: range is far lower and the peaks stand out); scaled by sqrt(feature
    #: scale) within :data:`MOUNTAIN_SCALE_RANGE`
    mountain_height: float = 4800.0
    #: At least ~3x the mountain wavelength, or a range is a single ridge
    mountain_mask_wavelength: float = 140e3
    #: Raises (positive) or lowers the share of land that is mountain range
    mountain_coverage: float = 0.0
    hills_wavelength: float = 14e3
    hills_octaves: int = 5
    hills_height: float = 320.0
    detail_wavelength: float = 900.0
    detail_octaves: int = 8
    detail_height: float = 35.0
    feature_scale: float | None = None


@dataclass(frozen=True)
class ClimateParams:
    """Climate bias. Temperatures in degrees Celsius, humidity in [0, 1].

    :ivar temperature_mean: sea-level temperature, planet mean
    :ivar temperature_spread: amplitude of the regional variation
    :ivar temperature_latitude: equator-to-pole contrast
    :ivar lapse_rate: cooling per metre of altitude (6.5 C/km on Earth); this
        alone is what puts snow on peaks
    :ivar humidity_mean: planet mean humidity
    :ivar humidity_spread: amplitude of the regional variation
    :ivar temperature_wavelength: size of the regional temperature patches,
        metres (scaled with the radius, as the elevation's planetary lengths)
    :ivar humidity_wavelength: same for humidity
    """

    temperature_mean: float = 14.0
    temperature_spread: float = 9.0
    temperature_latitude: float = 40.0
    lapse_rate: float = 0.0065
    humidity_mean: float = 0.5
    humidity_spread: float = 0.35
    temperature_wavelength: float = 140e3
    temperature_octaves: int = 4
    humidity_wavelength: float = 105e3
    humidity_octaves: int = 4


@dataclass(frozen=True)
class PlanetSpec:
    """Everything that defines one planet; frozen and picklable.

    :ivar radius: sea-level radius, metres; the planetary-scale features
        follow it (see the module docstring)
    :ivar reference_lat: latitude of the origin of chart-metre coordinates in
        level files (stamp centres, camera poses), degrees
    :ivar reference_lon: its longitude, degrees
    :ivar has_vegetation: False for a barren world (no grass, no forest)
    :ivar stamps: hand-made heightmaps (:class:`stamps.Stamp`), in order
    :ivar paints: biome paint masks (:class:`stamps.Paint`), in order
    :ivar sea_bias: continental-field offset; None until :meth:`resolved`
    """

    name: str
    seed: int = 1
    radius: float = TERRAIN_RADIUS_M
    reference_lat: float = 42.0
    reference_lon: float = 10.0
    has_ocean: bool = True
    has_vegetation: bool = True
    elevation: ElevationParams = field(default_factory=ElevationParams)
    climate: ClimateParams = field(default_factory=ClimateParams)
    stamps: tuple = ()
    paints: tuple = ()
    sea_bias: float | None = None

    def resolved(self) -> PlanetSpec:
        """:returns: this spec with its derived sea bias filled in"""
        if self.sea_bias is not None:
            return self
        return replace(self, sea_bias=_sea_bias(self))


PRESETS: dict[str, PlanetSpec] = {
    "earthlike": PlanetSpec(name="earthlike"),
    "desert": PlanetSpec(
        name="desert",
        has_ocean=False,
        elevation=ElevationParams(
            ocean_fraction=0.3,
            mountain_height=2700.0,
            mountain_coverage=-0.1,
            hills_height=220.0,
            detail_height=25.0,
        ),
        climate=ClimateParams(
            temperature_mean=31.0,
            temperature_spread=6.0,
            temperature_latitude=20.0,
            humidity_mean=0.08,
            humidity_spread=0.1,
        ),
    ),
    "forest": PlanetSpec(
        name="forest",
        elevation=ElevationParams(
            ocean_fraction=0.35, mountain_height=3800.0, hills_height=420.0
        ),
        climate=ClimateParams(
            temperature_mean=17.0,
            temperature_spread=5.0,
            temperature_latitude=22.0,
            humidity_mean=0.78,
            humidity_spread=0.18,
        ),
    ),
    # A Mun-sized, airless ball of sand: dunes and crater-like basins (the
    # continental lows, dry), rock only on cliffs. No sea, no vegetation, no
    # climate: hot and bone dry everywhere, no snow even on the peaks.
    "moon": PlanetSpec(
        name="moon",
        radius=200e3,
        has_ocean=False,
        has_vegetation=False,
        elevation=ElevationParams(
            ocean_fraction=0.4,
            mountain_coverage=-0.1,
            hills_height=260.0,
            detail_height=30.0,
        ),
        climate=ClimateParams(
            temperature_mean=30.0,
            temperature_spread=0.0,
            temperature_latitude=0.0,
            lapse_rate=0.0,
            humidity_mean=0.0,
            humidity_spread=0.0,
        ),
    ),
    "ice": PlanetSpec(
        name="ice",
        has_ocean=False,
        elevation=ElevationParams(ocean_fraction=0.4, mountain_coverage=0.1),
        climate=ClimateParams(
            temperature_mean=-9.0,
            temperature_spread=8.0,
            temperature_latitude=20.0,
            humidity_mean=0.4,
            humidity_spread=0.2,
        ),
    ),
}


def scaled_params(spec: PlanetSpec) -> tuple[ElevationParams, ClimateParams]:
    """The spec's parameters with the planetary-scale lengths fitted to its radius.

    :param spec: the planet
    :returns: elevation and climate parameters as evaluated, with
        ``feature_scale`` set to the factor applied
    """
    e, c = spec.elevation, spec.climate
    s = (
        e.feature_scale
        if e.feature_scale is not None
        else spec.radius / REFERENCE_RADIUS_M
    )
    s = max(float(s), MIN_FEATURE_SCALE)
    lo, hi = MOUNTAIN_SCALE_RANGE
    e = replace(
        e,
        warp_wavelength=e.warp_wavelength * s,
        warp_amplitude=e.warp_amplitude * s,
        continent_wavelength=e.continent_wavelength * s,
        mountain_wavelength=e.mountain_wavelength * s,
        mountain_mask_wavelength=e.mountain_mask_wavelength * s,
        mountain_height=e.mountain_height * min(max(s**0.5, lo), hi),
        feature_scale=s,
    )
    c = replace(
        c,
        temperature_wavelength=c.temperature_wavelength * s,
        humidity_wavelength=c.humidity_wavelength * s,
    )
    return e, c


@dataclass
class GridSample:
    """Fields sampled on a grid of one cube face (row = v, column = u).

    :ivar dirs: (rows * columns, 3) sphere directions, row-major
    :ivar elevation: (rows, columns) metres above sea level
    :ivar temperature: (rows, columns) degrees Celsius, altitude lapse included
    :ivar humidity: (rows, columns) in [0, 1]
    """

    dirs: np.ndarray
    elevation: np.ndarray
    temperature: np.ndarray
    humidity: np.ndarray


class Planet:
    """Evaluates a :class:`PlanetSpec`. Cheap to build once the spec is resolved."""

    def __init__(self, spec: PlanetSpec):
        """:param spec: the planet; resolved here if its sea bias is unset"""
        self.spec = spec.resolved()
        self.elevation, self.climate = scaled_params(self.spec)
        self.grid = grid_for(self.spec.radius)
        #: Level-0 steps between low-band lattice nodes: a power of two, so
        #: coarse chunks' samples land on it; follows the feature scale.
        target = LOW_BAND_TARGET * self.elevation.feature_scale / self.grid.spacing0
        self.low_band_step = min(1 << max(0, round(math.log2(target))), self.grid.n0)
        points = np.array(self.elevation.continent_curve)
        self._continent_curve = PchipInterpolator(points[:, 0], points[:, 1])
        self._curve_range = (points[0, 0], points[-1, 0])
        self._stamp_bases: dict[int, float] = {}

    # ── low band ───────────────────────────────────────────────────────────────

    def low_band(self, dirs: np.ndarray) -> np.ndarray:
        """Fields with no detail below ~10 km, safe to interpolate.

        :param dirs: (N, 3) unit vectors
        :returns: (N, 7): warp displacement xyz (metres), continental field
            (sea bias removed), mountain mask, base temperature, base humidity
        """
        spec = self.spec
        e, c = self.elevation, self.climate
        p = dirs * spec.radius
        seed = spec.seed * 1000
        q = warp(p, e.warp_wavelength, e.warp_amplitude, 3, seed + _SEED_WARP)
        continent = (
            fbm(q, e.continent_wavelength, e.continent_octaves, seed + _SEED_CONTINENT)
            - spec.sea_bias
        )
        mask = fbm(q, e.mountain_mask_wavelength, 3, seed + _SEED_MASK)
        lat, _ = latlon(dirs)
        temperature = (
            c.temperature_mean
            + c.temperature_spread
            * 2.0
            * fbm(
                p,
                c.temperature_wavelength,
                c.temperature_octaves,
                seed + _SEED_TEMPERATURE,
            )
            + c.temperature_latitude * (np.cos(np.radians(lat)) ** 2 - 0.66)
        )
        humidity = c.humidity_mean + c.humidity_spread * 2.0 * fbm(
            p, c.humidity_wavelength, c.humidity_octaves, seed + _SEED_HUMIDITY
        )
        return np.column_stack([q - p, continent, mask, temperature, humidity])

    # ── full evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self, dirs: np.ndarray, low: np.ndarray, min_wavelength: float | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Elevation and climate from the low band plus per-sample detail.

        Stamps are not applied here; see :meth:`sample_face_grid`.

        :param dirs: (N, 3) unit vectors
        :param low: (N, 7) low band at the same points (:meth:`low_band`)
        :param min_wavelength: shortest wavelength to keep, metres; None = all
        :returns: elevation (metres), base temperature, humidity; each (N,)
        """
        spec = self.spec
        e = self.elevation
        seed = spec.seed * 1000
        p = dirs * spec.radius
        q = p + low[:, 0:3]
        continent = low[:, 3]
        lo, hi = self._curve_range
        base = self._continent_curve(np.clip(continent, lo, hi))
        land = smoothstep(0.0, 0.08, continent)

        range_mask = smoothstep(
            -0.05 - e.mountain_coverage, 0.3 - e.mountain_coverage, low[:, 4]
        )
        mountains = np.zeros(len(dirs))
        active = range_mask * land > 1e-4
        if np.any(active):
            crest = ridged(
                q[active],
                e.mountain_wavelength,
                e.mountain_octaves,
                seed + _SEED_MOUNTAIN,
                min_wavelength=min_wavelength,
            )
            mountains[active] = (
                e.mountain_height * crest**3 * (range_mask * land)[active]
            )

        hills = (
            e.hills_height
            * fbm(
                q,
                e.hills_wavelength,
                e.hills_octaves,
                seed + _SEED_HILLS,
                min_wavelength=min_wavelength,
            )
            * (0.4 + 0.6 * land)
        )
        detail = (
            e.detail_height
            * fbm(
                q,
                e.detail_wavelength,
                e.detail_octaves,
                seed + _SEED_DETAIL,
                min_wavelength=min_wavelength,
            )
            * (1.0 + 2.0 * range_mask * land)
        )
        elevation = base + mountains + hills + detail

        # Local climate patches: a few km, so low flight crosses biome edges.
        jitter_t = fbm(p, 5e3, 1, seed + _SEED_JITTER)
        jitter_h = fbm(p, 7e3, 2, seed + _SEED_JITTER + 1)
        temperature = low[:, 5] + 3.0 * jitter_t
        humidity = np.clip(low[:, 6] + 0.12 * jitter_h, 0.0, 1.0)
        return elevation, temperature, humidity

    def lapse(self, temperature: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """:returns: *temperature* cooled by altitude above sea level"""
        return temperature - self.spec.climate.lapse_rate * np.maximum(elevation, 0.0)

    def sample_dirs(
        self,
        dirs: np.ndarray,
        min_wavelength: float | None = None,
        apply_stamps: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct (uninterpolated) evaluation at arbitrary directions.

        :param dirs: (N, 3) unit vectors
        :param min_wavelength: shortest wavelength to keep, metres
        :param apply_stamps: whether to blend the spec's stamps in (their mip
            level follows min_wavelength / 4)
        :returns: elevation, temperature (lapse included), humidity; each (N,)
        """
        dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
        elevation, temperature, humidity = self.evaluate(
            dirs, self.low_band(dirs), min_wavelength
        )
        if apply_stamps:
            spacing = min_wavelength / 4.0 if min_wavelength else 1.0
            elevation = self._apply_stamps(dirs, elevation, spacing)
        return elevation, self.lapse(temperature, elevation), humidity

    def sample_face_grid(
        self,
        face: int,
        ka: np.ndarray,
        kb: np.ndarray,
        step: int,
        apply_stamps: bool = True,
    ) -> GridSample:
        """Sample a grid of one cube face.

        The low band comes from the face lattice when the grid is finer than
        it, and is evaluated directly otherwise. On a coarse chunk, whose
        samples lie on the lattice, both give the same values.

        :param face: cube face, 0-5
        :param ka: (columns,) level-0 indices along u, ascending, evenly spaced
        :param kb: (rows,) level-0 indices along v, same spacing
        :param step: level-0 steps between samples (2**level)
        :param apply_stamps: whether to blend the spec's stamps in
        :returns: the sampled fields
        """
        grid = self.grid
        ga, gb = np.meshgrid(grid.param(ka), grid.param(kb))
        dirs = face_dirs(face, ga, gb)
        if step >= self.low_band_step:
            low = self.low_band(dirs)
        else:
            low = self._interpolated_low_band(face, ka, kb)
        spacing = grid.spacing0 * step
        elevation, temperature, humidity = self.evaluate(dirs, low, 4.0 * spacing)
        if apply_stamps:
            elevation = self._apply_stamps(dirs, elevation, spacing)
        temperature = self.lapse(temperature, elevation)
        shape = (len(kb), len(ka))
        return GridSample(
            dirs=dirs,
            elevation=elevation.reshape(shape),
            temperature=temperature.reshape(shape),
            humidity=humidity.reshape(shape),
        )

    def stamp_base(self, index: int) -> float:
        """Procedural ground height at a stamp's centre (what ``flatten`` levels to).

        Full detail, no stamps, so every LOD level flattens to the same height.

        :param index: index into the spec's stamps
        :returns: metres above sea level
        """
        if index not in self._stamp_bases:
            placement = self.spec.stamps[index].placement
            dirs = dirs_from_latlon(placement.lat, placement.lon)
            elevation, _, _ = self.evaluate(dirs, self.low_band(dirs), None)
            self._stamp_bases[index] = float(elevation[0])
        return self._stamp_bases[index]

    def _apply_stamps(self, dirs, elevation, spacing):
        for index, stamp in enumerate(self.spec.stamps):
            elevation = stamp.apply(
                dirs, elevation, spacing, self.spec.radius, self.stamp_base(index)
            )
        return elevation

    def _interpolated_low_band(
        self, face: int, ka: np.ndarray, kb: np.ndarray
    ) -> np.ndarray:
        """Low band at face grid points, by Catmull-Rom from the face lattice.

        :param face: cube face
        :param ka: (columns,) level-0 indices
        :param kb: (rows,) level-0 indices
        :returns: (rows * columns, 7), row-major
        """
        step = float(self.low_band_step)
        wa, la = _catmull_rom_weights(np.asarray(ka, dtype=np.float64), step)
        wb, lb = _catmull_rom_weights(np.asarray(kb, dtype=np.float64), step)
        nodes = self._lattice_nodes(face, la, lb)  # (len(lb), len(la), 7)
        # rows: wb @ nodes, columns: @ wa.T, per channel
        out = np.einsum("ri,ijc,sj->rsc", wb, nodes, wa, optimize=True)
        return out.reshape(-1, nodes.shape[2])

    def _lattice_nodes(self, face: int, la: np.ndarray, lb: np.ndarray) -> np.ndarray:
        """Low band at the lattice nodes la x lb (level-0 indices; nodes past
        the face edge are points of the extended face, still on the sphere).

        :returns: (len(lb), len(la), 7)
        """
        ga, gb = np.meshgrid(self.grid.param(la), self.grid.param(lb))
        dirs = face_dirs(face, ga, gb)
        return self.low_band(dirs).reshape(len(lb), len(la), -1)


def _catmull_rom_weights(coords: np.ndarray, step: float):
    """Separable Catmull-Rom weights onto a lattice of multiples of *step*.

    Interpolates exactly at the nodes, so a coarse chunk whose vertices sit on
    the lattice gets the node values.

    :param coords: (n,) sorted sample coordinates
    :param step: lattice spacing
    :returns: (n, m) weights and the (m,) lattice coordinates they apply to
    """
    first = np.floor(coords[0] / step) - 1.0
    last = np.floor(coords[-1] / step) + 2.0
    lattice = step * np.arange(first, last + 1.0)
    u = coords / step - first
    cell = np.floor(u).astype(np.intp)
    t = u - cell
    t2, t3 = t * t, t * t * t
    weights = np.zeros((len(coords), len(lattice)))
    rows = np.arange(len(coords))
    weights[rows, cell - 1] = 0.5 * (-t + 2.0 * t2 - t3)
    weights[rows, cell] = 0.5 * (2.0 - 5.0 * t2 + 3.0 * t3)
    weights[rows, cell + 1] = 0.5 * (t + 4.0 * t2 - 3.0 * t3)
    weights[rows, cell + 2] = 0.5 * (-t2 + t3)
    return weights, lattice


@functools.lru_cache(maxsize=16)
def _sea_bias(spec: PlanetSpec) -> float:
    """The continental offset putting ``ocean_fraction`` of the sphere underwater.

    :param spec: the planet (its own sea_bias is ignored)
    :returns: the bias to subtract from the continental field
    """
    rng = np.random.default_rng(spec.seed)
    dirs = rng.normal(size=(_SEA_BIAS_SAMPLES, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    unbiased = Planet.__new__(Planet)
    unbiased.spec = replace(spec, sea_bias=0.0)
    unbiased.elevation, unbiased.climate = scaled_params(spec)
    continent = unbiased.low_band(dirs)[:, 3]
    return float(np.quantile(continent, spec.elevation.ocean_fraction))
