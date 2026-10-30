"""
Planet definition: elevation and climate as pure functions of a sphere direction.

A :class:`PlanetSpec` is a small frozen, picklable description (seed, noise
parameters, climate bias, hand-made stamps). :class:`Planet` evaluates it.

Elevation, in metres above sea level, sums these layers at the sample point
``p = dir * R``:

1. **warp**: displace p by a smooth vector field, which bends coastlines and
   ranges into meanders
2. **continents**: low-frequency fBm, plus the tectonic plates' crust and
   belts (:mod:`plates`), through a smooth curve (abyss, shelf, coast,
   plains, highlands). The sea bias is set so ``ocean_fraction`` of the
   sphere lies below sea level. Plates also add mid-ocean ridges and rifts
3. **mountains**: ridged multifractal through a crest profile, under a
   summit level that varies along each range (massifs), masked into ranges
   (the plates' closing belts, and a few from a noise mask), on land only
4. **hills**, then **detail**: fBm down to the sampling resolution (band
   limited, so coarse samples never alias detail they cannot represent),
   scaled by the region's character (plains to rough country) and, near the
   sea, its coast style (smooth to rugged)
5. **desert landforms** on arid land (:mod:`landforms`): dunes, mesas and
   terraces, canyons
6. **stamps**: hand-made heightmaps, in list order

Planet scale: the radius is the one knob. :class:`ElevationParams` and
:class:`ClimateParams` hold the planetary-scale lengths (warp, continents,
mountains, climate) for a :data:`REFERENCE_RADIUS_M` planet; :class:`Planet`
multiplies them by ``feature_scale`` (radius / reference unless set), so a
bigger planet gets bigger continents. Ground-level character (hills, detail,
local climate patches) stays in absolute metres at every radius.

Fields with no wavelength below ~10 km (warp, continents, plates, the range
mask, character, coast style, the base climate and the desert masks) are the
"low band". :meth:`Planet.sample_face_grid` evaluates
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

from world_builder.terrain import landforms
from world_builder.terrain.biomes import aridity
from world_builder.terrain.chart import dirs_from_latlon, latlon
from world_builder.terrain.cubesphere import face_dirs, grid_for
from world_builder.terrain.noise import fbm, native_library, ridged, smoothstep, warp
from world_builder.terrain.plates import PlateFields, make_plates, plate_fields

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
_SEED_CHARACTER = 101
_SEED_COAST = 103
_SEED_PLATES = 107
_SEED_MASSIF = 109
_SEED_DUNES = 127
_SEED_MESAS = 131
_SEED_CANYONS = 137

# Columns of the low band (:meth:`Planet.low_band`).
LOW_WARP = slice(0, 3)
LOW_CONTINENT = 3
LOW_RANGE = 4
LOW_TEMPERATURE = 5
LOW_HUMIDITY = 6
LOW_CHARACTER = 7
LOW_COAST = 8
LOW_TECTONIC = 9
LOW_ARID = 10
LOW_ERG = 11
LOW_CANYON = 12
LOW_CHANNELS = 13

#: Mountain height (fraction of the range's summit level) against the ridged
#: crest field: broad valley floors, steep flanks, sharp crests. The crest
#: field's median is about 0.47, so a range is mostly mountain, not spikes.
MOUNTAIN_PROFILE = (
    (0.0, 0.0),
    (0.1, 0.05),
    (0.3, 0.18),
    (0.5, 0.42),
    (0.7, 0.72),
    (0.9, 0.97),
    (1.0, 1.0),
)

#: Continental-field distance from the coast over which the coast style
#: replaces the inland relief (about 4 km of shelf and shore)
COAST_ZONE = 0.06


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
    #: Height of the tallest crests (through :data:`MOUNTAIN_PROFILE`, under
    #: a summit level that varies from massif to massif); scaled by
    #: sqrt(feature scale) within :data:`MOUNTAIN_SCALE_RANGE`
    mountain_height: float = 4800.0
    #: At least ~3x the mountain wavelength, or a range is a single ridge
    mountain_mask_wavelength: float = 140e3
    #: Raises (positive) or lowers the share of land that is mountain range
    mountain_coverage: float = 0.0
    #: Tectonic plates (:mod:`plates`); 0 = none, the ranges then come from
    #: the noise mask alone
    plates: int = 14
    #: The warp bending plate edges (scaled); keep the ratio near 0.06
    plate_warp_wavelength: float = 500e3
    plate_warp_amplitude: float = 30e3
    #: Half-width of the belt along a plate boundary: mountains where plates
    #: close, a ridge or rift where they open (scaled)
    belt_width: float = 80e3
    #: Pull of each plate's crust on the continental field, so that some
    #: coasts follow plate edges
    plate_crust: float = 0.2
    #: Rise of the continental field along closing plate edges: their
    #: mountain belts stand on land (island arcs, coastal ranges)
    plate_uplift: float = 0.15
    #: Mid-ocean ridge height and rift valley depth where plates open, metres
    ridge_height: float = 1200.0
    rift_depth: float = 500.0
    #: Size of the regions of plains or rough country (scaled)
    character_wavelength: float = 250e3
    #: Size of the stretches of smooth or rugged coast (scaled)
    coast_wavelength: float = 200e3
    #: Desert landforms on arid land (:mod:`landforms`); each costs only
    #: where it is. Sand seas (ergs) of dunes on every arid plain, at any
    #: height: wherever the ground is otherwise flat (no range, no mesas, no
    #: canyons)
    dunes: bool = True
    #: Dune size, times Sandastray's (layers 42 m to 1.4 km), and height
    #: scale, metres (Sandastray's 200: 100-150 m from trough to crest)
    dune_scale: float = 1.0
    dune_height: float = 200.0
    #: Mesas and terraced slopes on arid uplands
    mesas: bool = True
    mesa_wavelength: float = 8e3
    mesa_height: float = 300.0
    terrace_step: float = 80.0
    #: Canyons cut into arid uplands
    canyons: bool = True
    canyon_wavelength: float = 40e3
    canyon_depth: float = 900.0
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

    :ivar name: preset name, for display
    :ivar seed: noise seed; each seed is a different planet
    :ivar radius: sea-level radius, metres; the planetary-scale features
        follow it (see the module docstring)
    :ivar reference_lat: latitude of the origin of chart-metre coordinates in
        level files (stamp centres, camera poses), degrees
    :ivar reference_lon: its longitude, degrees
    :ivar has_ocean: whether sea level holds water: beaches, a drawn sea
        surface, and the sea counted as solid by queries. Without one the
        lowlands are dry basins
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

    def evaluator(self) -> Planet:
        """:returns: what samples it (the terrain's body protocol)"""
        return Planet(self)


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
            plates=0,
            dunes=False,
            mesas=False,
            canyons=False,
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
        plate_warp_wavelength=e.plate_warp_wavelength * s,
        plate_warp_amplitude=e.plate_warp_amplitude * s,
        belt_width=e.belt_width * s,
        character_wavelength=e.character_wavelength * s,
        coast_wavelength=e.coast_wavelength * s,
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
    :ivar weights: (rows, columns, 6) biome weights summing to 1, for a body
        that sets them itself; None = from the climate (biomes.weights)
    """

    dirs: np.ndarray
    elevation: np.ndarray
    temperature: np.ndarray
    humidity: np.ndarray
    weights: np.ndarray | None = None


#: The heights a planet's ground can take, metres above sea level: before a
#: chunk is generated, its ground is assumed anywhere in them.
PLANET_HEIGHT_BOUNDS = (-7000.0, 9000.0)


class Planet:
    """Evaluates a :class:`PlanetSpec`. Cheap to build once the spec is resolved.

    The terrain's **body protocol** (what the chunk pipeline, the LOD pass,
    the queries and the collision ask of a body; ``asteroids.rock.RockBody``
    is another):

    - ``spec``: its spec, with ``radius``, ``has_ocean``, ``has_vegetation``,
      ``paints`` and ``evaluator()``;
    - ``grid``: its :class:`cubesphere.CubeGrid`;
    - ``height_bounds``: (lowest, highest) ground, metres from the radius;
    - ``sample_face_grid(face, ka, kb, step)`` and
      ``sample_dirs(dirs, min_wavelength, apply_stamps)``.
    """

    #: (lowest, highest) ground, metres above sea level
    height_bounds = PLANET_HEIGHT_BOUNDS

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
        self._continent_curve = Pchip(points[:, 0], points[:, 1])
        self._curve_range = (points[0, 0], points[-1, 0])
        profile = np.array(MOUNTAIN_PROFILE)
        self._mountain_profile = Pchip(profile[:, 0], profile[:, 1])
        self._stamp_bases: dict[int, float] = {}

    # ── low band ───────────────────────────────────────────────────────────────

    def low_band(self, dirs: np.ndarray) -> np.ndarray:
        """Fields with no detail below ~10 km, safe to interpolate.

        :param dirs: (N, 3) unit vectors
        :returns: (N, :data:`LOW_CHANNELS`), columns ``LOW_*``: warp
            displacement xyz (metres), continental field (sea bias removed),
            range mask in [0, 1], base temperature, base humidity, character
            (0 plains, 1 rough country), coast style (0 smooth, 1 rugged) and
            the plates' ridges and rifts (metres), arid land, sand seas and
            canyon regions (all three in [0, 1])
        """
        spec = self.spec
        e, c = self.elevation, self.climate
        seed = spec.seed * 1000
        p, q, continent, plates = self._continental(dirs)
        continent = continent - spec.sea_bias
        mask = fbm(q, e.mountain_mask_wavelength, 3, seed + _SEED_MASK)
        character = smoothstep(
            -0.3, 0.3, fbm(q, e.character_wavelength, 2, seed + _SEED_CHARACTER)
        )
        coast = smoothstep(
            -0.25, 0.25, fbm(q, e.coast_wavelength, 2, seed + _SEED_COAST)
        )
        cover = e.mountain_coverage
        # Massifs: a range's summit level rises and falls along it, so it is
        # a row of mountain groups with lower passes between.
        massif = ridged(q, 2.5 * e.mountain_wavelength, 2, seed + _SEED_MASSIF)
        if plates is None:
            ranges = smoothstep(-0.05 - cover, 0.3 - cover, mask)
            tectonic = np.zeros(len(dirs))
        else:
            # Belts along closing plate edges, broken where the mask is low;
            # the mask alone still raises a few ranges inside the plates.
            belts = plates.convergent * smoothstep(-0.45, -0.2, mask + cover)
            ranges = np.maximum(belts, smoothstep(0.25 - cover, 0.5 - cover, mask))
            sea = 1.0 - smoothstep(-0.1, 0.0, continent)
            tectonic = plates.divergent * (
                e.ridge_height * sea - e.rift_depth * (1.0 - sea)
            )
            # Plates sliding past each other leave long, straight coasts.
            coast = coast * (1.0 - plates.transform)
        ranges = ranges * (0.45 + 0.55 * massif)
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
        # Arid land, from the climate at sea level (the altitude's cooling
        # would make the shape depend on itself), and its sand seas: on every
        # plain, away from ranges (and from canyons: see _erg).
        arid = aridity(temperature, humidity) * smoothstep(0.0, 0.08, continent)
        erg = arid * smoothstep(0.3, 0.1, character) * smoothstep(0.05, 0.0, ranges)
        # Where canyons may cut (into arid uplands): some regions only.
        canyon = smoothstep(
            0.0,
            0.2,
            fbm(q, 8.0 * e.canyon_wavelength, 2, seed + _SEED_CANYONS + 1),
        )
        return np.column_stack(
            [
                q - p,
                continent,
                ranges,
                temperature,
                humidity,
                character,
                coast,
                tectonic,
                arid,
                erg,
                canyon,
            ]
        )

    def _continental(
        self, dirs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, PlateFields | None]:
        """:returns: the sample points p (metres), the warped points q, the
        continental field at q (sea bias not removed, plate crust included)
        and the plates' fields (None without plates)"""
        e = self.elevation
        seed = self.spec.seed * 1000
        p = dirs * self.spec.radius
        q = warp(p, e.warp_wavelength, e.warp_amplitude, 3, seed + _SEED_WARP)
        continent = fbm(
            q, e.continent_wavelength, e.continent_octaves, seed + _SEED_CONTINENT
        )
        if e.plates <= 0:
            return p, q, continent, None
        x = warp(
            q, e.plate_warp_wavelength, e.plate_warp_amplitude, 2, seed + _SEED_PLATES
        )
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        plates = plate_fields(
            make_plates(seed + _SEED_PLATES, e.plates),
            x,
            self.spec.radius,
            e.belt_width,
        )
        continent = (
            continent
            + e.plate_crust * plates.crust
            + e.plate_uplift * plates.convergent
        )
        return p, q, continent, plates

    # ── full evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self, dirs: np.ndarray, low: np.ndarray, min_wavelength: float | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Elevation and climate from the low band plus per-sample detail.

        Stamps are not applied here; see :meth:`sample_face_grid`.

        :param dirs: (N, 3) unit vectors
        :param low: (N, :data:`LOW_CHANNELS`) low band at the same points
            (:meth:`low_band`)
        :param min_wavelength: shortest wavelength to keep, metres; None = all
        :returns: elevation (metres), base temperature, humidity; each (N,)
        """
        spec = self.spec
        e = self.elevation
        seed = spec.seed * 1000
        p = dirs * spec.radius
        q = p + low[:, LOW_WARP]
        continent = low[:, LOW_CONTINENT]
        base = self._base(low)
        land = smoothstep(0.0, 0.08, continent)
        range_mask = low[:, LOW_RANGE]

        # Relief of hills and detail: low on plains, high in rough country
        # and ranges; near the coast the coast style takes over, so a smooth
        # coast follows the continental field and a rugged one keeps islets.
        rough = np.maximum(low[:, LOW_CHARACTER], range_mask * land)
        shore = 1.0 - smoothstep(0.0, COAST_ZONE, np.abs(continent))
        coast = low[:, LOW_COAST]
        relief = (0.12 + 1.08 * rough) * (1.0 - shore) + (0.04 + 1.16 * coast) * shore
        fine = (0.3 + 0.7 * rough) * (1.0 - shore) + (0.1 + 0.9 * coast) * shore

        # Under the sand seas the ground is otherwise flat.
        erg = self._erg(low, base)

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
                e.mountain_height
                * self._mountain_profile(np.clip(crest, 0.0, 1.0))
                * (range_mask * land)[active]
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
            * relief
            * (1.0 - 0.85 * erg)
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
            * fine
            * (1.0 + 2.0 * range_mask * land)
            * (1.0 - 0.8 * erg)
        )
        elevation = self._desert(
            base + mountains + hills, q, base, erg, low, min_wavelength
        )
        elevation = elevation + detail

        # Local climate patches: a few km, so low flight crosses biome edges.
        jitter_t = fbm(p, 5e3, 1, seed + _SEED_JITTER)
        jitter_h = fbm(p, 7e3, 2, seed + _SEED_JITTER + 1)
        temperature = low[:, LOW_TEMPERATURE] + 3.0 * jitter_t
        humidity = np.clip(low[:, LOW_HUMIDITY] + 0.12 * jitter_h, 0.0, 1.0)
        return elevation, temperature, humidity

    def _desert(
        self,
        elevation: np.ndarray,
        q: np.ndarray,
        base: np.ndarray,
        erg: np.ndarray,
        low: np.ndarray,
        min_wavelength: float | None,
    ) -> np.ndarray:
        """Dunes on sand seas (*erg*, in [0, 1]); mesas, canyons and terraces
        on arid uplands. Each is evaluated only where it is, and only if the
        spec has it on.

        :returns: *elevation* with them (a new array)
        """
        e = self.elevation
        seed = self.spec.seed * 1000
        elevation = elevation.copy()
        on = erg > 1e-3
        if e.dunes and np.any(on):
            elevation[on] += erg[on] * landforms.dunes(
                q[on],
                e.dune_scale,
                e.dune_height,
                seed + _SEED_DUNES,
                min_wavelength,
            )
        upland = self._upland(low, base)
        on = upland > 1e-3
        if not (e.mesas or e.canyons) or not np.any(on):
            return elevation
        z, weight, qs = elevation[on], upland[on], q[on]
        # Where cliffs stand, they are terraced (flat ground stays flat).
        cliffs = np.zeros(len(z))
        if e.mesas:
            # Tables stand in rough upland; flat upland is tableland already
            # (or sand sea).
            mesa = weight * self._rough(low[on])
            tops = landforms.mesa_tops(
                qs, e.mesa_wavelength, seed + _SEED_MESAS, min_wavelength
            )
            z = z + mesa * e.mesa_height * tops
            cliffs = np.maximum(cliffs, mesa * smoothstep(0.0, 0.2, tops))
        if e.canyons:
            # Deeper into higher plateaus, never down to the sea.
            canyon = weight * low[on, LOW_CANYON]
            cut = landforms.canyons(
                qs, e.canyon_wavelength, seed + _SEED_CANYONS, min_wavelength
            )
            depth = e.canyon_depth * smoothstep(200.0, 1500.0, base[on])
            z = z - canyon * depth * cut
            cliffs = np.maximum(cliffs, canyon * smoothstep(0.0, 0.2, cut))
        fade = landforms.terrace_fade(e.terrace_step, min_wavelength)
        if e.mesas and fade > 0.0:
            z = z + fade * cliffs * (landforms.terrace(z, e.terrace_step) - z)
        elevation[on] = z
        return elevation

    def _base(self, low: np.ndarray) -> np.ndarray:
        """:returns: the continental base height and the plates' ridges and
        rifts, metres: the ground before mountains, hills and detail"""
        lo, hi = self._curve_range
        curve = self._continent_curve(np.clip(low[:, LOW_CONTINENT], lo, hi))
        return curve + low[:, LOW_TECTONIC]

    @classmethod
    def _erg(cls, low: np.ndarray, base: np.ndarray) -> np.ndarray:
        """:returns: the sand seas, in [0, 1]: arid plains at any height,
        but never where canyons cut (mesas stand on rougher ground)"""
        return low[:, LOW_ERG] * (1.0 - cls._upland(low, base) * low[:, LOW_CANYON])

    @staticmethod
    def _upland(low: np.ndarray, base: np.ndarray) -> np.ndarray:
        """:returns: arid upland, in [0, 1], where mesas and canyons may be"""
        return low[:, LOW_ARID] * smoothstep(300.0, 500.0, base)

    @staticmethod
    def _rough(low: np.ndarray) -> np.ndarray:
        """:returns: country rough enough for mesas, in [0, 1]: from where
        the sand seas' plains end"""
        return smoothstep(0.3, 0.5, low[:, LOW_CHARACTER])

    def landform_masks(self, dirs: np.ndarray) -> dict[str, np.ndarray]:
        """Where the desert landforms are, whether or not the spec has them on.

        :param dirs: (N, 3) unit vectors
        :returns: {"dunes", "mesas", "canyons"}: (N,) weights in [0, 1]
        """
        low = self.low_band(dirs)
        base = self._base(low)
        upland = self._upland(low, base)
        return {
            "dunes": self._erg(low, base),
            "mesas": upland * self._rough(low),
            "canyons": upland * low[:, LOW_CANYON],
        }

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
        dirs = face_grid(face, grid.param(ka), grid.param(kb))
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

    def _apply_stamps(
        self, dirs: np.ndarray, elevation: np.ndarray, spacing: float
    ) -> np.ndarray:
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
        dirs = face_grid(face, self.grid.param(la), self.grid.param(lb))
        return self.low_band(dirs).reshape(len(lb), len(la), -1)


class Pchip:
    """Monotone piecewise-cubic interpolation (PCHIP), as scipy's
    ``PchipInterpolator`` (the same slopes: Fritsch-Butland, with its
    one-sided end slopes) for inputs within the points' range. Here so that
    every worker process does not import ``scipy.interpolate`` (~0.1 s).

    :param x: (n,) increasing abscissae, n >= 2
    :param y: (n,) values
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        h = np.diff(self.x)
        m = np.diff(self.y) / h
        d = np.zeros_like(self.y)
        if len(m) == 1:
            d[:] = m[0]
        else:
            # Interior: weighted harmonic mean of the neighbouring secants,
            # zero at a local extremum (keeps the curve monotone).
            flat = (np.sign(m[1:]) != np.sign(m[:-1])) | (m[1:] == 0) | (m[:-1] == 0)
            w1, w2 = 2.0 * h[1:] + h[:-1], h[1:] + 2.0 * h[:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = (w1 / m[:-1] + w2 / m[1:]) / (w1 + w2)
                d[1:-1] = np.where(flat, 0.0, 1.0 / mean)
            d[0] = _pchip_end(h[0], h[1], m[0], m[1])
            d[-1] = _pchip_end(h[-1], h[-2], m[-1], m[-2])
        self.slopes = d

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """:returns: the curve at *x* (within the points' range)"""
        x = np.asarray(x, dtype=np.float64)
        k = np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, len(self.x) - 2)
        h = self.x[k + 1] - self.x[k]
        t = (x - self.x[k]) / h
        t2, t3 = t * t, t * t * t
        return (
            (2.0 * t3 - 3.0 * t2 + 1.0) * self.y[k]
            + (t3 - 2.0 * t2 + t) * h * self.slopes[k]
            + (3.0 * t2 - 2.0 * t3) * self.y[k + 1]
            + (t3 - t2) * h * self.slopes[k + 1]
        )


def _pchip_end(h0: float, h1: float, m0: float, m1: float) -> float:
    """One-sided three-point slope at an end, kept monotone (scipy's)."""
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3.0 * abs(m0):
        return 3.0 * m0
    return float(d)


def face_grid(face: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """:returns: ``face_dirs`` over the grid a x b, rows along b: in the
    native library when it is built (one tangent per row and column)"""
    native = native_library()
    if native is not None:
        return native.face_grid(face, a, b)
    ga, gb = np.meshgrid(a, b)
    return face_dirs(face, ga, gb)


def _catmull_rom_weights(
    coords: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
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
    unbiased.spec = spec
    unbiased.elevation, unbiased.climate = scaled_params(spec)
    # The continental field alone (the low band's other fields cost as much)
    _, _, continent, _ = unbiased._continental(dirs)
    return float(np.quantile(continent, spec.elevation.ocean_fraction))
