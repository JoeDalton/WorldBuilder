"""
The weather's scales, from the planet's radius, rotation and stratification.

Every relation is in docs/literature/synoptic-weather-review.tex, section
"Planetary scales" (keys of docs/literature/synoptic-weather.bib in the
comments). Lengths are metres, speeds m/s, latitudes degrees (one hemisphere).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from world_builder.terrain.planet import PlanetSpec

# ── Earth calibration ─────────────────────────────────────────────────────────
#: Earth's radius and sidereal day: the anchors the scalings are calibrated to.
EARTH_RADIUS_M = 6.371e6
EARTH_DAY_H = 23.934
#: Earth's N*H, m/s (holton2013).
EARTH_STRATIFICATION = 100.0
#: Earth's Hadley cell edge, degrees (schneider2006).
EARTH_HADLEY_DEG = 30.0
#: Exponent of the Hadley edge in the thermal Rossby number: 1/4, the
#: eddy-limited edge (proportional to Omega^-1/2; schneider2006, Eq. 3);
#: Held & Hou's 1/2 (held1980) is the upper bound.
HADLEY_EXPONENT = 0.25
HADLEY_CLAMP = (10.0, 60.0)
#: Most unstable Eady wavelength in deformation radii, 2 pi / 1.61
#: (holton2013, vallis2017).
EADY_WAVELENGTHS = 3.9
#: Eddy velocity of the Rhines / Williams jet spacing, m/s (williams1978).
EDDY_SPEED = 20.0
#: Poleward limit of the eddy-driven jets, degrees.
JET_LIMIT_DEG = 75.0
MAX_JETS = 8
#: Widest half-width of a storm band around its eddy-driven jet, degrees.
STORM_BAND_HALF_DEG = 12.5
#: Speed of a tropical cyclone's circulation that sets its size with f(20°).
TC_SPEED = 20.0
#: Jet and steering speeds, m/s (tunable).
JET_SPEED = 25.0
#: The subtropical jet's speed, m/s: the fastest (tunable).
SUBTROPICAL_JET_SPEED = 30.0
STEERING_SHARE = 0.6


def _omega(day_length_h: float) -> float:
    return 2.0 * math.pi / (3600.0 * day_length_h)


def _thermal_rossby(stratification: float, omega: float, radius: float) -> float:
    return (stratification / (omega * radius)) ** 2


_EARTH_RO_T = _thermal_rossby(EARTH_STRATIFICATION, _omega(EARTH_DAY_H), EARTH_RADIUS_M)


@dataclass(frozen=True)
class Scales:
    """A planet's weather scales.

    :ivar radius: the planet's radius, metres
    :ivar omega: rotation rate, rad/s
    :ivar f45: Coriolis parameter at 45 degrees, 1/s
    :ivar beta45: its meridional gradient at 45 degrees, 1/(m s)
    :ivar deformation_radius: N H / f45, metres
    :ivar wavelength: the storm track's Eady wavelength, metres
    :ivar wavenumber: zonal wavenumber of that wavelength at 45 degrees
    :ivar cyclone_radius: radius of an extratropical system, metres
    :ivar tc_radius: radius of a tropical cyclone, metres
    :ivar jet_spacing: the Rhines / Williams jet spacing, metres
    :ivar hadley_edge: the Hadley cell's edge, degrees
    :ivar jet_lats: the jets' latitudes in one hemisphere, degrees: the
        subtropical jet, then the eddy-driven ones poleward
    :ivar storm_bands: one band of extratropical cyclones around each
        eddy-driven jet, degrees: centred on it, half the jet spacing (at
        most :data:`STORM_BAND_HALF_DEG`) either side
    :ivar storm_track: from the first storm band's edge to the last's, degrees
    :ivar tc_band: the tropical cyclones' band, degrees
    :ivar jet_speed: the jets' speed, m/s
    :ivar steering: the lows' speed along the jet, m/s
    """

    radius: float
    omega: float
    f45: float
    beta45: float
    deformation_radius: float
    wavelength: float
    wavenumber: float
    cyclone_radius: float
    tc_radius: float
    jet_spacing: float
    hadley_edge: float
    jet_lats: tuple[float, ...]
    storm_bands: tuple[tuple[float, float], ...]
    storm_track: tuple[float, float]
    tc_band: tuple[float, float]
    jet_speed: float
    steering: float

    @staticmethod
    @functools.cache
    def from_planet(spec: PlanetSpec) -> Scales:
        """:returns: the scales of *spec* (cached)"""
        radius = spec.radius
        omega = _omega(spec.day_length_h)
        s45 = math.sin(math.radians(45.0))
        f45 = 2.0 * omega * s45
        beta45 = 2.0 * omega * s45 / radius
        deformation = spec.stratification / f45
        wavelength = EADY_WAVELENGTHS * deformation
        wavenumber = 2.0 * math.pi * radius * s45 / wavelength
        cyclone = 0.25 * wavelength

        spacing = math.pi * math.sqrt(2.0 * EDDY_SPEED / beta45)  # williams1978
        spacing_deg = math.degrees(spacing / radius)
        ratio = _thermal_rossby(spec.stratification, omega, radius) / _EARTH_RO_T
        hadley = min(
            max(EARTH_HADLEY_DEG * ratio**HADLEY_EXPONENT, HADLEY_CLAMP[0]),
            HADLEY_CLAMP[1],
        )

        extratropics = JET_LIMIT_DEG - hadley
        jets = min(max(round(extratropics / spacing_deg), 1), MAX_JETS)
        jet_lats = (hadley,) + tuple(
            hadley + (i + 0.5) * extratropics / jets for i in range(jets)
        )
        half = min(0.5 * extratropics / jets, STORM_BAND_HALF_DEG)
        storm_bands = tuple((lat - half, lat + half) for lat in jet_lats[1:])
        storm_track = (storm_bands[0][0], storm_bands[-1][1])
        tc_band = (5.0, min(25.0, 0.8 * hadley))

        f20 = 2.0 * omega * math.sin(math.radians(20.0))
        tc = min(max(TC_SPEED / f20, 0.1 * cyclone), 0.5 * cyclone)
        return Scales(
            radius=radius,
            omega=omega,
            f45=f45,
            beta45=beta45,
            deformation_radius=deformation,
            wavelength=wavelength,
            wavenumber=wavenumber,
            cyclone_radius=cyclone,
            tc_radius=tc,
            jet_spacing=spacing,
            hadley_edge=hadley,
            jet_lats=jet_lats,
            storm_bands=storm_bands,
            storm_track=storm_track,
            tc_band=tc_band,
            jet_speed=JET_SPEED,
            steering=STEERING_SHARE * JET_SPEED,
        )
