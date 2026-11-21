"""
The weather systems: lows, highs, tropical cyclones and cut-off lows, each
an analytic object on a closed-form track with a life cycle.

The population at a time t is a pure function of the spec and t
(stateless renewal): each storm band is cut into sectors, one per wave of
its jet's storm-track wave; each sector has seeded time slots, and a slot
may spawn a system there. Only the slots that can be alive at t are
looked at, so any t costs the same.

docs/literature/synoptic-weather-review.tex, section "Synoptic objects".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from world_builder.clouds.synoptic.jets import (
    JetWave,
    extreme_near,
    jet_lat,
    meander_slope,
)
from world_builder.clouds.synoptic.scales import Scales
from world_builder.terrain.chart import dirs_from_latlon

_SEED_LOW = 601
_SEED_CUTOFF = 607
_SEED_HIGH = 613
_SEED_SUBTROPICAL = 617
_SEED_TROPICAL = 619
#: Slots are hashed with this offset so negative slot indices stay valid.
_SLOT_OFFSET = 1 << 40
#: A permanent system's birth and lifetime, seconds (its stage stays ~0.5).
_FOREVER = 1e15

#: Peak strength of each kind, hPa (tunable).
LOW_STRENGTH = 25.0
HIGH_STRENGTH = 15.0
TROPICAL_STRENGTH = 60.0
CUTOFF_STRENGTH = 15.0
#: Chances a slot spawns a system.
LOW_CHANCE = 1.0
#: How far a low drifts poleward over its life, in its radii.
LOW_POLEWARD = 0.5
#: A low's life, as a share of its sector's slot period: a sector (one
#: trough of its jet's wave) hosts one low at a time, the next born soon
#: after the last dies.
LOW_LIFE_SHARE = (0.9, 1.0)
CUTOFF_CHANCE = 0.3
HIGH_CHANCE = 0.7
TROPICAL_CHANCE = 0.5
BLOCKING_SHARE = 0.1
SHAPIRO_KEYSER_SHARE = 0.3
#: A tropical cyclone's westward and poleward drift, m/s; it recurves (its
#: westward drift spent) at this share of its life.
TC_WEST = 5.0
TC_POLEWARD = 2.5
TC_RECURVE = 0.4
#: Peak of the strength curve, in stage (after the occlusion starts:
#: schultz2011).
PEAK_STAGE = 0.6


class Kind(Enum):
    LOW = "low"
    HIGH = "high"
    TROPICAL = "tropical"
    CUTOFF = "cutoff"


#: +1 for the systems that lower the pressure, -1 for those that raise it.
PRESSURE_SIGN = {Kind.LOW: 1, Kind.CUTOFF: 1, Kind.TROPICAL: 1, Kind.HIGH: -1}


@dataclass(frozen=True)
class SynopticObject:
    """One weather system on a closed-form track.

    ``lat(tau) = lat0 + dlat tau``, ``lon(tau) = lon0 + dlon tau + curve
    tau^2 / 2`` with ``tau = t - birth`` held within its life.

    :ivar kind: what it is
    :ivar birth: when it forms, seconds
    :ivar lifetime: how long it lives, seconds
    :ivar lat0: its latitude at birth, degrees
    :ivar lon0: its longitude at birth, degrees
    :ivar radius: its radius, metres
    :ivar strength: its peak pressure anomaly, hPa (positive)
    :ivar variant: ``"norwegian"`` or ``"sk"`` (Shapiro-Keyser) for a low,
        ``"subtropical"``, ``"migratory"`` or ``"blocking"`` for a high
    :ivar heading: its long axis (the jet's direction at birth), radians
        from east toward north
    :ivar seed: its own seed (shape details)
    :ivar dlon: eastward drift, degrees of longitude per second
    :ivar dlat: northward drift, degrees per second
    :ivar curve: eastward acceleration, degrees of longitude per second^2
        (a tropical cyclone's recurvature)
    """

    kind: Kind
    birth: float
    lifetime: float
    lat0: float
    lon0: float
    radius: float
    strength: float
    variant: str
    heading: float
    seed: int
    dlon: float = 0.0
    dlat: float = 0.0
    curve: float = 0.0


def _smoothstep(e0: float, e1: float, x):
    u = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def stage(obj: SynopticObject, t: float) -> float:
    """:returns: its age as a share of its life, in [0, 1]"""
    return float(np.clip((t - obj.birth) / obj.lifetime, 0.0, 1.0))


def active(obj: SynopticObject, t: float) -> bool:
    """:returns: whether it is alive at *t*"""
    return obj.birth <= t <= obj.birth + obj.lifetime


def intensity(obj: SynopticObject, t: float) -> float:
    """:returns: its strength's share of its peak at *t*: rising to 1 at
    :data:`PEAK_STAGE`, gone at birth and death (0 when not alive)"""
    if not active(obj, t):
        return 0.0
    s = stage(obj, t)
    return float(
        _smoothstep(0.0, PEAK_STAGE, s) * (1.0 - _smoothstep(PEAK_STAGE, 1.0, s))
    )


def position(obj: SynopticObject, t: float) -> tuple[float, float]:
    """:returns: its centre's (latitude, longitude) at *t*, degrees"""
    tau = min(max(t - obj.birth, 0.0), obj.lifetime)
    lat = min(max(obj.lat0 + obj.dlat * tau, -89.0), 89.0)
    lon = obj.lon0 + obj.dlon * tau + 0.5 * obj.curve * tau * tau
    return lat, (lon + 180.0) % 360.0 - 180.0


def low_lifetime(scales: Scales) -> float:
    """:returns: an extratropical low's mean lifetime, seconds"""
    return 2.5 * scales.wavelength / scales.steering


def _rng(seed: int, salt: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed, salt, *[k + _SLOT_OFFSET for k in keys]])


def _slots(t: float, period: float) -> range:
    """The slot indices whose systems may be alive at *t* (lives up to
    three slot periods)."""
    n = math.floor(t / period)
    return range(n - 3, n + 1)


def _deg(metres: float, radius: float) -> float:
    return math.degrees(metres / radius)


def _heading(jet: JetWave, lon: float, t: float, lat: float) -> float:
    slope = float(meander_slope(jet, np.array([lon]), t)[0]) * jet.hemisphere
    return math.atan2(slope, math.cos(math.radians(lat)))


def _wave_speed(jet: JetWave, lat: float) -> float:
    """The jet's storm-track wave's eastward phase speed, m/s at its axis."""
    m = 1
    circle = jet.radius * math.cos(math.radians(jet.lat0))
    return float(jet.omega[m] * circle / jet.k[m])


def storm_sectors(scales: Scales, lat0: float) -> int:
    """The sectors of the storm band around a jet at |latitude| *lat0*:
    its storm-track wavenumber there (the 45 degree one scaled by the
    latitude circle), at least 1."""
    ratio = math.cos(math.radians(lat0)) / math.cos(math.radians(45.0))
    return max(1, round(scales.wavenumber * ratio))


def _band(scales: Scales, lat0: float) -> tuple[float, float]:
    """The storm band around the eddy-driven jet at |latitude| *lat0*."""
    return min(scales.storm_bands, key=lambda b: abs(0.5 * (b[0] + b[1]) - lat0))


def _sector_drift(jet: JetWave, t: float) -> float:
    """How far east the storm band's sectors have moved by *t*, degrees:
    they ride the jet's storm-track wave, as its lows do, so a low stays in
    its sector all its life."""
    rate = float(jet.omega[1] / jet.k[1])  # rad/s of longitude
    return math.degrees(math.fmod(rate * t, 2.0 * math.pi))


def _eddy_jets(jets):
    return [(i, j) for i, j in enumerate(jets) if not j.subtropical]


def _lows(spec, scales, jets, t) -> list[SynopticObject]:
    """The extratropical lows (and cut-off lows) alive at *t*."""
    out = []
    radius = scales.radius
    mean_life = low_lifetime(scales)
    for ji, jet in _eddy_jets(jets):
        sectors = storm_sectors(scales, jet.lat0)
        width = 360.0 / sectors
        speed = _wave_speed(jet, jet.lat0)
        period = mean_life
        for q in range(sectors):
            # The sector's phase: its lows are born at the same point of
            # every slot, so their lives never overlap.
            eta = _rng(spec.seed, _SEED_LOW, ji, q).uniform(0.0, 1.0)
            for n in _slots(t, period):
                rng = _rng(spec.seed, _SEED_LOW, ji, q, n)
                chance, life, size, power, kind = rng.uniform(0.0, 1.0, 5)
                birth = (n + eta) * period
                lo, hi = LOW_LIFE_SHARE
                lifetime = period * (lo + (hi - lo) * life)
                if chance >= LOW_CHANCE or not birth <= t <= birth + lifetime:
                    continue
                centre = -180.0 + (q + 0.5) * width + _sector_drift(jet, birth)
                trough = extreme_near(jet, centre, 0.5 * width, birth)
                # 0.6-1.05 of the scale: 1100-1950 km across on Earth, the
                # observed 1000-2000 km (synoptic weather plan, Task 8; it was
                # 0.75-1.3, 1400-2450 km, 40 % of them over 2000)
                r = scales.cyclone_radius * (0.6 + 0.45 * size)
                axis = abs(float(jet_lat(jet, np.array([trough]), birth)[0]))
                lon0 = trough + _deg(
                    0.25 * scales.wavelength, radius * math.cos(math.radians(axis))
                )
                lat_axis = abs(float(jet_lat(jet, np.array([lon0]), birth)[0]))
                # On the jet's poleward side, but in its storm band even
                # after its poleward drift.
                ceiling = _band(scales, jet.lat0)[1] - _deg(LOW_POLEWARD * r, radius)
                lat0 = jet.hemisphere * min(lat_axis + _deg(0.3 * r, radius), ceiling)
                circle = radius * math.cos(math.radians(lat0))
                out.append(
                    SynopticObject(
                        kind=Kind.LOW,
                        birth=birth,
                        lifetime=lifetime,
                        lat0=lat0,
                        lon0=lon0,
                        radius=r,
                        strength=LOW_STRENGTH * (0.7 + 0.6 * power),
                        variant="sk" if kind < SHAPIRO_KEYSER_SHARE else "norwegian",
                        heading=_heading(jet, lon0, birth, lat0),
                        seed=int(rng.integers(1 << 31)),
                        dlon=_deg(speed, circle),
                        dlat=jet.hemisphere * _deg(LOW_POLEWARD * r / lifetime, radius),
                    )
                )
        # Cut-off lows: rarer, slow, equatorward of a deep trough.
        cut_sectors = max(1, sectors // 4)
        cut_width = 360.0 / cut_sectors
        cut_life = 1.5 * mean_life
        for n in _slots(t, cut_life):
            for q in range(cut_sectors):
                rng = _rng(spec.seed, _SEED_CUTOFF, ji, q, n)
                eta, chance, life, power = rng.uniform(0.0, 1.0, 4)
                birth = (n + eta) * cut_life
                lifetime = cut_life * (0.8 + 0.4 * life)
                if chance >= CUTOFF_CHANCE or not birth <= t <= birth + lifetime:
                    continue
                centre = -180.0 + (q + 0.5) * cut_width
                trough = extreme_near(jet, centre, 0.5 * cut_width, birth)
                r = 0.6 * scales.cyclone_radius
                axis = abs(float(jet_lat(jet, np.array([trough]), birth)[0]))
                lat0 = jet.hemisphere * max(axis - _deg(0.5 * r, radius), 1.0)
                circle = radius * math.cos(math.radians(lat0))
                out.append(
                    SynopticObject(
                        kind=Kind.CUTOFF,
                        birth=birth,
                        lifetime=lifetime,
                        lat0=lat0,
                        lon0=trough,
                        radius=r,
                        strength=CUTOFF_STRENGTH * (0.7 + 0.6 * power),
                        variant="norwegian",
                        heading=0.0,
                        seed=int(rng.integers(1 << 31)),
                        dlon=_deg(0.2 * speed, circle),
                    )
                )
    return out


def _migratory_highs(spec, scales, jets, t) -> list[SynopticObject]:
    """Highs in the ridges between the lows; a few blocking ones."""
    out = []
    radius = scales.radius
    base_life = 1.5 * low_lifetime(scales)
    for ji, jet in _eddy_jets(jets):
        sectors = max(1, storm_sectors(scales, jet.lat0) // 2)
        width = 360.0 / sectors
        speed = _wave_speed(jet, jet.lat0)
        for n in _slots(t, base_life):
            for q in range(sectors):
                rng = _rng(spec.seed, _SEED_HIGH, ji, q, n)
                eta, chance, life, size, power, block = rng.uniform(0.0, 1.0, 6)
                blocking = block < BLOCKING_SHARE
                birth = (n + eta) * base_life
                lifetime = base_life * (0.8 + 0.4 * life) * (3.0 if blocking else 1.0)
                if chance >= HIGH_CHANCE or not birth <= t <= birth + lifetime:
                    continue
                centre = -180.0 + (q + 0.5) * width + _sector_drift(jet, birth)
                ridge = extreme_near(jet, centre, 0.5 * width, birth, trough=False)
                r = scales.cyclone_radius * (0.9 + 0.4 * size)
                axis = abs(float(jet_lat(jet, np.array([ridge]), birth)[0]))
                side = 0.5 if blocking else -0.6
                lat0 = jet.hemisphere * min(
                    max(axis + side * _deg(r, radius), 1.0), 85.0
                )
                circle = radius * math.cos(math.radians(lat0))
                out.append(
                    SynopticObject(
                        kind=Kind.HIGH,
                        birth=birth,
                        lifetime=lifetime,
                        lat0=lat0,
                        lon0=ridge,
                        radius=r,
                        strength=HIGH_STRENGTH * (0.7 + 0.6 * power),
                        variant="blocking" if blocking else "migratory",
                        heading=_heading(jet, ridge, birth, lat0),
                        seed=int(rng.integers(1 << 31)),
                        # Ridges travel with the wave, as the troughs do.
                        dlon=_deg((0.2 if blocking else 1.0) * speed, circle),
                    )
                )
    return out


def _subtropical_highs(spec, scales, planet) -> list[SynopticObject]:
    """The permanent subtropical highs, on the eastern flanks of the ocean
    basins (klein1993, wood2012), evenly spaced on a dry planet."""
    from world_builder.clouds.weather import _sampler

    radius = scales.radius
    count = max(1, round(scales.wavenumber / 3.0))
    r = 1.5 * scales.cyclone_radius
    out = []
    lon = np.linspace(-180.0, 180.0, 360, endpoint=False)
    sample = _sampler(spec, planet)[0] if spec.planet.has_ocean else None
    for hi, h in enumerate((1, -1)):
        lat = h * scales.hadley_edge
        rng = _rng(spec.seed, _SEED_SUBTROPICAL, hi)
        picks: list[float] = []
        if sample is not None:
            elevation, _, _ = sample(dirs_from_latlon(np.full_like(lon, lat), lon))
            sea = elevation < 0.0
            reach = int(min(60.0, _deg(r, radius * math.cos(math.radians(lat)))))
            land_east = np.array(
                [
                    np.mean(~np.roll(sea, -i - 1)[:reach]) if reach else 0.0
                    for i in range(360)
                ]
            )
            score = sea * land_east
            gap = 0.6 * 360.0 / count
            for i in np.argsort(-score):
                if score[i] <= 0.0 or len(picks) == count:
                    break
                if all(abs((lon[i] - p + 180.0) % 360.0 - 180.0) >= gap for p in picks):
                    picks.append(float(lon[i]))
        start = rng.uniform(-180.0, 180.0)
        while len(picks) < count:
            picks.append((start + len(picks) * 360.0 / count + 180.0) % 360.0 - 180.0)
        for i, p in enumerate(picks):
            out.append(
                SynopticObject(
                    kind=Kind.HIGH,
                    birth=-_FOREVER,
                    lifetime=2.0 * _FOREVER,
                    lat0=lat,
                    lon0=p,
                    radius=r,
                    strength=HIGH_STRENGTH,
                    variant="subtropical",
                    heading=0.0,
                    seed=int(
                        _rng(spec.seed, _SEED_SUBTROPICAL, hi, i).integers(1 << 31)
                    ),
                )
            )
    return out


def genesis_potential(lat, temperature, humidity, elevation) -> np.ndarray:
    """A stand-in for camargo2007's genesis potential index: absolute
    vorticity (|sin lat|), humidity, warm sea; 0 over land."""
    sea = np.asarray(elevation) < -200.0
    return (
        np.abs(np.sin(np.radians(lat))) ** 1.5
        * np.asarray(humidity) ** 3
        * _smoothstep(24.0, 28.0, temperature)
        * sea
    )


def _tropical(spec, scales, planet, t) -> list[SynopticObject]:
    """The tropical cyclones alive at *t*: born over warm sea in the band,
    drifting west and poleward, recurving, dying over land or past the
    Hadley cell's edge."""
    if not spec.planet.has_ocean:
        return []
    from world_builder.clouds.weather import _sampler

    sample = _sampler(spec, planet)[0]
    radius = scales.radius
    mean_life = 1.5 * low_lifetime(scales)
    sectors = max(2, round(scales.wavenumber / 2.0))
    width = 360.0 / sectors
    lo, hi_lat = scales.tc_band
    out = []
    for hi, h in enumerate((1, -1)):
        for n in _slots(t, mean_life):
            for q in range(sectors):
                rng = _rng(spec.seed, _SEED_TROPICAL, hi, q, n)
                eta, chance, life, power = rng.uniform(0.0, 1.0, 4)
                birth = (n + eta) * mean_life
                lifetime = mean_life * (0.8 + 0.4 * life)
                if chance >= TROPICAL_CHANCE or not birth <= t <= birth + lifetime:
                    continue
                lats = h * rng.uniform(lo, hi_lat, 12)
                lons = -180.0 + (q + rng.uniform(0.0, 1.0, 12)) * width
                elevation, temperature, humidity = sample(dirs_from_latlon(lats, lons))
                gp = genesis_potential(lats, temperature, humidity, elevation)
                best = int(np.argmax(gp))
                if gp[best] <= 0.0:
                    continue
                lat0, lon0 = float(lats[best]), float(lons[best])
                circle = radius * math.cos(math.radians(lat0))
                west = _deg(TC_WEST, circle)
                candidate = SynopticObject(
                    kind=Kind.TROPICAL,
                    birth=birth,
                    lifetime=lifetime,
                    lat0=lat0,
                    lon0=lon0,
                    radius=scales.tc_radius,
                    strength=TROPICAL_STRENGTH * (0.7 + 0.6 * power),
                    variant="",
                    heading=0.0,
                    seed=int(rng.integers(1 << 31)),
                    dlon=-west,
                    dlat=h * _deg(TC_POLEWARD, radius),
                    curve=west / (TC_RECURVE * lifetime),
                )
                candidate = _decay(candidate, sample, scales)
                if active(candidate, t):
                    out.append(candidate)
    return out


def _decay(obj: SynopticObject, sample, scales: Scales) -> SynopticObject:
    """Shorten a tropical cyclone's life to end soon after it makes landfall
    or leaves the Hadley cell (it decays rather than turning extratropical)."""
    taus = np.linspace(0.0, obj.lifetime, 33)
    track = [position(obj, obj.birth + tau) for tau in taus]
    lats = np.array([p[0] for p in track])
    lons = np.array([p[1] for p in track])
    elevation, _, _ = sample(dirs_from_latlon(lats, lons))
    out = (elevation > 0.0) | (np.abs(lats) > scales.hadley_edge)
    if not out.any():
        return obj
    exit_tau = taus[int(np.argmax(out))]
    lifetime = min(
        obj.lifetime, max(exit_tau + 0.25 * obj.lifetime, 0.3 * obj.lifetime)
    )
    return SynopticObject(**{**obj.__dict__, "lifetime": lifetime})


def spacing_radius(obj: SynopticObject, t: float) -> float:
    """The room a system needs at *t*, metres: its radius, half of it at
    birth and death (a filling low takes less room than a deep one)."""
    return obj.radius * (0.5 + 0.5 * intensity(obj, t))


class _Kept:
    """The systems kept so far, in arrays grown as needed (the spacing check
    runs once per candidate: thousands on a large planet)."""

    def __init__(self, capacity: int = 64):
        self.n = 0
        self.dirs = np.empty((capacity, 3))
        self.radii = np.empty(capacity)
        self.signs = np.empty(capacity, dtype=np.int8)
        self.strict = np.empty(capacity, dtype=bool)

    def add(self, centre, radius: float, sign: int, strict: bool) -> None:
        if self.n == len(self.radii):
            grow = 2 * len(self.radii)
            self.dirs = np.resize(self.dirs, (grow, 3))
            self.radii = np.resize(self.radii, grow)
            self.signs = np.resize(self.signs, grow)
            self.strict = np.resize(self.strict, grow)
        self.dirs[self.n] = centre
        self.radii[self.n] = radius
        self.signs[self.n] = sign
        self.strict[self.n] = strict
        self.n += 1

    def crowds(self, centre, radius: float, sign: int, planet_radius: float) -> bool:
        """Same-sign systems keep 0.75 (r_a + r_b) apart; a low may sit
        beside a high (the gradient between them is the wind), but not on
        its core: 0.75 min(r_a, r_b). A pinned system (strict) keeps every
        other at the full 0.75 (r_a + r_b): what its pin asks for is what
        is seen."""
        n = self.n
        if n == 0:
            return False
        radii = self.radii[:n]
        gc = planet_radius * np.arccos(np.clip(self.dirs[:n] @ centre, -1.0, 1.0))
        need = np.where(
            (self.signs[:n] == sign) | self.strict[:n],
            0.75 * (radii + radius),
            0.75 * np.minimum(radii, radius),
        )
        return bool(np.any(gc < need))


def population(spec, scales: Scales, jets, planet, t: float, pinned=()) -> list:
    """The weather systems alive at *t*: the pinned ones first, then the
    seeded ones in order of birth, each dropped if it crowds one kept
    already (:meth:`_Kept.crowds`).

    :param spec: the :class:`weather.WeatherSpec` (its seed and planet)
    :param jets: :func:`jets.jet_waves`
    :param planet: the evaluated :class:`terrain.planet.Planet`
    :param pinned: systems placed by pins (:mod:`presets`), kept first
    :returns: list of :class:`SynopticObject`
    """
    seeded = (
        _subtropical_highs(spec, scales, planet)
        + _lows(spec, scales, jets, t)
        + _migratory_highs(spec, scales, jets, t)
        + _tropical(spec, scales, planet, t)
    )
    seeded.sort(key=lambda o: (o.birth, o.seed))
    kept, room_taken = [], _Kept()
    for i, obj in enumerate(list(pinned) + seeded):
        is_pinned = i < len(pinned)
        if not active(obj, t):
            continue
        lat, lon = position(obj, t)
        centre = dirs_from_latlon(lat, lon)[0]
        sign = PRESSURE_SIGN[obj.kind]
        room = spacing_radius(obj, t)
        # Pinned systems are always kept (:mod:`presets` checked them).
        if not is_pinned and room_taken.crowds(centre, room, sign, scales.radius):
            continue
        kept.append(obj)
        room_taken.add(centre, room, sign, is_pinned)
    return kept
