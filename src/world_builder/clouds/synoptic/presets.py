"""
Local weather presets: a situation pinned at a coordinate, the global
weather arranged around it.

A preset is a *role* relative to a weather system ("on the cold front of a
mature low"), not a cloud stamp: :func:`solve` builds the system so that
the pin sits at its role point at the weather's time (its track worked
back from there), refits the jets so a pinned low lies downstream of a
trough and a pinned high under a ridge, and lets the free systems form
round it (:func:`systems.population` keeps the pinned ones first). A pin
the planet cannot hold raises :class:`ValueError` with the reason; it never
moves.

docs/literature/synoptic-weather-review.tex, section "Local presets".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from world_builder.clouds.synoptic.fields import frame
from world_builder.clouds.synoptic.fronts import fronts_in_radii
from world_builder.clouds.synoptic.jets import JetWave, jet_waves, meander_amplitude
from world_builder.clouds.synoptic.rules import instability
from world_builder.clouds.synoptic.scales import Scales
from world_builder.clouds.synoptic.systems import (
    HIGH_STRENGTH,
    LOW_POLEWARD,
    LOW_STRENGTH,
    PRESSURE_SIGN,
    TC_POLEWARD,
    TC_RECURVE,
    TC_WEST,
    TROPICAL_STRENGTH,
    Kind,
    SynopticObject,
    _deg,
    _wave_speed,
    low_lifetime,
    position,
)
from world_builder.terrain.chart import dirs_from_latlon, latlon


class Preset(Enum):
    FAIR_HIGH = "fair_high"
    WARM_FRONT_APPROACH = "warm_front_approach"
    COLD_FRONT = "cold_front"
    THUNDERSTORMS = "thunderstorms"
    SQUALL_LINE = "squall_line"
    HURRICANE = "hurricane"
    OVERCAST_DRIZZLE = "overcast_drizzle"


@dataclass(frozen=True)
class WeatherPin:
    """A preset pinned at (*lat*, *lon*) degrees, at the weather's time.

    :ivar heading_deg: which way the pinned system faces (its long axis,
        degrees from east toward north); None for east, the westerlies'
    """

    lat: float
    lon: float
    preset: Preset
    heading_deg: float | None = None


def parse_pin(lat: str, lon: str, name: str) -> WeatherPin:
    """A pin from the command line's ``LAT LON PRESET``.

    :raises ValueError: an unknown preset (naming the known ones) or a
        number that is not one
    """
    key = name.strip().lower()
    names = [p.value for p in Preset]
    if key not in names:
        raise ValueError(f"unknown preset {name!r}: one of {', '.join(names)}")
    return WeatherPin(float(lat), float(lon), Preset(key))


#: The genus that dominates overhead at the pin ("clear": none above 0.2).
PRESET_GENUS = {
    Preset.FAIR_HIGH: "clear",
    Preset.WARM_FRONT_APPROACH: "altostratus",
    Preset.COLD_FRONT: "cumulonimbus",
    Preset.THUNDERSTORMS: "cumulonimbus",
    Preset.SQUALL_LINE: "cumulonimbus",
    Preset.HURRICANE: "clear",  # its eye
    Preset.OVERCAST_DRIZZLE: "stratus",
}
#: The low-based presets, and the stage their low is pinned at.
LOW_PRESETS = (
    Preset.WARM_FRONT_APPROACH,
    Preset.COLD_FRONT,
    Preset.SQUALL_LINE,
    Preset.OVERCAST_DRIZZLE,
)
STAGE = {Preset.FAIR_HIGH: 0.5, Preset.HURRICANE: 0.5, **{p: 0.4 for p in LOW_PRESETS}}
#: A thunderstorm cluster's radius, in cyclone radii.
THUNDERSTORM_RADIUS = 0.3
#: Pins of the same kind of system closer than this many of its radii
#: share it.
SHARE = 0.6


def dominant_genus(fields, i: int) -> str:
    """The genus with the most cover at index *i* of :class:`rules.GenusFields`
    (the convective continuum as ``cumulonimbus`` where it is deep,
    ``cumulus`` otherwise); ``"clear"`` with none above 0.2."""
    covers = {g: float(c[i]) for g, c in fields.layers.items()}
    convective = "cumulonimbus" if fields.depth[i] > 0.8 else "cumulus"
    covers[convective] = float(fields.convective[i])
    best = max(covers, key=covers.get)
    return best if covers[best] > 0.2 else "clear"


def _along(polyline: np.ndarray, share: float) -> np.ndarray:
    """The point *share* of the way along a polyline, by arc length."""
    steps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    at = share * steps[-1]
    return np.array(
        [np.interp(at, steps, polyline[:, 0]), np.interp(at, steps, polyline[:, 1])]
    )


def _cold_normal(polyline: np.ndarray, share: float) -> np.ndarray:
    """The unit normal toward a front's cold side (its left), *share* along."""
    steps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    k = int(np.clip(np.searchsorted(steps, share * steps[-1]), 1, len(polyline) - 1))
    tangent = polyline[k] - polyline[k - 1]
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    return np.array([-tangent[1], tangent[0]])


def role_point(preset: Preset, obj: SynopticObject, t: float) -> np.ndarray:
    """Where the pin sits relative to its system at *t*, in the system's
    radii (along its heading, poleward)."""
    if preset in (Preset.FAIR_HIGH, Preset.HURRICANE):
        return np.zeros(2)
    f = fronts_in_radii(obj, t)
    if preset is Preset.WARM_FRONT_APPROACH:
        return _along(f.warm, 0.5) + 0.3 * _cold_normal(f.warm, 0.5)  # in the As
    if preset is Preset.OVERCAST_DRIZZLE:
        return _along(f.warm, 0.3) - 0.2 * _cold_normal(f.warm, 0.3)  # warm sector
    # The cold front runs from its tip back to the triple point: 0.3 of the
    # way out from the triple point is 0.7 along it.
    on_front = _along(f.cold, 0.7)
    if preset is Preset.COLD_FRONT:
        return on_front
    return on_front - 0.17 * _cold_normal(f.cold, 0.7)  # SQUALL_LINE: ahead of it


def _offset(centre: np.ndarray, metres_east: float, metres_north: float, radius: float):
    """The exponential map: *centre* moved by a tangent offset, metres."""
    east, north = frame(centre)
    step = (metres_east * east + metres_north * north) / radius
    angle = float(np.linalg.norm(step))
    if angle < 1e-12:
        return centre
    return math.cos(angle) * centre + math.sin(angle) * step / angle


def _local_of(
    point: np.ndarray, centre: np.ndarray, heading: float, radius: float, r: float
):
    """*point*'s gnomonic (along, poleward) coordinates round *centre*, in
    radii *r* (as :func:`fields.local` and :mod:`rules` read them)."""
    east, north = frame(centre)
    depth = max(float(point @ centre), 1e-6)
    x = radius * float(point @ east) / depth
    y = radius * float(point @ north) / depth
    c, s = math.cos(heading), math.sin(heading)
    lat, _ = latlon(centre[None])
    h = 1.0 if lat[0] >= 0.0 else -1.0
    return np.array([x * c + y * s, h * (-x * s + y * c)]) / r


def _centre_for(pin_dir, role: np.ndarray, heading: float, radius: float, r: float):
    """The centre that puts *pin_dir* at local *role* (radii), solved (a
    system may span a good share of a small planet: no flat offset)."""
    lat, _ = latlon(pin_dir[None])
    h = 1.0 if lat[0] >= 0.0 else -1.0
    c, s = math.cos(heading), math.sin(heading)

    def tangent(local):
        along, pole = local * r
        across = h * pole
        return along * c - across * s, along * s + across * c  # east, north

    centre = _offset(pin_dir, *(-np.array(tangent(role))), radius)
    # Newton steps (a fixed-point one oscillates near a pole, where moving
    # the centre east also turns its frame): the pin's local coordinates
    # against tangent moves of the centre, by finite differences.
    step = 0.01 * r
    for _ in range(12):
        here = _local_of(pin_dir, centre, heading, radius, r)
        miss = role - here
        if np.linalg.norm(miss) < 1e-6:
            break
        jacobian = np.empty((2, 2))
        for k, (de, dn) in enumerate(((step, 0.0), (0.0, step))):
            moved = _local_of(
                pin_dir, _offset(centre, de, dn, radius), heading, radius, r
            )
            jacobian[:, k] = (moved - here) / step
        de, dn = np.linalg.solve(jacobian, miss)
        centre = _offset(centre, de, dn, radius)
    return centre


def _nearest_eddy_jet(jets, lat: float) -> JetWave:
    h = 1 if lat >= 0.0 else -1
    eddy = [j for j in jets if not j.subtropical and j.hemisphere == h]
    return min(eddy, key=lambda j: abs(j.lat0 - abs(lat)))


def _system_for(
    pin: WeatherPin, spec, scales: Scales, jets, index: int
) -> SynopticObject | None:
    """The system *pin* asks for, its track worked back so that its centre
    puts the pin at its role point at the weather's time."""
    if pin.preset is Preset.THUNDERSTORMS:
        return None
    radius = scales.radius
    t = spec.time_s
    s = STAGE[pin.preset]
    heading = math.radians(pin.heading_deg or 0.0)
    pin_dir = dirs_from_latlon(pin.lat, pin.lon)[0]
    seed = (spec.seed * 7919 + index * 104729) & 0x7FFFFFFF
    if pin.preset is Preset.HURRICANE:
        kind, r, strength, life, variant = (
            Kind.TROPICAL,
            scales.tc_radius,
            TROPICAL_STRENGTH,
            1.5 * low_lifetime(scales),
            "",
        )
    elif pin.preset is Preset.FAIR_HIGH:
        kind, r, strength, life, variant = (
            Kind.HIGH,
            1.2 * scales.cyclone_radius,
            HIGH_STRENGTH,
            3.0 * low_lifetime(scales),
            "pinned",
        )
    else:
        variant = (
            "norwegian-squall" if pin.preset is Preset.SQUALL_LINE else "norwegian"
        )
        kind, r, strength, life = (
            Kind.LOW,
            scales.cyclone_radius,
            LOW_STRENGTH,
            low_lifetime(scales),
        )
    birth = t - s * life
    probe = SynopticObject(
        kind, birth, life, pin.lat, pin.lon, r, strength, variant, heading, seed
    )
    role = role_point(pin.preset, probe, t)
    centre = _centre_for(pin_dir, role, heading, radius, r)
    clat, clon = (float(v[0]) for v in latlon(centre[None]))
    tau = s * life
    circle = radius * math.cos(math.radians(clat))
    dlon = dlat = curve = 0.0
    h = 1.0 if clat >= 0.0 else -1.0
    if kind is Kind.LOW:
        dlon = _deg(_wave_speed(_nearest_eddy_jet(jets, clat), clat), circle)
        dlat = h * _deg(LOW_POLEWARD * r / life, radius)
    elif kind is Kind.TROPICAL:
        west = _deg(TC_WEST, circle)
        dlon, dlat, curve = (
            -west,
            h * _deg(TC_POLEWARD, radius),
            west / (TC_RECURVE * life),
        )
    # Pinned highs stand still (a blocking high): the fair spot stays fair.
    return SynopticObject(
        kind=kind,
        birth=birth,
        lifetime=life,
        lat0=clat - dlat * tau,
        lon0=clon - dlon * tau - 0.5 * curve * tau * tau,
        radius=r,
        strength=strength,
        variant=variant,
        heading=heading,
        seed=seed,
        dlon=dlon,
        dlat=dlat,
        curve=curve,
    )


def _check(pin: WeatherPin, spec, scales: Scales, planet) -> None:
    """Raise :class:`ValueError` if the planet cannot hold *pin* there."""
    from world_builder.clouds.weather import _sampler

    lat = abs(pin.lat)
    where = f"{pin.preset.value} at ({pin.lat:.1f}, {pin.lon:.1f})"
    if pin.preset is Preset.HURRICANE:
        if not spec.planet.has_ocean:
            raise ValueError(
                f"{where}: a hurricane needs an ocean; this planet has none"
            )
        lo, hi = scales.tc_band
        if not lo - 2.0 <= lat <= hi + 2.0:
            raise ValueError(
                f"{where}: hurricanes live at latitude {lo:.0f}-{hi:.0f}"
                " degrees on this planet"
            )
    if pin.preset in LOW_PRESETS:
        reach = math.degrees(scales.deformation_radius / scales.radius)
        if not any(b0 - reach <= lat <= b1 + reach for b0, b1 in scales.storm_bands):
            bands = ", ".join(f"{b0:.0f}-{b1:.0f}" for b0, b1 in scales.storm_bands)
            raise ValueError(
                f"{where}: fronts form in the storm track ({bands}"
                " degrees on this planet)"
            )
    if pin.preset in (Preset.HURRICANE, Preset.THUNDERSTORMS):
        sample, _ = _sampler(spec, planet)
        elevation, temperature, humidity = (
            float(v[0]) for v in sample(dirs_from_latlon(pin.lat, pin.lon))
        )
        if pin.preset is Preset.HURRICANE:
            if elevation > -100.0:
                raise ValueError(f"{where}: a hurricane needs open ocean, this is land")
            if temperature < 24.0:
                raise ValueError(
                    f"{where}: the sea is too cold ({temperature:.0f} C)"
                    " for a hurricane"
                )
        elif instability(np.array([humidity]), np.array([temperature]))[0] < 0.3:
            raise ValueError(
                f"{where}: thunderstorms need warm, humid air "
                f"({temperature:.0f} C, humidity {humidity:.2f})"
            )


def _merge_or_raise(objects: list, scales: Scales, t: float) -> list:
    """Pins of the same low share it (a cold front and the squall line
    ahead of it); any other pinned systems crowding each other raise."""
    kept: list = []
    for obj in objects:
        centre = dirs_from_latlon(*position(obj, t))[0]
        clash = None
        for i, other in enumerate(kept):
            gap = scales.radius * math.acos(
                float(
                    np.clip(
                        dirs_from_latlon(*position(other, t))[0] @ centre, -1.0, 1.0
                    )
                )
            )
            if (
                obj.kind is Kind.LOW
                and other.kind is Kind.LOW
                and gap < SHARE * obj.radius
            ):
                clash = i
                break
            same = PRESSURE_SIGN[obj.kind] == PRESSURE_SIGN[other.kind]
            need = (
                0.75 * (obj.radius + other.radius)
                if same
                else 0.75 * min(obj.radius, other.radius)
            )
            if gap < need:
                raise ValueError(
                    f"pins too close: a {obj.kind.value} and a {other.kind.value} "
                    f"{gap / 1e3:.0f} km apart (they need {need / 1e3:.0f} km)"
                )
        if clash is None:
            kept.append(obj)
        elif "squall" in obj.variant and "squall" not in kept[clash].variant:
            kept[clash] = replace(kept[clash], variant=kept[clash].variant + "-squall")
    return kept


def _refit(jets, pinned, scales: Scales, t: float, mu: float = 0.1) -> tuple:
    """The jets' waves refitted so each pinned low lies downstream of a
    trough and each pinned high under a ridge: ridge regression on each
    wave's (sin, cos) amplitudes, pulled toward the seeded ones."""
    total = meander_amplitude(scales)
    out = []
    for jet in jets:
        rows, targets = [], []
        for obj in pinned:
            if obj.kind not in (Kind.LOW, Kind.HIGH):
                continue
            lat, lon = position(obj, t)
            if jet.subtropical or (1 if lat >= 0 else -1) != jet.hemisphere:
                continue
            if _nearest_eddy_jet(jets, lat) is not jet:
                continue
            if obj.kind is Kind.LOW:
                shift = math.degrees(
                    0.25
                    * scales.wavelength
                    / (scales.radius * math.cos(math.radians(lat)))
                )
                at, target = lon - shift, -total
            else:
                at, target = lon, total
            drift = np.fmod(jet.omega * t, 2.0 * math.pi)
            arg = jet.k * math.radians(at) - drift
            rows.append(np.concatenate([np.sin(arg), np.cos(arg)]))
            targets.append(target)
        if not rows:
            out.append(jet)
            continue
        m = np.array(rows)
        x0 = np.concatenate([jet.amp * np.cos(jet.phase), jet.amp * np.sin(jet.phase)])
        lhs = m.T @ m + mu * np.eye(len(x0))
        x = np.linalg.solve(lhs, m.T @ np.array(targets) + mu * x0)
        n = len(jet.k)
        amp = np.hypot(x[:n], x[n:])
        phase = np.arctan2(x[n:], x[:n])
        if amp.sum() > 1.5 * total:
            amp = amp * (1.5 * total / amp.sum())
        out.append(replace(jet, amp=amp, phase=phase))
    return tuple(out)


def solve(
    spec, scales: Scales, planet
) -> tuple[tuple[SynopticObject, ...], tuple[JetWave, ...]]:
    """The pinned systems of :attr:`weather.WeatherSpec.pins` and the jets
    refitted round them.

    :raises ValueError: a pin the planet cannot hold, or pins crowding each
        other, with the reason
    """
    jets = jet_waves(scales, spec.seed)
    for pin in spec.pins:
        _check(pin, spec, scales, planet)
    objects = [
        obj
        for i, pin in enumerate(spec.pins)
        if (obj := _system_for(pin, spec, scales, jets, i)) is not None
    ]
    pinned = _merge_or_raise(objects, scales, spec.time_s)
    return tuple(pinned), _refit(jets, pinned, scales, spec.time_s)


def thunderstorm_boost(spec, scales: Scales, dirs: np.ndarray) -> np.ndarray:
    """The THUNDERSTORMS pins' instability at *dirs*: a cluster of deep
    convection round each, :data:`THUNDERSTORM_RADIUS` cyclone radii wide."""
    boost = np.zeros(len(dirs))
    for pin in spec.pins:
        if pin.preset is not Preset.THUNDERSTORMS:
            continue
        centre = dirs_from_latlon(pin.lat, pin.lon)[0]
        angle = np.arccos(np.clip(dirs @ centre, -1.0, 1.0))
        width = THUNDERSTORM_RADIUS * scales.cyclone_radius / scales.radius
        boost = np.maximum(boost, np.exp(-((angle / width) ** 2)))
    return boost
