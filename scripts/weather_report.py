"""
Is the Earth-scale planet's weather credible? Its synoptic weather
(clouds.synoptic) against observed climatology, over several weather seeds
and times (synoptic weather plan, Task 8)::

    python scripts/weather_report.py [--planet earth] [--seeds 5] [--times 5]
        [--day-step 1.0]

Prints each quantity's spread over the samples (min, median, max) beside
its band, and exits 1 if one fails. The bands (the literature review,
docs/literature/synoptic-weather-review.tex, has their sources):

* the subtropical jet at 25-35 degrees, the eddy-driven one at 40-60;
* the Hadley cell's edge at 27-33 degrees;
* 5-10 active extratropical lows per hemisphere, 1000-2000 km across;
* the subtropical highs centred on the eastern half of their ocean basin
  (at least 60 % of them);
* 0-5 tropical cyclones per hemisphere, formed at 5-25 degrees over sea
  (they recurve poleward and make landfall later, as the real ones do);
* the warm-front cloud sequence, cirrus to nimbostratus toward the front;
* the circulation: cyclonic around the lows (counter-clockwise in the north,
  clockwise in the south), anticyclonic around the highs (each system's own
  wind, the background's subtracted).

How a band is judged: the planet's scales (the jets, the Hadley edge) in
every sample; a count (lows, cyclones) by its median, and in at least 80 %
of the samples; a share of all the samples' systems (diameters 90 %,
eastern highs 60 %, cyclones' latitude and sea 100 %, circulation 90 %,
warm-front order 70 %).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np

from world_builder.clouds.synoptic.chart import _to_dirs
from world_builder.clouds.synoptic.fields import wind
from world_builder.clouds.synoptic.fronts import fronts_in_radii
from world_builder.clouds.synoptic.state import synoptic_state
from world_builder.clouds.synoptic.systems import Kind, intensity, position, stage
from world_builder.clouds.weather import (
    SYNOPTIC_CHANNELS,
    WeatherSpec,
    _sampler,
    sample_cube,
    synoptic_cube,
)
from world_builder.terrain.chart import dirs_from_latlon
from world_builder.terrain.planet import PRESETS, Planet

#: The warm front's sequence, far ahead of it to at it.
SHIELD = ("cirrus", "cirrostratus", "altostratus", "nimbostratus")


@dataclass
class Tally:
    """What the samples gave, pooled."""

    subtropical_jet: list = field(default_factory=list)
    eddy_jet: list = field(default_factory=list)
    hadley: list = field(default_factory=list)
    lows: list = field(default_factory=list)  # per sample and hemisphere
    diameters: list = field(default_factory=list)  # km
    highs_east: list = field(default_factory=list)  # bool per subtropical high
    tcs: list = field(default_factory=list)  # per sample and hemisphere
    tc_ok: list = field(default_factory=list)  # bool per cyclone, its genesis
    tc_now: list = field(default_factory=list)  # the same, where it is now
    sense_ok: list = field(default_factory=list)  # bool per system
    front_ok: list = field(default_factory=list)  # bool per mature low


def _basin_east(lat: float, lon: float, sample) -> bool | None:
    """Whether (lat, lon) lies on the eastern half of its ocean basin along
    its parallel (None: over land, or the sea goes all the way round)."""
    lons = (lon + np.arange(-180.0, 180.0, 0.5) + 180.0) % 360.0 - 180.0
    elevation, _, _ = sample(dirs_from_latlon(np.full_like(lons, lat), lons))
    sea = elevation < 0.0
    centre = len(lons) // 2  # lon itself
    if not sea[centre] or sea.all():
        return None
    west = centre
    while west > 0 and sea[west - 1]:
        west -= 1
    east = centre
    while east < len(sea) - 1 and sea[east + 1]:
        east += 1
    width = east - west
    return width > 0 and (centre - west) / width >= 0.5


def _sense(obj, t, scales) -> bool:
    """Whether the surface wind turns the right way round *obj*: cyclonic
    (counter-clockwise in the north) round a low, the other way round a
    high, at half its radius."""
    lat, lon = position(obj, t)
    centre = dirs_from_latlon(lat, lon)[0]
    east = np.cross([0.0, 0.0, 1.0], centre)
    east /= np.linalg.norm(east)
    north = np.cross(centre, east)
    angles = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    offsets = 0.5 * obj.radius / scales.radius
    ring = np.array(
        [centre + offsets * (math.cos(a) * east + math.sin(a) * north) for a in angles]
    )
    ring /= np.linalg.norm(ring, axis=1, keepdims=True)
    # Its own: less the background's (the jets', the trades')
    uv = wind(ring, [obj], scales, t, surface=False) - wind(
        ring, [], scales, t, surface=False
    )
    # The circulation: the radial offset crossed with the wind, (x v - y u)
    curl = np.sum(np.cos(angles) * uv[:, 1] - np.sin(angles) * uv[:, 0])
    counter = curl > 0.0
    cyclonic = obj.kind in (Kind.LOW, Kind.CUTOFF, Kind.TROPICAL)
    north_hemisphere = lat >= 0.0
    return counter == (cyclonic == north_hemisphere)


def _front_order(obj, t, scales, cube) -> bool | None:
    """Whether the cloud ahead of *obj*'s warm front runs cirrus,
    cirrostratus, altostratus, nimbostratus toward it (each strong sample's
    dominant genus never steps back), None where too little cloud."""
    warm = fronts_in_radii(obj, t).warm
    if len(warm) < 3:
        return None
    mid = len(warm) // 2
    tangent = warm[mid + 1] - warm[mid - 1]
    tangent = tangent / np.linalg.norm(tangent)
    cold_side = np.array([-tangent[1], tangent[0]])  # left: ahead of it
    xi = np.linspace(1.2, 0.02, 60)
    dirs = _to_dirs(obj, t, warm[mid] + xi[:, None] * cold_side, scales.radius)
    picks = [SYNOPTIC_CHANNELS.index(g) for g in SHIELD]
    stack = sample_cube(cube[..., picks], dirs)
    strong = stack.max(axis=-1) > 0.1
    if strong.sum() < 10:
        return None
    seq = np.argmax(stack, axis=-1)[strong]
    return bool(np.all(np.diff(seq) >= 0)) and len(set(seq)) >= 3


def survey(planet_name: str, seeds: int, times: int, day_step: float) -> Tally:
    spec = PRESETS[planet_name]
    planet = Planet(spec)
    tally = Tally()
    for seed in range(seeds):
        for k in range(times):
            weather = WeatherSpec(spec, seed=11 + seed, time_s=k * day_step * 86400.0)
            state = synoptic_state(weather)
            scales, t = state.scales, state.t
            sample = _sampler(weather, planet)[0]
            for jet in state.jets:
                (tally.subtropical_jet if jet.subtropical else tally.eddy_jet).append(
                    jet.lat0
                )
            tally.hadley.append(scales.hadley_edge)
            alive = [o for o in state.systems if intensity(o, t) > 0.0]
            for h in (1, -1):
                lows = [
                    o for o in alive if o.kind is Kind.LOW and h * position(o, t)[0] > 0
                ]
                tally.lows.append(len(lows))
                tcs = [
                    o
                    for o in alive
                    if o.kind is Kind.TROPICAL and h * position(o, t)[0] > 0
                ]
                tally.tcs.append(len(tcs))
            for o in alive:
                lat, lon = position(o, t)
                if o.kind is Kind.LOW:
                    tally.diameters.append(2.0 * o.radius / 1000.0)
                if o.kind is Kind.HIGH and o.variant == "subtropical":
                    east = _basin_east(lat, lon, sample)
                    if east is not None:
                        tally.highs_east.append(east)
                if o.kind is Kind.TROPICAL:
                    # Where it formed (later it recurves poleward and may
                    # make landfall, as the real ones do)
                    elevation = sample(dirs_from_latlon(o.lat0, o.lon0))[0][0]
                    tally.tc_ok.append(5.0 <= abs(o.lat0) <= 25.0 and elevation < 0.0)
                    elevation = sample(dirs_from_latlon(lat, lon))[0][0]
                    tally.tc_now.append(5.0 <= abs(lat) <= 25.0 and elevation < 0.0)
                if o.kind in (Kind.LOW, Kind.HIGH, Kind.TROPICAL):
                    tally.sense_ok.append(_sense(o, t, scales))
            if k == 0:
                # The warm fronts' sequence, from the weather's own maps (one
                # bake a seed: ~4 s each)
                cube = synoptic_cube(weather)
                for o in alive:
                    if o.kind is Kind.LOW and 0.3 < stage(o, t) < 0.7:
                        order = _front_order(o, t, scales, cube)
                        if order is not None:
                            tally.front_ok.append(order)
            print(f"  seed {11 + seed}, day {k * day_step:g}: {len(alive)} systems")
    return tally


def _spread(values) -> str:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return "none"
    return f"{v.min():7.1f} {np.median(v):7.1f} {v.max():7.1f}"


def _share(flags) -> float:
    return float(np.mean(flags)) if len(flags) else float("nan")


def report(tally: Tally) -> bool:
    rows = []

    def every(name, values, lo, hi):
        ok = len(values) > 0 and all(lo <= v <= hi for v in values)
        rows.append((name, _spread(values), f"{lo:g}-{hi:g}, every sample", ok))

    def count(name, values, lo, hi):
        v = np.asarray(values)
        inside = np.mean((v >= lo) & (v <= hi)) if v.size else 0.0
        ok = v.size > 0 and lo <= np.median(v) <= hi and inside >= 0.8
        rows.append(
            (name, _spread(values), f"{lo:g}-{hi:g}, median; {inside:.0%} in", ok)
        )

    def share(name, flags, least, spread=""):
        s = _share(flags)
        ok = len(flags) > 0 and s >= least
        rows.append(
            (name, spread or f"{len(flags)} systems", f"{s:.0%} >= {least:.0%}", ok)
        )

    every("subtropical jet (deg)", tally.subtropical_jet, 25.0, 35.0)
    every("eddy-driven jet (deg)", tally.eddy_jet, 40.0, 60.0)
    every("Hadley edge (deg)", tally.hadley, 27.0, 33.0)
    count("lows per hemisphere", tally.lows, 5, 10)
    d = np.asarray(tally.diameters)
    share(
        "low diameter 1000-2000 km",
        list((d >= 1000.0) & (d <= 2000.0)),
        0.9,
        _spread(d),
    )
    share("subtropical highs, east half", tally.highs_east, 0.6)
    count("cyclones per hemisphere", tally.tcs, 0, 5)
    share("cyclones form at 5-25 deg, at sea", tally.tc_ok, 1.0)
    share("circulation sense", tally.sense_ok, 0.9)
    share("warm-front Ci-Cs-As-Ns order", tally.front_ok, 0.7)

    print(f"\n{'quantity':34s} {'min / median / max':>24s}   band")
    for name, spread, band, ok in rows:
        print(f"{name:34s} {spread:>24s}   {band:28s} {'ok' if ok else 'FAIL'}")
    print(
        f"(cyclones now at 5-25 deg and at sea: {_share(tally.tc_now):.0%}; "
        "the others have recurved or made landfall)"
    )
    return all(ok for *_, ok in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--planet", default="earth", help="a terrain.planet preset")
    parser.add_argument("--seeds", type=int, default=5, help="weather seeds")
    parser.add_argument("--times", type=int, default=5, help="times a seed")
    parser.add_argument("--day-step", type=float, default=1.0, help="days apart")
    args = parser.parse_args()
    print(f"{args.planet}: {args.seeds} seeds x {args.times} times")
    tally = survey(args.planet, args.seeds, args.times, args.day_step)
    sys.exit(0 if report(tally) else 1)


if __name__ == "__main__":
    main()
