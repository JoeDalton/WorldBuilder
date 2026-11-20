"""
The cloud genera from the weather: each system's zones (its fronts, warm
sector, comma head, dry slot, cold air; a high's subsidence and its
stratocumulus deck; a tropical cyclone's eye, eyewall and bands) add the
genera the conceptual models put there, over a background of fair-weather
cumulus from the climate.

Coverages add as ``1 - prod(1 - c_i)``. Clearing (a high's subsidence, a
tropical cyclone's eye, a low's dry slot) multiplies what was added before
it; a high's deck, fog and altocumulus are added after.

docs/literature/synoptic-weather-review.tex, section "Genus rules".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from world_builder.clouds.synoptic.fields import DirIndex, local
from world_builder.clouds.synoptic.fronts import (
    FAR,
    front_query,
    fronts_in_radii,
    length,
    overshoot,
)
from world_builder.clouds.synoptic.scales import Scales
from world_builder.clouds.synoptic.systems import (
    Kind,
    intensity,
    position,
    stage,
)
from world_builder.clouds.synoptic.systems import (
    _smoothstep as S,
)
from world_builder.terrain.chart import dirs_from_latlon

#: The genera other than the convective continuum, by their
#: ``genera.GENUS_PRESETS`` names.
LAYERS = (
    "stratus",
    "stratocumulus",
    "nimbostratus",
    "altostratus",
    "altocumulus",
    "cirrus",
    "cirrostratus",
    "cirrocumulus",
)
#: The layers a high's subsidence clears (not the ice cloud above it).
_SUBSIDENCE = (
    "convective",
    "stratus",
    "stratocumulus",
    "nimbostratus",
    "altostratus",
    "altocumulus",
)
#: The warm-front shield, by distance ahead of the front in radii:
#: (genus, start, end, coverage) (carlson1980).
WARM_SHIELD = (
    ("nimbostratus", 0.00, 0.20, 0.95),
    ("altostratus", 0.15, 0.50, 0.85),
    ("cirrostratus", 0.45, 0.85, 0.70),
    ("cirrus", 0.80, 1.20, 0.55),
)
#: How far the warm-front shield's bands fade into each other, in radii.
BAND_EDGE = 0.15
#: A tropical cyclone's spiral bands' pitch, degrees.
BAND_PITCH_DEG = 15.0
#: How much stable air (under the warm front's shield, in the warm sector)
#: clears the fair-weather cumulus.
STABLE_CLEARING = 0.8
#: Systems reach this many radii.
REACH = 4.0
#: A front's genera fade out over this many radii past its ends, the broad
#: zones it bounds (the warm sector, the cold air) over this many.
END_FADE = 0.3
ZONE_FADE = 1.0


@dataclass
class GenusFields:
    """The genera at some directions.

    :ivar convective: the cumulus continuum's coverage
    :ivar depth: its convective depth, 0 (humilis) to 1 (cumulonimbus)
    :ivar shelf: a cumulonimbus anvil's spread
    :ivar layers: each other genus's coverage (:data:`LAYERS`)
    """

    convective: np.ndarray
    depth: np.ndarray
    shelf: np.ndarray
    layers: dict = field(default_factory=dict)


def _gauss(x, x0, sigma):
    return np.exp(-(((x - x0) / sigma) ** 2))


def _bump(x, x0, x1, edge):
    """1 between *x0* and *x1*, falling to 0 over *edge* outside."""
    return S(x0 - edge, x0, x) * (1.0 - S(x1, x1 + edge, x))


class _Canvas:
    """The channels being drawn, with soft additions and clearing. The
    convective background is kept apart: stable air calms it alone
    (:meth:`calm`), not the systems' own convection."""

    def __init__(self, n: int, base: np.ndarray, depth: np.ndarray):
        self.base = np.clip(base, 0.0, 1.0).astype(np.float64)
        self.calmed = np.ones(n)
        self.c = {"convective": np.zeros(n)}
        for g in LAYERS:
            self.c[g] = np.zeros(n)
        self.depth = depth.astype(np.float64)
        self.shelf = np.zeros(n)
        self.clear = {k: np.ones(n) for k in self.c}
        self.post: list = []

    def add(self, name, idx, value):
        v = np.clip(value, 0.0, 1.0)
        self.c[name][idx] = 1.0 - (1.0 - self.c[name][idx]) * (1.0 - v)

    def deepen(self, idx, value):
        self.depth[idx] = np.maximum(self.depth[idx], np.clip(value, 0.0, 1.0))

    def calm(self, idx, factor):
        """Thin the fair-weather background (stable air)."""
        self.calmed[idx] *= np.clip(factor, 0.0, 1.0)

    def clearing(self, names, idx, factor):
        for name in names:
            self.clear[name][idx] *= np.clip(factor, 0.0, 1.0)

    def later(self, name, idx, value):
        self.post.append((name, idx, value))

    def finish(self) -> GenusFields:
        background = self.base * self.calmed
        self.c["convective"] = 1.0 - (1.0 - background) * (1.0 - self.c["convective"])
        for name, factor in self.clear.items():
            self.c[name] *= factor
        for name, idx, value in self.post:
            self.add(name, idx, value)
        self.shelf = np.maximum(
            self.shelf, S(0.8, 0.95, self.depth) * (self.c["convective"] > 0.05)
        )
        return GenusFields(
            convective=self.c["convective"],
            depth=self.depth,
            shelf=self.shelf,
            layers={g: self.c[g] for g in LAYERS},
        )


def _low(canvas, idx, x, y, obj, t, sea):
    """An extratropical low: its fronts' genera (x along it, y poleward, in
    its radii)."""
    power = intensity(obj, t)
    s = stage(obj, t)
    f = fronts_in_radii(obj, t)
    pts = np.stack([x, y], axis=-1)

    # The warm front's shield: Ci, Cs, As, Ns as it gets closer.
    d_w, a_w = front_query(pts, f.warm, FAR)
    taper_w = (1.0 - S(0.7 * length(f.warm), length(f.warm), a_w)) * (
        1.0 - S(0.0, END_FADE, overshoot(pts, f.warm))
    )
    overcast = np.zeros(len(idx))
    for genus, lo, hi, cover in WARM_SHIELD:
        layer = cover * _bump(d_w, lo, hi, BAND_EDGE) * taper_w * power
        canvas.add(genus, idx, layer)
        if genus in ("nimbostratus", "altostratus"):
            overcast = np.maximum(overcast, layer)
    # Stable overrunning air: no fair-weather cumulus under the shield.
    canvas.calm(idx, 1.0 - STABLE_CLEARING * overcast)

    # A Shapiro-Keyser low's bent-back front, wrapped in its cloud head.
    if len(f.bent) >= 2:
        d_b, _ = front_query(pts, f.bent, FAR)
        gap = np.abs(d_b)
        fade = 1.0 - S(0.0, END_FADE, overshoot(pts, f.bent))
        canvas.add("nimbostratus", idx, 0.8 * _gauss(gap, 0.0, 0.25) * fade * power)
        canvas.add("altostratus", idx, 0.6 * _gauss(gap, 0.0, 0.5) * fade * power)

    # The cold front: a narrow Cb line, congestus ahead of it.
    d_c, a_c = front_query(pts, f.cold, FAR)
    cold_len = length(f.cold)
    from_triple = cold_len - a_c
    taper_c = (1.0 - S(0.7 * cold_len, cold_len, from_triple)) * (
        1.0 - S(0.0, END_FADE, overshoot(pts, f.cold))
    )
    line = _gauss(d_c, 0.0, 0.08)
    canvas.add("convective", idx, 0.9 * line * taper_c * power)
    canvas.deepen(idx, line * (1.0 - 0.5 * from_triple / max(cold_len, 1e-6)))
    ahead = _bump(-d_c, 0.08, 0.25, 0.05) * taper_c
    if "squall" in obj.variant:
        # A pre-frontal squall line: the band ahead is cumulonimbus.
        canvas.add("convective", idx, 0.85 * ahead * power)
        canvas.deepen(idx, ahead)
    else:
        canvas.add("convective", idx, 0.5 * ahead * power)
        canvas.deepen(idx, 0.5 * ahead)

    # The warm sector: overcast and drizzle near the warm front, Sc
    # elsewhere (mostly over sea).
    rho = np.hypot(x, y)
    # Soft edges: the warm sector's stratus carries on under the warm
    # front's nimbostratus a little way (pre-frontal stratus).
    warm_side = (1.0 - S(-0.25, 0.25, d_w)) * (1.0 - S(-0.25, 0.25, d_c))
    # Broad zones bounded by the fronts fade past the fronts' ends, where
    # which side of them a point is on no longer means anything.
    past_w = 1.0 - S(0.0, ZONE_FADE, overshoot(pts, f.warm))
    past_c = 1.0 - S(0.0, ZONE_FADE, overshoot(pts, f.cold))
    sector = warm_side * (1.0 - S(2.0, 2.5, rho)) * past_w * past_c
    canvas.add("stratus", idx, 0.6 * _gauss(d_w, 0.0, 0.4) * sector * power)
    canvas.add("stratocumulus", idx, (0.35 + 0.3 * sea) * sector * power)
    # The warm sector's moist, stable air: stratus and stratocumulus, not
    # fair-weather cumulus.
    canvas.calm(idx, 1.0 - STABLE_CLEARING * sector * power)

    # The comma head, then the dry slot cutting in behind the cold front.
    # The cloud head outlives the low's deepening: it forms as the low
    # matures and lasts through the occlusion (schultz2011), fading only at
    # the end.
    lasting = float(S(0.3, 0.6, s) * (1.0 - S(0.85, 1.0, s)))
    head = _gauss(x, 0.3, 0.9) * _gauss(y, 0.6, 0.5) * lasting
    canvas.add("nimbostratus", idx, 0.8 * head)
    canvas.add("altostratus", idx, 0.6 * head)
    canvas.add("convective", idx, 0.3 * head)
    canvas.deepen(idx, 0.3 * head)
    slot = _gauss(x, -0.3, 0.4) * _gauss(y, -0.6, 0.5) * float(S(0.2, 0.4, s))
    canvas.clearing(canvas.c.keys(), idx, 1.0 - 0.9 * slot)

    # Cold air behind the cold front over sea: open-cell convection.
    cold_air = _bump(d_c, 0.2, 2.0, 0.2) * sea * power * past_c
    canvas.add("convective", idx, 0.45 * cold_air)
    canvas.deepen(idx, 0.3 * cold_air)
    canvas.add("stratocumulus", idx, 0.3 * cold_air)


def _cutoff(canvas, idx, x, y, obj, t):
    power = intensity(obj, t)
    rho = np.hypot(x, y)
    canvas.add("nimbostratus", idx, 0.5 * _gauss(rho, 0.0, 0.8) * power)
    canvas.add("altostratus", idx, 0.4 * _gauss(rho, 0.0, 1.2) * power)
    showers = _gauss(rho, 0.0, 1.0)
    canvas.add("convective", idx, 0.3 * showers * power)
    canvas.deepen(idx, 0.5 * showers)


def _high(canvas, idx, x, y, obj, t, sea, temperature, elevation):
    """A high: subsidence clears its core; a stratocumulus deck over cold
    sea on its eastern flank (wood2012, klein1993), fog near the coast,
    altocumulus round its edge."""
    power = intensity(obj, t)
    rho = np.hypot(x, y)
    canvas.clearing(_SUBSIDENCE, idx, 1.0 - 0.9 * power * _gauss(rho, 0.0, 0.6))
    stable = (
        power * (1.0 - S(14.0, 26.0, temperature)) * sea * (0.5 + 0.5 * S(-0.2, 0.6, x))
    )
    ring = _bump(rho, 0.5, 1.6, 0.3)
    canvas.later("stratocumulus", idx, 0.85 * stable * ring)
    canvas.later("stratus", idx, 0.7 * stable * S(-200.0, -20.0, elevation) * ring)
    canvas.later("altocumulus", idx, 0.35 * power * _bump(rho, 1.0, 1.6, 0.2))


def _tropical(canvas, idx, x, y, obj, t, h):
    """A tropical cyclone (x, y in metres): a clear eye, the eyewall, the
    spiral bands, the central dense overcast's cirrostratus."""
    power = intensity(obj, t)
    r_max = 0.1 * obj.radius
    rho = np.hypot(x, y)
    canvas.clearing(canvas.c.keys(), idx, S(0.3 * r_max, 0.6 * r_max, rho))
    wall = _gauss(rho, r_max, 0.5 * r_max)
    canvas.add("convective", idx, 0.95 * wall * power)
    canvas.deepen(idx, wall)
    azimuth = np.arctan2(y, x)
    spiral = 2.0 * azimuth - 2.0 * h * np.log(np.maximum(rho, 1.0) / r_max) / math.tan(
        math.radians(BAND_PITCH_DEG)
    )
    bands = (
        (0.5 + 0.5 * np.cos(spiral)) ** 3
        * _gauss(rho, 0.0, 4.0 * r_max)
        * S(1.5 * r_max, 2.0 * r_max, rho)
    )
    canvas.add("convective", idx, 0.6 * bands * power)
    canvas.deepen(idx, 0.7 * bands)
    canvas.add("cirrostratus", idx, 0.8 * _gauss(rho, 0.0, 3.0 * r_max) * power)


def apply_systems(
    dirs: np.ndarray,
    systems,
    scales: Scales,
    t: float,
    base: np.ndarray,
    sea: np.ndarray,
    temperature: np.ndarray,
    elevation: np.ndarray,
    depth: np.ndarray | None = None,
) -> GenusFields:
    """The genera at *dirs*: the background (*base* convective coverage and
    *depth*) with every system's zones drawn over it.

    :param sea: share of sea, in [0, 1]
    :param temperature: degrees Celsius
    :param elevation: metres above sea level
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    n = len(dirs)
    canvas = _Canvas(n, base, np.zeros(n) if depth is None else depth)
    index = DirIndex(dirs)
    for obj in systems:
        if intensity(obj, t) <= 0.0:
            continue
        lat, lon = position(obj, t)
        centre = dirs_from_latlon(lat, lon)[0]
        idx = index.query(centre, REACH * obj.radius / scales.radius)
        if len(idx) == 0:
            continue
        h = 1.0 if lat >= 0.0 else -1.0
        x_east, y_north, x_along, y_across = local(dirs[idx], obj, t, scales.radius)
        if obj.kind is Kind.TROPICAL:
            _tropical(canvas, idx, x_east, y_north, obj, t, h)
            continue
        x = x_along / obj.radius
        y = h * y_across / obj.radius  # poleward: the south mirrors the north
        if obj.kind is Kind.LOW:
            _low(canvas, idx, x, y, obj, t, sea[idx])
        elif obj.kind is Kind.CUTOFF:
            _cutoff(canvas, idx, x, y, obj, t)
        else:
            _high(
                canvas,
                idx,
                x_east / obj.radius,
                y_north / obj.radius,
                obj,
                t,
                sea[idx],
                temperature[idx],
                elevation[idx],
            )
    return canvas.finish()


# ── The background ────────────────────────────────────────────────────────────


def zonal_cover(lat: np.ndarray, scales: Scales, itcz: float = 0.0) -> np.ndarray:
    """Fair-weather cover by latitude: the wet equatorial band, the storm
    tracks, a floor between."""
    track = 0.5 * (scales.storm_track[0] + scales.storm_track[1])
    return (
        0.15
        + 0.5 * _gauss(lat, itcz, scales.hadley_edge / 3.0)
        + 0.35 * _gauss(np.abs(lat), track, 12.0)
    )


def instability(humidity: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Air-mass convection's strength: warm and humid air."""
    return S(0.4, 0.8, humidity) * S(20.0, 30.0, temperature)


#: Each kind's swirl of the mesoscale texture at its centre, radians (a
#: high's anticyclonic), and its reach, in its radii.
SWIRL = {Kind.LOW: 3.4, Kind.CUTOFF: 2.0, Kind.TROPICAL: 5.0, Kind.HIGH: -1.0}
SWIRL_REACH = {Kind.LOW: 1.6, Kind.CUTOFF: 1.2, Kind.TROPICAL: 0.8, Kind.HIGH: 1.6}


def swirled(dirs: np.ndarray, systems, scales: Scales, t: float) -> np.ndarray:
    """*dirs* wound by each system's swirl (cyclonic: counter-clockwise in
    the north), most at its centre: the mesoscale texture's coordinates,
    so it curls into spiral arms round the lows."""
    warped = np.array(dirs, dtype=np.float64, copy=True)
    index = DirIndex(dirs)
    for obj in systems:
        power = intensity(obj, t)
        if power <= 0.0:
            continue
        lat, lon = position(obj, t)
        centre = dirs_from_latlon(lat, lon)[0]
        width = SWIRL_REACH[obj.kind] * obj.radius / scales.radius
        idx = index.query(centre, 2.0 * width)
        if len(idx) == 0:
            continue
        angle = np.arccos(np.clip(dirs[idx] @ centre, -1.0, 1.0))
        h = 1.0 if lat >= 0.0 else -1.0
        turn = -h * SWIRL[obj.kind] * power * np.exp(-((angle / width) ** 2))
        p = warped[idx]
        c, s = np.cos(turn)[:, None], np.sin(turn)[:, None]
        along = (p @ centre)[:, None] * centre
        warped[idx] = p * c + np.cross(centre, p) * s + along * (1.0 - c)
    return warped
