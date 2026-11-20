"""
The jet streams: each a zonal axis meandering as three Rossby waves.

``phi_j(lambda, t) = h (phi_0 + sum_m A_m sin(k_m lambda - w_m t + theta_m))``
with the barotropic dispersion ``c_m = U - beta / K_m^2`` (rossby1939), so
long waves stand nearly still (or drift west) and short ones travel east.
Troughs are the axis's equatorward excursions, ridges its poleward ones.

The upper wind blows along the meandering axis with a Gaussian profile
across it; its stream function (:func:`upper_stream`, metres: the wind at
the fastest jet's speed is 1) has the sign convention of
:func:`weather.stream_table`, plus a bump per weather system.

docs/literature/synoptic-weather-review.tex, section "Jet axes as Rossby waves".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from world_builder.clouds.synoptic.scales import (
    JET_SPEED,
    SUBTROPICAL_JET_SPEED,
    Scales,
)
from world_builder.terrain.chart import latlon

_SEED_JET = 503
#: The jets' total meander amplitude, degrees, and as a share of the jet
#: spacing (the narrower of the two).
MEANDER_DEG = 8.0
MEANDER_SPACING = 0.25
#: The subtropical jet meanders this share of the eddy-driven ones' amplitude.
SUBTROPICAL_MEANDER = 0.5
#: The jet's half-width across its axis, in jet spacings, and at most this
#: share of the gap to its neighbouring jet.
JET_WIDTH = 0.25
JET_GAP_WIDTH = 0.4
#: A weather system's upper-wind swirl at its fastest, in the fastest jet's
#: speed (the cirrus bend around it), and its reach in its radii.
SYSTEM_SWIRL = 0.8
_GAUSS_SLOPE = math.sqrt(2.0) * math.exp(-0.5)
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True, eq=False)
class JetWave:
    """One jet: its axis's mean |latitude| and its three waves.

    :ivar hemisphere: +1 north, -1 south
    :ivar lat0: the axis's mean |latitude|, degrees
    :ivar k: (3,) integer zonal wavenumbers
    :ivar amp: (3,) amplitudes, degrees (poleward positive)
    :ivar omega: (3,) angular frequencies, rad/s (positive: moving east)
    :ivar phase: (3,) phases, radians
    :ivar speed: the jet's speed, m/s
    :ivar width: its half-width across the axis, metres
    :ivar radius: the planet's radius, metres
    :ivar subtropical: the subtropical jet (no storm band of its own)
    """

    hemisphere: int
    lat0: float
    k: np.ndarray
    amp: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    speed: float
    width: float
    radius: float
    subtropical: bool = False


def meander_amplitude(scales: Scales) -> float:
    """:returns: the eddy-driven jets' total meander amplitude, degrees"""
    spacing = math.degrees(scales.jet_spacing / scales.radius)
    return min(MEANDER_DEG, MEANDER_SPACING * spacing)


def wavenumbers(scales: Scales) -> np.ndarray:
    """:returns: (3,) the jets' integer wavenumbers: half, once and one and a
    half times the storm track's (at least 1 each)"""
    k = scales.wavenumber
    return np.array([max(1, round(f * k)) for f in (0.5, 1.0, 1.5)], dtype=np.int64)


def jet_waves(
    scales: Scales, seed: int, phases: tuple | None = None
) -> tuple[JetWave, ...]:
    """The planet's jets: north then south, each from the subtropical one
    poleward (:attr:`Scales.jet_lats`).

    :param seed: the weather's seed (amplitude shares and phases)
    :param phases: optional (3,) phases per jet in the same order, replacing
        the seeded ones (pins refit them)
    """
    radius = scales.radius
    k = wavenumbers(scales)
    total = meander_amplitude(scales)
    # A jet is narrower than the gap to its neighbour (on a slow rotator the
    # subtropical and eddy-driven jets crowd together).
    gaps = np.diff(scales.jet_lats)
    width = JET_WIDTH * scales.jet_spacing
    if len(gaps):
        width = min(width, JET_GAP_WIDTH * radius * math.radians(float(gaps.min())))
    out = []
    for hi, hemisphere in enumerate((1, -1)):
        for ji, lat0 in enumerate(scales.jet_lats):
            subtropical = ji == 0
            speed = SUBTROPICAL_JET_SPEED if subtropical else JET_SPEED
            phi = math.radians(lat0)
            circle = radius * math.cos(phi)
            beta = 2.0 * scales.omega * math.cos(phi) / radius
            big_k = k / circle
            c = speed - beta / big_k**2
            omega = k * c / circle
            rng = np.random.default_rng([seed, _SEED_JET, hi, ji])
            # The storm track's own wave dominates: the lows ride it.
            weights = np.array([rng.uniform(0.3, 0.7), 1.0, rng.uniform(0.3, 0.7)])
            amp = total * weights / weights.sum()
            if subtropical:
                amp = amp * SUBTROPICAL_MEANDER
            phase = rng.uniform(0.0, _TWO_PI, 3)
            if phases is not None:
                phase = np.asarray(phases[len(out)], dtype=np.float64)
            out.append(
                JetWave(
                    hemisphere=hemisphere,
                    lat0=lat0,
                    k=k.copy(),
                    amp=amp,
                    omega=omega,
                    phase=phase,
                    speed=speed,
                    width=width,
                    radius=radius,
                    subtropical=subtropical,
                )
            )
    return tuple(out)


def _arguments(jet: JetWave, lon: np.ndarray, t: float) -> np.ndarray:
    """(..., 3) the waves' arguments at longitudes *lon* (degrees), time *t*;
    ``omega * t`` reduced modulo 2 pi in float64 first (large *t* keep
    their precision)."""
    drift = np.fmod(np.asarray(jet.omega, dtype=np.float64) * float(t), _TWO_PI)
    lam = np.radians(np.asarray(lon, dtype=np.float64))[..., None]
    return jet.k * lam - drift + jet.phase


def meander(jet: JetWave, lon: np.ndarray, t: float) -> np.ndarray:
    """:returns: the axis's poleward excursion from its mean, degrees"""
    return np.sum(jet.amp * np.sin(_arguments(jet, lon, t)), axis=-1)


def meander_slope(jet: JetWave, lon: np.ndarray, t: float) -> np.ndarray:
    """:returns: d(poleward excursion, degrees) / d(longitude, degrees)"""
    return np.sum(jet.amp * jet.k * np.cos(_arguments(jet, lon, t)), axis=-1) * (
        math.pi / 180.0
    )


def jet_lat(jet: JetWave, lon: np.ndarray, t: float) -> np.ndarray:
    """:returns: the axis's signed latitude at longitudes *lon*, degrees"""
    return jet.hemisphere * (jet.lat0 + meander(jet, lon, t))


def _extrema(values: np.ndarray, lon: np.ndarray, sign: float) -> np.ndarray:
    """Minima of ``sign * values`` on a periodic grid, parabola-refined."""
    v = sign * values
    prev, nxt = np.roll(v, 1), np.roll(v, -1)
    at = np.nonzero((v < prev) & (v <= nxt))[0]
    curvature = prev[at] - 2.0 * v[at] + nxt[at]
    offset = np.where(
        curvature > 1e-12,
        0.5 * (prev[at] - nxt[at]) / np.maximum(curvature, 1e-12),
        0.0,
    )
    step = lon[1] - lon[0]
    return (lon[at] + offset * step + 180.0) % 360.0 - 180.0


def _grid(jet: JetWave) -> np.ndarray:
    count = max(512, 16 * int(np.max(jet.k)))
    return np.linspace(-180.0, 180.0, count, endpoint=False)


def troughs(jet: JetWave, t: float) -> np.ndarray:
    """:returns: the longitudes of the axis's equatorward extremes, degrees"""
    lon = _grid(jet)
    return _extrema(meander(jet, lon, t), lon, 1.0)


def ridges(jet: JetWave, t: float) -> np.ndarray:
    """:returns: the longitudes of the axis's poleward extremes, degrees"""
    lon = _grid(jet)
    return _extrema(meander(jet, lon, t), lon, -1.0)


def extreme_near(
    jet: JetWave, lon: float, half_width: float, t: float, trough: bool = True
) -> float:
    """The trough (or ridge) within *half_width* degrees of *lon*: the
    deepest one there, parabola-refined (cheaper than all of them).

    :returns: its longitude, degrees in [-180, 180)
    """
    grid = lon + np.linspace(-half_width, half_width, 65)
    v = meander(jet, grid, t) * (1.0 if trough else -1.0)
    i = int(np.clip(np.argmin(v), 1, len(v) - 2))
    curvature = v[i - 1] - 2.0 * v[i] + v[i + 1]
    offset = 0.5 * (v[i - 1] - v[i + 1]) / curvature if curvature > 1e-12 else 0.0
    step = grid[1] - grid[0]
    return float((grid[i] + offset * step + 180.0) % 360.0 - 180.0)


def _stream_scale() -> float:
    return max(JET_SPEED, SUBTROPICAL_JET_SPEED)


def across_axis(jet: JetWave, lat: np.ndarray, lon: np.ndarray, t: float) -> np.ndarray:
    """Distance from a jet's meandering axis, metres, measured across it
    (perpendicular, not along the meridian: where the axis tilts, the
    meridian crosses it at a slant), positive to the north."""
    north = jet.radius * np.radians(lat - jet_lat(jet, lon, t))
    slope = jet.hemisphere * meander_slope(jet, lon, t)  # d(lat)/d(lon), deg/deg
    tilt = slope / np.maximum(np.cos(np.radians(lat)), 1e-6)
    return north / np.sqrt(1.0 + tilt * tilt)


def system_rows(
    systems, t: float, radius: float, swirl: float = SYSTEM_SWIRL
) -> np.ndarray:
    """Each active system's Gaussian bump of the upper wind's stream
    function, ``a exp(-|d - c|^2 k)`` over unit directions d: cyclonic over
    a low (a minimum in the north), anticyclonic over a high and over a
    tropical cyclone's outflow, its fastest wind *swirl* of the fastest
    jet's, scaled by its intensity.

    :returns: (N, 5) float64: centre (3), k, a (metres)
    """
    from world_builder.clouds.synoptic.systems import Kind, intensity, position

    rows = []
    for obj in systems:
        strength = intensity(obj, t)
        if strength <= 0.0:
            continue
        clat, clon = position(obj, t)
        reach = 0.8 if obj.kind is Kind.TROPICAL else 1.6
        width = reach * obj.radius / radius
        amplitude = swirl * width * radius / _GAUSS_SLOPE * strength
        h = 1.0 if clat >= 0.0 else -1.0
        sign = -h if obj.kind in (Kind.LOW, Kind.CUTOFF) else h
        rows.append((*_unit(clat, clon), 1.0 / width**2, sign * amplitude))
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def upper_stream(
    jets, systems, t: float, dirs: np.ndarray, swirl: float = SYSTEM_SWIRL
) -> np.ndarray:
    """The upper wind's stream function at *dirs*, metres (the fastest jet's
    speed is 1): each jet a Gaussian speed profile across its meandering
    axis, ``psi = -(U / U_max) w sqrt(pi)/2 erf(y / w)``, plus a Gaussian
    bump per active system (:func:`system_rows`).

    :param jets: :func:`jet_waves`
    :param systems: :class:`systems.SynopticObject` s
    :param dirs: (N, 3) unit directions
    :param swirl: the systems' fastest wind, in the fastest jet's
    """
    from scipy.special import erf

    dirs = np.asarray(dirs, dtype=np.float64)
    lat, lon = latlon(dirs)
    psi = np.zeros(len(dirs))
    top = _stream_scale()
    for jet in jets:
        y = across_axis(jet, lat, lon, t)
        share = jet.speed / top
        psi -= share * jet.width * 0.5 * math.sqrt(math.pi) * erf(y / jet.width)
    if jets:
        for cx, cy, cz, k, a in system_rows(systems, t, jets[0].radius, swirl):
            offset = dirs - np.array([cx, cy, cz])
            psi += a * np.exp(-np.einsum("ij,ij->i", offset, offset) * k)
    return psi


def _unit(lat: float, lon: float) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    return np.array(
        [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]
    )
