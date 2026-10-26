"""
Distributions of one variable on a closed interval: rock sizes, spin rates,
cone angles, ice fractions.

Each is a known family truncated to [lo, hi]. What a field draws from is its
inverse CDF tabulated at :data:`TABLE_SIZE` evenly spaced probabilities
(:meth:`Distribution.table`): a draw is a linear interpolation in that table,
the same arithmetic in numpy and in the native scatter, so both draw the same
values from the same uniforms.

Families::

    uniform(lo, hi)
    normal(mean, sd, lo, hi)
    lognormal(median, sigma, lo, hi)       sigma of the logarithm
    power_law(q, lo, hi)                   density ~ x^-q (q = 3.5: a
                                           collisional cascade, as in rings
                                           and belts)
    u_shaped(a, lo, hi, log=True)          beta(a, a) with a < 1 across the
                                           interval (in log x by default):
                                           mostly small and large, few between
    table(xs, pdf)                         a piecewise-linear density
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import special

#: Points of the tabulated inverse CDF (2^10 intervals).
TABLE_SIZE = 1025

_KINDS = ("uniform", "normal", "lognormal", "power_law", "u_shaped", "table")


@dataclass(frozen=True)
class Distribution:
    """A truncated distribution on [lo, hi]. Build one with the class methods.

    :ivar kind: the family (see the module)
    :ivar lo: smallest value
    :ivar hi: largest value
    :ivar params: the family's parameters
    """

    kind: str
    lo: float
    hi: float
    params: tuple = ()

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"unknown distribution {self.kind!r}")
        if not self.lo <= self.hi:
            raise ValueError(f"empty interval [{self.lo}, {self.hi}]")
        if self.kind in ("lognormal", "power_law") and self.lo <= 0.0:
            raise ValueError(f"{self.kind} needs lo > 0")
        if self.kind == "u_shaped" and self.params[1] and self.lo <= 0.0:
            raise ValueError("a u_shaped distribution in log x needs lo > 0")

    # ── families ──────────────────────────────────────────────────────────

    @classmethod
    def constant(cls, value: float) -> Distribution:
        """Always *value*."""
        return cls("uniform", float(value), float(value))

    @classmethod
    def uniform(cls, lo: float, hi: float) -> Distribution:
        return cls("uniform", float(lo), float(hi))

    @classmethod
    def normal(cls, mean: float, sd: float, lo: float, hi: float) -> Distribution:
        return cls("normal", float(lo), float(hi), (float(mean), float(sd)))

    @classmethod
    def lognormal(
        cls, median: float, sigma: float, lo: float, hi: float
    ) -> Distribution:
        """:param sigma: standard deviation of ln x"""
        return cls("lognormal", float(lo), float(hi), (float(median), float(sigma)))

    @classmethod
    def power_law(cls, q: float, lo: float, hi: float) -> Distribution:
        """Density proportional to x^-q."""
        return cls("power_law", float(lo), float(hi), (float(q),))

    @classmethod
    def u_shaped(cls, a: float, lo: float, hi: float, log: bool = True) -> Distribution:
        """beta(a, a) across [lo, hi]: U-shaped for a < 1, flat at 1.

        :param log: across log x (sizes spanning decades), else across x
        """
        return cls("u_shaped", float(lo), float(hi), (float(a), bool(log)))

    @classmethod
    def table(cls, xs: ArrayLike, pdf: ArrayLike) -> Distribution:
        """A piecewise-linear density through (xs, pdf), xs increasing; the
        pdf need not be normalised."""
        xs = tuple(float(x) for x in np.asarray(xs, dtype=np.float64))
        pdf = tuple(float(p) for p in np.asarray(pdf, dtype=np.float64))
        if len(xs) != len(pdf) or len(xs) < 2 or any(np.diff(xs) <= 0.0):
            raise ValueError("table needs two or more increasing xs, one pdf each")
        if min(pdf) < 0.0 or sum(pdf) <= 0.0:
            raise ValueError("table pdf must be non-negative and not all zero")
        return cls("table", xs[0], xs[-1], (xs, pdf))

    # ── the untruncated family ────────────────────────────────────────────

    def _raw_cdf(self, x: np.ndarray) -> np.ndarray:
        """The family's CDF before truncation (any increasing function of x
        with the right shape will do: it is renormalised on [lo, hi])."""
        k, p = self.kind, self.params
        if k == "uniform":
            return x
        if k == "normal":
            return special.ndtr((x - p[0]) / p[1])
        if k == "lognormal":
            return special.ndtr(np.log(np.maximum(x, 1e-300) / p[0]) / p[1])
        if k == "power_law":
            q = p[0]
            if abs(q - 1.0) < 1e-12:
                return np.log(x)
            # increasing in x for either sign of 1 - q
            return np.power(x, 1.0 - q) / (1.0 - q)
        if k == "u_shaped":
            t = self._unit(x)
            return special.betainc(p[0], p[0], np.clip(t, 0.0, 1.0))
        xs, pdf = np.asarray(p[0]), np.asarray(p[1])
        cumulative = _table_cumulative(p[0], p[1])
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
        dx = np.clip(x - xs[i], 0.0, xs[i + 1] - xs[i])
        slope = (pdf[i + 1] - pdf[i]) / (xs[i + 1] - xs[i])
        return cumulative[i] + pdf[i] * dx + 0.5 * slope * dx * dx

    def _unit(self, x: np.ndarray) -> np.ndarray:
        """u_shaped: x as a fraction of the interval (in log x if so made)"""
        if self.hi == self.lo:
            return np.zeros_like(x)
        if self.params[1]:
            return np.log(x / self.lo) / math.log(self.hi / self.lo)
        return (x - self.lo) / (self.hi - self.lo)

    def _raw_ppf(self, u: np.ndarray) -> np.ndarray | None:
        """The family's inverse of :meth:`_raw_cdf`, where closed: None
        otherwise (the table is then found by bisection)"""
        k, p = self.kind, self.params
        if k == "uniform":
            return u
        if k == "normal":
            return p[0] + p[1] * special.ndtri(u)
        if k == "lognormal":
            return p[0] * np.exp(p[1] * special.ndtri(u))
        if k == "power_law":
            q = p[0]
            if abs(q - 1.0) < 1e-12:
                return np.exp(u)
            return np.power(u * (1.0 - q), 1.0 / (1.0 - q))
        if k == "u_shaped":
            t = special.betaincinv(p[0], p[0], u)
            if self.params[1]:
                return self.lo * np.power(self.hi / self.lo, t)
            return self.lo + t * (self.hi - self.lo)
        return None

    # ── the truncated distribution ────────────────────────────────────────

    @functools.cached_property
    def _bounds(self) -> tuple[float, float]:
        lo, hi = self._raw_cdf(np.array([self.lo, self.hi], dtype=np.float64))
        return float(lo), float(hi)

    def cdf(self, x: ArrayLike) -> np.ndarray:
        """:returns: P(X <= x)"""
        x = np.asarray(x, dtype=np.float64)
        if self.hi == self.lo:
            return (x >= self.lo).astype(np.float64)
        lo, hi = self._bounds
        inside = (self._raw_cdf(np.clip(x, self.lo, self.hi)) - lo) / (hi - lo)
        return np.clip(inside, 0.0, 1.0)

    def ppf(self, u: ArrayLike) -> np.ndarray:
        """:returns: the value below which a share *u* of draws fall"""
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        if self.hi == self.lo:
            return np.full_like(u, self.lo)
        lo, hi = self._bounds
        raw = lo + u * (hi - lo)
        x = self._raw_ppf(raw)
        if x is None:
            x = self._bisect(u)
        return np.clip(x, self.lo, self.hi)

    def _bisect(self, u: np.ndarray) -> np.ndarray:
        a = np.full_like(u, self.lo)
        b = np.full_like(u, self.hi)
        for _ in range(64):
            mid = 0.5 * (a + b)
            below = self.cdf(mid) < u
            a = np.where(below, mid, a)
            b = np.where(below, b, mid)
        return 0.5 * (a + b)

    def pdf(self, x: ArrayLike, eps: float = 1e-6) -> np.ndarray:
        """:returns: the density at x (a central difference of the CDF;
        0 outside [lo, hi])"""
        x = np.asarray(x, dtype=np.float64)
        h = eps * max(self.hi - self.lo, 1e-300)
        a = np.clip(x - h, self.lo, self.hi)
        b = np.clip(x + h, self.lo, self.hi)
        width = np.where(b > a, b - a, 1.0)
        density = (self.cdf(b) - self.cdf(a)) / width
        return np.where((x >= self.lo) & (x <= self.hi), density, 0.0)

    def table_values(self, size: int = TABLE_SIZE) -> np.ndarray:
        """:returns: (size,) the inverse CDF at evenly spaced probabilities
        from 0 to 1: what draws interpolate"""
        return self._table(size)

    @functools.lru_cache(maxsize=4)
    def _table(self, size: int) -> np.ndarray:
        table = self.ppf(np.linspace(0.0, 1.0, size))
        # The ends exactly, and never decreasing (the closed forms round).
        table[0], table[-1] = self.lo, self.hi
        table = np.maximum.accumulate(table)
        table.flags.writeable = False
        return table

    def sample(self, u: ArrayLike) -> np.ndarray:
        """Values for uniforms u in [0, 1), as a field draws them: a linear
        interpolation in :meth:`table_values`."""
        return interpolate(self.table_values(), u)

    def median(self) -> float:
        return float(self.ppf(0.5))

    def mean(self) -> float:
        """The mean of the tabulated distribution (what draws average to)."""
        table = self.table_values()
        return float(0.5 * (table[:-1] + table[1:]).mean())

    def moment(self, power: float) -> float:
        """E[X^power] of the tabulated distribution (by the midpoint rule on
        the table: each interval is an equal share of the draws)."""
        u = (np.arange(4096) + 0.5) / 4096
        return float(np.mean(np.power(self.sample(u), power)))


def interpolate(table: np.ndarray, u: ArrayLike) -> np.ndarray:
    """A draw from a tabulated inverse CDF: the value at probability u by
    linear interpolation. The same operations as ``table_lerp`` in
    native/scatter.c, so both give the same bits.

    :param table: (n,) values at probabilities 0, 1/(n-1), ..., 1
    :param u: uniforms in [0, 1]
    """
    u = np.asarray(u, dtype=np.float64)
    last = len(table) - 1
    x = u * last
    i = np.minimum(np.floor(x).astype(np.int64), last - 1)
    f = x - i
    return table[i] + (table[i + 1] - table[i]) * f


@functools.lru_cache(maxsize=64)
def _table_cumulative(xs: tuple, pdf: tuple) -> np.ndarray:
    """table distributions: the integral of the density up to each x"""
    xs, pdf = np.asarray(xs), np.asarray(pdf)
    areas = 0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xs)
    return np.concatenate(([0.0], np.cumsum(areas)))
