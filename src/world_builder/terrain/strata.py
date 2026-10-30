"""
Rock strata: the level layers a planet's tablelands are cut into.

One table per planet: from :data:`LOW` to :data:`HIGH` metres above sea
level, layers 20 to 250 m thick, about one in three hard. A tableland's
walls take their shape from it: a hard layer stands as a cliff with a narrow
ledge on its top, shedding gently outward, a soft one weathers back to a
slope. The layers are level
and shared, so the same cliff and the same ledge stand at the same height on
every mesa and butte, and the shader tints the ground by the same table
(:func:`strata_lut`): bands that line up across the whole country.

The cross-section is a run: :meth:`Strata.run` is the horizontal distance a
wall takes to climb from :data:`LOW` to an altitude, the sum of every layer's
run below it (thickness / tan(slope), plus the ledge on a hard layer's top).
A wall whose foot is at height f is, at horizontal distance s into its
table, at altitude ``rise(run(f) + s)`` (:meth:`Strata.profile`).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

#: Altitude range the table covers, metres above sea level.
LOW, HIGH = -500.0, 6000.0
#: Wall slopes, degrees: hard layers stand as cliffs, soft ones as steep
#: slopes; the debris at a wall's foot (its talus) lies at a gentler one.
HARD_SLOPE, SOFT_SLOPE, TALUS_SLOPE = 80.0, 50.0, 30.0
#: A hard layer's ledge sheds outward at this slope, degrees: a wall's
#: altitude then rises strictly with its run, so :meth:`Strata.run` is the
#: exact inverse of :meth:`Strata.rise` (a level ledge has no single run).
LEDGE_SLOPE = 8.0
#: Placeholder tints of the layers (albedo multipliers): grey, red, ochre,
#: buff, pale, brown.
PALETTE = np.array(
    [
        (0.86, 0.86, 0.90),
        (1.12, 0.74, 0.58),
        (1.12, 0.92, 0.60),
        (1.05, 1.00, 0.84),
        (1.16, 1.12, 1.04),
        (0.82, 0.66, 0.50),
    ]
)


@dataclass(frozen=True, eq=False)
class Strata:
    """One planet's layers, bottom first.

    :ivar tops: (L,) each layer's top, metres above sea level
    :ivar hard: (L,) bool, whether the layer stands as a cliff
    :ivar tints: (L, 3) albedo multipliers
    :ivar heights: (M,) altitudes of the run's nodes, increasing (a ledge
        climbs its last metres)
    :ivar runs: (M,) the run at each node, increasing
    """

    tops: np.ndarray
    hard: np.ndarray
    tints: np.ndarray
    heights: np.ndarray
    runs: np.ndarray

    def run(self, altitude: np.ndarray) -> np.ndarray:
        """:returns: the horizontal run of a wall from :data:`LOW` up to
        *altitude*, metres"""
        return np.interp(altitude, self.heights, self.runs)

    def rise(self, run: np.ndarray) -> np.ndarray:
        """:returns: the altitude a wall reaches after *run* metres"""
        return np.interp(run, self.runs, self.heights)

    def profile(
        self,
        foot: np.ndarray,
        s: np.ndarray,
        ceiling: np.ndarray,
        talus: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """A wall's cross-section.

        :param foot: (N,) altitude of the wall's foot, metres
        :param s: (N,) horizontal distance into the table from the foot,
            metres (at most 0 off the table)
        :param ceiling: (N,) the table's top surface
        :param talus: height of the debris slope at the foot, metres (at
            :data:`TALUS_SLOPE`), before the layers' own walls
        :returns: (N,) altitude, from *foot* up to *ceiling*
        """
        s = np.maximum(s, 0.0)
        slope = math.tan(math.radians(TALUS_SLOPE))
        talus_run = talus / slope
        above = foot + talus
        wall = self.rise(self.run(above) + np.maximum(s - talus_run, 0.0))
        altitude = np.where(s < talus_run, foot + s * slope, wall)
        return np.clip(altitude, foot, np.maximum(ceiling, foot))


@functools.lru_cache(maxsize=16)
def make_strata(seed: int) -> Strata:
    """:returns: a planet's strata, the same for the same seed"""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, 0x57A7A])
    tops, hard, tints = [], [], []
    altitude, colour = LOW, -1
    while altitude < HIGH:
        # Hard layers are the thick ones: walls stand mostly as cliffs.
        is_hard = rng.random() < 1.0 / 3.0
        altitude += rng.uniform(60.0, 250.0) if is_hard else rng.uniform(20.0, 100.0)
        tops.append(altitude)
        hard.append(is_hard)
        # Neighbouring layers never share a tint.
        colour = (colour + 1 + rng.integers(len(PALETTE) - 1)) % len(PALETTE)
        tints.append(PALETTE[colour])
    tops = np.array(tops)
    hard = np.array(hard)

    heights, runs = [LOW], [0.0]
    bottom = LOW
    for top, is_hard in zip(tops, hard):
        if is_hard:  # a cliff, then its ledge up to the top
            ledge = rng.uniform(10.0, 30.0)
            lip = top - ledge * math.tan(math.radians(LEDGE_SLOPE))
            runs.append(runs[-1] + (lip - bottom) / math.tan(math.radians(HARD_SLOPE)))
            heights.append(lip)
            runs.append(runs[-1] + ledge)
        else:
            runs.append(runs[-1] + (top - bottom) / math.tan(math.radians(SOFT_SLOPE)))
        heights.append(top)
        bottom = top
    return Strata(
        tops=tops,
        hard=hard,
        tints=np.array(tints),
        heights=np.array(heights),
        runs=np.array(runs),
    )


def strata_lut(strata: Strata, size: int = 1024) -> np.ndarray:
    """The layers' tints by altitude, for the shader.

    :param strata: the planet's layers
    :param size: texels from :data:`LOW` to :data:`HIGH`
    :returns: (size, 3) float32 albedo multipliers (linear filtering
        softens the band edges by a texel, about 6 m)
    """
    altitude = LOW + (np.arange(size) + 0.5) * (HIGH - LOW) / size
    layer = np.minimum(np.searchsorted(strata.tops, altitude), len(strata.tops) - 1)
    return strata.tints[layer].astype(np.float32)
