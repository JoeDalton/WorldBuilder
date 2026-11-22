"""
The asteroid fields the tests draw on: the library keeps no presets, a world
file defines every field (src/world_builder/worlds/). These are the specs the
tests were written around; ``cube`` and ``ring`` are also shipped as field
files (checked in test_world_file.py).
"""

import math

import numpy as np

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.spec import (
    Annulus,
    AroundAxis,
    Box,
    Clumping,
    Density,
    FieldSpec,
    KeepOut,
    MaterialSpec,
    Sphere,
    SpinSpec,
    Tumble,
)


def _ringlets() -> tuple[float, ...]:
    """A ring's radial profile: ringlets on a fainter sheet, two gaps and a
    dense band (129 values, inner rim to outer)."""
    x = np.linspace(0.0, 1.0, 129)
    p = 0.7 + 0.2 * np.sin(2.0 * math.pi * 9.0 * x) * np.sin(2.0 * math.pi * 2.0 * x)
    p[(x > 0.30) & (x < 0.34)] = 0.03
    p[(x > 0.71) & (x < 0.72)] = 0.1
    p[(x > 0.50) & (x < 0.60)] = 1.0
    return tuple(round(float(v), 4) for v in np.clip(p, 0.0, 1.0))


PRESETS: dict[str, FieldSpec] = {
    # SpaceFlight's missions 1-2 setting: a 15 km cube, with a corridor along
    # y at z = 500 m (scripts/asteroid_bench.py's cruise flies it).
    "cube": FieldSpec(
        shape=Box((7500.0, 7500.0, 7500.0), edge=500.0),
        density=Density(2.0, Clumping(scale=4000.0, coverage=0.6, softness=0.12)),
        sizes=Distribution.power_law(3.5, 50.0, 2000.0),
        material=MaterialSpec(Distribution.uniform(0.0, 0.15)),
        keep_outs=(KeepOut((0.0, -7500.0, 500.0), 300.0, end=(0.0, 7500.0, 500.0)),),
    ),
    # A ball of ice and rock, spinning around one axis.
    "icy_ball": FieldSpec(
        shape=Sphere(10_000.0, edge=2000.0),
        density=Density(1.0),
        sizes=Distribution.lognormal(150.0, 0.6, 50.0, 1000.0),
        spin=SpinSpec(
            Distribution.lognormal(0.05, 0.5, 0.005, 0.3),
            axis=AroundAxis((0.0, 0.0, 1.0), 20.0),
            tumble=Tumble(
                0.3, Distribution.uniform(0.2, 0.8), Distribution.uniform(0.3, 1.5)
            ),
        ),
        material=MaterialSpec(Distribution.u_shaped(0.4, 0.0, 1.0, log=False)),
    ),
    # A planet ring: 90,000 to 110,000 km from its axis, 4 km thick, icy
    # rocks turning with the orbit, in a sheen of dust.
    "ring": FieldSpec(
        shape=Annulus(
            9.0e7,
            1.1e8,
            4000.0,
            edge=1000.0,
            radial_edge=2.0e6,
            radial_profile=_ringlets(),
        ),
        density=Density(
            2.0, Clumping(scale=20_000.0, coverage=0.7, softness=0.15), dust=0.03
        ),
        sizes=Distribution.power_law(3.0, 50.0, 2000.0),
        spin=SpinSpec(
            Distribution.lognormal(0.03, 0.6, 0.003, 0.3),
            axis=AroundAxis((0.0, 0.0, 1.0), 10.0),
        ),
        material=MaterialSpec(Distribution.uniform(0.6, 1.0)),
        seed=3,
    ),
}
