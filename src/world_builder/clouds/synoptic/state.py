"""
A weather's synoptic state at its time: its scales, jets and the systems
alive then (cached per spec, as the weather's maps are).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from world_builder.clouds.synoptic.jets import JetWave, jet_waves
from world_builder.clouds.synoptic.scales import Scales
from world_builder.clouds.synoptic.systems import SynopticObject, population


@dataclass(frozen=True, eq=False)
class SynopticState:
    """:ivar scales: the planet's weather scales
    :ivar jets: the jets (:func:`jets.jet_waves`)
    :ivar systems: the weather systems alive at :attr:`t` (none when the
        weather has :attr:`weather.WeatherSpec.systems` off)
    :ivar t: the time, seconds
    """

    scales: Scales
    jets: tuple[JetWave, ...]
    systems: tuple[SynopticObject, ...]
    t: float


@functools.cache
def synoptic_state(spec) -> SynopticState:
    """:returns: the state of :class:`weather.WeatherSpec` *spec* at its
    ``time_s`` (cached per process)"""
    from world_builder.terrain.planet import Planet

    scales = Scales.from_planet(spec.planet)
    jets = jet_waves(scales, spec.seed)
    systems: tuple = ()
    if spec.systems:
        planet = Planet(spec.planet)
        systems = tuple(population(spec, scales, jets, planet, spec.time_s))
    return SynopticState(scales=scales, jets=jets, systems=systems, t=spec.time_s)
