"""
A planet's sea, drawn by its terrain's chunks (see :mod:`world_builder.ocean.sea`).

    ocean.py   what the sea costs (OceanSettings) and the CPU halves of its
               shading: wave directions and phases, the unresolved octaves'
               variance, the rough fresnel table, the tonemap
    sea.py     Sea: the near look's inputs on the terrain, its planar
               reflection, the frame at the camera's nadir

Shaders: ``shaders/sea.glsl`` (the near look, included by terrain.frag) and
``water.glsl`` (the colour by depth).
"""

from world_builder.ocean.ocean import (
    OceanSettings,
    aces_tonemap,
    bind_sea_defaults,
    compute_wave_dirs,
)
from world_builder.ocean.sea import Sea

__all__ = [
    "OceanSettings",
    "Sea",
    "aces_tonemap",
    "bind_sea_defaults",
    "compute_wave_dirs",
]
