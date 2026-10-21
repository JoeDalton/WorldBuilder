"""
The sky: a star field (procedural or painted), the sun's disc, and the light
scattered by every atmosphere along each view direction.

:class:`Sky` draws a camera-centred sphere first, behind everything; terrain,
clouds and water drawn over it apply the atmospheres themselves (see
:mod:`world_builder.atmosphere`), so a moon in the daytime sky is dimmed and
tinted by the air in front of it, and stars fade out by day.
"""

from world_builder.sky.sky import Sky
from world_builder.sky.starfield import (
    NEBULA_PALETTES,
    NebulaStyle,
    Starfield,
    load_cube_map,
    make_glow,
    make_stars,
)

__all__ = [
    "NEBULA_PALETTES",
    "NebulaStyle",
    "Sky",
    "Starfield",
    "load_cube_map",
    "make_glow",
    "make_stars",
]
