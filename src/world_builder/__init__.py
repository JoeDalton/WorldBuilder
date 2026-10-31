"""
WorldBuilder: procedural worlds for Panda3D.

    terrain  procedural heightmap terrain on a true spherical planet, from
             orbit to the ground: biomes, hand-made stamps, cube-sphere
             LOD, collision queries
    ocean    the sea, drawn by the terrain's chunks: its near look (waves,
             swell, reflection) and what it costs
    clouds   a planet's clouds: volumetric billboards near, a deck far, one
             pattern following a weather map, and high cirrus above them
    atmosphere  single-scattering atmospheres (plus a twilight fill),
             applied by every shader; CPU copies to light anything else
    sky      star field (procedural or painted) and the sun, through the air
    asteroids  asteroid fields and planet rings: rocks scattered from a
             spec, the same every time, stored nowhere
    gravity  the bodies' summed point-mass gravity at many points (native or
             numpy)
    depth    near/far camera split, for views from the ground to orbit
    world    every part a game draws around its camera (bodies, air,
             clouds, ocean, sky, sun, lights), built from the game's looks
             and updated as one
    utils    small numeric helpers

Every part is real geometry around its body's centre, of the body's own
``PlanetSpec.radius``. Parts that load assets take an optional ``game``
object, for its ``game.app.asset_manager``.
"""

from world_builder import _version

__version__ = _version.__version__

#: Shallow water's colour (linear), and the depth over which the deep water's
#: colour takes over from it (Beer-Lambert e-folding, metres). Shared by the
#: sea's near and far looks, so its colour holds as they fade. The deep
#: colour is the sea's ``water_color``.
SHALLOW_WATER_COLOR = (0.03, 0.16, 0.15)
SHALLOW_WATER_DEPTH_M = 8.0
