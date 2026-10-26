"""
WorldBuilder: procedural worlds for Panda3D.

    terrain  procedural heightmap terrain on a true spherical planet, from
             orbit to the ground: biomes, hand-made stamps, cube-sphere
             LOD, collision queries
    ocean    camera-locked reflective ocean with an optional geometric swell
    clouds   volumetric billboard clouds: a flat field, or a whole planet's
             (clouds.sphere: billboards near, a deck far, one pattern)
    atmosphere  single-scattering atmospheres (plus a twilight fill),
             applied by every shader; CPU copies to light anything else
    sky      star field (procedural or painted) and the sun, through the air
    gravity  the bodies' summed point-mass gravity at many points (native or
             numpy)
    depth    near/far camera split, for views from the ground to orbit
    world    every part a game draws around its camera (bodies, air,
             clouds, ocean, sky, sun, lights), built from the game's looks
             and updated as one
    utils    small numeric helpers

The ocean and the flat cloud field (clouds.field) are flat-world objects that
curve with the planet by a camera-relative drop, d^2 / 2R, so they must share
one radius (:data:`PLANET_RADIUS_M` by default) to meet at the horizon. The
terrain is real geometry around the planet centre, of its own
``PlanetSpec.radius``, and so are the planet clouds (clouds.sphere); next to
them, build the ocean with that radius, on the exact sphere
(``exact_sphere=True``), in a frame that follows the camera's nadir (as
:class:`world.World` does).

The ocean and clouds take a ``game`` object by duck typing: ``game.app`` (the
ShowBase), ``game.root_node`` (the world's parent node), and, for the classes
that update themselves, ``game.method_lists`` (a dict of per-frame callables)
and ``game.game_time``. See each class for exactly what it reads.
"""

from world_builder import _version

__version__ = _version.__version__

#: Default planet radius of the ocean and the cloud deck, which must droop
#: identically (SpaceFlight's world). The terrain's is ``PlanetSpec.radius``.
#: Metres.
PLANET_RADIUS_M = 6371000.0

#: Shallow water's colour (linear), and the depth over which the deep water's
#: colour takes over from it (Beer-Lambert e-folding, metres). Shared by the
#: sea's near and far looks, so its colour holds as they fade. The deep
#: colour is the sea's ``water_color``.
SHALLOW_WATER_COLOR = (0.03, 0.16, 0.15)
SHALLOW_WATER_DEPTH_M = 8.0
