"""
WorldBuilder: procedural worlds for Panda3D.

    terrain  procedural heightmap terrain on a true spherical planet, from
             orbit to the ground: biomes, hand-made stamps, cube-sphere
             LOD, collision queries
    ocean    camera-locked reflective ocean with an optional geometric swell
    clouds   volumetric billboard clouds
    depth    near/far camera split, for views from the ground to orbit

The ocean and clouds are flat-world objects that curve with the planet by a
camera-relative drop, d^2 / 2R, so they must share one radius
(:data:`PLANET_RADIUS_M` by default) to meet at the horizon. The terrain is
real geometry around the planet centre, of its own ``PlanetSpec.radius``;
next to it, build the ocean and clouds with that radius and place them in a
frame that follows the camera's nadir (as scripts/demo_terrain.py does).

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
