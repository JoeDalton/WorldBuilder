"""
WorldBuilder: procedural worlds for Panda3D.

    terrain  procedural heightmap terrain on a spherical planet: biomes,
             hand-made stamps, quadtree LOD, collision queries
    ocean    camera-locked reflective ocean with an optional geometric swell
    clouds   volumetric billboard clouds

All three curve with the planet by the same camera-relative drop, d^2 / 2R, so
they must share one radius (:data:`PLANET_RADIUS_M`) to meet at the horizon.

The ocean and clouds take a ``game`` object by duck typing: ``game.app`` (the
ShowBase), ``game.root_node`` (the world's parent node), and, for the classes
that update themselves, ``game.method_lists`` (a dict of per-frame callables)
and ``game.game_time``. See each class for exactly what it reads.
"""

from world_builder import _version

__version__ = _version.__version__

#: Planet radius shared by the ocean, the cloud deck and the terrain, which
#: must droop identically. Metres.
PLANET_RADIUS_M = 6371000.0
