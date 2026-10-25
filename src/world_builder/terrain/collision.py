"""
Terrain collision through Panda3D's own collision traverser.

:meth:`Terrain.enable_collision` gives the terrain one ``CollisionNode``
holding a native ``CollisionTerrain`` solid (:mod:`world_builder.terrain.native`)
that answers for every drawn chunk within ``max_distance`` of the camera. A
game's ``CollisionTraverser`` over ``render`` then sees the terrain like any
other into-solid: spheres (ships, ordnance, sensors) and segments (lasers) get
ordinary ``CollisionEntry`` objects, one per collider per frame, with the
surface point and the normal of the very triangle drawn there.

The node does not hang under the terrain root: its collide mask would make
the traverser visit every chunk node there for every collider. It sits beside
the root and copies the root's transform each frame.

The node's owner tag points at a :class:`TerrainSurface`, whose
:meth:`~TerrainSurface.material_at` names the ground at a hit point.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import BitMask32, CollisionNode, NodePath

from world_builder.terrain.biomes import BIOMES
from world_builder.terrain.native import NativeTerrain

if TYPE_CHECKING:
    from world_builder.terrain.terrain import Terrain

#: Sphere footprint samples per side (see terrain_collision.cpp).
SPHERE_SAMPLES = 5


class TerrainSurface:
    """The owner of a terrain's collision node (its python tag), for the
    game's handlers.

    :ivar terrain: the terrain it belongs to
    """

    #: No single material: ask :meth:`material_at` with the hit point.
    material = None

    def __init__(self, collision: TerrainCollision):
        self._collision = weakref.ref(collision)

    @property
    def terrain(self) -> Terrain | None:
        collision = self._collision()
        return None if collision is None else collision.terrain

    def material_at(
        self, point: ArrayLike, relative_to: NodePath | None = None
    ) -> str | None:
        """What the ground is made of at a point on it, such as a collision
        entry's surface point.

        :param point: a point on the surface, in *relative_to*'s frame
        :param relative_to: the frame of *point*; None = the top of the scene
            graph (``render``)
        :returns: "water" over the sea, else the dominant biome's name
            (``biomes.BIOMES``); None where no collision chunk is registered
        """
        collision = self._collision()
        if collision is None:
            return None
        return collision.material_at(point, relative_to)


class TerrainCollision:
    """A terrain's collision node and the chunks registered with it.

    :ivar node: the CollisionNode's path (into-only)
    :ivar surface: its owner (:class:`TerrainSurface`)
    :ivar max_distance: chunks farther than this from the camera are not
        registered, metres
    :ivar include_ocean: whether the sea surface is solid; None follows the
        planet (solid if it has an ocean), also after a spec change
    """

    def __init__(
        self,
        terrain: Terrain,
        into_mask: BitMask32,
        name: str,
        owner_tag: str,
        max_distance: float,
        include_ocean: bool | None = None,
        samples: int = SPHERE_SAMPLES,
    ):
        self.terrain = terrain
        self.max_distance = float(max_distance)
        self.include_ocean = include_ocean
        self._samples = samples
        self._registered: set[tuple] = set()
        self.native = self._make_native()
        self.native.set_swell(terrain.swell)
        self._root = terrain.root.get_parent().attach_new_node("terrain-collision")
        node = CollisionNode(name)
        node.set_from_collide_mask(BitMask32.all_off())
        node.set_into_collide_mask(into_mask)
        self.native.attach(node, samples)
        self.node = self._root.attach_new_node(node)
        self.surface = TerrainSurface(self)
        self.node.set_python_tag(owner_tag, self.surface)
        self._owner_tag = owner_tag

    def _make_native(self) -> NativeTerrain:
        terrain = self.terrain
        ocean = self.include_ocean
        return NativeTerrain(
            terrain.radius,
            terrain.grid.max_level,
            terrain.grid.n0,
            terrain.spec.has_ocean if ocean is None else ocean,
        )

    @property
    def registered(self) -> frozenset:
        """The chunk keys the solid answers for."""
        return frozenset(self._registered)

    def follow(self, root: NodePath, origin: np.ndarray, rebased: bool) -> None:
        """Copy the terrain root's placement (every frame), and its origin
        when it changed."""
        self._root.set_transform(root.get_transform())
        if rebased:
            self.native.set_origin(origin)

    def update(self, drawn: set, store, camera: np.ndarray) -> None:
        """Register the drawn chunks within reach, drop the others.

        :param drawn: keys of the chunks drawn now
        :param store: the terrain's chunk store
        :param camera: camera position, planet frame
        """
        grid = self.terrain.grid
        wanted = set()
        for key in drawn:
            chunk = store.get(key)
            if chunk is None:
                continue
            reach = 0.75 * grid.chunk_size(key[0])  # centre to corner, roughly
            if np.linalg.norm(chunk.centre - camera) - reach < self.max_distance:
                wanted.add(key)
        for key in self._registered - wanted:
            self.native.remove_chunk(key)
        for key in wanted - self._registered:
            self.native.add_chunk(store[key])
        self._registered = wanted

    def reset(self) -> None:
        """Forget every chunk and take the terrain's (new) planet."""
        node = self.node.node()
        node.clear_solids()
        self.native.close()
        self.native = self._make_native()
        self.native.attach(node, self._samples)
        self._registered = set()
        self.native.set_swell(self.terrain.swell)

    def set_swell(self, swell) -> None:
        """The sea's swell now (``Terrain.set_swell``): the water it raises,
        as drawn, is solid. None = calm water at sea level."""
        self.native.set_swell(swell)

    def material_at(self, point: ArrayLike, relative_to: NodePath | None) -> str | None:
        frame = relative_to if relative_to is not None else self.node.get_top()
        local = self.node.get_relative_point(frame, tuple(float(v) for v in point))
        planet = np.array(local, dtype=np.float64) + self.terrain._origin
        index = self.native.material(planet)
        if index == -2:
            return None
        return "water" if index == -1 else BIOMES[index]

    def destroy(self) -> None:
        self.node.clear_python_tag(self._owner_tag)
        self._root.remove_node()
        self.native.close()
        self._registered = set()
