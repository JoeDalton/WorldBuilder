"""
Asteroid field collision through Panda's own collision traverser.

:meth:`AsteroidField.enable_collision` gives the field one into-only
``CollisionNode`` holding a native ``CollisionAsteroids`` solid
(``terrain/native/asteroid_collision.cpp``) that answers for the rocks within
``max_distance`` of the camera. A game's ``CollisionTraverser`` over render
then sees the rocks like any other into-solid: spheres (ships, ordnance,
sensors) and segments (lasers) get ordinary ``CollisionEntry`` objects, one per
collider per frame, with the surface point and the normal of the very
triangle of the finest mesh, each rock turned to the time it was drawn at.

The node hangs beside the field's root, in the field's axes, at a snapped
origin near the camera, so its float32 coordinates stay small at any distance
from the field's centre.

The node's owner tag is an :class:`AsteroidSurface`: ``material_at`` names
what a hit point is made of ("snow" for an icy rock, else "rock") and
``velocity_at`` how fast the surface moves there (rocks turn).
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BitMask32,
    CollisionNode,
    LPoint3,
    LQuaternionf,
    LVector3,
    NodePath,
)

from world_builder.asteroids.motion import angular_velocity, orientation, rotate
from world_builder.asteroids.shape import ice_patches, ice_weight
from world_builder.terrain.native import NativeAsteroids

if TYPE_CHECKING:
    from world_builder.asteroids.field import AsteroidField

#: The node's origin snaps to this grid (metres) around the camera.
ORIGIN_SNAP = 4096.0
#: A hit point is snow past this much of ice (the patches' edges ramp).
ICE_MATERIAL = 0.5


class AsteroidSurface:
    """The owner of a field's collision node (its python tag), for the
    game's handlers."""

    #: No single material: ask :meth:`material_at` with the hit point.
    material = None

    def __init__(self, collision: AsteroidCollision):
        self._collision = weakref.ref(collision)

    @property
    def field(self) -> AsteroidField | None:
        collision = self._collision()
        return None if collision is None else collision.field

    def material_at(
        self, point: ArrayLike, relative_to: NodePath | None = None
    ) -> str | None:
        """What the rock at a hit point is made of.

        :param point: a point on a rock, in *relative_to*'s frame
        :param relative_to: the frame of *point*; None = render
        :returns: "snow" on the rock's ice patches, "rock" elsewhere, None
            where no rock is
        """
        collision = self._collision()
        return None if collision is None else collision.material_at(point, relative_to)

    def velocity_at(
        self, point: ArrayLike, relative_to: NodePath | None = None
    ) -> np.ndarray:
        """How fast the surface moves at a hit point: the rock's angular
        velocity crossed with the point's offset from its centre.

        :param point: a point on a rock, in *relative_to*'s frame
        :param relative_to: its frame, and the result's; None = render
        :returns: (3,) metres per second (zero where no rock is)
        """
        collision = self._collision()
        if collision is None:
            return np.zeros(3)
        return collision.velocity_at(point, relative_to)


class AsteroidCollision:
    """A field's collision node and the rocks registered with it.

    :ivar node: the CollisionNode's path (into-only)
    :ivar surface: its owner (:class:`AsteroidSurface`)
    :ivar max_distance: rocks farther than this from the camera are not
        solid
    """

    def __init__(
        self,
        field: AsteroidField,
        into_mask: BitMask32,
        name: str,
        owner_tag: str,
        max_distance: float,
        samples: int,
    ):
        self._field = weakref.ref(field)
        classes = field.scatter.classes
        self.native = NativeAsteroids(
            [c.cell for c in classes], [c.hi for c in classes]
        )
        node = CollisionNode(name)
        node.set_from_collide_mask(BitMask32.all_off())
        node.set_into_collide_mask(into_mask)
        self.native.attach(node, samples)
        self.node = field.root.get_parent().attach_new_node(node)
        self.surface = AsteroidSurface(self)
        self.node.set_python_tag(owner_tag, self.surface)
        self.max_distance = float(max_distance)
        self._registered = np.zeros(0, dtype=bool)
        self._origin = None
        self._time = 0.0

    @property
    def field(self) -> AsteroidField | None:
        return self._field()

    def set_prototypes(self, prototypes) -> None:
        """Register each prototype's finest mesh: the triangles collided."""
        for prototype in prototypes:
            mesh = prototype.meshes[-1]
            self.native.set_prototype(prototype.key, mesh.quads, mesh.vertices)

    def update(
        self,
        camera: np.ndarray,
        camera_render: ArrayLike,
        time: float,
        exclude: np.ndarray | None = None,
    ) -> None:
        """Per frame, after the field's: the rocks within reach, the field's
        time, and the node at the camera.

        :param camera: (3,) the camera, field frame
        :param camera_render: the camera in render
        :param time: the time the rocks were drawn at
        :param exclude: per slot, rocks solid otherwise (drawn as terrain)
        """
        field = self._field()
        pop = field.population
        if len(self._registered) < pop.capacity:
            grown = np.zeros(pop.capacity, dtype=bool)
            grown[: len(self._registered)] = self._registered
            self._registered = grown
        live = pop.live
        rel = pop.rocks["centre"] - camera
        reach = self.max_distance + pop.rocks["radius"]
        wanted = live & (np.sum(rel * rel, axis=1) < reach * reach)
        if exclude is not None:
            wanted[: len(exclude)] &= ~exclude[: len(wanted)]
        gone = np.flatnonzero(self._registered & ~wanted)
        new = np.flatnonzero(wanted & ~self._registered)
        if len(gone):
            self.native.remove(gone)
        if len(new):
            self.native.add(new, _rows(pop.rocks[new]))
        self._registered = wanted
        self._time = float(time)
        self.native.set_time(self._time)
        origin = np.round(camera / ORIGIN_SNAP) * ORIGIN_SNAP
        if self._origin is None or np.any(origin != self._origin):
            self._origin = origin
            self.native.set_origin(origin)
        # The node: the field's axes, its origin where `origin` is.
        rotation = field.rotation
        offset = rotation @ (origin - camera)
        quat = field.orientation / np.linalg.norm(field.orientation)
        self.node.set_pos_quat(
            LPoint3(*(np.asarray(camera_render, dtype=np.float64) + offset)),
            LQuaternionf(*quat),
        )

    def _field_point(
        self, point: ArrayLike, relative_to: NodePath | None
    ) -> np.ndarray:
        frame = relative_to if relative_to is not None else self.node.get_top()
        local = self.node.get_relative_point(frame, LPoint3(*point))
        return self._origin + np.array(local, dtype=np.float64)

    def _rock_at(self, point: np.ndarray) -> int | None:
        """:returns: the slot of the registered rock nearest *point* that
        could hold it, or None"""
        pop = self._field().population
        slots = np.flatnonzero(self._registered[: pop.capacity])
        if len(slots) == 0:
            return None
        rocks = pop.rocks[slots]
        d = np.linalg.norm(rocks["centre"] - point, axis=1) / rocks["radius"]
        best = int(np.argmin(d))
        return int(slots[best]) if d[best] <= 1.05 else None

    def material_at(self, point: ArrayLike, relative_to: NodePath | None) -> str | None:
        if self._origin is None:
            return None
        p = self._field_point(point, relative_to)
        slot = self._rock_at(p)
        if slot is None:
            return None
        field = self._field()
        rock = field.population.rocks[slot]
        # The point's direction in the rock's body frame, at the frame's time
        q = orientation(
            rock["q0"], rock["axis"], rock["precession"], rock["spin"], self._time
        )
        inverse = q * np.array([1.0, -1.0, -1.0, -1.0])
        body = rotate(inverse, p - rock["centre"])
        patches = ice_patches(
            field.spec.seed, int(rock["proto"]), field.spec.material.patch_scale
        )
        snow = ice_weight(patches.rank(body / np.linalg.norm(body)), rock["ice"])
        return "snow" if float(snow[0]) > ICE_MATERIAL else "rock"

    def velocity_at(self, point: ArrayLike, relative_to: NodePath | None) -> np.ndarray:
        if self._origin is None:
            return np.zeros(3)
        p = self._field_point(point, relative_to)
        slot = self._rock_at(p)
        if slot is None:
            return np.zeros(3)
        rock = self._field().population.rocks[slot]
        omega = angular_velocity(
            rock["q0"], rock["axis"], rock["precession"], rock["spin"], self._time
        )
        v = np.cross(omega, p - rock["centre"])
        frame = relative_to if relative_to is not None else self.node.get_top()
        out = frame.get_relative_vector(self.node, LVector3(*v))
        return np.array(out, dtype=np.float64)

    def destroy(self) -> None:
        self.node.remove_node()
        self.native.close()


def _rows(rocks: np.ndarray) -> np.ndarray:
    """:returns: (n, 15) the native solid's rows for rock records"""
    rows = np.empty((len(rocks), NativeAsteroids.ROW))
    rows[:, 0:3] = rocks["centre"]
    rows[:, 3] = rocks["radius"]
    rows[:, 4:8] = rocks["q0"]
    rows[:, 8:11] = rocks["axis"]
    rows[:, 11] = rocks["precession"]
    rows[:, 12] = rocks["spin"]
    rows[:, 13] = rocks["proto"]
    rows[:, 14] = rocks["cls"]
    return rows
