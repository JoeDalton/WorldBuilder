"""
Panda3D lights per node: a game's model lit by the light at its own position,
not the light at the camera.

The world's Panda3D lights on ``render`` (``World.sun_light``,
``World.ambient_light``) are the light where the camera is. That is wrong for
a model elsewhere: a ship inside an eclipse while the camera is outside it,
or on the night side of a terminator the camera has not crossed. A tracked
node gets its own ``DirectionalLight`` and ``AmbientLight``, and the
world's are switched off on it (and its children); whoever owns the lights
(:meth:`World.update <world_builder.world.World.update>`, or a hand-built
demo) sets them every frame from the light at each node, in one vectorised
call.

Usage::

    lights = TrackedLights(render)
    lights.track(ship)
    # every frame:
    points = origin_world + lights.positions()       # (n, 3), world frame
    direct, ambient = light_at_many(points)          # (n, 3) each
    lights.apply(direct, ambient, sun_direction)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import AmbientLight, DirectionalLight, LightAttrib, NodePath, Vec3


@dataclass
class _Tracked:
    node: NodePath
    cast: bool
    sun: NodePath
    ambient: NodePath
    before: LightAttrib | None  # the node's own light attrib, put back on untrack


class TrackedLights:
    """The nodes lit by their own Panda3D lights (see the module).

    :ivar render: what the nodes' positions are read in, and where their
        lights hang
    """

    def __init__(self, render: NodePath) -> None:
        self.render = render
        self._tracked: list[_Tracked] = []

    def __len__(self) -> int:
        self._drop_removed()
        return len(self._tracked)

    def track(self, node: NodePath, cast: bool = False) -> None:
        """Light *node* (and its children) by the light at its own position:
        its own sun and ambient lights, the others off on it. Tracking a
        node again only changes *cast*.

        :param cast: whether it casts shadows on the world (kept for the
            caster map, not drawn yet)
        """
        for tracked in self._tracked:
            if tracked.node == node:
                tracked.cast = cast
                return
        name = node.get_name() or "tracked"
        sun = self.render.attach_new_node(DirectionalLight(f"{name}_sun"))
        ambient = self.render.attach_new_node(AmbientLight(f"{name}_ambient"))
        before = node.get_attrib(LightAttrib) if node.has_attrib(LightAttrib) else None
        node.set_light_off()  # every other light, the world's first
        node.set_light(sun)
        node.set_light(ambient)
        self._tracked.append(_Tracked(node, cast, sun, ambient, before))

    def untrack(self, node: NodePath) -> None:
        """Back to the lights it had (the world's, inherited)."""
        for index, tracked in enumerate(self._tracked):
            if tracked.node == node:
                self._release(tracked)
                del self._tracked[index]
                return

    def is_tracked(self, node: NodePath) -> bool:
        return any(tracked.node == node for tracked in self._tracked)

    def casts(self, node: NodePath) -> bool:
        """:returns: whether a tracked node casts shadows
        :raises KeyError: for a node not tracked"""
        for tracked in self._tracked:
            if tracked.node == node:
                return tracked.cast
        raise KeyError(node)

    @property
    def nodes(self) -> list[NodePath]:
        """The tracked nodes, in the order of :meth:`positions`."""
        self._drop_removed()
        return [tracked.node for tracked in self._tracked]

    def positions(self) -> np.ndarray:
        """:returns: (n, 3) the tracked nodes' positions in :attr:`render`,
        float64, in the order of :attr:`nodes` (nodes removed meanwhile are
        dropped first)"""
        self._drop_removed()
        return np.array(
            [tuple(tracked.node.get_pos(self.render)) for tracked in self._tracked],
            dtype=np.float64,
        ).reshape(-1, 3)

    def apply(
        self, direct: ArrayLike, ambient: ArrayLike, sun_direction: ArrayLike
    ) -> None:
        """Set each node's lights, in the order of :meth:`positions`.

        :param direct: (n, 3) its sunlight, linear rgb
        :param ambient: (n, 3) its sky light, linear rgb
        :param sun_direction: toward the sun (render's axes)
        """
        direct = np.asarray(direct, dtype=np.float64).reshape(-1, 3)
        ambient = np.asarray(ambient, dtype=np.float64).reshape(-1, 3)
        toward = -np.asarray(sun_direction, dtype=np.float64)
        for tracked, sun, sky in zip(self._tracked, direct, ambient):
            tracked.sun.node().set_color((*sun, 1.0))
            tracked.sun.node().set_direction(Vec3(*toward))
            tracked.ambient.node().set_color((*sky, 1.0))

    def clear(self) -> None:
        """Untrack every node."""
        for tracked in self._tracked:
            self._release(tracked)
        self._tracked = []

    def _drop_removed(self) -> None:
        """Forget nodes the game removed (their lights with them)."""
        kept = []
        for tracked in self._tracked:
            if tracked.node.is_empty():
                tracked.sun.remove_node()
                tracked.ambient.remove_node()
            else:
                kept.append(tracked)
        self._tracked = kept

    @staticmethod
    def _release(tracked: _Tracked) -> None:
        if not tracked.node.is_empty():
            if tracked.before is None:
                tracked.node.clear_attrib(LightAttrib)
            else:
                tracked.node.set_attrib(tracked.before)
        tracked.sun.remove_node()
        tracked.ambient.remove_node()
