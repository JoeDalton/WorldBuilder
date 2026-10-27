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

A node tracked with ``shader=True`` is drawn by the world's model shader
(:mod:`world_builder.objects`) instead of Panda3D's fixed-function lighting:
its owner gives it the shader's state every frame; its ambient light is the
sky light the shader reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    LightAttrib,
    NodePath,
    ShaderAttrib,
    Vec3,
)


@dataclass
class _Tracked:
    node: NodePath
    cast: bool
    sun: NodePath
    ambient: NodePath
    before: LightAttrib | None  # the node's own light attrib, put back on untrack
    shader: bool = False  # drawn by the world's model shader
    shader_before: ShaderAttrib | None = None  # its own, put back


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

    def track(self, node: NodePath, cast: bool = False, shader: bool = False) -> None:
        """Light *node* (and its children) by the light at its own position:
        its own sun and ambient lights, the others off on it. Tracking a
        node again only changes *cast* and *shader*.

        :param cast: whether it casts shadows on the world (the objects'
            shadow map: :attr:`casting`)
        :param shader: drawn by the world's model shader (its owner applies
            it, :attr:`shaded`); its own shader state is put back when no
            longer
        """
        for tracked in self._tracked:
            if tracked.node == node:
                tracked.cast = cast
                if tracked.shader and not shader:
                    self._release_shader(tracked)
                elif shader and not tracked.shader:
                    tracked.shader_before = self._shader_attrib(node)
                tracked.shader = shader
                return
        name = node.get_name() or "tracked"
        sun = self.render.attach_new_node(DirectionalLight(f"{name}_sun"))
        ambient = self.render.attach_new_node(AmbientLight(f"{name}_ambient"))
        before = node.get_attrib(LightAttrib) if node.has_attrib(LightAttrib) else None
        node.set_light_off()  # every other light, the world's first
        node.set_light(sun)
        node.set_light(ambient)
        shader_before = self._shader_attrib(node) if shader else None
        self._tracked.append(
            _Tracked(node, cast, sun, ambient, before, shader, shader_before)
        )

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
    def casting(self) -> list[NodePath]:
        """The tracked nodes that cast shadows."""
        self._drop_removed()
        return [tracked.node for tracked in self._tracked if tracked.cast]

    @property
    def shaded(self) -> list[bool]:
        """Per node, in the order of :attr:`nodes`: whether the world's
        model shader draws it."""
        self._drop_removed()
        return [tracked.shader for tracked in self._tracked]

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
    def _shader_attrib(node: NodePath) -> ShaderAttrib | None:
        return node.get_attrib(ShaderAttrib) if node.has_attrib(ShaderAttrib) else None

    @staticmethod
    def _release_shader(tracked: _Tracked) -> None:
        if not tracked.node.is_empty():
            if tracked.shader_before is None:
                tracked.node.clear_attrib(ShaderAttrib)
            else:
                tracked.node.set_attrib(tracked.shader_before)
        tracked.shader_before = None

    @classmethod
    def _release(cls, tracked: _Tracked) -> None:
        if tracked.shader:
            cls._release_shader(tracked)
        if not tracked.node.is_empty():
            if tracked.before is None:
                tracked.node.clear_attrib(LightAttrib)
            else:
                tracked.node.set_attrib(tracked.before)
        tracked.sun.remove_node()
        tracked.ambient.remove_node()
