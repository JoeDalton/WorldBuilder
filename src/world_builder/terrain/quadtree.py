"""
Quadtree LOD selection over the six faces of the cube-sphere.

Chunk keys are ``(level, face, i, j)`` (see :mod:`cubesphere`), so a chunk's
identity (and its cached data) does not depend on where the camera is. A node
splits when the camera is closer than ``K`` times its size, measured in 3D to
the node's box: the patch's tangent-frame box, bounded radially by its height
range. That keeps every chunk at about ``QUADS / K`` vertices across its own
distance, i.e. roughly constant on screen.

Nodes wholly behind the planet's bulge are culled (exact sphere geometry).
There is no frustum culling here: turning the camera must never reveal holes
or coarse ground, and Panda culls the drawn nodes itself.

The selection never shows a hole: a node is replaced by its children only
once all of them (the ones not culled) are ready to draw, and until then the
missing children are reported as wanted. Cracks between neighbours of
different levels are left to the chunk skirts, so no 2:1 balancing is needed.

The traversal runs on Python floats (numpy per node would be slower); the
per-key geometry is memoised by :meth:`CubeGrid.box`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from world_builder.terrain.cubesphere import CubeGrid, Key

#: Height bounds assumed for a node nobody has generated yet, metres.
UNKNOWN_BOUNDS = (-7000.0, 9000.0)


@dataclass
class Selection:
    """Result of one LOD pass.

    :ivar draw: keys to show this frame; all were ready
    :ivar wanted: keys that should exist but are not ready, most urgent first
    :ivar prefetch: culled children of split nodes that are not ready. They
        are not drawn, but one step of camera motion can bring them into view,
        and their parent cannot stay split without them, so they are
        generated ahead of time (after the wanted keys)
    :ivar keep: keys whose chunks must stay resident: drawn chunks, the roots,
        and every child (culled or not) of a split node
    """

    draw: list[Key] = field(default_factory=list)
    wanted: list[Key] = field(default_factory=list)
    prefetch: list[Key] = field(default_factory=list)
    keep: set[Key] = field(default_factory=set)


class LodSelector:
    """Picks the chunks to draw for a camera position."""

    def __init__(
        self,
        grid: CubeGrid,
        split_factor: float = 2.0,
        hysteresis: float = 1.15,
        max_distance: float | None = None,
        horizon_floor: float = 0.0,
        min_level: int = 0,
    ):
        """
        :param grid: the planet's chunk grid (carries the radius)
        :param split_factor: K; a node splits when nearer than K * its size
        :param hysteresis: a split node merges back only beyond
            hysteresis * K * size, so a camera hovering at the threshold does
            not make chunks flicker
        :param max_distance: nodes entirely beyond this are culled; None keeps
            everything up to the horizon
        :param horizon_floor: lowest height that can hide terrain behind the
            planet's bulge, metres (0 with an ocean, since water is opaque)
        :param min_level: finest level drawn; nodes at it never split, so
            raising it drops the finest detail (0 = full detail)
        """
        self.grid = grid
        self.radius = grid.radius
        self.split_factor = split_factor
        self.hysteresis = hysteresis
        self.max_distance = max_distance
        self.horizon_floor = horizon_floor
        self.min_level = min_level
        self._split: set[Key] = set()

    def reset(self):
        """Forget the hysteresis state (after a spec change)."""
        self._split.clear()

    def select(
        self,
        camera: tuple[float, float, float],
        ready: Callable[[Key], bool],
        bounds: Callable[[Key], tuple[float, float] | None],
    ) -> Selection:
        """One LOD pass.

        :param camera: camera position in the planet frame (centre at the
            origin), metres
        :param ready: whether a key's chunk can be drawn now
        :param bounds: a key's (h_min, h_max) when known, else None
        :returns: what to draw and what to generate
        """
        self._begin(camera)
        selection = Selection()
        split_now: set[Key] = set()
        for root in self.grid.roots():
            selection.keep.add(root)
            box = bounds(root) or UNKNOWN_BOUNDS
            distance = self.distance(root, box)
            if not self._culled(root, box, distance):
                self._visit(root, box, distance, ready, bounds, selection, split_now)
        self._split = split_now
        # Coarse first, so the world is covered before it is refined; nearer
        # first within a level.
        selection.wanted.sort(key=lambda k: (-k[0], self.distance(k, UNKNOWN_BOUNDS)))
        selection.prefetch.sort(key=lambda k: self.distance(k, UNKNOWN_BOUNDS))
        return selection

    def distance(self, key: Key, box: tuple[float, float]) -> float:
        """:returns: 3D distance from the camera of the last pass to the node's
        box with heights *box* = (h_min, h_max), metres"""
        kb = self.grid.box(key)
        r0 = self.radius + box[0]
        r1 = self.radius + box[1]
        x, y, z = self._camera
        u0, u1 = kb.u_range
        v0, v1 = kb.v_range
        a, b, c = kb.t_u
        pu = a * x + b * y + c * z
        a, b, c = kb.t_v
        pv = a * x + b * y + c * z
        a, b, c = kb.up
        pw = a * x + b * y + c * z
        du = max(0.0, min(u0 * r0, u0 * r1) - pu, pu - max(u1 * r0, u1 * r1))
        dv = max(0.0, min(v0 * r0, v0 * r1) - pv, pv - max(v1 * r0, v1 * r1))
        dw = max(0.0, kb.cos_radius * r0 - pw, pw - r1)
        return math.sqrt(du * du + dv * dv + dw * dw)

    # ── internals ──────────────────────────────────────────────────────────────

    def _begin(self, camera) -> None:
        """Per-pass camera terms."""
        x, y, z = (float(c) for c in camera)
        self._camera = (x, y, z)
        rho = math.sqrt(x * x + y * y + z * z)
        self._rho = rho
        self._dir = (x / rho, y / rho, z / rho) if rho > 0.0 else (0.0, 0.0, 1.0)
        floor = self.radius + self.horizon_floor
        self._floor = floor
        # Angle from the camera's nadir to the tangent point on the floor
        # sphere; 0 when the camera is below it (under water: nothing beyond
        # the floor's horizon is visible past the sea surface anyway).
        cos_eye = floor / rho if rho > floor else 1.0
        self._cos_eye = cos_eye
        self._sin_eye = math.sqrt(max(0.0, 1.0 - cos_eye * cos_eye))

    def _visit(self, key, box, distance, ready, bounds, selection, split_now):
        """Visit a node that is not culled.

        :param box: its height bounds (its own, or its parent's if unknown)
        :param distance: camera distance to its box
        """
        limit = self.split_factor * self.grid.chunk_size(key[0])
        if key in self._split:
            limit *= self.hysteresis
        if key[0] > self.min_level and distance < limit:
            split_now.add(key)
            every_child = CubeGrid.children(key)
            selection.keep.update(every_child)
            children = []
            for child in every_child:
                child_box = bounds(child) or box
                child_distance = self.distance(child, child_box)
                if not self._culled(child, child_box, child_distance):
                    children.append((child, child_box, child_distance))
            visible = {c[0] for c in children}
            selection.prefetch.extend(
                c for c in every_child if c not in visible and not ready(c)
            )
            missing = [c for c in visible if not ready(c)]
            if not missing:
                for child, child_box, child_distance in children:
                    self._visit(
                        child,
                        child_box,
                        child_distance,
                        ready,
                        bounds,
                        selection,
                        split_now,
                    )
                return
            selection.wanted.extend(c[0] for c in children if not ready(c[0]))
        if ready(key):
            selection.draw.append(key)
            selection.keep.add(key)
        else:
            selection.wanted.append(key)

    def _culled(self, key, box, distance) -> bool:
        """Whether the node is wholly beyond the horizon or the distance cap."""
        if self.max_distance is not None and distance > self.max_distance:
            return True
        kb = self.grid.box(key)
        dx, dy, dz = self._dir
        ux, uy, uz = kb.up
        cos_phi = dx * ux + dy * uy + dz * uz  # camera direction to patch centre
        if cos_phi >= kb.cos_radius:
            return False  # the camera is over the patch
        # Nearest patch point is alpha = phi - patch radius away (angle); the
        # line of sight grazing the floor sphere reaches beta = eye angle +
        # the angle at which the patch's top pokes over it.
        sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
        cos_alpha = cos_phi * kb.cos_radius + sin_phi * kb.sin_radius
        top = self.radius + box[1]
        cos_top = self._floor / top if top > self._floor else 1.0
        sin_top = math.sqrt(max(0.0, 1.0 - cos_top * cos_top))
        cos_beta = self._cos_eye * cos_top - self._sin_eye * sin_top
        return cos_alpha < cos_beta
