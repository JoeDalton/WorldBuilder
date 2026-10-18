"""
Quadtree LOD selection over the flat chart.

Chunk keys are world-aligned ``(level, ix, iy)``, so a chunk's identity (and
its cached data) does not depend on where the camera is. A node splits when
the camera is closer than ``K`` times its size, measured in 3D to the node's
height-bounded box; that keeps every chunk at about ``QUADS / K`` vertices
across its own distance, i.e. roughly constant on screen.

The selection never shows a hole: a node is replaced by its children only
once all of them (the ones not culled) are ready to draw, and until then the
missing children are reported as wanted. Cracks between neighbours of
different levels are left to the chunk skirts, so no 2:1 balancing is needed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from world_builder.terrain.chunkgen import MAX_LEVEL, chunk_size

Key = tuple[int, int, int]
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
        radius: float,
        split_factor: float = 2.0,
        hysteresis: float = 1.15,
        max_distance: float | None = None,
        horizon_floor: float = 0.0,
        min_level: int = 0,
    ):
        """
        :param radius: planet radius, metres (for the horizon cull)
        :param split_factor: K; a node splits when nearer than K * its size
        :param hysteresis: a split node merges back only beyond
            hysteresis * K * size, so a camera hovering at the threshold does
            not make chunks flicker
        :param max_distance: nodes entirely beyond this are culled; None keeps
            everything up to the horizon
        :param min_level: finest level drawn; nodes at it never split, so
            raising it drops the finest detail (0 = full detail)
        :param horizon_floor: lowest height that can hide terrain behind the
            planet's bulge, metres (0 with an ocean, since water is opaque)
        """
        self.radius = radius
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

        :param camera: camera (x, y, z) on the chart, metres
        :param ready: whether a key's chunk can be drawn now
        :param bounds: a key's (h_min, h_max) when known, else None
        :returns: what to draw and what to generate
        """
        selection = Selection()
        split_now: set[Key] = set()
        root = chunk_size(MAX_LEVEL)
        cx, cy, _ = camera
        rx, ry = math.floor(cx / root), math.floor(cy / root)
        for iy in range(ry - 1, ry + 2):
            for ix in range(rx - 1, rx + 2):
                selection.keep.add((MAX_LEVEL, ix, iy))
                self._visit(
                    (MAX_LEVEL, ix, iy),
                    UNKNOWN_BOUNDS,
                    camera,
                    ready,
                    bounds,
                    selection,
                    split_now,
                )
        self._split = split_now
        # Coarse first, so the world is covered before it is refined; nearer
        # first within a level.
        selection.wanted.sort(key=lambda k: (-k[0], self._distance(k, camera, None)))
        selection.prefetch.sort(key=lambda k: self._distance(k, camera, None))
        return selection

    # ── internals ──────────────────────────────────────────────────────────────

    def _visit(self, key, parent_bounds, camera, ready, bounds, selection, split_now):
        known = bounds(key)
        box = known if known is not None else parent_bounds
        if self._culled(key, camera, box):
            return
        distance = self._distance(key, camera, box)
        size = chunk_size(key[0])
        limit = self.split_factor * size
        if key in self._split:
            limit *= self.hysteresis
        if key[0] > self.min_level and distance < limit:
            split_now.add(key)
            every_child = _children(key)
            selection.keep.update(every_child)
            children = [
                c for c in every_child if not self._culled(c, camera, bounds(c) or box)
            ]
            selection.prefetch.extend(
                c for c in every_child if c not in children and not ready(c)
            )
            missing = [c for c in children if not ready(c)]
            if not missing:
                for child in children:
                    self._visit(child, box, camera, ready, bounds, selection, split_now)
                return
            selection.wanted.extend(missing)
        if ready(key):
            selection.draw.append(key)
            selection.keep.add(key)
        else:
            selection.wanted.append(key)

    def _distance(self, key, camera, box) -> float:
        """:returns: 3D distance from the camera to the node's box, metres"""
        level, ix, iy = key
        size = chunk_size(level)
        cx, cy, cz = camera
        dx = max(0.0, ix * size - cx, cx - (ix + 1) * size)
        dy = max(0.0, iy * size - cy, cy - (iy + 1) * size)
        h_min, h_max = box if box is not None else UNKNOWN_BOUNDS
        dz = max(0.0, h_min - cz, cz - h_max)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _culled(self, key, camera, box) -> bool:
        """Whether the node is wholly beyond the horizon or the distance cap."""
        level, ix, iy = key
        size = chunk_size(level)
        cx, cy, cz = camera
        dx = max(0.0, ix * size - cx, cx - (ix + 1) * size)
        dy = max(0.0, iy * size - cy, cy - (iy + 1) * size)
        horizontal = math.hypot(dx, dy)
        if self.max_distance is not None and horizontal > self.max_distance:
            return True
        # Line of sight grazing a sphere at horizon_floor: the eye sees
        # sqrt(2 R h_eye) to the tangent point, the target pokes up from
        # sqrt(2 R h_target) beyond it.
        h_max = box[1]
        floor = self.horizon_floor
        reach = math.sqrt(2.0 * self.radius * max(cz - floor, 1.0)) + math.sqrt(
            2.0 * self.radius * max(h_max - floor, 0.0)
        )
        return horizontal > reach


def _children(key: Key) -> list[Key]:
    level, ix, iy = key
    return [(level - 1, 2 * ix + dx, 2 * iy + dy) for dy in (0, 1) for dx in (0, 1)]


def parent(key: Key) -> Key:
    """:returns: the key of the chunk containing *key* one level up"""
    level, ix, iy = key
    return level + 1, ix >> 1, iy >> 1
