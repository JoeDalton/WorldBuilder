"""
Quadtree LOD selection over the six faces of the cube-sphere.

Chunk keys are ``(level, face, i, j)`` (see :mod:`cubesphere`), so a chunk's
identity (and its cached data) does not depend on where the camera is. A node
splits when the camera is closer than ``K`` times its size, measured in 3D to
the node's box: the patch's tangent-frame box, bounded radially by its height
range. That keeps every chunk at about ``QUADS / K`` vertices across its own
distance, i.e. roughly constant on screen. A split node merges back only
past ``hysteresis`` times that distance, so a hovering camera does not make
chunks flicker.

Nodes wholly behind the planet's bulge are culled (exact sphere geometry), as
are nodes wholly beyond the optional ``max_distance``.
There is no frustum culling here: turning the camera must never reveal holes
or coarse ground, and Panda culls the drawn nodes itself.

The selection never shows a hole: a node is replaced by its children only
once all of them (the ones not culled) are ready to draw, and until then the
missing children are reported as wanted. Cracks between neighbours of
different levels are left to the chunk skirts, so no 2:1 balancing is needed.

What is wanted is ordered by benefit: ground with nothing drawn over it
first (coarse first), then the chunks largest on screen (distance over size),
those behind the view after the others when the view direction is given.

The traversal runs on Python floats (numpy per node would be slower); the
per-key geometry is memoised by :meth:`CubeGrid.box`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from numpy.typing import ArrayLike

from world_builder.terrain.cubesphere import CubeGrid, Key

#: Height bounds assumed for a node nobody has generated yet, metres.
UNKNOWN_BOUNDS = (-7000.0, 9000.0)
#: Cached per-node geometry entries before the cache is reset.
_GEOMETRY_CACHE_LIMIT = 200_000
#: A wanted chunk behind the view counts as this many times farther. Only the
#: order changes: turning still finds the ground, a little coarser for longer.
BEHIND_PENALTY = 3.0
#: Cosine of the angle off the view direction past which a chunk is behind it
#: (about 107 degrees: the screen's corners and some margin are in view).
_BEHIND_COS = -0.3


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
        self._geometry: dict[Key, tuple] = {}

    def reset(self) -> None:
        """Forget the hysteresis state (after a spec change)."""
        self._split.clear()

    def select(
        self,
        camera: ArrayLike,
        ready: Callable[[Key], bool],
        bounds: Callable[[Key], tuple[float, float] | None],
        forward: ArrayLike | None = None,
    ) -> Selection:
        """One LOD pass.

        :param camera: (3,) camera position in the planet frame (centre at
            the origin), metres
        :param ready: whether a key's chunk can be drawn now
        :param bounds: a key's (h_min, h_max) when known, else None. Return
            the same tuple object for a key while its bounds do not change
            (a dict's ``get`` does): the per-key geometry cache then
            recognises it by identity
        :param forward: (3,) view direction in the planet frame, to want the
            chunks in view first; None = no preference
        :returns: what to draw and what to generate
        """
        self._begin(camera)
        self._distances: dict[Key, float] = {}
        selection = Selection()
        split_now: set[Key] = set()
        for root in self.grid.roots():
            selection.keep.add(root)
            box = bounds(root) or UNKNOWN_BOUNDS
            distance, culled = self._measure(root, box)
            self._distances[root] = distance
            if not culled:
                self._visit(root, box, distance, ready, bounds, selection, split_now)
        self._split = split_now
        priority = self._priority(ready, forward)
        selection.wanted.sort(key=priority)
        selection.prefetch.sort(key=self._distances.__getitem__)
        return selection

    def _priority(
        self, ready: Callable[[Key], bool], forward: ArrayLike | None
    ) -> Callable[[Key], tuple]:
        """:returns: the sort key of wanted chunks, most urgent first.

        Ground nothing covers yet (a root, or the child of a root not ready)
        comes first, coarse first, so the world is covered before it is
        refined. Every other wanted chunk refines one already drawn: the
        largest on screen first (distance over size), those behind the view
        as if farther. Distances as measured in the traversal (with each
        key's own bounds): re-measuring with other bounds would evict the
        geometry cache entries the next pass needs.
        """
        distances = self._distances
        top = self.grid.max_level
        size = self.grid.chunk_size
        x, y, z = self._camera
        if forward is not None:
            fx, fy, fz = (float(c) for c in forward)
            norm = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
            fx, fy, fz = fx / norm, fy / norm, fz / norm
        radius = self.radius
        box = self.grid.box

        def key(k: Key) -> tuple:
            level = k[0]
            if level == top or not ready(CubeGrid.parent(k)):
                return (0, -level, distances[k])
            score = distances[k] / size(level)
            if forward is not None and score > 1.0:
                ux, uy, uz = box(k).up
                dx, dy, dz = ux * radius - x, uy * radius - y, uz * radius - z
                along = dx * fx + dy * fy + dz * fz
                if along < _BEHIND_COS * math.sqrt(dx * dx + dy * dy + dz * dz):
                    score *= BEHIND_PENALTY
            return (1, score, 0.0)

        return key

    def distance(self, key: Key, box: tuple[float, float]) -> float:
        """:returns: 3D distance from the camera of the last pass to the node's
        box with heights *box* = (h_min, h_max), metres"""
        return self._measure(key, box)[0]

    # ── internals ──────────────────────────────────────────────────────────────

    def _begin(self, camera: ArrayLike) -> None:
        """Per-pass camera terms."""
        x, y, z = (float(c) for c in camera)
        self._camera = (x, y, z)
        rho = math.sqrt(x * x + y * y + z * z)
        self._rho = rho
        self._dir = (x / rho, y / rho, z / rho) if rho > 0.0 else (0.0, 0.0, 1.0)
        floor = self.radius + self.horizon_floor
        if floor != getattr(self, "_floor", None):
            self._geometry = {}  # cached horizon terms depend on the floor
        self._floor = floor
        # Angle from the camera's nadir to the tangent point on the floor
        # sphere; 0 when the camera is below it (under water: nothing beyond
        # the floor's horizon is visible past the sea surface anyway).
        cos_eye = floor / rho if rho > floor else 1.0
        self._cos_eye = cos_eye
        self._sin_eye = math.sqrt(max(0.0, 1.0 - cos_eye * cos_eye))
        level_size = self.grid.chunk_size
        self._limits = [
            self.split_factor * level_size(level)
            for level in range(self.grid.max_level + 1)
        ]

    def _visit(self, key, box, distance, ready, bounds, selection, split_now):
        """Visit a node that is not culled.

        :param box: its height bounds (its own, or its parent's if unknown)
        :param distance: camera distance to its box
        """
        level = key[0]
        limit = self._limits[level]
        if key in self._split:
            limit *= self.hysteresis
        if level > self.min_level and distance < limit:
            split_now.add(key)
            _, face, i, j = key
            i, j, child_level = 2 * i, 2 * j, level - 1
            every_child = (
                (child_level, face, i, j),
                (child_level, face, i + 1, j),
                (child_level, face, i, j + 1),
                (child_level, face, i + 1, j + 1),
            )
            selection.keep.update(every_child)
            measure = self._measure
            distances = self._distances
            visible = []
            missing = False
            for child in every_child:
                child_box = bounds(child) or box
                child_distance, culled = measure(child, child_box)
                distances[child] = child_distance
                if culled:
                    if not ready(child):
                        selection.prefetch.append(child)
                    continue
                visible.append((child, child_box, child_distance))
                if not missing and not ready(child):
                    missing = True
            if not missing:
                for child, child_box, child_distance in visible:
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
            selection.wanted.extend(c[0] for c in visible if not ready(c[0]))
        if ready(key):
            selection.draw.append(key)
            selection.keep.add(key)
        else:
            selection.wanted.append(key)

    def _measure(self, key, box) -> tuple[float, bool]:
        """Distance to the node's box, and whether the node is wholly beyond
        the horizon or the distance cap.

        The hot loop of the pass: the box extents for these bounds and the
        horizon terms that depend only on the node are cached per key.
        """
        entry = self._geometry.get(key)
        if entry is None or (entry[0] is not box and entry[0] != box):
            entry = self._prepare(key, box)
        (
            _,
            ux,
            uy,
            uz,
            vx,
            vy,
            vz,
            wx,
            wy,
            wz,
            u0,
            u1,
            v0,
            v1,
            w0,
            w1,
            cos_r,
            sin_r,
            cos_top,
            sin_top,
        ) = entry
        x, y, z = self._camera
        p = ux * x + uy * y + uz * z
        d = u0 - p if p < u0 else (p - u1 if p > u1 else 0.0)
        total = d * d
        p = vx * x + vy * y + vz * z
        d = v0 - p if p < v0 else (p - v1 if p > v1 else 0.0)
        total += d * d
        p = wx * x + wy * y + wz * z
        d = w0 - p if p < w0 else (p - w1 if p > w1 else 0.0)
        distance = math.sqrt(total + d * d)

        if self.max_distance is not None and distance > self.max_distance:
            return distance, True
        dx, dy, dz = self._dir
        cos_phi = dx * wx + dy * wy + dz * wz  # camera direction to patch centre
        if cos_phi >= cos_r:
            return distance, False  # the camera is over the patch
        # Nearest patch point is alpha = phi - patch radius away (angle); the
        # line of sight grazing the floor sphere reaches beta = eye angle +
        # the angle at which the patch's top pokes over it.
        s = 1.0 - cos_phi * cos_phi
        cos_alpha = cos_phi * cos_r + (math.sqrt(s) if s > 0.0 else 0.0) * sin_r
        cos_beta = self._cos_eye * cos_top - self._sin_eye * sin_top
        return distance, cos_alpha < cos_beta

    def _prepare(self, key, box) -> tuple:
        """Cache the node's box extents for *box* and its horizon terms."""
        if len(self._geometry) > _GEOMETRY_CACHE_LIMIT:
            self._geometry.clear()
        kb = self.grid.box(key)
        r0 = self.radius + box[0]
        r1 = self.radius + box[1]
        (u_lo, u_hi), (v_lo, v_hi), (w_lo, w_hi) = kb.extents(r0, r1)
        cos_top = self._floor / r1 if r1 > self._floor else 1.0
        entry = (
            box,
            *kb.t_u,
            *kb.t_v,
            *kb.up,
            u_lo,
            u_hi,
            v_lo,
            v_hi,
            w_lo,
            w_hi,
            kb.cos_radius,
            kb.sin_radius,
            cos_top,
            math.sqrt(max(0.0, 1.0 - cos_top * cos_top)),
        )
        self._geometry[key] = entry
        return entry
