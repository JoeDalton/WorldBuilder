"""
Depth partitioning: one view drawn by a far camera, then a near camera.

A planet seen from orbit and a rock under the camera do not fit one 24-bit
depth buffer: with the near plane at 1 m, depth steps are hundreds of metres
at 100 km. :class:`SplitDepthCameras` (the trick KSP uses with its scaled and
local cameras) draws the scene twice into the same window: first a far camera
covering [split, far], then the usual camera covering [near, split] after
clearing depth. Each range gets the whole depth buffer, so over one planet
precision stays under a metre at the far end of each at any altitude (see
:meth:`SplitDepthCameras.update`; :meth:`SplitDepthCameras.system_ranges`
trades some of it far away for reaching other bodies).

Both cameras see the whole scene graph and Panda culls each to its own
frustum, so only objects straddling the split are drawn twice. Nothing else
changes: standard depth, the window's multisampling, existing shaders.

Masks: each camera has one bit of its own, :data:`NEAR_BIT` and
:data:`FAR_BIT` (its mask, :data:`NEAR_CAMERA_MASK` or
:data:`FAR_CAMERA_MASK`, is that bit alone). ``node.hide(HIDE_FROM_NEAR)``
keeps a node out of the near pass (e.g. a sky dome, which belongs to the far
pass); other cameras (the ocean's reflection) are unaffected. Panda draws a
node for a camera if any of the camera's bits is still shown on it, so a
camera with every bit but one could never be hidden from by one bit: with
the near camera's mask "all but FAR_BIT", ``hide(HIDE_FROM_NEAR)`` did
nothing, and the sky was kept out of the near pass only by lying beyond its
range.

Culling: Panda builds a lens's culling frustum (``Lens.make_bounds``) with its
far end cut at ``1 / lens-far-limit``, 10,000 km by default, whatever the far
distance. Past it a whole body was culled while its depth range still
covered it (a planet seen from beyond its moon, 10,000 km away, vanished; its
clouds' deck and the sky's atmosphere, which are never culled, stayed). The
far camera gets explicit cull bounds to its true far plane instead
(:func:`frustum_bounds`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from panda3d.core import (
    BitMask32,
    BoundingHexahedron,
    Camera,
    Lens,
    PandaNode,
    Point3,
)

#: Camera bit only the near camera has (outside the ocean reflection's 0-19).
NEAR_BIT = 21
#: Camera bit only the far camera has.
FAR_BIT = 22
#: ``node.hide(HIDE_FROM_NEAR)``: drawn by the far camera only.
HIDE_FROM_NEAR = BitMask32.bit(NEAR_BIT)
#: The ocean's reflection camera's mask: camera bits 0-19. Panda draws a node
#: for a camera if any of the camera's bits is still shown on it, so a node is
#: hidden from the reflection by hiding *all* of these
#: (``node.hide(REFLECTION_CAMERA_MASK)``); hiding one bit does nothing. The
#: main cameras keep other bits (21 and 22 here, or all of them by default)
#: and still see it.
REFLECTION_CAMERA_MASK = BitMask32(0xFFFFF)
#: ``node.hide(HIDE_FROM_FAR)``: drawn by the near camera only.
HIDE_FROM_FAR = BitMask32.bit(FAR_BIT)
#: The near camera's mask: its bit alone, so :data:`HIDE_FROM_NEAR` hides
#: from it.
NEAR_CAMERA_MASK = BitMask32.bit(NEAR_BIT)
#: The far camera's mask: its bit alone, so :data:`HIDE_FROM_FAR` hides
#: from it.
FAR_CAMERA_MASK = BitMask32.bit(FAR_BIT)


def frustum_bounds(lens: Lens) -> BoundingHexahedron:
    """A perspective lens's frustum to its true far plane.

    What ``lens.make_bounds()`` returns, without its far end cut at
    ``1 / lens-far-limit`` (10,000 km by default). No film offset. For a
    camera's ``set_cull_bounds``, which Panda then culls with instead.

    :param lens: a ``PerspectiveLens``
    """
    fov = lens.get_fov()
    half_x = np.tan(np.radians(fov[0]) / 2.0)
    half_z = np.tan(np.radians(fov[1]) / 2.0)

    def corners(distance):
        x, z = distance * half_x, distance * half_z
        return [
            Point3(-x, distance, -z),
            Point3(x, distance, -z),
            Point3(x, distance, z),
            Point3(-x, distance, z),
        ]

    return BoundingHexahedron(*corners(lens.get_far()), *corners(lens.get_near()))


class SplitDepthCameras:
    """A far camera under the app's camera, drawn first into the same window.

    Takes over the app's camera: its mask becomes :data:`NEAR_CAMERA_MASK`
    and its lens's near/far are set by :meth:`update` every frame.
    :meth:`destroy` restores its mask and depth clearing (the lens keeps the
    last near/far).

    :ivar far_camera: the far camera's NodePath (a child of ``app.camera``)
    :ivar far_lens: its lens, a copy of ``app.camLens`` kept in step with it
    :ivar near_region: the app camera's display region (clears depth)
    :ivar far_region: the far camera's, sorted just before it
    :ivar split: the current split distance, metres (0 before :meth:`update`)
    :ivar far: the current far distance, metres
    """

    def __init__(self, app: Any, near: float = 1.0, overlap: float = 1.02) -> None:
        """
        :param app: the ShowBase (``camera``, ``cam``, ``camLens``, ``win``)
        :param near: near plane of the near camera, metres
        :param overlap: the near camera reaches this factor past the split, so
            the two ranges overlap by a sliver and never leave a gap
        """
        self.app = app
        self.near = near
        self.overlap = overlap
        self.far_lens = app.camLens.make_copy()
        far_node = Camera("far_camera", self.far_lens)
        far_node.set_camera_mask(FAR_CAMERA_MASK)
        self.far_camera = app.camera.attach_new_node(far_node)
        app.cam.node().set_camera_mask(NEAR_CAMERA_MASK)

        self.near_region = app.cam.node().get_display_region(0)
        left, right, bottom, top = self.near_region.get_dimensions()
        self.far_region = app.win.make_display_region(left, right, bottom, top)
        self.far_region.set_sort(self.near_region.get_sort() - 1)
        self.far_region.set_camera(self.far_camera)
        self.near_region.set_clear_depth_active(True)
        self.near_region.set_clear_depth(1.0)
        self.split = 0.0
        self.far = 0.0

    def update(self, split: float, far: float) -> None:
        """Set the ranges for this frame: near camera [near, split], far
        camera [split, far].

        Depth steps at the end of a range are about d^2 / (near * 2^24): pick
        *split* near the nearest thing the far camera must draw finely and
        *far* just past the farthest thing visible.

        :param split: metres
        :param far: metres; raised to 2 * split if smaller
        """
        split = max(split, 2.0 * self.near)
        far = max(far, 2.0 * split)
        self.split, self.far = split, far
        lens = self.app.camLens
        lens.set_near_far(self.near, split * self.overlap)
        self.far_lens.set_near_far(split, far)
        # Follow window resizes and FOV changes.
        self.far_lens.set_fov(lens.get_fov())
        self.far_lens.set_aspect_ratio(lens.get_aspect_ratio())
        self.far_camera.node().set_cull_bounds(frustum_bounds(self.far_lens))

    @staticmethod
    def planet_ranges(
        radius: float,
        camera_distance: float,
        max_height: float = 9000.0,
        min_height: float = 0.0,
        near_split: float = 2000.0,
    ) -> tuple[float, float]:
        """Split and far distances for a camera over a sphere.

        The split sits halfway down to the highest possible ground, so the far
        camera starts well short of anything; the far plane sits just past the
        horizon of the highest ground.

        :param radius: sea-level radius, metres
        :param camera_distance: camera distance from the planet centre, metres
        :param max_height: highest ground above sea level, metres
        :param min_height: lowest ground that can block the view (0 with an
            ocean), metres
        :param near_split: smallest split, metres
        :returns: (split, far)
        """
        altitude = camera_distance - radius
        split = max(near_split, 0.5 * (altitude - max_height))
        floor = radius + min_height
        eye = math.sqrt(max(0.0, camera_distance**2 - floor**2))
        peak = math.sqrt(max(0.0, (radius + max_height) ** 2 - floor**2))
        return split, 1.05 * (eye + peak)

    @staticmethod
    def system_ranges(
        bodies: Iterable[tuple[float, ...]], near_split: float = 2000.0
    ) -> tuple[float, float]:
        """Split and far distances for a camera among several bodies.

        The split follows the nearest body (it bounds the nearest ground);
        the far plane reaches the farthest visible part of any body. Depth
        steps at a distance only depend on the split, so reaching a moon
        10,000 km away costs nothing up close: a few km of step on the moon
        itself, which is a few pixels wide per km anyway.

        :param bodies: (radius, camera distance from its centre, max height,
            min height) per body, as :meth:`planet_ranges` takes them
        :param near_split: smallest split, metres
        :returns: (split, far)
        """
        ranges = [
            SplitDepthCameras.planet_ranges(*body, near_split=near_split)
            for body in bodies
        ]
        return min(r[0] for r in ranges), max(r[1] for r in ranges)

    def destroy(self) -> None:
        """Remove the far camera and its display region."""
        self.app.win.remove_display_region(self.far_region)
        self.far_camera.remove_node()
        self.app.cam.node().set_camera_mask(PandaNode.get_all_camera_mask())
        self.near_region.set_clear_depth_active(False)
