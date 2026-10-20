"""
Unit tests for depth partitioning's range maths (world_builder.depth).

The cameras themselves need a window (see scripts/terrain_shots.py).
"""

import math

import numpy as np
import pytest
from panda3d.core import (
    BitMask32,
    BoundingBox,
    BoundingVolume,
    NodePath,
    PerspectiveLens,
    Point3,
)

from world_builder.depth import (
    FAR_CAMERA_MASK,
    HIDE_FROM_FAR,
    HIDE_FROM_NEAR,
    NEAR_CAMERA_MASK,
    SplitDepthCameras,
    frustum_bounds,
)

R = 600e3


def _precision(near: float, distance: float) -> float:
    """Depth step of a 24-bit buffer at *distance*, metres."""
    return distance * distance / (near * 2**24)


@pytest.mark.parametrize("altitude", [2.0, 1000.0, 20e3, 200e3, 1800e3])
def test_planet_ranges_keep_depth_steps_under_a_metre(altitude):
    split, far = SplitDepthCameras.planet_ranges(R, R + altitude)
    assert split >= 2000.0 and far > split
    assert _precision(split, far) < 1.0
    # The far plane reaches past the horizon of the highest ground.
    horizon = math.sqrt((R + altitude) ** 2 - R**2) + math.sqrt(
        (R + 9000.0) ** 2 - R**2
    )
    assert far >= horizon


def test_the_far_camera_starts_short_of_the_ground():
    """Everything nearer than the split is drawn by the near camera; from high
    up nothing is, so the near camera's coarse far end is never used."""
    for altitude in (50e3, 500e3):
        split, _ = SplitDepthCameras.planet_ranges(R, R + altitude)
        assert split < altitude - 9000.0


def test_system_ranges_follow_the_nearest_ground_and_reach_the_farthest_body():
    moon = (200e3, 9.4e6, 9000.0, -7000.0)  # 9,400 km from the camera
    planet = (R, R + 2.0, 9000.0, 0.0)  # standing on the planet
    split, far = SplitDepthCameras.system_ranges([planet, moon])
    assert split == SplitDepthCameras.planet_ranges(*planet)[0]
    assert far >= 9.4e6  # the moon's near side is in range
    # Up close, precision only depends on the split.
    assert _precision(split, 100e3) < 1.0


def _lens(near, far):
    lens = PerspectiveLens()
    lens.set_fov(70.0, 42.0)
    lens.set_near_far(near, far)
    return lens


def test_frustum_bounds_match_the_lens_within_its_own_limit():
    lens = _lens(500e3, 5e6)
    ours, panda = frustum_bounds(lens), lens.make_bounds()
    for i in range(8):
        np.testing.assert_allclose(ours.get_point(i), panda.get_point(i), rtol=1e-5)


def test_frustum_bounds_reach_past_ten_thousand_km():
    """Panda's own frustum stops at 1 / lens-far-limit (10,000 km): a planet
    beyond its moon was culled while the depth range covered it."""
    lens = _lens(570e3, 12e6)
    planet = BoundingBox(Point3(-1e6, 10.2e6, -1e6), Point3(1e6, 11.4e6, 1e6))
    assert lens.make_bounds().contains(planet) == BoundingVolume.IF_no_intersection
    assert frustum_bounds(lens).contains(planet) != BoundingVolume.IF_no_intersection
    assert frustum_bounds(lens).get_max()[1] == pytest.approx(12e6)


def _drawn(node: NodePath, camera_mask) -> bool:
    """Panda's rule (PandaNode::compare_draw_mask): drawn if any of the
    camera's bits is shown, or not controlled, on the node."""
    n = node.node()
    shown = n.get_draw_show_mask() | ~n.get_draw_control_mask()
    return not (shown & camera_mask).is_zero()


def test_hiding_from_one_camera_leaves_the_others():
    """The sky dome is hidden from the near camera by one bit; with a near
    camera mask of "every bit but one" that hid nothing."""
    reflection = BitMask32(0xFFFFF)  # the ocean's reflection camera's bits
    sky = NodePath("sky")
    sky.hide(HIDE_FROM_NEAR)
    assert not _drawn(sky, NEAR_CAMERA_MASK)
    assert _drawn(sky, FAR_CAMERA_MASK) and _drawn(sky, reflection)
    near_only = NodePath("near_only")
    near_only.hide(HIDE_FROM_FAR)
    assert not _drawn(near_only, FAR_CAMERA_MASK)
    assert _drawn(near_only, NEAR_CAMERA_MASK)
