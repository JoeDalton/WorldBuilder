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
    CardMaker,
    NodePath,
    PerspectiveLens,
    Point3,
)

from world_builder.depth import (
    FAR_CAMERA_MASK,
    HIDE_FROM_FAR,
    HIDE_FROM_NEAR,
    MIN_SPLIT_M,
    NEAR_CAMERA_MASK,
    SplitDepthCameras,
    camera_span,
    frustum_bounds,
    split_clear_of,
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


# ── translucent volumes ────────────────────────────────────────────────────────

OVERLAP = 1.02


def _crosses(split: float, span: tuple[float, float]) -> bool:
    """Whether a span crosses the band both cameras draw."""
    near, far = span
    return near / OVERLAP < split and far > split


@pytest.mark.parametrize("span", [(100.0, 1500.0), (2100.0, 4000.0)])
def test_a_span_clear_of_the_band_leaves_the_split(span):
    assert split_clear_of(2000.0, [span], OVERLAP) == 2000.0


def test_without_spans_the_split_stays():
    assert split_clear_of(2000.0, [], OVERLAP) == 2000.0


def test_the_split_moves_in_front_of_a_span_it_cuts():
    split = split_clear_of(2000.0, [(1900.0, 2300.0)], OVERLAP)
    assert split == pytest.approx(1900.0 / OVERLAP)
    assert not _crosses(split, (1900.0, 2300.0))


def test_inside_a_volume_the_split_moves_past_it():
    """The camera is inside: nothing is in front of the volume."""
    assert split_clear_of(2000.0, [(0.0, 2500.0)], OVERLAP) == 2500.0


def test_too_close_to_move_in_front_the_split_moves_past():
    span = (MIN_SPLIT_M / 2, 2200.0)
    assert split_clear_of(2000.0, [span], OVERLAP) == 2200.0


def test_moving_in_front_keeps_clear_of_the_next_span():
    spans = [(1900.0, 2300.0), (1700.0, 1880.0)]
    split = split_clear_of(2000.0, spans, OVERLAP)
    assert split == pytest.approx(1700.0 / OVERLAP)


def test_moving_past_keeps_clear_of_the_next_span():
    spans = [(0.0, 2100.0), (2050.0, 2600.0)]
    assert split_clear_of(2000.0, spans, OVERLAP) == 2600.0


def test_no_span_ever_crosses_the_band():
    rng = np.random.default_rng(5)
    for _ in range(200):
        centres = rng.uniform(0.0, 6000.0, 6)
        radii = rng.uniform(20.0, 1500.0, 6)
        spans = [(max(0.0, c - r), c + r) for c, r in zip(centres, radii)]
        split = split_clear_of(2000.0, spans, OVERLAP)
        assert split >= MIN_SPLIT_M
        assert not any(_crosses(split, span) for span in spans)


def _volume():
    """:returns: (a camera, a 100 m card 1 km ahead of it)"""
    render = NodePath("render")
    camera = render.attach_new_node("cam")
    maker = CardMaker("bubble")
    maker.set_frame(-30.0, 30.0, -40.0, 40.0)  # bounding radius 50 m
    holder = render.attach_new_node("ship")
    holder.set_pos(0.0, 1000.0, 0.0)
    bubble = holder.attach_new_node(maker.generate())
    return camera, bubble


def test_a_volume_spans_its_bounding_sphere_from_the_camera():
    camera, bubble = _volume()
    near, far = camera_span(bubble, camera)
    assert near == pytest.approx(950.0, abs=1.0)
    assert far == pytest.approx(1050.0, abs=1.0)


def test_the_span_follows_the_camera():
    camera, bubble = _volume()
    camera.set_pos(0.0, 1000.0, 0.0)  # inside it
    near, far = camera_span(bubble, camera)
    assert near == 0.0 and far == pytest.approx(50.0, abs=1.0)


def test_a_hidden_or_removed_volume_has_no_span():
    camera, bubble = _volume()
    bubble.hide()
    assert camera_span(bubble, camera) is None
    bubble.show()
    bubble.remove_node()
    assert camera_span(bubble, camera) is None
