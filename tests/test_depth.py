"""
Unit tests for depth partitioning's range maths (world_builder.depth).

The cameras themselves need a window (see scripts/terrain_shots.py).
"""

import math

import pytest

from world_builder.depth import SplitDepthCameras

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
