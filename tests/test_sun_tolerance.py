"""The shadow maps are not drawn again for a sun that has hardly moved."""

import math

import numpy as np

from world_builder.clouds.shadow_map import _Map
from world_builder.utils import SUN_TOLERANCE, sun_moved


def _turned(angle: float) -> np.ndarray:
    """:returns: the overhead sun turned by *angle* radians toward +x"""
    return np.array([math.sin(angle), 0.0, math.cos(angle)])


def test_a_sun_turned_less_than_the_tolerance_has_not_moved():
    sun = _turned(0.0)
    assert not sun_moved(sun, sun)
    assert not sun_moved(sun, _turned(0.5 * SUN_TOLERANCE))
    assert sun_moved(sun, _turned(2.0 * SUN_TOLERANCE))
    assert sun_moved(None, sun)


def test_a_cloud_shadow_map_wants_a_new_draw_only_past_the_tolerance():
    """_Map.wanted on a map drawn for the overhead sun (no buffer made)."""
    cloud_map = object.__new__(_Map)
    cloud_map.size, cloud_map.spacing = 1024, 50.0
    up = np.array([0.0, 0.0, 1.0])
    cloud_map.drawn = (up, _turned(0.0), 8, 0)
    assert not cloud_map.wanted(up, _turned(0.5 * SUN_TOLERANCE), 8, 0, 6e5)
    assert cloud_map.wanted(up, _turned(2.0 * SUN_TOLERANCE), 8, 0, 6e5)
