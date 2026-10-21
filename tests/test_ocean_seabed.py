"""
Unit tests for the ocean's seabed map (world_builder.ocean.seabed): its bakes
around the camera, and their mapping into a plane that moves.

A bare NodePath carries the shader inputs, so no window is needed. The bakes
run on the map's worker thread; :func:`_settle` waits for them.
"""

import math
import time

import numpy as np
import pytest
from panda3d.core import NodePath

import world_builder.ocean.seabed as seabed
from world_builder.ocean.seabed import DEEP_SEA_M, Cascade, SeabedMap

CASCADES = (Cascade(1000.0, 64), Cascade(8000.0, 64))


def _beach(points: np.ndarray, texel: float) -> np.ndarray:
    """A seabed rising 1 m every 100 m to the world's east (x), dry past 0."""
    return points[:, 0] * 0.01 - 5.0


def _settle(seabed_map: SeabedMap, camera_xy=(0.0, 0.0), timeout: float = 10.0):
    """Update until every cascade is baked and nothing is pending."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        seabed_map.update(camera_xy)
        if not seabed_map._pending and all(
            b is not None and b.heights is not None for b in seabed_map._bakes
        ):
            seabed_map.update(camera_xy)  # map the last one in
            return
        time.sleep(0.01)
    raise AssertionError("seabed bakes did not finish")


def _count(node: NodePath) -> int:
    """The cascades the shader reads. The ShaderInput must outlive the read:
    a temporary's vector dangles."""
    value = node.get_shader_input("uSeabedCount")
    return int(value.get_vector()[0])


@pytest.fixture
def made(monkeypatch):
    monkeypatch.setattr(seabed, "MIN_INTERVAL_S", 0.0)
    node = NodePath("ocean")
    seabed_map = SeabedMap(node, CASCADES)
    yield seabed_map, node
    seabed_map.destroy()


def test_no_sampler_is_deep_sea(made):
    seabed_map, node = made
    seabed_map.update((0.0, 0.0))
    assert _count(node) == 0
    assert np.all(seabed_map.heights_at([[0.0, 0.0], [300.0, 0.0]]) == DEEP_SEA_M)


def test_bakes_read_back_the_sampler(made):
    seabed_map, node = made
    seabed_map.set_sampler(_beach)
    _settle(seabed_map)
    assert _count(node) == 2
    xy = np.array([[0.0, 0.0], [-300.0, 120.0], [2500.0, -900.0]])
    world = np.column_stack([xy, np.zeros(len(xy))])
    # Nearest texel: within half a texel of slope (the near one is 15.6 m,
    # the far one 125 m).
    tolerance = np.array([0.08, 0.08, 0.63])
    assert np.all(np.abs(seabed_map.heights_at(xy) - _beach(world, 0.0)) <= tolerance)
    # Beyond the widest cascade, the sea is deep.
    assert seabed_map.heights_at([[9000.0, 0.0]])[0] == DEEP_SEA_M


def test_a_bake_stays_on_the_ground_as_the_frame_moves(made):
    """The plane turns and shifts (a tangent plane following the camera over
    a planet): a bake must keep reading the same ground points until rebaked."""
    seabed_map, node = made
    seabed_map.set_sampler(_beach)
    _settle(seabed_map)
    angle = math.radians(25.0)
    east = np.array([math.cos(angle), math.sin(angle), 0.0])
    north = np.array([-math.sin(angle), math.cos(angle), 0.0])
    origin = np.array([40.0, -25.0, 0.0])
    seabed_map.set_frame(origin, east, north)
    seabed_map.update((0.0, 0.0))  # remaps without baking (camera still central)
    xy = np.array([[100.0, 50.0], [-200.0, -150.0]])
    world = origin + xy[:, :1] * east + xy[:, 1:] * north
    assert np.all(np.abs(seabed_map.heights_at(xy) - _beach(world, 0.0)) <= 0.12)


def test_moving_away_rebakes_around_the_camera(made):
    seabed_map, _ = made
    seabed_map.set_sampler(_beach)
    _settle(seabed_map)
    first = seabed_map._bakes[0]
    # A third of the near cascade away: past the rebake fraction.
    _settle(seabed_map, camera_xy=(340.0, 0.0))
    second = seabed_map._bakes[0]
    assert second is not first
    centre = second.x0 + CASCADES[0].extent / 2.0
    assert abs(centre - 340.0) <= CASCADES[0].texel
    # Bake corners sit on the texel lattice, so ground points repeat.
    assert second.x0 / CASCADES[0].texel == pytest.approx(
        round(second.x0 / CASCADES[0].texel)
    )


def test_a_failing_sampler_keeps_the_ocean_going(made, capsys):
    seabed_map, node = made

    def broken(points, texel):
        raise RuntimeError("no terrain")

    seabed_map.set_sampler(broken)
    for _ in range(50):
        seabed_map.update((0.0, 0.0))
        time.sleep(0.01)
    assert _count(node) == 0
    assert "seabed bake failed" in capsys.readouterr().out


def test_at_most_three_cascades(made):
    seabed_map, _ = made
    with pytest.raises(ValueError):
        seabed_map.set_cascades([Cascade(1000.0, 8)] * 4)
