"""
Panda3D lights per node (world_builder.lights): a tracked node's own lights
replace the world's on it and its children, untracking puts back what it
had, removed nodes are forgotten, and the colours and direction are set in
order.
"""

import numpy as np
import pytest
from panda3d.core import AmbientLight, DirectionalLight, LightAttrib, NodePath

from world_builder.lights import TrackedLights


def _on_lights(node: NodePath) -> list[str]:
    """The lights that reach a node, by name."""
    attrib = node.get_net_state().get_attrib(LightAttrib)
    if attrib is None:
        return []
    count = attrib.get_num_on_lights()
    return sorted(attrib.get_on_light(i).name for i in range(count))


@pytest.fixture
def scene():
    render = NodePath("render")
    for light in (DirectionalLight("world_sun"), AmbientLight("world_ambient")):
        render.set_light(render.attach_new_node(light))
    return render


def test_a_tracked_node_has_its_own_lights_only(scene):
    ship = scene.attach_new_node("ship")
    hull = ship.attach_new_node("hull")
    other = scene.attach_new_node("other")
    lights = TrackedLights(scene)
    lights.track(ship, cast=True)
    assert _on_lights(hull) == ["ship_ambient", "ship_sun"]
    assert _on_lights(other) == ["world_ambient", "world_sun"]
    assert lights.casts(ship) and len(lights) == 1
    lights.track(ship)  # again: only the flag changes
    assert not lights.casts(ship) and len(lights) == 1
    lights.untrack(ship)
    assert _on_lights(hull) == ["world_ambient", "world_sun"]
    assert len(lights) == 0
    assert scene.find("**/ship_sun").is_empty()


def test_untracking_puts_back_the_nodes_own_lights(scene):
    lamp = scene.attach_new_node(DirectionalLight("lamp"))
    ship = scene.attach_new_node("ship")
    ship.set_light(lamp)
    lights = TrackedLights(scene)
    lights.track(ship)
    assert "lamp" not in _on_lights(ship)
    lights.untrack(ship)
    assert _on_lights(ship) == ["lamp", "world_ambient", "world_sun"]


def test_positions_and_colours_in_order_and_removed_nodes_dropped(scene):
    lights = TrackedLights(scene)
    a = scene.attach_new_node("a")
    b = scene.attach_new_node("b")
    a.set_pos(1.0, 2.0, 3.0)
    b.set_pos(-4.0, 5.0, 6.0)
    lights.track(a)
    lights.track(b)
    np.testing.assert_allclose(lights.positions(), [[1, 2, 3], [-4, 5, 6]])
    lights.apply([[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]], [[0.1] * 3, [0.2] * 3], (0, 0, 1))
    sun = scene.find("**/a_sun").node()
    assert tuple(sun.get_color())[:3] == pytest.approx((1.0, 0.5, 0.25))
    assert tuple(sun.get_direction()) == pytest.approx((0.0, 0.0, -1.0))
    sky = scene.find("**/b_ambient").node()
    assert tuple(sky.get_color())[:3] == pytest.approx((0.2, 0.2, 0.2))
    a.remove_node()
    assert lights.nodes == [b]
    np.testing.assert_allclose(lights.positions(), [[-4, 5, 6]])
    assert scene.find("**/a_sun").is_empty()
    lights.clear()
    assert len(lights) == 0 and scene.find("**/b_sun").is_empty()
