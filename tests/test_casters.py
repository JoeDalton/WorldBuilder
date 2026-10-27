"""
The objects' shadow map (world_builder.shadows.casters): fitted to the
casters down the sun's rays, and read by shadows.glsl's casterShadow as
their geometric shadow.
"""

from pathlib import Path

import numpy as np
import pytest
from panda3d.core import (
    Geom,
    GeomNode,
    GeomPoints,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    OmniBoundingVolume,
    Shader,
    Texture,
)

from world_builder.depth import CASTER_BIT
from world_builder.shadows import bind_defaults
from world_builder.shadows.casters import CASTER_CAMERA_MASK, CasterMap

SHADOWS = Path(__file__).parents[1] / "src/world_builder/shadows/shaders/shadows.glsl"
SIZE = 32


@pytest.fixture(scope="module")
def window(showbase):
    """The session's ShowBase with an offscreen window; skipped where none
    can be opened."""
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


def _ball(window, parent, radius, centre):
    ball = window.loader.load_model("models/smiley")
    ball.reparent_to(parent)
    ball.set_scale(radius)  # a unit sphere
    ball.set_pos(*centre)
    return ball


def _shows_through(node) -> bool:
    """Whether a node shows to the map's camera whatever hides its parents
    (``show_through``: its own control and show bits set)."""
    control = node.node().get_draw_control_mask().get_bit(CASTER_BIT)
    return control and node.node().get_draw_show_mask().get_bit(CASTER_BIT)


def test_only_the_casters_show_to_the_maps_camera(window):
    scene = NodePath("scene")
    casters = CasterMap(window.win, scene, 0)
    try:
        ship = _ball(window, scene, 10.0, (0.0, 100.0, 0.0))
        rock = _ball(window, scene, 10.0, (50.0, 100.0, 0.0))
        assert CASTER_CAMERA_MASK.get_bit(CASTER_BIT)
        assert scene.is_hidden(CASTER_CAMERA_MASK)
        casters.update([ship], (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        assert _shows_through(ship) and not _shows_through(rock)
        casters.update([rock], (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        assert _shows_through(rock) and not _shows_through(ship)
        assert casters._on[0] == 0  # no map: nothing to read
    finally:
        casters.destroy()
    assert not scene.is_hidden(CASTER_CAMERA_MASK)


def test_a_caster_whose_shadow_is_too_small_to_see_is_left_out(window):
    """A 10 m ball 50 km away casts a shadow under two pixels: not drawn,
    no map; the camera in its shadow, far under it, sees it."""
    scene = NodePath("scene")
    casters = CasterMap(window.win, scene, 256)
    try:
        ball = _ball(window, scene, 10.0, (0.0, 50e3, 0.0))
        sun = (0.0, 0.0, 1.0)
        pixel = 1e-3  # 1600 px over 90 degrees
        casters.update([ball], sun, (0.0, 0.0, 0.0), pixel)
        assert casters.casters == [] and casters._on[0] == 0
        assert not _shows_through(ball)
        casters.update([ball], sun, (0.0, 50e3, -20e3), pixel)  # under it
        assert casters.casters == [ball] and casters._on[0] == 1
        casters.update([ball], sun, (0.0, 0.0, 0.0))  # no pixel angle: all
        assert casters.casters == [ball]
    finally:
        casters.destroy()
    assert not scene.is_hidden(CASTER_CAMERA_MASK)


def _read_back(window, casters, rel, normals, sun):
    """casterShadow at each point (from the camera), its normal given."""
    vdata = GeomVertexData("points", GeomVertexFormat.get_v3n3(), Geom.UH_static)
    position = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    for p, n in zip(rel.astype(np.float32), normals.astype(np.float32)):
        position.add_data3f(*(float(v) for v in p))
        normal.add_data3f(*(float(v) for v in n))
    prim = GeomPoints(Geom.UH_static)
    prim.add_next_vertices(len(rel))
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("caster_points")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    include = SHADOWS.read_text().replace("#pragma once", "")
    vertex = f"""#version 330
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
uniform vec3 uSun;
flat out float vValue;
{include}
void main() {{
    vValue = casterShadow(p3d_Vertex.xyz, p3d_Normal, uSun);
    int x = gl_VertexID % {SIZE}, y = gl_VertexID / {SIZE};
    gl_Position = vec4((vec2(x, y) + 0.5) / {SIZE}.0 * 2.0 - 1.0, 0.0, 1.0);
}}
"""
    fragment = """#version 330
flat in float vValue;
out vec4 fragColor;
void main() {
    fragColor = vec4(vValue, 0.0, 0.0, 1.0);
}
"""
    root = NodePath("caster_check")
    points = root.attach_new_node(node)
    points.set_shader(Shader.make(Shader.SL_GLSL, vertex, fragment))
    bind_defaults(points)
    casters.bind(points)
    points.set_shader_input("uSun", tuple(float(v) for v in sun))
    texture = Texture("caster_check")
    buffer = window.win.make_texture_buffer("caster_check", SIZE, SIZE, texture, True)
    try:
        camera = window.makeCamera(buffer)
        camera.node().set_scene(root)
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGB").get_data(), np.uint8)
        got = ram.reshape(SIZE, SIZE, 3)[..., 0] / 255.0
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()
    return got.ravel()[: len(rel)]


def test_the_map_is_the_casters_geometric_shadow(window):
    """A ball over a plane: the plane is in shadow where its ray toward the
    sun crosses the ball, lit elsewhere, and outside the map's square; a
    point between the ball and the sun is lit; the ball's lit side too."""
    scene = NodePath("scene")
    casters = CasterMap(window.win, scene, 512)
    try:
        assert casters.buffer is not None
        centre = np.array([30.0, -20.0, 40.0])
        radius = 10.0
        ball = _ball(window, scene, radius, centre)
        sun = np.array([0.3, -0.2, 1.0])
        sun /= np.linalg.norm(sun)
        camera = np.array([5.0, -40.0, 3.0])
        casters.update([ball], sun, camera)
        assert casters._on[0] == 1
        window.graphicsEngine.render_frame()

        rng = np.random.default_rng(7)
        count = SIZE * SIZE
        # The plane z = 0 around the ball's shadow (centered at the foot of
        # its ray), and a few kinds of points apart.
        foot = centre - sun * (centre[2] / sun[2])
        plane = foot + np.c_[rng.uniform(-18.0, 18.0, (count, 2)), np.zeros(count)]
        normals = np.tile([0.0, 0.0, 1.0], (count, 1))
        # Its distance from the ball's centre, across the sun's ray.
        to_centre = centre - plane
        across = np.linalg.norm(to_centre - np.outer(to_centre @ sun, sun), axis=1)
        expected = (across > radius).astype(float)
        plane[:8] = centre + sun * 40.0 + rng.normal(size=(8, 3))  # over it
        expected[:8] = 1.0
        plane[8:16] = centre + np.array([500.0, 0.0, 0.0])  # off the map
        expected[8:16] = 1.0
        lit = centre + sun * (radius + 0.05)  # on the ball, facing the sun
        plane[16] = lit
        normals[16] = sun
        expected[16] = 1.0
        got = _read_back(window, casters, plane - camera, normals, sun)
        # The mesh is a little inside its sphere, and the edge is a few
        # texels soft.
        clear = np.abs(across - radius) > 0.6
        clear[:17] = True
        assert (expected[clear] == 0.0).sum() > 100
        np.testing.assert_allclose(got[clear], expected[clear], atol=0.02)
    finally:
        casters.destroy()
        scene.remove_node()
