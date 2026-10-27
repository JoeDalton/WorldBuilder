"""
World shadows (world_builder.shadows): the eclipse's disc coverage against
its limits and a Monte-Carlo estimate, the eclipse of spherical bodies
against geometry, the default inputs, and shadows.glsl read back from the GPU
against the CPU twin.
"""

import math

import numpy as np
import pytest
from panda3d.core import NodePath

from world_builder.shadows import (
    SHADER_PATH,
    SUN_ANGULAR_RADIUS,
    Shadows,
    bind_defaults,
    disc_coverage,
)

SUN = SUN_ANGULAR_RADIUS


# ── disc coverage ──────────────────────────────────────────────────────────────


def test_coverage_limits():
    """Apart: nothing covered; a wider disc centred: all; a narrower one
    inside: its area's share; touching from outside or within: continuous."""
    assert disc_coverage(SUN, SUN, 2.0 * SUN) == 0.0
    assert disc_coverage(SUN, 0.5 * SUN, 3.0 * SUN) == 0.0
    assert disc_coverage(SUN, 1.2 * SUN, 0.0) == 1.0
    assert disc_coverage(SUN, 1.2 * SUN, 0.1 * SUN) == 1.0
    assert disc_coverage(SUN, 0.5 * SUN, 0.2 * SUN) == pytest.approx(0.25)
    eps = 1e-9
    assert disc_coverage(SUN, 0.5 * SUN, 1.5 * SUN - eps) == pytest.approx(0, abs=1e-6)
    assert disc_coverage(SUN, 0.5 * SUN, 0.5 * SUN + eps) == pytest.approx(0.25)
    assert disc_coverage(SUN, 2.0 * SUN, SUN + eps) == pytest.approx(1.0)
    # Equal discs half overlapped (centres a radius apart): the lens formula.
    lens = (2.0 * math.pi / 3.0 - math.sqrt(3.0) / 2.0) / math.pi
    assert disc_coverage(SUN, SUN, SUN) == pytest.approx(lens)


def test_coverage_is_the_overlap_over_the_suns_area():
    """The overlap is one area: covered share times the sun's area is the
    same with the two discs swapped; and it only falls as they part."""
    s = np.linspace(0.0, 3.0 * SUN, 200)
    for body in (0.3 * SUN, SUN, 1.7 * SUN):
        forward = disc_coverage(SUN, body, s) * SUN**2
        swapped = disc_coverage(body, SUN, s) * body**2
        np.testing.assert_allclose(forward, swapped, atol=1e-12)
        assert np.all(np.diff(disc_coverage(SUN, body, s)) <= 1e-12)


def test_coverage_against_monte_carlo():
    rng = np.random.default_rng(3)
    n = 400_000
    radius = SUN * np.sqrt(rng.random(n))
    angle = 2.0 * math.pi * rng.random(n)
    x, y = radius * np.cos(angle), radius * np.sin(angle)  # uniform on the sun
    for body, separation in [(0.8, 0.9), (1.0, 1.0), (1.3, 0.6), (0.4, 0.7)]:
        hidden = np.mean((x - separation * SUN) ** 2 + y**2 < (body * SUN) ** 2)
        got = disc_coverage(SUN, body * SUN, separation * SUN)
        assert got == pytest.approx(hidden, abs=4e-3)


# ── eclipses ───────────────────────────────────────────────────────────────────

R = 200e3  # the occluder
D = 21_000e3  # its distance from the points
L = np.array([0.0, 0.0, 1.0])  # toward the sun


def _occluder() -> Shadows:
    shadows = Shadows()
    shadows.add(R, D * L)
    return shadows


def test_eclipse_by_geometry():
    """Behind the body: as much of the sun as its disc hides from there;
    off its axis by the sum of the radii: none; on the sun's side: none."""
    shadows = _occluder()
    body = math.asin(R / D)
    assert shadows.eclipse([0.0, 0.0, 0.0], L)[()] == pytest.approx(
        1.0 - disc_coverage(SUN, body, 0.0)
    )
    for angle in (0.3, 0.7, 1.0):  # off the axis, in sun radii
        side = D * math.tan(angle * SUN)
        point = np.array([side, 0.0, 0.0])
        separation = math.atan2(side, D)
        distance = math.hypot(side, D)
        expected = 1.0 - disc_coverage(SUN, math.asin(R / distance), separation)
        assert shadows.eclipse(point, L)[()] == pytest.approx(expected, abs=1e-9)
    clear = D * math.tan(SUN + body) * 1.001
    assert shadows.eclipse([clear, 0.0, 0.0], L)[()] == 1.0
    assert shadows.eclipse([0.0, 0.0, 2.0 * D], L)[()] == 1.0
    assert shadows.eclipse([0.0, 0.0, D], L)[()] == 0.0  # inside it


def test_eclipse_is_vectorised_multiplies_and_skips_its_own_body():
    shadows = _occluder()
    shadows.add(R, D * L + [0.0, 3.0 * R, 0.0])
    points = np.array([[0.0, 0.0, 0.0], [0.0, 3.0 * R, 0.0], [5e6, 0.0, 0.0]])
    both = shadows.eclipse(points, L)
    assert both.shape == (3,)
    np.testing.assert_allclose(
        both, shadows.eclipse(points, L, exclude=0) * shadows.eclipse(points, L, 1)
    )
    assert both[2] == 1.0
    assert shadows.eclipse(points[:1], L, exclude=0)[0] > both[0]


def test_defaults_bind_every_input():
    node = NodePath("n")
    bind_defaults(node)
    for name in (
        "uShadowBodies",
        "uShadowCount",
        "uSunAngularRadius",
        "uShadowSelf",
        "uAtmoBody",
    ):
        # A missing input comes back blank: value type 0 (M_invalid).
        assert node.get_shader_input(name).get_value_type() != 0
    assert NodePath("unbound").get_shader_input("uShadowCount").get_value_type() == 0


def test_inputs_are_shared_and_rewritten_in_place():
    """The bodies are arrays bound once: an update reaches every node bound."""
    shadows = _occluder()
    node = NodePath("bound")
    shadows.bind(node, 0)
    bound = node.get_shader_input("uShadowBodies")
    shadows.update([1.0, 2.0, 3.0])
    assert node.get_shader_input("uShadowBodies") == bound
    assert tuple(shadows._spheres[0]) == pytest.approx((-1.0, -2.0, D - 3.0, R))


def test_only_shadows_that_can_reach_a_body_are_looked_at():
    """A body's casters are the bodies whose penumbra meets its reach: never
    one missing where a point within the reach is eclipsed, and none where
    the sun is elsewhere."""
    rng = np.random.default_rng(4)
    seen = 0
    for _ in range(40):
        sun = rng.normal(size=3)
        sun /= np.linalg.norm(sun)
        shadows = Shadows()
        reach = 260e3
        planet = shadows.add(200e3, (0.0, 0.0, 0.0), reach=reach)
        moon_at = (sun + 0.03 * rng.normal(size=3)) * rng.uniform(2e6, 3e7)
        moon = shadows.add(rng.uniform(5e4, 3e5), moon_at)
        shadows.set_sun(sun)
        points = rng.normal(size=(4000, 3))
        points *= (
            reach * rng.random(4000) ** (1 / 3) / np.linalg.norm(points, axis=1)
        )[:, None]
        eclipsed = np.any(shadows.eclipse(points, sun, exclude=planet) < 1.0)
        if eclipsed:
            seen += 1
            assert moon in shadows.casters(planet)
    assert seen >= 10
    shadows.set_sun(-sun)  # the moon behind the planet: no shadow on it
    assert shadows.casters(planet) == []


# ── shadows.glsl read back ─────────────────────────────────────────────────────


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


def test_the_shader_eclipse_is_the_cpus(window):
    """eclipseVisibility (float32, relative to the camera) matches
    Shadows.eclipse (float64, world frame) across a penumbra, with the
    skipped body read from uAtmoBody (an int array)."""
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomPoints,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        OmniBoundingVolume,
        Shader,
        Texture,
    )

    size = 32
    sun = np.array([0.3, -0.2, 1.0])
    sun /= np.linalg.norm(sun)
    camera = np.array([4.0e6, -1.0e6, 2.0e5])
    shadows = Shadows()
    planet = shadows.add(600e3, camera + [0.0, 0.0, -601e3], reach=660e3)
    shadows.add(R, camera + sun * D)  # the moon, over the camera's ground
    shadows.set_atmosphere(1, planet)
    shadows.set_sun(sun)
    shadows.update(camera)
    assert shadows.casters(planet) == [1]
    # Points on the planet's ground (around the camera) across the moon's
    # penumbra (~400 km wide), and a few in full sun.
    rng = np.random.default_rng(9)
    offsets = rng.uniform(-500e3, 500e3, size=(size * size, 3)) * [1.0, 1.0, 0.0]
    points = camera + offsets
    expected = shadows.eclipse(points, sun, exclude=planet)
    assert expected.min() < 0.1 and expected.max() == 1.0  # umbra to full sun

    vdata = GeomVertexData("points", GeomVertexFormat.get_v3(), Geom.UH_static)
    writer = GeomVertexWriter(vdata, "vertex")
    for x, y, z in (points - camera).astype(np.float32):
        writer.add_data3f(float(x), float(y), float(z))
    prim = GeomPoints(Geom.UH_static)
    prim.add_next_vertices(len(points))
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("eclipse_points")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    include = (SHADER_PATH / "shadows.glsl").read_text().replace("#pragma once", "")
    vertex = f"""#version 330
in vec4 p3d_Vertex;
uniform vec3 uSun;
uniform int uAtmoBody[2];
flat out float vVisible;
{include}
void main() {{
    vVisible = eclipseVisibility(p3d_Vertex.xyz, uSun, uAtmoBody[1]);
    int x = gl_VertexID % {size}, y = gl_VertexID / {size};
    gl_Position = vec4((vec2(x, y) + 0.5) / {size}.0 * 2.0 - 1.0, 0.0, 1.0);
}}
"""
    fragment = """#version 330
flat in float vVisible;
out vec4 fragColor;
void main() {
    float v = floor(clamp(vVisible, 0.0, 1.0) * 16711680.0 + 0.5);
    float r = floor(v / 65536.0);
    float g = floor((v - r * 65536.0) / 256.0);
    fragColor = vec4(r, g, v - r * 65536.0 - g * 256.0, 255.0) / 255.0;
}
"""
    root = NodePath("eclipse_check")
    points_np = root.attach_new_node(node)
    points_np.set_shader(Shader.make(Shader.SL_GLSL, vertex, fragment))
    bind_defaults(points_np)
    shadows.bind(points_np)
    points_np.set_shader_input("uSun", tuple(float(v) for v in sun))
    texture = Texture("eclipse_check")
    buffer = window.win.make_texture_buffer("eclipse_check", size, size, texture, True)
    try:
        camera_np = window.makeCamera(buffer)
        camera_np.node().set_scene(root)
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGB").get_data(), np.uint8)
        rgb = ram.reshape(size, size, 3).astype(np.int64)
        got = (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]) / 16711680.0
        # float32 positions and angles: a few metres on a 400 km penumbra.
        np.testing.assert_allclose(got.ravel(), expected, atol=2e-3)
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()
