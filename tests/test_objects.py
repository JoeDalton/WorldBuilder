"""
The models' shader (world_builder.objects): the sunlight it lights a model
by is the CPU's (the air, the eclipses, an airless body's own horizon, the
mountains in the body's own axes), and the whole shader lights and
tonemaps as the terrain does.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from panda3d.core import (
    AmbientLight,
    CardMaker,
    Geom,
    GeomNode,
    GeomPoints,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    OmniBoundingVolume,
    Shader,
    ShaderAttrib,
    Texture,
)

from world_builder.atmosphere import AtmosphereParams, Atmospheres
from world_builder.clouds.weather import sample_cube
from world_builder.lights import TrackedLights
from world_builder.objects import SHADER_PATH, ModelLighting, model_shader
from world_builder.ocean import aces_tonemap
from world_builder.shadows import SUN_ANGULAR_RADIUS, Shadows
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.shadow_map import terrain_visibility

SRC = Path(__file__).parents[1] / "src/world_builder"
R = 600e3


def _rotation(axis, angle):
    """World from planet frame: *angle* radians about *axis*."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


# ── binding ────────────────────────────────────────────────────────────────────


def test_tracked_lights_put_a_shaderless_node_back():
    lights = TrackedLights(NodePath("render"))
    node = lights.render.attach_new_node("ship")
    lights.track(node, shader=True)
    assert lights.shaded == [True]
    node.set_shader(model_shader())  # what the owner applies
    lights.untrack(node)
    assert not node.has_attrib(ShaderAttrib)


# ── read back ──────────────────────────────────────────────────────────────────


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


SIZE = 32

_ENCODE = """#version 330
flat in float vValue;
out vec4 fragColor;
void main() {
    float v = floor(clamp(vValue, 0.0, 1.0) * 16711680.0 + 0.5);
    float r = floor(v / 65536.0);
    float g = floor((v - r * 65536.0) / 256.0);
    fragColor = vec4(r, g, v - r * 65536.0 - g * 256.0, 255.0) / 255.0;
}
"""


def _sunlight_points(window, tmp_path, lighting, rel, channels):
    """modelSunlight at each point *rel* (from the camera, render frame),
    channel *channels[i]* of it, read back (the sun's colour must be 1)."""
    vdata = GeomVertexData("points", GeomVertexFormat.get_v3n3(), Geom.UH_static)
    position = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    for p, c in zip(rel.astype(np.float32), channels):
        position.add_data3f(*(float(v) for v in p))
        normal.add_data3f(float(c), 0.0, 0.0)
    prim = GeomPoints(Geom.UH_static)
    prim.add_next_vertices(len(rel))
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("model_sunlight_points")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    vertex = tmp_path / "sunlight.vert"
    vertex.write_text(
        f"""#version 330
#pragma include "{SRC}/atmosphere/shaders/atmosphere.glsl"
#pragma include "{SRC}/clouds/shaders/cloud_field.glsl"
#pragma include "{SHADER_PATH}/model_light.glsl"
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
uniform vec3 uSunDir;
flat out float vValue;
void main() {{
    vValue = modelSunlight(p3d_Vertex.xyz, uSunDir, 0.0)[int(p3d_Normal.x + 0.5)];
    int x = gl_VertexID % {SIZE}, y = gl_VertexID / {SIZE};
    gl_Position = vec4((vec2(x, y) + 0.5) / {SIZE}.0 * 2.0 - 1.0, 0.0, 1.0);
}}
"""
    )
    fragment = tmp_path / "sunlight.frag"
    fragment.write_text(_ENCODE)
    shader = Shader.load(Shader.SL_GLSL, vertex=str(vertex), fragment=str(fragment))
    root = NodePath("sunlight_check")
    points = root.attach_new_node(node)
    points.set_attrib(lighting.inputs.get_attrib(ShaderAttrib).set_shader(shader))
    texture = Texture("sunlight_check")
    buffer = window.win.make_texture_buffer("sunlight_check", SIZE, SIZE, texture, True)
    try:
        camera = window.makeCamera(buffer)
        camera.node().set_scene(root)
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGB").get_data(), np.uint8)
        rgb = ram.reshape(SIZE, SIZE, 3).astype(np.int64)
        got = (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]) / 16711680.0
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()
    return got.ravel()[: len(rel)]


def _cube_texture(cube):
    """A ceiling cube as ShadowMap uploads it: half floats, km."""
    texture = Texture("ceiling")
    texture.setup_cube_map(cube.shape[1], Texture.T_half_float, Texture.F_rg16)
    texture.set_ram_image(np.ascontiguousarray(cube / 1e3, dtype=np.float16).tobytes())
    return texture


def test_the_shader_model_sunlight_is_the_cpus(window, tmp_path):
    """Over a planet with air, near its terminator, turned, in a moon's
    penumbra and under a ceiling that grows toward its own +x: the air's
    reddening, the eclipse and the mountains' term, in the body's own axes,
    are the CPU's."""
    up = np.array([0.3, -0.5, 0.8])
    up /= np.linalg.norm(up)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    elevation = math.radians(5.0)
    sun = math.sin(elevation) * up + math.cos(elevation) * east
    centre = np.array([1e6, 2e6, -3e5])
    camera = centre + up * (R + 1000.0)
    moon_radius = 50e3  # its penumbra ~100 km across
    moon_at = centre + up * R + sun * (moon_radius / math.sin(SUN_ANGULAR_RADIUS))

    atmospheres = Atmospheres()
    air = atmospheres.add(R, AtmosphereParams(), centre)
    shadows = Shadows()
    planet = shadows.add(R, centre, reach=R + 60e3)
    shadows.add(moon_radius, moon_at)
    shadows.set_atmosphere(air, planet)
    shadows.set_sun(sun)
    atmospheres.update(camera)
    shadows.update(camera)

    rotation = _rotation((0.2, 1.0, -0.4), 1.1)
    size = 32
    dirs = cube_directions(size).reshape(6, size, size, 3)
    cube = np.stack(
        [(dirs[..., 0] + 1.0) * 2000.0, np.full(dirs.shape[:-1], 20e3)],
        axis=-1,
    )
    lighting = ModelLighting(atmosphere=air, terrain=SimpleNamespace(rotation=rotation))
    atmospheres.bind(lighting.inputs)
    shadows.bind(lighting.inputs, planet)
    lighting.inputs.set_shader_inputs(
        uTerrainShadow=_cube_texture(cube), uTerrainShadowOn=1
    )
    lighting.set_sun(sun, (1.0, 1.0, 1.0))
    lighting.update((0.0, 0.0, 0.0))

    rng = np.random.default_rng(4)
    count = SIZE * SIZE // 3
    north = np.cross(up, east)
    spread = rng.uniform(-200e3, 200e3, (count, 2))
    ground = up * R + spread[:, :1] * east + spread[:, 1:] * north
    ground /= np.linalg.norm(ground, axis=1, keepdims=True)
    altitude = rng.uniform(0.0, 4000.0, count)
    points = centre + ground * (R + altitude)[:, None]
    planet_dirs = (points - centre) @ rotation  # rows: R^T (p - c)
    planet_dirs /= np.linalg.norm(planet_dirs, axis=1, keepdims=True)
    stored = (cube / 1e3).astype(np.float16).astype(np.float64) * 1e3
    ceiling, distance = np.moveaxis(sample_cube(stored, planet_dirs), -1, 0)
    terrain = terrain_visibility(ceiling, distance, altitude, SUN_ANGULAR_RADIUS)
    eclipse = shadows.eclipse(points, sun, exclude=planet)
    air_light = atmospheres.transmittance_to_sun(air, points, sun)
    expected = air_light * (eclipse * terrain)[:, None]
    assert eclipse.min() < 0.3 and eclipse.max() == 1.0
    assert terrain.min() < 0.1 and terrain.max() == 1.0
    assert air_light[:, 2].min() < 0.5 * air_light[:, 2].max()  # reddened low

    rel = np.repeat(points - camera, 3, axis=0)
    channels = np.tile([0, 1, 2], count)
    got = _sunlight_points(window, tmp_path, lighting, rel, channels)
    np.testing.assert_allclose(got.reshape(count, 3), expected, atol=4e-3)


def test_an_airless_bodys_own_horizon(window, tmp_path):
    """On an airless moon the sun sets at its horizon with the sun's
    penumbra (Shadows.own_shadow), and a crater's floor under its sea-level
    radius is lit by day, not inside the body."""
    shadows = Shadows()
    moon = shadows.add(170e3, (0.0, 0.0, 0.0))
    sun = np.array([1.0, 0.0, 0.0])
    shadows.set_sun(sun)
    camera = np.array([0.0, 0.0, 170e3 + 500.0])
    shadows.update(camera)
    lighting = ModelLighting(terrain=SimpleNamespace(rotation=np.eye(3)))
    shadows.bind(lighting.inputs, moon)
    lighting.set_sun(sun, (1.0, 1.0, 1.0))
    lighting.update((0.0, 0.0, 0.0))

    rng = np.random.default_rng(5)
    count = SIZE * SIZE
    # Around the terminator (the sun on the horizon at the pole): a few
    # sun widths either side, a few km up or down.
    angle = rng.uniform(-0.03, 0.03, count)
    altitude = rng.uniform(-3000.0, 3000.0, count)
    points = (170e3 + altitude)[:, None] * np.stack(
        [np.sin(angle), np.zeros(count), np.cos(angle)], axis=1
    )
    expected = shadows.own_shadow(points, sun, moon)
    assert expected.min() == 0.0 and expected.max() == 1.0
    assert ((expected > 0.05) & (expected < 0.95)).sum() > 20  # penumbrae
    # Lit by day, from under the sea-level radius too.
    day = angle > 0.02
    assert np.all(expected[day] == 1.0) and np.any(altitude[day] < -1000.0)

    got = _sunlight_points(window, tmp_path, lighting, points - camera, np.zeros(count))
    np.testing.assert_allclose(got, expected, atol=3e-3)


def test_the_model_shader_lights_and_tonemaps_as_the_terrain(window):
    """A card under the sun in open space: its colour decoded, Lambert
    under the sun plus the sky light (its ambient light, less the light
    scale), exposed and tonemapped as the terrain is; no channel clipped."""
    size = 16
    sun = np.array([0.0, -1.0, 1.0]) / math.sqrt(2.0)  # 45 degrees off the card
    lighting = ModelLighting(exposure=0.9, light_scale=0.5)
    lighting.set_sun(sun, (2.6, 2.4, 2.2))
    lighting.update((0.0, 0.0, 0.0))

    root = NodePath("model_check")
    maker = CardMaker("card")
    maker.set_frame(-1.0, 1.0, -1.0, 1.0)
    card = root.attach_new_node(maker.generate())  # faces -y, toward the camera
    card.set_pos(0.0, 2.0, 0.0)
    card.set_color(0.8, 0.6, 0.4, 1.0)
    sky = root.attach_new_node(AmbientLight("sky"))
    sky.node().set_color((0.15, 0.2, 0.3, 1.0))  # the world's 0.3, 0.4, 0.6 at 0.5
    card.set_light(sky)
    lighting.apply(card)

    texture = Texture("model_check")
    buffer = window.win.make_texture_buffer("model_check", size, size, texture, True)
    try:
        # At the scene's origin: under the app's camera, wherever earlier
        # tests left it, the card is out of view.
        camera = window.makeCamera(buffer)
        camera.reparent_to(root)
        camera.set_pos_hpr(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        camera.node().set_scene(root)
        camera.node().get_lens().set_fov(60.0)
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGB").get_data(), np.uint8)
        got = ram.reshape(size, size, 3)[size // 2, size // 2] / 255.0
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()

    albedo = np.array([0.8, 0.6, 0.4]) ** 2.2
    normal = np.array([0.0, -1.0, 0.0])
    lambert = float(normal @ sun)
    linear = albedo * (np.array([2.6, 2.4, 2.2]) * lambert + [0.3, 0.4, 0.6])
    expected = aces_tonemap(linear, 0.9)
    assert linear.max() > 1.0  # what Panda's lighting would clip
    np.testing.assert_allclose(got, expected, atol=2.5 / 255.0)
