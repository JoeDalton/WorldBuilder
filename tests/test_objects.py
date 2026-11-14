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
    LMatrix4f,
    LVecBase3f,
    Material,
    NodePath,
    OmniBoundingVolume,
    Shader,
    ShaderAttrib,
    Texture,
    TextureStage,
)

from world_builder.atmosphere import AtmosphereParams, Atmospheres
from world_builder.clouds.weather import sample_cube
from world_builder.lights import TrackedLights
from world_builder.objects import (
    LINEAR_TAG,
    PBR,
    PBR_NORMAL,
    PBR_SKY,
    SHADER_PATH,
    SIMPLE,
    ModelLighting,
    ModelSettings,
    mark_linear,
    model_shader,
    unmark_linear,
)
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


_CARD_SUN = (2.6, 2.4, 2.2)
_CARD_SKY = np.array([0.3, 0.4, 0.6])  # the world's, at light scale 0.5


def _card_pixel(window, shading, setup=None, sun=None, mark=False, size=16):
    """A card 2 m ahead of a camera at the origin, facing it (-y), in open
    space under the sun *sun* (toward it) of colour _CARD_SUN and the sky
    light _CARD_SKY, drawn by the model shader at tier *shading*: its centre
    pixel (display colour).

    :param setup: given the card, before it is drawn (its colour, material,
        textures)
    :param mark: mark_linear the card (as World.track_light does)"""
    if sun is None:
        sun = np.array([0.0, -1.0, 1.0]) / math.sqrt(2.0)  # 45 degrees off it
    lighting = ModelLighting(
        exposure=0.9, light_scale=0.5, settings=ModelSettings(shading=shading)
    )
    lighting.set_sun(sun, _CARD_SUN)
    lighting.update((0.0, 0.0, 0.0))

    root = NodePath("model_check")
    maker = CardMaker("card")
    maker.set_frame(-1.0, 1.0, -1.0, 1.0)
    card = root.attach_new_node(maker.generate())  # faces -y, toward the camera
    card.set_pos(0.0, 2.0, 0.0)
    if setup is not None:
        setup(card)
    if mark:
        mark_linear(card)
    sky = root.attach_new_node(AmbientLight("sky"))
    sky.node().set_color((*(_CARD_SKY * 0.5), 1.0))
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
        return ram.reshape(size, size, 3)[size // 2, size // 2] / 255.0
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()
        lighting.destroy()


def _texture(rgb, srgb=False):
    """A 1 x 1 texture of *rgb* (0-1), stored as sRGB or linear."""
    texture = Texture("texel")
    texture.setup_2d_texture(
        1, 1, Texture.T_unsigned_byte, Texture.F_srgb if srgb else Texture.F_rgb
    )
    bgr = bytes(round(v * 255.0) for v in reversed(rgb))
    texture.set_ram_image(bgr)
    return texture


def _pbr_twin(albedo, metallic, roughness, emission=(0.0, 0.0, 0.0)):
    """model_pbr.glsl on the CPU for _card_pixel's card (open space: the
    vertical is the normal, the environment the sky light): its display
    colour."""
    albedo = np.asarray(albedo, dtype=np.float64)
    N = np.array([0.0, -1.0, 0.0])
    # The centre pixel's centre: half a pixel right of and above the axis
    # (16 pixels over 60 degrees).
    off = math.tan(math.radians(30.0)) * 0.5 / 8.0
    V = -np.array([off, 1.0, off]) / math.sqrt(1.0 + 2.0 * off * off)
    L = np.array([0.0, -1.0, 1.0]) / math.sqrt(2.0)
    roughness = min(max(roughness, 0.045), 1.0)
    diffuse = albedo * (1.0 - metallic)
    f0 = 0.04 + (albedo - 0.04) * metallic
    H = (L + V) / np.linalg.norm(L + V)
    NoL, NoV, NoH, VoH = N @ L, N @ V, N @ H, V @ H
    a = roughness * roughness
    a2 = a * a
    D = a2 / (math.pi * (NoH * NoH * (a2 - 1.0) + 1.0) ** 2)
    ggx = NoL * math.sqrt(NoV * NoV * (1 - a2) + a2) + NoV * math.sqrt(
        NoL * NoL * (1 - a2) + a2
    )
    vis = 0.5 / ggx
    F = f0 + (1.0 - f0) * 2.0 ** ((-5.55473 * VoH - 6.98316) * VoH)
    sun = (diffuse + F * D * vis * math.pi) * np.array(_CARD_SUN) * NoL
    c0 = np.array([-1.0, -0.0275, -0.572, 0.022])
    c1 = np.array([1.0, 0.0425, 1.04, -0.04])
    r = roughness * c0 + c1
    a004 = min(r[0] * r[0], 2.0 ** (-9.28 * NoV)) * r[0] + r[1]
    env_brdf = f0 * (-1.04 * a004 + r[2]) + (1.04 * a004 + r[3])
    ambient = diffuse * _CARD_SKY + _CARD_SKY * env_brdf
    return aces_tonemap(sun + ambient + np.asarray(emission), 0.9)


def _pbr_material(base, metallic, roughness, emission=None):
    def setup(card):
        material = Material()
        material.set_base_color((*base, 1.0))
        material.set_metallic(metallic)
        material.set_roughness(roughness)
        if emission is not None:
            material.set_emission((*emission, 1.0))
        card.set_material(material)

    return setup


def test_the_model_shader_lights_and_tonemaps_as_the_terrain(window):
    """A card under the sun in open space, at the simple tier: its colour
    decoded, Lambert under the sun plus the sky light (its ambient light,
    less the light scale), exposed and tonemapped as the terrain is; no
    channel clipped."""
    got = _card_pixel(window, SIMPLE, lambda card: card.set_color(0.8, 0.6, 0.4, 1.0))
    albedo = np.array([0.8, 0.6, 0.4]) ** 2.2
    lambert = 1.0 / math.sqrt(2.0)
    linear = albedo * (np.array(_CARD_SUN) * lambert + _CARD_SKY)
    expected = aces_tonemap(linear, 0.9)
    assert linear.max() > 1.0  # what Panda's lighting would clip
    np.testing.assert_allclose(got, expected, atol=2.5 / 255.0)


@pytest.mark.parametrize("shading", [PBR, PBR_NORMAL, PBR_SKY])
@pytest.mark.parametrize(
    "base, metallic, roughness",
    [
        ((0.6, 0.45, 0.3), 0.0, 0.5),
        ((0.9, 0.6, 0.3), 1.0, 0.35),
        ((0.5, 0.5, 0.5), 0.0, 1.0),
    ],
)
def test_the_pbr_tiers_are_the_ported_brdf(window, shading, base, metallic, roughness):
    """A glTF-like card (a PBR material, its base colour linear) at every
    PBR tier is model_pbr.glsl's BRDF in the world's units: a dielectric,
    a metal (its reflections tinted), a rough one. Without a sky-view table
    the sky tier reflects the even sky, as the others."""
    got = _card_pixel(
        window, shading, _pbr_material(base, metallic, roughness), mark=True
    )
    np.testing.assert_allclose(
        got, _pbr_twin(base, metallic, roughness), atol=3 / 255.0
    )


def test_a_rough_dielectric_looks_alike_at_every_tier(window):
    """The PBR tiers are in the terrain's units: a matte grey card is about
    as bright under them as the simple tier lights it."""
    setup = _pbr_material((0.5, 0.5, 0.5), 0.0, 1.0)
    simple = _card_pixel(window, SIMPLE, setup, mark=True)
    pbr = _card_pixel(window, PBR, setup, mark=True)
    assert np.all(pbr >= simple - 1 / 255.0)
    np.testing.assert_allclose(pbr, simple, atol=6 / 255.0)


def test_a_metal_is_lit_as_a_metal(window):
    """Metal reflects in its own colour and has no diffuse light: at PBR a
    gold card's highlight-free face is darker and redder than the same
    colour as a dielectric."""
    gold = (1.0, 0.75, 0.35)
    metal = _card_pixel(window, PBR, _pbr_material(gold, 1.0, 0.6), mark=True)
    plastic = _card_pixel(window, PBR, _pbr_material(gold, 0.0, 0.6), mark=True)
    assert metal.sum() < plastic.sum()
    assert metal[0] - metal[2] > plastic[0] - plastic[2]  # tinted reflections


def test_an_srgb_texture_is_decoded_once(window):
    """A glTF texture is sRGB (decoded by the GPU): marked linear, it comes
    out as the same colour authored as displayed does, not decoded twice."""
    colour = (0.7, 0.5, 0.3)

    def textured(card):
        card.set_texture(_texture(colour, srgb=True))

    for shading in (SIMPLE, PBR):
        got = _card_pixel(window, shading, textured, mark=True)
        expected = _card_pixel(
            window, shading, lambda card: card.set_color(*colour, 1.0)
        )
        np.testing.assert_allclose(got, expected, atol=3 / 255.0)
        twice = _card_pixel(window, shading, textured)  # not marked
        assert twice.sum() < got.sum() - 0.2


def _normal_map(texel):
    def setup(card):
        stage = TextureStage("normal")
        stage.set_mode(TextureStage.M_normal)
        card.set_texture(stage, _texture(texel))

    return setup


def test_normal_maps(window):
    """At PBR_NORMAL a flat normal map changes nothing; one leaning toward
    the sun (the card has no tangents: the frame from the screen) lights it
    more; PBR ignores it."""
    flat = _card_pixel(window, PBR_NORMAL, _normal_map((0.5, 0.5, 1.0)))
    none = _card_pixel(window, PBR_NORMAL)
    np.testing.assert_allclose(flat, none, atol=2 / 255.0)
    # The card's +z (its texture's +v) is toward the sun: the map's +y.
    toward = (0.5, 0.5 + 0.5 * math.sin(0.6), 0.5 + 0.5 * math.cos(0.6))
    leaning = _card_pixel(window, PBR_NORMAL, _normal_map(toward))
    assert leaning.sum() > none.sum() + 0.05
    ignored = _card_pixel(window, PBR, _normal_map(toward))
    np.testing.assert_allclose(ignored, _card_pixel(window, PBR), atol=2 / 255.0)


def test_emission_glows_in_the_dark(window):
    """A glowing part shows on the night side at the PBR tiers (its
    emission, linear) and not at the simple one."""
    night = np.array([0.0, 1.0, 0.0])  # the sun behind the card
    setup = _pbr_material((0.2, 0.2, 0.2), 0.0, 0.8, emission=(0.0, 0.8, 0.4))
    glow = _card_pixel(window, PBR, setup, sun=night, mark=True)
    dark = _card_pixel(window, SIMPLE, setup, sun=night, mark=True)
    assert glow[1] > dark[1] + 0.3
    expected = _pbr_twin((0.2, 0.2, 0.2), 0.0, 0.8)  # the sun's part aside
    assert glow[1] > expected[1]


def test_mark_linear_marks_the_gltf_parts_only():
    """Parts with an sRGB texture or a PBR material are marked linear; the
    others are not; unmarking clears the input."""
    root = NodePath("model")
    maker = CardMaker("part")
    textured = root.attach_new_node(maker.generate())
    textured.set_texture(_texture((0.5, 0.5, 0.5), srgb=True))
    pbr = root.attach_new_node(maker.generate())
    _pbr_material((0.5, 0.5, 0.5), 0.0, 1.0)(pbr)
    plain = root.attach_new_node(maker.generate())
    plain.set_texture(_texture((0.5, 0.5, 0.5)))
    marked = mark_linear(root)
    assert set(marked) == {textured, pbr}
    assert textured.get_attrib(ShaderAttrib).has_shader_input("uBaseLinear")
    assert not plain.has_attrib(ShaderAttrib)
    unmark_linear(marked)
    assert not textured.get_attrib(ShaderAttrib).has_shader_input("uBaseLinear")
    # A model that is itself the part: tagged, as apply replaces its state
    marked = mark_linear(textured)
    assert marked == [textured] and textured.has_tag(LINEAR_TAG)
    lighting = ModelLighting()
    lighting.apply(textured)
    assert textured.get_attrib(ShaderAttrib).has_shader_input("uBaseLinear")
    state = textured.get_state()
    lighting.apply(textured)
    assert textured.get_state() == state  # no churn
    unmark_linear(marked)
    assert not textured.has_tag(LINEAR_TAG)


def test_tracked_lights_mark_and_unmark_linear_parts():
    lights = TrackedLights(NodePath("render"))
    ship = lights.render.attach_new_node("ship")
    hull = ship.attach_new_node(CardMaker("hull").generate())
    hull.set_texture(_texture((0.5, 0.5, 0.5), srgb=True))
    lights.track(ship, shader=True)
    assert hull.get_attrib(ShaderAttrib).has_shader_input("uBaseLinear")
    lights.track(ship, shader=False)
    assert not hull.get_attrib(ShaderAttrib).has_shader_input("uBaseLinear")
    lights.track(ship, shader=True)
    lights.untrack(ship)
    assert not hull.get_attrib(ShaderAttrib).has_shader_input("uBaseLinear")


def test_the_settings_swap_the_shader_live():
    """set_settings changes the tier's shader, keeping the inputs; apply
    gives a model the new one, and nothing when it did not change."""
    lighting = ModelLighting(settings=ModelSettings(shading=PBR))
    assert lighting.inputs.get_shader() == model_shader(PBR)
    ship = NodePath("ship")
    lighting.apply(ship)
    lighting.set_settings(ModelSettings(shading=PBR))
    state = ship.get_state()
    lighting.apply(ship)
    assert ship.get_state() == state
    lighting.set_settings(ModelSettings(shading=SIMPLE))
    lighting.apply(ship)
    attrib = ship.get_attrib(ShaderAttrib)
    assert attrib.get_shader() == model_shader(SIMPLE)
    assert attrib.has_shader_input("uExposure")
    assert {
        model_shader(tier) for tier in (SIMPLE, PBR, PBR_NORMAL, PBR_SKY)
    }.__len__() == 4


def test_apply_leaves_a_node_with_the_inputs_already_alone():
    """Setting the same shader attrib again would make the node a new state
    every frame: apply only sets it when it differs."""
    lighting = ModelLighting()
    ship = NodePath("ship")
    lighting.apply(ship)
    state = ship.get_state()
    lighting.apply(ship)
    assert ship.get_state() == state
    # The inputs' attrib changed (e.g. a texture bound again): set again
    lighting.inputs.set_shader_input("uExposure", 0.5)
    lighting.apply(ship)
    assert ship.get_attrib(ShaderAttrib) == lighting.inputs.get_attrib(ShaderAttrib)


def test_the_planet_axes_follow_its_rotation_only_when_it_turns():
    """update gives the shader the body's rotation once, then again only when
    it changed."""
    terrain = SimpleNamespace(rotation=np.eye(3))
    lighting = ModelLighting(terrain=terrain)
    lighting.update((0.0, 0.0, 0.0))
    assert LMatrix4f(lighting._axes[0]) == LMatrix4f.ident_mat()
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    terrain.rotation = turned
    lighting.update((0.0, 0.0, 0.0))
    # Transposed: Panda's rows are GLSL's columns
    assert LMatrix4f(lighting._axes[0]).get_row3(0) == LVecBase3f(*turned.T[0])
