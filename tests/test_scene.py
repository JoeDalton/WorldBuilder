"""
Scene-graph tests of the drawn pieces on a live ShowBase: the split depth
cameras and the ocean (which need a window: an offscreen one, skipped where
there is no GL context), the sky, the terrain's settings and the planet
clouds' bookkeeping. The shaders' output is checked by
scripts/terrain_shots.py, not here.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from panda3d.core import LVecBase3f, NodePath, Texture

from world_builder.atmosphere import AtmosphereParams, Atmospheres
from world_builder.clouds import sphere
from world_builder.clouds.cloud import CloudSettings
from world_builder.clouds.sphere import CloudVolume
from world_builder.depth import (
    FAR_CAMERA_MASK,
    NEAR_CAMERA_MASK,
    REFLECTION_CAMERA_MASK,
    SplitDepthCameras,
)
from world_builder.ocean import OceanSettings, aces_tonemap
from world_builder.sky import Sky
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.workers import ChunkExecutor

R = 600e3


def _input(node: NodePath, name: str) -> tuple:
    """A shader input's value. The ShaderInput must outlive the read: a
    temporary's vector dangles."""
    value = node.get_shader_input(name)
    return tuple(value.get_vector())


@pytest.fixture(scope="module")
def window(showbase):
    """The session's ShowBase with an offscreen window (the tests are
    headless otherwise); skipped where none can be opened."""
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


# ── depth partitioning ─────────────────────────────────────────────────────────


def test_split_cameras_draw_into_the_window(window):
    cameras = SplitDepthCameras(window, near=1.0)
    try:
        assert window.cam.node().get_camera_mask() == NEAR_CAMERA_MASK
        assert cameras.far_camera.node().get_camera_mask() == FAR_CAMERA_MASK
        assert cameras.far_region.get_sort() < cameras.near_region.get_sort()
        cameras.update(5000.0, 12e6)
        assert window.camLens.get_near() == 1.0
        assert window.camLens.get_far() == pytest.approx(5000.0 * cameras.overlap)
        assert cameras.far_lens.get_near() == 5000.0
        assert cameras.far_lens.get_far() == 12e6
        # Culled to the true far plane, past Panda's 10,000 km.
        bounds = cameras.far_camera.node().get_cull_bounds()
        assert bounds.get_max()[1] == pytest.approx(12e6)
        cameras.update(0.5, 0.6)  # too small: raised
        assert cameras.split == 2.0 and cameras.far == 4.0
        window.graphicsEngine.render_frame()
    finally:
        cameras.destroy()
    assert cameras.far_region not in list(window.win.get_display_regions())


def _card(parent, name, distance, frame, color):
    """A flat, unlit card *distance* ahead of the camera, its frame in
    fractions of the view's half-width (x0, x1, z0, z1)."""
    from panda3d.core import CardMaker

    lens_half = math.tan(math.radians(0.5 * 60.0)) * distance
    maker = CardMaker(name)
    x0, x1, z0, z1 = frame
    maker.set_frame(x0 * lens_half, x1 * lens_half, z0 * lens_half, z1 * lens_half)
    card = parent.attach_new_node(maker.generate())
    card.set_y(distance)
    card.set_color(*color)
    card.set_light_off(1)
    card.set_shader_off(1)
    card.set_texture_off(1)
    card.set_two_sided(True)
    return card


def test_half_resolution_clouds_are_hidden_by_what_is_in_front(window):
    """Each depth range's clouds go into their own half-size layer: near
    ground hides far clouds (one buffer for both ranges showed them through
    it), and near clouds cover near ground."""
    from world_builder.clouds.half_res import HalfResClouds

    fov, transform = window.camLens.get_fov(), window.camera.get_transform()
    window.camLens.set_fov(60.0, 60.0)
    window.camera.set_pos_hpr(0, 0, 0, 0, 0, 0)
    cameras = SplitDepthCameras(window, near=1.0)
    clouds = window.render.attach_new_node("test_clouds")
    occluder = _card(
        window.render, "occluder", 50.0, (-0.4, 0.4, -0.4, 0.4), (0, 1, 0, 1)
    )
    _card(clouds, "far_cloud", 2000.0, (-2, 2, -2, 2), (1, 0, 0, 1))
    _card(clouds, "near_cloud", 20.0, (-2, -0.5, -2, 2), (0, 0, 1, 1))
    half = HalfResClouds(window, cameras)
    half.add(clouds)
    try:
        cameras.update(500.0, 5000.0)
        half.update()
        for _ in range(3):
            window.graphicsEngine.render_frame()
        shot = window.win.get_screenshot()
        size = shot.get_x_size()
        ram = np.frombuffer(shot.get_ram_image_as("RGB").get_data(), np.uint8)
        rgb = ram.reshape(shot.get_y_size(), size, 3)

        def at(x, y):  # fractions of the view, from the bottom left
            return tuple(int(c) for c in rgb[int(y * size), int(x * size)])

        assert at(0.55, 0.5) == (0, 255, 0)  # the occluder hides the far cloud
        assert at(0.9, 0.9) == (255, 0, 0)  # the far cloud elsewhere
        assert at(0.1, 0.5) == (0, 0, 255)  # the near cloud, in front
        assert half.pixel_scale == 2.0
    finally:
        half.destroy()
        cameras.destroy()
        clouds.remove_node()
        occluder.remove_node()
        window.camLens.set_fov(fov)
        window.camera.set_transform(transform)
    assert clouds.is_empty()


def test_split_cameras_keep_the_split_off_translucent_volumes(window):
    """A translucent volume the wanted split would cut moves it in front:
    blended twice where the cameras overlap, it would show a ring there."""
    from panda3d.core import CardMaker

    maker = CardMaker("bubble")
    maker.set_frame(-30.0, 30.0, -40.0, 40.0)  # bounding radius 50 m
    bubble = window.render.attach_new_node(maker.generate())
    camera = window.cam.get_pos(window.render)
    bubble.set_pos(camera + (0.0, 2000.0, 0.0))  # 2 km ahead
    cameras = SplitDepthCameras(window, near=1.0)
    try:
        cameras.update(2000.0, 12e6, translucent=[bubble])
        assert cameras.split == pytest.approx(1950.0 / cameras.overlap, abs=1.0)
        cameras.update(2000.0, 12e6)  # none: as wanted
        assert cameras.split == 2000.0
    finally:
        cameras.destroy()
        bubble.remove_node()


def test_split_cameras_follow_the_scene_into_a_post_process_buffer(window):
    """With the scene rendered into a texture (a game's render scale or
    anti-aliasing pipeline), the far camera draws into that buffer, before
    the near one, not into the window under the composited quad."""
    from direct.filter.FilterManager import FilterManager

    manager = FilterManager(window.win, window.cam)
    assert manager.renderSceneInto(colortex=Texture("scene")) is not None
    try:
        buffer = window.cam.node().get_display_region(0).get_window()
        assert buffer != window.win
        cameras = SplitDepthCameras(window, near=1.0)
        try:
            assert cameras.output == buffer
            assert cameras.far_region in list(buffer.get_display_regions())
            assert cameras.far_region not in list(window.win.get_display_regions())
            assert cameras.far_region.get_sort() < cameras.near_region.get_sort()
            cameras.update(5000.0, 12e6)
            window.graphicsEngine.render_frame()
        finally:
            cameras.destroy()
        assert cameras.far_region not in list(buffer.get_display_regions())
    finally:
        manager.cleanup()


# ── sea ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def sea(window):
    """A Sea on an ocean planet's coarse terrain, its camera 30 m over the
    water at a point of the planet."""
    from world_builder.ocean import Sea
    from world_builder.terrain.terrain import Terrain, TerrainSettings

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    terrain = Terrain(
        window.render,
        PRESETS["earthlike"],
        executor=executor,
        settings=TerrainSettings(min_level=7),
    )
    made = Sea(
        window,
        terrain,
        clock=lambda: 1.5,
        settings=OceanSettings(reflection_scale=0.25),
    )
    up = TangentFrame.at(10.0, 20.0).up
    camera = up * (R + 30.0)
    terrain.update(camera)
    window.camera.set_pos(window.render, 0.0, 0.0, 0.0)
    yield made, terrain, camera
    made.destroy()
    terrain.destroy()
    executor.shutdown()


def test_the_terrain_draws_the_sea_of_an_ocean_planet(sea):
    _, terrain, _ = sea
    assert _input(terrain.root, "uSeaSurface")[0] == 1


def test_sea_draws_before_its_first_update(window, sea):
    """A game may render frames between building the sea and its first
    update (a loading screen): every shader input it reads is already set."""
    window.graphicsEngine.render_frame()  # raises on a missing input


def test_sea_updates_and_mirrors_the_camera(window, sea):
    made, terrain, camera = sea
    window.camera.set_hpr(window.render, 15.0, -10.0, 0.0)
    made.update(camera)
    assert made.near == 1.0
    assert terrain.sea_root.get_shader() is not None  # the near look's variant
    # The frame is the tangent plane at the nadir's sea level: the camera
    # 30 m above it, its mirror image 30 m below.
    np.testing.assert_allclose(
        tuple(window.camera.get_pos(made.frame)), (0.0, 0.0, 30.0), atol=1e-2
    )
    np.testing.assert_allclose(
        tuple(made.refl_cam.get_pos(made.frame)), (0.0, 0.0, -30.0), atol=1e-2
    )
    up = made.frame.get_quat(window.render).get_up()
    np.testing.assert_allclose(tuple(up), camera / np.linalg.norm(camera), atol=1e-5)
    assert made._frame_inputs["iTime"][0] == 1.5
    # The octaves' phases reach the shader's array (written through a view).
    from world_builder.ocean.ocean import wave_phases

    phases = made._frame_inputs["uWavePhase"]
    expected = wave_phases(made._directions, made._chart_xy, 1.5)
    np.testing.assert_allclose(
        [phases[i] for i in range(len(expected))], expected, rtol=1e-6, atol=1e-5
    )
    # The swell handed to the terrain is the drift's at the clock's time.
    np.testing.assert_array_equal(terrain.swell.rot_a, made._drift.at(1.5)[0])
    window.graphicsEngine.render_frame()
    made.update(camera)  # the reflection texture is realized: the UV scale is set
    assert made._uv_scale_set


def test_the_near_look_stops_above_its_band(window, sea):
    made, terrain, camera = sea
    high = camera / np.linalg.norm(camera) * (R + made.near_band[1] + 100.0)
    made.update(high)
    assert made.near == 0.0 and not made.refl_buffer.is_active()
    assert made._frame_inputs["uSeaNear"][0] == 0.0
    assert terrain.sea_root.get_shader() is None  # the root's, without it


def test_reflection_follows_the_main_field_of_view(window, sea):
    made, _, camera = sea
    scale = made.settings.reflection_fov_scale
    lens = window.camLens
    before = lens.get_fov()
    try:
        lens.set_fov(50.0, 30.0)
        made.update(camera)
        fov = made.refl_cam_node.get_lens().get_fov()
        np.testing.assert_allclose(tuple(fov), (50.0 * scale, 30.0 * scale))
    finally:
        lens.set_fov(before)


def test_sea_light(sea):
    made, _, _ = sea
    made.set_light((0.5, 0.25, 0.0))
    light = made._frame_inputs["iWaterLight"][0]
    np.testing.assert_allclose(tuple(light), (0.5, 0.25, 0.0))
    # The shader decodes the reflection as display colours, so where it
    # renders nothing it must find the lit water's colour tonemapped.
    clear = made.refl_buffer.get_clear_color()
    lit = np.multiply(tuple(made._water_color), (0.5, 0.25, 0.0))
    np.testing.assert_allclose(tuple(clear)[:3], aces_tonemap(lit), atol=1e-6)


def test_sea_settings_set_its_uniforms(sea):
    from dataclasses import replace

    from world_builder.ocean import OceanSettings
    from world_builder.ocean.ocean import wave_weight_sum

    made, terrain, _ = sea
    node = terrain.root
    made.set_settings(replace(OceanSettings.preset("low"), reflection_scale=0.25))
    assert _input(node, "iIterationsNormal")[0] == 16
    assert _input(node, "uWaveWeightSum")[0] == pytest.approx(wave_weight_sum(16))
    assert _input(node, "uAAEnd")[0] == pytest.approx(2.0 * math.pi / 6.0)
    assert _input(node, "uRoughFresnel")[0] == 0
    assert _input(node, "uReflBlur")[0] == 0.0
    assert not made.refl_tex.uses_mipmaps()  # nothing reads them
    assert _input(node, "uFoamNoiseOn")[0] == 0
    made.set_settings(replace(OceanSettings.preset("high"), reflection_scale=0.25))
    assert _input(node, "uRoughFresnel")[0] == 1
    assert made.refl_tex.uses_mipmaps()
    made.set_settings(replace(made.settings, foam=False))
    assert _input(node, "uFoamOn")[0] == 0
    made.set_settings(made.settings.with_value("wave_iterations", 10**6))
    assert _input(node, "iIterationsNormal")[0] == 64  # clamped
    made.set_debug_mode(10)
    assert _input(node, "uSeaDebug")[0] == 10
    made.set_swell_amplitude(45.0)
    assert _input(node, "uSwellAmplitude")[0] == 45.0
    # A cut-off sharper than full strength is held at it.
    made.set_settings(replace(made.settings, aa_full_px=5.0, aa_gone_px=9.0))
    assert _input(node, "uAAEnd")[0] == pytest.approx(2.0 * math.pi / 5.0)


def test_sea_reflection_rebuild(window, sea):
    """A new reflection scale makes the buffer again on the same camera (its
    lens, far plane and active flag kept)."""
    from dataclasses import replace

    made, terrain, _ = sea
    kept = made.settings
    old_buffer, camera = made.refl_buffer, made.refl_cam
    far = made.refl_cam_node.get_lens().get_far()
    width = old_buffer.get_x_size()
    old_buffer.set_active(False)
    made.set_settings(replace(kept, reflection_scale=kept.reflection_scale * 2))
    assert made.refl_buffer is not old_buffer
    assert made.refl_buffer.get_x_size() == pytest.approx(2 * width, abs=1)
    assert not made.refl_buffer.is_active()
    assert made.refl_cam == camera
    assert made.refl_cam_node.get_lens().get_far() == far
    engine = window.graphicsEngine
    assert old_buffer not in [
        engine.get_window(i) for i in range(engine.get_num_windows())
    ]
    rebound = terrain.root.get_shader_input("iReflectionTex").get_texture()
    assert rebound == made.refl_tex


def test_sea_destroy_releases_its_buffer(window, sea):
    made, terrain, _ = sea
    buffer = made.refl_buffer
    made.destroy()
    assert buffer not in [
        window.graphicsEngine.get_window(i)
        for i in range(window.graphicsEngine.get_num_windows())
    ]
    assert made._frame_inputs["uSeaNear"][0] == 0.0  # the far look stays
    made.destroy = lambda: None  # the fixture's teardown


def test_the_shader_swell_is_the_cpus(window):
    """swell.glsl and world_builder.ocean.swell are one field: the GPU's
    float32 evaluation, read back, matches the CPU's at the same points (the
    collisions stand on the CPU's, the eye sees the GPU's)."""
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomPoints,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        LMatrix4f,
        OmniBoundingVolume,
        Shader,
    )

    from world_builder.ocean.ocean import swell_rotations
    from world_builder.ocean.swell import Swell, swell_field

    size = 32
    frame = TangentFrame.at(10.0, 20.0)
    a, b = swell_rotations(
        frame.up, frame.east, frame.north, (0.6, 0.8), 0.004, 0.1, R, 777.0
    )
    rng = np.random.default_rng(5)
    dirs = frame.up + 0.05 * rng.normal(size=(size * size, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = (dirs * R).astype(np.float32)  # what the GPU gets
    max_freq = 0.25 / 29.0
    swell = Swell(20.0, 0.004, a, b)
    expected = swell_field(
        swell, points.astype(np.float64), np.full(len(points), max_freq)
    )

    vdata = GeomVertexData("points", GeomVertexFormat.get_v3(), Geom.UH_static)
    writer = GeomVertexWriter(vdata, "vertex")
    for x, y, z in points:
        writer.add_data3f(float(x), float(y), float(z))
    prim = GeomPoints(Geom.UH_static)
    prim.add_next_vertices(len(points))
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("swell_points")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    import world_builder.ocean

    shaders = Path(world_builder.ocean.__file__).parent / "shaders"
    swell_glsl = (shaders / "swell.glsl").read_text()
    vertex = f"""#version 330
in vec4 p3d_Vertex;
uniform float uMaxFreq;
flat out float vField;
{swell_glsl}
void main() {{
    vec3 g;
    vField = swellField(p3d_Vertex.xyz, uMaxFreq, g);
    int x = gl_VertexID % {size}, y = gl_VertexID / {size};
    gl_Position = vec4((vec2(x, y) + 0.5) / {size}.0 * 2.0 - 1.0, 0.0, 1.0);
}}
"""
    fragment = """#version 330
flat in float vField;
out vec4 fragColor;
void main() {
    float v = floor(clamp(vField, 0.0, 1.0) * 16777215.0 + 0.5);
    float r = floor(v / 65536.0);
    float g = floor((v - r * 65536.0) / 256.0);
    fragColor = vec4(r, g, v - r * 65536.0 - g * 256.0, 255.0) / 255.0;
}
"""
    root = NodePath("swell_check")
    points_np = root.attach_new_node(node)
    points_np.set_shader(Shader.make(Shader.SL_GLSL, vertex, fragment))

    def mat(m):
        out = np.eye(4)
        out[:3, :3] = m.T  # Panda multiplies rows
        return LMatrix4f(*out.ravel())

    points_np.set_shader_inputs(
        uMaxFreq=max_freq,
        iSwellScale=0.004,
        uSwellAmplitude=20.0,
        uSwellRotA=mat(a),
        uSwellRotB=mat(b),
        uSwellByDepth=0,
        uSwellShallow=5.0,
        uSwellDeep=60.0,
    )
    texture = Texture("swell_check")
    buffer = window.win.make_texture_buffer("swell_check", size, size, texture, True)
    try:
        camera = window.makeCamera(buffer)
        camera.node().set_scene(root)
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGB").get_data(), np.uint8)
        rgb = ram.reshape(size, size, 3).astype(np.int64)
        got = (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]) / 16777215.0
        # float32 planet-frame points (6 cm apart at 600 km) move the noise a
        # little: 3e-4 of the field is 6 mm of water at 20 m of swell.
        np.testing.assert_allclose(got.ravel(), expected, atol=3e-4)
        assert np.ptp(expected) > 0.1  # a field, not a constant
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()


# ── sky ────────────────────────────────────────────────────────────────────────


def test_sky_with_a_painted_cube_map(showbase):
    painted = Texture("painted")
    painted.setup_cube_map(4, Texture.T_unsigned_byte, Texture.F_rgb8)
    painted.set_ram_image(bytes(6 * 4 * 4 * 3))
    atmospheres = Atmospheres()
    atmospheres.add(R, AtmosphereParams())
    sky = Sky(showbase.render, painted, atmospheres, star_hpr=(30.0, 0.0, 0.0))
    try:
        assert sky.stars is None  # no point stars over a painted map
        assert _input(sky.node, "uGlowMode")[0] == 0
        sky.set_sun((0.0, 3.0, 4.0))
        np.testing.assert_allclose(tuple(sky._sun[0]), (0.0, 0.6, 0.8), atol=1e-6)
        sky.set_star_brightness(4.0)
        sky.set_star_rotation((0.0, 90.0, 0.0))
        sky.update((1.0, 2.0, 3.0), 1000.0)
        assert tuple(sky.node.get_pos()) == (1.0, 2.0, 3.0)
        assert sky.node.get_scale()[0] == pytest.approx(900.0)
    finally:
        sky.remove()
    assert sky.node.is_empty()


# ── terrain settings ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def terrain(showbase):
    from world_builder.terrain.terrain import Terrain, TerrainSettings

    executor = ChunkExecutor(backend="inline", inline_budget=16)
    made = Terrain(
        showbase.render,
        PRESETS["moon"],
        executor=executor,
        settings=TerrainSettings(min_level=7),
    )
    yield made
    made.destroy()
    executor.shutdown()


def test_terrain_settings(terrain):
    root = terrain.root
    terrain.set_debug_mode(3)
    assert _input(root, "uDebugMode")[0] == 3
    terrain.set_blend_depth(0.3)
    assert terrain.blend_depth == 0.3
    assert _input(root, "uSeaSurface")[0] == 0  # the moon has no sea
    from world_builder import SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M

    np.testing.assert_allclose(_input(root, "uShallowWater")[:3], SHALLOW_WATER_COLOR)
    assert _input(root, "uShallowDepth")[0] == pytest.approx(SHALLOW_WATER_DEPTH_M)
    terrain.set_shallow_water((0.1, 0.2, 0.3), 4.0)
    np.testing.assert_allclose(
        _input(root, "uShallowWater")[:3], (0.1, 0.2, 0.3), rtol=1e-6
    )
    terrain.set_wireframe(True)
    assert root.has_render_mode()
    terrain.set_wireframe(False)
    assert not root.has_render_mode()
    terrain.set_max_distance(5e3)
    assert terrain.selector.max_distance == 5e3
    terrain.set_max_distance(None)
    terrain.set_sun((0.0, 0.0, 2.0), (1.0, 1.0, 1.0), (0.1, 0.1, 0.1))
    assert tuple(terrain._frame_inputs["uSunDir"][0]) == (0.0, 0.0, 1.0)


def test_the_gpu_reads_each_chunks_values(window, terrain):
    """The chunk table, as the GPU fetches it: each uploaded chunk's slot
    holds its centre and swell band, uv origin, step and skirt depth, level
    and face, in that channel order (Panda keeps RGBA as BGRA in RAM)."""
    from panda3d.core import CardMaker, FrameBufferProperties, Shader

    camera = np.array([0.0, 0.0, terrain.radius + 3000.0])
    for _ in range(6):
        terrain.update(camera)
    assert terrain.nodes
    slots = sorted(terrain._slots.values())
    width = max(slots) + 1
    props = FrameBufferProperties()
    props.set_rgba_bits(32, 32, 32, 32)
    props.set_float_color(True)
    texture = Texture("chunk_table_check")
    buffer = window.win.make_texture_buffer(
        "chunk_table_check", width, 3, texture, True, props
    )
    maker = CardMaker("copy")
    maker.set_frame(-1, 1, -1, 1)
    card = NodePath(maker.generate())
    card.set_shader(
        Shader.make(
            Shader.SL_GLSL,
            vertex="#version 330\nin vec4 p3d_Vertex;\n"
            "void main() { gl_Position = vec4(p3d_Vertex.x, p3d_Vertex.z, 0, 1); }",
            fragment="#version 330\nuniform sampler2D uChunkTable;\n"
            "out vec4 o;\nvoid main() { o = texelFetch(uChunkTable,"
            " ivec2(gl_FragCoord.xy), 0); }",
        )
    )
    card.set_shader_input("uChunkTable", terrain._chunk_texture)
    card.set_depth_test(False)
    try:
        cam = window.makeCamera(buffer)
        cam.node().set_scene(card)
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGBA").get_data(), np.float32)
        got = ram.reshape(3, width, 4)
        for key, slot in terrain._slots.items():
            chunk = terrain.store[key]
            u0, v0 = terrain.grid.uv_origin(key)
            expected = np.array(
                [
                    [*chunk.centre, 0.25 / terrain.grid.spacing(key[0])],
                    [u0, v0, (1 << key[0]) * terrain.grid.uv_unit, chunk.skirt_depth],
                    [key[0], key[1], 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(got[:, slot], expected)
            # Every vertex of the chunk's node names the slot.
            vdata = terrain.nodes[key].node().get_geom(0).get_vertex_data()
            rows = np.frombuffer(memoryview(vdata.get_array(0)).cast("B"), np.uint8)
            rows = rows.reshape(-1, 36).astype(np.int64)
            assert np.all(rows[:, 34] + 256 * (rows[:, 35] & 127) == slot)
            assert np.all(rows[-256:, 35] >= 128) and np.all(rows[:-256, 35] < 128)
    finally:
        window.graphicsEngine.remove_window(buffer)


def test_terrain_frames(terrain):
    point = np.array([1.0, 2.0, R])
    np.testing.assert_array_equal(terrain.to_planet(point), point - terrain.position)
    assert terrain.altitude(np.array([0.0, 0.0, terrain.radius + 5.0])) == 5.0


# ── planet clouds ──────────────────────────────────────────────────────────────


@pytest.fixture
def clouds(showbase):
    executor = ChunkExecutor(backend="inline", inline_budget=64)
    made = CloudVolume(
        showbase.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(shells=2),
        planet=PRESETS["forest"],
    )
    yield made, executor
    made.remove()
    executor.shutdown()


def test_cloud_settings(clouds):
    made, _ = clouds
    made.set_sun((0.0, 0.0, 2.0), sun_color=(1.0, 0.5, 0.2), sky_color=(0.2, 0.3, 0.4))
    assert tuple(made._sun[0]) == (0.0, 0.0, 1.0)
    assert _input(made.root, "sunColor")[1] == 0.5
    made.set_sky_light(0.7)
    assert _input(made.root, "skyStrength")[0] == (pytest.approx(0.7))
    made.set_pixel_angle(0.002)
    made.set_debug(5)
    assert _input(made.root, "uCloudDebug")[0] == 5


def test_clouds_forget_tiles_far_out_of_use(clouds):
    """Flying away, the cache keeps at most ``tile_cache`` tiles nobody
    wants; tiles for an older spec version are dropped on arrival."""
    from dataclasses import replace

    made, _ = clouds
    kept = made.settings
    made.set_settings(replace(kept, tile_cache=0))  # live: no rebuild
    assert made.settings.tile_cache == 0
    first = TangentFrame.at(20.0, 40.0).up * (R + 1000.0)
    for _ in range(10):
        made.update(first)
    before = set(made._tiles)
    assert before
    elsewhere = TangentFrame.at(-30.0, -100.0).up * (R + 1000.0)
    for _ in range(60):  # until the new area's tiles are all in
        made.update(elsewhere)
    assert not (before & set(made._tiles))  # the first area's tiles are gone
    # A tile generated for an older version is not uploaded.
    key = next(iter(made._wanted))
    stale = sphere.generate_cloud_tile(made.spec, key, version=made._version - 1)
    made._tiles.pop(key).node.remove_node()
    made._ready[key] = stale
    made._upload()
    assert key not in made._tiles
    made.set_settings(kept)


def test_clouds_settings_live_and_rebuilt(clouds):
    """The sun march changes live (same tiles, same worker spec); the
    billboard count re-plans the shells and regenerates every tile."""
    from dataclasses import replace

    from world_builder.depth import REFLECTION_CAMERA_MASK

    made, _ = clouds
    kept = made.settings
    camera = TangentFrame.at(20.0, 40.0).up * (R + 1000.0)
    try:
        for _ in range(10):
            made.update(camera)
        tiles, version, spec = set(made._tiles), made._version, made.spec
        assert tiles
        made.set_settings(replace(kept, march_scale=0.5, reflected=False))
        assert made._version == version and made.spec == spec
        assert set(made._tiles) == tiles
        assert made.root.is_hidden(REFLECTION_CAMERA_MASK)
        made.set_settings(replace(made.settings, count_scale=0.5))
        assert made._version == version + 1 and not made._tiles
        # Fewer billboards, larger: as count^(-1/3).
        assert made.spec.shells[0].radius[0] > spec.shells[0].radius[0]
        made.set_settings(made.settings)  # nothing changes: no rebuild
        assert made._version == version + 1
    finally:
        made.set_settings(kept)
    assert not made.root.is_hidden(REFLECTION_CAMERA_MASK)


def test_clouds_rebase_with_altitude(clouds):
    """The rebasing grid coarsens with altitude, so float32 only sees small
    numbers near the camera."""
    made, _ = clouds
    camera = TangentFrame.at(20.0, 40.0).up * (R + 300e3)
    made.update(camera, camera_render=(0.0, 0.0, 0.0))
    origin = made._origin
    assert np.all(np.abs(camera - origin) <= 0.5 * 4096.0 * 2**8)
    np.testing.assert_allclose(
        tuple(made.tiles_root.get_pos()), -(camera - origin), atol=1.0
    )


def test_cloud_tile_node_bounds_hold_the_billboards(clouds):
    made, _ = clouds
    camera = TangentFrame.at(20.0, 40.0).up * (R + 1000.0)
    for _ in range(8):
        made.update(camera)
    tile = next(t for t in made._tiles.values() if t.tris is not None)
    bounds = tile.node.node().get_internal_bounds()
    p = tile.data.particles
    assert bounds.contains(LVecBase3f(*p[0, 0:3]))
    assert tile.reach >= float(np.linalg.norm(p[:, 0:3], axis=1).max())


def test_terrain_settings_apply_live(terrain):
    """Every live terrain knob reaches what it drives; the LOD factor survives
    a new planet; the texture size waits for the next terrain."""
    from dataclasses import replace

    from world_builder.terrain.terrain import TerrainSettings

    kept = terrain.settings
    try:
        new = replace(
            TerrainSettings.preset("low"),
            min_level=2,
            split_factor=1.6,
            textured=False,
            reflected=False,
        )
        terrain.set_settings(new)
        assert terrain.settings == new
        assert terrain.selector.split_factor == 1.6
        assert terrain.min_level == terrain.query.level == 2
        assert _input(terrain.root, "uDetailFar")[0] == pytest.approx(1500.0)
        assert _input(terrain.root, "uTextured")[0] == 0
        assert all(t.get_anisotropic_degree() == 2 for t in terrain._textures)
        assert terrain.root.is_hidden(REFLECTION_CAMERA_MASK)
        terrain.set_spec(terrain.spec)
        assert terrain.selector.split_factor == 1.6
        # Past the grid's deepest level, the deepest is drawn.
        terrain.set_settings(replace(new, min_level=8))
        assert terrain.min_level == min(8, terrain.grid.max_level)
    finally:
        terrain.set_settings(kept)
    assert terrain.selector.split_factor == kept.split_factor


def test_terrain_caches_follow_their_settings(terrain):
    """A lower RAM cap evicts on the next update, never a chunk the drawn
    selection needs."""
    from dataclasses import replace

    kept = terrain.settings
    camera = np.array([0.0, 0.0, terrain.radius + 2000.0])
    for _ in range(40):
        terrain.update(camera)
    try:
        terrain.set_settings(replace(kept, store_size=500, node_cache_size=300))
        stored = set(terrain.store)
        for _ in range(5):
            terrain.update(camera)
        protected = terrain._keep | set(terrain._wanted)
        assert len(terrain.store) <= max(500, len(protected & set(terrain.store)))
        assert len(terrain.nodes) <= max(300, len(protected & set(terrain.nodes)))
        # What the drawn selection needs and was in memory stays there.
        assert (terrain._keep & stored) <= set(terrain.store)
    finally:
        terrain.set_settings(kept)


def test_atmosphere_steps_are_shared_live():
    """The march's steps are arrays bound once: a change reaches every node
    already bound, without touching them."""
    from world_builder.atmosphere import AtmosphereSettings, bind_defaults

    air = Atmospheres(AtmosphereSettings.preset("low"))
    air.add(R, AtmosphereParams())
    node = NodePath("bound")
    air.bind(node)
    assert air._steps[0] == 4
    bound = node.get_shader_input("uAtmoSteps")
    air.set_settings(AtmosphereSettings.preset("ultra"))
    # The same array, rewritten: the node's input is unchanged.
    assert node.get_shader_input("uAtmoSteps") == bound
    assert (air._steps[0], air._vertex_steps[0]) == (12, 8)
    bare = NodePath("bare")
    bind_defaults(bare)
    assert bare.get_shader_input("uAtmoSteps").get_name().get_name() == "uAtmoSteps"


def test_sky_settings_thin_the_stars(showbase):
    from world_builder.sky import SkySettings, Starfield

    starfield = Starfield(stars=400, glow_size=16)
    sky = Sky(showbase.render, starfield)
    try:
        # No settings: the starfield's own glow, not a new one.
        assert sky.node.get_shader_input("uStars").get_texture() == starfield.glow
        points = lambda: sky.stars.node().get_geom(0).get_primitive(0)  # noqa: E731
        assert points().get_num_vertices() == 400
        sky.set_settings(SkySettings(star_fraction=0.25, glow_size=64))
        assert points().get_num_vertices() == 100
        assert sky.node.get_shader_input("uStars").get_texture().get_x_size() == 64
        # Brightest first: the stars kept are the brightest.
        flux = np.frombuffer(
            sky.stars.node()
            .get_geom(0)
            .get_vertex_data()
            .get_array(0)
            .get_handle()
            .get_data(),
            dtype=np.float32,
        ).reshape(-1, 7)[:, 6]
        assert np.all(np.diff(flux) <= 0)
    finally:
        sky.remove()
