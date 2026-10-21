"""
Scene-graph tests of the drawn pieces on a live ShowBase: the split depth
cameras and the ocean (which need a window: an offscreen one, skipped where
there is no GL context), the sky, the terrain's settings and the planet
clouds' bookkeeping. The shaders' output is checked by
scripts/terrain_shots.py, not here.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from panda3d.core import BitMask32, CullBinManager, LVecBase3f, NodePath, Texture

from world_builder.atmosphere import AtmosphereParams, Atmospheres
from world_builder.clouds import field, sphere
from world_builder.clouds.sphere import SphericalClouds
from world_builder.depth import (
    FAR_CAMERA_MASK,
    NEAR_CAMERA_MASK,
    SplitDepthCameras,
)
from world_builder.ocean import aces_tonemap
from world_builder.sky import Sky
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.terrain import REFLECTION_CAMERA_BITS
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


# ── ocean ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def ocean(window):
    from world_builder.ocean import Ocean

    root = window.render.attach_new_node("sea")
    game = SimpleNamespace(
        app=window,
        root_node=root,
        method_lists={},
        game_time=SimpleNamespace(get_current_time=lambda: 1.5),
    )
    made = Ocean(
        game=game,
        geometric_swell=True,
        planet_radius=R,
        exact_sphere=True,
        reflection_scale=0.25,
    )
    yield made, game
    if made.game is not None:
        made.clean()
    root.remove_node()


def test_ocean_updates_and_mirrors_the_camera(window, ocean):
    made, game = ocean
    assert made.id in game.method_lists
    window.camera.set_pos(made.base_node, 10.0, 20.0, 30.0)
    window.camera.set_hpr(made.base_node, 15.0, -10.0, 0.0)
    for method in game.method_lists[made.id]:
        method()
    mirrored = made.refl_cam.get_pos(made.base_node)
    np.testing.assert_allclose(tuple(mirrored), (10.0, 20.0, -30.0), atol=1e-4)
    assert _input(made.ocean_node, "iTime")[0] == 1.5
    window.graphicsEngine.render_frame()
    made.update()  # the reflection texture is realized: the UV scale is set
    assert getattr(made, "_uv_scale_set", False)


def _drawn(node: NodePath, camera_mask: BitMask32) -> bool:
    """Panda's rule: drawn if any of the camera's bits is shown, or not
    controlled, on the node."""
    n = node.node()
    shown = n.get_draw_show_mask() | ~n.get_draw_control_mask()
    return not (shown & camera_mask).is_zero()


def test_ocean_is_not_in_its_own_mirror(ocean):
    """Hidden from the whole reflection mask (one bit hid nothing), seen by
    the main cameras."""
    from world_builder.ocean.ocean import REFLECTION_CAMERA_MASK

    made, _ = ocean
    reflection = made.refl_cam_node.get_camera_mask()
    assert reflection == REFLECTION_CAMERA_MASK
    assert not _drawn(made.ocean_node, reflection)
    assert _drawn(made.ocean_node, NEAR_CAMERA_MASK)
    assert _drawn(made.ocean_node, FAR_CAMERA_MASK)
    assert _drawn(made.ocean_node, BitMask32.all_on())  # a default camera


def test_reflection_follows_the_main_field_of_view(window, ocean):
    from world_builder.ocean.ocean import REFLECTION_FOV_SCALE

    made, _ = ocean
    lens = window.camLens
    before = lens.get_fov()
    try:
        lens.set_fov(50.0, 30.0)
        made.update()
        fov = made.refl_cam_node.get_lens().get_fov()
        np.testing.assert_allclose(
            tuple(fov), (50.0 * REFLECTION_FOV_SCALE, 30.0 * REFLECTION_FOV_SCALE)
        )
    finally:
        lens.set_fov(before)


def test_everything_drawn_in_the_mirror_writes_its_clip_distance():
    """The reflection pass turns a clip plane on; a vertex shader drawn
    through it that leaves gl_ClipDistance[0] unwritten is clipped at
    random."""
    src = Path(__file__).parent.parent / "src" / "world_builder"
    shaders = [
        "terrain/shaders/terrain.vert",
        "clouds/shaders/cloud.vert",
        "clouds/shaders/cloud_sphere.vert",
        "clouds/shaders/cloud_deck.vert",
        "sky/shaders/sky.vert",
        "sky/shaders/stars.vert",
    ]
    for name in shaders:
        assert "gl_ClipDistance[0]" in (src / name).read_text(), name


def test_cloud_bins_never_tie(showbase):
    """The planet clouds' bins (deck, then one per shell) are ordered, sit
    between opaque and transparent, and never tie with the flat field's."""
    sphere._ensure_bin()
    field._ensure_cloud_bins()
    manager = CullBinManager.get_global_ptr()

    def sort(name):
        return manager.get_bin_sort(manager.find_bin(name))

    deck = sort(sphere.CLOUD_BIN)
    shells = [sort(sphere._shell_bin(k)) for k in range(sphere.MAX_LAYERS - 1)]
    flat = sort(field._CLOUD_BIN)
    assert all(20 < s < 30 for s in [deck, *shells])
    assert shells == sorted(shells, reverse=True) and len(set(shells)) == len(shells)
    assert deck < min(shells)
    assert flat not in [deck, *shells]


def test_ocean_light_and_settings(ocean):
    made, _ = ocean
    made.set_light((0.5, 0.25, 0.0))
    light = _input(made.ocean_node, "iWaterLight")
    np.testing.assert_allclose(tuple(light)[:3], (0.5, 0.25, 0.0))
    # The shader decodes the reflection as display colours, so where it
    # renders nothing it must find the lit water's colour tonemapped.
    clear = made.refl_buffer.get_clear_color()
    lit = np.multiply(tuple(made._water_color), (0.5, 0.25, 0.0))
    np.testing.assert_allclose(tuple(clear)[:3], aces_tonemap(lit), atol=1e-6)
    made.set_atmosphere_frame(made.base_node.get_mat(), (0.0, 0.0, 1.0))
    made.set_wave_iterations(12)
    made.set_wave_iterations(10**6)  # clamped to the shader's maximum


def test_ocean_clean_releases_its_buffer(window, ocean):
    made, game = ocean
    buffer = made.refl_buffer
    made.clean()
    assert made.id not in game.method_lists
    assert buffer not in [
        window.graphicsEngine.get_window(i)
        for i in range(window.graphicsEngine.get_num_windows())
    ]


def test_flat_ocean_is_one_quad(window):
    from world_builder.ocean.ocean import make_plane_mesh

    node = make_plane_mesh(100.0)
    vdata = node.get_geom(0).get_vertex_data()
    assert vdata.get_num_rows() == 4
    assert node.get_geom(0).get_primitive(0).get_num_primitives() == 2


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
    from world_builder.terrain.terrain import Terrain

    executor = ChunkExecutor(backend="inline", inline_budget=16)
    made = Terrain(showbase.render, PRESETS["moon"], executor=executor, min_level=7)
    yield made
    made.destroy()
    executor.shutdown()


def test_terrain_settings(terrain):
    root = terrain.root
    terrain.set_debug_mode(3)
    assert _input(root, "uDebugMode")[0] == 3
    terrain.set_textured(False)
    assert _input(root, "uTextured")[0] == 0
    terrain.set_blend_depth(0.3)
    assert terrain.blend_depth == 0.3
    terrain.set_sea_surface(True)
    assert _input(root, "uSeaSurface")[0] == 1
    terrain.set_wireframe(True)
    assert root.has_render_mode()
    terrain.set_wireframe(False)
    assert not root.has_render_mode()
    terrain.set_reflected(False)
    assert root.is_hidden(REFLECTION_CAMERA_BITS)
    terrain.set_reflected(True)
    assert not root.is_hidden(REFLECTION_CAMERA_BITS)
    terrain.set_min_level(99)
    assert terrain.min_level == terrain.grid.max_level
    terrain.set_min_level(7)
    terrain.set_max_distance(5e3)
    assert terrain.selector.max_distance == 5e3
    terrain.set_max_distance(None)
    terrain.set_sun((0.0, 0.0, 2.0), (1.0, 1.0, 1.0), (0.1, 0.1, 0.1))
    assert tuple(terrain._frame_inputs["uSunDir"][0]) == (0.0, 0.0, 1.0)


def test_terrain_frames(terrain):
    point = np.array([1.0, 2.0, R])
    np.testing.assert_array_equal(terrain.to_planet(point), point - terrain.position)
    assert terrain.altitude(np.array([0.0, 0.0, terrain.radius + 5.0])) == 5.0


# ── planet clouds ──────────────────────────────────────────────────────────────


@pytest.fixture
def clouds(showbase):
    executor = ChunkExecutor(backend="inline", inline_budget=64)
    made = SphericalClouds(
        showbase.render,
        R,
        executor,
        altitude=2200.0,
        shells=2,
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


def test_clouds_forget_tiles_far_out_of_use(clouds, monkeypatch):
    """Flying away, the cache keeps at most TILE_CACHE tiles nobody wants;
    tiles for an older spec version are dropped on arrival."""
    made, _ = clouds
    monkeypatch.setattr(sphere, "TILE_CACHE", 0)
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
