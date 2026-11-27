"""
Unit tests for world_builder.depth: the far distance's maths, culling bounds,
camera masks and the reversed camera.
"""

import math

import numpy as np
import pytest
from panda3d.core import (
    BitMask32,
    BoundingBox,
    BoundingSphere,
    BoundingVolume,
    CardMaker,
    NodePath,
    PerspectiveLens,
    Point3,
    TransformState,
    Vec3,
)

from world_builder.depth import (
    FAR_CAMERA_MASK,
    HIDE_FROM_FAR,
    HIDE_FROM_NEAR,
    NEAR_CAMERA_MASK,
    frustum_bounds,
    planet_far,
    system_far,
)

R = 600e3


@pytest.mark.parametrize("altitude", [2.0, 1000.0, 20e3, 200e3, 1800e3])
def test_the_far_distance_reaches_past_the_horizon_of_the_highest_ground(altitude):
    far = planet_far(R, R + altitude)
    horizon = math.sqrt((R + altitude) ** 2 - R**2) + math.sqrt(
        (R + 9000.0) ** 2 - R**2
    )
    assert far >= horizon


def test_the_far_distance_reaches_the_farthest_body():
    moon = (200e3, 9.4e6, 9000.0, -7000.0)  # 9,400 km from the camera
    planet = (R, R + 2.0, 9000.0, 0.0)  # standing on the planet
    far = system_far([planet, moon])
    assert far == planet_far(*moon) and far >= 9.4e6


def _lens(near, far):
    lens = PerspectiveLens()
    lens.set_fov(70.0, 42.0)
    lens.set_near_far(near, far)
    return lens


def _reaches(bounds, point) -> bool:
    sphere = BoundingSphere(Point3(*point), 1e-3)
    return bounds.contains(sphere) != BoundingVolume.IF_no_intersection


def test_frustum_bounds_match_the_lens_within_its_own_limit():
    lens = _lens(500e3, 5e6)
    ours, panda = frustum_bounds(lens), lens.make_bounds()
    for y in (400e3, 600e3, 2e6, 4.9e6, 5.1e6):
        for x in (-4e6, -0.6 * y, 0.0, 0.6 * y, 4e6):
            for z in (-0.3 * y, 0.0, 0.5 * y):
                assert _reaches(ours, (x, y, z)) == _reaches(panda, (x, y, z))


def test_frustum_bounds_reach_past_ten_thousand_km():
    """Panda's own frustum stops at 1 / lens-far-limit (10,000 km): a planet
    beyond its moon was culled while the depth range covered it."""
    lens = _lens(570e3, 12e6)
    planet = BoundingBox(Point3(-1e6, 10.2e6, -1e6), Point3(1e6, 11.4e6, 1e6))
    assert lens.make_bounds().contains(planet) == BoundingVolume.IF_no_intersection
    assert frustum_bounds(lens).contains(planet) != BoundingVolume.IF_no_intersection
    assert _reaches(frustum_bounds(lens), (0.0, 11.9e6, 0.0))
    assert not _reaches(frustum_bounds(lens), (0.0, 12.1e6, 0.0))


def test_frustum_bounds_stay_exact_near_the_camera_in_a_models_frame():
    """Panda culls a model's parts with the volume carried into the model's
    frame. A hexahedron rebuilt its planes there from corners 20,000 km out,
    in single precision, a metre or two off near the camera: a game's
    cockpit lost its panels at the screen's edge. Planes carry over exactly:
    a part a few centimetres inside the bottom edge, 1.5 m off, stays in,
    and one a few centimetres outside stays out."""
    lens = _lens(1.0, 2.2e7)
    half_z = math.tan(math.radians(lens.get_fov()[1]) / 2.0)
    # a ship turned anyhow, its cockpit model offset, turned and scaled
    ship = TransformState.make_hpr(Vec3(120.4, 69.472, 0.0))
    model = TransformState.make_pos_hpr_scale(
        Vec3(0.0, 0.8, -0.2), Vec3(180.0, 0.0, 0.0), Vec3(0.8)
    )
    net = ship.compose(model)
    for inside, margin in ((True, -0.03), (False, 0.03)):
        # 1.5 m ahead, *margin* metres off the bottom plane (outward +)
        y = 1.5
        z = -(half_z * y) - margin / math.cos(math.atan(half_z))
        part = BoundingSphere(Point3(0.0, y, z), 0.01)
        bounds = frustum_bounds(lens)
        bounds.xform(net.get_inverse().get_mat())
        part.xform(net.get_inverse().get_mat())
        assert (bounds.contains(part) != BoundingVolume.IF_no_intersection) == inside


def _drawn(node: NodePath, camera_mask) -> bool:
    """Panda's rule (PandaNode::compare_draw_mask): drawn if any of the
    camera's bits is shown, or not controlled, on the node."""
    n = node.node()
    shown = n.get_draw_show_mask() | ~n.get_draw_control_mask()
    return not (shown & camera_mask).is_zero()


def test_hiding_from_both_view_bits_keeps_a_node_out_of_the_view_alone():
    """The view camera carries both bits: a node hidden from one is still
    drawn, hidden from both it is not; the sea's reflection still sees it
    (the clouds' decks)."""
    view = NEAR_CAMERA_MASK | FAR_CAMERA_MASK
    reflection = BitMask32(0xFFFFF)  # the ocean's reflection camera's bits
    one = NodePath("one")
    one.hide(HIDE_FROM_NEAR)
    assert _drawn(one, view)
    both = NodePath("both")
    both.hide(HIDE_FROM_NEAR | HIDE_FROM_FAR)
    assert not _drawn(both, view) and _drawn(both, reflection)


# ── the single reversed-Z camera (FRAME_COST_PLAN.md, Task 6.1) ─────────────


def test_depth_mode_follows_the_clip_control_setting():
    """0 is the split's standard depth; 1 and 2 the reversed camera's, with
    OpenGL's 0-1 depth (clip control) or its [-1, 1] (macOS)."""
    from world_builder import depth

    assert depth.DEPTH_STANDARD == 0
    expected = (
        depth.DEPTH_REVERSED_01 if depth.zero_to_one() else depth.DEPTH_REVERSED_11
    )
    assert depth.reversed_depth_mode() == expected
    assert {depth.DEPTH_REVERSED_01, depth.DEPTH_REVERSED_11} == {1, 2}


def test_a_reversed_lens_maps_near_to_one_and_infinity_to_minus_one():
    """Panda's lens as near = inf, far = the near plane: reversed and infinite
    (OpenGL's [-1, 1] in the lens; the GSG maps it to 1..0 with clip control)."""
    from panda3d.core import LVecBase4f

    from world_builder.depth import reversed_lens

    lens = PerspectiveLens()
    lens.set_fov(60.0)
    reversed_lens(lens, 2.0)
    m = lens.get_projection_mat()

    def ndc(d):
        p = m.xform(LVecBase4f(0.0, d, 0.0, 1.0))
        return p[2] / p[3]

    assert ndc(2.0) == pytest.approx(1.0, abs=1e-6)
    assert ndc(4.0) == pytest.approx(0.0, abs=1e-6)
    assert ndc(1e3) > ndc(1e6) >= ndc(1e9)  # toward -1 at infinity
    assert ndc(1e9) == pytest.approx(-1.0, abs=1e-6)
    assert math.isinf(lens.get_near()) and lens.get_far() == pytest.approx(2.0)


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


def test_the_reversed_camera_draws_instead_of_the_app_camera_and_gives_it_back(window):
    """Its own camera draws the view into a scene buffer of its own (float
    depth: a window's framebuffer has none), which a card presents in the
    window; the app camera's region is off meanwhile and its lens stays
    standard (a game's picking)."""
    from panda3d.core import DepthTestAttrib

    from world_builder.depth import (
        DEPTH_STANDARD,
        ReversedDepthCamera,
        depth_mode_input,
        reversed_depth_mode,
    )

    app = window
    app_region = app.cam.node().get_display_region(0)
    app_near = app.camLens.get_near()
    assert not app.win.get_fb_properties().get_float_depth()
    camera = ReversedDepthCamera(app, near=1.0)
    try:
        node = camera.camera.node()
        mask = node.get_camera_mask()
        # both of the split's bits: nodes kept to one range are drawn
        assert (mask & NEAR_CAMERA_MASK) == NEAR_CAMERA_MASK
        assert (mask & FAR_CAMERA_MASK) == FAR_CAMERA_MASK
        assert camera.region.get_camera() == camera.camera
        assert camera.output != app.win and camera.float_depth
        assert (camera.output.get_x_size(), camera.output.get_y_size()) == (
            app.win.get_x_size(),
            app.win.get_y_size(),
        )
        assert camera.present.get_window() == app.win and camera.present.is_active()
        assert camera.region.get_clear_depth_active()
        assert camera.region.get_clear_depth() == 0.0
        assert not app_region.is_active()
        state = node.get_initial_state()
        mode = state.get_attrib(DepthTestAttrib).get_mode()
        assert mode == DepthTestAttrib.M_greater
        bound = app.render.get_shader_input("uDepthMode")
        assert bound.get_name().get_name() == "uDepthMode"
        assert camera.mode == reversed_depth_mode()
        # the one shared input every renderer binds
        assert depth_mode_input()[0] == camera.mode
        app.camLens.set_fov(70.0)
        camera.update(far=5.0e4)
        assert math.isinf(camera.lens.get_near())
        assert camera.lens.get_far() == pytest.approx(camera.near)
        assert camera.lens.get_fov()[0] == pytest.approx(70.0)
        assert camera.near == pytest.approx(1.0) and camera.far == 5.0e4
        assert app.camLens.get_near() == pytest.approx(app_near)  # standard
        bounds = node.get_cull_bounds()
        assert _reaches(bounds, (0.0, 4.9e4, 0.0))
        assert not _reaches(bounds, (0.0, 5.1e4, 0.0))
        scene = camera.output
    finally:
        camera.destroy()
    assert app_region.is_active()
    assert camera.camera.is_empty()
    assert camera.present not in list(app.win.get_display_regions())
    assert scene not in [
        app.graphicsEngine.get_window(i)
        for i in range(app.graphicsEngine.get_num_windows())
    ]
    assert depth_mode_input()[0] == DEPTH_STANDARD


def test_the_reversed_camera_draws_into_a_float_window_itself(window):
    """Given a window with float depth (or told to), no scene buffer: it
    draws into the app camera's output."""
    from world_builder.depth import ReversedDepthCamera

    app = window
    camera = ReversedDepthCamera(app, near=1.0, scene_buffer=False)
    try:
        assert camera.output == app.win and camera.present is None
        assert camera.region.get_window() == app.win
    finally:
        camera.destroy()


@pytest.mark.parametrize("distance", [5.0, 5.0e4])
def test_the_reversed_camera_hides_what_is_behind(window, distance):
    """A red card at *distance* and a blue one 5 % behind it, drawn after
    it: the red one stays in front (Task 6.4: anything drawn through Panda's
    projection, models and trees, sorts by the reversed depth as is)."""

    from world_builder.depth import ReversedDepthCamera

    app = window
    app.camera.set_pos_hpr(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    cards = []
    for colour, d, order in (
        ((1, 0, 0, 1), distance, 0),
        ((0, 0, 1, 1), distance * 1.05, 1),
    ):
        maker = CardMaker("card")
        half = d  # wider than the view at any distance
        maker.set_frame(-half, half, -half, half)
        card = app.render.attach_new_node(maker.generate())
        card.set_pos(0.0, d, 0.0)
        card.set_color(*colour)
        card.set_light_off()
        card.set_shader_off()
        card.set_bin("fixed", order)  # the far one drawn last
        cards.append(card)
    camera = ReversedDepthCamera(app, near=1.0)
    try:
        camera.update(far=distance * 10.0)
        app.graphicsEngine.render_frame()
        app.graphicsEngine.render_frame()
        shot = app.win.get_screenshot()
        data = np.frombuffer(shot.get_ram_image_as("RGB"), dtype=np.uint8)
        pixels = data.reshape(shot.get_y_size(), shot.get_x_size(), 3)
        centre = pixels[shot.get_y_size() // 2, shot.get_x_size() // 2]
        assert centre[0] > 200 and centre[2] < 50, centre
    finally:
        camera.destroy()
        for card in cards:
            card.remove_node()


def test_the_reversed_camera_copies_the_views_depth_for_the_next_frame(window):
    """The view's depth, copied into a texture after each frame (not bound:
    the clouds' composite reads it while the view draws): what the clouds
    read instead of drawing the scene's depth again (Task 7)."""
    from panda3d.core import GraphicsOutput

    from world_builder.depth import ReversedDepthCamera

    camera = ReversedDepthCamera(window, near=1.0)
    try:
        texture = camera.depth_texture
        output = camera.output
        planes = [
            output.get_rtm_mode(i)
            for i in range(output.count_textures())
            if output.get_texture(i) == texture
        ]
        assert planes == [GraphicsOutput.RTM_copy_texture]
        camera.update(1.0e5)
        window.graphicsEngine.render_frame()
        assert (texture.get_x_size(), texture.get_y_size()) == (
            output.get_x_size(),
            output.get_y_size(),
        )
    finally:
        camera.destroy()


def test_the_view_test_is_less_in_standard_depth_and_greater_reversed():
    from panda3d.core import DepthTestAttrib

    from world_builder import depth

    assert depth.view_test(depth.DEPTH_STANDARD) == DepthTestAttrib.M_less
    for mode in (depth.DEPTH_REVERSED_01, depth.DEPTH_REVERSED_11):
        assert depth.view_test(mode) == DepthTestAttrib.M_greater


def test_the_scene_buffer_follows_the_windows_size(window):
    """A window resized: the next update makes the scene buffer its size
    again (the march and its passes follow the region's size)."""
    from world_builder.depth import ReversedDepthCamera

    camera = ReversedDepthCamera(window, near=1.0)
    try:
        size = (window.win.get_x_size(), window.win.get_y_size())
        camera.output.set_size(size[0] // 2, size[1] // 2)
        assert (camera.output.get_x_size(), camera.output.get_y_size()) != size
        camera.update(1.0e5)
        assert (camera.output.get_x_size(), camera.output.get_y_size()) == size
    finally:
        camera.destroy()


def test_drawing_into_the_window_leaves_its_other_render_textures(window):
    """Without a scene buffer the depth copy is added to the app's own
    output: destroy takes it off and nothing else (a game's own render
    textures stay)."""
    from panda3d.core import GraphicsOutput, Texture

    from world_builder.depth import ReversedDepthCamera

    output = window.win
    theirs = Texture("theirs")
    output.add_render_texture(
        theirs, GraphicsOutput.RTM_copy_ram, GraphicsOutput.RTP_color
    )
    try:
        camera = ReversedDepthCamera(window, near=1.0, scene_buffer=False)
        ours = camera.depth_texture
        textures = [output.get_texture(i) for i in range(output.count_textures())]
        assert ours in textures and theirs in textures
        camera.destroy()
        textures = [output.get_texture(i) for i in range(output.count_textures())]
        assert ours not in textures and theirs in textures
    finally:
        output.clear_render_textures()


@pytest.mark.parametrize("samples", [0, 4])
def test_the_scene_buffer_keeps_the_outputs_multisampling(samples):
    """A game's anti-aliasing pipeline (MSAA on the buffer the view drew
    into): the scene buffer that now holds the view asks for the same
    samples, its float depth besides."""
    from panda3d.core import FrameBufferProperties

    from world_builder.depth import scene_buffer_props

    host = FrameBufferProperties()
    host.set_multisamples(samples)
    props = scene_buffer_props(host)
    assert props.get_multisamples() == samples
    assert props.get_float_depth() and props.get_depth_bits() == 32
    assert (props.get_red_bits(), props.get_alpha_bits()) == (8, 8)
