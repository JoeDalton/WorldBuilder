"""
The clouds in front of a game's translucent surfaces (world_builder.clouds.cover,
cloud_cover.glsl): a translucent card reading cloudCoverBefore, drawn over
a stand-in cloud of known coverage and distance.
"""

import numpy as np
import pytest
from panda3d.core import (
    CardMaker,
    NodePath,
    PerspectiveLens,
    Shader,
    Texture,
    TransparencyAttrib,
)

from world_builder.clouds.cover import GLSL_DIR, CloudCover

#: A straight-alpha red surface, faded by the cloud in front of it (as a
#: shield bubble does)
_VERT = """
#version 330
in vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
out vec3 vView;
void main() {
    vView = (p3d_ModelViewMatrix * p3d_Vertex).xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""
_FRAG = """
#version 330
#pragma include "cloud_cover.glsl"
in vec3 vView;
out vec4 fragColor;
void main() {
    float cover = cloudCoverBefore(gl_FragCoord.xy, length(vView));
    fragColor = vec4(1.0, 0.0, 0.0, 1.0 - cover);
}
"""


@pytest.fixture(scope="module")
def window(showbase):
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


def _cloud(alpha: float, km: float) -> tuple[Texture, Texture]:
    """A stand-in for the march's result: one texel of cloud, *alpha* opaque,
    *km* from the camera."""
    colour, distance = Texture("colour"), Texture("distance")
    colour.setup_2d_texture(1, 1, Texture.T_float, Texture.F_rgba32)
    colour.set_clear_color((alpha, alpha, alpha, alpha))
    distance.setup_2d_texture(1, 1, Texture.T_float, Texture.F_rgba32)
    distance.set_clear_color((km, 0.0, 0.0, 0.0))
    for texture in (colour, distance):
        texture.make_ram_image()
    return colour, distance


def _card(window, metres: float) -> NodePath:
    maker = CardMaker("surface")
    maker.set_frame(-1e4, 1e4, -1e4, 1e4)
    card = window.render.attach_new_node(maker.generate())
    card.set_pos(0.0, metres, 0.0)
    card.set_transparency(TransparencyAttrib.M_alpha)
    card.set_depth_write(False)
    card.set_two_sided(True)
    card.set_shader(Shader.make(Shader.SL_GLSL, _VERT, _FRAG))
    return card


def _red(window) -> float:
    """The mean red of the view, 0-255, over a black background."""
    window.graphicsEngine.render_frame()
    window.graphicsEngine.render_frame()
    shot = window.win.get_screenshot()
    data = np.frombuffer(shot.get_ram_image_as("RGB"), dtype=np.uint8)
    return float(data.reshape(-1, 3)[:, 0].mean())


@pytest.fixture
def scene(window):
    window.set_background_color(0, 0, 0, 1)
    lens = PerspectiveLens()
    lens.set_near_far(1.0, 1e5)
    lens.set_fov(60.0)
    window.cam.node().set_lens(lens)
    window.camera.set_pos_hpr(0, 0, 0, 0, 0, 0)
    cover = CloudCover()
    yield window, cover
    cover.clear()
    window.cam.node().clear_tag_states()
    window.cam.node().set_tag_state_key("")


def test_a_cloud_in_front_hides_the_surface_one_behind_does_not(scene):
    window, cover = scene
    colour, distance = _cloud(1.0, 2.0)  # a dense cloud 2 km away
    for metres, red in ((1000.0, 255.0), (3000.0, 0.0)):
        card = _card(window, metres)
        cover.track(card)
        cover.update((colour, distance, (1.0, 1.0)))
        try:
            assert _red(window) == pytest.approx(red, abs=2.0), metres
        finally:
            cover.untrack(card)
            card.remove_node()


def test_a_thin_cloud_lets_the_surface_through(scene):
    window, cover = scene
    colour, distance = _cloud(0.25, 2.0)
    card = _card(window, 3000.0)
    cover.track(card)
    cover.update((colour, distance, (1.0, 1.0)))
    try:
        assert _red(window) == pytest.approx(0.75 * 255.0, abs=3.0)
    finally:
        card.remove_node()


def test_no_clouds_and_other_views_cover_nothing(scene):
    window, cover = scene
    card = _card(window, 3000.0)
    cover.track(card)
    try:
        cover.update(None)  # no clouds (open space, hidden, loading)
        assert _red(window) == pytest.approx(255.0, abs=2.0)
        cover.update((*_cloud(1.0, 2.0), (1.0, 1.0)))
        assert _red(window) == pytest.approx(0.0, abs=2.0)
        # A view the clouds are not marched for (a rear-view mirror's)
        CloudCover.off_in(window.cam)
        assert _red(window) == pytest.approx(255.0, abs=2.0)
    finally:
        card.remove_node()


def test_the_glsl_is_where_its_include_finds_it():
    assert (GLSL_DIR / "cloud_cover.glsl").is_file()
