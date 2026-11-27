"""The depth convention's one source (FRAME_COST_PLAN.md, Phase 2): the shared
``uDepthMode`` (world_builder.depth) and shaders/depth_convention.glsl, and the
renderers that pin or read depth through them."""

from __future__ import annotations

from pathlib import Path

import pytest
from panda3d.core import DepthTestAttrib

import world_builder
from world_builder import depth

PACKAGE = Path(world_builder.__file__).parent
CONVENTION = PACKAGE / "shaders" / "depth_convention.glsl"


def test_the_convention_pins_the_far_plane_in_every_mode():
    """One function places a vertex at the far plane: the split's standard
    depth (just short of 1), reversed 0-1 (just above 0: infinity), reversed
    [-1, 1] (just above -1)."""
    source = CONVENTION.read_text()
    assert "uniform int uDepthMode;" in source
    assert "float depthFarZ(float w)" in source
    assert "float depthNearZ()" in source
    for helper in (
        "float depthClear()",
        "bool depthDrawn(float depth)",
        "float depthFarNdc()",
        "float depthWindow(vec4 clip)",
    ):
        assert helper in source, helper
    for name in ("DEPTH_STANDARD", "DEPTH_REVERSED_01", "DEPTH_REVERSED_11"):
        assert f"const int {name} = {getattr(depth, name)};" in source


@pytest.mark.parametrize("name", ["sky.vert", "stars.vert"])
def test_the_sky_and_its_stars_pin_depth_through_the_convention(name):
    source = (PACKAGE / "sky" / "shaders" / name).read_text()
    assert '#pragma include "../../shaders/depth_convention.glsl"' in source
    assert "gl_Position.z = depthFarZ(gl_Position.w);" in source
    assert "0.999999" not in source  # no far plane of its own


def test_the_far_plane_test_is_the_cameras_with_equality():
    """What sits at the far plane passes only where nothing nearer was drawn:
    "less or equal" against the standard clear of 1, "greater or equal"
    against the reversed clear of 0 (equal: a 24-bit buffer rounds the far
    plane onto the clear value)."""
    assert depth.far_plane_test(depth.DEPTH_STANDARD) == DepthTestAttrib.M_less_equal
    for mode in (depth.DEPTH_REVERSED_01, depth.DEPTH_REVERSED_11):
        assert depth.far_plane_test(mode) == DepthTestAttrib.M_greater_equal


def test_the_sky_follows_the_depth_mode(showbase):
    """The sky binds the shared mode and takes the far-plane test of the mode
    it is given; its stars inherit both."""
    from world_builder.sky import Sky, Starfield

    sky = Sky(showbase.render, Starfield(stars=50, glow_size=16))
    try:
        assert sky.node.get_shader_input("uDepthMode").get_name().get_name() == (
            "uDepthMode"
        )
        mode = lambda: sky.node.get_attrib(DepthTestAttrib).get_mode()  # noqa: E731
        assert mode() == DepthTestAttrib.M_less_equal
        sky.set_depth_mode(depth.DEPTH_REVERSED_01)
        assert mode() == DepthTestAttrib.M_greater_equal
        assert not sky.stars.has_attrib(DepthTestAttrib)  # inherited
        sky.set_depth_mode(depth.DEPTH_STANDARD)
        assert mode() == DepthTestAttrib.M_less_equal
    finally:
        sky.node.remove_node()


@pytest.mark.parametrize(
    "name",
    [
        "cloud_march.glsl",
        "cloud_resolve.frag",
        "cloud_composite.frag",
        "ground_shadow.frag",
    ],
)
def test_the_clouds_read_depth_through_the_convention(name):
    """No cloud shader assumes the standard clear (1) or near plane (-1): what
    was drawn, the near plane, a direction's far point and the composite's
    own depth go through depth_convention.glsl."""
    source = (PACKAGE / "clouds" / "shaders" / name).read_text()
    assert '#pragma include "../../shaders/depth_convention.glsl"' in source
    assert "depthDrawn(" in source or name == "cloud_composite.frag"
    for standard_only in (
        "< 1.0 ?",
        ".r : 1.0",
        "Depth < 1.0",
        "Depth >= 1.0",
        "ndc, -1.0",
    ):
        assert standard_only not in source, standard_only


def test_no_cloud_shader_unprojects_the_reversed_infinity():
    """Clip z = 0 is infinity in reversed 0-1 depth (w = 0 after unprojection):
    a direction or footprint takes depthFarNdc's finite point instead."""
    for name in ("cloud_march.glsl", "cloud_resolve.frag", "ground_shadow.frag"):
        source = (PACKAGE / "clouds" / "shaders" / name).read_text()
        assert "vec3(ndc, 0.0)" not in source and ", 0.0), 0.0)" not in source, name
