"""
Unit tests for the atmospheres (world_builder.atmosphere) and the star field
(world_builder.sky.starfield). The shaders need a window (see
scripts/terrain_shots.py); these check the CPU side.
"""

import numpy as np
import pytest
from panda3d.core import NodePath, Texture

from world_builder.atmosphere import (
    MAX_ATMOSPHERES,
    AtmosphereParams,
    Atmospheres,
    bind_defaults,
    depth_toward,
    in_shadow,
    twilight,
)
from world_builder.sky import NEBULA_PALETTES, NebulaStyle, Sky, Starfield
from world_builder.sky.starfield import (
    STAR_BLUE,
    STAR_RED,
    cube_directions,
    directions_to_texels,
    glow_texture,
    make_glow,
    make_stars,
)

R = 600e3


def test_uniforms_follow_the_camera_in_float64():
    atmospheres = Atmospheres()
    params = AtmosphereParams()
    index = atmospheres.add(R, params, position=(1e7, 0.0, 0.0))
    camera = np.array([1e7 + R + 1500.0, 0.0, 0.0])
    atmospheres.update(camera)
    cam = atmospheres._arrays["uAtmoCam"][index]
    assert cam[0] == pytest.approx(R + 1500.0)
    # |cam|^2 - top^2, computed before rounding to float32.
    top = R + params.height
    assert cam[3] == pytest.approx((R + 1500.0) ** 2 - top**2, rel=1e-6)
    mie = atmospheres._arrays["uAtmoMie"][index]
    assert mie[3] == pytest.approx((R + 1500.0) ** 2 - R**2, rel=1e-6)


def test_at_most_max_atmospheres():
    atmospheres = Atmospheres()
    for _ in range(MAX_ATMOSPHERES):
        atmospheres.add(R, AtmosphereParams())
    with pytest.raises(ValueError):
        atmospheres.add(R, AtmosphereParams())


def test_sunlight_through_the_air():
    atmospheres = Atmospheres()
    params = AtmosphereParams()
    atmospheres.add(R, params)
    ground = np.array([0.0, 0.0, R + 1.0])
    overhead = atmospheres.transmittance_to_sun(0, ground, (0.0, 0.0, 1.0))
    expected = np.exp(
        -(
            np.asarray(params.rayleigh) * params.rayleigh_height
            + params.mie_extinction * params.mie_height
        )
    )
    np.testing.assert_allclose(overhead, expected, rtol=0.02)
    assert overhead[2] < overhead[0]  # blue scattered out first
    low = atmospheres.transmittance_to_sun(0, ground, (1.0, 0.0, 0.08))
    assert np.all(low < overhead)  # a low sun is dimmer and redder
    assert low[2] / low[0] < overhead[2] / overhead[0]
    night = atmospheres.transmittance_to_sun(0, ground, (0.0, 0.0, -1.0))
    np.testing.assert_array_equal(night, 0.0)  # the planet is in the way
    space = np.array([0.0, 0.0, R + 200e3])
    np.testing.assert_array_equal(
        atmospheres.transmittance_to_sun(0, space, (0.0, 0.0, 1.0)), 1.0
    )


def test_sunset_radius_reddens_only_a_low_sun():
    """A larger sunset radius lengthens the grazing path (redder near the
    terminator) and leaves the sun overhead as it was."""
    physical, earthlike = Atmospheres(), Atmospheres()
    physical.add(R, AtmosphereParams(sunset_radius=None))
    earthlike.add(R, AtmosphereParams(sunset_radius=6371e3))
    ground = np.array([0.0, 0.0, R + 1.0])
    overhead = (0.0, 0.0, 1.0)
    np.testing.assert_allclose(
        earthlike.transmittance_to_sun(0, ground, overhead),
        physical.transmittance_to_sun(0, ground, overhead),
        rtol=1e-2,  # the Chapman function's curvature term, a few tenths of 1 %
    )
    low = (1.0, 0.0, 0.05)
    a = physical.transmittance_to_sun(0, ground, low)
    b = earthlike.transmittance_to_sun(0, ground, low)
    assert b[2] / b[0] < 0.5 * (a[2] / a[0])  # much redder


def test_slant_depth_is_continuous_across_the_horizon():
    """Above and below the horizontal, the two branches of the Chapman
    formula meet."""
    r = R + 2000.0
    for curve in (1.0, 10.0):
        up = depth_toward(r, 1e-6, R, 8000.0, curve)
        down = depth_toward(r, -1e-6, R, 8000.0, curve)
        assert down == pytest.approx(up, rel=1e-3)


def test_light_at_follows_the_sun_and_the_planet_shadow():
    atmospheres = Atmospheres()
    atmospheres.add(R, AtmosphereParams())
    ground = np.array([0.0, 0.0, R + 10.0])
    sun, sky = (2.6, 2.5, 2.35), (0.3, 0.4, 0.6)
    noon_direct, noon_sky = atmospheres.light_at(0, ground, (0, 0, 1), sun, sky)
    assert np.all(noon_direct > 0.5 * np.asarray(sun))
    np.testing.assert_allclose(noon_sky, sky)
    night_direct, night_sky = atmospheres.light_at(0, ground, (0, 0, -1), sun, sky)
    np.testing.assert_array_equal(night_direct, 0.0)
    np.testing.assert_array_equal(night_sky, 0.0)
    assert twilight(0.0) == pytest.approx(0.5)


def test_in_shadow_is_the_bodys_geometric_shadow():
    assert in_shadow((0.0, 0.0, -2.0), (0.0, 0.0, 1.0), 1.0)
    assert not in_shadow((0.0, 0.0, 2.0), (0.0, 0.0, 1.0), 1.0)
    assert not in_shadow((1.5, 0.0, -2.0), (0.0, 0.0, 1.0), 1.0)


def test_defaults_bind_every_input():
    node = NodePath("n")
    bind_defaults(node)
    for name in (
        "uAtmoCount",
        "uAtmoCam",
        "uAtmoShape",
        "uAtmoScatter",
        "uAtmoMie",
        "uAtmoExtra",
    ):
        # A missing input comes back blank: value type 0 (M_invalid).
        assert node.get_shader_input(name).get_value_type() != 0
    assert NodePath("unbound").get_shader_input("uAtmoCount").get_value_type() == 0


# ── star field ─────────────────────────────────────────────────────────────────


def test_cube_texels_round_trip():
    size = 16
    dirs = cube_directions(size)
    face, row, col = directions_to_texels(dirs.reshape(-1, 3), size)
    expected_face = np.repeat(np.arange(6), size * size)
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    np.testing.assert_array_equal(face, expected_face)
    np.testing.assert_allclose(row, np.tile(rows.ravel(), 6) + 0.5, atol=1e-9)
    np.testing.assert_allclose(col, np.tile(cols.ravel(), 6) + 0.5, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=-1), 1.0)


def test_faces_follow_opengl_order():
    """+X -X +Y -Y +Z -Z: each face's centre texel looks down its axis."""
    dirs = cube_directions(2)
    centres = dirs.mean(axis=(1, 2))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    axes = np.array(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    )
    np.testing.assert_allclose(centres, axes, atol=1e-12)


def test_stars_crowd_the_band_and_are_mostly_faint():
    pole = np.array([0.0, 0.0, 1.0])
    stars = make_stars(seed=1, count=20000, galactic_pole=pole)
    assert stars.dirs.shape == (20000, 3) and stars.dirs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(stars.dirs, axis=1), 1.0, atol=1e-6)
    near_band = np.abs(stars.dirs[:, 2]) < 0.1
    near_pole = np.abs(stars.dirs[:, 2]) > 0.9
    # Equal areas of sky (bands of equal z width): the band holds more.
    assert near_band.sum() > 1.5 * near_pole.sum()
    # A power law: a few bright stars, most faint.
    assert np.percentile(stars.flux, 99) > 5.0 * np.median(stars.flux)


def test_some_stars_are_slightly_red_and_some_slightly_blue():
    stars = make_stars(seed=2, count=10000, red=0.25, blue=0.2)
    c = stars.colour
    reddish = (c[:, 0] > c[:, 2] + 0.05).mean()
    bluish = (c[:, 2] > c[:, 0] + 0.05).mean()
    white = np.all(np.abs(c - 1.0) < 1e-6, axis=1).mean()
    assert reddish == pytest.approx(0.25, abs=0.02)
    assert bluish == pytest.approx(0.2, abs=0.02)
    assert white == pytest.approx(0.55, abs=0.02)
    # Slightly: never past the full tint.
    assert c.max() <= max(STAR_RED.max(), STAR_BLUE.max()) + 1e-6
    assert c.min() >= min(STAR_RED.min(), STAR_BLUE.min()) - 1e-6


def test_glow_holds_the_band_and_nebula_parameters():
    pole = np.array([0.0, 0.0, 1.0])
    glow = make_glow(size=32, seed=1, galactic_pole=pole)
    assert glow.shape == (6, 32, 32, 3)
    assert glow.min() >= 0.0 and glow.max() <= 1.0
    # The band (r) along the galactic equator, not at the poles (+Z, -Z).
    equator = glow[0:4, 14:18, 14:18, 0].mean()
    poles = glow[4:6, 14:18, 14:18, 0].mean()
    assert equator > 5.0 * poles
    # More coverage, more nebula.
    thin = make_glow(size=32, seed=1, nebula_coverage=0.3)[..., 1]
    thick = make_glow(size=32, seed=1, nebula_coverage=0.8)[..., 1]
    assert (thick > 0).mean() > (thin > 0).mean() + 0.2


def test_glow_texture_is_a_cube_map():
    tex = glow_texture(make_glow(size=16, seed=2))
    assert tex.get_texture_type() == Texture.TT_cube_map
    assert tex.get_x_size() == 16
    assert len(tex.get_ram_image()) == 6 * 16 * 16 * 3


def test_sky_draws_point_stars_and_tunes_the_nebula(showbase):
    sky = Sky(showbase.render, Starfield(stars=500, glow_size=16))
    try:
        assert sky.stars is not None
        geom = sky.stars.node().get_geom(0)
        assert geom.get_vertex_data().get_num_rows() == 500
        style = NebulaStyle(palette=NEBULA_PALETTES["ember"], intensity=0.05)
        sky.set_nebula(style)
        assert sky.nebula == style
    finally:
        sky.remove()
