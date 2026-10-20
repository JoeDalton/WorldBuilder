"""
Unit tests for the atmospheres (world_builder.atmosphere) and the star field
(world_builder.sky.starfield). The shaders need a window (see
scripts/terrain_shots.py); these check the CPU side.
"""

import math

import numpy as np
import pytest
from panda3d.core import NodePath, Texture

from world_builder.atmosphere import (
    MAX_ATMOSPHERES,
    AtmosphereParams,
    Atmospheres,
    bind_defaults,
)
from world_builder.sky.starfield import (
    cube_directions,
    directions_to_texels,
    make_starfield,
    make_starfield_texture,
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


def test_defaults_bind_every_input():
    node = NodePath("n")
    bind_defaults(node)
    for name in ("uAtmoCount", "uAtmoCam", "uAtmoShape", "uAtmoScatter", "uAtmoMie"):
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


def test_starfield_has_stars_and_a_band():
    image = make_starfield(size=64, seed=1, stars=3000)
    assert image.shape == (6, 64, 64, 3)
    assert image.min() >= 0.0
    brightest = np.sort(image.max(axis=-1).ravel())[::-1]
    # Points of light over a dark sky: a few bright texels, most near black.
    assert brightest[20] > 10.0 * np.median(brightest)


def test_starfield_texture_is_a_cube_map():
    tex = make_starfield_texture(size=32, seed=2, stars=500)
    assert tex.get_texture_type() == Texture.TT_cube_map
    assert tex.get_x_size() == 32
    assert len(tex.get_ram_image()) == 6 * 32 * 32 * 3


def test_angular_scale_is_sane():
    # A 1024 cube map resolves ~0.1 degree at a face centre: stars are points.
    assert math.degrees(2.0 * math.atan(1.0 / 1024)) < 0.12
