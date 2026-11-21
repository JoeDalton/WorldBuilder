"""The clouds' sky light through twilight and at night: it fades with the
sunlit share of the air above them (cloud_shading.glsl's cloudSkyFade, its
CPU twin noise.sky_light_fade), so a planet's size sets how long its
clouds' twilight lasts."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from world_builder.atmosphere import AtmosphereParams
from world_builder.clouds.noise import _smoothstep, sky_light_fade

SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"
H = AtmosphereParams().rayleigh_height
KERBIN, EARTH = 600e3, 6.371e6


def _mu(degrees_below: float) -> float:
    return -math.sin(math.radians(degrees_below))


def test_by_day_the_fade_is_the_sun_heights_alone():
    mu = np.linspace(0.0, 1.0, 11)
    for ground in (KERBIN, EARTH):
        np.testing.assert_allclose(
            sky_light_fade(ground + 5000.0, mu, ground, H), _smoothstep(-0.3, 0.1, mu)
        )


def test_earths_clouds_go_dark_at_nautical_twilight():
    """The sun 8 degrees down (the user's capture): the planet's shadow ~60
    km over a cloud 5 km up, the sky's light gone; the bare curve kept 35 %."""
    fade = float(sky_light_fade(EARTH + 5000.0, _mu(7.8), EARTH, H))
    assert fade < 0.01
    assert float(_smoothstep(-0.3, 0.1, _mu(7.8))) > 0.3


def test_a_small_planets_clouds_keep_their_long_twilight():
    """On the 600 km planet the shadow rises slowly: at 8 degrees down a
    cloud 5 km up keeps most of the curve's light, as before."""
    curve = float(_smoothstep(-0.3, 0.1, _mu(8.0)))
    fade = float(sky_light_fade(KERBIN + 5000.0, _mu(8.0), KERBIN, H))
    assert fade == pytest.approx(curve, rel=0.15)
    # Deep night: dark on both
    for ground in (KERBIN, EARTH):
        assert float(sky_light_fade(ground + 5000.0, _mu(25.0), ground, H)) == 0.0


def test_the_fade_falls_as_the_sun_sinks_and_rises_with_height():
    mu = -np.sin(np.radians(np.linspace(0.0, 15.0, 31)))
    fade = sky_light_fade(EARTH + 3000.0, mu, EARTH, H)
    assert np.all(np.diff(fade) <= 1e-12)
    low = sky_light_fade(EARTH + 1000.0, _mu(4.0), EARTH, H)
    high = sky_light_fade(EARTH + 10000.0, _mu(4.0), EARTH, H)
    assert high > low


def test_the_shader_uses_the_same_fade():
    shading = (SHADERS / "cloud_shading.glsl").read_text()
    fade = shading[shading.index("float cloudSkyFade(") :]
    fade = fade[: fade.index("\n}\n")]
    assert "smoothstep(-0.3, 0.1, mu)" in fade
    assert "uAtmoShape[atmo]" in fade
    assert re.search(r"exp\(-max\(lit - r, 0\.0\) / shape\.z\)", fade)
    light = shading[shading.index("void planetCloudLight(") :]
    assert "skylight *= cloudSkyFade(atmo, p, sunDir);" in light[: light.index("\n}\n")]
