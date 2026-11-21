"""The clouds' light along a view ray (CloudSettings.light_refresh_*,
cloud_march.glsl's lightStale): taken again as the ray goes on, so a far
cloud behind a near one gets its own light (its sunset colour), not the
near one's; tuned by quality preset, cycled in the demo (L)."""

import math
import sys
from pathlib import Path

import pytest

from world_builder.clouds.cloud import CloudSettings
from world_builder.clouds.march import LIGHT_REFRESH_RADIUS
from world_builder.quality import LIVE, QualityLevel

SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def _takes(growth: float, least: float, first: float, last: float) -> int:
    """How many times lightStale takes the light along a ray of cloud
    samples every metre from *first* to *last* (its CPU twin)."""
    takes, lit_at, t = 1, first, first
    while t < last:
        t += 1.0
        if growth > 0.0 and t > max(growth * lit_at, lit_at + least):
            takes += 1
            lit_at = t
    return takes


def test_the_presets_go_from_coarse_to_fine():
    growth = {
        level: CloudSettings.preset(level).light_refresh_growth
        for level in QualityLevel
    }
    assert growth[QualityLevel.LOW] == 2.0 and growth[QualityLevel.MID] == 1.6
    assert growth[QualityLevel.HIGH] == 1.4 and growth[QualityLevel.ULTRA] == 1.25
    knobs = dict(CloudSettings.knobs())
    assert knobs["light_refresh_growth"].apply == LIVE
    assert 0.0 in knobs["light_refresh_growth"].choices.values()  # once a pixel
    assert LIGHT_REFRESH_RADIUS == 600e3


@pytest.mark.parametrize("growth", [2.0, 1.4, 1.25])
def test_the_takes_along_a_ray_are_bounded_by_its_distances_ratio(growth):
    """However much cloud a grazing ray crosses, the takes grow with the
    log of its far/near distance ratio, not its length."""
    takes = _takes(growth, 2000.0, 2000.0, 300e3)
    assert takes <= math.log(300e3 / 2000.0) / math.log(growth) + 2
    assert _takes(0.0, 2000.0, 2000.0, 300e3) == 1  # once a pixel


def test_the_shader_takes_the_light_again_in_both_loops():
    march = (SHADERS / "cloud_march.glsl").read_text()
    stale = march[march.index("bool lightStale(") :]
    stale = stale[: stale.index("\n}\n")]
    assert "uLightRefresh.y > 0.0" in stale
    assert "max(uLightRefresh.y * litAt, litAt + uLightRefresh.x)" in stale
    assert march.count("if (!lit || lightStale(t, litAt))") == 2
    assert march.count("litAt = t;") == 2


def test_the_demo_cycles_the_light_choices():
    from demo_terrain import LIGHT_REFRESH_CHOICES, _light_refresh_label

    choices = set(dict(CloudSettings.knobs())["light_refresh_growth"].choices.values())
    assert set(LIGHT_REFRESH_CHOICES) == choices
    assert _light_refresh_label(CloudSettings(light_refresh_growth=0.0)) == "once"
    assert _light_refresh_label(CloudSettings()) == "1.4x"
