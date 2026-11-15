"""
Diagnostics for cumulus congestus (the convective field at depth 1/2,
clouds.genera's ``congestus`` features): a congestus is a tower, cloud from
its base to a top several kilometres up, not blobs scattered through a tall
slab. Heights are above sea level; the convective slab starts at 2.2 km.

Measured before the fix (2026-11-15; coverage 0.45, depth 1/2, local top
6.85 km): the share of samples that are cloud peaks at 0.14 at 3.3 km and
falls to 0.05 at 4.8 km, 0.02 at 5.3 km and nothing above 5.8 km; only 53 %
of the cloud above 4 km stands over cloud at 3 km. The threshold's rise
toward the local top stops nearly every column 1.5-2 km short of it, and
what rises is scattered: the congestus reads as cumulus mediocris.
"""

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.field_setup import _noise
from world_builder.clouds.genera import (
    DEFAULT_FEATURES,
    depth_shift,
    depth_shift_tables,
)
from world_builder.clouds.noise import (
    density_on_sphere,
    fbm_sigma,
    sphere_column_peaks,
    with_shelf,
    with_threshold,
)

R = 600e3
#: A density above this counts as cloud.
CLOUD = 0.05
#: The demo's congestus features: their core coverage, at depth 1/2.
CONGESTUS_COVERAGE = next(
    f.coverage for f in DEFAULT_FEATURES if f.genus == "congestus"
)
CONGESTUS_DEPTH = 0.5


@pytest.fixture(scope="module")
def convective():
    raw = PRESETS[CloudType.CONVECTIVE].field
    volume = _noise(7)
    peaks = sphere_column_peaks(volume, raw, R)
    field = with_shelf(with_threshold(raw, peaks, fbm_sigma(volume, raw)), volume, R)
    return field, depth_shift_tables(volume, field, R)


def column_cloud_profile(field, tables, coverage, depth, heights, columns=3000, seed=0):
    """(columns, len(heights)) booleans: cloud at each height of random
    columns under a weather of *coverage* and *depth* everywhere."""
    cube = np.empty((6, 4, 4, 4), dtype=np.float32)
    cube[..., 0] = coverage
    cube[..., 1] = depth_shift(
        np.full((6, 4, 4), coverage), np.full((6, 4, 4), depth), tables
    )
    cube[..., 2] = depth
    cube[..., 3] = 0.0  # no anvil shelf
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(columns, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = dirs[:, None, :] * (R + np.asarray(heights))[None, :, None]
    return density_on_sphere(_noise(7), cube, field, points, R) > CLOUD


def _band(heights, lo, hi):
    heights = np.asarray(heights)
    return (heights >= lo) & (heights <= hi)


HEIGHTS = np.arange(2300.0, 7000.0, 100.0)


def test_congestus_reaches_its_top(convective):
    """The share of columns holding cloud at 4-6 km is at least a quarter of
    their share at 2.5-3.5 km: cloud reaches the top of the tower."""
    field, tables = convective
    cloud = column_cloud_profile(
        field, tables, CONGESTUS_COVERAGE, CONGESTUS_DEPTH, HEIGHTS
    )
    low = cloud[:, _band(HEIGHTS, 2500, 3500)].any(axis=1).mean()
    high = cloud[:, _band(HEIGHTS, 4000, 6000)].any(axis=1).mean()
    assert high >= 0.25 * low, (low, high)


def test_congestus_is_towers_not_scatter(convective):
    """Cloud above 4 km stands over cloud at 3 km (within 300 m) in at least
    60 % of its samples: towers connected to their base, not blobs."""
    field, tables = convective
    cloud = column_cloud_profile(
        field, tables, CONGESTUS_COVERAGE, CONGESTUS_DEPTH, HEIGHTS
    )
    under = cloud[:, _band(HEIGHTS, 2700, 3300)].any(axis=1)
    high = cloud[:, _band(HEIGHTS, 4000, 7000)]
    connected = (high & under[:, None]).sum() / max(high.sum(), 1)
    assert connected >= 0.6, connected


def test_congestus_tops_near_its_local_top(convective):
    """Columns holding cloud at 5.5-6.8 km, under the local top (6.85 km),
    are at least 0.4 of those at 2.5-3.5 km: the towers reach their top."""
    field, tables = convective
    cloud = column_cloud_profile(
        field, tables, CONGESTUS_COVERAGE, CONGESTUS_DEPTH, HEIGHTS
    )
    low = cloud[:, _band(HEIGHTS, 2500, 3500)].any(axis=1).mean()
    top = cloud[:, _band(HEIGHTS, 5500, 6800)].any(axis=1).mean()
    assert top >= 0.4 * low, (low, top)


# ── The deep profile (noise.DensityField.deep_exponent, deep_erosion,
# cell_power): neutral by default, a late rise at depth ────────────────────


def _reference_lift(spec, h, depth):
    """threshold_lift as it was before the deep profile (2026-11-15)."""
    from world_builder.clouds.noise import (
        ANVIL_BAND,
        ANVIL_DEPTH,
        _smoothstep,
        top_fraction,
    )

    local = np.asarray(h) / top_fraction(spec, depth)
    lift = spec.top_erosion * np.clip(local, 0.0, 1.0) ** spec.top_exponent
    if spec.anvil:
        lift = lift - spec.anvil * _smoothstep(*ANVIL_DEPTH, depth) * _smoothstep(
            *ANVIL_BAND, local
        )
    if spec.min_top < 1.0:
        lift = np.where(local < 1.0, lift, np.inf)
    return lift


def test_neutral_deep_profile_changes_nothing():
    from dataclasses import replace

    from world_builder.clouds.noise import threshold_lift

    spec = replace(
        PRESETS[CloudType.CONVECTIVE].field, deep_exponent=None, deep_erosion=None
    )
    h = np.linspace(0.0, 1.0, 41)[:, None]
    depth = np.linspace(0.0, 1.0, 11)[None, :]
    got, want = threshold_lift(spec, h, depth), _reference_lift(spec, h, depth)
    assert np.array_equal(np.isinf(got), np.isinf(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=0, atol=1e-12)


def test_the_deep_profile_rises_late_and_leaves_depth_0_alone():
    from dataclasses import replace

    from world_builder.clouds.noise import threshold_lift, top_fraction

    spec = replace(
        PRESETS[CloudType.CONVECTIVE].field, deep_exponent=10.0, deep_erosion=0.25
    )
    neutral = replace(spec, deep_exponent=None, deep_erosion=None)
    at0 = top_fraction(spec, 0.0) * np.array([0.3, 0.6, 0.9])
    np.testing.assert_allclose(
        threshold_lift(spec, at0, 0.0), threshold_lift(neutral, at0, 0.0)
    )
    mid = 0.6 * top_fraction(spec, 1.0)
    assert threshold_lift(spec, mid, 1.0) < 0.2 * threshold_lift(neutral, mid, 1.0)
    # exponent and erosion halfway at depth 1/2
    half = 0.6 * top_fraction(spec, 0.5)
    expected = (0.4 + (0.25 - 0.4) * 0.5) * 0.6 ** (3.0 + (10.0 - 3.0) * 0.5)
    np.testing.assert_allclose(threshold_lift(spec, half, 0.5), expected, rtol=1e-9)


def test_the_cell_octave_grows_as_a_power_of_depth():
    from dataclasses import replace

    from world_builder.clouds.noise import cell_shift

    volume = _noise(7)
    spec = replace(PRESETS[CloudType.CONVECTIVE].field, cell_weight=1.0, cell_power=0.5)
    coords = np.random.default_rng(3).uniform(0, 64, (500, 3))
    full = cell_shift(volume, spec, coords, 1.0)
    np.testing.assert_allclose(
        cell_shift(volume, spec, coords, 0.25), 0.5 * full, rtol=1e-6
    )


def test_the_deep_profile_is_packed_and_read_by_the_shader():
    import re
    from dataclasses import replace
    from pathlib import Path

    from world_builder.clouds.noise import pack_layer_params

    raw = replace(
        PRESETS[CloudType.CONVECTIVE].field,
        threshold=(0.4, 0.5),
        deep_exponent=9.0,
        deep_erosion=0.2,
        cell_power=0.7,
    )
    rows = pack_layer_params(raw, 0.01)
    assert tuple(rows[10, 2:]) == pytest.approx((9.0, 0.2))
    assert rows[12, 3] == pytest.approx(0.7)
    neutral = pack_layer_params(
        replace(raw, deep_exponent=None, deep_erosion=None), 0.01
    )
    assert tuple(neutral[10, 2:]) == pytest.approx((raw.top_exponent, raw.top_erosion))
    text = (
        Path(__file__).parent.parent
        / "src/world_builder/clouds/shaders/cloud_field.glsl"
    ).read_text()
    assert re.search(r"f\.deepExponent\s*=\s*v10\.z;", text)
    assert re.search(r"f\.deepErosion\s*=\s*v10\.w;", text)
    assert re.search(r"f\.cellPower\s*=\s*v12\.w;", text)
