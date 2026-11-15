"""
The union march's geometry and genus model (clouds.march.sky_bands,
band_segments; noise.GenusField, pack_genus_params; genera.GENUS_PRESETS):
genera whose slabs may overlap, marched as one.
"""

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.genera import GENUS_PRESETS
from world_builder.clouds.march import band_segments, family_segments, sky_bands
from world_builder.clouds.noise import (
    COVERAGE_SOURCES,
    DETAIL_LEVELS,
    GENUS_VEC4S,
    MAX_GENERA,
    GenusField,
    pack_genus_params,
    pack_layer_params,
)

R = 600e3


# ── Sky bands ────────────────────────────────────────────────────────────────


def test_disjoint_slabs_stay_apart_sorted_by_base():
    assert sky_bands([(2200.0, 8300.0), (900.0, 500.0)]) == [
        (900.0, 500.0),
        (2200.0, 8300.0),
    ]


def test_overlapping_slabs_merge():
    assert sky_bands([(2200.0, 8300.0), (3000.0, 1000.0), (9000.0, 3000.0)]) == [
        (2200.0, 9800.0)
    ]


def test_nested_and_touching_slabs_merge():
    assert sky_bands([(1000.0, 1000.0), (2000.0, 500.0)]) == [(1000.0, 1500.0)]
    assert sky_bands([(1000.0, 5000.0), (2000.0, 500.0)]) == [(1000.0, 5000.0)]


def test_more_than_four_bands_is_refused():
    slabs = [(1000.0 * k, 100.0) for k in range(1, 6)]
    with pytest.raises(ValueError):
        sky_bands(slabs)


@pytest.mark.parametrize("height, sign", [(100.0, 1.0), (20e3, -1.0), (1500.0, 1.0)])
def test_band_segments_are_the_families_on_disjoint_slabs(height, sign):
    slabs = [(2200.0, 8300.0), (900.0, 500.0)]
    up = np.array([0.0, 0.6, 0.8])
    tilt = np.array([0.8, 0.0, -0.6])
    for direction in (sign * up, sign * up * 0.2 + tilt):
        direction = direction / np.linalg.norm(direction)
        origin = up * (R + height)
        old = [(t0, t1) for t0, t1, _ in family_segments(origin, direction, R, slabs)]
        new = [(t0, t1) for t0, t1, _ in band_segments(origin, direction, R, slabs)]
        np.testing.assert_allclose(new, old)


def test_band_segments_name_their_band():
    up = np.array([0.0, 0.0, 1.0])
    segments = band_segments(
        up * (R + 100.0), up, R, [(2200.0, 8300.0), (900.0, 500.0)]
    )
    # Looking up from under both: the stratus's band (0, the lower) first
    assert [band for *_, band in segments] == [0, 1]


# ── The genus model ──────────────────────────────────────────────────────────


def _resolved(kind=CloudType.STRATUS):
    from dataclasses import replace

    return replace(PRESETS[kind].field, threshold=(0.45, 0.55))


def test_a_genus_packs_its_field_rows_then_its_coverage_row():
    field = _resolved()
    genus = GenusField(
        "stratus", field, PRESETS[CloudType.STRATUS].optics, ("A", 2), 1.0
    )
    rows = pack_genus_params(genus, 0.02, octave_mean=0.5, octaves=4)
    assert rows.shape == (GENUS_VEC4S, 4)
    expected = pack_layer_params(
        field, 0.02, octave_mean=0.5, optics=genus.optics, octaves=4
    )
    np.testing.assert_array_equal(rows[: len(expected)], expected)
    assert tuple(rows[13]) == (COVERAGE_SOURCES["A"], 2.0, 0.0, 1.0)
    assert not rows[14:].any()


def test_the_primary_genus_reads_the_weather_cube():
    genus = GenusField(
        "convective",
        _resolved(CloudType.CONVECTIVE),
        PRESETS[CloudType.CONVECTIVE].optics,
        ("weather", 0),
        0.0,
    )
    rows = pack_genus_params(genus, 0.02, shift_row=0)
    assert rows[13, 0] == COVERAGE_SOURCES["weather"] == 0


def test_the_uniforms_fit_the_gl_minimum():
    """Every genus's rows and the deck's, the per-(genus, level) threshold
    offsets (4 a vec4) and a few vec4s of detail bands: under GL 3.3's
    guaranteed 1024 components (256 vec4) of a stage's uniforms."""
    params = (MAX_GENERA + 1) * GENUS_VEC4S
    offsets = MAX_GENERA * DETAIL_LEVELS // 4
    assert params + offsets + 16 <= 256


def test_the_presets_name_the_two_genera_of_phase_3():
    convective = GENUS_PRESETS["convective"]
    stratus = GENUS_PRESETS["stratus"]
    assert convective.cloud_type is CloudType.CONVECTIVE
    assert convective.coverage == ("weather", 0) and convective.far_shadow == 0.0
    assert stratus.cloud_type is CloudType.STRATUS
    assert stratus.coverage == ("A", 0) and stratus.far_shadow == 1.0


# ── Coverage cubes, the shift texture, the deck light array ─────────────────


@pytest.fixture(scope="module")
def genera_spec():
    from world_builder.clouds.genera import GeneraSpec
    from world_builder.clouds.weather import WeatherSpec
    from world_builder.terrain.planet import PlanetSpec

    return GeneraSpec(WeatherSpec(PlanetSpec(name="clouds", radius=R)))


def test_cube_a_holds_the_stratus_coverage_in_its_channel(genera_spec):
    from world_builder.clouds.genera import coverage_cube, genus_planes

    cube = coverage_cube(genera_spec, "A")
    planes = genus_planes(genera_spec)
    assert cube.shape == planes.shape[:-1] + (4,) and cube.dtype == np.float32
    np.testing.assert_array_equal(cube[..., 0], planes[..., 2])
    assert not cube[..., 1:].any()  # no other genus in A yet


def test_a_genus_coverage_turns_into_its_shift_through_its_table():
    """genus_weather_at: the coverage read off its channel, the shift its
    table's at that coverage; the coverage that shift gives is the one asked
    for (within 5 %)."""
    from dataclasses import replace

    from world_builder.clouds.field_setup import _noise
    from world_builder.clouds.genera import genus_weather_at
    from world_builder.clouds.noise import (
        density_on_sphere,
        fbm_sigma,
        sphere_column_peaks,
        with_threshold,
    )
    from world_builder.clouds.weather import shift_table

    raw = replace(PRESETS[CloudType.STRATUS].field, slab=(900.0, 500.0))
    volume = _noise(7)
    peaks = sphere_column_peaks(volume, raw, R)
    field = with_threshold(raw, peaks, fbm_sigma(volume, raw))
    table = shift_table(peaks, field.threshold[0])
    for coverage in (0.3, 0.7):
        cube = np.zeros((6, 4, 4, 4), dtype=np.float32)
        cube[..., 1] = coverage  # channel 1
        rng = np.random.default_rng(2)
        dirs = rng.normal(size=(4000, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        got, shift = genus_weather_at(cube, 1, table, dirs)
        np.testing.assert_allclose(got, coverage, atol=1e-6)
        weather = np.stack([got, shift], -1)
        weather = np.broadcast_to(weather[0], (6, 4, 4, 2)).astype(np.float32)
        heights = R + 900.0 + (np.arange(24) + 0.5) / 24 * 500.0
        points = dirs[:, None, :] * heights[None, :, None]
        cloudy = (density_on_sphere(volume, weather, field, points, R) > 0).any(1)
        assert abs(cloudy.mean() - coverage) < 0.05 * max(coverage, 0.1) + 0.02


def test_the_shift_texture_holds_one_table_a_row():
    from world_builder.clouds.weather import SHIFT_TABLE_SIZE, build_shift_texture

    tables = [
        np.linspace(-0.3, 0.2, SHIFT_TABLE_SIZE),
        np.linspace(0.1, 0.4, SHIFT_TABLE_SIZE),
    ]
    tex = build_shift_texture(tables)
    assert (tex.get_x_size(), tex.get_y_size()) == (SHIFT_TABLE_SIZE, MAX_GENERA)
    ram = np.frombuffer(tex.get_ram_image(), np.float32).reshape(MAX_GENERA, -1)
    np.testing.assert_allclose(ram[0], tables[0], rtol=1e-6)
    np.testing.assert_allclose(ram[1], tables[1], rtol=1e-6)
    assert not ram[2:].any()


def test_the_deck_light_array_holds_one_genus_a_page():
    from world_builder.clouds.noise import build_deck_light_array

    tables = [
        np.full((8, 16, 4), 0.25, np.float32),
        np.full((8, 16, 4), 0.75, np.float32),
    ]
    tex = build_deck_light_array(tables)
    assert (tex.get_x_size(), tex.get_y_size(), tex.get_z_size()) == (16, 8, 2)
    ram = np.frombuffer(tex.get_ram_image(), np.float32).reshape(2, 8, 16, 4)
    np.testing.assert_allclose(ram[0], 0.25)
    np.testing.assert_allclose(ram[1], 0.75)
