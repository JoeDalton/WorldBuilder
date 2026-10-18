"""
Unit tests for the terrain prototype's queries, stamps, levels and materials
(world_builder.terrain). Pure NumPy plus Panda image I/O; no window.
"""

from dataclasses import replace

import numpy as np
import pytest

from world_builder.terrain import materials
from world_builder.terrain.biomes import BIOMES, ROCK
from world_builder.terrain.chunkgen import (
    VERTEX_DTYPE,
    VERTS,
    chunk_size,
    chunk_spacing,
    generate_chunk,
)
from world_builder.terrain.level import DEMO_LEVEL, load_level
from world_builder.terrain.planet import PRESETS, Planet
from world_builder.terrain.pngio import read_image, write_png
from world_builder.terrain.query import TerrainQuery
from world_builder.terrain.stamps import Paint, Placement, Stamp


@pytest.fixture(scope="module")
def planet():
    return Planet(replace(PRESETS["earthlike"], seed=19))


@pytest.fixture(scope="module")
def query(planet):
    return TerrainQuery(planet, include_ocean=False)


# ── queries ────────────────────────────────────────────────────────────────────


def test_height_at_vertices_matches_the_mesh(query):
    chunk = generate_chunk(query.planet.spec, (0, 10, -40))
    query.store[chunk.key] = chunk
    spacing = chunk_spacing(0)
    x0, y0 = 10 * 128.0, -40 * 128.0
    cols, rows = np.meshgrid(np.arange(0, VERTS - 1, 7), np.arange(0, VERTS - 1, 5))
    h = query.height_at(x0 + cols * spacing, y0 + rows * spacing)
    np.testing.assert_allclose(h, chunk.elevation[rows, cols], atol=1e-3)


def test_height_inside_triangles_matches_the_rendered_plane(query):
    """Interpolate exactly as the GPU does, on the mesh's own triangles."""
    chunk = generate_chunk(query.planet.spec, (0, -3, 7))
    query.store[chunk.key] = chunk
    vertices = np.frombuffer(chunk.vertex_bytes, VERTEX_DTYPE)["vertex"].astype(
        np.float64
    )
    centre = np.array([-3 * 128.0 + 64.0, 7 * 128.0 + 64.0])
    rng = np.random.default_rng(4)
    for _ in range(50):
        r, c = rng.integers(0, VERTS - 1, 2)
        fu, fv = sorted(rng.uniform(0, 1, 2))[::-1]  # fu >= fv: triangle (00, 01, 11)
        p00 = vertices[r * VERTS + c]
        p01 = vertices[r * VERTS + c + 1]
        p11 = vertices[(r + 1) * VERTS + c + 1]
        expected = p00 + fu * (p01 - p00) + fv * (p11 - p01)
        got = query.height_at(centre[0] + expected[0], centre[1] + expected[1])
        assert float(got) == pytest.approx(expected[2], abs=2e-3)


def test_query_follows_the_drawn_level(planet):
    """With coarser detail drawn, collision reads that coarser mesh."""
    coarse = TerrainQuery(planet, include_ocean=False, level=3)
    chunk = generate_chunk(planet.spec, (3, 0, -10))
    coarse.store[chunk.key] = chunk
    spacing = chunk_spacing(3)
    x0, y0 = 0.0, -10 * chunk_size(3)
    cols = np.arange(1, VERTS - 1, 9)
    h = coarse.height_at(x0 + cols * spacing, np.full(cols.shape, y0 + 20 * spacing))
    np.testing.assert_allclose(h, chunk.elevation[20, cols], atol=1e-3)
    # Midway between coarse vertices the fine grid has detail the coarse one lacks.
    mid = x0 + (cols + 0.5) * spacing
    fine = TerrainQuery(planet, include_ocean=False)
    y = np.full(mid.shape, y0 + 20 * spacing)
    assert np.abs(coarse.height_at(mid, y) - fine.height_at(mid, y)).max() > 0.05
    hit = coarse.segment_intersect([mid[0], y[0], 5000.0], [mid[0], y[0], -5000.0])
    assert hit[1][2] == pytest.approx(float(coarse.height_at(mid[0], y[0])), abs=0.01)


def test_normal_points_up_and_matches_slope(query):
    n = query.normal_at(np.array([100.0, 2000.0]), np.array([-300.0, -9000.0]))
    np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0)
    assert np.all(n[:, 2] > 0.0)


def test_sphere_contact(query):
    x, y = 500.0, -9000.0
    ground = float(query.height_at(x, y))
    assert query.sphere_contact([x, y, ground + 50.0], 20.0) is None
    hit, normal = query.sphere_contact([x, y, ground + 10.0], 20.0)
    assert np.linalg.norm(hit - np.array([x, y, ground + 10.0])) <= 20.0 + 1e-6
    assert normal[2] > 0.3
    assert abs(hit[2] - float(query.height_at(hit[0], hit[1]))) < 2.0
    # A buried sphere still reports the surface above its centre.
    hit, normal = query.sphere_contact([x, y, ground - 5.0], 2.0)
    assert hit[2] > ground - 5.0


def test_large_sensor_sphere_uses_a_coarse_level(query):
    x, y = 500.0, -9000.0
    ground = float(query.height_at(x, y))
    before = set(query.store)
    assert query.sphere_contact([x, y, ground + 60.0], 100.0) is not None
    new_levels = {k[0] for k in set(query.store) - before}
    assert new_levels and min(new_levels) >= 2


def test_segment_intersect(query):
    x, y = 300.0, -9500.0
    ground = float(query.height_at(x, y))
    hit = query.segment_intersect(
        [x, y, ground + 100.0], [x, y, ground - 100.0], generate=True
    )
    t, point, normal = hit
    assert t == pytest.approx(0.5, abs=0.01)
    assert point[2] == pytest.approx(ground, abs=0.05)
    assert (
        query.segment_intersect([x, y, ground + 100.0], [x + 50, y, ground + 200.0])
        is None
    )


def test_ocean_surface_is_solid_when_included(planet):
    wet = TerrainQuery(planet, include_ocean=True)
    # Offshore, north-east of the demo start (about 1 km deep for seed 19).
    x, y = 45000.0, 55000.0
    assert float(wet.ground_height_at(x, y)) < -500.0
    assert float(wet.height_at(x, y)) == 0.0
    assert wet.material_at(x, y) == "water"


# ── stamps ─────────────────────────────────────────────────────────────────────


def test_png_round_trip_keeps_16_bits(tmp_path):
    image = (np.arange(64 * 32, dtype=np.uint32).reshape(32, 64) * 31 % 65536).astype(
        np.uint16
    )
    write_png(tmp_path / "a.png", image)
    back = read_image(tmp_path / "a.png")
    assert back.shape == (32, 64, 1)
    np.testing.assert_array_equal(
        np.rint(back[:, :, 0] * 65535).astype(np.uint16), image
    )


@pytest.fixture
def bump(tmp_path):
    yy, xx = np.mgrid[-1:1:65j, -1:1:65j]
    image = np.clip(1.0 - np.hypot(xx, yy), 0.0, 1.0)
    path = tmp_path / "bump.png"
    write_png(path, (image * 65535).astype(np.uint16))
    return str(path)


def _stamped(planet, stamp, x, y):
    dirs = planet.chart.to_dirs(np.atleast_1d(x), np.atleast_1d(y))
    base = np.full(len(dirs), 100.0)
    return stamp.apply(dirs, base, 2.0, planet.spec.radius, base=40.0)


def test_stamp_blends(planet, bump):
    placement = Placement(
        lat=planet.spec.reference_lat,
        lon=planet.spec.reference_lon,
        width=1000.0,
        height=1000.0,
    )
    centre, outside = (0.0, 0.0), (0.0, 900.0)
    add = Stamp(bump, placement, height_scale=50.0, blend="add")
    assert _stamped(planet, add, *centre)[0] == pytest.approx(150.0, abs=0.5)
    assert _stamped(planet, add, *outside)[0] == 100.0
    replace_ = Stamp(
        bump, placement, height_scale=50.0, height_offset=10.0, blend="replace"
    )
    assert _stamped(planet, replace_, *centre)[0] == pytest.approx(60.0, abs=0.5)
    flatten = Stamp(
        bump, placement, height_scale=0.0, height_offset=2.0, blend="flatten"
    )
    assert _stamped(planet, flatten, 200.0, 0.0)[0] == pytest.approx(42.0)
    soft = Stamp(
        bump,
        placement,
        height_scale=0.0,
        height_offset=0.0,
        blend="replace",
        falloff=200.0,
    )
    edge = _stamped(planet, soft, np.array([0.0, 450.0, 499.0]), np.zeros(3))
    assert (
        edge[0] == pytest.approx(0.0)
        and 0.0 < edge[1] < 100.0
        and edge[2] == pytest.approx(100.0, abs=0.5)
    )


def test_rotated_stamp(planet, tmp_path):
    image = np.zeros((16, 64), dtype=np.uint16)
    image[:, :] = 65535  # a wide bar
    path = tmp_path / "bar.png"
    write_png(path, image)
    placement = Placement(
        planet.spec.reference_lat,
        planet.spec.reference_lon,
        1000.0,
        250.0,
        rotation=90.0,
    )
    stamp = Stamp(str(path), placement, height_scale=10.0, blend="add")
    # Rotated a quarter turn, the bar runs north-south.
    assert _stamped(planet, stamp, 0.0, 400.0)[0] == pytest.approx(110.0)
    assert _stamped(planet, stamp, 400.0, 0.0)[0] == 100.0


def test_paint_forces_biomes(planet, tmp_path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (128, 128, 128)  # west half: rock
    path = tmp_path / "paint.png"
    write_png(path, image)
    placement = Placement(
        planet.spec.reference_lat, planet.spec.reference_lon, 1000.0, 1000.0
    )
    paint = Paint(str(path), placement, colors=(("#808080", "rock"),))
    dirs = planet.chart.to_dirs(np.array([-300.0, 300.0]), np.array([0.0, 0.0]))
    w = np.zeros((2, len(BIOMES)), dtype=np.float32)
    w[:, 2] = 1.0  # grass
    out = paint.apply(dirs, w, planet.spec.radius)
    assert out[0].argmax() == ROCK
    np.testing.assert_allclose(out[1], w[1])
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_demo_level_loads_and_stamps_show():
    spec, camera = load_level(DEMO_LEVEL)
    assert len(spec.stamps) == 4 and len(spec.paints) == 3
    assert camera["z"] > 0
    planet = Planet(spec)
    pad = spec.stamps[2]
    assert pad.blend == "flatten"
    # Around the pad's centre the ground is level, at its base height.
    x, y = planet.chart.from_dirs(
        np.array([[*_dir(pad.placement.lat, pad.placement.lon)]])
    )
    s = planet.sample_grid(float(x[0]) - 40.0, float(y[0]) - 40.0, 4.0, 21)
    base = planet.stamp_base(2)
    assert np.ptp(s.elevation) < 1.0
    assert np.abs(s.elevation - (base + 0.5)).max() < 1.0


def test_cliff_lake():
    """Water at sea level, deep up to a near-vertical cliff on the north shore."""
    spec, _ = load_level(DEMO_LEVEL)
    query = TerrainQuery(Planet(spec), include_ocean=False)
    y = np.arange(4000.0, 5600.0, 2.0)
    h = query.ground_height_at(np.full_like(y, -4500.0), y)
    assert h.min() < -50.0  # the lake bed
    shore = int(np.argmax(h > 0.0))
    assert h[shore - 4] < -20.0  # still deep 8 m from the foot of the wall
    assert np.max(h) > 300.0
    rim = int(np.argmax(h > 290.0))
    assert (rim - shore) * 2.0 <= 16.0  # waterline to rim within 16 m
    wall = int(np.argmax(np.diff(h)))
    assert query.material_at(-4500.0, y[wall] + 1.0) == "rock"
    assert query.material_at(-4500.0, 5900.0) == "forest"  # painted plateau


def _dir(lat, lon):
    from world_builder.terrain.chart import dirs_from_latlon

    return dirs_from_latlon(lat, lon)[0]


def test_level_rejects_unknown_fields(tmp_path):
    level = tmp_path / "bad.yaml"
    level.write_text("planet: {preset: desert, elevation: {mountian_height: 3}}\n")
    with pytest.raises(ValueError, match="mountian_height"):
        load_level(level)


# ── materials ──────────────────────────────────────────────────────────────────


def test_texture_name_matching(tmp_path):
    folder = tmp_path / "rock"
    folder.mkdir()
    names = [
        "Rock051_2K-PNG_Color.png",
        "Rock051_2K-PNG_NormalGL.png",
        "Rock051_2K-PNG_Displacement.png",
        "Rock051_2K-PNG_Roughness.png",
        "Rock051_2K-PNG_AmbientOcclusion.png",
        "rock_diff_2k.jpg",
        "rock_preview.png",
        "readme.txt",
    ]
    for name in names:
        (folder / name).write_bytes(b"")
    maps = materials._find_maps(folder)
    assert maps["albedo"].name == "Rock051_2K-PNG_Color.png"
    assert maps["normal"].name == "Rock051_2K-PNG_NormalGL.png"
    assert maps["height"].name == "Rock051_2K-PNG_Displacement.png"
    assert maps["roughness"].name == "Rock051_2K-PNG_Roughness.png"
    assert maps["ao"].name == "Rock051_2K-PNG_AmbientOcclusion.png"


def test_polyhaven_names_and_directx_detection(tmp_path):
    folder = tmp_path / "grass"
    folder.mkdir()
    for name in (
        "grass_diff_2k.jpg",
        "grass_nor_dx_2k.png",
        "grass_disp_2k.png",
        "grass_ao_2k.jpg",
        "grass_rough_2k.jpg",
    ):
        (folder / name).write_bytes(b"")
    maps = materials._find_maps(folder)
    assert set(maps) == {"albedo", "normal", "height", "ao", "roughness"}
    assert materials._is_directx(maps["normal"])


def test_placeholders_and_packing(tmp_path):
    m = materials.load_materials(tmp_path)  # no textures at all: placeholders
    assert m.albedo_height.shape == (
        len(BIOMES),
        materials.PLACEHOLDER_SIZE,
        materials.PLACEHOLDER_SIZE,
        4,
    )
    assert m.albedo_height.dtype == np.uint8
    assert set(m.sources.values()) == {"placeholder"}
    assert np.all((m.mean_albedo > 0.0) & (m.mean_albedo < 1.0))
    for tile in m.tile_m:
        assert materials.UV_PERIOD % (tile * materials.MACRO_SCALE) == 0


def test_real_texture_replaces_the_placeholder(tmp_path):
    folder = tmp_path / "snow"
    folder.mkdir()
    write_png(folder / "snow_albedo.png", np.full((256, 256, 3), 200, dtype=np.uint8))
    m = materials.load_materials(tmp_path)
    assert m.albedo_height.shape[1] == 256
    layer = BIOMES.index("snow")
    assert np.all(m.albedo_height[layer, :, :, :3] == 200)
    assert m.sources["snow"] == "albedo=snow_albedo.png"


def test_bad_tile_size_is_rejected(tmp_path):
    (tmp_path / "textures.yaml").write_text("grass: {tile_m: 3}\n")
    with pytest.raises(ValueError, match="tile_m"):
        materials.load_materials(tmp_path)


def test_placeholder_tiles_seamlessly():
    p = materials.make_placeholder(ROCK, 128)
    for key in ("height", "albedo"):
        image = p[key]
        # The wrap-around step is no bigger than a typical interior step.
        wrap = np.abs(image[0] - image[-1]).mean()
        inner = np.abs(np.diff(image, axis=0)).mean()
        assert wrap < 3.0 * inner
