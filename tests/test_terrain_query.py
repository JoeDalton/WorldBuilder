"""
Unit tests for the terrain prototype's queries, stamps, levels and materials
(world_builder.terrain). Pure NumPy plus Panda image I/O; no window.
"""

from dataclasses import replace

import numpy as np
import pytest

from world_builder.terrain import materials
from world_builder.terrain.biomes import BIOMES, ROCK
from world_builder.terrain.chart import (
    AzimuthalEquidistantChart,
    TangentFrame,
    dirs_from_latlon,
)
from world_builder.terrain.chunkgen import VERTS, generate_chunk
from world_builder.terrain.cubesphere import face_dirs
from world_builder.terrain.level import DEMO_LEVEL, load_level
from world_builder.terrain.planet import PRESETS, Planet
from world_builder.terrain.pngio import read_image, write_png
from world_builder.terrain.query import TerrainQuery
from world_builder.terrain.stamps import Paint, Placement, Stamp

#: The demo level's reference point: a coast, with a range to the south.
REFERENCE = (38.85, -125.72)


@pytest.fixture(scope="module")
def planet():
    spec = replace(
        PRESETS["earthlike"],
        seed=19,
        reference_lat=REFERENCE[0],
        reference_lon=REFERENCE[1],
    )
    return Planet(spec)


@pytest.fixture(scope="module")
def chart(planet):
    spec = planet.spec
    return AzimuthalEquidistantChart(
        spec.reference_lat, spec.reference_lon, spec.radius
    )


@pytest.fixture(scope="module")
def query(planet):
    return TerrainQuery(planet, include_ocean=False)


def at(chart, x, y, height):
    """:returns: (..., 3) planet-frame point *height* m above sea level over
    chart (x, y)"""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    dirs = chart.to_dirs(x, y).reshape(*x.shape, 3)
    return dirs * (chart.radius + height)


def _key_near(planet, chart, level, x=0.0, y=0.0):
    face, i, j, _, _ = planet.grid.locate(
        chart.to_dirs(np.array([x]), np.array([y])), level
    )
    return level, int(face[0]), int(i[0]), int(j[0])


def _positions(chunk):
    return chunk.vertices()["vertex"].astype(np.float64) + chunk.centre


# ── queries ────────────────────────────────────────────────────────────────────


def test_height_at_vertices_matches_the_mesh(query, chart):
    chunk = generate_chunk(
        query.planet.spec, _key_near(query.planet, chart, 0, 500.0, -9000.0)
    )
    query.store[chunk.key] = chunk
    p = _positions(chunk)[: VERTS * VERTS]
    # Anywhere on the radial line through a vertex, above or below it.
    pick = np.arange(0, VERTS * VERTS, 37)
    for scale in (1.01, 0.99):
        h = query.height_at(p[pick] * scale)
        np.testing.assert_allclose(h, chunk.elevation.ravel()[pick], atol=1e-3)


def test_height_inside_triangles_matches_the_rendered_plane(query, chart):
    """Intersect the very triangles the GPU draws."""
    chunk = generate_chunk(
        query.planet.spec, _key_near(query.planet, chart, 0, -300.0, -9200.0)
    )
    query.store[chunk.key] = chunk
    p = _positions(chunk)
    rng = np.random.default_rng(4)
    for _ in range(50):
        r, c = rng.integers(0, VERTS - 1, 2)
        fu, fv = sorted(rng.uniform(0, 1, 2))[::-1]  # fu >= fv: triangle (00, 01, 11)
        p00, p01, p11 = (
            p[r * VERTS + c],
            p[r * VERTS + c + 1],
            p[(r + 1) * VERTS + c + 1],
        )
        point = p00 + fu * (p01 - p00) + fv * (p11 - p01)
        assert float(query.altitude(point)) == pytest.approx(0.0, abs=2e-3)


def test_query_follows_the_drawn_level(planet, chart):
    """With coarser detail drawn, collision reads that coarser mesh."""
    coarse = TerrainQuery(planet, include_ocean=False, level=3)
    chunk = generate_chunk(planet.spec, _key_near(planet, chart, 3, 0.0, -9000.0))
    coarse.store[chunk.key] = chunk
    p = _positions(chunk)
    row = 20
    cols = np.arange(1, VERTS - 1, 9)
    h = coarse.height_at(p[row * VERTS + cols] * 1.001)
    np.testing.assert_allclose(h, chunk.elevation[row, cols], atol=1e-3)
    # Midway between coarse vertices the fine grid has detail the coarse one lacks.
    mid = 0.5 * (p[row * VERTS + cols] + p[row * VERTS + cols + 1])
    fine = TerrainQuery(planet, include_ocean=False)
    assert np.abs(coarse.height_at(mid) - fine.height_at(mid)).max() > 0.05
    up = mid[0] / np.linalg.norm(mid[0])
    hit = coarse.segment_intersect(mid[0] + 5000.0 * up, mid[0] - 5000.0 * up)
    height = float(np.linalg.norm(hit[1])) - planet.spec.radius
    assert height == pytest.approx(float(coarse.height_at(mid[0])), abs=0.01)


def test_normal_points_out_of_the_ground(query, chart):
    points = at(chart, [100.0, 2000.0], [-300.0, -9000.0], 3000.0)
    n = query.normal_at(points)
    np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0)
    up = points / np.linalg.norm(points, axis=-1, keepdims=True)
    assert np.all(np.einsum("ij,ij->i", n, up) > 0.0)


def test_sphere_contact(query, chart):
    surface = query.surface_point(at(chart, 500.0, -9000.0, 0.0))
    up = surface / np.linalg.norm(surface)
    assert query.sphere_contact(surface + 50.0 * up, 20.0) is None
    centre = surface + 10.0 * up
    hit, normal = query.sphere_contact(centre, 20.0)
    assert np.linalg.norm(hit - centre) <= 20.0 + 1e-6
    assert normal @ up > 0.3
    assert abs(float(query.altitude(hit))) < 2.0
    # A buried sphere still reports the surface above its centre.
    buried = surface - 5.0 * up
    hit, normal = query.sphere_contact(buried, 2.0)
    assert (hit - buried) @ up > 0.0


@pytest.mark.parametrize(
    "where",
    [(0, 1.0, 0.3), (2, 0.4, 1.0), (0, 1.0, 1.0)],
    ids=["seam", "rotated seam", "corner"],
)
def test_sphere_contact_across_seams_and_corners(query, where):
    face, a, b = where
    d = face_dirs(face, np.array([a]), np.array([b]))[0]
    surface = query.surface_point(d)
    hit, normal = query.sphere_contact(surface + 8.0 * d, 20.0)
    assert normal @ d > 0.3
    assert abs(float(query.altitude(hit))) < 2.0


def test_large_sensor_sphere_uses_a_coarse_level(query, chart):
    surface = query.surface_point(at(chart, 500.0, -9000.0, 0.0))
    up = surface / np.linalg.norm(surface)
    before = set(query.store)
    assert query.sphere_contact(surface + 60.0 * up, 100.0) is not None
    new_levels = {k[0] for k in set(query.store) - before}
    assert new_levels and min(new_levels) >= 2


def test_segment_intersect(query, chart):
    surface = query.surface_point(at(chart, 300.0, -9500.0, 0.0))
    up = surface / np.linalg.norm(surface)
    t, point, normal = query.segment_intersect(
        surface + 100.0 * up, surface - 100.0 * up, generate=True
    )
    assert t == pytest.approx(0.5, abs=0.01)
    np.testing.assert_allclose(point, surface, atol=0.05)
    assert normal @ up > 0.0
    side = TangentFrame.at(*REFERENCE).east
    assert (
        query.segment_intersect(
            surface + 100.0 * up, surface + 200.0 * up + 50.0 * side
        )
        is None
    )


def test_segment_starting_underground_hits_at_once(query, chart):
    surface = query.surface_point(at(chart, 300.0, -9500.0, 0.0))
    up = surface / np.linalg.norm(surface)
    start = surface - 5.0 * up
    t, point, normal = query.segment_intersect(start, start + 100.0 * up, True)
    assert t == 0.0
    np.testing.assert_array_equal(point, start)


def test_ocean_surface_is_solid_when_included(planet, chart):
    wet = TerrainQuery(planet, include_ocean=True)
    offshore = at(chart, -10000.0, 30000.0, 100.0)  # about 1 km deep for seed 19
    assert float(wet.ground_height_at(offshore)) < -500.0
    # The drawn water: a triangle of sea-level corners (its chord a hair
    # under the sphere), facing up.
    assert float(wet.height_at(offshore)) == pytest.approx(0.0, abs=1e-3)
    up = offshore / np.linalg.norm(offshore)
    np.testing.assert_allclose(wet.normal_at(offshore), up, atol=1e-5)
    assert wet.material_at(offshore) == "water"


def test_queries_without_chunks_fall_back_to_the_planet(planet):
    query = TerrainQuery(planet, include_ocean=False)
    point = TangentFrame.at(10.0, 10.0).up * (planet.spec.radius + 9000.0)
    h = float(query.height_at(point, generate=False))
    expected = planet.sample_dirs(
        point[None, :] / np.linalg.norm(point), apply_stamps=True
    )[0]
    assert h == pytest.approx(float(expected[0]))
    assert not query.store


def test_queries_on_no_points(query):
    empty = np.zeros((0, 3))
    assert query.height_at(empty, generate=False).shape == (0,)
    assert query.ground_height_at(empty, generate=False).shape == (0,)
    assert query.normal_at(empty, generate=False).shape == (0, 3)
    assert query.altitude(empty, generate=False).shape == (0,)
    assert query.surface_point(empty, generate=False).shape == (0, 3)


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
    chart = AzimuthalEquidistantChart(*REFERENCE, planet.spec.radius)
    dirs = chart.to_dirs(np.atleast_1d(x), np.atleast_1d(y))
    base = np.full(len(dirs), 100.0)
    return stamp.apply(dirs, base, 2.0, planet.spec.radius, base=40.0)


def test_stamps_ignore_the_rest_of_the_planet(planet, bump):
    """Far directions never reach the stamp's chart maths (singular at its
    antipode), and come back untouched."""
    placement = Placement(
        lat=REFERENCE[0], lon=REFERENCE[1], width=1000.0, height=1000.0
    )
    stamp = Stamp(bump, placement, height_scale=50.0, blend="add")
    antipode = -TangentFrame.at(*REFERENCE).up
    dirs = np.array([antipode, TangentFrame.at(0.0, 0.0).up])
    with np.errstate(all="raise"):
        out = stamp.apply(dirs, np.array([7.0, 8.0]), 2.0, planet.spec.radius)
    np.testing.assert_array_equal(out, [7.0, 8.0])


def test_stamp_blends(planet, bump):
    placement = Placement(
        lat=REFERENCE[0], lon=REFERENCE[1], width=1000.0, height=1000.0
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


def test_stamp_max_and_min_blends(planet, bump):
    """max raises the ground to the stamp, never lowers it; min the
    reverse (the ground here is 100 m, the stamp 10-60 m)."""
    placement = Placement(
        lat=REFERENCE[0], lon=REFERENCE[1], width=1000.0, height=1000.0
    )
    high = Stamp(bump, placement, height_scale=50.0, height_offset=90.0, blend="max")
    assert _stamped(planet, high, 0.0, 0.0)[0] == pytest.approx(140.0, abs=0.5)
    assert _stamped(planet, high, 0.0, 450.0)[0] == pytest.approx(100.0, abs=0.5)
    low = Stamp(bump, placement, height_scale=50.0, height_offset=10.0, blend="min")
    assert _stamped(planet, low, 0.0, 0.0)[0] == pytest.approx(60.0, abs=0.5)
    with pytest.raises(ValueError, match="unknown blend"):
        Stamp(bump, placement, blend="multiply")


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
    chart = AzimuthalEquidistantChart(*REFERENCE, planet.spec.radius)
    dirs = chart.to_dirs(np.array([-300.0, 300.0]), np.array([0.0, 0.0]))
    w = np.zeros((2, len(BIOMES)), dtype=np.float32)
    w[:, 2] = 1.0  # grass
    out = paint.apply(dirs, w, planet.spec.radius)
    assert out[0].argmax() == ROCK
    np.testing.assert_allclose(out[1], w[1])
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_demo_level_loads_and_stamps_show():
    spec, camera = load_level(DEMO_LEVEL)
    assert len(spec.stamps) == 4 and len(spec.paints) == 3
    assert camera["alt"] > 0 and -90.0 <= camera["lat"] <= 90.0
    assert spec.radius == 600e3
    planet = Planet(spec)
    pad = spec.stamps[2]
    assert pad.blend == "flatten"
    # Around the pad's centre the ground is level, at its base height.
    chart = AzimuthalEquidistantChart(pad.placement.lat, pad.placement.lon, spec.radius)
    coords = 4.0 * np.arange(-10, 11)
    gx, gy = np.meshgrid(coords, coords)
    elevation, _, _ = planet.sample_dirs(chart.to_dirs(gx, gy), 16.0, apply_stamps=True)
    base = planet.stamp_base(2)
    assert np.ptp(elevation) < 1.0
    assert np.abs(elevation - (base + 0.5)).max() < 1.0


def test_level_radius_override():
    spec, camera = load_level(DEMO_LEVEL, radius=1200e3)
    assert spec.radius == 1200e3
    # Chart metres stay metres: the crater is as far from the reference point.
    chart = AzimuthalEquidistantChart(
        spec.reference_lat, spec.reference_lon, spec.radius
    )
    crater = spec.stamps[0].placement
    x, y = chart.from_dirs(dirs_from_latlon(crater.lat, crater.lon))
    assert (float(x[0]), float(y[0])) == pytest.approx((-4500.0, -2500.0), abs=0.01)


def test_cliff_lake():
    """Water at sea level, deep up to a near-vertical cliff on the north shore."""
    spec, _ = load_level(DEMO_LEVEL)
    query = TerrainQuery(Planet(spec), include_ocean=False)
    chart = AzimuthalEquidistantChart(
        spec.reference_lat, spec.reference_lon, spec.radius
    )
    y = np.arange(4000.0, 5600.0, 2.0)
    line = at(chart, np.full_like(y, -4500.0), y, 1000.0)
    h = query.ground_height_at(line)
    assert h.min() < -50.0  # the lake bed
    shore = int(np.argmax(h > 0.0))
    assert h[shore - 4] < -20.0  # still deep 8 m from the foot of the wall
    assert np.max(h) > 300.0
    rim = int(np.argmax(h > 290.0))
    assert (rim - shore) * 2.0 <= 16.0  # waterline to rim within 16 m
    wall = int(np.argmax(np.diff(h)))
    assert query.material_at(at(chart, -4500.0, y[wall] + 1.0, 500.0)) == "rock"
    assert query.material_at(at(chart, -4500.0, 5900.0, 500.0)) == "forest"  # painted


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
