"""
Unit tests for the terrain prototype's planet, biomes, chunks and LOD
(world_builder.terrain). Pure NumPy; no window.
"""

import math
import pickle
from dataclasses import replace

import numpy as np
import pytest
from timing import TIME_SCALE, wait_until

from world_builder.terrain import biomes
from world_builder.terrain.bench import _best_ms
from world_builder.terrain.chart import (
    AzimuthalEquidistantChart,
    TangentFrame,
    dirs_from_latlon,
    latlon,
)
from world_builder.terrain.chunkgen import (
    QUADS,
    VERTEX_DTYPE,
    VERTS,
    generate_chunk,
    parent_interpolation,
    skirt_ring,
    triangle_indices,
)
from world_builder.terrain.cubesphere import CubeGrid, face_dirs, grid_for
from world_builder.terrain.planet import (
    LOW_CHARACTER,
    LOW_CONTINENT,
    LOW_RANGE,
    PLANET_HEIGHT_BOUNDS,
    PRESETS,
    REFERENCE_RADIUS_M,
    Planet,
    scaled_params,
)
from world_builder.terrain.quadtree import UNKNOWN_BOUNDS, LodSelector
from world_builder.terrain.workers import ChunkExecutor

R = 600e3


@pytest.fixture(scope="module")
def spec():
    return replace(PRESETS["earthlike"], seed=19).resolved()


@pytest.fixture(scope="module")
def planet(spec):
    return Planet(spec)


def camera_at(lat, lon, altitude, radius=R):
    return TangentFrame.at(lat, lon).up * (radius + altitude)


# ── planet ─────────────────────────────────────────────────────────────────────


def test_spec_is_picklable_and_hashable(spec):
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert hash(spec) == hash(replace(spec))
    assert spec.radius == R


def test_sea_bias_sets_the_ocean_fraction(spec, planet):
    rng = np.random.default_rng(99)
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    continent = planet.low_band(dirs)[:, LOW_CONTINENT]
    assert np.mean(continent < 0.0) == pytest.approx(
        spec.elevation.ocean_fraction, abs=0.03
    )


@pytest.mark.parametrize(
    "radius, tolerance", [(600e3, 0.05), (200e3, 0.1), (6371e3, 0.1)]
)
def test_interpolated_low_band_matches_direct(radius, tolerance):
    """The lattice interpolation is invisible: centimetres of elevation, at a
    face centre and right at a face edge (where the lattice runs past it)."""
    planet = Planet(replace(PRESETS["earthlike"], seed=19, radius=radius))
    n0 = planet.grid.n0
    for ka0 in (n0 // 2 + 12345, n0 - 40):
        ka = ka0 + 3 * np.arange(60)
        kb = n0 // 2 + 777 + 3 * np.arange(60)
        interpolated = planet._interpolated_low_band(1, ka, kb)
        ga, gb = np.meshgrid(planet.grid.param(ka), planet.grid.param(kb))
        dirs = face_dirs(1, ga, gb)
        e_i, _, _ = planet.evaluate(dirs, interpolated, 12.0)
        e_d, _, _ = planet.evaluate(dirs, planet.low_band(dirs), 12.0)
        assert np.abs(e_i - e_d).max() < tolerance


def test_sample_face_grid_is_deterministic(planet):
    ka = 1000 + 4 * np.arange(33)
    kb = 90000 + 4 * np.arange(33)
    a = planet.sample_face_grid(3, ka, kb, 4)
    b = planet.sample_face_grid(3, ka, kb, 4)
    np.testing.assert_array_equal(a.elevation, b.elevation)
    assert a.elevation.shape == (33, 33)


def test_temperature_falls_with_altitude(planet):
    t = np.array([15.0, 15.0])
    cooled = planet.lapse(t, np.array([0.0, 2000.0]))
    assert cooled[1] == pytest.approx(15.0 - 13.0)


def _random_dirs(n, seed=0):
    dirs = np.random.default_rng(seed).normal(size=(n, 3))
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


@pytest.mark.parametrize("radius", [600e3, 2400e3])
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_heights_stay_within_the_bounds(preset, radius):
    """The LOD pass assumes ungenerated ground lies within these."""
    planet = Planet(replace(PRESETS[preset], seed=3, radius=radius))
    elevation, _, _ = planet.sample_dirs(_random_dirs(60000), 400.0)
    lo, hi = PLANET_HEIGHT_BOUNDS
    assert lo < elevation.min() and elevation.max() < hi


def test_plains_are_flat_and_ranges_rough(planet):
    """Plains (low character, no range) are nearly flat; ranges are not."""
    dirs = _random_dirs(4000, seed=5)
    low = planet.low_band(dirs)
    land = low[:, LOW_CONTINENT] > 0.1
    plains = dirs[land & (low[:, LOW_CHARACTER] < 0.05) & (low[:, LOW_RANGE] < 0.01)]
    ranges = dirs[land & (low[:, LOW_RANGE] > 0.7)]
    assert len(plains) > 20 and len(ranges) > 20

    def median_slope(centres):
        lat, lon = latlon(centres[:20])
        coords = 100.0 * (np.arange(24) - 12)
        gx, gy = np.meshgrid(coords, coords)
        slopes = []
        for a, b in zip(lat, lon):
            chart = AzimuthalEquidistantChart(a, b, planet.spec.radius)
            z, _, _ = planet.sample_dirs(chart.to_dirs(gx, gy), 400.0)
            slope = biomes.slope_degrees(z.reshape(24, 24), 100.0)
            slopes.append(np.median(slope[1:-1, 1:-1]))
        return float(np.median(slopes))

    assert median_slope(plains) < 2.0
    assert median_slope(ranges) > 10.0


def test_dunes_cover_the_desert_plains_only():
    """Sand seas cover most plains of the desert planet, at any height, and
    only otherwise flat ground: no range, no tablelands."""
    planet = Planet(replace(PRESETS["desert"], seed=19))
    dirs = _random_dirs(40000, seed=7)
    masks = planet.landform_masks(dirs)
    low = planet.low_band(dirs)
    plains = (
        (low[:, LOW_CONTINENT] > 0.1)
        & (low[:, LOW_CHARACTER] < 0.1)
        & (low[:, LOW_RANGE] < 0.01)
    )
    assert np.mean(masks["dunes"][plains] > 0.5) > 0.6
    sand = masks["dunes"] > 0.01
    assert low[sand, LOW_CHARACTER].max() < 0.3
    assert low[sand, LOW_RANGE].max() < 0.05
    # Sand and tablelands share the ground at a region's edge, never more.
    assert np.all(masks["tablelands"] + masks["dunes"] <= 1.0 + 1e-9)
    deep = masks["tablelands"] > 0.99
    assert deep.any() and masks["dunes"][deep].max() < 0.01


def test_landforms_and_plates_can_be_turned_off():
    """Every optional landform off: the elevation is still a sane, repeatable
    function, and arid land loses its dunes."""
    desert = replace(PRESETS["desert"], seed=4)
    off = replace(
        desert,
        elevation=replace(desert.elevation, plates=0, dunes=False, tablelands=False),
    )
    dirs = _random_dirs(20000, seed=6)
    a, _, _ = Planet(off).sample_dirs(dirs, 8.0)
    b, _, _ = Planet(off).sample_dirs(dirs, 8.0)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a))
    on, _, _ = Planet(desert).sample_dirs(dirs, 8.0)
    assert not np.allclose(a, on)


def test_planetary_features_scale_with_the_radius():
    base = PRESETS["earthlike"]
    e600, c600 = scaled_params(base)
    assert e600.feature_scale == 1.0
    assert e600.continent_wavelength == base.elevation.continent_wavelength
    earth, earth_climate = scaled_params(replace(base, radius=6371e3))
    factor = 6371e3 / REFERENCE_RADIUS_M
    assert earth.continent_wavelength == pytest.approx(
        base.elevation.continent_wavelength * factor
    )
    assert earth_climate.temperature_wavelength == pytest.approx(
        c600.temperature_wavelength * factor
    )
    # Ground-level character does not scale.
    assert earth.hills_wavelength == base.elevation.hills_wavelength
    assert earth.detail_wavelength == base.elevation.detail_wavelength
    # Mountains grow with sqrt(scale), within bounds.
    assert e600.mountain_height < earth.mountain_height <= 7000.0
    # An explicit feature scale wins over the radius.
    fixed = replace(base.elevation, feature_scale=1.0)
    kept, _ = scaled_params(replace(base, radius=6371e3, elevation=fixed))
    assert kept.continent_wavelength == base.elevation.continent_wavelength


@pytest.mark.parametrize(
    "preset, expected",
    [("desert", biomes.DESERT), ("forest", biomes.FOREST), ("ice", biomes.SNOW)],
)
def test_presets_bias_the_biomes(preset, expected):
    planet = Planet(replace(PRESETS[preset], seed=2))
    spacing, n = 1500.0, 101
    coords = spacing * (np.arange(n) - n // 2)
    gx, gy = np.meshgrid(coords, coords)
    chart = AzimuthalEquidistantChart(20.0, 0.0, planet.spec.radius)
    elevation, temperature, humidity = planet.sample_dirs(
        chart.to_dirs(gx, gy), 4.0 * spacing
    )
    elevation = elevation.reshape(n, n)
    slope = biomes.slope_degrees(elevation, spacing)
    w = biomes.weights(temperature, humidity, elevation, slope, planet.spec.has_ocean)
    land = elevation.ravel() > 0.0
    counts = np.bincount(biomes.dominant(w[land]), minlength=len(biomes.BIOMES))
    assert counts.argmax() == expected


# ── biomes ─────────────────────────────────────────────────────────────────────


def _weights(t, h, z, slope, ocean=True):
    return biomes.weights(
        np.array([t]), np.array([h]), np.array([z]), np.array([slope]), ocean
    )[0]


def test_weights_are_normalised():
    rng = np.random.default_rng(0)
    w = biomes.weights(
        rng.uniform(-30, 40, 1000),
        rng.uniform(0, 1, 1000),
        rng.uniform(-100, 5000, 1000),
        rng.uniform(0, 70, 1000),
        True,
    )
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert w.min() >= 0.0


def test_biome_rules():
    assert _weights(-20, 0.5, 3000, 5).argmax() == biomes.SNOW
    assert _weights(15, 0.5, 300, 60).argmax() == biomes.ROCK
    assert _weights(-20, 0.5, 3000, 60).argmax() == biomes.ROCK  # cliffs shed snow
    assert _weights(30, 0.05, 300, 5).argmax() == biomes.DESERT
    assert _weights(15, 0.8, 300, 5).argmax() == biomes.FOREST
    assert _weights(15, 0.35, 300, 5).argmax() == biomes.GRASS
    assert _weights(15, 0.5, 1.0, 3).argmax() == biomes.BEACH
    assert _weights(15, 0.5, 1.0, 3, ocean=False).argmax() != biomes.BEACH


def test_slope_of_a_plane():
    x = np.arange(20) * 10.0
    plane = np.tile(x, (20, 1))  # rises 1 m per metre: 45 degrees
    slope = biomes.slope_degrees(plane, 10.0, step=2)
    np.testing.assert_allclose(slope[5:-5, 5:-5], 45.0)


def test_slope_from_normals():
    up = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, math.sin(0.5), math.cos(0.5)], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        biomes.slope_from_normals(normals, up), [math.degrees(0.5), 0.0], atol=1e-6
    )


# ── chunks ─────────────────────────────────────────────────────────────────────


def _positions(chunk):
    return chunk.vertices()["vertex"].astype(np.float64) + chunk.centre


def test_vertices_carry_five_weights_and_the_strata_mask():
    """Beach travels as the rest of the weights; the sixth byte is where the
    strata tint the ground (a canyon of the desert planet, seed 19)."""
    desert = replace(PRESETS["desert"], seed=19).resolved()
    planet = Planet(desert)
    face, i, j, _, _ = planet.grid.locate(dirs_from_latlon(-0.423, 11.1071), 4)
    chunk = generate_chunk(desert, (4, int(face[0]), int(i[0]), int(j[0])), 0)
    vertices = np.frombuffer(chunk.vertex_bytes, VERTEX_DTYPE)[: VERTS * VERTS]
    stored = np.column_stack([vertices["biome_a"], vertices["biome_b"][:, :1]])
    beach = 1.0 - stored.sum(axis=1) / 255.0
    np.testing.assert_allclose(beach, chunk.weights[:, 5] / 255.0, atol=3 / 255)
    assert vertices["biome_b"][:, 1].max() > 200  # in the tablelands


def test_chunk_layout(spec):
    grid = grid_for(spec.radius)
    key = (0, 1, grid.side(0) // 3, grid.side(0) // 2)
    chunk = generate_chunk(spec, key)
    vertices = chunk.vertices()
    assert len(vertices) == VERTS * VERTS + 4 * QUADS
    assert chunk.elevation.shape == (VERTS, VERTS)
    p = _positions(chunk)
    radial = np.linalg.norm(p, axis=1)
    # Grid vertices sit at R + h along their direction.
    np.testing.assert_allclose(
        radial[: VERTS * VERTS] - spec.radius, chunk.elevation.ravel(), atol=1e-2
    )
    np.testing.assert_allclose(
        vertices["height"][: VERTS * VERTS], chunk.elevation.ravel(), atol=1e-3
    )
    normals = vertices["normal"]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    up = p / radial[:, None]
    assert np.all(np.einsum("ij,ij->i", normals, up) > 0.0)  # outward
    skirt = vertices[VERTS * VERTS :]
    assert np.all(skirt["biome_b"][:, 3] == 255)
    edge = skirt_ring()
    assert np.all(radial[VERTS * VERTS :] < radial[edge])  # hangs below its edge
    np.testing.assert_allclose(  # straight down
        up[VERTS * VERTS :], up[edge], atol=1e-6
    )
    local = vertices["vertex"]
    assert np.all(local >= chunk.aabb_min) and np.all(local <= chunk.aabb_max)


def test_indices_reference_valid_vertices():
    indices = triangle_indices()
    assert indices.max() == VERTS * VERTS + 4 * QUADS - 1
    assert len(indices) % 3 == 0


def test_triangles_face_outwards(spec):
    chunk = generate_chunk(spec, (2, 5, 40, 77))
    p = _positions(chunk)
    tris = triangle_indices()[: 6 * QUADS * QUADS].reshape(-1, 3)
    n = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
    assert np.all(np.einsum("ij,ij->i", n, p[tris[:, 0]]) > 0.0)


def test_neighbour_chunks_share_edges(spec):
    """Same-level neighbours compute the same heights on their shared edge."""
    for level in (0, 3):
        west = generate_chunk(spec, (level, 2, 5, 2))
        east = generate_chunk(spec, (level, 2, 6, 2))
        np.testing.assert_allclose(
            west.elevation[:, -1], east.elevation[:, 0], atol=1e-6
        )
        south = generate_chunk(spec, (level, 2, 5, 1))
        np.testing.assert_allclose(
            south.elevation[-1, :], west.elevation[0, :], atol=1e-6
        )


@pytest.mark.parametrize("level", [0, 3, 8])
def test_chunks_meet_across_face_seams(spec, level):
    """Across a seam the two faces' edge vertices are the same points (1 cm;
    the lattices of the low band differ by millimetres)."""
    grid = grid_for(spec.radius)
    last = grid.side(level) - 1
    j = last // 3
    rows = VERTS * np.arange(VERTS)
    # +X's east edge is +Y's west edge (an equatorial seam).
    east = _positions(generate_chunk(spec, (level, 0, last, j)))[rows + QUADS]
    west = _positions(generate_chunk(spec, (level, 2, 0, j)))[rows]
    np.testing.assert_allclose(east, west, atol=0.01)
    # +Y's north edge is +Z's east edge, in the same order (a rotated seam).
    top = _positions(generate_chunk(spec, (level, 2, j, last)))[
        QUADS * VERTS + np.arange(VERTS)
    ]
    side = _positions(generate_chunk(spec, (level, 4, last, j)))[rows + QUADS]
    np.testing.assert_allclose(top, side, atol=0.01)


def test_parent_interpolation_keeps_even_samples():
    e = np.random.default_rng(0).normal(size=(VERTS, VERTS))
    m = parent_interpolation(e)
    np.testing.assert_array_equal(m[::2, ::2], e[::2, ::2])
    plane = np.add.outer(np.arange(VERTS) * 2.0, np.arange(VERTS) * 3.0)
    np.testing.assert_allclose(parent_interpolation(plane), plane)


def test_pchip_matches_scipy():
    """The continent curve's interpolation, as scipy's PchipInterpolator."""
    from scipy.interpolate import PchipInterpolator

    from world_builder.terrain.planet import ElevationParams, Pchip

    rng = np.random.default_rng(0)
    curves = [np.array(ElevationParams().continent_curve), np.array([[0, 1], [1, 3]])]
    for _ in range(50):
        x = np.unique(rng.uniform(-1.0, 1.0, rng.integers(3, 10)))
        curves.append(np.column_stack([x, 1000.0 * rng.normal(size=len(x))]))
    for curve in curves:
        x, y = curve[:, 0], curve[:, 1]
        q = np.linspace(x[0], x[-1], 501)
        np.testing.assert_allclose(
            Pchip(x, y)(q), PchipInterpolator(x, y)(q), rtol=1e-12, atol=1e-9
        )


# ── LOD selection ──────────────────────────────────────────────────────────────

GRID = grid_for(R)
ALL_READY = dict(ready=lambda k: True, bounds=lambda k: (0.0, 300.0))


def _ancestors(key):
    while key[0] < GRID.max_level:
        key = CubeGrid.parent(key)
        yield key


def test_selection_tiles_without_overlap():
    selector = LodSelector(GRID)
    selection = selector.select(camera_at(40.0, 20.0, 200.0), **ALL_READY)
    draw = set(selection.draw)
    assert not selection.wanted
    for key in draw:  # no drawn chunk is an ancestor of another drawn chunk
        assert not draw.intersection(_ancestors(key))
    # Fine near the camera, coarse far away.
    assert min(k[0] for k in draw) == 0
    assert max(k[0] for k in draw) >= 7


def test_selection_never_drops_a_parent_before_all_children_are_ready():
    selector = LodSelector(GRID)
    ready = set(GRID.roots())
    selection = selector.select(
        camera_at(40.0, 20.0, 100.0), ready=ready.__contains__, bounds=lambda k: None
    )
    assert set(selection.draw) <= ready
    assert any(k[0] == GRID.max_level - 1 for k in selection.wanted)
    assert all(k[0] >= GRID.max_level - 1 for k in selection.wanted)


def test_selection_wants_coarse_first():
    selector = LodSelector(GRID)
    selection = selector.select(
        camera_at(40.0, 20.0, 100.0), ready=lambda k: False, bounds=lambda k: None
    )
    assert selection.draw == []
    levels = [k[0] for k in selection.wanted]
    assert levels[0] == GRID.max_level
    assert levels == sorted(levels, reverse=True)


def test_from_orbit_only_coarse_chunks_cover_the_visible_hemisphere():
    camera = camera_at(10.0, 30.0, 3.0 * R)
    selection = LodSelector(GRID).select(camera, **ALL_READY)
    assert len(selection.draw) <= 40
    assert min(k[0] for k in selection.draw) >= GRID.max_level - 2
    # Every visible point of the planet is under a drawn chunk.
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    visible = dirs @ (camera / np.linalg.norm(camera)) > 1.0 / 4.0  # R / distance
    face, i, j, _, _ = GRID.locate(dirs[visible], GRID.max_level)
    for f, ki, kj in zip(face, i, j):
        root = (GRID.max_level, int(f), int(ki), int(kj))
        drawn = [k for k in selection.draw if k[1] == f and root in (k, *_ancestors(k))]
        assert drawn


def test_nothing_behind_the_horizon_is_drawn():
    camera = camera_at(40.0, 20.0, 10.0)
    up = camera / np.linalg.norm(camera)
    selection = LodSelector(GRID).select(
        camera, ready=lambda k: True, bounds=lambda k: (0.0, 300.0)
    )
    # Horizon angle of the eye (10 m) plus that of a 300 m peak.
    reach = math.acos(R / (R + 10.0)) + math.acos(R / (R + 300.0))
    for key in selection.draw:
        box = GRID.box(key)
        nearest = math.acos(np.clip(up @ np.array(box.up), -1, 1)) - math.acos(
            box.cos_radius
        )
        assert nearest <= reach + 1e-9
    assert len(selection.draw) < 1000


def test_cube_corner_draws_fine_chunks_from_three_faces():
    corner = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    selection = LodSelector(GRID).select(corner * (R + 20.0), **ALL_READY)
    fine_faces = {k[1] for k in selection.draw if k[0] <= 1}
    assert fine_faces == {0, 2, 4}  # +X, +Y, +Z


def test_inline_executor_generates_and_versions(spec):
    executor = ChunkExecutor(backend="inline", inline_budget=2)
    executor.set_spec(spec)
    keys = [(9, 0, 0, 0), (9, 0, 1, 0), (9, 0, 2, 0)]
    executor.request(keys)
    done = executor.collect()
    assert [c.key for c in done] == keys[:2]
    executor.request(keys[2:])
    executor.set_spec(replace(spec, seed=spec.seed + 1, sea_bias=None))
    assert executor.collect() == []  # the old version's chunk is dropped


def test_thread_executor_completes(spec):
    executor = ChunkExecutor(backend="thread", workers=2)
    executor.set_spec(spec)
    keys = [(10, 3, ix, 0) for ix in range(3)]
    executor.request(keys)
    done = []

    def all_done():
        done.extend(executor.collect())
        return len(done) == len(keys)

    try:
        wait_until(all_done, timeout=10.0, msg="chunks never arrived")
    finally:
        executor.shutdown()
    assert sorted(c.key for c in done) == keys


def test_selection_keeps_and_prefetches_every_child_of_split_nodes():
    """Regression: a culled child drifting into view must already be ready.

    Otherwise its parent cannot stay split, and the whole subtree (the fine
    detail under the camera included) collapses to one coarse chunk.
    """
    selector = LodSelector(GRID)
    camera = camera_at(40.0, 20.0, 200.0)
    everything = selector.select(camera, ready=lambda k: True, bounds=lambda k: None)
    for key in everything.draw:
        for node in _ancestors(key):
            assert set(CubeGrid.children(node)) <= everything.keep
    roots = set(GRID.roots())
    split = {CubeGrid.parent(k) for k in everything.keep if k not in roots}
    children = everything.keep - roots
    culled_children = children - set(everything.draw) - split
    assert culled_children  # some kept children are neither drawn nor split

    drawable = set(everything.draw) | split
    selector.reset()
    partial = selector.select(
        camera, ready=drawable.__contains__, bounds=lambda k: None
    )
    assert set(partial.prefetch) == culled_children - drawable
    assert partial.draw == everything.draw


def test_selection_wants_uncovered_ground_first_then_the_nearest_by_size():
    camera = camera_at(40.0, 20.0, 100.0)
    under = _root_under(camera)
    # Every root ready but the one under the camera; the others' children too.
    ready = set(_every_key_down_to(GRID.max_level - 1)) - {under}
    ready -= set(CubeGrid.children(under))
    selector = LodSelector(GRID)
    wanted = selector.select(
        camera, ready=ready.__contains__, bounds=lambda k: None
    ).wanted
    uncovered = [k[0] == GRID.max_level or CubeGrid.parent(k) == under for k in wanted]
    assert uncovered[0] and not uncovered[-1]
    assert uncovered == sorted(uncovered, reverse=True)  # all uncovered first
    covered = [k for k, u in zip(wanted, uncovered) if not u]
    ratios = [
        selector.distance(k, UNKNOWN_BOUNDS) / GRID.chunk_size(k[0]) for k in covered
    ]
    assert ratios == sorted(ratios)


def test_selection_wants_ground_in_view_first():
    camera = camera_at(40.0, 20.0, 300.0)
    ready = set(_every_key_down_to(4))
    east = TangentFrame.at(40.0, 20.0).east
    for forward in (east, -east):
        wanted = (
            LodSelector(GRID)
            .select(
                camera,
                ready=ready.__contains__,
                bounds=lambda k: None,
                forward=forward,
            )
            .wanted
        )
        ahead = [
            float((np.array(GRID.box(k).up) * R - camera) @ forward) > 0.0
            for k in wanted
        ]
        # More of the first third is in view than of the last third.
        third = len(ahead) // 3
        assert np.mean(ahead[:third]) > np.mean(ahead[-third:])


def _every_key_down_to(level):
    keys = list(GRID.roots())
    out = set(keys)
    while keys and keys[0][0] > level:
        keys = [c for k in keys for c in CubeGrid.children(k)]
        out.update(keys)
    return out


def _root_under(camera):
    up = np.asarray(camera) / np.linalg.norm(camera)
    face, i, j, _, _ = GRID.locate(up[None, :], GRID.max_level)
    return (GRID.max_level, int(face[0]), int(i[0]), int(j[0]))


def test_min_level_drops_the_finest_levels():
    full = LodSelector(GRID)
    coarse = LodSelector(GRID, min_level=3)
    camera = camera_at(40.0, 20.0, 50.0)
    kwargs = dict(ready=lambda k: True, bounds=lambda k: None)
    draw_full = full.select(camera, **kwargs).draw
    draw_coarse = coarse.select(camera, **kwargs).draw
    assert min(k[0] for k in draw_full) == 0
    assert min(k[0] for k in draw_coarse) == 3
    assert len(draw_coarse) < len(draw_full)
    # Only the finest levels change: coarse chunks far away are the same.
    assert {k for k in draw_full if k[0] >= 5} == {k for k in draw_coarse if k[0] >= 5}


def test_lod_pass_cost_is_bounded():
    """Guards the LOD pass from regressing badly (it runs in the frame)."""
    selector = LodSelector(GRID)
    camera = camera_at(40.0, 20.0, 10.0)
    selection = selector.select(camera, **ALL_READY)  # warm the per-key boxes
    assert 300 < len(selection.draw) < 900
    # The best of a few runs: one sample also times the scheduler, a GC pause
    # or another xdist worker.
    assert _best_ms(lambda: selector.select(camera, **ALL_READY)) < 50.0 * TIME_SCALE
