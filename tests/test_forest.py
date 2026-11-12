"""The vegetation's placement and data (world_builder.terrain.forest: the
jittered grids, the rules, the far forest's per-vertex data), its density
paints, its assets (world_builder.datafiles) and the far forest's tables
(world_builder.terrain.far_forest)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from world_builder.datafiles import TREES_PATH, tree_assets
from world_builder.terrain import forest
from world_builder.terrain.biomes import BIOMES, DESERT, FOREST, GRASS, SNOW
from world_builder.terrain.cubesphere import UV_PERIOD, VERTS, face_dirs, grid_for
from world_builder.terrain.planet import PRESETS


def _chunk(biome: int, key=(1, 2, 300, 400)):
    spec = PRESETS["tiny"].resolved()
    grid = grid_for(spec.radius)
    weights = np.zeros((VERTS * VERTS, len(BIOMES)), dtype=np.uint8)
    weights[:, biome] = 255
    elevation = np.full((VERTS, VERTS), 40.0, dtype=np.float32)
    ka, kb = grid.grid_indices(key)
    a, b = np.meshgrid(grid.param(ka), grid.param(kb))
    dirs = face_dirs(key[1], a.ravel(), b.ravel()).reshape(VERTS, VERTS, 3)
    centre = grid.centre_dir(key) * (spec.radius + 40.0)
    return grid, key, dirs, elevation, weights, centre, spec


def _place(biome, key=(1, 2, 300, 400), **kwargs):
    return forest.chunk_forest(*_chunk(biome, key), **kwargs)


def test_the_grids_divide_the_period_and_follow_the_crowns():
    for grid in forest.GRIDS:
        n = forest.cells_per_period(grid)
        assert abs(n * forest.cell_size(grid) - UV_PERIOD) < 1e-9
    # A tree cell holds about a crown: its side near the trees' mean crown
    # diameter (the baked crowns).
    groups = [g for g in forest.SPECIES if g.grid == "tree"]
    areas = [forest.group_crown_area(g) for g in groups]
    diameters = 2.0 * np.sqrt(np.array(areas) / np.pi)
    assert diameters.min() * 0.7 < forest.TREE_CELL < diameters.max() * 1.5
    assert forest.SMALL_CELL < forest.TREE_CELL


def test_the_crowns_are_baked_for_every_asset():
    baked = json.loads(forest.CROWNS_PATH.read_text())
    assert set(forest.ASSETS) <= set(baked)
    for name in forest.ASSETS:
        assert baked[name]["area"] >= 0.0 and baked[name]["height"] > 0.0


def test_placement_is_deterministic():
    a, fa = _place(FOREST)
    b, fb = _place(FOREST)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fa, fb)


def test_one_tree_per_cell_at_most():
    plants, _ = _place(FOREST, key=(0, 2, 600, 800))
    grid, key, *_rest, spec = _chunk(FOREST, (0, 2, 600, 800))
    tree_assets_ = {
        forest.ASSETS.index(a)
        for g in forest.SPECIES
        if g.grid == "tree"
        for a in g.assets
    }
    trees = plants[np.isin(plants[:, 5].astype(int), list(tree_assets_))]
    d = np.linalg.norm(trees[:, None, :3] - trees[None, :, :3], axis=-1)
    d += np.eye(len(trees)) * 1e9
    # Jittered: two trees of neighbouring cells are at least (1 - 2 JITTER)
    # of a cell apart in uv metres (true metres: the cube face's
    # distortion shrinks them, by 0.6 at worst).
    assert d.min() >= (1.0 - 2.0 * forest.JITTER) * forest.TREE_CELL * 0.55


@pytest.mark.parametrize(
    "biome, groups",
    [
        (FOREST, {"conifers", "birches", "bushes"}),
        (GRASS, {"conifers", "birches", "bushes", "rocks"}),
        (DESERT, {"dry bushes", "rocks"}),
        (SNOW, {"snowy pines", "rocks"}),
    ],
)
def test_each_biome_grows_its_own(biome, groups):
    plants, _ = _place(biome, key=(0, 2, 600, 800))
    assert plants is not None
    names = {forest.ASSETS[int(a)] for a in plants[:, 5]}
    allowed = set()
    for g in forest.SPECIES:
        if g.name in groups:
            allowed |= set(g.assets)
    assert names <= allowed


def test_a_forest_is_dense_and_a_desert_bare():
    forest_n = len(_place(FOREST)[0])
    desert = _place(DESERT)[0]
    assert forest_n > 10 * (0 if desert is None else len(desert))


def test_neighbours_share_no_plant():
    """A sample belongs to the chunk whose uv square holds it."""
    a = _chunk(FOREST, (1, 2, 300, 400))
    b = _chunk(FOREST, (1, 2, 301, 400))
    pa, _ = forest.chunk_forest(*a)
    pb, _ = forest.chunk_forest(*b)
    wa = pa[:, :3] + a[5]
    wb = pb[:, :3] + b[5]
    d = np.linalg.norm(wa[:, None] - wb[None], axis=-1)
    assert d.min() > 0.5


def test_coarse_chunks_keep_a_subset_of_their_childrens_trees():
    assert forest.thinning(4.0) == (1.0, 1.0)
    keep, grow = forest.thinning(16.0)
    assert keep == 0.5 and grow > 1.0
    assert forest.thinning(64.0) == (0.0, 0.0)


def test_the_rules_ramp_slope_density_and_convexity():
    w = np.zeros((4, 6))
    w[:, FOREST] = 1.0
    ones = np.ones(4)
    conifers = forest.SPECIES.index(
        next(g for g in forest.SPECIES if g.name == "conifers")
    )
    flat = forest.probabilities(w, ones, 0 * ones, 0 * ones, 100 * ones)
    steep = forest.probabilities(w, ones, 50 * ones, 0 * ones, 100 * ones)
    sparse = forest.probabilities(w, 0.5 * ones, 0 * ones, 0 * ones, 100 * ones)
    crest = forest.probabilities(w, ones, 0 * ones, 0.1 * ones, 100 * ones)
    assert flat[0, conifers] > 0.5
    assert steep[0, conifers] == 0.0
    np.testing.assert_allclose(sparse[0, conifers], 0.5 * flat[0, conifers])
    assert crest[0, conifers] < flat[0, conifers]
    # A grid's groups share its cells.
    for k in range(len(forest.GRIDS)):
        mine = forest._tables()["grid"] == k
        assert np.all(flat[:, mine].sum(axis=1) <= 1.0 + 1e-9)


def test_the_density_thins_the_trees():
    count = VERTS * VERTS
    full, far_full = _place(FOREST, density=np.ones(count))
    half, far_half = _place(FOREST, density=np.full(count, 0.4))
    assert len(half) < 0.6 * len(full)
    assert far_half[:, 1].mean() < 0.6 * far_full[:, 1].mean()


def test_the_far_data_matches_the_trees():
    """The cover (summed crown area per area) of the trees placed matches
    its expected value: the far forest is as dense as the trees it stands
    for."""
    _, far = _place(FOREST, key=(0, 2, 600, 800))
    cover = far[:, 0].astype(float) / 255.0 * forest.COVER_MAX
    lam = far[:, 1:].astype(float) / 255.0
    expected = lam[:, : len(forest.FAR_GROUPS)] @ forest._tables()["far_cover"]
    assert abs(cover.mean() - expected.mean()) < 0.25 * expected.mean()
    # A coarse chunk (none placed) has the expected cover itself.
    _, far = _place(FOREST, key=(6, 2, 9, 12))
    cover = far[:, 0].astype(float) / 255.0 * forest.COVER_MAX
    lam = far[:, 1:].astype(float) / 255.0
    expected = lam[:, : len(forest.FAR_GROUPS)] @ forest._tables()["far_cover"]
    np.testing.assert_allclose(cover, expected, atol=2.5 / 255.0 * forest.COVER_MAX)


def test_plants_stand_on_the_ground():
    grid, key, dirs, elevation, weights, centre, spec = _chunk(FOREST)
    plants, _ = forest.chunk_forest(grid, key, dirs, elevation, weights, centre, spec)
    height = np.linalg.norm(plants[:, :3] + centre, axis=1) - spec.radius
    np.testing.assert_allclose(height, 40.0 - forest.SINK, atol=0.05)


def test_a_barren_planet_grows_nothing():
    grid, key, dirs, elevation, weights, centre, spec = _chunk(FOREST)
    from dataclasses import replace

    plants, far = forest.chunk_forest(
        grid, key, dirs, elevation, weights, centre, replace(spec, has_vegetation=False)
    )
    assert plants is None and not far.any()


def test_a_chunk_carries_the_far_forest_in_its_vertices():
    from world_builder.terrain.chunkgen import VERTEX_DTYPE, generate_chunk
    from world_builder.terrain.level import TINY_LEVEL, load_level

    assert VERTEX_DTYPE.itemsize == 40
    spec, _ = load_level(TINY_LEVEL)
    grid = grid_for(spec.radius)
    from world_builder.terrain.chart import dirs_from_latlon

    d = np.asarray(dirs_from_latlon(32.0, -147.6)).reshape(1, 3)
    face, i, j, _, _ = grid.locate(d, 2)
    chunk = generate_chunk(spec, (2, int(face[0]), int(i[0]), int(j[0])))
    v = chunk.vertices()
    assert v["forest"][: VERTS * VERTS, 1].mean() > 20  # the south-east forest
    from world_builder.terrain.chunkgen import skirt_ring

    np.testing.assert_array_equal(
        v["forest"][VERTS * VERTS :], v["forest"][skirt_ring()]
    )
    assert (
        chunk.density is not None
        and 0.0 <= chunk.density.min() <= chunk.density.max() <= 1.0
    )
    assert chunk.convexity is not None and np.isfinite(chunk.convexity).all()


def test_a_density_paint_overrides_the_procedural_density(tmp_path):
    from world_builder.terrain.chart import AzimuthalEquidistantChart
    from world_builder.terrain.pngio import write_png
    from world_builder.terrain.stamps import DensityPaint, Placement

    mask = np.zeros((64, 64, 4), dtype=np.uint8)
    mask[:, :32, 3] = 255  # the west half painted bare, the rest untouched
    path = tmp_path / "density.png"
    write_png(path, mask)
    placement = Placement(lat=10.0, lon=20.0, width=1000.0, height=1000.0)
    paint = DensityPaint(mask=str(path), placement=placement)
    radius = 20000.0
    chart = AzimuthalEquidistantChart(10.0, 20.0, radius)
    dirs = chart.to_dirs(np.array([-250.0, 250.0, 2000.0]), np.zeros(3)).reshape(-1, 3)
    out = paint.apply(dirs, np.full(3, 0.8), radius)
    assert out[0] < 0.05  # painted bare
    assert abs(out[1] - 0.8) < 1e-6  # not painted (alpha 0)
    assert abs(out[2] - 0.8) < 1e-6  # outside


def test_every_asset_and_impostor_is_there():
    names = {a.name for a in tree_assets()}
    assert set(forest.ASSETS) <= names
    for a in tree_assets():
        assert (TREES_PATH / a.file).exists()
    meta = json.loads((TREES_PATH / "impostors" / "impostors.json").read_text())
    for e in meta["assets"]:
        for kind in ("albedo", "normal"):
            assert (TREES_PATH / "impostors" / f"{kind}_{e['layer']:02d}.png").exists()
    assert len(forest.ASSETS) <= 32  # tree_impostor.vert's uImpostors[32]
    assert (TREES_PATH / "CREDITS.md").exists()


def test_the_far_forest_tables_are_baked_for_these_trees():
    from world_builder.terrain import far_forest

    data = far_forest.load_tables()
    assert data is not None, "re-run scripts/bake_forest_tables.py"
    table = data["table"]
    assert table.shape == (len(forest.FAR_GROUPS), 8, 16, 16, 16, 4)
    assert np.all((table >= 0.0) & (table <= 1.0))
    # No trees: all the ground seen and lit; more trees: less of it.
    assert np.all(table[:, 0, ..., :2] == 1.0)
    seen = table[:, :, 0, 0, 0, 1]  # straight down
    assert np.all(np.diff(seen, axis=1) <= 1e-6)
    # Sunlit ground is seen ground (kg <= kg + kgs); a lower sun lights less.
    assert np.all(table[..., 0] <= table[..., 1] + 1e-6)
    assert np.all(np.diff(table[:, 1:, 0, :, 0, 0], axis=2) <= 0.02)
    assert np.all(np.diff(data["D"][:, 1:], axis=1) <= 1e-6)
