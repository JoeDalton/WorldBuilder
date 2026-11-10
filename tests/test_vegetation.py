"""The vegetation (world_builder.terrain.vegetation): its placement (the numpy
twin of shaders/vegetation_cells.glsl), the shells' data and settings."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from world_builder.quality import QualityLevel
from world_builder.terrain import vegetation as veg
from world_builder.terrain.biomes import BIOMES, DESERT, FOREST, GRASS, ROCK
from world_builder.terrain.cubesphere import QUADS, UV_PERIOD, VERTS, grid_for
from world_builder.terrain.planet import PRESETS

SHADERS = Path(veg.__file__).parent / "shaders"
SEED = 7


def _cells(n: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n) + offset
    return i % 1000 + 3 * 640, i // 1000 + 640


def _weights(n: int, **w: float) -> np.ndarray:
    out = np.zeros((n, 4))
    for k, name in enumerate(("forest", "grass", "desert", "rock")):
        out[:, k] = w.get(name, 0.0)
    return out


def _pcg3d(x: int, y: int, z: int) -> tuple[int, int, int]:
    """pcg3d in plain Python integers (mod 2**32): the reference."""
    m = 0xFFFFFFFF
    x, y, z = ((v * 1664525 + 1013904223) & m for v in (x, y, z))
    x = (x + y * z) & m
    y = (y + z * x) & m
    z = (z + x * y) & m
    x, y, z = x ^ (x >> 16), y ^ (y >> 16), z ^ (z >> 16)
    x = (x + y * z) & m
    y = (y + z * x) & m
    z = (z + x * y) & m
    return x, y, z


def test_the_hash_wraps_as_the_shader_does():
    """The numpy hash is pcg3d on uint32, bit for bit."""
    xs = np.array([0, 1, 12345, 2**31 + 7, 2**32 - 1], dtype=np.int64)
    ys = np.array([5, 0, 98765, 3, 2**32 - 2], dtype=np.int64)
    zs = np.array([9, 2, 4, 2**30, 1], dtype=np.int64)
    got = veg._hash(xs, ys, zs)
    for k in range(len(xs)):
        want = _pcg3d(int(xs[k]), int(ys[k]), int(zs[k]))
        assert tuple(int(h[k]) for h in got) == want


def test_placement_is_deterministic_and_seeded():
    cu, cv = _cells(5000)
    w = _weights(5000, forest=1.0)
    a = veg.plants(2, cu, cv, w, SEED)
    b = veg.plants(2, cu, cv, w, SEED)
    c = veg.plants(2, cu, cv, w, SEED + 1)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert (a["kind"] != c["kind"]).any()


@pytest.mark.parametrize(
    "biome, shares",
    [
        ("forest", {veg.TREE: veg.TREE_FOREST}),
        ("grass", {veg.TREE: veg.TREE_GRASS, veg.BUSH: veg.BUSH_GRASS}),
        ("desert", {veg.DRY: veg.DRY_DESERT}),
        ("rock", {veg.TREE: veg.TREE_ROCK, veg.BUSH: veg.BUSH_ROCK}),
    ],
)
def test_each_biome_grows_its_share_of_plants(biome, shares):
    """Dense trees in forest, sparse trees and bushes on grass and rock, very
    sparse dry bushes in desert (a Monte Carlo over 10^5 cells)."""
    n = 100_000
    cu, cv = _cells(n)
    p = veg.plants(0, cu, cv, _weights(n, **{biome: 1.0}), SEED)
    for kind in (veg.TREE, veg.BUSH, veg.DRY):
        share = float((p["kind"] == kind).mean())
        assert share == pytest.approx(shares.get(kind, 0.0), abs=0.006), kind


def test_snow_and_beach_grow_nothing():
    n = 20_000
    cu, cv = _cells(n)
    p = veg.plants(0, cu, cv, _weights(n), SEED)
    assert (p["kind"] == veg.NONE).all()


def test_a_crown_stays_in_its_cell():
    """The march tests a cell's own plant only: no crown may cross it."""
    n = 50_000
    cu, cv = _cells(n)
    p = veg.plants(1, cu, cv, _weights(n, forest=0.5, grass=0.5), SEED)
    grown = p["kind"] != veg.NONE
    reach = np.abs(p["offset"][grown]).max(axis=1) + p["radius"][grown]
    assert reach.max() <= 0.5 * veg.CELL + 1e-9


def test_sizes_stay_in_their_ranges():
    n = 50_000
    cu, cv = _cells(n)
    p = veg.plants(3, cu, cv, _weights(n, grass=0.6, desert=0.4), SEED)
    for kind, heights, radii in (
        (veg.TREE, veg.TREE_HEIGHT, veg.TREE_RADIUS),
        (veg.BUSH, veg.BUSH_HEIGHT, veg.BUSH_RADIUS),
        (veg.DRY, veg.DRY_HEIGHT, veg.DRY_RADIUS),
    ):
        mask = p["kind"] == kind
        assert mask.any()
        assert (
            heights[0]
            <= p["height"][mask].min()
            <= p["height"][mask].max()
            <= heights[1]
        )
        assert (
            radii[0] <= p["radius"][mask].min() <= p["radius"][mask].max() <= radii[1]
        )


def test_the_coverage_is_the_crowns_mean_area():
    """vegCoverage (the far look) against the plants' own crowns."""
    n = 100_000
    cu, cv = _cells(n)
    w = _weights(n, forest=0.6, grass=0.3, desert=0.1)
    p = veg.plants(0, cu, cv, w, SEED)
    area = np.where(p["kind"] != veg.NONE, np.pi * p["radius"] ** 2, 0.0)
    measured = area.mean() / veg.CELL**2
    assert veg.coverage(w[:1])[0] == pytest.approx(measured, rel=0.03)


def _glsl_constants(text: str) -> dict[str, list[float]]:
    out = {}
    for name, value in re.findall(r"const (?:float|int|vec2) (\w+) = ([^;]+);", text):
        numbers = re.findall(r"(?<![\w.])-?\d+\.?\d*", value.replace("vec2", ""))
        out[name] = [float(v) for v in numbers]
    return out


def test_the_shader_constants_are_the_twin_s():
    """vegetation_cells.glsl and vegetation.py place the same plants."""
    glsl = _glsl_constants((SHADERS / "vegetation_cells.glsl").read_text())
    pairs = {
        "VEG_CELL": [veg.CELL],
        "VEG_CELLS_PER_PERIOD": [veg.CELLS_PER_PERIOD],
        "VEG_TREE_FOREST": [veg.TREE_FOREST],
        "VEG_TREE_GRASS": [veg.TREE_GRASS],
        "VEG_BUSH_GRASS": [veg.BUSH_GRASS],
        "VEG_DRY_DESERT": [veg.DRY_DESERT],
        "VEG_TREE_ROCK": [veg.TREE_ROCK],
        "VEG_BUSH_ROCK": [veg.BUSH_ROCK],
        "VEG_TREE_HEIGHT": list(veg.TREE_HEIGHT),
        "VEG_TREE_RADIUS": list(veg.TREE_RADIUS),
        "VEG_BUSH_HEIGHT": list(veg.BUSH_HEIGHT),
        "VEG_BUSH_RADIUS": list(veg.BUSH_RADIUS),
        "VEG_DRY_HEIGHT": list(veg.DRY_HEIGHT),
        "VEG_DRY_RADIUS": list(veg.DRY_RADIUS),
        "VEG_REACH": [veg.REACH],
    }
    for name, value in pairs.items():
        assert glsl[name] == pytest.approx(value), name
    kinds = {
        "VEG_NONE": veg.NONE,
        "VEG_TREE": veg.TREE,
        "VEG_BUSH": veg.BUSH,
        "VEG_DRY": veg.DRY,
    }
    for name, value in kinds.items():
        assert glsl[name] == [value], name
    # The layer's rows as the shaders read them.
    frag = (SHADERS / "canopy_frag.glsl").read_text()
    assert f"vec2(65.0, {float(veg.LAYER_ROWS)})" in frag
    assert f"ivec3(q.x, {veg.LAYER_LIFT_ROW} + q.y" in frag


def test_biome_columns_are_forest_grass_desert_rock():
    w = np.zeros((VERTS * VERTS, len(BIOMES)), dtype=np.uint8)
    w[:, FOREST], w[:, GRASS], w[:, DESERT], w[:, ROCK] = 255, 128, 64, 32
    four = veg.layer_weights(w)
    assert four[0, 0] == pytest.approx([1.0, 128 / 255, 64 / 255, 32 / 255])


# ── a chunk ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny():
    spec = PRESETS["tiny"].resolved()
    return spec, grid_for(spec.radius)


def _chunk(grid, level=1, face=1, i=37, j=91):
    key = (level, face, i, j)
    ground = (
        30.0 + 4.0 * np.sin(np.arange(VERTS)[None, :] / 9.0) + np.zeros((VERTS, VERTS))
    )
    return key, ground.astype(np.float32)


def test_the_period_and_uv_match_the_grid(tiny):
    _, grid = tiny
    key = (1, 2, 1023, 77)
    uv0, step, period = veg.chunk_uv(grid, key)
    assert uv0 == grid.uv_origin(key)
    unwrapped = ((key[2] * QUADS) << key[0]) * grid.uv_unit
    assert period[0] * UV_PERIOD + uv0[0] == pytest.approx(unwrapped)
    assert step == (1 << key[0]) * grid.uv_unit


def test_the_frame_s_dual_axes_invert_its_steps(tiny):
    """F_u . E_u = 1, F_u . E_v = 0 (and v alike), all on the tangent plane."""
    spec, grid = tiny
    key = (1, 4, 200, 311)
    centre = grid.centre_dir(key) * (spec.radius + 50.0)
    frame = veg.chunk_frame(grid, key, centre, spec.radius).astype(np.float64)
    eu, ev, fu, fv, offset, up = (frame[k, :3] for k in range(6))
    assert fu @ eu == pytest.approx(1.0, abs=1e-5)
    assert fv @ ev == pytest.approx(1.0, abs=1e-5)
    assert abs(fu @ ev) < 1e-5 and abs(fv @ eu) < 1e-5
    assert abs(eu @ up) < 1e-4 and abs(fu @ up) < 1e-4
    assert np.linalg.norm(eu) == pytest.approx(grid.spacing(1), rel=0.35)
    # O is the middle vertex at sea level.
    assert np.linalg.norm(centre + offset) == pytest.approx(spec.radius, rel=1e-6)


def test_the_shell_covers_every_crown(tiny):
    """Every vertex around a crown lifts the shell over the crown's top."""
    spec, grid = tiny
    key, ground = _chunk(grid)
    uv0, step, period = veg.chunk_uv(grid, key)
    w = np.zeros((VERTS, VERTS, 4), dtype=np.float32)
    w[..., 0] = 0.7
    w[..., 1] = 0.3
    found = veg.chunk_plants(uv0, step, period, key[1], ground, w, SEED, -1e9)
    assert len(found["base"]) > 100
    lift = veg.shell_lift(found, ground, step)
    top = ground + lift
    for g, base, height, radius in zip(
        found["g"], found["base"], found["height"], found["radius"]
    ):
        reach = radius / step
        u = np.arange(VERTS)
        near_u = np.abs(u - g[0]) <= reach
        near_v = np.abs(u - g[1]) <= reach
        if not near_u.any() or not near_v.any():
            continue  # a margin cell's plant: its crown is outside the chunk
        assert (top[np.ix_(near_v, near_u)] >= base + height).all()


def test_a_bare_chunk_draws_no_shell(tiny):
    spec, grid = tiny
    key, ground = _chunk(grid)
    weights = np.zeros((VERTS * VERTS, len(BIOMES)), dtype=np.uint8)
    weights[:, 0] = 255  # snow
    assert veg.chunk_lift(grid, key, ground, weights, spec) is None
    weights[:, 0], weights[:, FOREST] = 0, 255
    assert veg.chunk_lift(grid, key, ground, weights, spec) is not None
    coarse = (6, 1, 0, 0)  # vertex spacing ~120 m: the far look's
    assert grid.spacing(6) > veg.MAX_SPACING
    assert veg.chunk_lift(grid, coarse, ground, weights, spec) is None


def test_the_layer_holds_ground_weights_frame_and_lift(tiny):
    spec, grid = tiny
    key, ground = _chunk(grid)
    weights = np.zeros((VERTS * VERTS, len(BIOMES)), dtype=np.uint8)
    weights[:, FOREST], weights[:, DESERT], weights[:, ROCK] = 200, 30, 25
    lift = veg.chunk_lift(grid, key, ground, weights, spec)
    centre = grid.centre_dir(key) * spec.radius
    layer = veg.canopy_layer(grid, key, centre, spec.radius, ground, weights, lift)
    assert layer.shape == (veg.LAYER_ROWS, veg.LAYER_SIDE, 4)
    np.testing.assert_array_equal(layer[:VERTS, :, 0], ground)
    assert layer[0, 0, 1] == pytest.approx(200 / 255)
    assert layer[0, 0, 3] == 30 + 256 * 25  # desert + 256 rock (bytes)
    np.testing.assert_array_equal(layer[veg.LAYER_LIFT_ROW :, :, 0], lift)
    # Per quad: never over its corners' ground, never under their top.
    quads = layer[veg.LAYER_LIFT_ROW : veg.LAYER_LIFT_ROW + QUADS, :QUADS]
    top = np.where(lift > 0, ground + lift, -1e9)
    assert (quads[..., 1] <= ground[:-1, :-1]).all()
    assert (quads[..., 2] >= top[1:, 1:]).all()


def test_the_shell_s_index_buffer_is_its_grid_and_walls():
    tris = veg.shell_triangles()
    assert tris.get_num_vertices() == 6 * QUADS * QUADS + 6 * 4 * QUADS


def test_lower_presets_draw_fewer_plants_one_by_one():
    ranges = [veg.VegetationSettings.preset(level).range for level in QualityLevel]
    assert ranges == sorted(ranges) and ranges[0] == 0.0
    assert veg.VegetationSettings().range == veg.VegetationSettings.preset("high").range
