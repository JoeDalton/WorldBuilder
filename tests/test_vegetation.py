"""The vegetation's setting (world_builder.terrain.vegetation) and the far
forest's shaders against the tables they read (world_builder.terrain.far_forest)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from world_builder.quality import QualityLevel
from world_builder.terrain import far_forest, forest
from world_builder.terrain import vegetation as veg

SHADERS = Path(veg.__file__).parent / "shaders"


def test_lower_presets_draw_fewer_trees_one_by_one():
    assert veg.VegetationSettings().enabled
    presets = [veg.VegetationSettings.preset(level) for level in QualityLevel]
    for name in ("mesh_range", "tree_range"):
        values = [getattr(p, name) for p in presets]
        assert values == sorted(values)
    assert all(p.mesh_range < p.tree_range for p in presets)


def _const(source: str, name: str) -> float:
    return float(re.search(rf"const float {name} = ([0-9.]+);", source).group(1))


def test_the_shader_reads_the_tables_layout():
    source = (SHADERS / "forest_table.glsl").read_text()
    data = far_forest.load_tables()
    assert data is not None
    groups, lambdas, phis, n_l, n_v, channels = data["table"].shape
    assert _const(source, "FOREST_LAMBDAS") == lambdas == len(data["lambdas"])
    assert _const(source, "FOREST_ANGLES") == phis == n_l == n_v
    assert channels == 4 and groups == len(forest.FAR_GROUPS) <= forest.MAX_FAR_GROUPS
    # uForestD[24], uForestAlbedo[3], uForestCrown[3]
    assert groups * lambdas <= 24 and groups <= 3
    # The angles' axes the shader assumes: linear over a right angle (θ)
    # and a half turn (φ), λ over [0, 1].
    np.testing.assert_allclose(data["theta_deg"], np.linspace(0, 90, n_v), atol=1e-4)
    np.testing.assert_allclose(data["phi_deg"], np.linspace(0, 180, phis), atol=1e-4)
    np.testing.assert_allclose(data["lambdas"], np.linspace(0, 1, lambdas), atol=1e-6)


def test_the_cover_byte_scale_matches_the_shader():
    source = (SHADERS / "forest_table.glsl").read_text()
    assert f"forest.x * {forest.COVER_MAX}" in source


def test_the_table_texture_holds_the_table():
    data = far_forest.load_tables()
    tex = far_forest.table_texture(data["table"])
    groups, lambdas, phis, n_l, n_v, _ = data["table"].shape
    assert (tex.get_x_size(), tex.get_y_size(), tex.get_z_size()) == (
        n_v,
        n_l,
        groups * lambdas * phis,
    )


def test_the_plants_follow_their_chunks_distance():
    """Trees.update: impostors for the drawn chunks with plants within
    tree_range (by their box's distance), the meshes for those within
    mesh_range and its margin, nothing for chunks without plants or nodes."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from world_builder.terrain.trees import MESH_MARGIN, Trees

    def chunk(x, plants=True):
        return SimpleNamespace(
            centre=np.array([x, 0.0, 0.0]),
            aabb_min=np.array([-100.0, -100.0, -100.0]),
            aabb_max=np.array([100.0, 100.0, 100.0]),
            forest=np.zeros((1, 7), np.float32) if plants else None,
        )

    store = {
        "close": chunk(0.0),  # the camera in its box
        "mid": chunk(1100.0),  # 1000 m off its box
        "far": chunk(5100.0),  # 5000 m: past both
        "bare": chunk(0.0, plants=False),
        "unbuilt": chunk(0.0),
    }
    settings = SimpleNamespace(mesh_range=500.0, tree_range=3000.0)
    trees = object.__new__(Trees)
    trees._terrain = SimpleNamespace(
        store=store,
        nodes={k for k in store if k != "unbuilt"},
        vegetation=SimpleNamespace(settings=settings, active=True),
    )
    trees._points, trees._counts, trees._bare, trees.stats = {}, {}, set(), {}
    trees._point_node = MagicMock(side_effect=lambda key, chunk, plants: key)
    trees._fill_meshes = MagicMock()
    trees.update(set(store), np.zeros(3), np.zeros(3))
    assert set(trees._points) == {"close", "mid"}
    near = trees._fill_meshes.call_args.args[0]
    assert near == ["close"] and 1000.0 > 500.0 + MESH_MARGIN


def test_a_chunk_without_impostor_plants_is_looked_at_once():
    """A chunk in range whose plants have no impostor (rocks, bushes) is
    looked at on its first pass only, until it is dropped."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from world_builder.terrain.trees import Trees

    chunk = SimpleNamespace(
        centre=np.zeros(3),
        aabb_min=np.full(3, -100.0),
        aabb_max=np.full(3, 100.0),
        forest=np.zeros((1, 7), np.float32),
    )
    trees = object.__new__(Trees)
    trees._terrain = SimpleNamespace(
        store={"bare": chunk},
        nodes={"bare"},
        vegetation=SimpleNamespace(
            settings=SimpleNamespace(mesh_range=500.0, tree_range=3000.0),
            active=True,
        ),
    )
    trees._points, trees._counts, trees._bare, trees.stats = {}, {}, set(), {}
    trees._point_node = MagicMock(return_value=None)
    trees._fill_meshes = MagicMock()
    for _ in range(3):
        trees.update({"bare"}, np.zeros(3), np.zeros(3))
    assert trees._point_node.call_count == 1
    trees.drop("bare")
    trees.update({"bare"}, np.zeros(3), np.zeros(3))
    assert trees._point_node.call_count == 2
