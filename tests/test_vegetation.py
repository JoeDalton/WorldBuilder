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
