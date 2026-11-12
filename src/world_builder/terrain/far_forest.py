"""
The far forest's tables on the GPU (VEGETATION_PLAN.md, part V2): Bruneton
and Neyret's G, T, D and E [BN12 §6], baked from our trees by
``scripts/bake_forest_tables.py`` (``datafiles/trees/forest_tables.npz``),
for the terrain's shader-map (``shaders/vegetation_far.glsl``).

One 3D texture holds every far group's G, T and E (x the view's zenith
angle, y the sun's, z the group, the density and the azimuth: (group 8 +
lambda) 16 + phi; RGBA: kg, kg + kgs, T, E), linear-filtered along the three
(the shader mixes two densities itself), 16-bit unsigned normalised. D,
the crowns' mean albedo and their mean area per cell go in uniforms.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
from panda3d.core import NodePath, PTA_float, PTA_LVecBase3f, SamplerState, Texture

from world_builder.datafiles import TREES_PATH
from world_builder.terrain import forest

LOGGER = logging.getLogger(__name__)
TABLES_PATH = TREES_PATH / "forest_tables.npz"


@functools.cache
def load_tables() -> dict | None:
    """:returns: the baked tables (``forest_tables.npz``'s arrays), or None
    if missing or baked for other far groups or another trees' grid"""
    if not TABLES_PATH.exists():
        LOGGER.warning(
            "no far forest tables (%s): run scripts/bake_forest_tables.py", TABLES_PATH
        )
        return None
    data = dict(np.load(TABLES_PATH))
    names = [str(n) for n in data["groups"]]
    expected = [forest.SPECIES[g].name for g in forest.FAR_GROUPS]
    if names != expected or abs(float(data["tree_cell"]) - forest.TREE_CELL) > 1e-3:
        LOGGER.warning(
            "the far forest tables are stale (groups %s, cell %.3f m): "
            "re-run scripts/bake_forest_tables.py",
            names,
            float(data["tree_cell"]),
        )
        return None
    return data


def table_texture(table: np.ndarray) -> Texture:
    """:returns: the 3D texture of a (groups, lambda, phi, theta_l, theta_v,
    4) table"""
    groups, lambdas, phis, n_l, n_v, _ = table.shape
    tex = Texture("forest_tables")
    # 16-bit unsigned normalised: the fractions are in [0, 1], and half
    # the floats' bytes (the cache: the shader fetches it per pixel).
    tex.setup_3d_texture(
        n_v, n_l, groups * lambdas * phis, Texture.T_unsigned_short, Texture.F_rgba16
    )
    pages = np.ascontiguousarray(
        np.rint(np.clip(table, 0.0, 1.0) * 65535.0).reshape(
            groups * lambdas * phis, n_l, n_v, 4
        )[..., [2, 1, 0, 3]],
        dtype=np.uint16,
    )
    tex.set_ram_image(pages.tobytes())
    tex.set_compression(Texture.CM_off)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    for wrap in (tex.set_wrap_u, tex.set_wrap_v, tex.set_wrap_w):
        wrap(SamplerState.WM_clamp)
    return tex


def bind(root: NodePath) -> bool:
    """Give *root* (a terrain's) the tables' inputs. :returns: whether there
    are tables (else the far forest is off: ``uForestGroups`` 0)"""
    data = load_tables()
    d = PTA_float.empty_array(24)
    albedo = PTA_LVecBase3f.empty_array(3)
    crown = PTA_float.empty_array(3)
    if data is None:
        root.set_shader_input("uForestGroups", 0)
        root.set_shader_input("uForestTable", _empty_texture())
    else:
        groups = data["table"].shape[0]
        for g in range(groups):
            for k in range(8):
                d[g * 8 + k] = float(data["D"][g, k])
            albedo[g] = tuple(float(c) for c in data["albedo"][g])
            crown[g] = float(data["crown_cover"][g])
        root.set_shader_input("uForestGroups", groups)
        root.set_shader_input("uForestTable", _texture())
    root.set_shader_input("uForestD", d)
    root.set_shader_input("uForestAlbedo", albedo)
    root.set_shader_input("uForestCrown", crown)
    return data is not None


@functools.cache
def _texture() -> Texture:
    return table_texture(load_tables()["table"])


@functools.cache
def _empty_texture() -> Texture:
    return table_texture(np.ones((1, 8, 16, 16, 16, 4), np.float32))
