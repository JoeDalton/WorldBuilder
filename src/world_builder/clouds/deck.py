"""
The deck: a sphere at the clouds' slab carrying their far field (the
expected coverage and its light), for the sea's reflection and the cloud
bins it draws in. Moved out of :mod:`sphere` (the billboards).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from panda3d.core import (
    CullBinManager,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    OmniBoundingVolume,
    Shader,
)

from world_builder.terrain.cubesphere import face_dirs

_SHADERS = Path(__file__).parent / "shaders"


#: Cull bins, between opaque (20) and transparent (30), after the sky's (21,
#: ``sky.SKY_BIN_SORT``): the deck's first, then one back-to-front bin per
#: shell, the coarsest first. A cirrus sheet above (:mod:`cirrus`) draws in
#: the sky's bin where it is behind them and in "transparent" where it is in
#: front.
CLOUD_BIN = "cloud_sphere"


_CLOUD_BIN_SORT = 22


#: Quads per face edge of the deck sphere.
DECK_QUADS = 48


def _ensure_bin():
    manager = CullBinManager.get_global_ptr()
    if manager.find_bin(CLOUD_BIN) == -1:
        # In a fixed order: two decks (a genus's) are ordered by their caller.
        manager.add_bin(CLOUD_BIN, CullBinManager.BT_fixed, _CLOUD_BIN_SORT)


def deck_mesh(
    radius: float, quads: int = DECK_QUADS, name: str = "cloud_deck"
) -> GeomNode:
    """A cube-sphere of radius *radius*, never culled (the caller makes it
    two-sided: the camera may be above or below it). The deck's, and the
    cirrus sheet's (:mod:`cirrus`)."""
    t = np.linspace(-1.0, 1.0, quads + 1)
    a, b = np.meshgrid(t, t)
    positions, indices = [], []
    for face in range(6):
        positions.append(face_dirs(face, a, b) * radius)
        first = face * (quads + 1) ** 2
        r, c = np.meshgrid(np.arange(quads), np.arange(quads), indexing="ij")
        v00 = first + (r * (quads + 1) + c).ravel()
        v01, v10 = v00 + 1, v00 + quads + 1
        indices.append(np.stack([v00, v01, v10 + 1, v00, v10 + 1, v10], 1).ravel())
    verts = np.concatenate(positions).astype(np.float32)
    index = np.concatenate(indices).astype(np.uint32)
    vdata = GeomVertexData("cloud_deck", GeomVertexFormat.get_v3(), GeomEnums.UH_static)
    vdata.unclean_set_num_rows(len(verts))
    memoryview(vdata.modify_array(0)).cast("B")[:] = verts.tobytes()
    tris = GeomTriangles(GeomEnums.UH_static)
    tris.set_index_type(GeomEnums.NT_uint32)
    tris.add_next_vertices(len(index))
    memoryview(tris.modify_vertices()).cast("B")[:] = index.tobytes()
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode(name)
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    return node


def deck_shader() -> Shader:
    """The deck's shader: its vertex stage (``cloud_deck.vert``) and the
    clouds' fragment stage (``cloud.frag``)."""
    return Shader.load(
        Shader.SL_GLSL,
        vertex=_SHADERS / "cloud_deck.vert",
        fragment=_SHADERS / "cloud.frag",
    )
