"""
Unit tests for the terrain prototype's Panda geometry
(world_builder.terrain.mesh).

These build Geoms without a window. They cannot check the GLSL: Shader.load
succeeds with no graphics context, so the terrain shaders need the demo (or
an offscreen GL context) to validate.
"""

from dataclasses import replace

import numpy as np
from panda3d.core import BitMask32, NodePath

from world_builder.depth import (
    CLOUD_FAR_BIT,
    CLOUD_NEAR_BIT,
    FAR_CAMERA_MASK,
    NEAR_CAMERA_MASK,
    REFLECTION_CAMERA_MASK,
)
from world_builder.terrain.chunkgen import (
    COARSE_STRIDE,
    QUADS,
    VERTEX_DTYPE,
    VERTS,
    generate_chunk,
    skirt_ring,
    triangle_indices,
)
from world_builder.terrain.mesh import (
    build_chunk_node,
    shared_triangles,
    vertex_format,
)
from world_builder.terrain.planet import PRESETS


def test_vertex_format_matches_the_packed_records():
    array = vertex_format().get_array(0)
    assert array.get_stride() == VERTEX_DTYPE.itemsize
    for name in VERTEX_DTYPE.names:
        assert array.get_column(name).get_start() == VERTEX_DTYPE.fields[name][1]


def test_chunk_node_holds_the_bytes_and_shared_indices():
    spec = replace(PRESETS["desert"], seed=3).resolved()
    chunk = generate_chunk(spec, (2, 4, 1, 1))
    node = build_chunk_node(chunk)
    fine, coarse = (child.node() for child in NodePath(node).get_children())
    geom = fine.get_geom(0)
    vdata = geom.get_vertex_data()
    assert vdata.get_num_rows() == len(chunk.vertex_bytes) // VERTEX_DTYPE.itemsize
    assert bytes(memoryview(vdata.get_array(0))) == chunk.vertex_bytes
    assert geom.get_primitive(0).get_num_vertices() == len(triangle_indices())
    assert geom.get_primitive(0) == shared_triangles()
    # The coarse mesh: the same vertices, the shared coarse indices
    assert coarse.get_geom(0).get_vertex_data() == vdata
    assert coarse.get_geom(0).get_primitive(0) == shared_triangles(COARSE_STRIDE)
    bounds = node.get_internal_bounds()  # the box we set (culling wraps it in a sphere)
    np.testing.assert_allclose(bounds.get_min(), chunk.aabb_min)
    np.testing.assert_allclose(bounds.get_max(), chunk.aabb_max)
    local = chunk.vertices()["vertex"]
    assert np.all(local >= chunk.aabb_min) and np.all(local <= chunk.aabb_max)
    # Never a collision target (Terrain.enable_collision is the solid ground).
    for part in (node, fine, coarse):
        assert part.get_into_collide_mask().is_zero()


def _shown_to(node, camera_mask: BitMask32) -> bool:
    control = node.get_draw_control_mask()
    shown = (control & node.get_draw_show_mask()) | ~control
    return not (shown & camera_mask).is_zero()


def test_the_coarse_passes_draw_the_coarse_mesh_alone():
    """The clouds' depth passes and the sea's reflection draw every other
    vertex (a quarter of the triangles: performance.md §34); the view's
    cameras the fine mesh alone."""
    spec = replace(PRESETS["desert"], seed=3).resolved()
    node = build_chunk_node(generate_chunk(spec, (2, 4, 1, 1)))
    fine, coarse = (child.node() for child in NodePath(node).get_children())
    for camera in (NEAR_CAMERA_MASK, FAR_CAMERA_MASK):
        assert _shown_to(fine, camera) and not _shown_to(coarse, camera)
    for bit in (CLOUD_NEAR_BIT, CLOUD_FAR_BIT):
        camera = BitMask32.bit(bit)
        assert _shown_to(coarse, camera) and not _shown_to(fine, camera)
    assert _shown_to(coarse, REFLECTION_CAMERA_MASK)
    assert not _shown_to(fine, REFLECTION_CAMERA_MASK)


def test_the_coarse_indices_take_every_other_vertex():
    fine, coarse = triangle_indices(), triangle_indices(COARSE_STRIDE)
    grid = 6 * QUADS * QUADS
    coarse_grid = coarse[: grid // 4].reshape(-1, 3)
    # Every corner on the even grid, each coarse quad over four fine ones
    rows, cols = np.divmod(coarse_grid, VERTS)
    assert np.all(rows % 2 == 0) and np.all(cols % 2 == 0)
    assert len(coarse_grid) * 3 * 4 == grid
    # The skirt hangs from every other edge vertex, its own below each
    skirt = coarse[grid // 4 :]
    top = skirt[skirt < VERTS * VERTS]
    assert set(top) == set(skirt_ring()[::2])
    hanging = skirt[skirt >= VERTS * VERTS] - VERTS * VERTS
    assert set(hanging) == set(range(0, 4 * QUADS, 2))
    # The first quad's two triangles are the fine ones scaled up: the same
    # diagonal and winding
    halved = np.stack(np.divmod(coarse_grid[:2], VERTS), axis=-1) // 2
    first = np.stack(np.divmod(fine[:6].reshape(-1, 3), VERTS), axis=-1)
    np.testing.assert_array_equal(halved, first)


def test_vertex_format_carries_the_height_column():
    assert vertex_format().get_array(0).has_column("height")


def test_the_demo_names_every_terrain_debug_view():
    """The demo's HUD names the view by index (``_DEBUG_VIEWS``): a view the
    terrain's shader has (``uDebugMode == N``) and the HUD does not name
    crashed it (the clouds' shadow, 6)."""
    import ast
    import re
    from pathlib import Path

    root = Path(__file__).parent.parent
    shader = (root / "src/world_builder/terrain/shaders/terrain_frag.glsl").read_text()
    modes = {int(n) for n in re.findall(r"uDebugMode == (\d+)", shader)}
    demo = ast.parse((root / "scripts/demo_terrain.py").read_text())
    (names,) = [
        node.value
        for node in demo.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "_DEBUG_VIEWS" for t in node.targets)
    ]
    assert len(names.elts) == max(modes) + 1
    assert 6 in modes  # the clouds' shadow alone
