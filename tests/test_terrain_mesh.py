"""
Unit tests for the terrain prototype's Panda geometry
(world_builder.terrain.mesh).

These build Geoms without a window. They cannot check the GLSL: Shader.load
succeeds with no graphics context, so the terrain shaders need the demo (or
an offscreen GL context) to validate.
"""

from dataclasses import replace

import numpy as np

from world_builder.terrain.chunkgen import (
    VERTEX_DTYPE,
    generate_chunk,
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
    geom = node.get_geom(0)
    vdata = geom.get_vertex_data()
    assert vdata.get_num_rows() == len(chunk.vertex_bytes) // VERTEX_DTYPE.itemsize
    assert bytes(memoryview(vdata.get_array(0))) == chunk.vertex_bytes
    assert geom.get_primitive(0).get_num_vertices() == len(triangle_indices())
    assert geom.get_primitive(0) == shared_triangles()
    bounds = node.get_internal_bounds()  # the box we set (culling wraps it in a sphere)
    np.testing.assert_allclose(bounds.get_min(), chunk.aabb_min)
    np.testing.assert_allclose(bounds.get_max(), chunk.aabb_max)
    local = chunk.vertices()["vertex"]
    assert np.all(local >= chunk.aabb_min) and np.all(local <= chunk.aabb_max)
    # Never a collision target (Terrain.enable_collision is the solid ground).
    assert node.get_into_collide_mask().is_zero()


def test_vertex_format_carries_the_height_column():
    assert vertex_format().get_array(0).has_column("height")
