"""
Chunk data to Panda geometry.

The vertex bytes arrive pre-packed from the worker in :data:`VERTEX_DTYPE`
layout, so building a chunk is one memory copy. All chunks share one index
buffer. Each GeomNode gets the chunk's precomputed bounding box, so Panda
never walks the vertices to compute one.

Chunk nodes are not collision targets: Panda would otherwise collide
colliders whose from-mask has GeomNode's default bit with every drawn
triangle, at about a microsecond each. Terrain collision goes through
``Terrain.enable_collision`` instead.
"""

from __future__ import annotations

import functools

from panda3d.core import (
    BitMask32,
    BoundingBox,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    InternalName,
    Point3,
)

from world_builder.terrain.chunkgen import VERTEX_DTYPE, ChunkData, triangle_indices


@functools.cache
def vertex_format() -> GeomVertexFormat:
    """:returns: the registered format matching :data:`VERTEX_DTYPE` byte for byte"""
    array = GeomVertexArrayFormat()
    array.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    array.add_column(InternalName.get_normal(), 3, Geom.NT_float32, Geom.C_normal)
    array.add_column(InternalName.make("height"), 1, Geom.NT_float32, Geom.C_other)
    # C_color makes Panda pass the bytes normalised to [0, 1].
    array.add_column(InternalName.make("biome_a"), 4, Geom.NT_uint8, Geom.C_color)
    array.add_column(InternalName.make("biome_b"), 4, Geom.NT_uint8, Geom.C_color)
    for name in VERTEX_DTYPE.names:
        column = array.get_column(name)
        expected = VERTEX_DTYPE.fields[name][1]
        if column.get_start() != expected:
            raise RuntimeError(f"column {name} at {column.get_start()}, not {expected}")
    if array.get_stride() != VERTEX_DTYPE.itemsize:
        raise RuntimeError(f"stride {array.get_stride()} != {VERTEX_DTYPE.itemsize}")
    return GeomVertexFormat.register_format(GeomVertexFormat(array))


@functools.cache
def shared_triangles() -> GeomTriangles:
    """:returns: the one index buffer all chunks draw with"""
    indices = triangle_indices()
    tris = GeomTriangles(Geom.UH_static)
    tris.set_index_type(GeomEnums.NT_uint16)
    tris.add_next_vertices(indices.size)
    memoryview(tris.modify_vertices()).cast("B")[:] = indices.tobytes()
    return tris


def build_chunk_node(chunk: ChunkData) -> GeomNode:
    """A GeomNode for one chunk, relative to its centre (:attr:`ChunkData.centre`).

    :param chunk: generated chunk data
    :returns: the node, bounds final
    """
    vdata = GeomVertexData(f"terrain{chunk.key}", vertex_format(), Geom.UH_static)
    rows = len(chunk.vertex_bytes) // VERTEX_DTYPE.itemsize
    vdata.unclean_set_num_rows(rows)
    memoryview(vdata.modify_array(0)).cast("B")[:] = chunk.vertex_bytes
    bounds = BoundingBox(Point3(*chunk.aabb_min), Point3(*chunk.aabb_max))
    geom = Geom(vdata)
    geom.add_primitive(shared_triangles())
    # On the Geom too, in case the cull tests Geoms individually.
    geom.set_bounds(bounds)
    node = GeomNode(f"chunk{chunk.key}")
    node.add_geom(geom)
    node.set_bounds(bounds)
    node.set_final(True)
    node.set_into_collide_mask(BitMask32.all_off())
    return node
