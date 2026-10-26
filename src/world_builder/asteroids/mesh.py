"""
Rock prototypes to Panda: their meshes as Geoms, their normal maps as one
texture array.

A mesh's vertices are its positions, normals and face parameters
(:class:`~world_builder.asteroids.prototype.PrototypeMesh`), interleaved in
:data:`VERTEX_DTYPE` and uploaded in one copy. Every prototype's mesh of one
level of detail shares that level's index buffer.

The normal maps go in one 2D texture array, six layers per prototype (layer
``6 * index + face``), sampled at ``((a + 1) / 2, (b + 1) / 2)`` for a
direction's face parameters (a, b): each layer's texels sit at the centres of
the face's equal-angle grid.
"""

from __future__ import annotations

import functools

import numpy as np
from panda3d.core import (
    Geom,
    GeomEnums,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    InternalName,
    SamplerState,
    Texture,
)

from world_builder.asteroids.prototype import Prototype, PrototypeMesh, mesh_indices

#: Interleaved vertex record of a prototype mesh.
VERTEX_DTYPE = np.dtype(
    [
        ("vertex", np.float32, 3),
        ("normal", np.float32, 3),
        ("face_uv", np.float32, 3),
    ]
)


@functools.cache
def vertex_format() -> GeomVertexFormat:
    """:returns: the registered format matching :data:`VERTEX_DTYPE`"""
    array = GeomVertexArrayFormat()
    array.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    array.add_column(InternalName.get_normal(), 3, Geom.NT_float32, Geom.C_normal)
    array.add_column(InternalName.make("face_uv"), 3, Geom.NT_float32, Geom.C_other)
    if array.get_stride() != VERTEX_DTYPE.itemsize:
        raise RuntimeError(f"stride {array.get_stride()} != {VERTEX_DTYPE.itemsize}")
    return GeomVertexFormat.register_format(GeomVertexFormat(array))


@functools.cache
def shared_triangles(quads: int) -> GeomTriangles:
    """:returns: the index buffer every prototype's mesh of this level uses"""
    indices = mesh_indices(quads)
    tris = GeomTriangles(Geom.UH_static)
    wide = indices.dtype == np.uint32
    tris.set_index_type(GeomEnums.NT_uint32 if wide else GeomEnums.NT_uint16)
    # Sized directly: add_next_vertices(n) would add the indices 0 to n - 1
    # and widen the buffer to 32 bits past 65,535 of them.
    array = tris.modify_vertices()
    array.unclean_set_num_rows(indices.size)
    memoryview(array).cast("B")[:] = indices.tobytes()
    return tris


def build_geom(mesh: PrototypeMesh, name: str = "rock") -> Geom:
    """:returns: one level of detail of a prototype, as a Geom"""
    records = np.empty(len(mesh.vertices), dtype=VERTEX_DTYPE)
    records["vertex"] = mesh.vertices
    records["normal"] = mesh.normals
    records["face_uv"] = mesh.face_uv
    data = GeomVertexData(name, vertex_format(), Geom.UH_static)
    data.unclean_set_num_rows(len(records))
    memoryview(data.modify_array(0)).cast("B")[:] = records.tobytes()
    geom = Geom(data)
    geom.add_primitive(shared_triangles(mesh.quads))
    return geom


def normal_map_texture(
    prototypes: list[Prototype], name: str = "rock_normals"
) -> Texture:
    """:returns: the prototypes' normal maps as one mipmapped 2D texture
    array, six layers each, in the list's order"""
    maps = np.concatenate([p.normal_map for p in prototypes])  # (6 k, s, s, 3)
    count, size = maps.shape[0], maps.shape[1]
    tex = Texture(name)
    tex.setup_2d_texture_array(
        size, size, count, Texture.T_unsigned_byte, Texture.F_rgb8
    )
    # Row 0 of a layer is the face's v = -1 edge: already Panda's bottom row.
    # Panda keeps RGB RAM images in BGR order.
    tex.set_ram_image(np.ascontiguousarray(maps[..., ::-1]).tobytes())
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_compression(Texture.CM_off)
    return tex
