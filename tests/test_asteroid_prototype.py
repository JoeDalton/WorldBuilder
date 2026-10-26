"""Rock prototypes (asteroids.shape, asteroids.prototype, asteroids.mesh):
closed outward meshes inside the bounding radius, errors, the collision grid
and the normal map, and the worker job."""

import pickle
import time

import numpy as np
import pytest

from world_builder.asteroids.prototype import (
    PrototypeJob,
    generate_prototype,
    interpolate_grid,
    mesh_indices,
)
from world_builder.asteroids.shape import face_grid_dirs, prototype_shape
from world_builder.asteroids.spec import ShapeSpec
from world_builder.terrain.workers import ChunkExecutor

JOB = PrototypeJob(ShapeSpec(), seed=3, lods=(4, 8, 16), normal_size=48)


@pytest.fixture(scope="module")
def prototypes():
    return [generate_prototype(JOB, k) for k in range(3)]


def test_meshes_are_closed_and_face_outward(prototypes):
    for prototype in prototypes:
        for mesh in prototype.meshes:
            # weld the seams' duplicated vertices (bit-identical directions)
            keys = np.round(mesh.vertices.astype(np.float64), 6)
            _, weld = np.unique(keys, axis=0, return_inverse=True)
            tris = weld.ravel()[mesh.indices.astype(np.int64)].reshape(-1, 3)
            directed = np.concatenate(
                [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]
            )
            # each directed edge once, and its reverse once: closed, consistent
            assert len(np.unique(directed, axis=0)) == len(directed)
            reverse = {tuple(e) for e in directed[:, ::-1].tolist()}
            assert reverse == {tuple(e) for e in directed.tolist()}
            v = mesh.vertices.astype(np.float64)[mesh.indices.astype(np.int64)].reshape(
                -1, 3, 3
            )
            normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            assert np.all(np.sum(normal * v.mean(axis=1), axis=1) > 0.0)


def test_rocks_stay_inside_their_bounding_radius(prototypes):
    for prototype in prototypes:
        for mesh in prototype.meshes:
            radius = np.linalg.norm(mesh.vertices, axis=1)
            assert radius.max() <= 1.0 and radius.min() > 0.2
            np.testing.assert_allclose(
                np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5
            )
            # vertex normals face out of the rock
            assert np.all(np.sum(mesh.normals * mesh.vertices, axis=1) > 0.0)
        # the longest axis is z
        assert prototype.axes[2] == 1.0 and np.all(prototype.axes <= 1.0)


def test_errors_shrink_to_the_finest(prototypes):
    for prototype in prototypes:
        to_finest = [m.error_to_finest for m in prototype.meshes]
        assert to_finest[-1] == 0.0
        assert to_finest[0] > to_finest[-2] > 0.0
        # the finest is the closest to the full detail
        full = [m.error for m in prototype.meshes]
        assert full[-1] == min(full) and full[-1] > 0.0


def test_the_radius_grid_is_the_finest_mesh(prototypes):
    for prototype in prototypes:
        finest = prototype.meshes[-1]
        n = finest.quads + 1
        assert prototype.radius_grid.shape == (6, n, n)
        np.testing.assert_allclose(
            prototype.radius_grid.ravel(),
            np.linalg.norm(finest.vertices, axis=1),
            rtol=1e-6,
        )
        # interpolated at the grid's nodes, the grid itself
        uv = finest.face_uv.astype(np.float64)
        at_nodes = interpolate_grid(
            prototype.radius_grid.astype(np.float64),
            uv[:, 2].astype(int),
            uv[:, 0],
            uv[:, 1],
        )
        np.testing.assert_allclose(at_nodes, prototype.radius_grid.ravel(), rtol=1e-12)


def test_the_normal_map_faces_out(prototypes):
    size = JOB.normal_size
    dirs = face_grid_dirs(size, centred=True)
    for prototype in prototypes:
        assert prototype.normal_map.shape == (6, size, size, 3)
        normal = prototype.normal_map.astype(np.float64) / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1.0, atol=0.02)
        assert np.mean(np.sum(normal * dirs, axis=-1) > 0.0) > 0.99


def test_prototypes_are_deterministic_and_distinct():
    a = generate_prototype(JOB, 1)
    b = generate_prototype(JOB, 1)
    np.testing.assert_array_equal(a.meshes[-1].vertices, b.meshes[-1].vertices)
    np.testing.assert_array_equal(a.normal_map, b.normal_map)
    c = generate_prototype(JOB, 2)
    assert not np.array_equal(a.meshes[-1].vertices, c.meshes[-1].vertices)
    # the shape is cached per process, keyed by its spec
    assert prototype_shape(JOB.shape, JOB.seed, 1) is prototype_shape(
        JOB.shape, JOB.seed, 1
    )


def test_seams_share_their_vertices():
    """Both faces of a seam compute the same directions, so the same
    positions: the mesh has no cracks."""
    mesh = generate_prototype(JOB, 0).meshes[-1]
    keys = mesh.vertices.view(np.dtype((np.void, 12))).ravel()
    n = mesh.quads + 1
    # 6 n^2 vertices, of which the 12 edges' (n - 2) inner ones are doubled
    # and the 8 corners tripled
    assert len(np.unique(keys)) == 6 * n * n - 12 * (n - 2) - 16


def test_mesh_indices():
    indices = mesh_indices(4)
    assert indices.dtype == np.uint16 and len(indices) == 6 * 4 * 4 * 6
    assert mesh_indices(128).dtype == np.uint32


@pytest.mark.parametrize("backend", ["inline", "thread", "process"])
def test_the_workers_build_prototypes(backend):
    job = PrototypeJob(ShapeSpec(prototypes=2), seed=0, lods=(4, 8), normal_size=16)
    assert pickle.loads(pickle.dumps(job)) == job
    executor = ChunkExecutor(backend=backend, workers=2, inline_budget=4)
    try:
        version = executor.set_spec(job, "asteroids", generator=generate_prototype)
        executor.request([0, 1], "asteroids")
        done = []
        for _ in range(600):
            done += executor.collect("asteroids")
            if len(done) == 2:
                break
            time.sleep(0.05)
        assert sorted(p.key for p in done) == [0, 1]
        assert all(p.version == version for p in done)
    finally:
        executor.shutdown()


def test_panda_geometry():
    from world_builder.asteroids.mesh import build_geom, normal_map_texture

    prototype = generate_prototype(JOB, 0)
    mesh = prototype.meshes[1]
    geom = build_geom(mesh)
    assert geom.get_vertex_data().get_num_rows() == len(mesh.vertices)
    assert geom.get_primitive(0).get_num_vertices() == len(mesh.indices)
    tex = normal_map_texture([prototype, prototype])
    assert tex.get_z_size() == 12 and tex.get_x_size() == JOB.normal_size


@pytest.mark.parametrize("quads", [64, 128])
def test_fine_levels_keep_their_index_buffer(quads):
    """64 quads a side: 147,456 indices of 25,350 vertices, still 16-bit;
    128: 32-bit."""
    from world_builder.asteroids.mesh import shared_triangles

    tris = shared_triangles(quads)
    indices = mesh_indices(quads)
    assert tris.get_num_vertices() == len(indices)
    assert tris.get_vertex(len(indices) - 1) == int(indices[-1])
    wide = 6 * (quads + 1) ** 2 > 65536
    assert tris.get_index_stride() == (4 if wide else 2)
