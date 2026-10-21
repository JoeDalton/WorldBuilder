"""
Unit tests for the cube-sphere chunk grid (world_builder.terrain.cubesphere).
Pure NumPy; no window.
"""

from collections import Counter

import numpy as np
import pytest

from world_builder.terrain.cubesphere import (
    FACE_FRAMES,
    QUADS,
    UV_PERIOD,
    CubeGrid,
    dirs_to_face,
    face_dirs,
    grid_for,
)

RADII = (200e3, 600e3, 6371e3)


def test_face_frames_are_orthonormal_and_right_handed():
    for n, u, v in FACE_FRAMES:
        np.testing.assert_allclose(np.cross(u, v), n)
        assert u @ v == 0.0 and u @ n == 0.0 and v @ n == 0.0


def test_directions_round_trip_through_faces():
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    face, a, b = dirs_to_face(dirs)
    for f in range(6):
        sel = face == f
        np.testing.assert_allclose(face_dirs(f, a[sel], b[sel]), dirs[sel], atol=1e-14)


def test_face_points_are_unit_and_on_their_face():
    t = np.linspace(-1.0, 1.0, 9)
    a, b = np.meshgrid(t, t)
    for f in range(6):
        d = face_dirs(f, a, b)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-15)
        assert np.all(d @ FACE_FRAMES[f][0] > 0.5)


def test_seams_are_bit_identical():
    """Every edge sample is computed identically by the faces sharing it:
    12 edges x 63 inner samples, plus the 8 corners, each seen 2 or 3 times."""
    grid = grid_for(600e3)
    t = grid.param(np.arange(0, grid.n0 + 1, grid.n0 // 64))
    one = np.ones_like(t)
    points = np.concatenate(
        [
            face_dirs(f, a, b)
            for f in range(6)
            for a, b in ((one, t), (-one, t), (t, one), (t, -one))
        ]
    )
    counts = Counter(row.tobytes() for row in points)
    assert len(counts) == 12 * 63 + 8
    assert sorted(set(counts.values())) == [2, 6]  # edges; corners (3 faces x 2 edges)


@pytest.mark.parametrize("radius", RADII)
def test_grid_levels_follow_the_radius(radius):
    grid = grid_for(radius)
    assert grid.n0 == QUADS << grid.max_level
    assert 1.5 < grid.spacing0 < 2.5  # the finest level stays near 2 m
    assert grid.spacing(grid.max_level) * QUADS == pytest.approx(np.pi * radius / 2.0)
    # A face edge holds a whole number of texture periods.
    assert (grid.uv_unit * grid.n0) % UV_PERIOD == pytest.approx(0.0, abs=1e-6)


def test_max_level_for_known_radii():
    assert grid_for(600e3).max_level == 13
    assert grid_for(6371e3).max_level == 16


def test_tree_navigation():
    grid = grid_for(600e3)
    key = (5, 2, 17, 40)
    children = CubeGrid.children(key)
    assert len(set(children)) == 4
    assert all(CubeGrid.parent(c) == key for c in children)
    ka, kb = grid.grid_indices(key)
    for child in children:
        ca, cb = grid.grid_indices(child)
        assert ka[0] <= ca[0] and ca[-1] <= ka[-1]
        assert kb[0] <= cb[0] and cb[-1] <= kb[-1]
    assert grid.roots() == [(13, f, 0, 0) for f in range(6)]


@pytest.mark.parametrize("radius", RADII)
def test_key_boxes_contain_their_patches(radius):
    grid = grid_for(radius)
    rng = np.random.default_rng(1)
    for level in (0, 4, grid.max_level - 3, grid.max_level):
        for _ in range(12):
            key = (
                level,
                int(rng.integers(6)),
                int(rng.integers(grid.side(level))),
                int(rng.integers(grid.side(level))),
            )
            box = grid.box(key)
            ka, kb = grid.grid_indices(key)
            ga, gb = np.meshgrid(grid.param(ka[::4]), grid.param(kb[::4]))
            d = face_dirs(key[1], ga, gb)
            du, dv, dw = d @ box.t_u, d @ box.t_v, d @ box.up
            assert box.u_range[0] <= du.min() and du.max() <= box.u_range[1]
            assert box.v_range[0] <= dv.min() and dv.max() <= box.v_range[1]
            assert dw.min() >= box.cos_radius - 1e-12


def test_locate_finds_the_chunk_and_position():
    grid = grid_for(600e3)
    key = (3, 4, 100, 37)
    ka, kb = grid.grid_indices(key)
    # A point 10.25 quads along u and 30.5 along v, inside the chunk.
    a = grid.param(ka[0] + 10.25 * 8)
    b = grid.param(kb[0] + 30.5 * 8)
    d = face_dirs(4, np.array([a]), np.array([b]))
    face, i, j, fu, fv = grid.locate(d, 3)
    assert (face[0], i[0], j[0]) == key[1:]
    assert fu[0] == pytest.approx(10.25, abs=1e-6)
    assert fv[0] == pytest.approx(30.5, abs=1e-6)


def test_uv_origin_is_wrapped():
    grid = grid_for(600e3)
    u, v = grid.uv_origin((0, 0, grid.side(0) - 1, 5))
    assert 0.0 <= u < UV_PERIOD and 0.0 <= v < UV_PERIOD
