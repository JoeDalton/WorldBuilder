"""
A rock prototype's meshes, built in worker processes from its shape
(:mod:`world_builder.asteroids.shape`).

- **Meshes**, one per level of detail: a cube-sphere grid of ``quads`` x
  ``quads`` per face (the terrain's face frames and equal-angle grid), each
  quad cut along its (i, j)-(i+1, j+1) diagonal, counter-clockwise from
  outside, as the terrain's chunks. A level samples the shape band-limited to
  four times its spacing, so a coarse mesh is a smoothed rock. The faces
  share no vertices: seam vertices are computed twice from bit-identical
  directions.
- **Normals** by central differences of the shape about each vertex, in a
  tangent frame that depends on the direction only: the same on both sides
  of a seam.
- **Errors**, in units of the bounding radius: each mesh's largest radial
  distance from the finest mesh (how far a level of detail can be seen
  without showing it) and from the full-detail shape (the normal map's: how
  close the finest can be seen).
- **Radius grid**: the finest mesh's vertex radii, for collision: the very
  triangles drawn at that level.
- **Normal map**: the full-detail shape's normals in the body frame, per
  texel of each face's equal-angle grid (6 x size x size, RGB8): drawn at
  every level, it lends the coarse meshes the fine detail.

Vertices are in the body frame, in units of the rock's bounding radius.
Pure numpy (and the native noise), for worker processes.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass

import numpy as np

from world_builder.asteroids.shape import face_grid_dirs, prototype_shape
from world_builder.asteroids.spec import ShapeSpec

#: Quads per face side of each level of detail, coarse to fine.
LODS = (4, 8, 16, 32)
#: Texels per face side of the normal map.
NORMAL_SIZE = 128
#: Band limit of a mesh, in its spacings (as the terrain's chunks).
BAND_SPACINGS = 4.0
#: Band limit of the normal map, in texels.
NORMAL_BAND_TEXELS = 2.0


@dataclass(frozen=True)
class PrototypeJob:
    """What a field's prototypes are built from: the workers' spec.

    :ivar shape: the field's shapes
    :ivar seed: the field's seed
    :ivar lods: quads per face side of each level of detail, coarse to fine
    :ivar normal_size: texels per face side of the normal map
    """

    shape: ShapeSpec
    seed: int
    lods: tuple[int, ...] = LODS
    normal_size: int = NORMAL_SIZE


@dataclass
class PrototypeMesh:
    """One level of detail.

    :ivar quads: quads per face side
    :ivar vertices: (n, 3) float32 positions, body frame, bounding radius 1
    :ivar normals: (n, 3) float32 unit normals
    :ivar face_uv: (n, 3) float32: the face parameters a and b (each in
        [-1, 1], the equal-angle grid's) and the face (0-5)
    :ivar error: largest radial distance from the full-detail shape (what
        only the normal map carries, at the finest level)
    :ivar error_to_finest: largest radial distance from the finest mesh: what
        drawing this level instead of the finest changes
    """

    quads: int
    vertices: np.ndarray
    normals: np.ndarray
    face_uv: np.ndarray
    error: float
    error_to_finest: float = 0.0

    @property
    def indices(self) -> np.ndarray:
        return mesh_indices(self.quads)


@dataclass
class Prototype:
    """A prototype, as the workers return it.

    :ivar key: its index in the field
    :ivar version: the executor's channel version
    :ivar axes: its ellipsoid's semi-axes (x, y, z), z the longest
    :ivar meshes: its levels of detail, coarse to fine
    :ivar radius_grid: (6, n, n) float32, the finest mesh's vertex radii
        [face, row (v), column (u)]
    :ivar normal_map: (6, size, size, 3) uint8 body-frame normals,
        (n + 1) / 2 * 255 [face, row (v), column (u)]
    :ivar gen_ms: generation time
    """

    key: int
    version: int
    axes: np.ndarray
    meshes: list[PrototypeMesh]
    radius_grid: np.ndarray
    normal_map: np.ndarray
    gen_ms: float


@functools.cache
def mesh_indices(quads: int) -> np.ndarray:
    """The triangles every prototype's mesh of this level shares.

    :returns: uint16 (uint32 past 65,536 vertices) indices, counter-clockwise
        seen from outside
    """
    n = quads + 1
    r, c = np.meshgrid(np.arange(quads), np.arange(quads), indexing="ij")
    v00 = (r * n + c).ravel()
    v01 = v00 + 1  # +u
    v10 = v00 + n  # +v
    v11 = v10 + 1
    face = np.stack([v00, v01, v11, v00, v11, v10], axis=1).ravel()
    all_faces = np.concatenate([face + f * n * n for f in range(6)])
    dtype = np.uint16 if 6 * n * n <= 65536 else np.uint32
    indices = all_faces.astype(dtype)
    indices.flags.writeable = False
    return indices


def grid_spacing(quads: int) -> float:
    """:returns: the nominal spacing of a face grid on the unit sphere"""
    return 0.5 * math.pi / quads


def interpolate_grid(
    grid: np.ndarray, face: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """A face grid's values at face parameters, on its triangles (the
    meshes' diagonal): the radius a mesh has there.

    :param grid: (6, n, n) values at the nodes [face, row (v), column (u)]
    :param face: (m,) faces
    :param a: (m,) u parameters in [-1, 1]
    :param b: (m,) v parameters in [-1, 1]
    """
    quads = grid.shape[1] - 1
    x = (np.asarray(a) + 1.0) * 0.5 * quads
    y = (np.asarray(b) + 1.0) * 0.5 * quads
    i = np.clip(np.floor(x).astype(np.int64), 0, quads - 1)
    j = np.clip(np.floor(y).astype(np.int64), 0, quads - 1)
    fu, fv = x - i, y - j
    r00 = grid[face, j, i]
    r01 = grid[face, j, i + 1]
    r10 = grid[face, j + 1, i]
    r11 = grid[face, j + 1, i + 1]
    lower = fu >= fv  # the (v00, v01, v11) triangle
    return np.where(
        lower,
        r00 + fu * (r01 - r00) + fv * (r11 - r01),
        r00 + fv * (r10 - r00) + fu * (r11 - r10),
    )


def _vertex_normals(shape, dirs: np.ndarray, eps: float, band: float) -> np.ndarray:
    """Normals of the radius surface at *dirs*, by central differences in a
    tangent frame of the direction alone."""
    helper = np.where(
        (np.abs(dirs[:, 0]) < 0.9)[:, None], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    )
    t1 = np.cross(dirs, helper)
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(dirs, t1)

    def point(offset: np.ndarray) -> np.ndarray:
        d = dirs + offset
        d /= np.linalg.norm(d, axis=1)[:, None]
        return d * shape.radius(d, band)[:, None]

    du = point(eps * t1) - point(-eps * t1)
    dv = point(eps * t2) - point(-eps * t2)
    normals = np.cross(du, dv)
    flip = np.sum(normals * dirs, axis=1) < 0.0
    normals[flip] = -normals[flip]
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def generate_prototype(job: PrototypeJob, key: int, version: int = 0) -> Prototype:
    """Build prototype *key* of a field. The worker-process entry point.

    :param job: the field's shapes and the meshes wanted
    :param key: the prototype's index
    :param version: the channel's version, carried through
    """
    start = time.perf_counter()
    shape = prototype_shape(job.shape, job.seed, key)
    # The full detail: the normal map's samples, with a border for the
    # normals' differences.
    size = job.normal_size
    texel = grid_spacing(size)
    full_band = NORMAL_BAND_TEXELS * texel
    texel_dirs = face_grid_dirs(size, centred=True, border=1)
    full = shape.radius(texel_dirs.reshape(-1, 3), full_band).reshape(
        texel_dirs.shape[:3]
    )
    points = texel_dirs * full[..., None]
    du = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    dv = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    normal = np.cross(du, dv)
    normal /= np.linalg.norm(normal, axis=-1)[..., None]
    normal_map = np.rint((normal * 0.5 + 0.5) * 255.0).astype(np.uint8)
    # Where the error is measured: the texel centres.
    centres = -1.0 + (2.0 * np.arange(size) + 1.0) / size
    ta, tb = np.meshgrid(centres, centres)
    t_face = np.repeat(np.arange(6), size * size)
    t_a = np.tile(ta.ravel(), 6)
    t_b = np.tile(tb.ravel(), 6)
    truth = full[:, 1:-1, 1:-1].ravel()

    meshes = []
    radius_grid = None
    for quads in job.lods:
        spacing = grid_spacing(quads)
        band = BAND_SPACINGS * spacing
        dirs = face_grid_dirs(quads)
        n = quads + 1
        flat = dirs.reshape(-1, 3)
        radius = shape.radius(flat, band)
        grid = radius.reshape(6, n, n)
        error = float(np.max(np.abs(interpolate_grid(grid, t_face, t_a, t_b) - truth)))
        params = -1.0 + 2.0 * np.arange(n) / quads
        a, b = np.meshgrid(params, params)
        face_uv = np.column_stack(
            (
                np.tile(a.ravel(), 6),
                np.tile(b.ravel(), 6),
                np.repeat(np.arange(6), n * n),
            )
        )
        meshes.append(
            PrototypeMesh(
                quads=quads,
                vertices=(flat * radius[:, None]).astype(np.float32),
                normals=_vertex_normals(shape, flat, 0.5 * spacing, band).astype(
                    np.float32
                ),
                face_uv=face_uv.astype(np.float32),
                error=error,
            )
        )
        radius_grid = grid.astype(np.float32)
    finest = meshes[-1]
    finest_params = finest.face_uv
    finest_radius = radius_grid.ravel()
    for mesh in meshes[:-1]:
        n = mesh.quads + 1
        grid = np.linalg.norm(mesh.vertices.astype(np.float64), axis=1).reshape(6, n, n)
        coarse = interpolate_grid(
            grid,
            finest_params[:, 2].astype(np.int64),
            finest_params[:, 0].astype(np.float64),
            finest_params[:, 1].astype(np.float64),
        )
        mesh.error_to_finest = float(np.max(np.abs(coarse - finest_radius)))
    return Prototype(
        key=key,
        version=version,
        axes=shape.axes.copy(),
        meshes=meshes,
        radius_grid=radius_grid,
        normal_map=normal_map,
        gen_ms=(time.perf_counter() - start) * 1e3,
    )
