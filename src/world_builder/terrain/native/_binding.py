"""
ctypes binding of the native terrain collision library.

The only module that knows how the library is called: a CPython extension can
replace it behind the same :class:`NativeTerrain` class. The library is built
per Panda3D series (``scripts/build_native.sh``) as
``libterrain_collision-panda<major>.<minor>.so`` next to this file.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from world_builder.terrain.chunkgen import VERTEX_DTYPE, ChunkData

HERE = Path(__file__).parent
#: Floats per vertex record; the position is the first three.
_STRIDE = VERTEX_DTYPE.itemsize // 4
# Panda's Python wrapper (Dtool_PyInstDef, py_panda.h) on 64-bit CPython:
# PyObject_HEAD (16 bytes), the type pointer, then the C++ pointer, then a
# signature. Only this module depends on it.
_POINTER_OFFSET = 24
_SIGNATURE_OFFSET = 32
_SIGNATURE = 0xBEAF


class NativeCollisionError(RuntimeError):
    """The native terrain collision library is missing or does not match the
    running Panda3D."""


def panda_series() -> tuple[int, int]:
    """:returns: (major, minor) of the running Panda3D"""
    import panda3d

    major, minor = panda3d.__version__.split(".")[:2]
    return int(major), int(minor)


def library_path(series: tuple[int, int] | None = None) -> Path:
    """:returns: where the library for a Panda3D series (default: the
    running one) is expected"""
    major, minor = series or panda_series()
    return HERE / f"libterrain_collision-panda{major}.{minor}.so"


@functools.cache
def _library() -> ctypes.CDLL:
    import panda3d
    import panda3d.core  # noqa: F401 (libpanda must be loaded first)

    series = panda_series()
    path = library_path(series)
    if not path.exists():
        built = sorted(p.name for p in HERE.glob("libterrain_collision-panda*.so"))
        raise NativeCollisionError(
            f"terrain collision is not built for Panda3D {panda3d.__version__}: "
            f"{path.name} is missing (found: {', '.join(built) or 'none'}). Build it "
            "with scripts/build_native.sh, see NATIVE_COLLISION_BUILD.md."
        )
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as error:
        raise NativeCollisionError(
            f"cannot load {path.name} with Panda3D {panda3d.__version__}: {error}. "
            "Rebuild it with scripts/build_native.sh."
        ) from error
    built = lib.wb_panda_version()
    if built != series[0] * 100 + series[1]:
        raise NativeCollisionError(
            f"{path.name} was built against Panda3D {built // 100}.{built % 100}, "
            f"but {panda3d.__version__} is running. Rebuild it with "
            "scripts/build_native.sh."
        )

    P, N, D, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_float
    L = ctypes.c_longlong
    signatures = {
        "wb_terrain_new": (P, [D, N, L, N]),
        "wb_terrain_free": (None, [P]),
        "wb_terrain_set_origin": (None, [P, D, D, D]),
        "wb_terrain_add_chunk": (None, [P, N, N, L, L, P, P, N, P, F, F, P]),
        "wb_terrain_remove_chunk": (None, [P, N, N, L, L]),
        "wb_terrain_clear": (None, [P]),
        "wb_terrain_num_chunks": (N, [P]),
        "wb_terrain_attach": (N, [P, P, N]),
        "wb_terrain_segment": (N, [P, P, P, P]),
        "wb_terrain_sphere": (N, [P, P, D, N, P]),
        "wb_terrain_material": (N, [P, P]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    lib.wb_init()
    return lib


def _collision_node_pointer(node) -> int:
    """The C++ CollisionNode behind a Python ``panda3d.core.CollisionNode``."""
    from panda3d.core import CollisionNode

    if not isinstance(node, CollisionNode):
        raise TypeError(f"expected a CollisionNode, got {type(node).__name__}")
    address = id(node)
    if ctypes.c_ushort.from_address(address + _SIGNATURE_OFFSET).value != _SIGNATURE:
        raise TypeError("not a Panda3D wrapper object")
    return ctypes.c_void_p.from_address(address + _POINTER_OFFSET).value


def _vec(point: ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(point, dtype=np.float64).reshape(3)


class NativeTerrain:
    """One planet's drawn chunks, as the native collision solid sees them.

    Positions are float64 metres in the planet frame. The solids attached to
    collision nodes (:meth:`attach`) share these chunks; they work in the
    frame of the planet shifted by :meth:`set_origin`.
    """

    def __init__(self, radius: float, max_level: int, n0: int, ocean: bool):
        """
        :param radius: sea-level radius, metres
        :param max_level: the chunk grid's root level (``CubeGrid.max_level``)
        :param n0: level-0 steps per face edge (``CubeGrid.n0``)
        :param ocean: the sea surface is solid (``TerrainQuery.include_ocean``)
        :raises NativeCollisionError: the library is missing or mismatched
        """
        self._handle = None
        self._lib = _library()
        self._handle = self._lib.wb_terrain_new(radius, max_level, n0, int(ocean))
        self._out = np.zeros(7)

    def close(self) -> None:
        """Release the chunks (attached solids keep them until removed)."""
        if self._handle:
            self._lib.wb_terrain_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self._lib.wb_terrain_num_chunks(self._handle)

    def set_origin(self, origin: ArrayLike) -> None:
        """The planet-frame point at the collision node's origin."""
        x, y, z = (float(v) for v in origin)
        self._lib.wb_terrain_set_origin(self._handle, x, y, z)

    def add_chunk(self, chunk: ChunkData) -> None:
        """Register a drawn chunk (copied; replaces one with the same key)."""
        level, face, i, j = chunk.key
        vertices = np.frombuffer(chunk.vertex_bytes, np.float32)
        centre = _vec(chunk.centre)
        elevation = np.ascontiguousarray(chunk.elevation, np.float32)
        weights = np.ascontiguousarray(chunk.weights, np.uint8)
        self._lib.wb_terrain_add_chunk(
            self._handle,
            level,
            face,
            i,
            j,
            centre.ctypes.data,
            vertices.ctypes.data,
            _STRIDE,
            elevation.ctypes.data,
            float(chunk.h_min),
            float(chunk.h_max),
            weights.ctypes.data,
        )

    def remove_chunk(self, key: tuple) -> None:
        level, face, i, j = key
        self._lib.wb_terrain_remove_chunk(self._handle, level, face, i, j)

    def clear(self) -> None:
        self._lib.wb_terrain_clear(self._handle)

    def attach(self, node, samples: int = 5) -> None:
        """Add a solid answering for these chunks to a CollisionNode.

        :param node: a ``panda3d.core.CollisionNode``
        :param samples: sphere footprint samples per side (odd)
        """
        self._lib.wb_terrain_attach(
            self._handle, _collision_node_pointer(node), samples
        )

    def segment(self, a: ArrayLike, b: ArrayLike):
        """:returns: (t, point (3,), normal (3,)) of the first crossing into
        the surface, planet frame, or None"""
        a, b = _vec(a), _vec(b)
        out = self._out
        if not self._lib.wb_terrain_segment(
            self._handle, a.ctypes.data, b.ctypes.data, out.ctypes.data
        ):
            return None
        return float(out[0]), out[1:4].copy(), out[4:7].copy()

    def sphere(self, centre: ArrayLike, radius: float, samples: int = 5):
        """:returns: (point (3,), normal (3,)) where a sphere touches the
        surface, planet frame, or None"""
        c = _vec(centre)
        out = self._out
        if not self._lib.wb_terrain_sphere(
            self._handle, c.ctypes.data, float(radius), samples, out.ctypes.data
        ):
            return None
        return out[0:3].copy(), out[3:6].copy()

    def material(self, point: ArrayLike) -> int:
        """:returns: the dominant biome index at the vertex nearest a surface
        point, -1 over the sea, -2 where nothing is registered"""
        return self._lib.wb_terrain_material(self._handle, _vec(point).ctypes.data)
