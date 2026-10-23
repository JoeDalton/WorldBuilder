"""
Native (C++) terrain collision for Panda3D's collision traverser.

``terrain_collision.cpp`` defines ``CollisionTerrain``, a collision solid that
answers sphere and segment tests against the drawn chunks' triangles. The
Linux (x86_64, aarch64), Windows x64 and macOS wheels ship it built; from a
source checkout,
``invoke native`` builds it (``scripts/build_native.sh``; on Windows
``scripts/build_native.ps1``; see ``NATIVE_COLLISION_BUILD.md``).
Use it through
:meth:`world_builder.terrain.terrain.Terrain.enable_collision`.
"""

from world_builder.terrain.native._binding import (
    NativeCollisionError,
    NativeTerrain,
    library_path,
)

__all__ = ["NativeCollisionError", "NativeTerrain", "library_path"]
