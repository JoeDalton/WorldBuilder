"""
A camera-locked reflective ocean, curved with the planet.

    ocean.py  the surface mesh (a flat quad, or a dense swell grid ringed by
              geometric border cells), its planar reflection, and the per-frame
              re-centring under the camera

Shaders: ``shaders/ocean.vert`` and ``shaders/ocean.frag``. Known open issue
with the geometric swell: ``OCEAN_SWELL_ARTIFACT_REPORT.md``.
"""

from world_builder.ocean.ocean import (
    Ocean,
    border_coords,
    compute_wave_dirs,
    make_plane_mesh,
    make_swell_grid_mesh,
)

__all__ = [
    "Ocean",
    "border_coords",
    "compute_wave_dirs",
    "make_plane_mesh",
    "make_swell_grid_mesh",
]
