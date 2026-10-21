"""
A camera-locked reflective ocean, curved with the planet.

    ocean.py   the surface mesh (a flat quad, or a dense swell grid ringed by
               geometric border cells), its planar reflection, the per-frame
               re-centring under the camera, and its quality presets
    seabed.py  the sea floor under it, baked around the camera from an app's
               sampler: colour by depth, shoreline foam, swell over shallows

Shaders: ``shaders/ocean.vert`` and ``shaders/ocean.frag``, with
``seabed.glsl`` and ``water.glsl`` (the colour by depth, shared with the
terrain's sea surface). Known open issue with the geometric swell:
``OCEAN_SWELL_ARTIFACT_REPORT.md``.
"""

from world_builder.ocean.ocean import (
    OCEAN_QUALITY,
    Ocean,
    OceanQuality,
    aces_tonemap,
    border_coords,
    compute_wave_dirs,
    make_plane_mesh,
    make_swell_grid_mesh,
)
from world_builder.ocean.seabed import Cascade, SeabedMap

__all__ = [
    "Cascade",
    "OCEAN_QUALITY",
    "Ocean",
    "OceanQuality",
    "SeabedMap",
    "aces_tonemap",
    "border_coords",
    "compute_wave_dirs",
    "make_plane_mesh",
    "make_swell_grid_mesh",
]
