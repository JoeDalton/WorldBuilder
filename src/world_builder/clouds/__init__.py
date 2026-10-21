"""
In-scene billboard clouds.

    noise.py   — what a cloud type looks like: its density field (DensityField)
                 and the tileable 3D value-noise volume it is built from.
    cloud.py   — where a type's billboards go (sample_field_particles), presets
                 (PRESETS, CloudSpec) and quality scaling (CloudQuality).
    field.py   — the drawable flat field (CloudField: geometry, sorting, wind,
                 live set_sun / set_optics / set_coverage) and the game wrapper
                 (Clouds).
    sphere.py  — a whole planet's clouds (SphericalClouds: billboard shells
                 generated per cube-sphere tile in the worker pool, a deck
                 beyond); not re-exported here.
    weather.py — the planet's weather: a cube map of target coverage
                 (WeatherSpec) and the fbm shift that gives it.

A billboard is a sample of a volume, not a sprite, and a cloud type is one density
field over a slab of sky whose features are the clouds. Design notes:
docs/source/clouds.md (planet clouds: docs/source/atmosphere.md); profiling and an
open alternative: view_aligned_slicing.md.
"""

from world_builder.clouds.cloud import (
    PRESETS,
    QUALITY_SCALES,
    SHELL_OCTAVES,
    VOLUME_FRACTIONS,
    CloudQuality,
    CloudSettings,
    CloudSpec,
    CloudType,
    at_quality,
    scale_count,
    scale_march,
    with_settings,
)
from world_builder.clouds.field import CloudField, CloudLayer, Clouds
from world_builder.clouds.noise import (
    CloudOptics,
    DensityField,
    build_noise_texture,
    measure_coverage,
    resolve_field,
)

__all__ = [
    "Clouds",
    "CloudField",
    "CloudLayer",
    "CloudType",
    "CloudSpec",
    "CloudOptics",
    "CloudQuality",
    "CloudSettings",
    "DensityField",
    "PRESETS",
    "QUALITY_SCALES",
    "SHELL_OCTAVES",
    "VOLUME_FRACTIONS",
    "at_quality",
    "scale_count",
    "scale_march",
    "with_settings",
    "build_noise_texture",
    "measure_coverage",
    "resolve_field",
]
