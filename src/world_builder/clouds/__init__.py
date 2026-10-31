"""
A planet's clouds.

    planet.py  — a whole planet's clouds (PlanetClouds): the low clouds'
                 volume and the cirrus sheet above it, as one part; not
                 re-exported here, nor are the two layers:
    sphere.py  — the volume (CloudVolume: billboard shells generated per
                 cube-sphere tile in the worker pool, a deck beyond);
    cirrus.py  — the sheet (CloudSheet: high cirrus, baked patches stamped
                 along the upper wind, drawn behind and in front of the
                 volume).
    noise.py   — what a cloud type looks like: its density field (DensityField)
                 and the tileable 3D value-noise volume it is built from.
    cloud.py   — the cloud types (PRESETS, CloudSpec), what they cost
                 (CloudSettings) and the billboards' sprite atlas.
    weather.py — the planet's weather: a cube map of target coverage
                 (WeatherSpec) and the fbm shift that gives it; the upper
                 wind and the cirrus's cover.

A billboard is a sample of a volume, not a sprite, and a cloud type is one density
field over a slab of sky whose features are the clouds. Design notes:
docs/source/clouds.md and docs/source/atmosphere.md.
"""

from world_builder.clouds.cloud import (
    PRESETS,
    CloudSettings,
    CloudSpec,
    CloudType,
    scale_count,
    scale_march,
    with_settings,
)
from world_builder.clouds.noise import (
    CloudOptics,
    DensityField,
    build_noise_texture,
)

__all__ = [
    "CloudType",
    "CloudSpec",
    "CloudOptics",
    "CloudSettings",
    "DensityField",
    "PRESETS",
    "scale_count",
    "scale_march",
    "with_settings",
    "build_noise_texture",
]
