"""
A planet's clouds.

    planet.py  — a whole planet's clouds (PlanetClouds), as one part; not
                 re-exported here, nor is the volume:
    march.py   — the volume (CloudMarch: the low clouds raymarched and the
                 high layer's cirrus, a deck sphere for the sea's
                 reflection: deck.py; what the field is built from:
                 field_setup.py; the genera: genera.py).
    noise.py   — what a cloud type looks like: its density field (DensityField)
                 and the tileable 3D value-noise volume it is built from.
    cloud.py   — the cloud types (PRESETS, CloudSpec) and what they cost
                 (CloudSettings).
    weather.py — the planet's weather: a cube map of target coverage
                 (WeatherSpec) and the fbm shift that gives it; the upper
                 wind and the cirrus's cover.

A cloud type is one density field over a slab of sky whose features are the
clouds. Design notes:
docs/source/clouds.md and docs/source/atmosphere.md.
"""

from world_builder.clouds.cloud import (
    PRESETS,
    CloudSettings,
    CloudSpec,
    CloudType,
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
    "scale_march",
    "with_settings",
    "build_noise_texture",
]
