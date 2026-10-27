"""
What an asteroid field costs (:mod:`world_builder.quality`): how small a rock
is still drawn, how fine its mesh, how many at most, and its textures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from world_builder.quality import REBUILD, RESTART, QualityLevel, QualitySettings, knob


@dataclass(frozen=True)
class AsteroidSettings(QualitySettings):
    """What an asteroid field costs. All live but the meshes' and the normal
    maps' resolution (rebuilt) and the texture size (at the next start).

    :ivar min_pixels: rocks smaller than this on screen (their diameter,
        pixels) are not drawn; they fade out over the last quarter
    :ivar lod_error_px: a coarser level of detail is drawn while its error
        stays below this many pixels
    :ivar max_lod: quads per face side of the finest level
    :ivar normal_size: texels per face side of the rocks' normal maps
    :ivar instance_budget: rocks drawn (in view) at most: past it, the
        smallest drawn size rises until it fits
    :ivar held_budget: rocks kept around the camera (in every direction, on
        the CPU) at most: past it, the windows keep only bigger rocks, the
        others fading into the haze
    :ivar detail_far: textures fade to the materials' mean colours by this
        distance, metres
    :ivar texture_size: the rock and ice textures' size, texels
    :ivar terrain_min_radius: rocks this big and more turn into terrain near
        the camera (the terrain's level of detail), metres
    :ivar max_terrain_rocks: rocks drawn as terrain at once, at most (0:
        none)
    """

    min_pixels: float = knob(
        2.0,
        "Smallest rock drawn",
        unit="px",
        range=(0.5, 8.0),
        help="Rocks smaller than this on screen are not drawn: fewer rocks, "
        "nearer horizon.",
    )
    lod_error_px: float = knob(
        1.0,
        "Rock detail",
        unit="px",
        range=(0.25, 6.0),
        help="How far a rock's silhouette may be from its finest mesh, on "
        "screen: lower is sharper, more triangles.",
    )
    max_lod: int = knob(
        32,
        "Finest rock mesh",
        REBUILD,
        choices={"16": 16, "32": 32, "64": 64},
        help="Quads per side of each of the six faces of the finest mesh.",
    )
    normal_size: int = knob(
        128,
        "Rock detail maps",
        REBUILD,
        unit="px",
        choices={"64": 64, "128": 128, "256": 256},
        help="The surface detail's resolution (video memory, load time).",
    )
    instance_budget: int = knob(
        80_000,
        "Rocks drawn",
        range=(1000, 200_000),
        help="At most this many: past it, smaller rocks are dropped first.",
    )
    held_budget: int = knob(
        300_000,
        "Rocks kept",
        range=(1000, 2_000_000),
        advanced=True,
        help="Rocks kept around the camera in every direction (CPU time): "
        "past it, far small rocks fade into the haze.",
    )
    detail_far: float = knob(
        3000.0,
        "Rock texture distance",
        unit="m",
        range=(500.0, 10000.0),
        help="Textures fade to flat colours by this distance.",
    )
    texture_size: int = knob(
        1024,
        "Rock texture size",
        RESTART,
        unit="px",
        choices={"256": 256, "512": 512, "1024": 1024, "2048": 2048},
        help="Rock and ice texture resolution (video memory, load time).",
    )

    terrain_min_radius: float = knob(
        200.0,
        "Detailed rocks from",
        unit="m",
        range=(50.0, 5000.0),
        advanced=True,
        help="Rocks this big and more get the terrain's detail up close.",
    )
    max_terrain_rocks: int = knob(
        8,
        "Detailed rocks",
        range=(0, 32),
        help="Rocks drawn with the terrain's detail at once (0: none).",
    )

    LABEL: ClassVar = "Asteroids"
    PRESETS: ClassVar = {
        QualityLevel.LOW: {
            "min_pixels": 4.0,
            "lod_error_px": 3.0,
            "max_lod": 16,
            "normal_size": 64,
            "instance_budget": 15_000,
            "held_budget": 80_000,
            "detail_far": 1000.0,
            "texture_size": 256,
            "max_terrain_rocks": 0,
        },
        QualityLevel.MID: {
            "min_pixels": 3.0,
            "lod_error_px": 2.0,
            "instance_budget": 40_000,
            "held_budget": 150_000,
            "detail_far": 2000.0,
            "texture_size": 512,
            "max_terrain_rocks": 4,
        },
        QualityLevel.ULTRA: {
            "min_pixels": 1.5,
            "lod_error_px": 0.5,
            "max_lod": 64,
            "normal_size": 256,
            "instance_budget": 150_000,
            "held_budget": 600_000,
            "detail_far": 5000.0,
            "texture_size": 2048,
            "max_terrain_rocks": 16,
        },
    }
