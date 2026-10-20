"""
Biome weights from climate, altitude and slope.

The climate fields decide the broad biome: cold gives snow, hot and dry gives
desert, wet and mild gives forest, and grass fills whatever is left. Because
temperature falls with altitude (:meth:`Planet.lapse`), mountain tops come out
snowy with no special rule. Two terrain rules then override the climate:

* slope: cliffs are rock, whatever the climate
* coast: a band just above sea level is beach (only on planets with an ocean)

Weights are smooth and normalised, not hard ids, so biomes blend.
"""

from __future__ import annotations

import numpy as np

from world_builder.terrain.noise import smoothstep

#: Fixed order; also the texture-array layer order.
BIOMES = ("snow", "rock", "grass", "forest", "desert", "beach")
SNOW, ROCK, GRASS, FOREST, DESERT, BEACH = range(len(BIOMES))

#: Flat colours for maps, placeholders and the biome debug view (sRGB).
BIOME_COLORS = np.array(
    [
        (0.92, 0.94, 0.98),  # snow
        (0.42, 0.40, 0.38),  # rock
        (0.30, 0.48, 0.16),  # grass
        (0.10, 0.28, 0.10),  # forest
        (0.82, 0.66, 0.40),  # desert
        (0.90, 0.84, 0.62),  # beach
    ]
)

#: Slopes, in degrees, over which ground turns to rock.
ROCK_SLOPE = (30.0, 48.0)
#: Altitude, in metres, below which the coast is beach. Above zero, to clear
#: the ocean swell, which lifts the water near the camera.
BEACH_TOP = 6.0
#: Horizontal scale the slope is measured at, metres. Fixed, so rock does not
#: appear or vanish when a chunk changes LOD.
SLOPE_SCALE = 8.0


def slope_degrees(elevation: np.ndarray, spacing: float, step: int = 1) -> np.ndarray:
    """Terrain slope by central differences over +-*step* samples.

    The outermost *step* rows/columns are left at the nearest valid value;
    callers sample a border wide enough to discard them.

    :param elevation: (n, n) heights, metres
    :param spacing: sample spacing, metres
    :param step: half-width of the difference, in samples
    :returns: (n, n) slope in degrees
    """
    dz_dx = np.zeros_like(elevation)
    dz_dy = np.zeros_like(elevation)
    span = 2.0 * step * spacing
    dz_dx[:, step:-step] = (elevation[:, 2 * step :] - elevation[:, : -2 * step]) / span
    dz_dy[step:-step, :] = (elevation[2 * step :, :] - elevation[: -2 * step, :]) / span
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))


def slope_from_normals(normals: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Terrain slope from surface normals and the local vertical.

    :param normals: (..., 3) unit normals
    :param up: (..., 3) unit radial directions, same shape
    :returns: (...) slope in degrees
    """
    cos = np.clip(np.einsum("...i,...i->...", normals, up), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def weights(
    temperature: np.ndarray,
    humidity: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    has_ocean: bool,
) -> np.ndarray:
    """Normalised biome weights.

    :param temperature: degrees Celsius, altitude lapse included
    :param humidity: in [0, 1]
    :param elevation: metres above sea level
    :param slope: degrees
    :param has_ocean: whether the planet has a sea (beaches)
    :returns: (N, 6) float32 summing to 1, in :data:`BIOMES` order
    """
    t = np.ravel(temperature)
    h = np.ravel(humidity)
    z = np.ravel(elevation)
    s = np.ravel(slope)
    w = np.zeros((t.size, len(BIOMES)))

    snow = 1.0 - smoothstep(-8.0, 1.5, t)
    # Dryness makes desert; only the cold (steppe, tundra) is spared.
    desert = smoothstep(6.0, 14.0, t) * (1.0 - smoothstep(0.16, 0.38, h))
    # Trees need water, and neither frost nor scorching heat: the cold edge is
    # the tree line on mountain flanks.
    forest = (
        smoothstep(0.38, 0.62, h)
        * smoothstep(1.0, 6.0, t)
        * (1.0 - smoothstep(27.0, 33.0, t))
    )
    w[:, SNOW] = snow
    w[:, DESERT] = desert * (1.0 - snow)
    w[:, FOREST] = forest * (1.0 - snow) * (1.0 - desert)
    w[:, GRASS] = np.maximum(0.05, 1.0 - w.sum(axis=1))
    w /= w.sum(axis=1, keepdims=True)

    if has_ocean:
        beach = (
            (1.0 - smoothstep(BEACH_TOP - 4.0, BEACH_TOP + 1.0, z))
            * (1.0 - smoothstep(12.0, 26.0, s))
            * (1.0 - snow)
        )
        w *= (1.0 - beach)[:, None]
        w[:, BEACH] += beach

    rock = smoothstep(ROCK_SLOPE[0], ROCK_SLOPE[1], s)
    w *= (1.0 - rock)[:, None]
    w[:, ROCK] += rock
    return w.astype(np.float32)


def dominant(w: np.ndarray) -> np.ndarray:
    """:returns: index of the strongest biome per row of (N, 6) weights"""
    return np.argmax(w, axis=1)


def colors(w: np.ndarray) -> np.ndarray:
    """:returns: (N, 3) flat colours blended by the weights"""
    return w @ BIOME_COLORS
