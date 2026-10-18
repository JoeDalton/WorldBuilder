"""
Top-down terrain maps, rendered offline to PNG: tune the look with no 3D.

Each preset gets one row of maps centred on the planet's reference point, at
three scales: 400 km (the horizon from 20 km up), 20 km, and 1 km. The maps are
biome colour times hillshade, with water drawn blue on planets with an ocean.

    python -m world_builder.terrain.preview [--seed N] [--out DIR]
        [--preset NAME ...] [--size PX]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from world_builder.terrain import biomes
from world_builder.terrain.planet import PRESETS, Planet, PlanetSpec
from world_builder.terrain.pngio import write_png

SCALES_M = (400e3, 20e3, 1e3)
_LIGHT = np.array([-1.0, 1.0, 1.4]) / np.linalg.norm([-1.0, 1.0, 1.4])
_WATER_SHALLOW = np.array([0.12, 0.35, 0.45])
_WATER_DEEP = np.array([0.02, 0.06, 0.18])


def render_map(
    planet: Planet, extent: float, size: int, center=(0.0, 0.0)
) -> np.ndarray:
    """One square map.

    :param planet: the planet
    :param extent: side of the square, metres
    :param size: side, pixels
    :param center: chart (x, y) of the map centre, metres
    :returns: (size, size, 3) uint8, north up
    """
    spacing = extent / (size - 1)
    step = max(1, round(biomes.SLOPE_SCALE / spacing))
    border = step
    n = size + 2 * border
    offset = -extent / 2.0 - border * spacing
    sample = planet.sample_grid(center[0] + offset, center[1] + offset, spacing, n)
    elevation = sample.elevation
    slope = biomes.slope_degrees(elevation, spacing, step)
    w = biomes.weights(
        sample.temperature,
        sample.humidity,
        elevation,
        slope,
        planet.spec.has_ocean,
    )
    for paint in planet.spec.paints:
        w = paint.apply(sample.dirs, w, planet.spec.radius)
    color = biomes.colors(w).reshape(n, n, 3)

    gy, gx = np.gradient(elevation, spacing)
    normal = np.dstack([-gx, -gy, np.ones_like(gx)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    shade = 0.35 + 0.65 * np.clip(normal @ _LIGHT, 0.0, 1.0)
    color = color * shade[:, :, None]

    if planet.spec.has_ocean:
        depth = np.clip(-elevation / 3000.0, 0.0, 1.0)[:, :, None]
        water = _WATER_SHALLOW * (1.0 - depth) + _WATER_DEEP * depth
        color = np.where((elevation < 0.0)[:, :, None], water, color)

    color = color[border:-border, border:-border]
    # Row 0 is the southmost; images store the top (north) row first.
    srgb = np.clip(color, 0.0, 1.0) ** (1.0 / 2.2)
    return (srgb[::-1] * 255.0 + 0.5).astype(np.uint8)


def render_sheet(specs: list[PlanetSpec], size: int) -> np.ndarray:
    """:returns: one row of maps per spec, one column per scale, 4 px gutters"""
    gutter = 4
    rows = []
    for spec in specs:
        start = time.perf_counter()
        planet = Planet(spec)
        maps = [render_map(planet, extent, size) for extent in SCALES_M]
        print(
            f"  {spec.name:<10} seed {spec.seed}: {time.perf_counter() - start:.1f} s"
        )
        pad = np.full((size, gutter, 3), 255, dtype=np.uint8)
        rows.append(np.hstack([m for pair in zip(maps, [pad] * 3) for m in pair][:-1]))
        rows.append(np.full((gutter, rows[-1].shape[1], 3), 255, dtype=np.uint8))
    return np.vstack(rows[:-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--preset", nargs="*", default=list(PRESETS))
    parser.add_argument("--out", type=Path, default=Path("terrain_preview"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    specs = [replace(PRESETS[name], seed=args.seed) for name in args.preset]
    print(f"rendering {len(specs)} presets at {args.size} px ...")
    sheet = render_sheet(specs, args.size)
    path = args.out / f"preview_seed{args.seed}.png"
    write_png(path, sheet)
    print(f"wrote {path}  (columns: {', '.join(f'{s / 1e3:g} km' for s in SCALES_M)})")


if __name__ == "__main__":
    main()
