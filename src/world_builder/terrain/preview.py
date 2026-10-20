"""
Top-down terrain maps, rendered offline to PNG: tune the look with no 3D.

Each preset gets one row: the whole planet (equirectangular, twice as wide as
tall, the reference point marked), then maps centred on the planet's
reference point at three scales: 100 km, 20 km and 1 km. The maps are biome
colour times hillshade, with water drawn blue on planets with an ocean.

    python -m world_builder.terrain.preview [--seed N] [--radius M] [--out DIR]
        [--preset NAME ...] [--size PX]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from world_builder.terrain import biomes
from world_builder.terrain.chart import AzimuthalEquidistantChart, dirs_from_latlon
from world_builder.terrain.planet import PRESETS, Planet, PlanetSpec
from world_builder.terrain.pngio import write_png

SCALES_M = (100e3, 20e3, 1e3)
_LIGHT = np.array([-1.0, 1.0, 1.4]) / np.linalg.norm([-1.0, 1.0, 1.4])
_WATER_SHALLOW = np.array([0.12, 0.35, 0.45])
_WATER_DEEP = np.array([0.02, 0.06, 0.18])


def _shade(planet: Planet, elevation, temperature, humidity, dirs, spacing, step):
    """:returns: (rows, cols, 3) linear colour: biomes, hillshade, water"""
    slope = biomes.slope_degrees(elevation, spacing, step)
    w = biomes.weights(temperature, humidity, elevation, slope, planet.spec.has_ocean)
    for paint in planet.spec.paints:
        w = paint.apply(dirs, w, planet.spec.radius)
    color = biomes.colors(w).reshape(*elevation.shape, 3)

    gy, gx = np.gradient(elevation, spacing)
    normal = np.dstack([-gx, -gy, np.ones_like(gx)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    shade = 0.35 + 0.65 * np.clip(normal @ _LIGHT, 0.0, 1.0)
    color = color * shade[:, :, None]

    if planet.spec.has_ocean:
        depth = np.clip(-elevation / 3000.0, 0.0, 1.0)[:, :, None]
        water = _WATER_SHALLOW * (1.0 - depth) + _WATER_DEEP * depth
        color = np.where((elevation < 0.0)[:, :, None], water, color)
    return color


def _to_image(color: np.ndarray) -> np.ndarray:
    # Row 0 is the southmost; images store the top (north) row first.
    srgb = np.clip(color, 0.0, 1.0) ** (1.0 / 2.2)
    return (srgb[::-1] * 255.0 + 0.5).astype(np.uint8)


def render_map(planet: Planet, extent: float, size: int) -> np.ndarray:
    """One square map centred on the reference point.

    :param planet: the planet
    :param extent: side of the square, metres
    :param size: side, pixels
    :returns: (size, size, 3) uint8, north up
    """
    spec = planet.spec
    spacing = extent / (size - 1)
    step = max(1, round(biomes.SLOPE_SCALE / spacing))
    border = step
    n = size + 2 * border
    coords = spacing * (np.arange(n) - border) - extent / 2.0
    gx, gy = np.meshgrid(coords, coords)
    chart = AzimuthalEquidistantChart(
        spec.reference_lat, spec.reference_lon, spec.radius
    )
    dirs = chart.to_dirs(gx, gy)
    elevation, temperature, humidity = planet.sample_dirs(
        dirs, 4.0 * spacing, apply_stamps=True
    )
    color = _shade(
        planet,
        elevation.reshape(n, n),
        temperature.reshape(n, n),
        humidity.reshape(n, n),
        dirs,
        spacing,
        step,
    )
    return _to_image(color[border:-border, border:-border])


def render_globe(planet: Planet, height: int) -> np.ndarray:
    """The whole planet, equirectangular (longitude across, latitude up).

    :param planet: the planet
    :param height: pixels pole to pole; the map is twice as wide
    :returns: (height, 2 * height, 3) uint8, with the reference point marked
    """
    spec = planet.spec
    lat = np.linspace(-90.0, 90.0, height)
    lon = np.linspace(-180.0, 180.0, 2 * height)
    glon, glat = np.meshgrid(lon, lat)
    dirs = dirs_from_latlon(glat.ravel(), glon.ravel())
    # One pixel spans this much ground at the equator; detail below it is noise.
    spacing = np.pi * spec.radius / height
    elevation, temperature, humidity = planet.sample_dirs(dirs, 2.0 * spacing)
    shape = glat.shape
    color = _shade(
        planet,
        elevation.reshape(shape),
        temperature.reshape(shape),
        humidity.reshape(shape),
        dirs,
        spacing,
        1,
    )
    row = int(round((spec.reference_lat + 90.0) / 180.0 * (height - 1)))
    col = int(round((spec.reference_lon + 180.0) / 360.0 * (2 * height - 1)))
    color[max(row - 2, 0) : row + 3, max(col - 2, 0) : col + 3] = (1.0, 0.0, 0.0)
    return _to_image(color)


def render_sheet(specs: list[PlanetSpec], size: int) -> np.ndarray:
    """:returns: one row per spec: globe, then one map per scale, 4 px gutters"""
    gutter = 4
    rows = []
    for spec in specs:
        start = time.perf_counter()
        planet = Planet(spec)
        maps = [render_globe(planet, size)]
        maps += [render_map(planet, extent, size) for extent in SCALES_M]
        print(
            f"  {spec.name:<10} seed {spec.seed}: {time.perf_counter() - start:.1f} s"
        )
        pad = np.full((size, gutter, 3), 255, dtype=np.uint8)
        cells = []
        for image in maps:
            cells += [image, pad]
        rows.append(np.hstack(cells[:-1]))
        rows.append(np.full((gutter, rows[-1].shape[1], 3), 255, dtype=np.uint8))
    return np.vstack(rows[:-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--radius", type=float, default=None, help="metres")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--preset", nargs="*", default=list(PRESETS))
    parser.add_argument("--out", type=Path, default=Path("terrain_preview"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    specs = []
    for name in args.preset:
        spec = replace(PRESETS[name], seed=args.seed)
        if args.radius is not None:
            spec = replace(spec, radius=args.radius)
        specs.append(spec)
    print(f"rendering {len(specs)} presets at {args.size} px ...")
    sheet = render_sheet(specs, args.size)
    radius_km = specs[0].radius / 1e3
    path = args.out / f"preview_seed{args.seed}_r{radius_km:g}km.png"
    write_png(path, sheet)
    scales = ", ".join(f"{s / 1e3:g} km" for s in SCALES_M)
    print(f"wrote {path}  (columns: globe, {scales})")


if __name__ == "__main__":
    main()
