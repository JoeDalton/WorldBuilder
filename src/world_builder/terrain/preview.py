"""
Top-down terrain maps, rendered offline to PNG: tune the look with no 3D.

Each planet gets one row: the whole planet (equirectangular, twice as wide as
tall, the reference point marked), then maps centred on the planet's
reference point at three scales: 100 km, 20 km and 1 km. The maps are biome
colour times hillshade, with water drawn blue on planets with an ocean.

With --weather, each row is instead three maps of the planet's weather
(clouds.weather, its seed --seed, at --time seconds, with --pin presets): the
globe under all its cloud (white where the sky is covered), the cloud genera
by colour, and a surface chart (isobars, fronts, the jets, the systems'
centres), to tune the weather; airless planets (no humidity) are skipped.
Sheets are written to --out (``terrain_preview`` by default)::

    python -m world_builder.terrain.preview [--seed N] [--radius M] [--out DIR]
        [--planet FILE|NAME ...] [--size PX] [--weather] [--time S]
        [--pin LAT LON PRESET ...]

A planet is a body file or world file (``worlds/desert.yaml``) or the name of
a shipped one (``desert``, ``earth``, ...: ``world_file.SHIPPED_PLANETS``,
all of them by default).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from world_builder.terrain import biomes
from world_builder.terrain.chart import AzimuthalEquidistantChart, dirs_from_latlon
from world_builder.terrain.planet import Planet, PlanetSpec
from world_builder.terrain.pngio import write_png

SCALES_M = (100e3, 20e3, 1e3)
_LIGHT = np.array([-1.0, 1.0, 1.4]) / np.linalg.norm([-1.0, 1.0, 1.4])
_WATER_SHALLOW = np.array([0.12, 0.35, 0.45])
_WATER_DEEP = np.array([0.02, 0.06, 0.18])


def _shade(planet: Planet, elevation, temperature, humidity, dirs, spacing, step):
    """:returns: (rows, cols, 3) linear colour: biomes, hillshade, water"""
    slope = biomes.slope_degrees(elevation, spacing, step)
    w = biomes.weights(
        temperature,
        humidity,
        elevation,
        slope,
        planet.spec.has_ocean,
        planet.spec.has_vegetation,
    )
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


def _globe_dirs(height: int) -> np.ndarray:
    """:returns: (height, 2 * height, 3) directions of an equirectangular map,
    the southmost row first"""
    lat = np.linspace(-90.0, 90.0, height)
    lon = np.linspace(-180.0, 180.0, 2 * height)
    glon, glat = np.meshgrid(lon, lat)
    return dirs_from_latlon(glat.ravel(), glon.ravel()).reshape(height, -1, 3)


def render_globe(planet: Planet, height: int, linear: bool = False) -> np.ndarray:
    """The whole planet, equirectangular (longitude across, latitude up).

    :param planet: the planet
    :param height: pixels pole to pole; the map is twice as wide
    :param linear: return the linear colour (southmost row first) instead
    :returns: (height, 2 * height, 3) uint8, north up, with the reference
        point marked; float linear colour, south up, if *linear*
    """
    spec = planet.spec
    dirs = _globe_dirs(height)
    shape = dirs.shape[:2]
    dirs = dirs.reshape(-1, 3)
    # One pixel spans this much ground at the equator; detail below it is noise.
    spacing = np.pi * spec.radius / height
    elevation, temperature, humidity = planet.sample_dirs(dirs, 2.0 * spacing)
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
    return color if linear else _to_image(color)


def render_weather(
    planet: Planet, height: int, seed: int = 11, time_s: float = 0.0, pins: tuple = ()
) -> np.ndarray:
    """Three equirectangular maps of the weather, side by side: the globe
    under all its cloud (white = overcast), the genera by colour (low
    yellow, mid blue, high violet, cumulus white, cumulonimbus red), and a
    surface chart (isobars every 4 hPa, fronts, the jets' axes; lows
    magenta, highs cyan, tropical cyclones orange).

    :param planet: the planet the weather follows
    :param height: pixels pole to pole of each map
    :param seed: the weather's seed (:class:`clouds.weather.WeatherSpec`)
    :param time_s: the weather's time, seconds
    :param pins: local presets (:class:`clouds.synoptic.presets.WeatherPin`)
    :returns: (height, 6 * height + 8, 3) uint8, north up
    """
    from world_builder.clouds.synoptic.chart import draw_chart, genus_colours
    from world_builder.clouds.synoptic.state import synoptic_state
    from world_builder.clouds.weather import (
        SYNOPTIC_CHANNELS,
        WeatherSpec,
        sample_cube,
        synoptic_cube,
    )

    spec = WeatherSpec(planet.spec, seed=seed, time_s=time_s, pins=tuple(pins))
    dirs = _globe_dirs(height)
    values = sample_cube(synoptic_cube(spec), dirs)
    channels = {n: values[..., i] for i, n in enumerate(SYNOPTIC_CHANNELS)}
    cloud = np.maximum.reduce(
        [c for n, c in channels.items() if n not in ("depth", "shelf")]
    )[..., None]
    ground = render_globe(planet, height, linear=True)
    under = ground * (1.0 - 0.85 * cloud) + 0.85 * cloud
    genera = genus_colours(0.6 * ground, channels)
    state = synoptic_state(spec)
    pale = 0.55 + 0.45 * np.clip(ground, 0.0, 1.0) ** (1.0 / 2.2)
    chart = draw_chart(pale, dirs, state.systems, state.jets, state.scales, state.t)
    pad = np.full((height, 4, 3), 255, dtype=np.uint8)
    chart_image = (np.clip(chart, 0.0, 1.0)[::-1] * 255.0 + 0.5).astype(np.uint8)
    return np.hstack([_to_image(under), pad, _to_image(genera), pad, chart_image])


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


def add_weather_arguments(parser: argparse.ArgumentParser) -> None:
    """``--time SECONDS`` and ``--pin LAT LON PRESET`` (repeatable): the
    weather's time and its local presets (:mod:`clouds.synoptic.presets`)."""
    from world_builder.clouds.synoptic.presets import Preset

    parser.add_argument(
        "--time", type=float, default=0.0, help="the weather's time, seconds"
    )
    parser.add_argument(
        "--pin",
        nargs=3,
        action="append",
        default=[],
        metavar=("LAT", "LON", "PRESET"),
        help="a local weather preset pinned there: "
        + ", ".join(p.value for p in Preset),
    )


def pins_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple:
    """:returns: the ``--pin`` presets (argparse's error on a bad one)"""
    from world_builder.clouds.synoptic.presets import parse_pin

    try:
        return tuple(parse_pin(*pin) for pin in args.pin)
    except ValueError as error:
        parser.error(f"--pin: {error}")


def main() -> None:
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--radius", type=float, default=None, help="metres")
    parser.add_argument("--size", type=int, default=320)
    from world_builder.world_file import SHIPPED_PLANETS, load_planet

    parser.add_argument("--planet", nargs="*", default=list(SHIPPED_PLANETS))
    parser.add_argument("--out", type=Path, default=Path("terrain_preview"))
    parser.add_argument(
        "--weather", action="store_true", help="the clouds' weather map instead"
    )
    add_weather_arguments(parser)
    args = parser.parse_args()
    pins = pins_from_args(parser, args)
    args.out.mkdir(parents=True, exist_ok=True)
    specs = []
    for name in args.planet:
        spec = replace(load_planet(name), seed=args.seed)
        if args.radius is not None:
            spec = replace(spec, radius=args.radius)
        specs.append(spec)
    if args.weather:
        rows = []
        for spec in specs:
            if spec.climate.humidity_mean <= 0.0:
                continue  # airless: no weather
            start = time.perf_counter()
            rows.append(
                render_weather(Planet(spec), args.size, args.seed, args.time, pins)
            )
            print(f"  {spec.name:<10} {time.perf_counter() - start:.1f} s")
        path = args.out / f"weather_seed{args.seed}.png"
        write_png(path, np.vstack(rows))
        print(f"wrote {path}  (columns: globe under the weather, genera, chart)")
        return
    print(f"rendering {len(specs)} presets at {args.size} px ...")
    sheet = render_sheet(specs, args.size)
    radius_km = specs[0].radius / 1e3
    path = args.out / f"preview_seed{args.seed}_r{radius_km:g}km.png"
    write_png(path, sheet)
    scales = ", ".join(f"{s / 1e3:g} km" for s in SCALES_M)
    print(f"wrote {path}  (columns: globe, {scales})")


if __name__ == "__main__":
    main()
