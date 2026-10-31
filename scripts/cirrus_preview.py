"""
Preview the planet's cirrus from the CPU, as seen looking up: a tuning tool
for its stamps (scripts/bake_cirrus_atlas.py) and their placement
(world_builder.clouds.cirrus), much faster than the demo::

    python scripts/cirrus_preview.py --atlas [--out FILE]
    python scripts/cirrus_preview.py --image GREY.png [--out FILE]
    python scripts/cirrus_preview.py --field [--lat DEG] [--lon DEG]
        [--size KM KM] [--pixels N] [--reference PHOTO] [--out FILE]

    --atlas      the baked stamps, white over sky blue
    --image      any thickness image (a stamp baked with --only), likewise
    --field      the sheet over a patch of the demo planet, top-down, as the
                 stamps are placed there (the cover map, the wind); prints
                 the cover measured against the map's
    --lat/--lon  the patch's centre, degrees (default 45, 10: the
                 westerlies)
    --size       the patch, km (default 120 80)
    --pixels     its width in pixels (default 1200)
    --reference  a photo to put beside the field, scaled to its height
    --out        the PNG (default cirrus_preview.png)

The colour is a sky-blue background with the cirrus's optical depth (the
stamps' thickness, as the shader sums it) over it in white: no sun, no air.
It shows shapes, not light.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from panda3d.core import Filename, PNMImage, StringStream

SKY = np.array([0.20, 0.40, 0.80])
CLOUD = np.array([0.95, 0.96, 1.0])


def _decode(image: PNMImage, channels: int) -> np.ndarray:
    """A PNMImage's pixels through its binary PGM/PPM form: (rows, columns[,
    3]) in [0, 1], row 0 at the top."""
    if channels == 1:
        image.make_grayscale()
    else:
        image.make_rgb()
    stream = StringStream()
    image.write(stream, "image.pgm" if channels == 1 else "image.ppm")
    data = stream.get_data()
    header = data.split(maxsplit=4)
    columns, rows = int(header[1]), int(header[2])
    pixels = np.frombuffer(data[-columns * rows * channels :], dtype=np.uint8)
    shape = (rows, columns) if channels == 1 else (rows, columns, 3)
    return pixels.reshape(shape) / 255.0


def _read(path: Path) -> PNMImage:
    image = PNMImage()
    if not image.read(Filename.from_os_specific(str(path))):
        raise OSError(f"could not read {path}")
    return image


def read_grey(path: Path) -> np.ndarray:
    """A grey PNG as (rows, columns) in [0, 1], row 0 at the bottom."""
    return _decode(_read(path), 1)[::-1]


def read_rgb(path: Path) -> np.ndarray:
    """An RGB picture as (rows, columns, 3) in [0, 1], row 0 at the top."""
    return _decode(_read(path), 3)


def composite(depth: np.ndarray) -> np.ndarray:
    """Optical depth (row 0 at the bottom) as white cloud over sky, (rows,
    columns, 3) row 0 at the top."""
    alpha = 1.0 - np.exp2(-depth)
    return (SKY * (1.0 - alpha[..., None]) + CLOUD * alpha[..., None])[::-1]


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    rows, columns, _ = rgb.shape
    pixels = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = PNMImage()
    image.read(
        StringStream(f"P6 {columns} {rows} 255\n".encode() + pixels.tobytes()), "x.ppm"
    )
    image.write(Filename.from_os_specific(str(path)))


def field(lat: float, lon: float, size_km, pixels: int) -> np.ndarray:
    """The sheet's optical depth over a patch of the demo planet, top-down
    (east to the right, north up), row 0 at the bottom."""
    from world_builder.clouds.cirrus import (
        CirrusSpec,
        cirrus_stamps,
        stamp_cover,
        stamp_thickness,
    )
    from world_builder.clouds.weather import WeatherSpec, cirrus_cover_map, sample_cube
    from world_builder.terrain.chart import TangentFrame
    from world_builder.terrain.level import DEMO_LEVEL, load_level

    planet, _ = load_level(str(DEMO_LEVEL))
    spec = CirrusSpec()
    weather = WeatherSpec(planet)
    stamps = cirrus_stamps(spec, weather, planet.radius)
    frame = TangentFrame.at(lat, lon)
    width, height = size_km[0] * 1e3, size_km[1] * 1e3
    rows = int(round(pixels * height / width))
    x = ((np.arange(pixels) + 0.5) / pixels - 0.5) * width
    y = ((np.arange(rows) + 0.5) / rows - 0.5) * height
    xx, yy = np.meshgrid(x, y)
    flat = (
        frame.up * stamps.sheet
        + xx.reshape(-1, 1) * frame.east
        + yy.reshape(-1, 1) * frame.north
    )
    points = flat / np.linalg.norm(flat, axis=1, keepdims=True) * stamps.sheet
    depth = spec.extinction * spec.thickness * stamp_thickness(stamps, points)
    sample = points[:: max(1, len(points) // 40000)]
    covered = stamp_cover(stamps, sample).mean()
    wanted = np.clip(
        sample_cube(cirrus_cover_map(weather) * spec.amount, sample), 0.0, 0.95
    ).mean()
    print(
        f"{len(stamps)} stamps on the planet; this patch: cover {covered:.2f} "
        f"(the map asks {wanted:.2f})"
    )
    return depth.reshape(rows, pixels)


def beside(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """*right* scaled (nearest) to *left*'s height and put beside it."""
    rows = left.shape[0]
    pick = (
        np.arange(int(right.shape[1] * rows / right.shape[0])) * right.shape[0] / rows
    ).astype(int)
    scaled = right[(np.arange(rows) * right.shape[0] / rows).astype(int)][:, pick]
    gap = np.ones((rows, 8, 3))
    return np.concatenate([left, gap, scaled], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--atlas", action="store_true")
    mode.add_argument("--image", type=Path)
    mode.add_argument("--field", action="store_true")
    parser.add_argument("--lat", type=float, default=45.0)
    parser.add_argument("--lon", type=float, default=10.0)
    parser.add_argument("--size", type=float, nargs=2, default=(120.0, 80.0))
    parser.add_argument("--pixels", type=int, default=1200)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("cirrus_preview.png"))
    args = parser.parse_args()

    if args.image is not None:
        rgb = composite(1.5 * read_grey(args.image))
    elif args.atlas:
        from world_builder.clouds.cirrus import ATLAS_PNG

        rgb = composite(1.5 * read_grey(ATLAS_PNG))
    else:
        rgb = composite(field(args.lat, args.lon, args.size, args.pixels))
        if args.reference is not None:
            rgb = beside(rgb, read_rgb(args.reference))
    write_rgb(args.out, rgb)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
