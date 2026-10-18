"""
Level files: a planet preset with overrides, plus hand-made stamps and paints.

::

    planet:
      preset: earthlike          # one of planet.PRESETS
      seed: 7
      reference: {lat: 42.0, lon: 10.0}   # where the flat world's origin is
      has_ocean: true
      elevation: {mountain_height: 6000}  # any ElevationParams field
      climate: {temperature_mean: 12}     # any ClimateParams field
    stamps:
      - heightmap: stamps/crater.png      # relative to this file
        center: {x: 3000, y: 9000}        # chart metres, or {lat: .., lon: ..}
        size: [6000, 6000]                # metres east-west, north-south
        rotation: 0                       # degrees, counter-clockwise
        height_scale: 900                 # metres per full-white pixel
        height_offset: -315               # metres added to every pixel
        blend: add                        # replace | add | max | min | flatten
        falloff: 600                      # metres of edge fade
    paints:
      - mask: stamps/crater_paint.png
        stamp: 0                          # reuse stamp 0's placement, or give
                                          # center/size/rotation
        colors: {"#808080": rock, "#e0c080": desert}
        falloff: 80
        strength: 1.0
    camera: {x: 0, y: 0, z: 600, heading: 0, pitch: -8}   # demo start pose

Positions in chart metres are converted to latitude/longitude at load, so a
stamp stays put on the planet even if the reference point moves.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import yaml

from world_builder.terrain.chart import AzimuthalEquidistantChart, latlon
from world_builder.terrain.planet import (
    PRESETS,
    ClimateParams,
    ElevationParams,
)
from world_builder.terrain.pngio import write_png
from world_builder.terrain.stamps import Paint, Placement, Stamp

LEVELS_PATH = Path(__file__).parent / "levels"
DEMO_LEVEL = LEVELS_PATH / "demo.yaml"


def load_level(path: Path | str, preset: str | None = None, seed: int | None = None):
    """Read a level file.

    :param path: the YAML file
    :param preset: overrides the level's preset (the hand-made content stays)
    :param seed: overrides the level's seed
    :returns: (resolved PlanetSpec, camera pose dict or None)
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    planet = data.get("planet", {}) or {}
    base = PRESETS[preset or planet.get("preset", "earthlike")]
    reference = planet.get("reference", {}) or {}
    spec = replace(
        base,
        seed=int(seed if seed is not None else planet.get("seed", base.seed)),
        reference_lat=float(reference.get("lat", base.reference_lat)),
        reference_lon=float(reference.get("lon", base.reference_lon)),
        has_ocean=bool(planet.get("has_ocean", base.has_ocean)),
        elevation=_override(base.elevation, planet.get("elevation"), ElevationParams),
        climate=_override(base.climate, planet.get("climate"), ClimateParams),
    )
    chart = AzimuthalEquidistantChart(
        spec.reference_lat, spec.reference_lon, spec.radius
    )
    folder = path.parent

    stamps = []
    for entry in data.get("stamps", []) or []:
        stamps.append(
            Stamp(
                heightmap=str((folder / entry["heightmap"]).resolve()),
                placement=_placement(entry, chart),
                height_scale=float(entry.get("height_scale", 1000.0)),
                height_offset=float(entry.get("height_offset", 0.0)),
                blend=entry.get("blend", "add"),
                falloff=float(entry.get("falloff", 0.0)),
            )
        )
    paints = []
    for entry in data.get("paints", []) or []:
        placement = (
            stamps[int(entry["stamp"])].placement
            if "stamp" in entry
            else _placement(entry, chart)
        )
        paints.append(
            Paint(
                mask=str((folder / entry["mask"]).resolve()),
                placement=placement,
                colors=tuple((str(c), str(b)) for c, b in entry["colors"].items()),
                falloff=float(entry.get("falloff", 0.0)),
                strength=float(entry.get("strength", 1.0)),
                tolerance=float(entry.get("tolerance", 0.12)),
            )
        )
    spec = replace(spec, stamps=tuple(stamps), paints=tuple(paints)).resolved()
    return spec, data.get("camera")


def _override(params, values, cls):
    if not values:
        return params
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    converted = {
        k: tuple(tuple(p) for p in v) if isinstance(v, list) else v
        for k, v in values.items()
    }
    return replace(params, **converted)


def _placement(entry: dict, chart: AzimuthalEquidistantChart) -> Placement:
    center = entry["center"]
    if "lat" in center:
        lat, lon = float(center["lat"]), float(center["lon"])
    else:
        dirs = chart.to_dirs(np.array([center["x"]]), np.array([center["y"]]))
        lat_a, lon_a = latlon(dirs)
        lat, lon = float(lat_a[0]), float(lon_a[0])
    size = entry["size"]
    width, height = (size, size) if np.isscalar(size) else size
    return Placement(
        lat=lat,
        lon=lon,
        width=float(width),
        height=float(height),
        rotation=float(entry.get("rotation", 0.0)),
    )


# ── sample content ────────────────────────────────────────────────────────────


def write_sample_stamps(folder: Path = LEVELS_PATH / "stamps") -> None:
    """Generate the demo's sample stamp and paint images.

    :param folder: destination
    """
    folder.mkdir(parents=True, exist_ok=True)
    n = 512
    yy, xx = np.mgrid[-1.0 : 1.0 : n * 1j, -1.0 : 1.0 : n * 1j]
    r = np.hypot(xx, yy)
    angle = np.arctan2(yy, xx)
    wobble = 1.0 + 0.04 * np.sin(5 * angle + 1.0) + 0.025 * np.sin(11 * angle + 2.0)
    rr = r * wobble

    # Crater: bowl, sharp rim at 0.62, ejecta decaying outward, central peak.
    bowl = 0.05 + 0.95 * (rr / 0.62) ** 2.5
    ejecta = 0.35 + 0.65 * np.exp(-(((rr - 0.62) / 0.12) ** 2))
    crater = np.where(rr < 0.62, bowl, ejecta)
    crater += 0.22 * np.exp(-((rr / 0.09) ** 2))
    crater = np.clip(crater, 0.0, 1.0)
    write_png(folder / "crater.png", (crater * 65535).astype(np.uint16))

    paint = np.zeros((n, n, 3), dtype=np.uint8)
    paint[(rr > 0.5) & (rr < 0.78)] = (128, 128, 128)  # rim: rock
    paint[rr < 0.45] = (224, 192, 128)  # floor: sand
    write_png(folder / "crater_paint.png", paint)

    # Mesa: a lobed plateau with steep, terraced flanks.
    lobes = 0.55 + 0.12 * np.sin(3 * angle) + 0.07 * np.sin(7 * angle + 0.5)
    edge = np.clip((lobes - r) / 0.06, 0.0, 1.0)
    terraces = np.floor(edge * 4.0) / 4.0
    mesa = 0.65 * terraces + 0.35 * edge
    write_png(folder / "mesa.png", (np.clip(mesa, 0, 1) * 65535).astype(np.uint16))

    # Pad: flat, ringed by a low berm.
    m = 256
    yy, xx = np.mgrid[-1.0 : 1.0 : m * 1j, -1.0 : 1.0 : m * 1j]
    box = np.maximum(np.abs(xx), np.abs(yy))
    berm = np.exp(-(((box - 0.82) / 0.04) ** 2))
    write_png(folder / "pad.png", (berm * 65535).astype(np.uint16))
    pad_paint = np.zeros((m, m, 3), dtype=np.uint8)
    pad_paint[box < 0.9] = (255, 230, 120)  # sand
    write_png(folder / "pad_paint.png", pad_paint)

    lake, lake_paint = cliff_lake()
    write_png(folder / "lake.png", (lake * 65535).astype(np.uint16))
    write_png(folder / "lake_paint.png", lake_paint)


def cliff_lake(n: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """A lake whose north shore is a sheer cliff and whose other shores are beaches.

    Meant for ``blend: replace`` with ``height_scale: 600`` and
    ``height_offset: -60``: pixel value v is height -60 + 600 v metres, so
    v = 0.1 is sea level, where the game's water surface lies. The lake bed
    goes to -60 m; it is deep right up to the cliff and shelves gently
    elsewhere. The cliff rises from the waterline to about 310 m within a few
    pixels, onto a plateau that slopes back down northwards.

    :param n: image side, pixels
    :returns: (heightmap (n, n) in [0, 1], paint (n, n, 3) uint8: forest on
        the plateau, away from the cliff edge)
    """
    from world_builder.terrain.materials import tileable_noise

    coords = np.linspace(-1.0, 1.0, n)
    x, north = np.meshgrid(coords, -coords)  # image rows run north to south
    centre = -0.1
    angle = np.arctan2(north - centre, x)
    wobble = (
        1.0
        + 0.06 * np.sin(3 * angle + 0.7)
        + 0.035 * np.sin(7 * angle + 2.1)
        + 0.02 * np.sin(13 * angle)
    )
    q = np.hypot(x / 0.55, (north - centre) / 0.40) / wobble  # 1 on the shore
    d = q - 1.0  # distance from the shore, in lake radii
    degrees = np.degrees(angle)
    # 1 along the northern arc of the shore, tapering at both ends.
    cliff = _smoothstep(5.0, 30.0, degrees) * (1.0 - _smoothstep(150.0, 175.0, degrees))

    noise = (tileable_noise(n, 1.8, seed=11) - 0.5) * 0.035  # about +-10 m
    land_noise = noise * _smoothstep(0.0, 0.02, d)  # keep the waterline exact
    gentle = 0.1 + 0.25 * _smoothstep(0.0, 0.9, d) + land_noise
    plateau = (
        0.62
        + 0.06 * _smoothstep(0.0, 0.25, d)
        - 0.30 * _smoothstep(0.35, 1.3, d)
        + land_noise
    )
    # Three pixels from waterline to the top: about 300 m over 15 m.
    rise = cliff * _smoothstep(0.0, 0.006, d)
    land = gentle + rise * (plateau - gentle)
    # Under the cliff the bed stays deep to within metres of the wall.
    bed = 0.1 * np.clip(q, 0.0, 1.0) ** (2.0 + 200.0 * cliff)
    height = np.clip(np.where(d < 0.0, bed, land), 0.0, 1.0)

    paint = np.zeros((n, n, 3), dtype=np.uint8)
    paint[(cliff > 0.5) & (d > 0.04) & (d < 1.0)] = (46, 107, 46)
    return height, paint


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


if __name__ == "__main__":
    write_sample_stamps()
    print(f"wrote sample stamps to {LEVELS_PATH / 'stamps'}")
