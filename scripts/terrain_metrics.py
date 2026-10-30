"""
Numbers for how realistic a planet's relief looks, to compare before and after
a change to the elevation (issue #9)::

    python scripts/terrain_metrics.py [--preset NAME ...] [--seed N ...]
        [--radius M] [--patches N]

Per preset and seed, on patches picked at random (the same ones every run):

- **flat**: share of land patches at 100 m spacing whose slope is under 3
  degrees; plains should be flat
- **slope p50 / p95**: land slope percentiles at 100 m spacing, degrees
- **coast D**: box-count dimension of the coastline on 64 km patches; 1 is
  a straight line, about 1.25 a rugged (Norway-like) coast
- **peaks**: summits per 100 km^2 of mountain ground (over 1500 m) that
  stand 300 m above everything within 2 km
- **land p10 / p50 / p90**: land elevation percentiles over the globe, metres
- **lowland**: share of land under 200 m
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
from scipy import ndimage

from world_builder.terrain import biomes
from world_builder.terrain.chart import AzimuthalEquidistantChart, latlon
from world_builder.terrain.planet import PRESETS, Planet

#: Slope under which ground counts as flat, degrees.
FLAT_SLOPE = 3.0
#: Ground over this is mountain, for the peak count, metres.
MOUNTAIN_ABOVE = 1500.0


def _random_dirs(rng: np.random.Generator, n: int) -> np.ndarray:
    dirs = rng.normal(size=(n, 3))
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def _patch(planet: Planet, centre: np.ndarray, spacing: float, n: int) -> np.ndarray:
    """:returns: (n, n) elevation on a chart around *centre*, full detail
    down to the spacing"""
    lat, lon = latlon(centre[None, :])
    chart = AzimuthalEquidistantChart(float(lat[0]), float(lon[0]), planet.spec.radius)
    coords = spacing * (np.arange(n) - n // 2)
    gx, gy = np.meshgrid(coords, coords)
    elevation, _, _ = planet.sample_dirs(chart.to_dirs(gx, gy), 4.0 * spacing)
    return elevation.reshape(n, n)


def _box_dimension(mask: np.ndarray) -> float:
    """Box-count dimension of the True pixels of a square mask."""
    sizes, counts = [], []
    for s in (2, 4, 8, 16, 32):
        m = mask.shape[0] // s * s
        boxes = mask[:m, :m].reshape(m // s, s, m // s, s).any(axis=(1, 3))
        if boxes.sum() > 0:
            sizes.append(s)
            counts.append(boxes.sum())
    if len(sizes) < 3:
        return float("nan")
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


def measure(planet: Planet, seed: int, patches: int) -> dict[str, float]:
    """:returns: the metrics of one planet (see the module docstring)"""
    rng = np.random.default_rng(seed)
    dirs = _random_dirs(rng, 40000)
    coarse, _, _ = planet.sample_dirs(dirs, 2000.0)
    land = coarse[coarse > 0.0]
    out = {
        "land": float(np.mean(coarse > 0.0)),
        "land p10": float(np.percentile(land, 10)) if land.size else np.nan,
        "land p50": float(np.percentile(land, 50)) if land.size else np.nan,
        "land p90": float(np.percentile(land, 90)) if land.size else np.nan,
        "lowland": float(np.mean(land < 200.0)) if land.size else np.nan,
    }

    # Slopes on land patches.
    flat, slopes = [], []
    for centre in dirs[coarse > 50.0][:patches]:
        z = _patch(planet, centre, 100.0, 64)
        s = biomes.slope_degrees(z, 100.0)[1:-1, 1:-1]
        s = s[z[1:-1, 1:-1] > 0.0]
        if s.size:
            slopes.append(s)
            flat.append(np.median(s) < FLAT_SLOPE)
    if slopes:
        everything = np.concatenate(slopes)
        out["flat"] = float(np.mean(flat))
        out["slope p50"] = float(np.percentile(everything, 50))
        out["slope p95"] = float(np.percentile(everything, 95))

    # Coastlines, where the coarse field crosses sea level.
    dims = []
    if planet.spec.has_ocean or np.any(coarse < 0.0):
        near = dirs[np.abs(coarse) < 60.0][:patches]
        for centre in near:
            z = _patch(planet, centre, 250.0, 256)
            wet = z < 0.0
            edge = wet ^ ndimage.binary_erosion(wet, border_value=1)
            if 0.05 < wet.mean() < 0.95:
                dims.append(_box_dimension(edge))
    out["coast D"] = float(np.nanmean(dims)) if dims else np.nan

    # Peaks standing out in mountain ground.
    peaks, area = 0, 0.0
    for centre in dirs[coarse > MOUNTAIN_ABOVE][:patches]:
        z = _patch(planet, centre, 100.0, 128)
        top = z == ndimage.maximum_filter(z, size=11)
        relief = z - ndimage.minimum_filter(z, size=41)
        high = z > MOUNTAIN_ABOVE
        inner = np.zeros_like(high)
        inner[20:-20, 20:-20] = True
        peaks += int(np.sum(top & (relief > 300.0) & high & inner))
        area += np.sum(high & inner) * 0.01  # km^2 at 100 m
    out["peaks"] = 100.0 * peaks / area if area > 0.0 else np.nan
    return out


COLUMNS = (
    "land",
    "flat",
    "slope p50",
    "slope p95",
    "coast D",
    "peaks",
    "land p10",
    "land p50",
    "land p90",
    "lowland",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", nargs="*", default=list(PRESETS))
    parser.add_argument("--seed", type=int, nargs="*", default=[1, 19])
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--patches", type=int, default=24)
    args = parser.parse_args()

    print(f"{'preset':<10} {'seed':>4} " + " ".join(f"{c:>9}" for c in COLUMNS))
    for name in args.preset:
        for seed in args.seed:
            spec = replace(PRESETS[name], seed=seed)
            if args.radius:
                spec = replace(spec, radius=args.radius)
            start = time.perf_counter()
            m = measure(Planet(spec), seed, args.patches)
            cells = " ".join(f"{m.get(c, np.nan):>9.3g}" for c in COLUMNS)
            print(
                f"{name:<10} {seed:>4} {cells}  ({time.perf_counter() - start:.0f} s)"
            )


if __name__ == "__main__":
    main()
