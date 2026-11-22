"""
Find places to start the demo over each desert landform: a sand sea, mesas,
a canyon. Prints a ``--lat LAT --lon LON`` for each::

    python scripts/find_landforms.py [--planet desert] [--seed 19]
        [--radius METRES] [--samples N] [--count N]

    python scripts/demo_world.py my_desert_world.yaml --body planet --lat LAT --lon LON

Candidates are random directions, scored by the landform's weight there
(:meth:`Planet.landform_masks`) averaged over a ring 3 km around them, so a
place is well inside the landform, not on its edge. Mesas then prefer the
most relief around them (a wall within 150 m), canyons likewise. ``--count`` places per
landform (3), the best first, at least 50 km apart.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from world_builder.terrain.chart import (
    AzimuthalEquidistantChart,
    dirs_from_latlon,
    latlon,
)
from world_builder.terrain.planet import Planet
from world_builder.world_file import SHIPPED_PLANETS, load_planet

#: Radius of the ring a place's landform weight is averaged over, metres.
RING_M = 3000.0
#: The ring a place's cliff is measured over, metres, and the shortest
#: wavelength of the ground it counts: the tablelands' own walls.
CLIFF_RING_M = 150.0
CLIFF_WAVELENGTH_M = 30.0
#: Candidates kept, the best by landform weight, for the finer tests.
SHORTLIST = 40
#: Least distance between two places found for one landform, metres.
APART_M = 50e3


def _ring(planet: Planet, centre: np.ndarray, radius: float) -> np.ndarray:
    lat, lon = latlon(centre[None, :])
    chart = AzimuthalEquidistantChart(float(lat[0]), float(lon[0]), planet.spec.radius)
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    return chart.to_dirs(radius * np.cos(angles), radius * np.sin(angles))


def _relief(planet: Planet, dirs: np.ndarray) -> np.ndarray:
    """:returns: the ground's steepest rise (highest minus lowest, metres) over
    a ring of :data:`CLIFF_RING_M` around each direction: a cliff or a cut,
    where a mountain slope of the same length is a fraction of it"""
    out = np.empty(len(dirs))
    for i, centre in enumerate(dirs):
        points = np.vstack([centre[None, :], _ring(planet, centre, CLIFF_RING_M)])
        ground, _, _ = planet.sample_dirs(points, CLIFF_WAVELENGTH_M)
        out[i] = np.ptp(ground)
    return out


def _centre_score(planet: Planet, name: str, dirs: np.ndarray) -> np.ndarray:
    """:returns: how good each direction is as the centre of the view: mesas
    and canyons want cliffs and cuts in view (the steepest rise around), dunes need
    nothing more than the sand sea"""
    if name in ("mesas", "canyons"):
        return _relief(planet, dirs)
    return np.ones(len(dirs))


def find(
    planet: Planet, samples: int, count: int
) -> dict[str, list[tuple[float, float, float, float]]]:
    """:returns: per landform, up to *count* places, the best first, at least
    :data:`APART_M` apart: (lat, lon, ground height, score)"""
    rng = np.random.default_rng(planet.spec.seed)
    dirs = rng.normal(size=(samples, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    masks = planet.landform_masks(dirs)
    found = {}
    for name, weight in masks.items():
        shortlist = np.argsort(-weight)[: SHORTLIST * 4]
        shortlist = shortlist[weight[shortlist] > 0.5]
        if len(shortlist) == 0:
            continue
        ring = np.array(
            [
                planet.landform_masks(_ring(planet, dirs[i], RING_M))[name].mean()
                for i in shortlist
            ]
        )
        score = ring * _centre_score(planet, name, dirs[shortlist])
        places = []
        for i in np.argsort(-score):
            centre = dirs[shortlist[i]]
            if score[i] <= 0.0 or any(
                np.arccos(np.clip(centre @ other, -1.0, 1.0)) * planet.spec.radius
                < APART_M
                for other, _ in places
            ):
                continue
            places.append((centre, float(score[i])))
            if len(places) == count:
                break
        found[name] = []
        for centre, value in places:
            lat, lon = latlon(centre[None, :])
            ground, _, _ = planet.sample_dirs(centre[None, :], None, apply_stamps=True)
            found[name].append((float(lat[0]), float(lon[0]), float(ground[0]), value))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--planet",
        default="desert",
        help="a body file, or a shipped one: " + ", ".join(SHIPPED_PLANETS),
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--radius", type=float, default=None, help="metres")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--count", type=int, default=3, help="places per landform")
    args = parser.parse_args()
    spec = replace(load_planet(args.planet), seed=args.seed)
    if args.radius is not None:
        spec = replace(spec, radius=args.radius)
    planet = Planet(spec)
    found = find(planet, args.samples, args.count)
    for name in ("dunes", "mesas", "canyons"):
        if not found.get(name):
            print(f"{name:<8} none found")
            continue
        for lat, lon, ground, _ in found[name]:
            note = f"ground {ground:.0f} m"
            if name != "dunes":
                relief = _relief(planet, dirs_from_latlon(lat, lon))[0]
                note += f", a {relief:.0f} m rise within {CLIFF_RING_M:g} m"
            print(f"{name:<8} --lat {lat:.4f} --lon {lon:.4f}   ({note})")


if __name__ == "__main__":
    main()
