"""
Terrain generation benchmark: noise, chunks per level, LOD pass, worker throughput.

    python -m world_builder.terrain.bench [--workers N] [--radius M]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.chunkgen import VERTS, generate_chunk, planet_for
from world_builder.terrain.noise import simplex3
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.quadtree import LodSelector
from world_builder.terrain.workers import ChunkExecutor, default_workers

#: Camera altitudes the LOD pass is timed at, metres.
LOD_ALTITUDES = (2.0, 1000.0, 20000.0, 1200e3)


def _best_ms(fn, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1e3)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--radius", type=float, default=None, help="metres")
    args = parser.parse_args()
    spec = replace(PRESETS["earthlike"], seed=19)
    if args.radius is not None:
        spec = replace(spec, radius=args.radius)
    spec = spec.resolved()
    grid = planet_for(spec).grid
    print(
        f"planet radius {spec.radius / 1e3:g} km, levels 0-{grid.max_level}, "
        f"finest spacing {grid.spacing0:.2f} m"
    )

    points = np.random.default_rng(0).uniform(-1e5, 1e5, size=((VERTS + 8) ** 2, 3))
    noise_ms = _best_ms(lambda: simplex3(points, 1))
    print(f"simplex3, {len(points)} points: {noise_ms:.2f} ms")

    print("chunk generation (median of 5):")
    for level in sorted({0, 2, 4, 6, 8, 10, grid.max_level}):
        middle = grid.side(level) // 2
        times = [
            generate_chunk(
                spec, (level, 0, min(middle + i, grid.side(level) - 1), middle)
            ).gen_ms
            for i in range(5)
        ]
        print(f"  level {level:2d}: {np.median(times):5.1f} ms")

    selector = LodSelector(grid)
    up = TangentFrame.at(40.0, 20.0).up
    for altitude in LOD_ALTITUDES:
        camera = up * (spec.radius + altitude)

        def select(camera=camera):
            return selector.select(
                camera, ready=lambda k: True, bounds=lambda k: (0.0, 500.0)
            )

        select()  # build the per-key boxes once, as a flying camera would have
        ms = _best_ms(select)
        count = len(select().draw)
        print(f"LOD pass at {altitude:>9,.0f} m: {count:4d} chunks, {ms:.1f} ms")

    print(f"process pool, {args.workers} workers, 120 chunks:")
    executor = ChunkExecutor(backend="process", workers=args.workers, max_inflight=240)
    executor.set_spec(spec)
    executor.request([(0, 0, 0, 0)])  # warm up: spawning workers takes a moment
    while not executor.collect():
        time.sleep(0.01)
    keys = [(0, 0, 1000 + i % 12, 1000 + i // 12) for i in range(120)]
    start = time.perf_counter()
    executor.request(keys)
    done = 0
    while done < len(keys):
        done += len(executor.collect())
        time.sleep(0.002)
    elapsed = time.perf_counter() - start
    print(f"  {len(keys) / elapsed:.0f} chunks/s")
    executor.shutdown()


if __name__ == "__main__":
    main()
