"""
Terrain generation benchmark: noise, chunks per level, LOD pass, worker throughput.

    python -m world_builder.terrain.bench [--workers N]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

from world_builder.terrain.chunkgen import (
    MAX_LEVEL,
    VERTS,
    generate_chunk,
    planet_for,
)
from world_builder.terrain.noise import simplex3
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.quadtree import LodSelector
from world_builder.terrain.workers import ChunkExecutor, default_workers


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
    args = parser.parse_args()
    spec = replace(PRESETS["earthlike"], seed=19).resolved()
    planet_for(spec)

    points = np.random.default_rng(0).uniform(-1e5, 1e5, size=((VERTS + 8) ** 2, 3))
    noise_ms = _best_ms(lambda: simplex3(points, 1))
    print(f"simplex3, {len(points)} points: {noise_ms:.2f} ms")

    print("chunk generation (median of 5):")
    for level in (0, 2, 4, 6, 8, 10, MAX_LEVEL):
        times = [generate_chunk(spec, (level, i, 3)).gen_ms for i in range(5)]
        print(f"  level {level:2d}: {np.median(times):5.1f} ms")

    selector = LodSelector(spec.radius, max_distance=157500.0)
    for altitude in (10.0, 1000.0, 20000.0):
        ms = _best_ms(
            lambda a=altitude: selector.select(
                (1234.0, -5678.0, a),
                ready=lambda k: True,
                bounds=lambda k: (0.0, 500.0),
            )
        )
        count = len(
            selector.select(
                (1234.0, -5678.0, altitude),
                ready=lambda k: True,
                bounds=lambda k: (0.0, 500.0),
            ).draw
        )
        print(f"LOD pass at {altitude:>7.0f} m: {count} chunks, {ms:.1f} ms")

    print(f"process pool, {args.workers} workers, 120 chunks:")
    executor = ChunkExecutor(backend="process", workers=args.workers, max_inflight=240)
    executor.set_spec(spec)
    executor.request([(0, 0, 0)])  # warm up: spawning workers takes a moment
    while not executor.collect():
        time.sleep(0.01)
    keys = [(0, i % 12, i // 12) for i in range(120)]
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
