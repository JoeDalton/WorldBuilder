"""
Compare two folders of screenshots, image by image (same file names)::

    python scripts/compare_shots.py BEFORE AFTER [--diff DIR]

Per image: identical, or the number of pixels that differ and the largest
difference (in 8-bit steps). ``--diff`` writes each differing pair's
difference, amplified 16 times, to DIR. Exits 1 if any image differs or is
missing from AFTER, so a shader refactor meant to change nothing can be
checked with ``scripts/terrain_shots.py`` before and after.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from world_builder.terrain.pngio import read_image, write_png


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--diff", type=Path, default=None)
    args = parser.parse_args()
    if args.diff is not None:
        args.diff.mkdir(parents=True, exist_ok=True)
    differ = False
    for path in sorted(args.before.glob("*.png")):
        other = args.after / path.name
        if not other.exists():
            print(f"{path.name:<20} missing")
            differ = True
            continue
        a = np.rint(read_image(path) * 255.0).astype(np.int16)
        b = np.rint(read_image(other) * 255.0).astype(np.int16)
        if a.shape != b.shape:
            print(f"{path.name:<20} sizes differ {a.shape} {b.shape}")
            differ = True
            continue
        delta = np.abs(a - b).max(axis=2)
        changed = int(np.count_nonzero(delta))
        if changed == 0:
            print(f"{path.name:<20} identical")
            continue
        differ = True
        share = 100 * changed / delta.size
        print(
            f"{path.name:<20} {changed} pixels differ ({share:.3f} %),"
            f" largest by {int(delta.max())}"
        )
        if args.diff is not None:
            amplified = np.clip(delta * 16, 0, 255).astype(np.uint8)
            write_png(args.diff / path.name, amplified)
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
