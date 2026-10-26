"""
A field's rocks and statistics, rendered offline to PNG: tune a spec with no
3D.

Per field, a sheet of two maps, looking down the z axis and along the y axis:
every rock as a disc of its radius (grey rock to pale blue ice), over the
density at the map's middle plane (dark: none, lighter: denser), keep-outs
outlined. With matplotlib installed (a development dependency), a second
sheet: the rocks' sizes, spin rates, spin axes and ice, each against the
spec's distribution::

    python -m world_builder.asteroids.preview [--preset NAME ...] [--out DIR]
        [--size PX] [--slab M]

    --preset   the fields (spec.PRESETS; default all)
    --out      where the sheets go (default ./asteroid_preview)
    --size     each map's side, pixels (default 900)
    --slab     draw only rocks within this distance of the middle plane
               (default: all)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from world_builder.asteroids.scatter import Scatter
from world_builder.asteroids.spec import PRESETS, AroundAxis, FieldSpec
from world_builder.terrain.pngio import write_png

_ROCK = np.array([0.55, 0.52, 0.48])
_ICE = np.array([0.75, 0.88, 1.0])
_KEEP_OUT = np.array([1.0, 0.35, 0.2])


def _map(
    scatter: Scatter,
    rocks: np.ndarray,
    axes: tuple[int, int],
    size: int,
    slab: float | None,
) -> np.ndarray:
    """:returns: (size, size, 3) uint8: the rocks seen along the third axis"""
    lo, hi = scatter.spec.shape.bounds()
    extent = float(np.max(hi - lo)) * 1.05
    centre = 0.5 * (lo + hi)
    a, b = axes
    depth = 3 - a - b
    scale = size / extent
    # The density at the middle plane, as the background.
    pixels = (np.arange(size) + 0.5) / scale - 0.5 * extent
    u, v = np.meshgrid(pixels + centre[a], pixels[::-1] + centre[b])
    points = np.zeros((size * size, 3))
    points[:, a], points[:, b], points[:, depth] = u.ravel(), v.ravel(), centre[depth]
    density = scatter.density(points).reshape(size, size)
    image = (0.04 + 0.16 * density)[:, :, None] * np.ones(3)
    if slab is not None:
        rocks = rocks[np.abs(rocks["centre"][:, depth] - centre[depth]) < slab]
    # Far ones first, so near ones cover them.
    order = np.argsort(rocks["centre"][:, depth])
    for rock in rocks[order]:
        x = (rock["centre"][a] - centre[a] + 0.5 * extent) * scale
        y = (0.5 * extent - (rock["centre"][b] - centre[b])) * scale
        r = max(rock["radius"] * scale, 0.7)
        x0, x1 = int(max(0, x - r - 1)), int(min(size, x + r + 2))
        y0, y1 = int(max(0, y - r - 1)), int(min(size, y + r + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r
        colour = _ROCK + (_ICE - _ROCK) * rock["ice"]
        image[y0:y1, x0:x1][inside] = colour
    # Keep-outs: the outline of their projection.
    for keep in scatter.spec.keep_outs:
        start = np.asarray(keep.centre)[[a, b]]
        end = start if keep.end is None else np.asarray(keep.end)[[a, b]]
        axis = end - start
        length2 = float(axis @ axis)
        d = np.stack((u - start[0], v - start[1]), axis=-1)
        if length2 > 0.0:
            t = np.clip((d @ axis) / length2, 0.0, 1.0)
            d = d - t[..., None] * axis
        distance = np.sqrt(np.sum(d * d, axis=-1))
        image[np.abs(distance - keep.radius) < 0.75 / scale] = _KEEP_OUT
    return (np.clip(image, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.uint8)


def _statistics(spec: FieldSpec, rocks: np.ndarray, path: Path) -> bool:
    """The rocks' distributions against the spec's, to *path*.

    :returns: False without matplotlib
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    sizes = spec.sizes
    bins = np.geomspace(sizes.lo, sizes.hi, 40)
    ax[0, 0].hist(rocks["radius"], bins=bins, density=True, alpha=0.6, label="rocks")
    x = np.geomspace(sizes.lo, sizes.hi, 400)
    ax[0, 0].plot(x, sizes.pdf(x), "k", label="spec")
    ax[0, 0].set(xscale="log", yscale="log", xlabel="radius (m)", title="sizes")
    ax[0, 0].legend()
    rate = spec.spin.rate
    base = rocks["precession"] * (rocks["radius"] / sizes.lo) ** spec.spin.size_exponent
    ax[0, 1].hist(base, bins=40, density=True, alpha=0.6, label="rocks, scaled back")
    x = np.linspace(rate.lo, rate.hi, 400)
    ax[0, 1].plot(x, rate.pdf(x), "k", label="spec")
    ax[0, 1].set(xlabel="rate of the smallest (rad/s)", title="spin rates")
    ax[0, 1].legend()
    axis = spec.spin.axis
    reference = (
        np.asarray(axis.axis) / np.linalg.norm(axis.axis)
        if isinstance(axis, AroundAxis)
        else np.array([0.0, 0.0, 1.0])
    )
    cosines = rocks["axis"] @ reference
    ax[1, 0].hist(cosines, bins=40, range=(-1, 1), density=True, alpha=0.6)
    tumbling = np.mean(rocks["spin"] != 0.0)
    ax[1, 0].set(
        xlabel="cos(axis, reference)",
        title=f"spin axes ({100 * tumbling:.0f} % tumble)",
    )
    ice = spec.material.ice_fraction
    if ice.hi > ice.lo:
        heights, _, _ = ax[1, 1].hist(
            rocks["ice"], bins=40, density=True, alpha=0.6, label="rocks"
        )
        x = np.linspace(ice.lo, ice.hi, 400)
        ax[1, 1].plot(x, ice.pdf(x), "k", label="spec")
        # a U-shaped density soars at the ends: keep the histogram readable
        ax[1, 1].set_ylim(0.0, 1.5 * max(heights.max(), 1e-9))
        ax[1, 1].legend()
    else:
        ax[1, 1].text(0.5, 0.5, f"ice fraction {ice.lo:g}", ha="center")
    ax[1, 1].set(xlabel="ice fraction", title="materials")
    fig.suptitle(f"{len(rocks)} rocks")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", nargs="+", choices=list(PRESETS), default=None)
    parser.add_argument("--out", type=Path, default=Path("asteroid_preview"))
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--slab", type=float, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.preset or list(PRESETS):
        spec = PRESETS[name]
        scatter = Scatter(spec)
        start = time.perf_counter()
        lo, hi = spec.shape.bounds()
        rocks = scatter.rocks_in_box(lo - 1.0, hi + 1.0)
        elapsed = (time.perf_counter() - start) * 1e3
        top = _map(scatter, rocks, (0, 1), args.size, args.slab)
        side = _map(scatter, rocks, (0, 2), args.size, args.slab)
        gap = np.full((args.size, 8, 3), 255, dtype=np.uint8)
        write_png(args.out / f"{name}_maps.png", np.hstack((top, gap, side)))
        stats = _statistics(spec, rocks, args.out / f"{name}_stats.png")
        summary = ", ".join(f"{k} {v:.4g}" for k, v in spec.summary().items())
        print(
            f"{name}: {len(rocks)} rocks in {elapsed:.0f} ms "
            f"({'native' if scatter.native else 'numpy'}); {summary}"
            + ("" if stats else "; no matplotlib, no statistics sheet")
        )


if __name__ == "__main__":
    main()
