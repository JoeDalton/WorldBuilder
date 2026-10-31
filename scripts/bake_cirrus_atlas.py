"""
Bake the cirrus stamps: a dozen patches of cirrus uncinus (tufts and their
fibrous comet tails, merging), drawn offline from a procedural model, into the
atlas the planet's cirrus sheet stamps along the wind
(world_builder.clouds.cirrus)::

    python scripts/bake_cirrus_atlas.py [--out DIR] [--only N]

    --out    where cirrus_atlas.png / .json go (default: the package's clouds
             directory, where the sheet loads them from)
    --only   bake patch N alone, into the scratch PNG patch_N.png (tuning)

A patch's frame: x along the wind, y across; 24 x 12 km at scale 1. In it:

1. **a flow**: the wind, bent by a stream function ``phi = y + curl * noise``
   (its contours are the flow's lines, so the tails and the fibres curve
   together);
2. **tufts**: 8-20 cottony blobs, 0.4-1.5 km, most of them along a curved
   leading front near +x (downwind), a few scattered upwind;
3. **tails**: each point marches downwind along the flow and takes the soft
   maximum of the tufts it meets, fading as ``exp(-s / tail)``, read from
   ever more blurred tufts the farther it goes, so the tails widen as they
   fall (fall streaks lag their generating cells upwind when the wind rises
   with height);
4. **fibres**: strands in the flow's own coordinates (x along, phi
   across), so they follow every curve: three widths (fine, medium, thick),
   each sparse, with gaps, and of a strength that wanders along and across,
   so no two strands are alike (one width at one contrast read as a brush);
   cut into wisps, more in the tails than on the tufts;
5. an irregular outline, faded to zero at the canvas's border, so a patch
   never shows a rectangle.

Pure numpy and scipy, deterministic. The output is 8-bit grey: thickness,
with the footprint area and the mean thickness of each patch in the JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from panda3d.core import Filename, PNMImage, StringStream
from scipy.ndimage import gaussian_filter, map_coordinates

from world_builder.clouds.cirrus import (
    ATLAS_GRID,
    FOOTPRINT_THRESHOLD,
    STAMP_METRES,
    STAMP_PIXELS,
    value_noise,
)

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent / "src" / "world_builder" / "clouds"
)
#: The tail's march downwind: steps and their length, metres.
TAIL_STEPS = 60
TAIL_STEP = 250.0
#: The soft maximum's exponent: high = each tuft point drags its own streak.
SOFT_MAX = 4.0
#: The tail's thickness next to its tuft, as a share of the tuft's.
TAIL_SHARE = 0.9
#: The tufts' blur (metres) the march reads from beyond each distance
#: (metres): the tails widen as they fall.
WIDENING = ((0.0, 0.0), (1000.0, 180.0), (3000.0, 400.0), (6000.0, 750.0))
#: The faint fibrous veil between the comets, as a share of a tuft's
#: thickness: it joins them into one field.
VEIL = 0.22
#: The fibres' layers: (spacing across, length along, share), metres.
STRANDS = ((45.0, 1400.0, 0.35), (110.0, 2600.0, 0.40), (260.0, 5000.0, 0.25))
#: How far the fibres wave off the flow's lines, metres, and over what length.
FIBRE_WAVE = (250.0, 3000.0)


@dataclass(frozen=True)
class Patch:
    """One stamp's shape.

    :ivar seed: its noise and its tufts' layout
    :ivar tufts: how many tufts
    :ivar tuft_size: (min, max) tuft radius, metres
    :ivar front: share of the tufts on the leading front (the rest scatter
        upwind)
    :ivar front_bend: how far the front curves back at its ends, metres
    :ivar curl: how far the flow's lines wander across, metres
    :ivar tail: the tails' e-folding length, metres
    :ivar fibre: how much the fibres cut the tails, in [0, 1]
    :ivar wisp: how much the tails break into wisps, in [0, 1]
    """

    seed: int
    tufts: int
    tuft_size: tuple
    front: float
    front_bend: float
    curl: float
    tail: float
    fibre: float
    wisp: float


#: The dozen patches: fronts straight and bent, dense and sparse, short and
#: long tails, calm and curling flows.
PATCHES = (
    Patch(1, 14, (500.0, 1300.0), 0.8, 2500.0, 1200.0, 5000.0, 0.75, 0.5),
    Patch(2, 10, (600.0, 1500.0), 0.7, -2000.0, 1800.0, 6000.0, 0.7, 0.6),
    Patch(3, 18, (400.0, 1000.0), 0.85, 3500.0, 900.0, 4000.0, 0.8, 0.5),
    Patch(4, 8, (700.0, 1500.0), 0.6, 0.0, 2200.0, 7000.0, 0.7, 0.6),
    Patch(5, 16, (450.0, 1200.0), 0.75, -3000.0, 1500.0, 5500.0, 0.75, 0.4),
    Patch(6, 12, (500.0, 1400.0), 0.5, 1500.0, 2600.0, 4500.0, 0.8, 0.7),
    Patch(7, 20, (400.0, 900.0), 0.9, 4000.0, 1000.0, 3500.0, 0.8, 0.5),
    Patch(8, 9, (600.0, 1500.0), 0.65, -1000.0, 2000.0, 8000.0, 0.65, 0.5),
    Patch(9, 15, (450.0, 1100.0), 0.8, 2000.0, 1700.0, 5000.0, 0.75, 0.6),
    Patch(10, 11, (550.0, 1400.0), 0.7, -3500.0, 1300.0, 6500.0, 0.7, 0.4),
    Patch(11, 13, (500.0, 1200.0), 0.6, 1000.0, 2400.0, 5000.0, 0.8, 0.6),
    Patch(12, 17, (400.0, 1100.0), 0.85, -2500.0, 1100.0, 4500.0, 0.75, 0.5),
)


def _more_patches(count: int, seed: int) -> tuple[Patch, ...]:
    """Patches drawn within the hand-picked ones' ranges (deterministic)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        small = rng.uniform(380.0, 700.0)
        out.append(
            Patch(
                seed=100 + i,
                tufts=int(rng.integers(8, 21)),
                tuft_size=(small, small * rng.uniform(1.8, 2.6)),
                front=float(rng.uniform(0.5, 0.9)),
                front_bend=float(rng.uniform(-4000.0, 4000.0)),
                curl=float(rng.uniform(900.0, 2600.0)),
                tail=float(rng.uniform(3500.0, 8000.0)),
                fibre=float(rng.uniform(0.65, 0.85)),
                wisp=float(rng.uniform(0.4, 0.7)),
            )
        )
    return tuple(out)


#: Twice as many as the hand-picked dozen, so repetition does not show.
PATCHES = PATCHES + _more_patches(12, 2026)


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fbm(x, y, scale, octaves, seed):
    """Value noise fbm over the plane (the sheet's own hash noise), in [0, 1]."""
    total = np.zeros_like(x)
    weight = 0.0
    for octave in range(octaves):
        f = 2.0**octave / scale
        coords = np.stack([x * f, y * f, np.full_like(x, 17.0 * seed + octave)], -1)
        amplitude = 0.5**octave
        total = total + amplitude * value_noise(coords)
        weight += amplitude
    return total / weight


def _strands(xx, phi, wave, seed):
    """The fibres, in [0, 1]: :data:`STRANDS`' layers, each a sparse set of
    strands (thin, with gaps) whose strength wanders slowly along and across."""
    total = np.zeros_like(xx)
    for i, (spacing, length, share) in enumerate(STRANDS):
        strand = _smoothstep(
            0.42, 0.82, _fbm(xx / length, wave / spacing, 1.0, 1, seed + 10 * i)
        )
        strength = _smoothstep(
            0.2,
            0.8,
            _fbm(xx / 7000.0, phi / (8.0 * spacing), 1.0, 2, seed + 10 * i + 5),
        )
        total += share * strand * (0.15 + 0.85 * strength)
    return np.clip(total / 0.6, 0.0, 1.0)


def patch(c: Patch) -> np.ndarray:
    """One stamp's thickness, (rows, columns) in [0, 1], row 0 at -y."""
    columns, rows = STAMP_PIXELS
    length, width = STAMP_METRES
    pixel = length / columns
    x = ((np.arange(columns) + 0.5) / columns - 0.5) * length
    y = ((np.arange(rows) + 0.5) / rows - 0.5) * width
    xx, yy = np.meshgrid(x, y)
    rng = np.random.default_rng(c.seed)

    # The flow: phi's contours, and its direction (d phi/dy, -d phi/dx)
    phi = yy + 2.0 * c.curl * (_fbm(xx, yy, 9000.0, 2, c.seed + 500) - 0.5)
    dphi_dy, dphi_dx = np.gradient(phi, pixel, pixel)
    norm = np.maximum(np.hypot(dphi_dx, dphi_dy), 1e-6)
    flow_x, flow_y = dphi_dy / norm, -dphi_dx / norm

    # The tufts: a curved front near +x, the rest scattered upwind
    tuft = np.zeros_like(xx)
    front_x = 0.28 * length
    for i in range(c.tufts):
        radius = rng.uniform(*c.tuft_size)
        room = 0.5 * width - 1.6 * radius
        cy = rng.uniform(-room, room)
        if i < round(c.front * c.tufts):
            cx = front_x - c.front_bend * (cy / (0.5 * width)) ** 2
            cx += rng.normal(0.0, 0.5 * radius)
        else:
            cx = rng.uniform(-0.35 * length, front_x - 2.0 * radius)
        cx = min(cx, 0.5 * length - 2.0 * radius)
        # Ragged, and sheared: its upwind side drawn out, curled one way
        warp = 0.9 * (_fbm(xx, yy, 0.7 * radius, 3, c.seed + 100 + i) - 0.5)
        dx, dy = xx - cx, yy - cy
        hook = rng.uniform(-0.6, 0.6)
        dy = dy - hook * np.minimum(dx, 0.0) ** 2 / radius
        stretch = np.where(dx < 0.0, 2.2, 1.0)
        r = np.hypot(dx / (stretch * radius), dy / radius) - warp
        tuft = np.maximum(
            tuft, (1.0 - _smoothstep(0.3, 1.0, r)) * rng.uniform(0.6, 1.0)
        )
    cotton = _smoothstep(0.25, 0.75, _fbm(xx, yy, 400.0, 3, c.seed))
    tuft *= 0.3 + 0.7 * cotton

    # The tails: march downwind along the flow through ever blurrier tufts
    blurred = [
        gaussian_filter(tuft, sigma / pixel) if sigma else tuft for _, sigma in WIDENING
    ]
    px, py = xx.copy(), yy.copy()
    acc = np.zeros_like(xx)
    for k in range(TAIL_STEPS):
        s = k * TAIL_STEP
        level = max(i for i, (start, _) in enumerate(WIDENING) if s >= start)
        at = np.stack([py / pixel + rows / 2 - 0.5, px / pixel + columns / 2 - 0.5])
        sampled = map_coordinates(blurred[level], at, order=1, mode="constant")
        weight = math.exp(-s / c.tail) * (1.0 if k == 0 else TAIL_SHARE)
        acc += (weight * sampled) ** SOFT_MAX
        fx = map_coordinates(flow_x, at, order=1, mode="nearest")
        fy = map_coordinates(flow_y, at, order=1, mode="nearest")
        px, py = px + TAIL_STEP * fx, py + TAIL_STEP * fy
    trail = acc ** (1.0 / SOFT_MAX)

    # Fibres along the flow's lines, waving a little, and wisps
    amplitude, wavelength = FIBRE_WAVE
    wave = phi + amplitude * (
        _fbm(xx / wavelength, phi / wavelength, 1.0, 2, c.seed + 400) - 0.5
    )
    hair = _strands(xx, phi, wave, c.seed + 200)
    wisps = _smoothstep(0.25, 0.7, _fbm(xx / 4000.0, phi / 350.0, 1.0, 2, c.seed + 300))
    tailness = np.clip(1.0 - tuft / np.maximum(trail, 1e-4), 0.0, 1.0)
    # Fibres in the tails, and fraying the tufts' edges a little too
    cut = c.fibre * (0.35 + 0.65 * tailness)
    thickness = trail * (1.0 - cut * (1.0 - hair))
    thickness *= 1.0 - c.wisp * tailness * (1.0 - wisps)
    veil = VEIL * hair * _smoothstep(0.2, 0.8, _fbm(xx, phi, 3000.0, 2, c.seed + 700))
    thickness = np.maximum(thickness, veil)

    # An irregular outline, to zero at the border
    u = (np.arange(columns) + 0.5) / columns
    v = (np.arange(rows) + 0.5) / rows
    border = np.outer(
        _smoothstep(0.0, 0.1, v) * _smoothstep(0.0, 0.1, 1.0 - v),
        # Long on the upwind side: a tail cut by the border shows a straight
        # line across the wind
        _smoothstep(0.0, 0.3, u) * _smoothstep(0.0, 0.06, 1.0 - u),
    )
    outline = _smoothstep(0.3, 0.55, _fbm(xx, yy, 7000.0, 2, c.seed + 600))
    thickness *= border * (0.3 + 0.7 * outline)
    return np.clip(thickness / max(float(thickness.max()), 1e-6), 0.0, 1.0)


def write_grey(path: Path, image: np.ndarray) -> None:
    """An (rows, columns) array in [0, 1] as an 8-bit grey PNG, row 0 at the
    bottom (OpenGL's v = 0), as Panda loads it."""
    rows, columns = image.shape
    pixels = np.round(image[::-1] * 255.0).astype(np.uint8)
    pgm = f"P5 {columns} {rows} 255\n".encode() + pixels.tobytes()
    out = PNMImage()
    if not out.read(StringStream(pgm), "stamp.pgm"):
        raise OSError("could not encode the stamp")
    out.write(Filename.from_os_specific(str(path)))


def bake(out: Path) -> None:
    columns, rows = STAMP_PIXELS
    grid_columns, grid_rows = ATLAS_GRID
    atlas = np.zeros((grid_rows * rows, grid_columns * columns))
    records = []
    for i, c in enumerate(PATCHES):
        stamp = np.round(patch(c) * 255.0) / 255.0
        gc, gr = i % grid_columns, i // grid_columns
        # Row 0 of the atlas array is v = 0 (bottom)
        atlas[gr * rows : (gr + 1) * rows, gc * columns : (gc + 1) * columns] = stamp
        records.append(
            {
                "u_min": gc / grid_columns,
                "v_min": gr / grid_rows,
                "u_size": 1.0 / grid_columns,
                "v_size": 1.0 / grid_rows,
                "footprint": float((stamp > FOOTPRINT_THRESHOLD).mean()),
                "mean_thickness": float(stamp.mean()),
                "patch": asdict(c),
            }
        )
        print(f"stamp {i}: footprint {records[-1]['footprint']:.3f}")
    write_grey(out / "cirrus_atlas.png", atlas)
    with open(out / "cirrus_atlas.json", "w") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()
    if args.only is not None:
        write_grey(args.out / f"patch_{args.only}.png", patch(PATCHES[args.only]))
        return
    bake(args.out)


if __name__ == "__main__":
    main()
