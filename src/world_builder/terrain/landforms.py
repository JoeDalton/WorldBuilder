"""
Desert landforms, as pure functions of position: dunes and tablelands.

They are shaped to stay band limited, as the rest of the elevation: a coarse
chunk gets a smoothed version, never aliased steps.

* **Dunes**: Sandastray's sand sea (a Panda3D game by entikan): nine
  stretched noise layers at sizes from 40 m to 1.4 km, combined in order by
  min, add and multiply. The min of two layers makes a sharp crest, so the
  dunes are rounded with knife-edge ridges, at several scales, all with one
  grain. The grain is a fixed axis of the planet frame, so on the ground it
  turns slowly from region to region, without a singular point anywhere.
* **Tablelands**: a layered plateau dissected by erosion, from young
  country (narrow canyons between broad tables) to old (wide valleys and
  plains, with isolated mesas, buttes and spires). The tables are the cells
  of a warped Voronoi noise (:func:`cells`); the valleys between them widen
  with the erosion stage, whole cells go to make plains, and small cells keep
  spires there. The walls' cross-section comes from the planet's rock strata
  (:mod:`strata`): cliffs in hard layers, slopes in soft ones, a ledge on
  each hard layer's top.
"""

from __future__ import annotations

import itertools

import numpy as np

from world_builder.terrain.noise import (
    _hash,
    _mix_seed,
    fbm,
    octave_frame,
    octave_weight,
    simplex3,
    smoothstep,
    warp,
)
from world_builder.terrain.strata import Strata

_MASK21 = np.uint64((1 << 21) - 1)

#: Sandastray's dune layers, in order: (period along the grain, period across
#: it, amount, op), periods in metres. Its images are 1 m a pixel, and a
#: perlin_noise_fill(sx, sy) layer has a period of sx * 8192 px, its (x, y)
#: scales times 168 m here. As PNMImage does it, with h starting at 1 and
#: clamped to [0, 1] after each: "min" is h = min(h, 1 - amount (1 - n))
#: (darken_sub_image), "add" h += amount n, "mult" h *= amount n.
DUNE_LAYERS = (
    (1428.0, 840.0, 0.8, "min"),
    (336.0, 302.0, 0.8, "add"),
    (134.0, 168.0, 0.5, "min"),
    (101.0, 84.0, 0.5, "min"),
    (67.0, 42.0, 0.005, "add"),
    (76.0, 92.0, 0.4, "min"),
    (504.0, 470.0, 0.2, "min"),
    (44.0, 76.0, 0.005, "add"),
    (336.0, 470.0, 1.0, "mult"),
)
#: Simplex frequency, per metre of Perlin period, giving the same feature
#: size (their autocorrelations first cross zero at 0.5 lattice unit and at
#: 0.757 period), and the gain giving Panda's Perlin spread (std 0.16 against
#: the simplex's 0.388) about its mean, 0.5.
_DUNE_FREQUENCY = 0.5 / 0.757
_DUNE_GAIN = 0.16 / 0.388
#: How round the crests are, in units of the dunes' height: Sandastray blurs
#: its image by 3 px; this smooth minimum does the same to each crest.
DUNE_CREST = 0.015
#: How much each layer, fading out after the finer ones, lowers the mean
#: height (units of the dunes' height): added back as it fades, so a coarse
#: chunk keeps the right average. Measured by :func:`dune_layer_offsets`
#: (tested).
DUNE_OFFSETS = (
    -0.0243,
    -0.0083,
    -0.0621,
    -0.0200,
    0.0,
    -0.0034,
    -0.0328,
    0.0,
    -0.0006,
)


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """min(a, b), rounded where they are within *k* of each other."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - 0.25 * k * h * h


def _dune_stack(
    points: np.ndarray,
    scale: float,
    seed: int,
    weights: tuple[float, ...],
) -> np.ndarray:
    """Sandastray's layers combined, each layer *weights*[i] of the way from
    neutral (0, faded out) to itself (1).

    :returns: (N,) in [0, 1]
    """
    axes, _ = octave_frame(seed, 0)
    along, across, normal = (points @ axes.T).T
    h = np.ones(len(points))
    for index, ((p_along, p_across, amount, op), w) in enumerate(
        zip(DUNE_LAYERS, weights)
    ):
        if op == "min" and w <= 0.0:
            continue
        f_along = _DUNE_FREQUENCY / (p_along * scale)
        f_across = _DUNE_FREQUENCY / (p_across * scale)
        coords = np.column_stack(
            [along * f_along, across * f_across, normal * f_across]
        )
        n = 0.5 + _DUNE_GAIN * simplex3(coords, seed + 1 + index)
        if op == "min":
            # Panda's darken_sub_image scales the layer toward white.
            low = 1.0 - amount * (1.0 - n)
            h = h + w * (_smooth_min(h, low, DUNE_CREST) - h)
        elif op == "add":
            h = h + amount * (0.5 + w * (n - 0.5))
        else:
            h = h * amount * (0.5 + w * (n - 0.5))
        h = np.clip(h, 0.0, 1.0)
    return h


def dunes(
    points: np.ndarray,
    scale: float,
    height: float,
    seed: int,
    min_wavelength: float | None,
) -> np.ndarray:
    """Dune relief, in metres, from 0 to *height*.

    :param points: (N, 3) positions, metres (warped, as the other fields)
    :param scale: times Sandastray's layer sizes
    :param height: the relief's full range, metres
    :param seed: field seed
    :param min_wavelength: shortest wavelength to keep, metres; None = all.
        Each layer fades out (to neutral, its mean effect added back) as its
        shorter period gets too short to sample
    :returns: (N,) heights
    """
    max_frequency = None if min_wavelength is None else 1.0 / min_wavelength
    weights = tuple(
        octave_weight(1.0 / (min(p_along, p_across) * scale), max_frequency)
        for p_along, p_across, _, _ in DUNE_LAYERS
    )
    h = _dune_stack(points, scale, seed, weights)
    h = h + sum((1.0 - w) * offset for w, offset in zip(weights, DUNE_OFFSETS))
    return height * h


def dune_layer_offsets(samples: int = 200000, seed: int = 5) -> tuple[float, ...]:
    """Measure :data:`DUNE_OFFSETS`. Layers fade finest first, so each
    layer's offset is the mean height lost when it fades after every finer
    one has: once all have faded, the offsets add up to the full mean.

    :returns: one offset per layer (in :data:`DUNE_LAYERS` order), units of
        the dunes' height
    """
    points = np.random.default_rng(seed).uniform(-2e5, 2e5, size=(samples, 3))
    order = sorted(range(len(DUNE_LAYERS)), key=lambda i: min(DUNE_LAYERS[i][:2]))
    weights = [1.0] * len(DUNE_LAYERS)
    mean = _dune_stack(points, 1.0, seed, tuple(weights)).mean()
    offsets = [0.0] * len(DUNE_LAYERS)
    for index in order:
        weights[index] = 0.0
        faded = _dune_stack(points, 1.0, seed, tuple(weights)).mean()
        offsets[index] = float(mean - faded)
        mean = faded
    return tuple(offsets)


def cells(
    points: np.ndarray, size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cellular (Voronoi) noise: one jittered site per lattice cube.

    :param points: (N, 3) positions, metres
    :param size: lattice spacing (the cells' mean size), metres
    :param seed: field seed
    :returns: (N,) each: the distance to the nearest site and to the edge of
        its cell (the bisector with the second nearest), metres, and the
        nearest site's id, uniform in [0, 1)
    """
    u = points / size
    base = np.floor(u).astype(np.int64)
    mixed = _mix_seed(seed)
    offsets = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
    cell = base[:, None, :] + offsets[None, :, :]  # (N, 27, 3)
    h = _hash(cell[..., 0].ravel(), cell[..., 1].ravel(), cell[..., 2].ravel(), mixed)
    jitter = np.stack(
        [
            ((h >> np.uint64(shift)) & _MASK21).astype(np.float64)
            for shift in (0, 21, 42)
        ],
        axis=1,
    ).reshape(cell.shape)
    site = cell + jitter / float(1 << 21)
    d2 = np.einsum("nkj,nkj->nk", site - u[:, None, :], site - u[:, None, :])
    nearest = np.argmin(d2, axis=1)
    rows = np.arange(len(points))
    best1 = d2[rows, nearest]
    site1 = site[rows, nearest]
    # Distance to the cell's edge: to the nearest bisector plane with any
    # other site (the second nearest alone jumps where the second and third
    # swap).
    gap = np.linalg.norm(site - site1[:, None, :], axis=2)
    to_bisector = (d2 - best1[:, None]) / (2.0 * np.maximum(gap, 1e-12))
    to_bisector[rows, nearest] = np.inf
    edge = to_bisector.min(axis=1)
    cell1 = cell[rows, nearest]
    ident = _hash(cell1[:, 0], cell1[:, 1], cell1[:, 2], _mix_seed(seed + 1))
    ident = (ident >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.sqrt(best1) * size, edge * size, ident


def tablelands(
    points: np.ndarray,
    floor: np.ndarray,
    weight: np.ndarray,
    stage: np.ndarray,
    stack: np.ndarray,
    strata: Strata,
    cell: float,
    spire_cell: float,
    top_relief: float,
    seed: int,
    min_wavelength: float | None,
) -> np.ndarray:
    """Tables, valleys and spires cut into the strata, as heights above the
    valley floor.

    :param points: (N, 3) positions, metres (warped, as the other fields)
    :param floor: (N,) the valley floor's altitude, metres
    :param weight: (N,) how much tableland there is, in [0, 1] (scales the
        stack, so tables sink into the plain at a region's edge)
    :param stage: (N,) erosion stage in [0, 1]: 0 narrow canyons between
        broad tables, 1 plains with a few mesas, buttes and spires
    :param stack: (N,) height of the tables above the floor, metres
    :param strata: the planet's layers (the walls' cross-section)
    :param cell: the tables' size, metres
    :param spire_cell: the spacing of spires and buttes, metres
    :param top_relief: the gentle hills on the tables' tops, metres
    :param seed: field seed
    :param min_wavelength: shortest wavelength to keep, metres; None = all.
        Coarser, the walls' cross-section is averaged over the sampling
        (a tent), so cliffs blur instead of stepping
    :returns: (N,) metres above *floor*, at least 0
    """
    # Cell edges bend, at two scales (each warp under the 0.06 ratio that
    # keeps space from folding).
    q = warp(points, 5.0 * cell, 0.25 * cell, 2, seed)
    q = warp(q, 1.5 * cell, 0.08 * cell, 2, seed + 1)

    # Tables: the valleys between them widen with the stage, their walls
    # crenellated; as erosion goes on, whole tables go (their walls retreat
    # to nothing once the cell's id falls under the stage's share).
    centre, edge, ident = cells(q, cell, seed + 2)
    # Old tables round off: their cell, cut by a disc about its site.
    rounded = np.minimum(edge, 0.45 * cell - centre + (1.0 - stage) * cell)
    width = 60.0 + (0.15 * cell - 60.0) * stage
    crenel = np.clip(fbm(q, 600.0, 2, seed + 3, min_wavelength=min_wavelength), -1, 1)
    gone = smoothstep(-0.05, 0.05, 0.6 * stage**2 - ident)
    # Crenellated walls: never more than the valley's half-width (so a
    # valley stays open) nor steep enough to undercut the strata's slopes.
    crenel = 0.8 * np.minimum(width, 60.0) * crenel
    s = rounded - width + crenel - 2.0 * cell * gone

    # Spires and buttes: small cells keep one, only where a table has gone
    # and away from its cell's edge (so no spire is cut by it). Computed
    # there only: elsewhere they would stand below the ground.
    away = gone * smoothstep(150.0, 300.0, edge)
    on = away > 0.0
    if np.any(on):
        f1, small_edge, small_ident = cells(q[on], spire_cell, seed + 4)
        radius = 150.0 + 250.0 * np.modf(small_ident * 4096.0)[0]
        kept = smoothstep(-0.05, 0.05, 0.35 * stage[on] - small_ident)
        spire = (
            np.minimum(radius - f1, 0.5 * small_edge)
            - 2.0 * spire_cell * (1.0 - kept)
            - 2.0 * cell * (1.0 - away[on])
        )
        s[on] = np.maximum(s[on], spire)

    # The tops: the stack's height, gently hilly above it.
    hills = 0.5 + 0.5 * np.clip(
        fbm(q, 1500.0, 3, seed + 5, min_wavelength=min_wavelength), -1.0, 1.0
    )
    ceiling = floor + weight * (stack + top_relief * hills)
    # Debris at the walls' foot, up to a fifth of their height.
    talus = 0.2 * weight * stack
    if not min_wavelength:
        return strata.profile(floor, s, ceiling, talus) - floor
    r = 0.25 * min_wavelength
    height = (
        0.25 * strata.profile(floor, s - r, ceiling, talus)
        + 0.5 * strata.profile(floor, s, ceiling, talus)
        + 0.25 * strata.profile(floor, s + r, ceiling, talus)
    )
    return height - floor
