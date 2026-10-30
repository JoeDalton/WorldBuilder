"""
Desert landforms, as pure functions of position: dunes, mesas, terraces, canyons.

They are shaped to stay band limited, as the rest of the elevation: a coarse
chunk gets a smoothed version, never aliased steps.

* **Dunes**: Sandastray's sand sea (a Panda3D game by entikan): nine
  stretched noise layers at sizes from 40 m to 1.4 km, combined in order by
  min, add and multiply. The min of two layers makes a sharp crest, so the
  dunes are rounded with knife-edge ridges, at several scales, all with one
  grain. The grain is a fixed axis of the planet frame, so on the ground it
  turns slowly from region to region, without a singular point anywhere.
* **Terraces**: a staircase of the elevation, flat treads and steep risers,
  faded in as the sampling gets fine enough to show the risers.
* **Mesas**: flat-topped tables with cliff edges, from a thresholded noise.
* **Canyons**: flat-floored, steep-walled cuts along the zero lines of a
  noise.
"""

from __future__ import annotations

import numpy as np

from world_builder.terrain.noise import (
    fbm,
    octave_frame,
    octave_weight,
    simplex3,
    smoothstep,
)

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


def terrace(elevation: np.ndarray, step: float, riser: float = 0.35) -> np.ndarray:
    """A staircase of *elevation*: flat treads, risers over *riser* of each step.

    :param elevation: (N,) metres
    :param step: height of a step, metres
    :param riser: share of each step's height range the riser takes
    :returns: (N,) terraced elevation, within half a step of the input
    """
    level = elevation / step
    floor = np.floor(level)
    rise = smoothstep(0.5 - 0.5 * riser, 0.5 + 0.5 * riser, level - floor)
    return step * (floor + rise)


def terrace_fade(step: float, min_wavelength: float | None) -> float:
    """:returns: how much of the terraces to show, 1 down to 0 as the
    sampling gets too coarse for their risers (a few steps wide)"""
    if min_wavelength is None:
        return 1.0
    return float(1.0 - smoothstep(1.5 * step, 6.0 * step, min_wavelength))


#: Share of the ground a mesa noise tops (n above :data:`MESA_LEVEL`), and
#: of a canyon region a canyon covers (abs(n) under :data:`CANYON_LEVEL`),
#: for a 3-octave fbm (measured over 400k points; the same for any seed).
MESA_LEVEL, MESA_SHARE = 0.145, 0.285
CANYON_LEVEL, CANYON_SHARE = 0.04, 0.125


def _band_limited_step(
    n: np.ndarray,
    level: float,
    width: float,
    metres: float,
    share: float,
    min_wavelength: float | None,
) -> np.ndarray:
    """A smooth step of a noise *n* at *level*, 0 below and 1 above, as a
    coarse chunk should see it.

    :param width: the step's width, in units of n
    :param metres: about how wide that is on the ground
    :param share: the share of the ground above the step (its mean)
    :param min_wavelength: shortest wavelength to keep, metres
    :returns: in [0, 1]. Coarser than the step, it widens (up to 4 times),
        then fades to its mean, so coarse chunks keep the right average
        height instead of aliasing a cliff or losing the feature.
    """
    blur = 1.0 if min_wavelength is None else max(1.0, min_wavelength / metres)
    half = 0.5 * width * min(blur, 4.0)
    step = smoothstep(level - half, level + half, n)
    fade = float(smoothstep(4.0, 16.0, np.float64(blur)))
    return step + fade * (share - step)


def mesa_tops(
    points: np.ndarray, wavelength: float, seed: int, min_wavelength: float | None
) -> np.ndarray:
    """Flat-topped tables: 1 on a mesa, 0 around it, cliffs between.

    :param points: (N, 3) positions, metres
    :param wavelength: size of the mesas' noise, metres
    :param seed: field seed
    :param min_wavelength: shortest wavelength to keep, metres
    :returns: (N,) in [0, 1]; mesas cover about 30 % of the ground, their
        cliffs about 1/40 of the wavelength wide
    """
    n = fbm(points, wavelength, 3, seed, min_wavelength=min_wavelength)
    return _band_limited_step(
        n, MESA_LEVEL, 0.05, wavelength / 40.0, MESA_SHARE, min_wavelength
    )


def canyons(
    points: np.ndarray, wavelength: float, seed: int, min_wavelength: float | None
) -> np.ndarray:
    """Canyon cuts: 1 on a canyon floor, 0 outside, steep walls between.

    :param points: (N, 3) positions, metres
    :param wavelength: spacing of the canyon network's noise, metres
    :param seed: field seed
    :param min_wavelength: shortest wavelength to keep, metres
    :returns: (N,) in [0, 1]; canyons about 1/20 of the wavelength wide, a
        third of that their floor, along the noise's zero lines
    """
    n = np.abs(fbm(points, wavelength, 3, seed, min_wavelength=min_wavelength))
    outside = _band_limited_step(
        n, CANYON_LEVEL, 0.04, wavelength / 60.0, 1.0 - CANYON_SHARE, min_wavelength
    )
    return 1.0 - outside
