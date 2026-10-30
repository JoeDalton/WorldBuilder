"""
Desert landforms, as pure functions of position: dunes, mesas, terraces, canyons.

They are shaped to stay band limited, as the rest of the elevation: a coarse
chunk gets a smoothed version, never aliased steps.

* **Dunes**: transverse ridges with a gentle windward side and a steep slip
  face. The profile is a short Fourier series of an asymmetric sawtooth, and
  each harmonic fades out as the octaves of :func:`noise.fbm` do. Crests are
  straight lines of the planet frame, bent by noise; the direction comes from
  three fixed axes blended by how flat each lies on the ground there, so the
  pattern turns from region to region (and crosses itself into star dunes
  where two axes meet) without a singular point anywhere on the sphere.
* **Terraces**: a staircase of the elevation, flat treads and steep risers,
  faded in as the sampling gets fine enough to show the risers.
* **Mesas**: flat-topped tables with cliff edges, from a thresholded noise.
* **Canyons**: flat-floored, steep-walled cuts along the zero lines of a
  noise.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from world_builder.terrain.noise import fbm, octave_frame, octave_weight, smoothstep

#: Share of a dune's wavelength on its windward (stoss) side.
DUNE_STOSS = 0.8
#: Harmonics of the dune profile at full resolution.
DUNE_HARMONICS = 8


@functools.lru_cache(maxsize=1)
def _dune_series() -> tuple[np.ndarray, np.ndarray, float, float]:
    """The dune profile's Fourier series.

    :returns: amplitudes and phases of harmonics 1..K, the series' mean, and
        its steepest slope (per unit of phase) relative to its height
    """
    n = 8192
    theta = np.arange(n) / n
    sawtooth = np.where(
        theta < DUNE_STOSS, theta / DUNE_STOSS, (1.0 - theta) / (1.0 - DUNE_STOSS)
    )
    spectrum = np.fft.rfft(sawtooth) / n
    k = np.arange(1, DUNE_HARMONICS + 1)
    amplitude = 2.0 * np.abs(spectrum[k])
    phase = np.angle(spectrum[k])
    profile = spectrum[0].real + np.sum(
        amplitude[:, None] * np.cos(2.0 * np.pi * k[:, None] * theta + phase[:, None]),
        axis=0,
    )
    slope = np.abs(np.gradient(profile, 1.0 / n)).max()
    return amplitude, phase, float(spectrum[0].real), float(slope / np.ptp(profile))


def dune_height(wavelength: float, slip: float) -> float:
    """:returns: crest-to-trough height, metres, of dunes of this wavelength
    whose slip face is *slip* degrees"""
    *_, steepest = _dune_series()
    return math.tan(math.radians(slip)) * wavelength / steepest


def dunes(
    points: np.ndarray,
    dirs: np.ndarray,
    wavelength: float,
    slip: float,
    seed: int,
    min_wavelength: float | None,
) -> np.ndarray:
    """Dune relief, in metres, about 0 to the dunes' full height.

    :param points: (N, 3) positions, metres (warped, as the other fields)
    :param dirs: (N, 3) unit vectors at the same points
    :param wavelength: crest spacing, metres
    :param slip: steepest slope, degrees, at full resolution
    :param seed: field seed
    :param min_wavelength: shortest wavelength to keep, metres; None = all
    :returns: (N,) heights
    """
    amplitude, phase, mean, _ = _dune_series()
    max_frequency = None if min_wavelength is None else 1.0 / min_wavelength
    height = dune_height(wavelength, slip)
    # Crests bend over a few wavelengths: the phase is pushed by noise.
    bend = (
        0.15
        * wavelength
        * fbm(points, 6.0 * wavelength, 2, seed + 1, min_wavelength=min_wavelength)
    )
    axes, _ = octave_frame(seed, 0)
    # How flat each axis lies on the ground: 1 - (axis.up)^2. The three values
    # sum to 2, so the largest is at least 2/3. The flattest axis sets the
    # crests; two blend (star dunes) only where they nearly tie.
    flat = 1.0 - (dirs @ axes.T) ** 2
    flattest = flat.max(axis=1)
    total = np.zeros(len(points))
    weights = np.zeros(len(points))
    for column, axis in enumerate(axes):
        weight = smoothstep(-0.06, 0.0, flat[:, column] - flattest)
        on = weight > 0.0
        if not np.any(on):
            continue
        theta = (points[on] @ axis + bend[on]) / wavelength
        profile = np.full(np.count_nonzero(on), mean)
        for k in range(1, DUNE_HARMONICS + 1):
            fade = octave_weight(k / wavelength, max_frequency)
            if fade <= 0.0:
                break
            profile += (
                fade * amplitude[k - 1] * np.cos(2.0 * np.pi * k * theta + phase[k - 1])
            )
        total[on] += weight[on] * profile
        weights[on] += weight[on]
    return height * total / weights


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
