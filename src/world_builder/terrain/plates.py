"""
Tectonic plates: rigid caps of the sphere, each spinning about its own pole.

A planet's plates are the Voronoi cells of a few centres spread over the
sphere (a Fibonacci lattice, jittered). Each plate turns about an Euler pole,
so its ground moves at ``spin x position``; the relative motion of two plates
where they meet says what their boundary is:

* **convergent** (they close): mountain belts
* **divergent** (they open): mid-ocean ridges, rift valleys on land
* **transform** (they slide past each other): long straight coasts

Each plate also carries a crust bias (continental or oceanic), which pulls
coastlines onto plate edges.

Everything here is a pure function of a direction, smooth enough for the
planet's low band (interpolated on a lattice): a boundary's effects fade
with distance from it, and every pair of nearby plates contributes with a
weight that vanishes before the pair stops being nearby, so nothing jumps at
triple junctions.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass

import numpy as np

from world_builder.terrain.noise import smoothstep

#: Plates considered at each point: the nearest few. More than three, so a
#: plate enters the set only well away from its edge (where its weight is 0).
NEAREST = 5


@dataclass(frozen=True, eq=False)
class Plates:
    """One planet's plates.

    :ivar centres: (N, 3) unit vectors, the Voronoi sites
    :ivar spins: (N, 3) angular velocities: Euler pole times rate, the rate
        in [0.5, 1] (arbitrary units: only relative motion matters)
    :ivar crust: (N,) in [-1, 1]: continental above 0, oceanic below
    :ivar chord: (N, N) distances between centres (1 on the diagonal)
    :ivar closing: (N, N, 3) per pair (a, b): ``normal x (spin_a - spin_b)``,
        whose dot product with a point is how fast a and b separate there
        (the normal points from b's centre to a's)
    :ivar relative: (N, N, 3) ``spin_a - spin_b``
    """

    centres: np.ndarray
    spins: np.ndarray
    crust: np.ndarray
    chord: np.ndarray
    closing: np.ndarray
    relative: np.ndarray

    @classmethod
    def of(cls, centres: np.ndarray, spins: np.ndarray, crust: np.ndarray) -> Plates:
        """:returns: the plates with these centres (unit vectors), spins and
        crust biases, their per-pair tables filled in"""
        # Per pair: a's motion relative to b at x is relative x x, and its
        # part along the normal is x . (normal x relative).
        chord = centres[:, None, :] - centres[None, :, :]
        length = np.linalg.norm(chord, axis=2)
        np.fill_diagonal(length, 1.0)
        normal = chord / length[:, :, None]
        relative = spins[:, None, :] - spins[None, :, :]
        return cls(
            centres=centres,
            spins=spins,
            crust=crust,
            chord=length,
            closing=np.cross(normal, relative),
            relative=relative,
        )


@dataclass
class PlateFields:
    """What the plates do at a set of points; each field (N,) in [0, 1].

    :ivar crust: the plates' crust bias, blended across their edges, in [-1, 1]
    :ivar convergent: strength of a closing boundary nearby
    :ivar divergent: strength of an opening boundary nearby
    :ivar transform: strength of a sliding boundary nearby
    """

    crust: np.ndarray
    convergent: np.ndarray
    divergent: np.ndarray
    transform: np.ndarray


@functools.lru_cache(maxsize=16)
def make_plates(seed: int, count: int) -> Plates:
    """A planet's plates, the same for the same seed and count.

    :param seed: field seed
    :param count: number of plates, >= 2
    :returns: the plates
    """
    rng = np.random.default_rng([seed & 0xFFFFFFFF, count, 0x91A7E5])
    # Fibonacci sphere: evenly spread, then turned at random and moved by up
    # to a quarter of the spacing, so plates differ in size and shape (but
    # none is a sliver).
    k = np.arange(count) + 0.5
    z = 1.0 - 2.0 * k / count
    phi = math.pi * (1.0 + math.sqrt(5.0)) * k
    r = np.sqrt(1.0 - z * z)
    centres = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    q, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(upper))
    centres = centres @ q.T
    spacing = math.sqrt(4.0 * math.pi / count)
    shift = rng.normal(size=centres.shape)
    shift -= np.sum(shift * centres, axis=1, keepdims=True) * centres
    shift /= np.linalg.norm(shift, axis=1, keepdims=True)
    centres += shift * rng.uniform(0.0, 0.25 * spacing, size=(count, 1))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    poles = rng.normal(size=(count, 3))
    poles /= np.linalg.norm(poles, axis=1, keepdims=True)
    spins = poles * rng.uniform(0.5, 1.0, size=(count, 1))
    crust = rng.uniform(-1.0, 1.0, size=count)
    return Plates.of(centres, spins, crust)


def plate_fields(
    plates: Plates, dirs: np.ndarray, radius: float, width: float
) -> PlateFields:
    """Crust and boundary strengths at some points.

    :param plates: the planet's plates
    :param dirs: (N, 3) unit vectors (warped by the caller, so edges meander)
    :param radius: planet radius, metres
    :param width: half-width of a boundary's belt, metres
    :returns: the fields
    """
    count = len(plates.centres)
    nearest = min(NEAREST, count)
    dots = dirs @ plates.centres.T
    top = np.argpartition(-dots, nearest - 1, axis=1)[:, :nearest]
    top_dots = np.take_along_axis(dots, top, axis=1)
    best = top_dots.max(axis=1, keepdims=True)
    # A plate's weight falls from 1 (nearest) to 0 a little over a belt's
    # width beyond its edge (x.c_a - x.c_b is about |c_a - c_b| * d / R).
    reach = 1.5 * width / radius
    near = smoothstep(-reach, 0.0, top_dots - best)
    sharp = smoothstep(-0.3 * reach, 0.0, top_dots - best)
    crust = np.sum(sharp * plates.crust[top], axis=1) / np.sum(sharp, axis=1)

    pairs = np.array(list(itertools.combinations(range(nearest), 2)))
    weight = near[:, pairs[:, 0]] * near[:, pairs[:, 1]]  # (N, pairs)
    # Only about one pair in six has weight: compute those alone.
    rows, columns = np.nonzero(weight > 0.0)
    a, b = pairs[columns, 0], pairs[columns, 1]
    ia, ib = top[rows, a], top[rows, b]
    x = dirs[rows]
    # Distance to the pair's bisector: x.n = (x.c_a - x.c_b) / |c_a - c_b|
    along = (top_dots[rows, a] - top_dots[rows, b]) / plates.chord[ia, ib]
    distance = radius * np.arcsin(np.clip(along, -1.0, 1.0))
    belt = weight[rows, columns] * (1.0 - smoothstep(0.0, width, np.abs(distance)))
    # a's motion relative to b, dv = relative x x: its part along the
    # normal (a heading toward b when negative) and across it.
    normal_part = np.einsum("ik,ik->i", plates.closing[ia, ib], x)
    relative = plates.relative[ia, ib]
    speed2 = (
        np.einsum("ik,ik->i", relative, relative)
        - np.einsum("ik,ik->i", relative, x) ** 2
    )
    closing = -normal_part / 2.0
    sliding = np.sqrt(np.maximum(speed2 - normal_part**2, 0.0)) / 2.0

    def strongest(strength: np.ndarray) -> np.ndarray:
        # The strongest pair wins: max of continuous terms stays continuous.
        per_pair = np.zeros(weight.shape)
        per_pair[rows, columns] = belt * strength
        return per_pair.max(axis=1)

    convergent = strongest(smoothstep(0.02, 0.2, closing))
    divergent = strongest(smoothstep(0.02, 0.2, -closing))
    transform = strongest(smoothstep(0.1, 0.4, sliding))
    return PlateFields(
        crust=crust, convergent=convergent, divergent=divergent, transform=transform
    )
