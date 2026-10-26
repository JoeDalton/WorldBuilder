"""
What an asteroid field is: its extent, how dense, the rocks' sizes, spins,
shapes and materials, and where it must leave room. Frozen, picklable
dataclasses; :mod:`world_builder.asteroids.scatter` turns one into rocks.

Lengths are metres, angles radians, rates radians per second. Everything is
in the **field frame**: the field's own centre and axes (an
:class:`~world_builder.asteroids.field.AsteroidField` places that frame in the
world).

A rock's ``radius`` is its bounding radius: no part of it reaches farther from
its centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from world_builder.asteroids.distributions import Distribution

# ── extent ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Box:
    """A box centred on the field's origin.

    :ivar half_size: half its extent along x, y and z
    :ivar edge: depth over which the density fades to nothing inside the
        faces (0 = a sharp edge)
    """

    half_size: tuple[float, float, float]
    edge: float = 0.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """:returns: the (min, max) corners of where rock centres can be"""
        half = np.asarray(self.half_size, dtype=np.float64)
        return -half, half

    def volume(self) -> float:
        """:returns: the integral of the profile, cubic metres"""
        return float(np.prod([2.0 * h - self.edge for h in self.half_size]))


@dataclass(frozen=True)
class Sphere:
    """A ball centred on the field's origin.

    :ivar radius: its radius
    :ivar edge: depth over which the density fades inside the surface
    """

    radius: float
    edge: float = 0.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(3, -self.radius), np.full(3, self.radius)

    def volume(self) -> float:
        r = np.linspace(0.0, self.radius, 4097)
        t = np.clip((self.radius - r) / max(self.edge, 1e-300), 0.0, 1.0)
        profile = t * t * (3.0 - 2.0 * t) if self.edge > 0.0 else np.ones_like(r)
        return float(np.trapezoid(4.0 * math.pi * r * r * profile, r))


FieldShape = Box | Sphere

# ── density ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Clumping:
    """Rocks gather where a 3D noise is high: the field is patchy, with
    clusters and empty stretches.

    :ivar scale: the noise's largest wavelength: the clusters' spacing
    :ivar coverage: the share of space populated (0-1); the noise threshold
        is set from it
    :ivar softness: how gradually the density falls off a cluster's edge,
        in noise units (0.02 sharp, 0.3 very soft)
    :ivar octaves: the noise's octaves (more: more ragged clusters)
    """

    scale: float
    coverage: float = 0.5
    softness: float = 0.1
    octaves: int = 4


@dataclass(frozen=True)
class Density:
    """:ivar count_per_km3: rocks per cubic kilometre where the field is
        densest (at full profile and full clumping), before rocks too close
        to a bigger one are removed
    :ivar clumping: patchiness; None = uniform"""

    count_per_km3: float
    clumping: Clumping | None = None


# ── spin ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Isotropic:
    """Spin axes in every direction alike."""


@dataclass(frozen=True)
class AroundAxis:
    """Spin axes gathered around one direction (von Mises-Fisher): a ring's
    rocks turning with its orbit.

    :ivar axis: the direction, field frame
    :ivar kappa: concentration: 0 isotropic, 10 within ~25 degrees, 100
        within ~8
    """

    axis: tuple[float, float, float]
    kappa: float


@dataclass(frozen=True)
class Tumble:
    """Some rocks tumble: free symmetric-top motion, their long axis
    (body z) sweeping a cone around the fixed axis while they turn about it.

    :ivar fraction: the share of rocks that tumble
    :ivar cone_angle: the cone's half-angle
    :ivar ratio: the body's own turn rate over the cone's, divided by
        cos(cone angle): I1 / I3 - 1 for moments of inertia I1 = I2 and I3
        (positive for an elongated body)
    """

    fraction: float
    cone_angle: Distribution
    ratio: Distribution


@dataclass(frozen=True)
class SpinSpec:
    """How the rocks turn: each about its own fixed axis, at a constant rate.

    :ivar rate: the turn rate of the smallest rocks (radius = the sizes' lo)
    :ivar axis: the axes' directions
    :ivar size_exponent: rate ~ (radius / smallest)^-size_exponent: big rocks
        turn slower
    :ivar tumble: some rocks tumble instead; None = none
    """

    rate: Distribution
    axis: Isotropic | AroundAxis = Isotropic()
    size_exponent: float = 1.5
    tumble: Tumble | None = None


# ── shapes, materials, keep-outs ──────────────────────────────────────────


@dataclass(frozen=True)
class ShapeSpec:
    """The rocks' shapes (:mod:`world_builder.asteroids.shape`). Every rock
    is one of a few prototypes, turned and scaled; lengths here are in units
    of a rock's radius, angles in radians.

    :ivar prototypes: distinct shapes
    :ivar elongation: each prototype's shortest axis over its longest (body
        z); the third axis lies between them
    :ivar lumps: the lumpy noise's amplitude, a share of the radius
    :ivar lump_scale: its largest wavelength
    :ivar ridges: ridged noise's amplitude: crests and grooves
    :ivar craters: craters per prototype, on average
    :ivar crater_size: their angular radius
    :ivar crater_depth: a crater's depth over its diameter (0.2 for a fresh
        bowl)
    """

    prototypes: int = 32
    elongation: Distribution = Distribution.uniform(0.5, 0.9)
    lumps: float = 0.16
    lump_scale: float = 1.2
    ridges: float = 0.02
    craters: float = 10.0
    crater_size: Distribution = Distribution.power_law(2.5, 0.06, 0.5)
    crater_depth: float = 0.2


@dataclass(frozen=True)
class MaterialSpec:
    """:ivar ice_fraction: each rock's share of ice (0 bare rock, 1 all ice)"""

    ice_fraction: Distribution = Distribution.constant(0.0)


@dataclass(frozen=True)
class KeepOut:
    """Room no rock may enter: a ball, or a capsule from *centre* to *end*.

    :ivar centre: the ball's centre, or one end of the capsule's axis
    :ivar radius: the clearance from that point (or axis)
    :ivar end: the capsule's other end; None = a ball
    """

    centre: tuple[float, float, float]
    radius: float
    end: tuple[float, float, float] | None = None

    def row(self) -> np.ndarray:
        """:returns: (7,) a, b, radius, as the scatter takes it"""
        end = self.centre if self.end is None else self.end
        return np.array([*self.centre, *end, self.radius], dtype=np.float64)


# ── the field ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FieldSpec:
    """An asteroid field.

    :ivar shape: where the field is (its density profile)
    :ivar density: how many rocks, and how patchy
    :ivar sizes: the rocks' bounding radii
    :ivar spin: how they turn
    :ivar shapes: what they look like
    :ivar material: what they are made of
    :ivar keep_outs: room no rock enters
    :ivar seed: each seed is a different field
    :ivar cell_factor: the scatter's cell edge over the smallest radius of
        each size class (at least 4). Changes the field; a larger factor
        means fewer, fuller cells
    """

    shape: FieldShape
    density: Density
    sizes: Distribution
    spin: SpinSpec = field(
        default_factory=lambda: SpinSpec(Distribution.lognormal(0.05, 0.7, 0.005, 0.5))
    )
    shapes: ShapeSpec = ShapeSpec()
    material: MaterialSpec = MaterialSpec()
    keep_outs: tuple[KeepOut, ...] = ()
    seed: int = 0
    cell_factor: float = 8.0

    def __post_init__(self):
        if self.sizes.lo <= 0.0:
            raise ValueError("rock radii must be positive")
        if self.cell_factor < 4.0:
            raise ValueError("cell_factor must be at least 4")
        if self.shapes.prototypes < 1:
            raise ValueError("at least one prototype")

    def summary(self) -> dict[str, float]:
        """What the spec amounts to, for tuning.

        :returns: expected rock count (before the hard core), the mean
            spacing and volume share filled where densest, and the median,
            mean and largest radius
        """
        coverage = (
            1.0 if self.density.clumping is None else self.density.clumping.coverage
        )
        per_m3 = self.density.count_per_km3 * 1e-9
        mean_volume = 4.0 / 3.0 * math.pi * self.sizes.moment(3.0)
        return {
            "expected_count": per_m3 * self.shape.volume() * coverage,
            "mean_spacing_m": per_m3 ** (-1.0 / 3.0) if per_m3 > 0.0 else math.inf,
            "volume_share": per_m3 * mean_volume,
            "median_radius_m": self.sizes.median(),
            "mean_radius_m": self.sizes.mean(),
            "max_radius_m": self.sizes.hi,
        }


#: Example fields, for the demos and the preview.
PRESETS: dict[str, FieldSpec] = {
    # SpaceFlight's missions 1-2 setting: a 15 km cube, with a corridor along
    # y at z = 500 m (scripts/asteroid_bench.py's cruise flies it).
    "cube": FieldSpec(
        shape=Box((7500.0, 7500.0, 7500.0), edge=500.0),
        density=Density(2.0, Clumping(scale=4000.0, coverage=0.6, softness=0.12)),
        sizes=Distribution.power_law(3.5, 50.0, 2000.0),
        material=MaterialSpec(Distribution.uniform(0.0, 0.15)),
        keep_outs=(KeepOut((0.0, -7500.0, 500.0), 300.0, end=(0.0, 7500.0, 500.0)),),
    ),
    # A ball of ice and rock, spinning around one axis.
    "icy_ball": FieldSpec(
        shape=Sphere(10_000.0, edge=2000.0),
        density=Density(1.0),
        sizes=Distribution.lognormal(150.0, 0.6, 50.0, 1000.0),
        spin=SpinSpec(
            Distribution.lognormal(0.05, 0.5, 0.005, 0.3),
            axis=AroundAxis((0.0, 0.0, 1.0), 20.0),
            tumble=Tumble(
                0.3, Distribution.uniform(0.2, 0.8), Distribution.uniform(0.3, 1.5)
            ),
        ),
        material=MaterialSpec(Distribution.u_shaped(0.4, 0.0, 1.0, log=False)),
    ),
}
