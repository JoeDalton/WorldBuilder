"""
A field's rocks that are not drawn, seen as a haze: what makes a ring a band
of light from afar and fills the distance between drawn rocks up close.

- **What is missing.** A rock is drawn while its diameter covers at least
  ``min_pixels`` (``population``), so at a distance t the rocks smaller than
  r_cut(t) = t * min_pixels * :data:`FADE_MIDDLE` / (2 pixels_per_radian)
  are not (they fade out between 1 and 1.25 times ``min_pixels``: the cut
  is that fade's middle), nor, past the field's ``far``, any rock: they fade
  out over its last fifth, and the haze takes them in over the same band. The haze is
  their cross-section: per metre of a ray, the density there (the shape's
  profile times the peak density times the clumping's coverage: the clumps
  averaged out) times the mean cross-section of the rocks under r_cut
  (:class:`HazeModel`). Approaching, the cut falls, rocks appear and the
  haze thins by as much. Plus the spec's dust (``Density.dust``), never
  drawn: a constant optical depth per metre at full profile.
- **The integral.** Along each pixel's ray, from where it enters the field
  to where it leaves (each interval of a ring's annulus apart), by
  :data:`SAMPLES` midpoints spaced geometrically: r_cut grows with the
  distance, and nothing is missing nearer than where it reaches the
  smallest rock. :func:`optical_depth` is the shader's quadrature in numpy.
- **Light.** The rocks are Lambert spheres of the rocks' mean albedo: seen
  from where the sun is behind, full; against it, dark (:func:`phase`),
  plus the sky light. Coverage 1 - exp(-optical depth), then the field's
  exposure and tonemap. No shadows (not even the planet's on its ring).
- **Drawing.** A hull around the field's shape, drawn on both sides: each
  pixel where a ray leaves the field integrates the interval that ends
  there, so nothing is counted twice. Depth-tested, not written: a planet in
  front hides it. Hull points beyond the far plane are pulled onto it (as the
  sky's), so a ring across the sky is drawn whole. Only the far camera of
  a split draws it (``HIDE_FROM_NEAR``), from its near plane on (read from
  its projection): nearer than the split, every rock is drawn (the haze
  starts where the smallest rock is under ``min_pixels``, about 57 km for
  50 m rocks at 2 pixels), so nothing is missed and nothing counted twice.
  A single camera draws it from its near plane.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    ColorBlendAttrib,
    CullFaceAttrib,
    DepthTestAttrib,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LVecBase3f,
    OmniBoundingVolume,
    PTA_float,
    PTA_LVecBase3f,
    Shader,
)

from world_builder.asteroids.distributions import interpolate
from world_builder.asteroids.population import FAR_FADE
from world_builder.asteroids.spec import Annulus, Box, FieldShape, FieldSpec, Sphere
from world_builder.depth import HIDE_FROM_NEAR

if TYPE_CHECKING:
    from world_builder.asteroids.field import AsteroidField

#: Midpoints along each ray (haze.frag's SAMPLES).
SAMPLES = 32
#: Entries of the cross-section table, over log radius (haze.frag's).
TABLE = 64
#: The per-frame table of what is missing per metre (haze.glsl): entries,
#: over log distance from MISSING_T0 to MISSING_T1 metres (9 % apart).
MISSING = 256
MISSING_T0 = 1.0
MISSING_T1 = 1.0e10
#: Entries of a ring's radial profile as the shader holds it.
PROFILE = 256
#: Drawn rocks fade out between 1 and 1.25 times min_pixels: the haze takes
#: those under the middle.
FADE_MIDDLE = 1.125
#: A rock's mean projected area over its bounding disc's, until its
#: prototypes are built (then theirs: a quarter of their surface).
DEFAULT_COVER = 0.55
#: Segments of a ring's hull: its outer wall is at most R * pi^2 / (2 N^2)
#: beyond the ring (500 m at 110,000 km).
RING_SEGMENTS = 1024
#: The cull bin: after the sky (21) and the clouds (22-28), blended over them.
HAZE_BIN = "fixed"
HAZE_BIN_SORT = 30


def phase(alpha: ArrayLike) -> np.ndarray:
    """:returns: a Lambert sphere's brightness at phase angle *alpha*, over
    its brightness seen from the sun's side (1 there, 0 against the sun)"""
    a = np.asarray(alpha, dtype=np.float64)
    return (np.sin(a) + (np.pi - a) * np.cos(a)) / np.pi


def radiance(
    albedo: ArrayLike,
    sun_direction: ArrayLike,
    sun_color: ArrayLike,
    ambient: ArrayLike,
    view: ArrayLike,
) -> np.ndarray:
    """:returns: the haze's light toward the camera, linear (haze.frag): the
    rocks as Lambert spheres of *albedo*, lit by the sun and the sky light;
    *view* is the ray's direction (from the camera)"""
    sun = np.asarray(sun_direction, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    view = np.asarray(view, dtype=np.float64)
    alpha = math.acos(float(np.clip(sun @ (-view / np.linalg.norm(view)), -1.0, 1.0)))
    return np.asarray(albedo) * (
        np.asarray(sun_color) * (2.0 / 3.0) * float(phase(alpha)) + np.asarray(ambient)
    )


def mean_projected_cover(prototypes) -> float:
    """:returns: the prototypes' mean projected area over their bounding
    disc's: a quarter of a convex body's surface (their finest meshes')"""
    covers = []
    for p in prototypes:
        mesh = p.meshes[-1]
        tri = mesh.vertices.astype(np.float64)[mesh.indices.reshape(-1, 3)]
        area = (
            0.5
            * np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            ).sum()
        )
        covers.append(area / 4.0 / math.pi)
    return float(np.mean(covers))


class HazeModel:
    """What a field's rocks under a size amount to per metre of a ray.

    :ivar radii: (TABLE,) the table's radii, log-spaced from the smallest
        rock to the largest
    :ivar table: (TABLE,) the mean of pi r^2 over all candidates, each
        weighted by the share the hard core keeps, counting only those up to
        each radius
    :ivar density: candidates per cubic metre at full profile, clumps
        averaged
    :ivar kept: the hard core's share kept, per size draw
    :ivar dust: the dust's optical depth per metre at full profile
    :ivar class_lo: (classes,) the scatter's size classes' smallest radii
    :ivar class_hi: (classes,) and largest
    """

    def __init__(self, spec: FieldSpec, cover: float = DEFAULT_COVER):
        self.spec = spec
        sizes = spec.sizes
        self.lo, self.hi = float(sizes.lo), float(sizes.hi)
        clumping = spec.density.clumping
        coverage = 1.0 if clumping is None else clumping.coverage
        self.density = spec.density.count_per_km3 * 1e-9 * coverage
        self.dust = spec.density.dust * 1e-3
        self.cover = float(cover)
        self.radii = self.lo * (self.hi / self.lo) ** np.linspace(0.0, 1.0, TABLE)
        # The rocks' sizes, by equally likely draws, each kept as the
        # scatter's hard core keeps it: a candidate survives unless a larger
        # one is within their radii's sum (Matern II, at the peak density:
        # rocks are where the clumps are)
        n = 16384
        r = np.sort(sizes.ppf((np.arange(n) + 0.5) / n))
        peak = spec.density.count_per_km3 * 1e-9
        self.kept = _kept(r, peak)
        area = np.cumsum(np.pi * r * r * self.kept) / n
        upto = np.searchsorted(r, self.radii, side="right")
        self.table = np.where(upto > 0, area[np.maximum(upto - 1, 0)], 0.0)
        self.table[-1] = area[-1]
        # The scatter's size classes (scatter._Tables): radii doubling
        lo, bounds = self.lo, []
        while True:
            hi = min(2.0 * lo, self.hi)
            bounds.append((lo, hi))
            if hi >= self.hi:
                break
            lo = hi
        self.class_lo, self.class_hi = (np.array(b) for b in zip(*bounds))

    @property
    def log_span(self) -> float:
        return max(math.log(self.hi / self.lo), 1e-6)

    def r_cut(
        self, t: ArrayLike, pixels_per_radian: float, min_pixels: float
    ) -> np.ndarray:
        """:returns: the largest rock too small to be drawn at distances *t*"""
        t = np.asarray(t, dtype=np.float64)
        return t * FADE_MIDDLE * min_pixels / (2.0 * pixels_per_radian)

    def cumulative(self, r: ArrayLike) -> np.ndarray:
        """:returns: the table at radii *r*: the mean of pi r'^2 over the
        candidates up to them (haze.glsl's hazeCrossSection)"""
        r = np.maximum(np.asarray(r, dtype=np.float64), self.lo)
        x = np.clip(np.log(r / self.lo) / self.log_span, 0.0, 1.0)
        return interpolate(self.table, x)

    def missing(
        self, t: ArrayLike, pixels_per_radian: float, min_pixels: float, limits
    ) -> np.ndarray:
        """:returns: per metre at full profile at distances *t*: each class's
        rocks too small to be drawn, the share of the others its limit fades
        out (:meth:`Population.limits`, or one far for all), and the dust
        (haze.glsl's hazeMissing)"""
        t = np.asarray(t, dtype=np.float64)
        cut = self.r_cut(t, pixels_per_radian, min_pixels)
        limits = np.broadcast_to(
            np.asarray(limits, dtype=np.float64), self.class_lo.shape
        )
        rocks = np.zeros_like(t)
        for lo, hi, limit in zip(self.class_lo, self.class_hi, limits):
            base = self.cumulative(lo)
            under = self.cumulative(np.clip(cut, lo, hi)) - base
            whole = self.cumulative(hi) - base
            if math.isinf(limit):
                drawn = np.ones_like(t)
            else:
                drawn = np.clip((limit - t) / max(FAR_FADE * limit, 1e-6), 0.0, 1.0)
            rocks = rocks + under * drawn + whole * (1.0 - drawn)
        return self.density * self.cover * rocks + self.dust

    def start(self, pixels_per_radian: float, min_pixels: float, limits) -> float:
        """:returns: the distance before which nothing is missing (the
        smallest rock is drawn, every class held, and there is no dust)"""
        if self.dust > 0.0:
            return 0.0
        limits = np.broadcast_to(
            np.asarray(limits, dtype=np.float64), self.class_lo.shape
        )
        return float(
            min(
                self.lo * 2.0 * pixels_per_radian / (FADE_MIDDLE * min_pixels),
                (1.0 - FAR_FADE) * float(np.min(limits)),
            )
        )

    def extinction(self, r_cut: ArrayLike) -> np.ndarray:
        """:returns: per metre at full profile: the cross-section of the
        rocks up to *r_cut*, and the dust (haze.frag's lookup)"""
        r = np.maximum(np.asarray(r_cut, dtype=np.float64), self.lo)
        x = np.clip(np.log(r / self.lo) / self.log_span, 0.0, 1.0)
        return self.density * self.cover * interpolate(self.table, x) + self.dust


def _kept(r: np.ndarray, per_m3: float) -> np.ndarray:
    """:returns: the share of candidates of each radius in *r* (sorted, equally
    likely draws) the hard core keeps: exp(-per_m3 * the mean over larger
    draws of 4/3 pi (r + r')^3), by suffix sums of r'^k"""
    n = len(r)
    suffix = [
        np.concatenate((np.cumsum((r**k)[::-1])[::-1][1:], [0.0])) / n for k in range(4)
    ]
    volume = r**3 * suffix[0] + 3.0 * r**2 * suffix[1] + 3.0 * r * suffix[2] + suffix[3]
    return np.exp(-per_m3 * 4.0 / 3.0 * np.pi * volume)


# ── rays ────────────────────────────────────────────────────────────────────


def _cylinder(origin, d, radius) -> tuple[float, float] | None:
    """:returns: where a ray is within *radius* of the z axis, or None"""
    a = d[0] * d[0] + d[1] * d[1]
    rho = math.hypot(origin[0], origin[1])
    c = (rho - radius) * (rho + radius)
    if a < 1e-18:
        return (-math.inf, math.inf) if c <= 0.0 else None
    b = origin[0] * d[0] + origin[1] * d[1]
    disc = b * b - a * c
    if disc < 0.0:
        return None
    q = -(b + math.copysign(math.sqrt(disc), b))
    if q == 0.0:
        return (0.0, 0.0)
    t1, t2 = q / a, c / q
    return (min(t1, t2), max(t1, t2))


def _slab(o: float, d: float, h: float) -> tuple[float, float] | None:
    if abs(d) < 1e-18:
        return (-math.inf, math.inf) if abs(o) <= h else None
    t1, t2 = (-h - o) / d, (h - o) / d
    return (min(t1, t2), max(t1, t2))


def intervals(shape: FieldShape, origin: ArrayLike, direction: ArrayLike) -> list:
    """:returns: [(a, b), ...] where a ray (unit *direction*) is in the
    shape's bounds, ahead of *origin* (a >= 0), as haze.frag cuts them"""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    found: list[tuple[float, float]] = []
    if isinstance(shape, Box):
        lo, hi = 0.0, math.inf
        for axis, h in enumerate(shape.half_size):
            span = _slab(o[axis], d[axis], h)
            if span is None:
                return []
            lo, hi = max(lo, span[0]), min(hi, span[1])
        found = [(lo, hi)]
    elif isinstance(shape, Sphere):
        b = float(o @ d)
        c = float(o @ o) - shape.radius * shape.radius
        disc = b * b - c
        if disc < 0.0:
            return []
        s = math.sqrt(disc)
        found = [(max(-b - s, 0.0), -b + s)]
    elif isinstance(shape, Annulus):
        z = _slab(o[2], d[2], 0.5 * shape.thickness)
        outer = _cylinder(o, d, shape.outer)
        if z is None or outer is None:
            return []
        lo, hi = max(z[0], outer[0], 0.0), min(z[1], outer[1])
        hole = _cylinder(o, d, shape.inner) if shape.inner > 0.0 else None
        if hole is None:
            found = [(lo, hi)]
        else:
            found = [(lo, min(hi, hole[0])), (max(lo, hole[1]), hi)]
    else:
        raise TypeError(f"unknown field shape {shape!r}")
    return [(a, b) for a, b in found if b > a]


def optical_depth(
    model: HazeModel,
    origin: ArrayLike,
    direction: ArrayLike,
    t0: float,
    t1: float,
    pixels_per_radian: float,
    min_pixels: float,
    limits,
    samples: int = SAMPLES,
) -> float:
    """:returns: the haze's optical depth along a ray from *t0* to *t1*:
    haze.glsl's quadrature (field frame, unit *direction*); *limits* per
    class (:meth:`Population.limits`) or one far for all"""
    a = max(t0, model.start(pixels_per_radian, min_pixels, limits))
    if t1 <= a:
        return 0.0
    a = max(a, 1e-3)
    edges = a * (t1 / a) ** (np.arange(samples + 1) / samples)
    mids = np.sqrt(edges[:-1] * edges[1:])
    points = np.asarray(origin, dtype=np.float64) + np.outer(
        mids, np.asarray(direction, dtype=np.float64)
    )
    sigma = model.missing(mids, pixels_per_radian, min_pixels, limits)
    return float(np.sum(model.spec.shape.profile(points) * sigma * np.diff(edges)))


def mean_rock_albedo(ice: ArrayLike, mean_albedo: np.ndarray) -> np.ndarray:
    """:returns: the rocks' mean colour (linear): each rock is snow on the
    share of its surface its ice is (in patches), rock elsewhere; averaged
    over *ice* (the rocks' ice fractions)"""
    share = float(np.mean(np.asarray(ice, dtype=np.float64)))
    return share * mean_albedo[0] + (1.0 - share) * mean_albedo[1]


# ── drawing ─────────────────────────────────────────────────────────────────


def hull_triangles(shape: FieldShape) -> tuple[np.ndarray, np.ndarray]:
    """A closed hull around the shape, each face with its outward normal.

    :returns: (n, 3, 3) triangles and (n, 3) normals, field frame"""
    tris: list = []
    normals: list = []

    def quad(a, b, c, d, n):
        tris.extend(([a, b, c], [a, c, d]))
        normals.extend((n, n))

    if isinstance(shape, Box):
        h = np.asarray(shape.half_size, dtype=np.float64) + 1.0
        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for sign in (-1.0, 1.0):
                n = np.zeros(3)
                n[axis] = sign
                corners = []
                for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    p = np.zeros(3)
                    p[axis], p[u], p[v] = sign * h[axis], su * h[u], sv * h[v]
                    corners.append(p)
                quad(*corners, n)
    elif isinstance(shape, Sphere):
        # latitude-longitude quads, pushed out past the sphere
        rings, segments = 24, 48
        theta = np.linspace(0.0, np.pi, rings + 1)
        phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)
        reach = shape.radius / (math.cos(np.pi / rings) * math.cos(np.pi / segments))
        reach += 1.0
        for i in range(rings):
            for j in range(segments):
                ps = [
                    reach
                    * np.array(
                        [
                            math.sin(t) * math.cos(f),
                            math.sin(t) * math.sin(f),
                            math.cos(t),
                        ]
                    )
                    for t, f in (
                        (theta[i], phi[j]),
                        (theta[i + 1], phi[j]),
                        (theta[i + 1], phi[j + 1]),
                        (theta[i], phi[j + 1]),
                    )
                ]
                n = sum(ps) / np.linalg.norm(sum(ps))
                quad(*ps, n)
    elif isinstance(shape, Annulus):
        segments = RING_SEGMENTS
        angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
        outer = shape.outer / math.cos(np.pi / segments) + 1.0
        inner = max(shape.inner - 1.0, 0.0)
        h = 0.5 * shape.thickness + 1.0
        for k in range(segments):
            a0, a1 = angles[k], angles[k + 1]
            e0 = np.array([math.cos(a0), math.sin(a0), 0.0])
            e1 = np.array([math.cos(a1), math.sin(a1), 0.0])
            mid = 0.5 * (e0 + e1)
            mid /= np.linalg.norm(mid)
            up, down = np.array([0.0, 0.0, h]), np.array([0.0, 0.0, -h])
            quad(
                outer * e0 + down,
                outer * e1 + down,
                outer * e1 + up,
                outer * e0 + up,
                mid,
            )
            quad(
                outer * e0 + up,
                outer * e1 + up,
                inner * e1 + up,
                inner * e0 + up,
                np.array([0.0, 0.0, 1.0]),
            )
            quad(
                inner * e0 + down,
                inner * e1 + down,
                outer * e1 + down,
                outer * e0 + down,
                np.array([0.0, 0.0, -1.0]),
            )
            if inner > 0.0:
                quad(
                    inner * e1 + down,
                    inner * e0 + down,
                    inner * e0 + up,
                    inner * e1 + up,
                    -mid,
                )
    else:
        raise TypeError(f"unknown field shape {shape!r}")
    return np.array(tris), np.array(normals)


class FieldHaze:
    """The haze of a field (see the module): a hull under the field's root,
    and its inputs on the root, so the rocks are veiled by the same haze
    (``asteroid.frag``). Without the hull (``enabled=False``) the rocks are
    not veiled."""

    def __init__(self, field: AsteroidField, shader_path, enabled: bool = True):
        self.field = field
        spec = field.spec
        self.model = HazeModel(spec)
        tris, normals = hull_triangles(spec.shape)
        self.node = field.root.attach_new_node(_hull_node(tris, normals))
        node = self.node
        node.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=shader_path / "haze.vert",
                fragment=shader_path / "haze.frag",
            )
        )
        node.hide(HIDE_FROM_NEAR)
        node.set_bin(HAZE_BIN, HAZE_BIN_SORT)
        node.set_depth_write(False)
        node.set_attrib(DepthTestAttrib.make(DepthTestAttrib.M_less_equal))
        node.set_attrib(CullFaceAttrib.make(CullFaceAttrib.M_cull_none))
        node.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_incoming_alpha,
                ColorBlendAttrib.O_one_minus_incoming_alpha,
            )
        )
        if not enabled:
            node.stash()
        root = field.root
        root.set_shader_input("uVeil", 1 if enabled else 0)
        self._cam = PTA_LVecBase3f.empty_array(1)
        root.set_shader_input("uCamField", self._cam)
        self._limits = PTA_float.empty_array(16)
        root.set_shader_input("uClassFar", self._limits)
        self._missing = PTA_float.empty_array(MISSING)
        root.set_shader_input("uMissing", self._missing)
        root.set_shader_input("uMissingT0", MISSING_T0)
        root.set_shader_input("uMissingLogSpan", math.log(MISSING_T1 / MISSING_T0))
        self._start = PTA_float.empty_array(1)
        root.set_shader_input("uHazeStart", self._start)
        self._distances = MISSING_T0 * (MISSING_T1 / MISSING_T0) ** np.linspace(
            0.0, 1.0, MISSING
        )
        self._set_shape(spec.shape)
        self.set_cover(DEFAULT_COVER)
        self.set_albedo(field.materials, spec)

    def _set_shape(self, shape: FieldShape) -> None:
        root = self.field.root
        profile = np.zeros(PROFILE)
        size = (0.0, 0.0, 0.0)
        edges = (shape.edge, 0.0)
        kind, profiled = 0, 0
        if isinstance(shape, Box):
            size = shape.half_size
        elif isinstance(shape, Sphere):
            kind, size = 1, (shape.radius, 0.0, 0.0)
        else:
            kind = 2
            size = (shape.inner, shape.outer, 0.5 * shape.thickness)
            edges = (shape.edge, shape.radial_edge)
            if shape.radial_profile:
                profiled = 1
                profile = interpolate(
                    np.asarray(shape.radial_profile, dtype=np.float64),
                    np.linspace(0.0, 1.0, PROFILE),
                )
        root.set_shader_input("uShape", kind)
        root.set_shader_input("uShapeSize", LVecBase3f(*size))
        root.set_shader_input("uEdges", edges)
        root.set_shader_input("uProfiled", profiled)
        root.set_shader_input("uProfile", _floats(profile, PROFILE))

    def set_cover(self, cover: float) -> None:
        """The rocks' mean projected area over their bounding disc's."""
        self.model.cover = float(cover)

    def set_albedo(self, materials, spec: FieldSpec) -> None:
        self.albedo = mean_rock_albedo(
            spec.material.ice_fraction.table_values(257), materials.mean_albedo
        )
        self.field.root.set_shader_input("uHazeAlbedo", LVecBase3f(*self.albedo))

    def update(
        self,
        camera: np.ndarray,
        limits: np.ndarray,
        pixels_per_radian: float,
        min_pixels: float,
    ) -> None:
        """Per frame: the camera, and what is missing at each distance (a
        table: the shader looks it up instead of summing the classes).

        :param camera: the camera, field frame
        :param limits: :meth:`Population.limits`, per class
        :param pixels_per_radian: the main view's
        :param min_pixels: the smallest diameter drawn"""
        self._cam[0] = LVecBase3f(*camera)
        for k, limit in enumerate(limits):
            self._limits[k] = float(min(limit, 1e30))
        missing = self.model.missing(
            self._distances, pixels_per_radian, min_pixels, limits
        )
        for i, value in enumerate(missing.tolist()):
            self._missing[i] = value
        self._start[0] = min(
            self.model.start(pixels_per_radian, min_pixels, limits), 1e30
        )

    def destroy(self) -> None:
        self.node.remove_node()


def _floats(values, size: int) -> PTA_float:
    """:returns: a float array input of *size*, *values* first"""
    out = PTA_float.empty_array(size)
    for i, v in enumerate(np.asarray(values, dtype=np.float64)):
        out[i] = float(v)
    return out


def _hull_node(tris: np.ndarray, normals: np.ndarray) -> GeomNode:
    vdata = GeomVertexData("haze_hull", GeomVertexFormat.get_v3n3(), Geom.UH_static)
    vdata.unclean_set_num_rows(3 * len(tris))
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    for tri, n in zip(tris, normals):
        for p in tri:
            vertex.set_data3f(*p)
            normal.set_data3f(*n)
    prim = GeomTriangles(Geom.UH_static)
    prim.set_index_type(Geom.NT_uint32)
    for i in range(len(tris)):
        prim.add_vertices(3 * i, 3 * i + 1, 3 * i + 2)
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("haze")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    return node
