"""
Ocean — camera-locked reflective ocean surface with planar reflections.

The surface is either a single huge flat quad (default; all wave detail is
computed per-pixel in ocean.frag) or, with ``geometric_swell=True``, a dense
grid displaced by the swell in ocean.vert and ringed by flat border cells.
Either mesh is re-centred under the camera every frame.  Reflections come
from an offscreen buffer rendered by a camera mirrored across Z = 0.

Usage:
    from world_builder.ocean import Ocean
    self.ocean = Ocean(game=game)

``Ocean`` registers its own no-argument :meth:`Ocean.update` in
``game.method_lists``, so no manual per-frame call is needed.

It reads from ``game``: ``root_node``, ``method_lists``, ``game_time``
(``get_current_time()``) and ``app`` (the ShowBase: ``camera``, ``camLens``,
``win``, ``graphicsEngine``). Two settings are read when the app has them,
with defaults otherwise: ``app.graphics_settings.config["render"]
["reflection_scale"]`` (unless the ``reflection_scale`` argument is given)
and ``app.graphics_manager.get_render_size()``.

Alone, distant water fades to a haze colour. In a world with atmospheres,
bind them to :attr:`Ocean.ocean_node` (``Atmospheres.bind``) and keep
:meth:`Ocean.set_atmosphere_frame` current: they replace the haze. By
night, :meth:`Ocean.set_light` dims the water's own colour.

The ocean has no collision of its own: physics is the game's business. Its
surface is the plane z = 0, exact at the camera, where the droop is zero.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BitMask32,
    Camera,
    ClipPlaneAttrib,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LMatrix4f,
    LPlane,
    LVecBase2f,
    LVecBase3f,
    LVecBase4f,
    PlaneNode,
    PTA_float,
    PTA_LVecBase2f,
    SamplerState,
    Shader,
    Texture,
)

from world_builder import PLANET_RADIUS_M, SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M
from world_builder.atmosphere import bind_defaults
from world_builder.ocean.seabed import Cascade, Sampler, SeabedMap
from world_builder.utils import compute_next_power_of_2

_SHADERS = Path(__file__).parent / "shaders"
#: Reflection resolution, as a fraction of the render size, when the app has no
#: graphics settings to read it from.
DEFAULT_REFLECTION_SCALE = 0.5

# The surface must reach past the horizon from the highest eye, so its own bulge
# hides its edge: sqrt(2*R*altitude) is ~505 km at 20 km, plus a margin.
_MAX_EYE_ALTITUDE_M = 20000.0
_HORIZON_MARGIN = 1.1

# Camera mask bit used to hide the ocean from the reflection camera
#: The reflection camera's mask: camera bits 0-19. Panda draws a node for a
#: camera if any of the camera's bits is still shown on it, so a node is
#: hidden from the reflection by hiding *all* of these
#: (``node.hide(REFLECTION_CAMERA_MASK)``); hiding one bit does nothing. The
#: main cameras keep other bits (world_builder.depth's 21 and 22, or all of
#: them by default) and still see it.
REFLECTION_CAMERA_MASK = BitMask32(0xFFFFF)
#: The reflection's field of view, relative to the main camera's.
REFLECTION_FOV_SCALE = 1.4

# Max wave iterations the shader's iWaveDirs[] array can hold (must match the
# MAX_WAVE_ITER #define in ocean.frag).
_MAX_WAVE_ITER = 64
# ocean.frag's small-wave constants the CPU side needs (keep them equal).
_WAVE_HEIGHT = 10.0
_WEIGHT_DECAY = 0.8  # weight = mix(weight, 0, 0.2) per octave
_FREQ_GROWTH = 1.18
#: Mean square of one octave's slope, (exp(sin x - 1) cos x)^2 over a period
#: (ocean.frag's C_WAVE).
C_WAVE = 0.1076346
#: The fresnel table's size and the slope spread its last row is at
#: (ocean.frag's FRESNEL_LUT_SIZE and FRESNEL_SLOPE_MAX).
FRESNEL_LUT_SIZE = 32
FRESNEL_LUT_SLOPE_MAX = 2.0


@dataclass(frozen=True)
class OceanQuality:
    """How much the ocean is allowed to cost: every cost/quality trade in one
    place, to tune without touching the shaders. See :data:`OCEAN_QUALITY`.

    :ivar wave_iterations: most small-wave octaves a pixel evaluates (the
        filter stops sooner where they are sub-pixel)
    :ivar aa_full_px: an octave is at full strength while its wavelength
        spans at least this many pixels
    :ivar aa_gone_px: and is gone below this many (Nyquist is 2). Larger
        values drop octaves sooner: softer, cheaper
    :ivar rough_fresnel: average the fresnel over the slopes the filter took
        away (one small texture lookup); off, distant water is shaded as a
        flatter mirror
    :ivar reflection_blur: blur the reflection by those slopes, through its
        mipmaps (regenerated every frame); 0 = a sharp mirror and no mipmaps,
        1 = over all the ripples would have moved it (0.25 keeps the
        reflections readable at distance)
    :ivar seabed_cascades: the seabed's squares around the camera, finest
        first (at most 3; only baked once :meth:`Ocean.set_seabed` is given
        a sampler): fewer or coarser bake faster and sample the same
    :ivar depth_colour: tint the water over shallow seabed
    :ivar foam: shoreline foam (:meth:`Ocean.set_foam` turns it alone)
    :ivar foam_noise: break the foam up with noise; off, plain lines
    :ivar swell_by_depth: the swell dies out over the shallows
    """

    wave_iterations: int = 36
    aa_full_px: float = 8.0
    aa_gone_px: float = 4.0
    rough_fresnel: bool = True
    reflection_blur: float = 0.25
    seabed_cascades: tuple[Cascade, ...] = (
        Cascade(1000.0, 128),
        Cascade(8000.0, 192),
        Cascade(64000.0, 128),
    )
    depth_colour: bool = True
    foam: bool = True
    foam_noise: bool = True
    swell_by_depth: bool = True


#: Quality presets, named like the clouds' (``CloudQuality``); HIGH is the
#: default.
OCEAN_QUALITY = {
    "low": OceanQuality(
        wave_iterations=16,
        aa_full_px=12.0,
        aa_gone_px=6.0,
        rough_fresnel=False,
        reflection_blur=0.0,
        seabed_cascades=(Cascade(8000.0, 128), Cascade(64000.0, 64)),
        foam_noise=False,
    ),
    "mid": OceanQuality(
        wave_iterations=24,
        aa_full_px=10.0,
        aa_gone_px=5.0,
        seabed_cascades=(
            Cascade(1000.0, 128),
            Cascade(8000.0, 128),
            Cascade(64000.0, 128),
        ),
    ),
    "high": OceanQuality(),
    "ultra": OceanQuality(
        wave_iterations=48,
        aa_full_px=6.0,
        aa_gone_px=3.0,
        seabed_cascades=(
            Cascade(2000.0, 256),
            Cascade(8000.0, 256),
            Cascade(64000.0, 192),
        ),
    ),
}
#: Pre-tonemap exposure; the sky's and the terrain's default, which the
#: shader's decoding of the reflection assumes.
DEFAULT_EXPOSURE = 0.9

# ocean.frag's ACES fit, as row-major matrices (GLSL's mat3 literals are
# column-major, so these are their transposes).
_ACES_M1 = np.array(
    [
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ]
)
_ACES_M2 = np.array(
    [
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602],
    ]
)


def aces_tonemap(color: ArrayLike, exposure: float = DEFAULT_EXPOSURE) -> np.ndarray:
    """ocean.frag's ``aces_tonemap(color * exposure)``: linear light to the
    display colour (gamma 1/2.2) that the sky and terrain write.

    :param color: linear (r, g, b)
    :param exposure: pre-tonemap exposure
    :return: display (r, g, b) in [0, 1]
    """
    v = _ACES_M1 @ (np.asarray(color, dtype=np.float64) * exposure)
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return np.clip(_ACES_M2 @ (a / b), 0.0, 1.0) ** (1.0 / 2.2)


def compute_wave_dirs(wind_dir: LVecBase2f, wind_strength: float) -> PTA_LVecBase2f:
    """Precompute the per-iteration wave directions the fragment shader uses.

    These depend only on the iteration index and the wind, not on the pixel, so
    we build them once on the CPU and pass them as a uniform array instead of
    recomputing sin/cos/normalize/mix per pixel per iteration in the shader.
    Mirrors the original in-shader formula exactly.

    :param wind_dir: Wind direction in the XY plane (normalised).
    :param wind_strength: How strongly directions align to the wind; 0 =
        random, 1 = fully aligned.
    :return: One normalised wave direction per shader iteration.
    """
    pta = PTA_LVecBase2f()
    it = 0.0
    ws = wind_strength
    for _ in range(_MAX_WAVE_ITER):
        rx, ry = math.sin(it), math.cos(it)
        # mix(rndDir, wind_dir, ws)
        mx = rx * (1.0 - ws) + wind_dir.x * ws
        my = ry * (1.0 - ws) + wind_dir.y * ws
        length = max(math.hypot(mx, my), 1e-5)
        pta.pushBack(LVecBase2f(mx / length, my / length))
        it += 1232.399963
    return pta


def wave_weight_sum(iterations: int) -> float:
    """:returns: the sum of the small-wave octave weights over *iterations*
    octaves (ocean.frag's ``uWaveWeightSum``)"""
    return (1.0 - _WEIGHT_DECAY**iterations) / (1.0 - _WEIGHT_DECAY)


def wave_tail_variance(iterations: int) -> np.ndarray:
    """The slope variance of small-wave octaves i.. *iterations* - 1, for
    each i, normalised as ocean.frag normalises the normal: what a pixel
    whose filter stops at octave i leaves to the rough fresnel.

    :param iterations: the octave count (``OceanQuality.wave_iterations``)
    :returns: (_MAX_WAVE_ITER,) float64, zero from *iterations* on
    """
    i = np.arange(_MAX_WAVE_ITER)
    amp = (_FREQ_GROWTH * _WEIGHT_DECAY) ** i  # frequency * weight
    per = C_WAVE * amp**2 * (_WAVE_HEIGHT / wave_weight_sum(iterations)) ** 2
    per[iterations:] = 0.0
    return np.cumsum(per[::-1])[::-1]


def rough_fresnel_table(
    size: int = FRESNEL_LUT_SIZE,
    slope_max: float = FRESNEL_LUT_SLOPE_MAX,
    nodes: int = 24,
) -> np.ndarray:
    """Schlick's fresnel averaged over the facets of a rough surface, as seen:
    slopes Gaussian about the resolved normal (spread s, the rms slope
    magnitude), each facet weighted by its area projected toward the eye
    (facets turned away are hidden). ocean.frag reads it by (N.V, s).

    :param size: entries per axis
    :param slope_max: the spread of the last row
    :param nodes: Gauss-Hermite nodes per slope axis
    :returns: (size, size) float32, rows by spread, columns by N.V
    """
    x, w = np.polynomial.hermite.hermgauss(nodes)
    sx, sy = np.meshgrid(x, x)
    weight = np.outer(w, w)
    cos_v = np.linspace(0.0, 1.0, size)
    table = np.empty((size, size))
    for row, spread in enumerate(np.linspace(0.0, slope_max, size)):
        # Gauss-Hermite integrates exp(-x^2): scale to a per-axis variance
        # of spread^2 / 2.
        dx, dy = sx * spread, sy * spread
        inv = 1.0 / np.sqrt(1.0 + dx * dx + dy * dy)
        for col, c in enumerate(cos_v):
            # Facet normal (-dx, -dy, 1) * inv; eye toward (sin, 0, cos).
            cos_f = (-dx * math.sqrt(1.0 - c * c) + c) * inv
            seen = weight * np.maximum(cos_f, 0.0) / inv
            total = seen.sum()
            f = 0.04 + 0.96 * (1.0 - np.clip(cos_f, 0.0, 1.0)) ** 5
            table[row, col] = (seen * f).sum() / total if total > 1e-12 else 1.0
    return table.astype(np.float32)


def make_fresnel_texture() -> Texture:
    """:returns: :func:`rough_fresnel_table` as a linear, clamped float texture"""
    table = rough_fresnel_table()
    tex = Texture("ocean_fresnel")
    tex.setup_2d_texture(table.shape[1], table.shape[0], Texture.T_float, Texture.F_r32)
    tex.set_ram_image(np.ascontiguousarray(table).tobytes())
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex


def make_plane_mesh(size: float) -> GeomNode:
    """
    Single flat XY quad of the given world size, centred on the origin.

    The ocean surface has no vertical displacement — all wave detail is
    computed per-pixel in the fragment shader from the interpolated world
    position.  Across a flat triangle that interpolation is exact, so a single
    quad is pixel-identical to any tessellation: no subdivision is needed, and
    the vertex carries only its position (no unused normal/texcoord).

    :param size: Edge length of the quad, in world units.
    :return: A :class:`GeomNode` holding the flat quad.
    """
    fmt = GeomVertexFormat.getV3()
    vdata = GeomVertexData("ocean_plane", fmt, Geom.UHStatic)
    vdata.setNumRows(4)

    vw = GeomVertexWriter(vdata, "vertex")
    half = size * 0.5
    for x, y in ((-half, -half), (half, -half), (half, half), (-half, half)):
        vw.addData3f(x, y, 0.0)

    tris = GeomTriangles(Geom.UHStatic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(0, 2, 3)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("ocean_plane")
    node.addGeom(geom)
    return node


def border_coords(grid_half: float, outer_half: float, rings: int) -> np.ndarray:
    """
    Radii of the border rings between the dense grid and the surface's edge.

    Geometrically spaced, so the growth factor is derived rather than chosen: the
    last ring lands exactly on outer_half whatever the ring count.  Geometric
    rather than uniform because what has to be resolved is the planet's CURVE, and
    the error in approximating it by flat cells is ``spacing^2 / (8R)`` — so a ring
    schedule that grows with radius keeps the sag a roughly constant fraction of
    the distance it is seen at, instead of spending vertices out where nothing can
    be resolved anyway.

    :param grid_half: Half-size of the dense inner grid, in world units.
    :param outer_half: Half-size of the whole surface, in world units.
    :param rings: Number of border rings.
    :return: Ring radii, ascending, ending at outer_half.
    """
    growth = (outer_half / grid_half) ** (1.0 / rings)
    return grid_half * growth ** np.arange(1, rings + 1, dtype=np.float64)


def make_swell_grid_mesh(
    grid_half: float,
    subdivs: int,
    outer_half: float,
    curvature: float = 0.0,
    border_rings: int = 28,
    exact_sphere: bool = False,
) -> GeomNode:
    """
    Mesh for the geometric-swell mode: a dense uniform grid of half-size
    grid_half (displaced by the swell in the vertex shader, tapered to zero
    before its edge), ringed by geometrically-spaced border rings out to
    outer_half.  Rings rather than one border quad so the surface can curve;
    28 keeps the curve's sag under 1 arcmin.  Curvature is baked into z, which
    is exact because the mesh is recentred on the camera every frame.

    :param grid_half: Half-size of the dense inner grid, in world units.
    :param subdivs: Number of subdivisions across the dense inner grid.
    :param outer_half: Half-size of the outer border, in world units.
    :param curvature: 1 / (2 * planet radius); 0 for a flat surface.
    :param border_rings: Number of geometrically-spaced border rings.
    :param exact_sphere: Bake the sphere of that radius instead of the
        paraboloid d^2 / 2R. They part by d^4 / 8R^3 (58 m at 100 km on a
        600 km planet), so an ocean meeting real spherical terrain needs it;
        one meeting the other drooped objects (clouds) must keep the
        paraboloid.
    :return: A :class:`GeomNode` holding the grid mesh.
    """
    # Per-axis coordinates: the border rings mirrored below the grid, the uniform
    # inner span, then the border rings above it. Vectorised: the default mesh is
    # ~324k verts.
    border = border_coords(grid_half, outer_half, border_rings)
    inner = -grid_half + 2.0 * grid_half * np.arange(subdivs + 1) / subdivs
    coords = np.concatenate([-border[::-1], inner, border]).astype(np.float32)
    m = coords.size

    # Vertex positions (m*m, 3): vertex j*m + i sits at (coords[i], coords[j]).
    verts = np.zeros((m * m, 3), dtype=np.float32)
    verts[:, 0] = np.tile(coords, m)
    verts[:, 1] = np.repeat(coords, m)
    # Droop from the true radius, so the grid samples a paraboloid of revolution
    # (or the sphere itself).
    r2 = verts[:, 0].astype(np.float64) ** 2 + verts[:, 1].astype(np.float64) ** 2
    if exact_sphere and curvature > 0.0:
        radius = 1.0 / (2.0 * curvature)
        verts[:, 2] = np.sqrt(np.maximum(radius * radius - r2, 0.0)) - radius
    else:
        verts[:, 2] = -curvature * r2

    fmt = GeomVertexFormat.getV3()
    vdata = GeomVertexData("ocean_grid", fmt, Geom.UHStatic)
    vdata.setNumRows(m * m)
    memoryview(vdata.modifyArray(0)).cast("B")[: verts.nbytes] = verts.tobytes()

    # Two triangles per cell:
    #   (v0, v0+1, v0+m) and (v0+1, v0+m+1, v0+m), with v0 = j*m + i.
    jj, ii = np.meshgrid(np.arange(m - 1), np.arange(m - 1), indexing="ij")
    v0 = (jj * m + ii).ravel().astype(np.uint32)
    tri = np.empty((v0.size, 6), dtype=np.uint32)
    tri[:, 0] = v0
    tri[:, 1] = v0 + 1
    tri[:, 2] = v0 + m
    tri[:, 3] = v0 + 1
    tri[:, 4] = v0 + m + 1
    tri[:, 5] = v0 + m
    indices = tri.reshape(-1)

    tris = GeomTriangles(Geom.UHStatic)
    # >65535 verts at the default subdivisions, so indices must be 32-bit.
    tris.setIndexType(GeomEnums.NT_uint32)
    tris.addNextVertices(indices.size)
    memoryview(tris.modifyVertices()).cast("B")[: indices.nbytes] = indices.tobytes()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("ocean_grid")
    node.addGeom(geom)
    return node


class Ocean:
    """A camera-locked ocean surface, its planar reflection and its shader.

    :ivar base_node: the ocean's frame (under ``game.root_node``); the water
        is its plane z = 0
    :ivar ocean_node: the surface mesh, carrying the shader inputs
    :ivar refl_buffer: the offscreen reflection buffer
    :ivar refl_tex: its texture
    :ivar refl_cam: the mirrored reflection camera's NodePath
    """

    def __init__(
        self,
        game: Any,
        water_color: LVecBase3f = LVecBase3f(0.02, 0.06, 0.14),
        ripple_strength: float = 0.02,
        wind_dir: LVecBase2f = LVecBase2f(1.0, 0.0),
        wind_strength: float = 0.5,
        quality: OceanQuality = OCEAN_QUALITY["high"],
        wave_iterations: int | None = None,
        geometric_swell: bool = False,
        swell_grid_subdivs: int = 512,
        swell_grid_half: float = 4000.0,
        swell_amplitude: float = 20.0,
        swell_scale: float = 0.004,
        swell_drift: float = 0.1,
        fm_depth: float = 0.2,
        haze_color: LVecBase3f = LVecBase3f(0.2, 0.2, 0.4),
        haze_distance: float = 150000.0,
        shallow_color: ArrayLike = SHALLOW_WATER_COLOR,
        shallow_depth: float = SHALLOW_WATER_DEPTH_M,
        foam_color: ArrayLike = (0.8, 0.8, 0.8),
        foam_depth: float = 2.5,
        foam_spacing: float = 0.6,
        foam_speed: float = 1.2,
        foam_strength: float = 0.85,
        swell_shallow: float = 5.0,
        swell_deep: float = 60.0,
        planet_radius: float = PLANET_RADIUS_M,
        exact_sphere: bool = False,
        reflection_scale: float | None = None,
        vert_shader: Path = _SHADERS / "ocean.vert",
        frag_shader: Path = _SHADERS / "ocean.frag",
    ) -> None:
        """
        Build the ocean surface, its shared reflection buffer, and shader.

        :param game: The game instance.
        :param water_color: Deep-water scatter colour.
        :param ripple_strength: Normal-based UV perturbation strength.
        :param wind_dir: Wind direction in the XY plane (normalised).
        :param wind_strength: How strongly the waves align to the wind; 0 =
            random, 1 = fully aligned.
        :param quality: the cost/quality trades (:data:`OCEAN_QUALITY`);
            :meth:`set_quality` changes them later.
        :param wave_iterations: overrides ``quality.wave_iterations``.
        :param geometric_swell: Prototype mode that geometrically displaces the
            surface by the swell rather than only shading it.
        :param swell_grid_subdivs: Tessellation of the dense centre grid used by
            the geometric-swell mode.
        :param swell_grid_half: Half-size of the dense grid, in world units.
        :param swell_amplitude: Vertical swell displacement in deep water, in
            world units (it dies out over the shallows: swell_shallow,
            swell_deep).
        :param swell_scale: Swell spatial frequency, shared by the vertex and
            fragment shaders; a long wavelength so the geometric-swell mesh
            samples it without aliasing.
        :param swell_drift: Swell scroll speed along the wind, shared by the
            vertex and fragment shaders.
        :param fm_depth: Small-wave frequency-modulation depth, in radians; the
            swell field modulates the wave phase to break the dominant octave's
            tiling (0 = off). Fragment-only; does not affect the geometric-swell
            displacement.
        :param haze_color: Colour distant water fades toward. Should match what
            the cloud field fades to, or sea and sky disagree at the horizon.
        :param haze_distance: Metres over which water reaches haze_color fully.
        :param shallow_color: The water's colour over shallow seabed (linear;
            water_color is the deep one). Needs a seabed (:meth:`set_seabed`).
        :param shallow_depth: Metres of water over which the deep colour
            takes over (Beer-Lambert e-folding depth). Give the terrain's sea
            surface the same two (``Terrain.set_shallow_water``).
        :param foam_color: Shoreline foam's colour in daylight (linear).
        :param foam_depth: Foam shows under this many metres of water.
        :param foam_spacing: Metres of depth between two foam lines.
        :param foam_speed: How fast they run shoreward (radians a second).
        :param foam_strength: Foam's opacity at its thickest.
        :param swell_shallow: Depth, metres, where the swell starts...
        :param swell_deep: ...and where it reaches swell_amplitude.
        :param planet_radius: Radius the surface curves with (its droop is
            d^2 / 2R) and whose horizon it must reach, in metres. Must match
            the clouds' and the terrain's.
        :param exact_sphere: Curve the geometric-swell surface on the exact
            sphere rather than the d^2 / 2R paraboloid (see
            :func:`make_swell_grid_mesh`): for real spherical terrain.
        :param reflection_scale: the reflection's resolution as a fraction of
            the view's; None = the game's graphics settings, or
            :data:`DEFAULT_REFLECTION_SCALE` without them
        :param vert_shader: Path to ocean.vert.
        :param frag_shader: Path to ocean.frag.
        """
        self.game = game
        self.id = uuid.uuid4()
        self.base_node = self.game.root_node.attachNewNode("ocean")
        self._exposure = DEFAULT_EXPOSURE

        # ── Reflection buffer ─────────────────────────────────────────────────
        settings = getattr(self.game.app, "graphics_settings", None)
        if reflection_scale is not None:
            refl_scale = reflection_scale
        elif settings is not None:
            refl_scale = settings.config["render"]["reflection_scale"]
        else:
            refl_scale = DEFAULT_REFLECTION_SCALE
        self.refl_tex, uv_scale = self.make_reflection_buffer(refl_scale, water_color)

        # ── Shader ────────────────────────────────────────────────────────────
        shader = Shader.load(Shader.SL_GLSL, vertex=vert_shader, fragment=frag_shader)

        # ── Camera-locked surface ─────────────────────────────────────────────
        # Follows the camera in XY each frame so the view is always covered to the
        # horizon. The reflection mirror (and a game's collision plane) stay flat
        # at z=0; both are exact at the camera, where the droop is zero.
        horizon = math.sqrt(2.0 * planet_radius * _MAX_EYE_ALTITUDE_M)
        outer_half = horizon * _HORIZON_MARGIN
        if geometric_swell:
            mesh = make_swell_grid_mesh(
                swell_grid_half,
                swell_grid_subdivs,
                outer_half,
                curvature=1.0 / (2.0 * planet_radius),
                exact_sphere=exact_sphere,
            )
        else:
            # Four vertices cannot curve, so this path stays flat.
            mesh = make_plane_mesh(outer_half * 2.0)
        self.ocean_node = self.base_node.attachNewNode(mesh)
        self.ocean_node.setShader(shader)
        self.ocean_node.setShaderInput("iTime", 0.0)
        self.ocean_node.setShaderInput("iCameraPos", LVecBase3f(0, 0, 20))
        self.ocean_node.setShaderInput("uCamForward", LVecBase3f(0, 1, 0))
        self.ocean_node.setShaderInput("uPixelAngle", 0.001)
        self.ocean_node.setShaderInput("uModelMatrix", LMatrix4f.identMat())
        self.ocean_node.setShaderInput("iWaterColor", water_color)
        self.ocean_node.setShaderInput("iWaterLight", LVecBase3f(1.0, 1.0, 1.0))
        self._water_color = LVecBase3f(water_color)
        self.ocean_node.setShaderInput("iReflectionTex", self.refl_tex)
        self.ocean_node.setShaderInput("iRippleStrength", ripple_strength)
        self.ocean_node.setShaderInput("iWindDir", wind_dir)
        self.ocean_node.setShaderInput("iWindStrength", wind_strength)
        self.ocean_node.setShaderInput(
            "iWaveDirs", compute_wave_dirs(wind_dir, wind_strength)
        )
        self.ocean_node.setShaderInput("uFresnelLUT", make_fresnel_texture())
        # The seabed: nothing until the app hands over a sampler.
        self.seabed = SeabedMap(self.ocean_node, quality.seabed_cascades)
        self.ocean_node.setShaderInput("uShallowColor", LVecBase3f(*shallow_color))
        self.ocean_node.setShaderInput("uShallowDepth", float(shallow_depth))
        self.ocean_node.setShaderInput("uFoamColor", LVecBase3f(*foam_color))
        self.ocean_node.setShaderInput("uFoamDepth", float(foam_depth))
        self.ocean_node.setShaderInput("uFoamSpacing", float(foam_spacing))
        self.ocean_node.setShaderInput("uFoamSpeed", float(foam_speed))
        self.ocean_node.setShaderInput("uFoamStrength", float(foam_strength))
        self.ocean_node.setShaderInput("uSwellShallow", float(swell_shallow))
        self.ocean_node.setShaderInput("uSwellDeep", float(swell_deep))
        if wave_iterations is not None:
            quality = replace(quality, wave_iterations=wave_iterations)
        self.set_quality(quality)
        self.ocean_node.setShaderInput("uReflMVP", LMatrix4f.identMat())
        self.ocean_node.setShaderInput("uReflUVScale", uv_scale)
        # Geometric-swell prototype: vertex-shader displacement (off by default).
        self.ocean_node.setShaderInput("uGeometricSwell", 1 if geometric_swell else 0)
        self.set_swell_amplitude(swell_amplitude)
        self.ocean_node.setShaderInput("uSwellGridHalf", float(swell_grid_half))
        # Swell frequency/drift shared by the vertex and fragment shaders.
        self.ocean_node.setShaderInput("iSwellScale", float(swell_scale))
        self.ocean_node.setShaderInput("iSwellDrift", float(swell_drift))
        # Swell-driven frequency modulation of the small waves (frag-only).
        self.ocean_node.setShaderInput("uFmDepth", float(fm_depth))
        self.ocean_node.setShaderInput("uDebugMode", 0)
        self.ocean_node.setShaderInput("uWaveOff", 0)
        self.ocean_node.setShaderInput("uExposure", self._exposure)
        # Aerial perspective, to meet the cloud haze at the horizon.
        self.ocean_node.setShaderInput("uHazeColor", haze_color)
        self.ocean_node.setShaderInput("uHazeDistance", float(haze_distance))
        # No atmosphere unless a world binds its own (atmosphere.Atmospheres):
        # then they replace the haze. The base frame's rotation and the sun
        # are the binder's to keep current (set_atmosphere_frame).
        bind_defaults(self.ocean_node)
        self.ocean_node.setShaderInput("uBaseToWorld", LMatrix4f.identMat())
        self.ocean_node.setShaderInput("uAtmoSunDir", LVecBase3f(0.0, 0.0, 1.0))
        # Not in its own mirror.
        self.ocean_node.hide(REFLECTION_CAMERA_MASK)

        self.game.method_lists[self.id] = [self.update]

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self) -> None:
        """Per-frame update (self-registered in ``game.method_lists``)."""
        current_time = self.game.game_time.get_current_time()
        camera_pos = self.game.app.camera.getPos(self.base_node)
        self.mirror_camera()
        # Follow the main camera's field of view (window resizes, zoom). Its
        # far plane is not followed: world_builder.depth sets the main lens
        # to the near camera's short range every frame.
        fov = self.game.app.camLens.getFov()
        self.refl_cam_node.getLens().setFov(
            fov.x * REFLECTION_FOV_SCALE, fov.y * REFLECTION_FOV_SCALE
        )

        # Reflection view-projection, for the shaders' reflection lookup
        view = LMatrix4f()
        view.invertFrom(self.refl_cam.getMat(self.base_node))
        proj = self.refl_cam_node.getLens().getProjectionMat()
        mvp = view * proj

        # The valid region of the reflection texture is buffer_size /
        # texture_size. The texture is padded to a power of two only on some
        # pipelines (the default one pads, simplepbr does not), and its real size
        # is known only once the buffer is realized, so refresh uReflUVScale from
        # the actual dimensions on the first realized frame.
        if not hasattr(self, "_uv_scale_set"):
            tex_w = self.refl_tex.getXSize()
            tex_h = self.refl_tex.getYSize()
            if tex_w > 0 and tex_h > 0:
                self._uv_scale_set = True
                uv_scale = LVecBase2f(
                    self.refl_buffer.getXSize() / tex_w,
                    self.refl_buffer.getYSize() / tex_h,
                )
                self.ocean_node.setShaderInput("uReflUVScale", uv_scale)

        # Keep the plane centred under the camera so the view is always covered.
        # The wave pattern is anchored in world space (shader reads world
        # position), so sliding the plane introduces no motion artifacts.
        self.ocean_node.setPos(camera_pos.x, camera_pos.y, 0.0)
        self.ocean_node.setShaderInput(
            "uModelMatrix", self.ocean_node.getMat(self.base_node)
        )
        self.ocean_node.setShaderInput("iTime", current_time)
        self.ocean_node.setShaderInput("iCameraPos", camera_pos)
        self.ocean_node.setShaderInput(
            "uCamForward",
            self.game.app.camera.getQuat(self.base_node).getForward(),
        )
        self.ocean_node.setShaderInput("uPixelAngle", self._pixel_angle(fov.y))
        self.ocean_node.setShaderInput("uReflMVP", mvp)
        self.seabed.update((camera_pos.x, camera_pos.y))

    def set_atmosphere_frame(
        self, base_to_world: LMatrix4f, sun_direction: ArrayLike
    ) -> None:
        """For a bound atmosphere: how the base node's axes sit in the world
        (the atmospheres work in world axes) and where the sun is.

        :param base_to_world: the base node's transform relative to render
            (an LMatrix4f, e.g. ``base_node.get_mat(render)``; only its
            rotation is used)
        :param sun_direction: toward the sun, world axes
        """
        self.ocean_node.setShaderInput("uBaseToWorld", base_to_world)
        self.ocean_node.setShaderInput("uAtmoSunDir", LVecBase3f(*sun_direction))

    def set_light(self, light: ArrayLike) -> None:
        """How lit the water is, relative to the daylight its colour was
        chosen in: (r, g, b) multiplying the water's own scattered colour
        (``iWaterLight``) and the colour the reflection is cleared to (seen
        where it renders nothing; 1 = as tuned, 0 = night). The sky's
        reflection darkens by itself.

        :param light: linear (r, g, b) factors
        """
        light = LVecBase3f(*light)
        self.ocean_node.setShaderInput("iWaterLight", light)
        self.refl_buffer.setClearColor(self._clear_color(self._water_color, light))

    def set_quality(self, quality: OceanQuality) -> None:
        """Change the cost/quality trades live (e.g. from a settings menu).

        :param quality: one of :data:`OCEAN_QUALITY`, or a custom one
        """
        iterations = max(1, min(int(quality.wave_iterations), _MAX_WAVE_ITER))
        self.quality = replace(quality, wave_iterations=iterations)
        node = self.ocean_node
        node.setShaderInput("iIterationsNormal", iterations)
        node.setShaderInput("uWaveWeightSum", wave_weight_sum(iterations))
        tail = PTA_float()
        for v in wave_tail_variance(iterations):
            tail.push_back(float(v))
        node.setShaderInput("uTailVar", tail)
        # Pixels per wavelength to phase per pixel.
        node.setShaderInput("uAAStart", 2.0 * math.pi / quality.aa_full_px)
        node.setShaderInput("uAAEnd", 2.0 * math.pi / quality.aa_gone_px)
        node.setShaderInput("uRoughFresnel", int(quality.rough_fresnel))
        node.setShaderInput("uDepthColourOn", int(quality.depth_colour))
        node.setShaderInput("uFoamOn", int(quality.foam))
        node.setShaderInput("uFoamNoiseOn", int(quality.foam_noise))
        node.setShaderInput("uSwellByDepth", int(quality.swell_by_depth))
        if self.seabed.cascades != tuple(quality.seabed_cascades):
            self.seabed.set_cascades(quality.seabed_cascades)
        # The reflection's mipmaps cost a regeneration every frame: only kept
        # when the blur reads them.
        blur = max(0.0, float(quality.reflection_blur))
        node.setShaderInput("uReflBlur", blur)
        self.refl_tex.set_minfilter(
            SamplerState.FT_linear_mipmap_linear
            if blur > 0.0
            else SamplerState.FT_linear
        )

    def set_seabed(self, sampler: Sampler | None) -> None:
        """Give the ocean the sea floor under it, for its colour by depth, its
        shoreline foam and its swell dying out over the shallows. Without one
        the sea is deep everywhere.

        The sampler is called on a worker thread, a cascade at a time
        (:class:`world_builder.ocean.seabed.SeabedMap`), so it must be safe to
        call from one: reading a store the main thread fills is fine.

        :param sampler: (N, 3) float64 points of the ocean's plane, in the
            world :meth:`set_seabed_frame` places it in, and the texel size
            in metres (a resolution hint) -> (N,) the sea floor's height
            under each, metres, negative under water; None = no seabed
        """
        self.seabed.set_sampler(sampler)

    def set_seabed_frame(
        self, origin: ArrayLike, east: ArrayLike, north: ArrayLike
    ) -> None:
        """Where :attr:`base_node`'s plane z = 0 is in the seabed sampler's
        world (float64): needed only where that frame moves, e.g. a tangent
        plane following the camera over a planet. Call it before
        :meth:`update`, every frame it changes; the default is the plane
        itself (origin 0, x east, y north).

        :param origin: the plane's (0, 0)
        :param east: its unit x axis
        :param north: its unit y axis
        """
        self.seabed.set_frame(origin, east, north)

    def set_swell_amplitude(self, amplitude: float) -> None:
        """The swell's height in deep water, metres, live: the geometric
        swell's displacement (``geometric_swell=True``) and the shading's
        tilt both follow it. The field mostly spans 0.1 to 0.5 of it, so
        crests stand some 0.4 times this above the troughs.

        :param amplitude: metres; the swell's wavelength (about
            1 / ``swell_scale``, 250 m) should stay well above it
        """
        self.swell_amplitude = float(amplitude)
        self.ocean_node.setShaderInput("uSwellAmplitude", self.swell_amplitude)

    def set_foam(self, enabled: bool) -> None:
        """Shoreline foam on or off (the quality's ``foam``, alone)."""
        self.quality = replace(self.quality, foam=bool(enabled))
        self.ocean_node.setShaderInput("uFoamOn", int(enabled))

    def set_debug_mode(self, mode: int) -> None:
        """Debug view (``uDebugMode`` in ocean.frag): 0 shaded, 1 normal,
        2 reflection UV, 3 fresnel, 4 world grid, 5 reflection UV clamp,
        6 raw reflection, 7 pre-tonemap clipping, 8 FM phase, 9 share of the
        wave octaves resolved, 10 unresolved slope spread, 11 wave octaves
        evaluated (the cost), 12 depth (black at the waterline, white at
        50 m; red on land, blue beyond the seabed's cascades)."""
        self.ocean_node.setShaderInput("uDebugMode", int(mode))

    def set_wave_iterations(self, iterations: int) -> None:
        """Adjust wave-detail quality at runtime (e.g. from a settings menu).

        Higher = sharper waves up close but more expensive per pixel; lower =
        smoother and cheaper. Far and grazing water evaluates fewer anyway,
        as its waves go sub-pixel.

        :param iterations: Wave-detail iteration count, clamped to the shader's
            maximum.
        """
        self.set_quality(replace(self.quality, wave_iterations=iterations))

    # ── Internal ──────────────────────────────────────────────────────────────

    def make_reflection_buffer(
        self, refl_scale: float, water_color: LVecBase3f
    ) -> tuple[Texture, LVecBase2f]:
        """
        Create the offscreen buffer and mirrored camera for planar reflections.

        :param refl_scale: Reflection resolution as a fraction of the 3D render
            size (< 1 renders reflections cheaper than the main view).
        :param water_color: Colour the buffer is cleared to, seen where the
            reflection camera renders nothing.
        :return: The reflection :class:`Texture` and the provisional UV scale
            (buffer size / padded texture size) for the first frame.
        """
        # Size off the 3D render resolution (which may be a downscale of the
        # window when render-scale is active) rather than the window itself, so
        # the reflection cost scales with the chosen internal resolution.
        render_w, render_h = self._render_size()
        buf_w = max(1, int(render_w * refl_scale))
        buf_h = max(1, int(render_h * refl_scale))

        refl_tex = Texture("refl_tex")
        refl_tex.setWrapU(Texture.WMClamp)
        refl_tex.setWrapV(Texture.WMClamp)

        self.refl_buffer = self.game.app.win.makeTextureBuffer(
            "refl_buffer", buf_w, buf_h, refl_tex
        )
        self.refl_buffer.setSort(-100)
        # Clear the reflection buffer to the water colour, so wherever the
        # reflection camera renders nothing (no skybox/geometry) the reflection
        # blends into the water.
        self.refl_buffer.setClearColor(self._clear_color(water_color))

        self.refl_cam_node = Camera("refl_cam")
        # Use a wider lens than the main camera (REFLECTION_FOV_SCALE times its
        # FOV) to avoid frustum culling artifacts when turning quickly — the
        # extra margin ensures reflected objects are never culled at the edges
        # of the reflection buffer. update() follows the main camera's FOV;
        # near and far are the ones taken here.
        refl_lens = self.game.app.camLens.makeCopy()
        fov = self.game.app.camLens.getFov()
        refl_lens.setFov(fov.x * REFLECTION_FOV_SCALE, fov.y * REFLECTION_FOV_SCALE)
        # The reflection camera sits ~2*altitude further from the skybox's far
        # side than the main camera, so with the same far plane a skybox that is
        # real geometry could be clipped at the reflected zenith. Push the
        # reflection far plane well past it so it always renders in full.
        # (world_builder's Sky needs none of it: sky.vert puts it on the far
        # plane whatever the range.)
        refl_lens.setFar(self.game.app.camLens.getFar() * 4.0)
        self.refl_cam_node.setLens(refl_lens)
        self.refl_cam = self.base_node.attachNewNode(self.refl_cam_node)

        self.refl_cam_node.setCameraMask(REFLECTION_CAMERA_MASK)

        # TODO(perf): the reflection re-renders all scene geometry. Objects that
        # contribute little (small, distant) could be kept out of it with
        # ``node.hide(REFLECTION_CAMERA_MASK)``, as the ocean itself is. Worth
        # it with many minor objects, not for one large hero object.

        self.refl_buffer.makeDisplayRegion(0, 1, 0, 1).setCamera(self.refl_cam)

        clip_plane = PlaneNode("water_clip")
        clip_plane.setPlane(LPlane(0, 0, 1, 0))
        self.clip_np = self.base_node.attachNewNode(clip_plane)
        clip_attrib = ClipPlaneAttrib.makeAllOff().addOnPlane(self.clip_np)
        self.refl_cam_node.setInitialState(
            self.refl_cam_node.getInitialState().addAttrib(clip_attrib)
        )

        # Provisional UV scale assuming power-of-two padding; update() corrects it
        # from the real texture size on the first realized frame.
        uv_scale = LVecBase2f(
            buf_w / compute_next_power_of_2(buf_w),
            buf_h / compute_next_power_of_2(buf_h),
        )
        return refl_tex, uv_scale

    def _render_size(self) -> tuple[int, int]:
        """:returns: the 3D render's size in pixels (a downscale of the window
        when the app's render scale is active)"""
        manager = getattr(self.game.app, "graphics_manager", None)
        if manager is not None:
            return manager.get_render_size()
        return self.game.app.win.get_x_size(), self.game.app.win.get_y_size()

    def _pixel_angle(self, fov_y: float) -> float:
        """:returns: a pixel's angle at the screen's centre, radians (the
        shader's wave filter measures the water's footprint from it)

        :param fov_y: the lens's vertical field of view, degrees
        """
        height = max(1, self._render_size()[1])
        return 2.0 * math.tan(math.radians(fov_y) / 2.0) / height

    def _clear_color(
        self, water_color: ArrayLike, light: ArrayLike = (1.0, 1.0, 1.0)
    ) -> LVecBase4f:
        """The reflection buffer's clear colour: the lit water colour as the
        display colour the shader decodes the reflection from (like the sky
        and terrain drawn into it, it is tonemapped).

        :param water_color: linear (r, g, b)
        :param light: linear (r, g, b) factors (:meth:`set_light`)
        :return: display RGBA
        """
        lit = np.asarray(water_color, dtype=np.float64) * np.asarray(light)
        return LVecBase4f(*aces_tonemap(lit, self._exposure), 1.0)

    def mirror_camera(self) -> None:
        """Mirror the reflection camera across the water plane (Z = 0)."""
        pos = self.game.app.camera.getPos(self.base_node)
        hpr = self.game.app.camera.getHpr(self.base_node)
        self.refl_cam.setPos(self.base_node, pos.x, pos.y, -pos.z)
        self.refl_cam.setHpr(self.base_node, hpr.x, -hpr.y, -hpr.z)

    def clean(self) -> None:
        """
        Cleans the Ocean object: unregisters its update, destroys the
        reflection buffer and removes its nodes.
        """
        if self.game.method_lists:
            self.game.method_lists.pop(self.id, None)
        # Destroy the offscreen reflection buffer; removing base_node alone
        # leaves the buffer registered with the graphics engine, so it keeps
        # rendering its camera every frame (a leak across scene reloads).
        self.game.app.graphicsEngine.removeWindow(self.refl_buffer)
        self.seabed.destroy()
        self.base_node.removeNode()
        self.game = None
