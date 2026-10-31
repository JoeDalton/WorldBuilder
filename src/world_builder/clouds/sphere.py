"""
A planet's low clouds as a volume (:class:`CloudVolume`), from inside the deck
to orbit, with one pattern. :class:`planet.PlanetClouds` puts it under the
cirrus sheet (:mod:`cirrus`).

The density field is the billboard field's (:mod:`noise`), moved onto the
planet: the slab is a height above the sea-level sphere, the tileable noise is
sampled at planet-frame positions (3D, so it has no seam anywhere on the
sphere), and the weather (:mod:`weather`: a cube map of coverage following the
planet's circulation, weather systems, climate and terrain) shifts the fbm by
exactly what gives its coverage there. Every view of the clouds evaluates that
same function, so the pattern seen from orbit is the one you fly into:

* **near** (the inner LOD shells): volumetric billboards, as the flat field
  draws them, generated per cube-sphere tile in the shared worker pool by
  rejection sampling of the density. Each shell doubles the billboards' size
  and the tiles' (``lod_shells``' trade: each shell costs half the previous)
  and crossfades with the next by distance, on optical depth. Below ULTRA the
  far shells are cheaper per pixel: fewer octaves (``CloudSettings.shell_octaves``; the
  dropped ones count as their mean and the shell's threshold moves to make up
  for the lost spread, so the coverage holds) and, at MID and LOW, fewer
  overlapping billboards (``overlap_near``, ``overlap_far``). The vertex stage drops a
  billboard outside its shell's band, faded out near the camera, or off
  screen, before working out its air and sunlight.
* **far** (beyond the last shell, up to orbit): the *deck*, a sphere at
  mid-slab height whose fragments evaluate the density through the slab's
  depth. Billboards would thin out at that scale (their covered area falls as
  1 / size); the deck covers the planet with a few thousand triangles.
  Where a pixel spans more than the clouds' features it fades to the
  weather's expected coverage, lit by the average of its own close-up
  self-shadowing (:func:`noise.deck_light_table`, by the sun's elevation and
  the coverage): no change of colour closing in, and a terminator in the
  reddened sunlight's colour.

The clouds are geometry that depth-tests against the scene, like the flat
field: terrain occludes them with no depth plumbing. Premultiplied "over"
needs back to front: the deck draws first, then each shell in its own
back-to-front bin, coarsest first (a coarse tile holding the camera has its
centre near it, but its visible billboards are all farther than the finer
shells'); within a tile the particles are re-sorted, a few tiles a frame,
the stalest first. Seen through the atmospheres
(``atmosphere.Atmospheres``), lit by sunlight reddened by the planet's own
air; drawn in the ocean's reflection too (all of it is above the sea, so the
shaders keep it whole against the mirror's clip plane). Static: no wind yet.

Design notes: docs/source/atmosphere.md, "Planet clouds".
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BoundingBox,
    ColorBlendAttrib,
    CullBinManager,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    InternalName,
    LVecBase3f,
    LVecBase4f,
    NodePath,
    OmniBoundingVolume,
    Point3,
    PTA_LVecBase3f,
    PTA_LVecBase4f,
    Shader,
    TransparencyAttrib,
    Vec3,
)

from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.clouds.cloud import (
    MAX_SHELLS,
    PRESETS,
    CloudSettings,
    CloudType,
    atlas_mean_alpha,
    atlas_rects,
    load_cloud_atlas,
    with_settings,
)
from world_builder.clouds.noise import (
    DECK_LIGHT_COS_MIN,
    LAYER_VEC4S,
    MAX_LAYERS,
    NOISE_SIZE,
    CloudOptics,
    DensityField,
    build_deck_light_texture,
    build_noise_texture,
    deck_light_table,
    density_on_sphere,
    fbm_sigma,
    pack_layer_params,
    quantise,
    sphere_column_peaks,
    value_noise_volume,
    with_threshold,
)
from world_builder.clouds.weather import (
    WeatherSpec,
    build_weather_texture,
    shift_table,
    weather_cube,
)
from world_builder.depth import REFLECTION_CAMERA_MASK
from world_builder.quality import REBUILD
from world_builder.terrain.cubesphere import QUADS, CubeGrid, face_dirs, grid_for
from world_builder.terrain.noise import native_library
from world_builder.terrain.planet import PlanetSpec
from world_builder.terrain.quadtree import LodSelector
from world_builder.terrain.workers import SharedArray, shared_array

if TYPE_CHECKING:
    from world_builder.terrain.workers import ChunkExecutor

_SHADERS = Path(__file__).parent / "shaders"
#: Cull bins, between opaque (20) and transparent (30), after the sky's (21,
#: ``sky.SKY_BIN_SORT``): the deck's first, then one back-to-front bin per
#: shell, the coarsest first. A cirrus sheet above (:mod:`cirrus`) draws in
#: the sky's bin where it is behind them and in "transparent" where it is in
#: front.
CLOUD_BIN = "cloud_sphere"
_CLOUD_BIN_SORT = 22
#: The shells' bin sorts, shell 0 (the nearest, drawn last) first: at most
#: :data:`cloud.MAX_SHELLS` shells. 25 is skipped: it is the flat field's bin
#: (``field._CLOUD_BIN_SORT``), and a tie would leave the two kinds' order
#: undefined in one scene.
_SHELL_BIN_SORTS = (29, 28, 27, 26, 24, 23)
#: Side of a first-shell tile, metres (a quarter of the flat field's 32 km box),
#: rounded to the nearest level of the planet's grid.
BASE_TILE_SIZE = 8000.0
#: Fewest candidate points per rejection-sampling batch of a tile: enough to
#: measure the accept rate its billboard count follows.
MIN_TILE_BATCH = 2048
#: Quads per face edge of the deck sphere.
DECK_QUADS = 48
#: The deck's optical depth cap (the billboards' is ~1.5 per quad).
DECK_MAX_OPTICAL_DEPTH = 6.0
Key = tuple[int, int, int, int]
_CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float32)
_QUAD_TRIS = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


# ── What the workers need ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShellSpec:
    """One billboard LOD shell.

    :ivar level: cube-sphere level of its tiles
    :ivar radius: (min, max) billboard radius, metres (at the settings' count)
    :ivar volume_fraction: billboard sphere volume per unit cloud volume
    :ivar octaves: noise octaves evaluated (placement and drawing agree)
    :ivar threshold_offset: added to the field's threshold, so the coverage
        holds with fewer octaves (:func:`octave_offset`)
    """

    level: int
    radius: tuple[float, float]
    volume_fraction: float
    octaves: int = 4
    threshold_offset: float = 0.0

    def field_of(self, field: DensityField) -> DensityField:
        """:returns: *field* as this shell evaluates it"""
        lo, hi = field.threshold
        offset = self.threshold_offset
        return replace(field, threshold=(lo + offset, hi + offset))


@dataclass(frozen=True)
class SphereCloudSpec:
    """Everything a worker needs to fill one tile; frozen and picklable.

    :ivar radius: the planet's sea-level radius, metres
    :ivar field: the cloud type's density field, threshold resolved
    :ivar noise_seed: seed of the shared noise volume
    :ivar weather: the planet's weather
    :ivar shift_table: the fbm shift per coverage (:func:`weather.shift_table`)
    :ivar shells: the billboard shells, innermost first
    :ivar shift: the weather map's fbm shift plane, shared with the workers
        (a :attr:`terrain.workers.SharedArray.handle`); None = each worker
        builds the map from *weather* (~0.6 s)
    """

    radius: float
    field: DensityField
    noise_seed: int
    weather: WeatherSpec
    shift_table: tuple
    shells: tuple
    shift: tuple | None = None

    def shell_of(self, level: int) -> int:
        """:returns: the index of the shell whose tiles are at *level*"""
        return level - self.shells[0].level


@dataclass
class CloudTile:
    """One tile's billboards; picklable.

    :ivar key: (level, face, i, j) on the planet's cube-sphere grid
    :ivar version: spec version it was generated for
    :ivar centre: (3,) float64 planet-frame point the particles are relative to
    :ivar particles: (n, 9) float32: x, y, z (relative to centre), radius, sprite
        rect (u, v, du, dv), layer (shell index)
    :ivar gen_ms: generation time, milliseconds
    """

    key: Key
    version: int
    centre: np.ndarray
    particles: np.ndarray
    gen_ms: float


@functools.cache
def _noise(seed: int) -> np.ndarray:
    return quantise(value_noise_volume(seed=seed))


@functools.cache
def _weather(weather: WeatherSpec, table: tuple) -> np.ndarray:
    return weather_cube(weather, np.asarray(table))


def _shift_plane(spec: SphereCloudSpec) -> np.ndarray:
    """:returns: (6, n, n) the weather's fbm shift, shared or built here"""
    if spec.shift is not None:
        return shared_array(spec.shift)
    return _weather(spec.weather, spec.shift_table)[..., 1]


def _peaks(seed: int, field: DensityField, radius: float, octaves: int) -> np.ndarray:
    """The planet's column peaks (the threshold does not change them)."""
    return _cached_peaks(seed, replace(field, threshold=None), radius, octaves)


@functools.cache
def _cached_peaks(seed, field, radius, octaves) -> np.ndarray:
    return sphere_column_peaks(_noise(seed), field, radius, octaves)


@functools.cache
def _deck_light(
    seed: int,
    field: DensityField,
    optics: CloudOptics,
    radius: float,
    table: tuple,
    octaves: int,
) -> np.ndarray:
    """The deck's far lighting (:func:`noise.deck_light_table`; ~0.4 s)."""
    volume = _noise(seed)
    return deck_light_table(
        volume, field, optics, radius, np.asarray(table), octaves, float(volume.mean())
    )


def octave_offset(full: np.ndarray, fewer: np.ndarray) -> float:
    """How far to move the threshold of a field evaluated with fewer octaves
    so its coverage matches the full field's.

    The dropped octaves count as their mean, which keeps the fbm's mean but
    not its spread: the columns' peaks sit lower, and the coverage drops a
    few percent. Matching the peaks' quantiles over the useful range of
    coverage gives it back (within 2 %, tested).

    :param full: the full field's sorted column peaks
        (:func:`noise.sphere_column_peaks`)
    :param fewer: the same columns' with fewer octaves
    :returns: the offset to add to both ends of the threshold
        (:attr:`ShellSpec.threshold_offset`)
    """
    q = np.linspace(0.05, 0.95, 19)
    return float(np.mean(np.quantile(fewer, q) - np.quantile(full, q)))


@functools.cache
def _rects() -> np.ndarray:
    return np.asarray(atlas_rects(), dtype=np.float32)


def _patch_area(face: int, a0: float, b0: float, span: float, radius: float) -> float:
    """Area of a face patch on a sphere: two spherical triangles' solid angles."""
    a = np.array([a0, a0 + span, a0 + span, a0])
    b = np.array([b0, b0, b0 + span, b0 + span])
    c = face_dirs(face, a, b)

    def solid_angle(p, q, r):
        numerator = abs(np.dot(p, np.cross(q, r)))
        denominator = 1.0 + p @ q + q @ r + r @ p
        return 2.0 * math.atan2(numerator, denominator)

    return (solid_angle(c[0], c[1], c[2]) + solid_angle(c[0], c[2], c[3])) * radius**2


def generate_cloud_tile(
    spec: SphereCloudSpec, key: Key, version: int = 0, batch: int = 20000
) -> CloudTile:
    """Place one tile's billboards where the density field is non-zero.

    The worker-process entry point. Rejection sampling, as the flat field's
    :func:`cloud.sample_field_particles`, with the shell's octaves and
    threshold: the accept rate measures the cloud volume, which sets the
    count for the shell's volume fraction. A sparse tile gives up after 40
    batches with what it found.

    :param spec: the clouds
    :param key: the tile, (level, face, i, j); its level picks the shell
    :param version: carried through to the result
    :param batch: candidate points per rejection-sampling batch, at most
    :returns: the tile's billboards, deterministic for *spec*'s seed and *key*
    """
    start = time.perf_counter()
    level, face, i, j = key
    shell = spec.shells[spec.shell_of(level)]
    grid = grid_for(spec.radius)
    volume = _noise(spec.noise_seed)
    weather = _shift_plane(spec)
    field = shell.field_of(spec.field)
    base, thickness = field.slab
    span = 2.0 * (QUADS << level) / grid.n0
    a0, b0 = -1.0 + i * span, -1.0 + j * span
    mid = spec.radius + base + 0.5 * thickness
    slab_volume = _patch_area(face, a0, b0, span, mid) * thickness

    r_lo, r_hi = shell.radius
    mean_r3 = (r_hi**4 - r_lo**4) / (4.0 * (r_hi - r_lo)) if r_hi > r_lo else r_lo**3
    sphere = 4.0 / 3.0 * math.pi * mean_r3
    rng = np.random.default_rng(np.random.SeedSequence([spec.noise_seed, *key]))
    # The target is proportional to the accept rate, so reaching it takes
    # about this many candidates whatever the rate: batches of that (a coarse
    # shell keeps ~700 billboards, it used to try 20,000 points for them).
    step = int(slab_volume * shell.volume_fraction / sphere * 1.1)
    step = min(batch, max(MIN_TILE_BATCH, step))

    kept, tried, accepted, target = [], 0, 0, None
    while target is None or accepted < target:
        a = a0 + span * rng.uniform(size=step)
        b = b0 + span * rng.uniform(size=step)
        height = base + thickness * rng.uniform(size=step)
        points = face_dirs(face, a, b) * (spec.radius + height)[:, None]
        inside = points[
            density_on_sphere(
                volume, weather, field, points, spec.radius, shell.octaves
            )
            > 0.0
        ]
        tried += step
        accepted += len(inside)
        kept.append(inside)
        if target is None:
            rate = accepted / tried
            target = int(slab_volume * rate * shell.volume_fraction / sphere)
        if tried >= 40 * batch:  # a sparse fringe tile: what was found will do
            break
    points = np.concatenate(kept)[:target] if kept else np.zeros((0, 3))

    centre = grid.centre_dir(key) * mid
    n = len(points)
    particles = np.empty((n, 9), dtype=np.float32)
    particles[:, 0:3] = points - centre
    particles[:, 3] = rng.uniform(r_lo, r_hi, n)
    rects = _rects()
    particles[:, 4:8] = rects[rng.integers(len(rects), size=n)]
    particles[:, 8] = spec.shell_of(level)
    return CloudTile(
        key=key,
        version=version,
        centre=centre,
        particles=particles,
        gen_ms=(time.perf_counter() - start) * 1e3,
    )


# ── Drawing ────────────────────────────────────────────────────────────────────


def _shell_bin(shell: int) -> str:
    return f"{CLOUD_BIN}_{shell}"


def _ensure_bin():
    manager = CullBinManager.get_global_ptr()
    if manager.find_bin(CLOUD_BIN) == -1:
        manager.add_bin(CLOUD_BIN, CullBinManager.BT_back_to_front, _CLOUD_BIN_SORT)
    for shell in range(MAX_SHELLS):
        if manager.find_bin(_shell_bin(shell)) == -1:
            manager.add_bin(
                _shell_bin(shell),
                CullBinManager.BT_back_to_front,
                _SHELL_BIN_SORTS[shell],
            )


@functools.cache
def _vertex_format() -> GeomVertexFormat:
    fmt = GeomVertexArrayFormat()
    fmt.add_column(
        InternalName.get_vertex(), 3, GeomEnums.NT_float32, GeomEnums.C_point
    )
    fmt.add_column(
        InternalName.get_texcoord(), 2, GeomEnums.NT_float32, GeomEnums.C_texcoord
    )
    fmt.add_column(
        InternalName.make("i_radius"), 1, GeomEnums.NT_float32, GeomEnums.C_other
    )
    fmt.add_column(
        InternalName.make("i_uv_st"), 4, GeomEnums.NT_float32, GeomEnums.C_texcoord
    )
    fmt.add_column(
        InternalName.make("i_layer"), 1, GeomEnums.NT_float32, GeomEnums.C_other
    )
    combined = GeomVertexFormat()
    combined.add_array(fmt)
    return GeomVertexFormat.register_format(combined)


@functools.cache
def _shaders() -> tuple[Shader, Shader]:
    tiles = Shader.load(
        Shader.SL_GLSL,
        vertex=_SHADERS / "cloud_sphere.vert",
        fragment=_SHADERS / "cloud.frag",
    )
    deck = Shader.load(
        Shader.SL_GLSL,
        vertex=_SHADERS / "cloud_deck.vert",
        fragment=_SHADERS / "cloud.frag",
    )
    return tiles, deck


def _tile_node(tile: CloudTile) -> tuple[NodePath, GeomTriangles, float]:
    """A GeomNode for one tile (4 vertices per particle), its index buffer,
    and the tile's reach (the farthest a billboard reaches from its centre).

    The vertices are written straight into Panda's buffer: by the native
    library when it is built (0.6-3.6 ms a tile in numpy, a fraction of
    that natively).
    """
    p = tile.particles
    n = len(p)
    vdata = GeomVertexData("cloud_tile", _vertex_format(), GeomEnums.UH_static)
    vdata.unclean_set_num_rows(4 * n)
    buffer = memoryview(vdata.modify_array(0)).cast("B")
    native = native_library()
    if native is not None:
        box, reach = native.billboard_vertices(p, np.frombuffer(buffer, np.float32))
        low, high = box[0:3], box[3:6]
    else:
        verts = np.frombuffer(buffer, np.float32).reshape(n, 4, 11)
        verts[:, :, 0:3] = p[:, None, 0:3]
        verts[:, :, 3:5] = _CORNERS[None, :, :]
        verts[:, :, 5] = p[:, None, 3]
        verts[:, :, 6:10] = p[:, None, 4:8]
        verts[:, :, 10] = p[:, None, 8]
        radius = p[:, 3].max()
        low = p[:, 0:3].min(axis=0) - radius
        high = p[:, 0:3].max(axis=0) + radius
        reach = float(np.sqrt((p[:, 0:3] ** 2).sum(axis=1).max()) + radius)
    del buffer
    tris = GeomTriangles(GeomEnums.UH_dynamic)
    tris.set_index_type(GeomEnums.NT_uint32)
    indices = tris.modify_vertices()
    indices.unclean_set_num_rows(6 * n)
    order = (np.arange(n, dtype=np.uint32)[:, None] * 4 + _QUAD_TRIS).ravel()
    memoryview(indices).cast("B")[:] = order.tobytes()
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode(f"clouds{tile.key}")
    node.add_geom(geom)
    node.set_bounds(BoundingBox(Point3(*low), Point3(*high)))
    node.set_final(True)
    return NodePath(node), tris, reach


def deck_mesh(
    radius: float, quads: int = DECK_QUADS, name: str = "cloud_deck"
) -> GeomNode:
    """A cube-sphere of radius *radius*, never culled (the caller makes it
    two-sided: the camera may be above or below it). The deck's, and the
    cirrus sheet's (:mod:`cirrus`)."""
    t = np.linspace(-1.0, 1.0, quads + 1)
    a, b = np.meshgrid(t, t)
    positions, indices = [], []
    for face in range(6):
        positions.append(face_dirs(face, a, b) * radius)
        first = face * (quads + 1) ** 2
        r, c = np.meshgrid(np.arange(quads), np.arange(quads), indexing="ij")
        v00 = first + (r * (quads + 1) + c).ravel()
        v01, v10 = v00 + 1, v00 + quads + 1
        indices.append(np.stack([v00, v01, v10 + 1, v00, v10 + 1, v10], 1).ravel())
    verts = np.concatenate(positions).astype(np.float32)
    index = np.concatenate(indices).astype(np.uint32)
    vdata = GeomVertexData("cloud_deck", GeomVertexFormat.get_v3(), GeomEnums.UH_static)
    vdata.unclean_set_num_rows(len(verts))
    memoryview(vdata.modify_array(0)).cast("B")[:] = verts.tobytes()
    tris = GeomTriangles(GeomEnums.UH_static)
    tris.set_index_type(GeomEnums.NT_uint32)
    tris.add_next_vertices(len(index))
    memoryview(tris.modify_vertices()).cast("B")[:] = index.tobytes()
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode(name)
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    return node


@dataclass
class _Tile:
    node: NodePath
    tris: GeomTriangles
    data: CloudTile
    #: farthest particle reach from the centre, metres
    reach: float = 0.0
    #: camera position (planet frame) of the last re-sort; None = never
    sorted_from: tuple[float, float, float] | None = None
    #: the tile's centre (planet frame), as floats for the per-frame scoring
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: its billboards farthest first at the last re-sort (uint32); the next
    #: one starts from it, so equal distances keep their order
    order: np.ndarray | None = None


def _sort_billboards(tile: _Tile, offset: np.ndarray) -> None:
    """Write a tile's triangles farthest billboard first, from the camera
    at *offset* (the tile's centre minus the camera): natively when the
    library is built (a radix sort, ~10x faster), else in numpy. Both sort
    stably over the previous order."""
    p = tile.data.particles
    if tile.order is None:
        tile.order = np.arange(len(p), dtype=np.uint32)  # as _tile_node built it
    native = native_library()
    if native is not None:
        index = np.empty(6 * len(p), dtype=np.uint32)
        native.billboard_order(p, offset, tile.order, index)
    else:
        xyz = p[tile.order, 0:3].astype(np.float64) + offset
        distance = np.einsum("ij,ij->i", xyz, xyz).astype(np.float32)
        tile.order = tile.order[np.argsort(-distance, kind="stable")]
        index = (tile.order[:, None] * 4 + _QUAD_TRIS).ravel()
    memoryview(tile.tris.modify_vertices()).cast("B")[:] = index.tobytes()


class CloudVolume:
    """A planet's low clouds as a volume (any of the cloud types): billboard
    shells near the camera, the deck beyond. The volume layer of
    :class:`planet.PlanetClouds`, which a game builds instead; a
    :class:`cirrus.CloudSheet` may lie above it.

    Usage::

        volume = CloudVolume(render, planet_radius, executor,
                             atmospheres=atmospheres, atmosphere=0)
        volume.set_sun(sun_direction)
        # every frame, camera in the world frame (float64):
        volume.update(camera_world)

    :ivar root: the node everything hangs under (hide it to hide the clouds)
    :ivar deck: the far deck's node
    :ivar tiles_root: the billboard tiles' rebased root
    :ivar field: the density field, threshold resolved for the planet
    :ivar weather: (6, n, n, 2) cube map of coverage and fbm shift
    :ivar spec: what the workers are given (:class:`SphereCloudSpec`)
    :ivar stats: tiles shown, billboards in them, tiles pending, and a
        running mean of a tile's generation time (``gen_ms``), per update
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = CloudSettings

    def __init__(
        self,
        parent: NodePath,
        radius: float,
        executor: ChunkExecutor,
        position: ArrayLike = (0.0, 0.0, 0.0),
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        cloud_type: CloudType = CloudType.CUMULUS,
        altitude: float | None = None,
        planet: PlanetSpec | None = None,
        weather: WeatherSpec | None = None,
        seed: int = 7,
        weather_seed: int = 11,
        settings: CloudSettings | None = None,
        sun_brightness: float = 2.8,
        sky_strength: float = 0.35,
        sky_occlusion: float = 0.55,
        exposure: float = 0.9,
        near_fade_radii: float = 2.5,
        channel: str | None = None,
        game: Any = None,
    ):
        """
        :param parent: usually ``render``
        :param radius: the planet's sea-level radius, metres
        :param executor: the world's :class:`terrain.workers.ChunkExecutor`
            (tiles get their own channel)
        :param position: the planet centre in the world frame (float64)
        :param atmospheres: the world's atmospheres (the clouds are seen
            through them)
        :param atmosphere: the planet's own atmosphere index (reddens the
            clouds' sunlight); None = none
        :param cloud_type: the cloud type (its preset's field and optics)
        :param altitude: base of the cloud slab above sea level, metres; None
            keeps the type's
        :param planet: the planet the weather follows (climate, terrain);
            None = an earthlike one of this radius
        :param weather: the weather in full; None = the defaults for
            *planet* with *weather_seed*
        :param seed: noise seed (field and placement)
        :param weather_seed: weather systems' seed
        :param settings: what the clouds cost (:class:`cloud.CloudSettings`:
            the billboards, their shells and octaves, the sun march, the
            tile budgets; :meth:`set_settings` changes it); None = HIGH
        :param sun_brightness: direct-term gain (as :class:`field.CloudField`'s)
        :param sky_strength: ambient sky weight (live: :meth:`set_sky_light`)
        :param sky_occlusion: fraction of sky light reaching the slab's
            underside
        :param exposure: cloud radiance scale before the sky's tonemap
            (display.glsl's displayColour); the sky's own by default
        :param near_fade_radii: fade a billboard within this many of its radii
            of the camera (0 = off)
        :param channel: the executor channel for the tiles; None = one of its
            own. Reusing a removed instance's channel cancels its work
        :param game: the game object, for its asset manager (None: from disk)
        """
        settings = settings or CloudSettings()
        _ensure_bin()
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=np.float64)
        self.executor = executor
        self._channel = channel or f"clouds-{id(self):x}"
        self.grid: CubeGrid = grid_for(self.radius)
        self.stats = {"tiles": 0, "billboards": 0, "pending": 0, "gen_ms": 0.0}

        preset = PRESETS[cloud_type]
        field = preset.field
        if altitude is not None:
            field = replace(field, slab=(float(altitude), field.slab[1]))
        volume = _noise(seed)
        sigma = fbm_sigma(volume, field)
        peaks = _peaks(seed, field, self.radius, 4)
        field = with_threshold(field, peaks, sigma)
        self.field = field
        if weather is None:
            planet = planet or PlanetSpec(name="clouds", radius=self.radius)
            weather = WeatherSpec(planet, seed=weather_seed)
        table = tuple(float(v) for v in shift_table(peaks, field.threshold[0]))
        # The weather map and the deck's light (~1 s together) are made in the
        # pool while the caller goes on; update() finishes the set-up when
        # they are in (see _finish).
        self._weather_job = executor.submit_once(_weather, weather, table)
        #: (6, n, n, 2): coverage and fbm shift; None until the set-up is done
        self.weather: np.ndarray | None = None
        self._shared_shift: SharedArray | None = None
        self._preset = preset
        self._seed = seed
        self._peaks = peaks
        self._weather_spec = weather
        self._table = table
        self._octave_mean = float(volume.mean())
        base_level = max(0, round(math.log2(BASE_TILE_SIZE / self.grid.chunk_size(0))))
        self._base_level = base_level
        self._settings = settings
        params, self._deck_job = self._configure(settings)

        # ── Nodes and shared inputs ──
        tile_shader, deck_shader = _shaders()
        self.root = parent.attach_new_node("spherical_clouds")
        self.root.stash()  # out of every camera until the set-up is done
        self.tiles_root = self.root.attach_new_node("cloud_tiles")
        self.tiles_root.set_shader(tile_shader)
        self.deck = self.root.attach_new_node(
            deck_mesh(self.radius + field.slab[0] + 0.5 * field.slab[1])
        )
        self.deck.set_shader(deck_shader)
        self.deck.set_two_sided(True)
        self.deck.set_shader_inputs(uDeck=1, uDeckLayer=self._deck_layer)
        self.tiles_root.set_shader_inputs(uDeck=0, uDeckLayer=0)
        atlas, _ = load_cloud_atlas(game)
        self.tiles_root.set_texture(atlas)
        self.deck.set_texture(atlas)

        root = self.root
        root.set_transparency(TransparencyAttrib.M_none)
        root.set_depth_test(True)
        root.set_depth_write(False)
        root.set_bin(CLOUD_BIN, 0)
        root.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_one,
                ColorBlendAttrib.O_one_minus_incoming_alpha,
            )
        )
        layer_params = PTA_LVecBase4f()
        for row in params.reshape(-1, 4):
            layer_params.push_back(LVecBase4f(*(float(v) for v in row)))
        self._layer_params = layer_params
        bind_defaults(root)
        if atmospheres is not None:
            atmospheres.bind(root)
        self._camera_planet = PTA_LVecBase3f.empty_array(1)
        self._camera_render = PTA_LVecBase3f.empty_array(1)
        self._planet_render = PTA_LVecBase3f.empty_array(1)
        self._sun = PTA_LVecBase3f.empty_array(1)
        self._sun[0] = LVecBase3f(0.0, 0.0, 1.0)
        root.set_shader_inputs(
            layerParams=layer_params,
            cloudNoise=build_noise_texture(volume),
            noiseSize=float(NOISE_SIZE),
            noiseOffset=Vec3(0, 0, 0),
            deckLightCosMin=DECK_LIGHT_COS_MIN,
            uSpherical=1,
            planetRadius=self.radius,
            uOwnAtmosphere=-1 if atmosphere is None else int(atmosphere),
            camPos=self._camera_planet,
            uCamRender=self._camera_render,
            uPlanetCentre=self._planet_render,
            sunDir=self._sun,
            # White: the planet's air colours it (atmoSunlight in cloud.frag).
            sunColor=Vec3(1.0, 1.0, 1.0),
            skyColor=Vec3(0.45, 0.6, 0.85),
            hazeColor=Vec3(0.45, 0.6, 0.85),
            sunBrightness=float(sun_brightness),
            skyStrength=float(sky_strength),
            skyOcclusion=float(sky_occlusion),
            exposure=float(exposure),
            horizonFade=0.0,
            uPixelAngle=0.001,
            earthCurvature=0.0,
            nearFadeRadii=float(near_fade_radii),
            uCloudDebug=0,
        )

        self._tiles: dict[Key, _Tile] = {}
        self._wanted: list[Key] = []
        self._shown: set[Key] = set()
        self._ready: dict[Key, CloudTile] = {}
        self._pixel_angle = 0.001
        self._sky_light: float | None = None
        self._origin = None
        self._camera = np.zeros(3)
        self._last_select = None
        self._frame = 0
        self._selector = LodSelector(self.grid)
        self._slab_box = (field.slab[0], field.slab[0] + field.slab[1])
        self._apply_live(settings)

    # ── settings ──────────────────────────────────────────────────────────────

    @property
    def settings(self) -> CloudSettings:
        """What the clouds cost (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: CloudSettings) -> None:
        """Change what the clouds cost. The sun march and the budgets apply
        live; the billboard count, overlap, octaves and shells re-plan the
        shells and regenerate every tile (on the same worker channel), so the
        near clouds refill over a few seconds; the deck stays.

        :param settings: the new settings (``resort_frames`` is the flat
            field's and does nothing here)
        """
        changes = self._settings.changes(settings)
        if not changes:
            return
        self._finish()
        self._settings = settings
        params, deck_job = self._configure(settings)
        deck_light = deck_job.result()
        if REBUILD in changes.values():
            self._version = self.executor.set_spec(
                self.spec, self._channel, generator=generate_cloud_tile
            )
            for tile in self._tiles.values():
                tile.node.remove_node()
            self._tiles.clear()
            self._ready.clear()
            self._shown.clear()
            self._wanted = []
            self._last_select = None
            self.root.set_shader_input(
                "deckLight", build_deck_light_texture(deck_light)
            )
            self.deck.set_shader_input("uDeckLayer", self._deck_layer)
        for i, row in enumerate(params.reshape(-1, 4)):
            self._layer_params[i] = LVecBase4f(*(float(v) for v in row))
        self._apply_live(settings)

    def _apply_live(self, settings: CloudSettings) -> None:
        if settings.reflected:
            self.root.show(REFLECTION_CAMERA_MASK)
        else:
            self.root.hide(REFLECTION_CAMERA_MASK)

    def _configure(self, settings: CloudSettings) -> tuple[np.ndarray, np.ndarray]:
        """Plan the shells and the deck for some settings: sets the bands,
        the deck's layer and :attr:`spec` (what the workers get).

        :returns: the packed layer parameters, and the deck's light table
            being made in the pool (a future)
        """
        preset, field, peaks, seed = self._preset, self.field, self._peaks, self._seed
        shells = settings.shells
        fractions = (settings.overlap_near, settings.overlap_far)
        schedule = settings.shell_octaves
        octave_mean = self._octave_mean
        base_level = self._base_level
        tile0 = self.grid.chunk_size(base_level)
        table = self._table

        # The shells and their crossfade bands (lod_shells' layout, for a
        # tile four times smaller than the box), then the deck.
        domain = 4.0 * tile0
        bands = [
            (0.62 * domain * 2**k * 0.5, 0.85 * domain * 2**k * 0.5)
            for k in range(shells)
        ]
        sprite = atlas_mean_alpha()
        shell_specs = []
        params = np.zeros((MAX_LAYERS, LAYER_VEC4S, 4), dtype=np.float32)
        for k in range(shells):
            scale = 2.0**k
            spec = with_settings(
                replace(
                    preset,
                    radius=(preset.radius[0] * scale, preset.radius[1] * scale),
                    optics=replace(
                        preset.optics,
                        max_optical_depth=preset.optics.max_optical_depth * scale,
                    ),
                ),
                settings,
            )
            fraction = fractions[min(k, len(fractions) - 1)]
            octaves = schedule[min(k, len(schedule) - 1)]
            shell = ShellSpec(
                base_level + k,
                spec.radius,
                fraction,
                octaves,
                octave_offset(peaks, _peaks(seed, field, self.radius, octaves)),
            )
            shell_specs.append(shell)
            params[k] = pack_layer_params(
                shell.field_of(field),
                extinction=field.density / max(fraction * sprite, 1e-6),
                wrap_radius=1e12,
                wrap_fade_band=0.0,
                octave_mean=octave_mean,
                optics=spec.optics,
                aspect=spec.aspect,
                fade_in=bands[k - 1] if k > 0 else None,
                fade_out=bands[k],
                octaves=octaves,
            )
        self._deck_layer = shells
        deck = ShellSpec(
            0,
            (0.0, 0.0),
            1.0,
            schedule[-1],
            octave_offset(peaks, _peaks(seed, field, self.radius, schedule[-1])),
        )
        deck_optics = replace(preset.optics, max_optical_depth=DECK_MAX_OPTICAL_DEPTH)
        params[shells] = pack_layer_params(
            deck.field_of(field),
            extinction=field.density,  # per metre of the slab's path
            wrap_radius=1e12,
            wrap_fade_band=0.0,
            octave_mean=octave_mean,
            optics=deck_optics,
            fade_in=bands[-1],
            octaves=schedule[-1],
        )
        deck_job = self.executor.submit_once(
            _deck_light,
            seed,
            deck.field_of(field),
            deck_optics,
            self.radius,
            table,
            schedule[-1],
        )
        self._bands = bands
        shared = self._shared_shift
        self.spec = SphereCloudSpec(
            radius=self.radius,
            field=field,
            noise_seed=seed,
            weather=self._weather_spec,
            shift_table=table,
            shells=tuple(shell_specs),
            shift=None if shared is None else shared.handle,
        )
        return params, deck_job

    # ── public ────────────────────────────────────────────────────────────────

    def set_sun(
        self,
        direction: ArrayLike,
        sun_color: ArrayLike | None = None,
        sky_color: ArrayLike | None = None,
    ) -> None:
        """Move the sun and/or restate its colours (uniform writes only).

        :param direction: toward the sun, world frame (need not be unit)
        :param sun_color: RGB of direct sunlight, or None to keep; white by
            default, since the planet's own air colours it
        :param sky_color: RGB of the ambient sky light, or None to keep
        """
        d = np.asarray(direction, dtype=float)
        self._sun[0] = LVecBase3f(*(d / np.linalg.norm(d)))
        if sun_color is not None:
            self.root.set_shader_input("sunColor", Vec3(*sun_color))
        if sky_color is not None:
            self.root.set_shader_input("skyColor", Vec3(*sky_color))

    def set_sky_light(self, strength: float) -> None:
        """The sky's (ambient) light on the clouds, as ``sky_strength``.

        :param strength: the new ambient sky weight
        """
        if strength != self._sky_light:  # a demo restates it every frame
            self._sky_light = strength
            self.root.set_shader_input("skyStrength", float(strength))

    def set_pixel_angle(self, radians: float) -> None:
        """The angle one pixel spans (vertical fov / window height): how far
        the deck can show the clouds' own detail and self-shadowing before
        averaging them (0.001 until set).

        :param radians: the angle per pixel
        """
        # Set every frame by World: a new value only when it changes, or the
        # root's render state (over every tile) is rebuilt each frame.
        if radians != self._pixel_angle:
            self._pixel_angle = radians
            self.root.set_shader_input("uPixelAngle", float(radians))

    def update(
        self, camera_world: ArrayLike, camera_render: ArrayLike = (0.0, 0.0, 0.0)
    ) -> None:
        """Per frame: select, generate, upload, sort, place.

        The tiles are reselected once the camera has moved the larger of
        200 m and 0.5 % of its altitude, and every 30 frames.

        :param camera_world: camera in the world frame, metres (float64)
        :param camera_render: camera in the render frame (the origin when
            the scene is drawn camera-relative)
        """
        if self._weather_job is not None:
            if not (self._weather_job.done() and self._deck_job.done()):
                self.stats["pending"] = 1  # the set-up: World.waiting() counts it
                return
            self._finish()
        self._frame += 1
        cam = np.asarray(camera_world, dtype=np.float64) - self.position
        render_cam = np.asarray(camera_render, dtype=np.float64)
        self._camera = cam
        self._rebase(cam, render_cam)
        self._camera_planet[0] = LVecBase3f(*cam)
        self._camera_render[0] = LVecBase3f(*render_cam)
        self._planet_render[0] = LVecBase3f(*(render_cam - cam))
        self.deck.set_pos(*(render_cam - cam))

        for tile in self.executor.collect(self._channel):
            self._ready[tile.key] = tile
            self.stats["gen_ms"] = 0.9 * self.stats["gen_ms"] + 0.1 * tile.gen_ms
        altitude = float(np.linalg.norm(cam)) - self.radius
        moved = self._last_select is None or np.linalg.norm(
            cam - self._last_select
        ) > max(200.0, 0.005 * abs(altitude))
        if moved or self._frame % 30 == 0:
            self._select(cam)
        self._upload()
        self._resort(cam)
        self.stats.update(
            tiles=len(self._shown),
            pending=self.executor.pending_for(self._channel),
            billboards=sum(len(self._tiles[k].data.particles) for k in self._shown),
        )

    def set_debug(self, flags: int) -> None:
        """Profiling switches (cloud_field.glsl's ``uCloudDebug`` bits): 1 no
        sun march, 2 no atmosphere, 4 two octaves only, 8 flat colour.

        :param flags: the bits, 0 = normal
        """
        self.root.set_shader_input("uCloudDebug", int(flags))

    def remove(self) -> None:
        """Take the clouds off the scene and drop their queued tiles."""
        self.executor.request([], self._channel)
        self.root.remove_node()
        if self._shared_shift is not None:
            self._shared_shift.release()
            self._shared_shift = None

    def _finish(self) -> None:
        """Finish the set-up begun in the pool (waiting for it if need be):
        the weather map and the deck's light become textures, the weather's
        shift is shared with the workers, and the tiles can be asked for."""
        if self._weather_job is None:
            return
        self.weather = self._weather_job.result()
        deck_light = self._deck_job.result()
        self._weather_job = self._deck_job = None
        if self.executor.backend == "process":
            self._shared_shift = SharedArray(self.weather[..., 1])
            self.spec = replace(self.spec, shift=self._shared_shift.handle)
        self._version = self.executor.set_spec(
            self.spec, self._channel, generator=generate_cloud_tile
        )
        self.root.set_shader_inputs(
            weatherMap=build_weather_texture(self.weather),
            deckLight=build_deck_light_texture(deck_light),
        )
        self.root.unstash()

    # ── internals ─────────────────────────────────────────────────────────────

    def _rebase(self, cam, render_cam) -> None:
        """Tiles hang under a root at ``origin - camera`` and sit at ``centre -
        origin``: float32 only ever sees numbers near the camera (as the
        terrain does)."""
        altitude = max(float(np.linalg.norm(cam)) - self.radius, 0.0)
        snap = 4096.0
        while snap * 16.0 < altitude:
            snap *= 2.0
        origin = np.round(cam / snap) * snap
        if self._origin is None or np.any(origin != self._origin):
            self._origin = origin
            for tile in self._tiles.values():
                tile.node.set_pos(*(tile.data.centre - origin))
        self.tiles_root.set_pos(*(render_cam - (cam - origin)))

    def _select(self, cam) -> None:
        """The tiles each shell needs: within its fade-out distance, not
        wholly inside its fade-in, above the horizon."""
        self._last_select = cam.copy()
        selector = self._selector
        selector._begin(cam)
        box = self._slab_box
        wanted = []
        levels = [s.level for s in self.spec.shells]
        stack = list(self.grid.roots())
        while stack:
            key = stack.pop()
            distance, culled = selector._measure(key, box)
            if culled:
                continue
            level = key[0]
            if level > levels[-1]:
                if distance < self._bands[-1][1]:
                    stack.extend(CubeGrid.children(key))
                continue
            k = level - levels[0]
            if distance >= self._bands[k][1]:
                continue
            if k > 0:
                # Wholly inside the shell's fade-in: weight zero there.
                reach = distance + self.grid.chunk_size(level) * 1.5
                if reach > self._bands[k - 1][0]:
                    wanted.append(key)
                if level > levels[0] and distance < self._bands[k - 1][1]:
                    stack.extend(CubeGrid.children(key))
            else:
                wanted.append(key)
        wanted.sort(key=lambda k: -k[0])  # coarse first
        self._wanted = wanted
        wanted_set = set(wanted)
        for key in self._shown - wanted_set:
            self._tiles[key].node.detach_node()
        self._shown &= wanted_set
        for key in wanted:
            if key in self._tiles and key not in self._shown:
                self._tiles[key].node.reparent_to(self.tiles_root)
                self._shown.add(key)
        self.executor.request(
            [k for k in wanted if k not in self._tiles and k not in self._ready],
            self._channel,
        )
        # Forget tiles far out of use.
        cache = self._settings.tile_cache
        if len(self._tiles) > len(wanted) + cache:
            for key in [k for k in self._tiles if k not in wanted_set][
                : len(self._tiles) - len(wanted) - cache
            ]:
                self._tiles.pop(key).node.remove_node()

    def _upload(self) -> None:
        count = 0
        for key in self._wanted:
            if count >= self._settings.uploads_per_frame:
                break
            tile = self._ready.pop(key, None)
            if tile is None or key in self._tiles:
                continue
            if tile.version != self._version:
                continue
            reach = 0.0
            if len(tile.particles) == 0:
                node, tris = NodePath("empty"), None
            else:
                node, tris, reach = _tile_node(tile)
                node.set_bin(_shell_bin(self.spec.shell_of(key[0])), 0)
            node.set_pos(*(tile.centre - self._origin))
            centre = tuple(float(c) for c in tile.centre)
            self._tiles[key] = _Tile(node, tris, tile, reach, centre=centre)
            if tris is not None:
                node.reparent_to(self.tiles_root)
                self._shown.add(key)
            count += 1
        # Ready tiles nobody wants any more.
        wanted = set(self._wanted)
        for key in [k for k in self._ready if k not in wanted]:
            del self._ready[key]

    def _resort(self, cam) -> None:
        """Back to front within the stalest few shown tiles (premultiplied over
        needs it; the bins sort the tiles themselves).

        A tile's order goes stale as the camera moves past it: by the
        distance moved since its last sort over its distance now (an angle).
        A stale order composites a cloud's far side over its near side, which
        reads as shadows sliding across it. New tiles come first.
        """
        stale = []
        here = tuple(float(c) for c in cam)  # math.dist: no numpy per tile
        for key in self._shown:
            tile = self._tiles[key]
            if tile.tris is None:
                continue
            if tile.sorted_from is None:
                stale.append((math.inf, key))
                continue
            moved = math.dist(here, tile.sorted_from)
            if moved == 0.0:
                continue
            gap = math.dist(tile.centre, here) - tile.reach
            stale.append((moved / max(gap, 0.05 * tile.reach, 1.0), key))
        stale.sort(reverse=True)
        for score, key in stale[: self._settings.resorts_per_frame]:
            if score < 1e-3:  # a tenth of a degree: nothing visible changes
                break
            tile = self._tiles[key]
            tile.sorted_from = here
            _sort_billboards(tile, tile.data.centre - cam)
