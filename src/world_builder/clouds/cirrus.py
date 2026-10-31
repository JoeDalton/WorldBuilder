"""
High cirrus over a planet: baked patches of fibrous cloud, stamped along the
wind on one sheet.

Cirrus is thin ice cloud, a few hundred metres to a kilometre deep and tens of
kilometres across, optically thin (it hides little of the sky behind it). The
billboards that draw the cumulus (:mod:`sphere`) would give it no volume worth
their cost, and their puffs fight its fibres. So it lies on a **surface**, a
sphere at the slab's mid-height, and a sight line crosses ``thickness / |cos|``
of the slab where it meets it.

**Its shapes are baked** (scripts/bake_cirrus_atlas.py, the atlas
``cirrus_atlas.png`` and its ``.json``): two dozen patches of cirrus uncinus,
24 x 12 km, drawn offline by a procedural model at a cost no frame could
pay:

- cottony tufts along a curved leading front;
- their comet tails, dragged upwind along a curving flow, widening as they
  fall and merging;
- strands of three widths, of uneven strength, along every curve;
- a faint fibrous veil joining them, and an irregular outline.

The tufts lead and the tails lag upwind, as fall streaks do when the wind
rises with height.

**Stamped as geometry** (:func:`cirrus_stamps`): a Poisson field of quads on
the sheet, thinned by the cirrus cover map (:func:`weather.cirrus_cover_map`)
so that the share of the sheet under a stamp is the map's cover, each turned
along the upper wind (:func:`upper_wind_at`: the jet streams bent around the
cyclones, plus meanders), all wandering off it together so that overlapping
tails stay parallel. A pixel pays one atlas
fetch per stamp over it (typically 1-3). Stamping in the shader instead would
search a 3D lattice per pixel (a 2D grid on the cube's faces leaves seams):
about 30 candidates for stamps this long, no cheaper than computing the
cloud. The atlas is mipmapped, with anisotropic filtering, so the far sheet
and the horizon need nothing else.

**Draw order**, against the cumulus below and anything else translucent. The
cirrus lies above the cumulus slab, and a ray's height along it is convex (it
falls, then rises), so where a ray *enters* the cirrus sphere it is above every
cumulus (the cirrus is in front of all of them), and after it *exits* it never
comes down again (the cirrus is behind all of them). For any camera: so the
stamps are drawn twice, each node keeping one kind of crossing, found per
fragment from the pass's own camera (the sea's mirror included):

* ``behind`` (exits) in the sky's bin after the sky dome and the stars, before
  every cloud bin;
* ``front`` (entries) in Panda's "transparent" bin, after every cloud bin;
  sorted there by their groups' centres, so a game's translucent effects
  nearer still go on top, as they do over the cumulus.

Overlapping stamps lie on one sphere with near-identical colours: their order
within a draw does not show.

Lit as the cumulus are (the ice's phase function, the planet's air reddening
the sunlight and twilight at the cirrus's own height: it stays pink after the
cumulus below have gone grey), seen through the air in front of it, per
vertex. No sun march: the sheet barely shadows itself.

Design notes: docs/source/atmosphere.md, "Cirrus".
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BoundingBox,
    ColorBlendAttrib,
    Filename,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    InternalName,
    LVecBase3f,
    NodePath,
    PNMImage,
    Point3,
    PTA_LVecBase3f,
    SamplerState,
    Shader,
    StringStream,
    Texture,
    TexturePool,
    TransparencyAttrib,
    Vec3,
    Vec4,
)
from scipy.ndimage import map_coordinates

from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.clouds.cloud import PRESETS, CloudSettings, CloudType
from world_builder.clouds.noise import (
    CloudOptics,
    clamped_anisotropy,
    field_offset,
    phase_weights,
)
from world_builder.clouds.weather import (
    WeatherSpec,
    cirrus_cover_map,
    sample_cube,
    stream_function,
    stream_table,
    stream_vortices,
)
from world_builder.depth import REFLECTION_CAMERA_MASK
from world_builder.sky.sky import SKY_BIN, ensure_sky_bin
from world_builder.terrain.planet import PlanetSpec

if TYPE_CHECKING:
    from world_builder.terrain.workers import ChunkExecutor

_SHADERS = Path(__file__).parent / "shaders"
#: The behind node's draw order in the sky's bin: after the dome (0) and the
#: stars (1).
BEHIND_DRAW_ORDER = 2
#: Panda's own bin for translucent objects, sorted back to front.
FRONT_BIN = "transparent"
#: The baked stamps (scripts/bake_cirrus_atlas.py): patches of tufts at +x
#: and their tails toward -x, in a grid of (columns, rows) of stamps.
ATLAS_PNG = _SHADERS.parent / "cirrus_atlas.png"
ATLAS_JSON = _SHADERS.parent / "cirrus_atlas.json"
ATLAS_GRID = (4, 6)
#: A stamp's pixels (along, across) and the metres they span at scale 1.
STAMP_PIXELS = (1024, 512)
STAMP_METRES = (24000.0, 12000.0)
#: A stamp covers the sheet where its thickness is above this.
FOOTPRINT_THRESHOLD = 0.02
#: Quads per stamp, along and across: its vertices lie on the sheet's sphere
#: (a 34 km stamp's 8.4 km segments sag 15 m under it on a 600 km planet).
STAMP_QUADS = (4, 2)
#: Stamps are drawn in groups by cube face and a GROUP_CELLS x GROUP_CELLS
#: grid on it, which Panda culls to the view.
GROUP_CELLS = 4
#: The atlas's anisotropic filtering: the sheet is mostly seen grazing.
ANISOTROPY = 8


@dataclass(frozen=True)
class CirrusSpec:
    """What a planet's cirrus looks like. Lengths in metres; the stamps' are
    the cloud's own, not scaled with the planet.

    :ivar base: the slab's base above sea level (it must clear the cumulus)
    :ivar thickness: the slab's depth (a sight line crosses thickness / |cos|)
    :ivar amount: the weather's cirrus cover is multiplied by it
    :ivar meander: how far the stamps' heading wanders off the wind's
        contours, in stream-function metres (metres across the wind at the
        jets' speed)
    :ivar meander_wavelength: over what distance it wanders
    :ivar extinction: optical depth per metre where a stamp's thickness is 1
    :ivar max_optical_depth: what a sight line's optical depth through the
        sheet saturates toward (softly)
    :ivar optics: the phase function and the light's fill (the ``CIRRUS``
        preset's: ice, strongly forward)
    :ivar seed: the stamps' placement
    :ivar scale_range: a stamp's size against the atlas's (24 x 12 km)
    :ivar aspect_range: how much wider (across the wind) a stamp is drawn
    :ivar opacity_range: a stamp's opacity
    :ivar turn: degrees the stamps' heading wanders off the wind, together:
        a slow noise over ``turn_wavelength``, so stamps that overlap stay
        parallel (each turned on its own, their tails crossed)
    :ivar turn_wavelength: over what distance that heading wanders, metres
    :ivar jitter: degrees each stamp turns off the others at random
    :ivar max_turn: degrees the upper wind may turn along a stamp's length;
        stamps where it turns more fade out (near a low's centre, where the
        wind nears calm): a straight stamp there would cross its neighbours
    """

    base: float = 8000.0
    thickness: float = 1000.0
    amount: float = 1.0
    meander: float = 3000.0
    meander_wavelength: float = 120000.0
    extinction: float = 1.5e-3
    max_optical_depth: float = 2.5
    optics: CloudOptics = PRESETS[CloudType.CIRRUS].optics
    seed: int = 0
    scale_range: tuple = (0.6, 1.4)
    aspect_range: tuple = (0.8, 1.25)
    opacity_range: tuple = (0.6, 1.0)
    turn: float = 10.0
    turn_wavelength: float = 200000.0
    jitter: float = 2.0
    max_turn: float = 20.0

    @property
    def top(self) -> float:
        """The slab's top above sea level, metres."""
        return self.base + self.thickness

    @property
    def mid(self) -> float:
        """Where the sheet is drawn, above sea level, metres."""
        return self.base + 0.5 * self.thickness


# ── The noise (the stamps' baking and the wind's meanders) ─────────────────────

#: Noise-space offsets decorrelating the meanders and the heading's wander.
OFFSETS = {"meander": np.array(field_offset(11)), "turn": np.array(field_offset(12))}


#: The hash's constants.
_HASH_AXES = tuple(np.uint32(v) for v in (0x8DA6B343, 0xD8163841, 0xCB1AB31F))
_HASH_MIX = tuple(np.uint32(v) for v in (0x2C1B3C6D, 0x297A2D39))


def _cell_value(cells: np.ndarray) -> np.ndarray:
    """A lattice cell's value in [0, 1).

    :param cells: (..., 3) int64 lattice coordinates
    """
    q = cells.astype(np.uint32)  # wraps, two's complement
    h = (q[..., 0] * _HASH_AXES[0]) ^ (q[..., 1] * _HASH_AXES[1])
    h ^= q[..., 2] * _HASH_AXES[2]
    h ^= h >> np.uint32(15)
    h *= _HASH_MIX[0]
    h ^= h >> np.uint32(12)
    h *= _HASH_MIX[1]
    h ^= h >> np.uint32(15)
    return (h >> np.uint32(8)).astype(np.float64) / 16777216.0


def value_noise(coords: ArrayLike) -> np.ndarray:
    """Hashed value noise with quintic interpolation (C2): one feature per
    unit, in [0, 1). The stamps are baked from it (scripts/bake_cirrus_atlas.py),
    and it bends the wind they follow (:func:`upper_wind_at`).

    :param coords: (..., 3) noise coordinates
    """
    c = np.asarray(coords, dtype=np.float64)
    i = np.floor(c)
    f = c - i
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    cell = i.astype(np.int64)
    out = 0.0
    for dz in (0, 1):
        wz = u[..., 2] if dz else 1.0 - u[..., 2]
        for dy in (0, 1):
            wy = u[..., 1] if dy else 1.0 - u[..., 1]
            for dx in (0, 1):
                wx = u[..., 0] if dx else 1.0 - u[..., 0]
                corner = _cell_value(cell + np.array([dx, dy, dz]))
                out = out + wx * wy * wz * corner
    return out


# ── The stamps ─────────────────────────────────────────────────────────────────


@functools.cache
def atlas_records() -> tuple[dict, ...]:
    """The baked stamps' descriptors (scripts/bake_cirrus_atlas.py): rect
    (u_min, v_min, u_size, v_size), footprint (share of the stamp with
    thickness above :data:`FOOTPRINT_THRESHOLD`), mean thickness, and the
    patch it was baked from."""
    with open(ATLAS_JSON) as f:
        return tuple(json.load(f))


@functools.cache
def atlas_array() -> np.ndarray:
    """The atlas's thickness, (rows, columns) float32 in [0, 1], row 0 at
    v = 0 (the bottom). No GL: read from the PNG."""
    image = PNMImage()
    if not image.read(Filename.from_os_specific(str(ATLAS_PNG))):
        raise OSError(f"could not read the cirrus atlas at {ATLAS_PNG}")
    stream = StringStream()
    image.write(stream, "atlas.pgm")
    data = stream.get_data()
    # Binary PGM: "P5 <columns> <rows> <max>" then one byte a pixel
    header = data.split(maxsplit=4)
    columns, rows = int(header[1]), int(header[2])
    pixels = np.frombuffer(data[-columns * rows :], dtype=np.uint8)
    return (pixels.reshape(rows, columns)[::-1] / 255.0).astype(np.float32)


@dataclass(frozen=True)
class Stamps:
    """The cirrus's stamps over a planet (:func:`cirrus_stamps`).

    :ivar centres: (N, 3) on the sheet, planet frame, metres
    :ivar along: (N, 3) unit, downwind: the stamp's +x (its tuft's end)
    :ivar across: (N, 3) unit, the stamp's +y (mirrored or not)
    :ivar index: (N,) the stamp in the atlas
    :ivar scale: (N,) size against the atlas's metres
    :ivar aspect: (N,) extra width across
    :ivar opacity: (N,)
    :ivar sheet: the sheet's radius, metres
    """

    centres: np.ndarray
    along: np.ndarray
    across: np.ndarray
    index: np.ndarray
    scale: np.ndarray
    aspect: np.ndarray
    opacity: np.ndarray
    sheet: float

    def __len__(self) -> int:
        return len(self.index)


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _tangent_frame(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East and north unit vectors at unit *dirs* (any at the poles)."""
    east = np.cross([0.0, 0.0, 1.0], dirs)
    norm = np.linalg.norm(east, axis=1, keepdims=True)
    east = np.where(norm > 1e-9, east / np.maximum(norm, 1e-12), [1.0, 0.0, 0.0])
    return east, np.cross(dirs, east)


def upper_wind_at(
    spec: CirrusSpec, weather: WeatherSpec, dirs: np.ndarray, radius: float
) -> np.ndarray:
    """The upper wind with the cirrus's meanders, tangent at unit *dirs*,
    the jets' 1 the fastest: ``k x grad psi'`` (psi from
    :func:`weather.stream_table` and :func:`weather.stream_vortices`, plus
    the meander noise), by central differences over 1 km.

    :returns: (N, 3)
    """
    table = stream_table(weather)
    rows = stream_vortices(weather)
    step = 1000.0

    def psi(points):
        unit = points / np.linalg.norm(points, axis=1, keepdims=True)
        wander = value_noise(points / spec.meander_wavelength + OFFSETS["meander"])
        return stream_function(table, rows, unit) + spec.meander * 2.0 * (wander - 0.5)

    east, north = _tangent_frame(dirs)
    centre = dirs * radius
    d_east = (psi(centre + step * east) - psi(centre - step * east)) / (2 * step)
    d_north = (psi(centre + step * north) - psi(centre - step * north)) / (2 * step)
    # u = -dpsi/dy, v = dpsi/dx (weather.stream_table's sign)
    return -d_north[:, None] * east + d_east[:, None] * north


def _mean_area(spec: CirrusSpec) -> float:
    """A stamp's mean footprint, m^2: the atlas's mean footprint times the
    mean of scale^2 x aspect."""
    lo, hi = spec.scale_range
    scale2 = (hi**3 - lo**3) / (3.0 * (hi - lo)) if hi > lo else lo * lo
    aspect = 0.5 * sum(spec.aspect_range)
    footprint = np.mean([r["footprint"] for r in atlas_records()])
    return float(footprint * STAMP_METRES[0] * STAMP_METRES[1] * scale2 * aspect)


def _wind_turn(
    spec: CirrusSpec,
    weather: WeatherSpec,
    dirs: np.ndarray,
    wind: np.ndarray,
    sheet: float,
) -> np.ndarray:
    """Degrees the upper wind turns over a stamp's length (the largest
    scale's) centred on each of *dirs*, along *wind*."""
    half = 0.5 * STAMP_METRES[0] * spec.scale_range[1]
    along = wind / np.maximum(np.linalg.norm(wind, axis=1, keepdims=True), 1e-12)
    ends = []
    for sign in (-1.0, 1.0):
        end = dirs * sheet + sign * half * along
        end /= np.linalg.norm(end, axis=1, keepdims=True)
        w = upper_wind_at(spec, weather, end, sheet)
        ends.append(w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12))
    cos = np.einsum("ij,ij->i", ends[0], ends[1])
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


#: The most cover the stamps are placed for: past it they pile up for little
#: more cover (a Poisson field's cover is 1 - exp(-density x area)).
MAX_COVER = 0.95


@functools.cache
def cirrus_stamps(spec: CirrusSpec, weather: WeatherSpec, radius: float) -> Stamps:
    """Place the stamps over a planet (about a second, then cached per
    process; top-level and picklable: run in the pool).

    A Poisson field thinned by the cover map (:func:`weather.cirrus_cover_map`
    times ``amount``): at cover c its density is ``-ln(1 - c) / A`` for a
    stamp's mean footprint A, so the share of the sheet under at least one
    stamp is c, but where the upper wind turns more than ``max_turn`` along a
    stamp's length (near a low's centre, or where the wind nears calm): a
    straight stamp there would cross its neighbours. Each turns its tufts
    downwind (:func:`upper_wind_at`): the tufts lead and their tails lag
    upwind, as fall streaks do when the wind rises with height. Their heading
    wanders off the wind together (``turn``, a slow noise), so overlapping
    stamps stay parallel: real streaks side by side never cross at an angle.

    :param radius: the planet's sea-level radius, metres
    """
    rng = np.random.default_rng([spec.seed, 0xC1225])
    sheet = radius + spec.mid
    cover_map = np.clip(cirrus_cover_map(weather) * spec.amount, 0.0, MAX_COVER)
    area = _mean_area(spec)
    density_max = -math.log1p(-float(cover_map.max())) / area
    count = rng.poisson(density_max * 4.0 * math.pi * sheet * sheet)
    dirs = rng.normal(size=(count, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cover = np.clip(sample_cube(cover_map, dirs), 0.0, MAX_COVER)
    keep = rng.random(count) * density_max < -np.log1p(-cover) / area
    dirs = dirs[keep]
    wind = upper_wind_at(spec, weather, dirs, sheet)
    turn = _wind_turn(spec, weather, dirs, wind, sheet)
    steady = rng.random(len(dirs)) < _smoothstep(
        spec.max_turn, 0.75 * spec.max_turn, turn
    )
    dirs, wind = dirs[steady], wind[steady]
    n = len(dirs)

    east, north = _tangent_frame(dirs)
    heading = np.arctan2(
        np.einsum("ij,ij->i", wind, north), np.einsum("ij,ij->i", wind, east)
    )
    wander = value_noise(dirs * sheet / spec.turn_wavelength + OFFSETS["turn"])
    heading += np.radians(spec.turn) * 2.0 * (wander - 0.5)
    heading += np.radians(spec.jitter) * rng.normal(size=n)
    along = np.cos(heading)[:, None] * east + np.sin(heading)[:, None] * north
    mirror = rng.choice([-1.0, 1.0], n)
    across = np.cross(dirs, along) * mirror[:, None]
    return Stamps(
        centres=dirs * sheet,
        along=along,
        across=across,
        index=rng.integers(len(atlas_records()), size=n),
        scale=rng.uniform(*spec.scale_range, n),
        aspect=rng.uniform(*spec.aspect_range, n),
        opacity=rng.uniform(*spec.opacity_range, n),
        sheet=sheet,
    )


def _sample_stamp(index: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear thickness of atlas stamp *index* at its (x, y) in [0, 1]
    (outside: 0), as the GPU's linear filter reads it."""
    atlas = atlas_array()
    r = atlas_records()[index]
    rows, columns = atlas.shape
    u = (r["u_min"] + x * r["u_size"]) * columns - 0.5
    v = (r["v_min"] + y * r["v_size"]) * rows - 0.5
    inside = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
    out = map_coordinates(atlas, np.stack([v, u]), order=1, mode="nearest")
    return np.where(inside, out, 0.0)


def stamp_thickness(stamps: Stamps, points: np.ndarray) -> np.ndarray:
    """The stamps' summed thickness x opacity at sheet *points* (what the
    shader integrates), on the CPU: for the preview and the tests.

    :param points: (N, 3) on (or near) the sheet, planet frame, metres
    :returns: (N,)
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    total = np.zeros(len(points))
    if len(stamps) == 0 or len(points) == 0:
        return total
    length, width = STAMP_METRES
    reach = (
        0.5
        * math.hypot(length, width)
        * float(np.max(stamps.scale) * max(1.0, float(np.max(stamps.aspect))))
    )
    tree = cKDTree(points)
    for i, near in enumerate(tree.query_ball_point(stamps.centres, reach)):
        if not near:
            continue
        d = points[near] - stamps.centres[i]
        x = d @ stamps.along[i] / (length * stamps.scale[i]) + 0.5
        y = d @ stamps.across[i] / (width * stamps.scale[i] * stamps.aspect[i]) + 0.5
        total[near] += stamps.opacity[i] * _sample_stamp(int(stamps.index[i]), x, y)
    return total


def stamp_cover(stamps: Stamps, points: np.ndarray) -> np.ndarray:
    """Whether each sheet point is under a stamp's footprint (any stamp's
    own thickness above :data:`FOOTPRINT_THRESHOLD`, opacity aside)."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    covered = np.zeros(len(points), dtype=bool)
    if len(stamps) == 0:
        return covered
    length, width = STAMP_METRES
    reach = (
        0.5
        * math.hypot(length, width)
        * float(np.max(stamps.scale) * max(1.0, float(np.max(stamps.aspect))))
    )
    tree = cKDTree(points)
    for i, near in enumerate(tree.query_ball_point(stamps.centres, reach)):
        if not near:
            continue
        d = points[near] - stamps.centres[i]
        x = d @ stamps.along[i] / (length * stamps.scale[i]) + 0.5
        y = d @ stamps.across[i] / (width * stamps.scale[i] * stamps.aspect[i]) + 0.5
        inside = _sample_stamp(int(stamps.index[i]), x, y) > FOOTPRINT_THRESHOLD
        covered[np.asarray(near)[inside]] = True
    return covered


def load_cirrus_atlas(game: Any = None) -> Texture:
    """The baked stamps as a texture: mipmapped and anisotropic (the sheet is
    mostly seen far and grazing), clamped at its edges.

    Through ``game.app.asset_manager`` when the game has one, as
    :func:`cloud.load_cloud_atlas`; straight from disk otherwise.
    """
    asset_manager = getattr(getattr(game, "app", None), "asset_manager", None)
    if asset_manager is not None:
        tex = asset_manager.get_asset(
            asset_type="texture", path=ATLAS_PNG
        ).get_texture()
    else:
        tex = TexturePool.load_texture(Filename.from_os_specific(str(ATLAS_PNG)))
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    tex.set_anisotropic_degree(ANISOTROPY)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


# ── The geometry ───────────────────────────────────────────────────────────────


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
        InternalName.make("i_opacity"), 1, GeomEnums.NT_float32, GeomEnums.C_other
    )
    combined = GeomVertexFormat()
    combined.add_array(fmt)
    return GeomVertexFormat.register_format(combined)


def stamp_vertices(stamps: Stamps) -> tuple[np.ndarray, np.ndarray]:
    """The stamps' vertices and triangles: a :data:`STAMP_QUADS` grid each,
    every vertex on the sheet's sphere.

    :returns: (N * per stamp, 6) float32 rows (x, y, z, u, v, opacity), and
        uint32 triangle indices
    """
    qx, qy = STAMP_QUADS
    length, width = STAMP_METRES
    sx = np.linspace(0.0, 1.0, qx + 1)
    sy = np.linspace(0.0, 1.0, qy + 1)
    gx, gy = (a.ravel() for a in np.meshgrid(sx, sy))  # row-major: y outer
    n = len(stamps)
    half_length = (stamps.scale * length)[:, None, None]
    half_width = (stamps.scale * stamps.aspect * width)[:, None, None]
    offset = (gx[None, :, None] - 0.5) * half_length * stamps.along[:, None, :] + (
        gy[None, :, None] - 0.5
    ) * half_width * stamps.across[:, None, :]
    position = stamps.centres[:, None, :] + offset
    position *= stamps.sheet / np.linalg.norm(position, axis=2, keepdims=True)
    records = atlas_records()
    rect = np.array(
        [[r["u_min"], r["v_min"], r["u_size"], r["v_size"]] for r in records]
    )
    r = rect[stamps.index]
    u = r[:, None, 0] + gx[None, :] * r[:, None, 2]
    v = r[:, None, 1] + gy[None, :] * r[:, None, 3]
    rows = np.concatenate(
        [
            position,
            u[..., None],
            v[..., None],
            np.broadcast_to(stamps.opacity[:, None, None], (n, len(gx), 1)),
        ],
        axis=2,
    ).reshape(-1, 6)
    per = len(gx)
    quads = []
    for j in range(qy):
        for i in range(qx):
            a = j * (qx + 1) + i
            quads += [a, a + 1, a + qx + 2, a, a + qx + 2, a + qx + 1]
    tris = (np.arange(n)[:, None] * per + np.array(quads)[None, :]).ravel()
    return rows.astype(np.float32), tris.astype(np.uint32)


def _groups(stamps: Stamps) -> np.ndarray:
    """Each stamp's group: its centre's cube face and cell on it."""
    d = stamps.centres / np.linalg.norm(stamps.centres, axis=1, keepdims=True)
    major = np.argmax(np.abs(d), axis=1)
    sign = np.sign(d[np.arange(len(d)), major])
    face = major * 2 + (sign < 0)
    others = np.array([[1, 2], [0, 2], [0, 1]])[major]
    plane = (
        np.take_along_axis(d, others, axis=1)
        / np.abs(d[np.arange(len(d)), major])[:, None]
    )
    cell = np.clip(((plane + 1.0) * 0.5 * GROUP_CELLS).astype(int), 0, GROUP_CELLS - 1)
    return (face * GROUP_CELLS + cell[:, 0]) * GROUP_CELLS + cell[:, 1]


def _group_node(rows: np.ndarray, tris: np.ndarray, name: str) -> GeomNode:
    vdata = GeomVertexData(name, _vertex_format(), GeomEnums.UH_static)
    vdata.unclean_set_num_rows(len(rows))
    memoryview(vdata.modify_array(0)).cast("B")[:] = rows.tobytes()
    prim = GeomTriangles(GeomEnums.UH_static)
    prim.set_index_type(GeomEnums.NT_uint32)
    prim.add_next_vertices(len(tris))
    memoryview(prim.modify_vertices()).cast("B")[:] = tris.tobytes()
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode(name)
    node.add_geom(geom)
    xyz = rows[:, 0:3]
    node.set_bounds(BoundingBox(Point3(*xyz.min(axis=0)), Point3(*xyz.max(axis=0))))
    node.set_final(True)
    return node


# ── Drawing ────────────────────────────────────────────────────────────────────


@functools.cache
def _shader() -> Shader:
    return Shader.load(
        Shader.SL_GLSL,
        vertex=_SHADERS / "cirrus.vert",
        fragment=_SHADERS / "cirrus.frag",
    )


class CloudSheet:
    """A planet's high cirrus as stamps on a surface: drawn behind and in
    front of the clouds below it (see the module). The sheet layer of
    :class:`planet.PlanetClouds`, which a game builds instead, above a
    :class:`sphere.CloudVolume`.

    Usage::

        sheet = CloudSheet(render, planet_radius, executor,
                           atmospheres=atmospheres, atmosphere=0)
        sheet.set_sun(sun_direction)
        # every frame, camera in the world frame (float64):
        sheet.update(camera_world)

    :ivar root: the node both of its nodes hang under (hide it to hide it)
    :ivar behind: the node drawing where sight lines leave the sheet
    :ivar front: the node drawing where they enter it
    :ivar spec: the cirrus
    :ivar stamps: where they are (:class:`Stamps`); None until the set-up is
        done
    :ivar stats: ``pending`` 1 while the set-up runs in the pool, ``stamps``
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``): the
    #: clouds' (only ``reflected`` applies).
    SETTINGS_TYPE: ClassVar = CloudSettings

    def __init__(
        self,
        parent: NodePath,
        radius: float,
        executor: ChunkExecutor,
        position: ArrayLike = (0.0, 0.0, 0.0),
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        spec: CirrusSpec = CirrusSpec(),
        planet: PlanetSpec | None = None,
        weather: WeatherSpec | None = None,
        weather_seed: int = 11,
        settings: CloudSettings | None = None,
        sun_brightness: float = 2.8,
        sky_strength: float = 0.35,
        exposure: float = 0.9,
        game: Any = None,
    ):
        """
        :param parent: usually ``render``
        :param radius: the planet's sea-level radius, metres
        :param executor: the world's :class:`terrain.workers.ChunkExecutor`
            (the placement runs in it)
        :param position: the planet centre in the world frame (float64)
        :param atmospheres: the world's atmospheres (the cirrus is seen
            through them)
        :param atmosphere: the planet's own atmosphere index (reddens the
            cirrus's sunlight); None = none
        :param spec: the cirrus (:class:`CirrusSpec`)
        :param planet: the planet the weather follows; None = an earthlike
            one of this radius
        :param weather: the weather in full (the cumulus's, for the same
            systems); None = the defaults for *planet* with *weather_seed*
        :param weather_seed: the weather systems' seed
        :param settings: :class:`cloud.CloudSettings` (``reflected``); None =
            HIGH
        :param sun_brightness: direct-term gain (as the cumulus's)
        :param sky_strength: ambient sky weight (live: :meth:`set_sky_light`)
        :param exposure: radiance scale before the sky's tonemap
        :param game: the game object, for its asset manager (None: from disk)
        """
        settings = settings or CloudSettings()
        ensure_sky_bin()
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=np.float64)
        self.spec = spec
        self.stats = {"pending": 0, "stamps": 0}
        if weather is None:
            planet = planet or PlanetSpec(name="clouds", radius=self.radius)
            weather = WeatherSpec(planet, seed=weather_seed)
        self._job = executor.submit_once(cirrus_stamps, spec, weather, self.radius)
        self.stamps: Stamps | None = None
        self._settings = settings
        self._sky_light: float | None = None

        self.root = parent.attach_new_node("cirrus")
        self.root.stash()  # out of every camera until the set-up is done
        self.behind = self.root.attach_new_node("cirrus_behind")
        self.front = self.root.attach_new_node("cirrus_front")
        self.behind.set_bin(SKY_BIN, BEHIND_DRAW_ORDER)
        self.front.set_bin(FRONT_BIN, 0)
        self.behind.set_shader_input("uFront", 0)
        self.front.set_shader_input("uFront", 1)

        root = self.root
        root.set_shader(_shader())
        root.set_two_sided(True)
        root.set_transparency(TransparencyAttrib.M_none)
        root.set_depth_test(True)
        root.set_depth_write(False)
        root.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_one,
                ColorBlendAttrib.O_one_minus_incoming_alpha,
            )
        )
        bind_defaults(root)
        if atmospheres is not None:
            atmospheres.bind(root)
        self._sun = PTA_LVecBase3f.empty_array(1)
        self._sun[0] = LVecBase3f(0.0, 0.0, 1.0)
        o = spec.optics
        forward, backward, isotropic = phase_weights(o)
        root.set_shader_inputs(
            cirrusAtlas=load_cirrus_atlas(game),
            planetRadius=self.radius,
            uThickness=float(spec.thickness),
            uExtinction=float(spec.extinction),
            uMaxOpticalDepth=float(spec.max_optical_depth),
            uOwnAtmosphere=-1 if atmosphere is None else int(atmosphere),
            sunDir=self._sun,
            sunColor=Vec3(1.0, 1.0, 1.0),
            skyColor=Vec3(0.45, 0.6, 0.85),
            sunBrightness=float(sun_brightness),
            skyStrength=float(sky_strength),
            exposure=float(exposure),
            uCloudDebug=0,
            uPhase=Vec4(forward, backward, isotropic, clamped_anisotropy(o)),
            uLight=Vec4(o.multiple_scattering, o.shadow_strength, 0.0, 0.0),
        )
        self._apply_live(settings)

    # ── settings ──────────────────────────────────────────────────────────────

    @property
    def settings(self) -> CloudSettings:
        """What the clouds cost (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: CloudSettings) -> None:
        """Take the clouds' settings: only ``reflected`` applies (live).

        :param settings: the new settings
        """
        self._settings = settings
        self._apply_live(settings)

    def _apply_live(self, settings: CloudSettings) -> None:
        if settings.reflected:
            self.root.show(REFLECTION_CAMERA_MASK)
        else:
            self.root.hide(REFLECTION_CAMERA_MASK)

    # ── live ──────────────────────────────────────────────────────────────────

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
        """The sky's (ambient) light on the cirrus, as ``sky_strength``."""
        if strength != self._sky_light:
            self._sky_light = strength
            self.root.set_shader_input("skyStrength", float(strength))

    def set_debug(self, flags: int) -> None:
        """Profiling switches (``uCloudDebug`` bits): 2 no atmosphere, 8 flat
        colour.

        :param flags: the bits, 0 = normal
        """
        self.root.set_shader_input("uCloudDebug", int(flags))

    def update(
        self, camera_world: ArrayLike, camera_render: ArrayLike = (0.0, 0.0, 0.0)
    ) -> None:
        """Per frame: build the stamps when the placement is in, and place
        the sheet.

        :param camera_world: camera in the world frame, metres (float64)
        :param camera_render: camera in the render frame (the origin when
            the scene is drawn camera-relative)
        """
        if self._job is not None:
            if not self._job.done():
                self.stats["pending"] = 1
                return
            self._finish()
        cam = np.asarray(camera_world, dtype=np.float64) - self.position
        self.root.set_pos(*(np.asarray(camera_render, dtype=np.float64) - cam))

    def remove(self) -> None:
        """Take the cirrus off the scene."""
        self.root.remove_node()

    def _finish(self) -> None:
        """The placement becomes geometry, in culled groups, each drawn by
        both nodes; the sheet is shown."""
        stamps = self._job.result()
        self._job = None
        self.stamps = stamps
        rows, tris = stamp_vertices(stamps)
        per_stamp_rows = len(rows) // max(len(stamps), 1)
        per_stamp_tris = len(tris) // max(len(stamps), 1)
        groups = _groups(stamps)
        for group in np.unique(groups):
            chosen = np.nonzero(groups == group)[0]
            row_index = (
                chosen[:, None] * per_stamp_rows + np.arange(per_stamp_rows)
            ).ravel()
            tri_index = (
                chosen[:, None] * per_stamp_tris + np.arange(per_stamp_tris)
            ).ravel()
            # Renumbered to the group's own vertices
            local = (
                tris[tri_index].reshape(len(chosen), -1)
                - (chosen[:, None] * per_stamp_rows)
                + (np.arange(len(chosen))[:, None] * per_stamp_rows)
            )
            node = _group_node(
                rows[row_index], local.ravel().astype(np.uint32), f"cirrus_{group}"
            )
            twin = GeomNode(node.get_name())
            twin.add_geom(node.modify_geom(0))  # the same vertices, drawn again
            twin.set_bounds(node.get_bounds())
            twin.set_final(True)
            self.behind.attach_new_node(node)
            self.front.attach_new_node(twin)
        self.stats.update(pending=0, stamps=len(stamps))
        self.root.unstash()

    def __repr__(self) -> str:
        return f"CloudSheet(radius={self.radius:g}, base={self.spec.base:g})"
