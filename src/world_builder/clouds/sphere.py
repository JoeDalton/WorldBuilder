"""
Clouds over a whole planet, from inside the deck to orbit, with one pattern.

The density field is the billboard field's (:mod:`noise`), moved onto the
planet: the slab is a height above the sea-level sphere, the tileable noise is
sampled at planet-frame positions (3D, so it has no seam anywhere on the
sphere), and a weather field of direction (:func:`noise.weather_volume`)
shifts the fbm, so some regions are clear and others overcast. Every view of
the clouds evaluates that same function, so the pattern seen from orbit is the
one you fly into:

* **near** (the inner LOD shells): volumetric billboards, as the flat field
  draws them, generated per cube-sphere tile in the shared worker pool by
  rejection sampling of the density. Each shell doubles the billboards' size
  and the tiles' (``lod_shells``' trade: each shell costs half the previous)
  and crossfades with the next by distance, on optical depth.
* **far** (beyond the last shell, up to orbit): the *deck*, a sphere at
  mid-slab height whose fragments evaluate the density through the slab's
  depth. Billboards would thin out at that scale (their covered area falls as
  1 / size); the deck covers the planet with a few thousand triangles.

The clouds are geometry that depth-tests against the scene, like the flat
field: terrain occludes them with no depth plumbing. Tiles sort back to front
in their own bin; particles are re-sorted within each tile, a few tiles per
frame. Seen through the atmospheres (``atmosphere.Atmospheres``), lit by
sunlight reddened by the planet's own air. Static: no wind yet.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
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
    LVecBase2f,
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
    PRESETS,
    CloudQuality,
    CloudType,
    at_quality,
    atlas_mean_alpha,
    atlas_rects,
    load_cloud_atlas,
)
from world_builder.clouds.noise import (
    LAYER_VEC4S,
    MAX_LAYERS,
    NOISE_SIZE,
    DensityField,
    build_noise_texture,
    build_weather_texture,
    column_peaks,
    density_on_sphere,
    fbm_sigma,
    pack_layer_params,
    quantise,
    value_noise_volume,
    weather_volume,
    with_threshold,
)
from world_builder.terrain.cubesphere import QUADS, CubeGrid, face_dirs, grid_for
from world_builder.terrain.quadtree import LodSelector

_SHADERS = Path(__file__).parent / "shaders"
#: Back to front by node: the deck (centred on the planet) first, then tiles.
CLOUD_BIN = "cloud_sphere"
_CLOUD_BIN_SORT = 25
#: Side of a first-shell tile, metres (a quarter of the flat field's 32 km box).
BASE_TILE_SIZE = 8000.0
#: Tiles uploaded and re-sorted per frame.
UPLOADS_PER_FRAME = 3
RESORTS_PER_FRAME = 3
#: Tiles kept (shown or not) beyond those wanted, for flying back and forth.
TILE_CACHE = 160
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
    :ivar radius: (min, max) billboard radius, metres
    :ivar volume_fraction: billboard sphere volume per unit cloud volume
    """

    level: int
    radius: tuple
    volume_fraction: float


@dataclass(frozen=True)
class SphereCloudSpec:
    """Everything a worker needs to fill one tile; frozen and picklable.

    :ivar radius: the planet's sea-level radius, metres
    :ivar field: the cloud type's density field, threshold resolved
    :ivar noise_seed: seed of the shared noise volume
    :ivar weather_seed: seed of the weather volume
    :ivar weather_gain: fbm shift between the clearest and cloudiest weather
    :ivar shells: the billboard shells, innermost first
    """

    radius: float
    field: DensityField
    noise_seed: int
    weather_seed: int
    weather_gain: float
    shells: tuple

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
    :ivar gen_ms: generation time
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
def _weather(seed: int) -> np.ndarray:
    return weather_volume(seed)


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
    :func:`cloud.sample_field_particles`: the accept rate measures the cloud
    volume, which sets the count for the shell's volume fraction.

    :param spec: the clouds
    :param key: the tile
    :param version: carried through to the result
    """
    start = time.perf_counter()
    level, face, i, j = key
    shell = spec.shells[spec.shell_of(level)]
    grid = grid_for(spec.radius)
    volume = _noise(spec.noise_seed)
    weather = _weather(spec.weather_seed)
    field = spec.field
    base, thickness = field.slab
    span = 2.0 * (QUADS << level) / grid.n0
    a0, b0 = -1.0 + i * span, -1.0 + j * span
    mid = spec.radius + base + 0.5 * thickness
    slab_volume = _patch_area(face, a0, b0, span, mid) * thickness

    r_lo, r_hi = shell.radius
    mean_r3 = (r_hi**4 - r_lo**4) / (4.0 * (r_hi - r_lo)) if r_hi > r_lo else r_lo**3
    sphere = 4.0 / 3.0 * math.pi * mean_r3
    rng = np.random.default_rng(np.random.SeedSequence([spec.noise_seed, *key]))

    kept, tried, accepted, target = [], 0, 0, None
    while target is None or accepted < target:
        a = a0 + span * rng.uniform(size=batch)
        b = b0 + span * rng.uniform(size=batch)
        height = base + thickness * rng.uniform(size=batch)
        points = face_dirs(face, a, b) * (spec.radius + height)[:, None]
        inside = points[
            density_on_sphere(
                volume, weather, field, points, spec.radius, spec.weather_gain
            )
            > 0.0
        ]
        tried += batch
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


def _ensure_bin():
    manager = CullBinManager.get_global_ptr()
    if manager.find_bin(CLOUD_BIN) == -1:
        manager.add_bin(CLOUD_BIN, CullBinManager.BT_back_to_front, _CLOUD_BIN_SORT)


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


def _tile_node(tile: CloudTile) -> tuple[NodePath, GeomTriangles]:
    """A GeomNode for one tile (4 vertices per particle), and its index buffer."""
    p = tile.particles
    n = len(p)
    verts = np.empty((n, 4, 11), dtype=np.float32)
    verts[:, :, 0:3] = p[:, None, 0:3]
    verts[:, :, 3:5] = _CORNERS[None, :, :]
    verts[:, :, 5] = p[:, None, 3]
    verts[:, :, 6:10] = p[:, None, 4:8]
    verts[:, :, 10] = p[:, None, 8]
    vdata = GeomVertexData("cloud_tile", _vertex_format(), GeomEnums.UH_static)
    vdata.unclean_set_num_rows(4 * n)
    memoryview(vdata.modify_array(0)).cast("B")[:] = verts.tobytes()
    tris = GeomTriangles(GeomEnums.UH_dynamic)
    tris.set_index_type(GeomEnums.NT_uint32)
    tris.add_next_vertices(6 * n)
    order = (np.arange(n, dtype=np.uint32)[:, None] * 4 + _QUAD_TRIS).ravel()
    memoryview(tris.modify_vertices()).cast("B")[:] = order.tobytes()
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode(f"clouds{tile.key}")
    node.add_geom(geom)
    reach = float(p[:, 3].max())
    low = p[:, 0:3].min(axis=0) - reach
    high = p[:, 0:3].max(axis=0) + reach
    node.set_bounds(BoundingBox(Point3(*low), Point3(*high)))
    node.set_final(True)
    return NodePath(node), tris


def _deck_mesh(radius: float, quads: int = DECK_QUADS) -> GeomNode:
    """A cube-sphere of radius *radius*, both sides drawn."""
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
    node = GeomNode("cloud_deck")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    return node


@dataclass
class _Tile:
    node: NodePath
    tris: GeomTriangles
    data: CloudTile


class SphericalClouds:
    """One planet's clouds: billboard shells near the camera, the deck beyond.

    Usage::

        clouds = SphericalClouds(render, planet_radius, executor,
                                 atmospheres=atmospheres, atmosphere=0)
        # every frame, camera in the world frame (float64):
        clouds.update(camera_world)
    """

    def __init__(
        self,
        parent: NodePath,
        radius: float,
        executor,
        position=(0.0, 0.0, 0.0),
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        cloud_type: CloudType = CloudType.CUMULUS,
        altitude: float | None = None,
        shells: int = 4,
        coverage: float | None = None,
        weather_gain: float = 0.55,
        seed: int = 7,
        weather_seed: int = 11,
        quality: CloudQuality = CloudQuality.HIGH,
        sun_brightness: float = 2.8,
        sky_strength: float = 0.35,
        sky_occlusion: float = 0.55,
        exposure: float = 1.15,
        near_fade_radii: float = 2.5,
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
        :param shells: billboard LOD shells before the deck takes over (at
            most :data:`noise.MAX_LAYERS` - 1)
        :param coverage: overrides the type's mean coverage (0-1)
        :param weather_gain: how strongly the weather field opens or closes
            the deck by region (fbm units; 0 = the same everywhere)
        :param seed: noise seed (field and placement)
        :param weather_seed: weather pattern seed
        :param quality: billboard budget (see :class:`cloud.CloudQuality`)
        :param game: the game object, for its asset manager (None: from disk)
        """
        if not 1 <= shells < MAX_LAYERS:
            raise ValueError(f"1 to {MAX_LAYERS - 1} shells")
        _ensure_bin()
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=np.float64)
        self.executor = executor
        self._channel = f"clouds-{id(self):x}"
        self.grid: CubeGrid = grid_for(self.radius)
        self.stats = {"tiles": 0, "billboards": 0, "pending": 0, "gen_ms": 0.0}

        preset = PRESETS[cloud_type]
        field = (
            preset.field
            if coverage is None
            else replace(preset.field, coverage=coverage)
        )
        if altitude is not None:
            field = replace(field, slab=(float(altitude), field.slab[1]))
        volume = _noise(seed)
        sigma = fbm_sigma(volume, field)
        field = with_threshold(field, column_peaks(volume, field), sigma)
        self.field = field
        fbm_mean = float(volume.mean()) * float(sum(field.octave_weights))
        base_level = max(0, round(math.log2(BASE_TILE_SIZE / self.grid.chunk_size(0))))
        tile0 = self.grid.chunk_size(base_level)

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
            spec = at_quality(
                replace(
                    preset,
                    radius=(preset.radius[0] * scale, preset.radius[1] * scale),
                    optics=replace(
                        preset.optics,
                        max_optical_depth=preset.optics.max_optical_depth * scale,
                    ),
                ),
                quality,
            )
            shell_specs.append(
                ShellSpec(base_level + k, spec.radius, spec.volume_fraction)
            )
            params[k] = pack_layer_params(
                field,
                extinction=field.density / max(spec.volume_fraction * sprite, 1e-6),
                wrap_radius=1e12,
                wrap_fade_band=0.0,
                octave_mean=float(volume.mean()),
                optics=spec.optics,
                aspect=spec.aspect,
                fade_in=bands[k - 1] if k > 0 else None,
                fade_out=bands[k],
            )
        self._deck_layer = shells
        params[shells] = pack_layer_params(
            field,
            extinction=field.density,  # per metre of the slab's path
            wrap_radius=1e12,
            wrap_fade_band=0.0,
            octave_mean=float(volume.mean()),
            optics=replace(preset.optics, max_optical_depth=DECK_MAX_OPTICAL_DEPTH),
            fade_in=bands[-1],
        )
        self._bands = bands
        self.spec = SphereCloudSpec(
            radius=self.radius,
            field=field,
            noise_seed=seed,
            weather_seed=weather_seed,
            weather_gain=weather_gain,
            shells=tuple(shell_specs),
        )
        self._version = executor.set_spec(
            self.spec, self._channel, generator=generate_cloud_tile
        )

        # ── Nodes and shared inputs ──
        tile_shader, deck_shader = _shaders()
        self.root = parent.attach_new_node("spherical_clouds")
        self.tiles_root = self.root.attach_new_node("cloud_tiles")
        self.tiles_root.set_shader(tile_shader)
        self.deck = self.root.attach_new_node(
            _deck_mesh(self.radius + field.slab[0] + 0.5 * field.slab[1])
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
            weatherNoise=build_weather_texture(_weather(weather_seed)),
            weatherGain=float(weather_gain),
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
            uDeckStats=LVecBase2f(fbm_mean, sigma),
            uPixelAngle=0.001,
            earthCurvature=0.0,
            nearFadeRadii=float(near_fade_radii),
        )

        self._tiles: dict[Key, _Tile] = {}
        self._wanted: list[Key] = []
        self._shown: set[Key] = set()
        self._ready: dict[Key, CloudTile] = {}
        self._origin = None
        self._camera = np.zeros(3)
        self._last_select = None
        self._resort_cursor = 0
        self._frame = 0
        self._selector = LodSelector(self.grid)
        self._slab_box = (field.slab[0], field.slab[0] + field.slab[1])

    # ── public ────────────────────────────────────────────────────────────────

    def set_sun(self, direction, sun_color=None, sky_color=None) -> None:
        """:param direction: toward the sun, world frame"""
        d = np.asarray(direction, dtype=float)
        self._sun[0] = LVecBase3f(*(d / np.linalg.norm(d)))
        if sun_color is not None:
            self.root.set_shader_input("sunColor", Vec3(*sun_color))
        if sky_color is not None:
            self.root.set_shader_input("skyColor", Vec3(*sky_color))

    def set_pixel_angle(self, radians: float) -> None:
        """The angle one pixel spans (vertical fov / window height): how far
        the deck can show the clouds' own detail before averaging it."""
        self.root.set_shader_input("uPixelAngle", float(radians))

    def update(self, camera_world, camera_render=(0.0, 0.0, 0.0)) -> None:
        """Per frame: select, generate, upload, sort, place.

        :param camera_world: camera in the world frame, metres (float64)
        :param camera_render: camera in the render frame
        """
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

    def remove(self) -> None:
        self.root.remove_node()

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
        if len(self._tiles) > len(wanted) + TILE_CACHE:
            for key in [k for k in self._tiles if k not in wanted_set][
                : len(self._tiles) - len(wanted) - TILE_CACHE
            ]:
                self._tiles.pop(key).node.remove_node()

    def _upload(self) -> None:
        count = 0
        for key in self._wanted:
            if count >= UPLOADS_PER_FRAME:
                break
            tile = self._ready.pop(key, None)
            if tile is None or key in self._tiles:
                continue
            if tile.version != self._version:
                continue
            if len(tile.particles) == 0:
                node, tris = NodePath("empty"), None
            else:
                node, tris = _tile_node(tile)
            node.set_pos(*(tile.centre - self._origin))
            self._tiles[key] = _Tile(node, tris, tile)
            if tris is not None:
                node.reparent_to(self.tiles_root)
                self._shown.add(key)
            count += 1
        # Ready tiles nobody wants any more.
        for key in [k for k in self._ready if k not in set(self._wanted)]:
            del self._ready[key]

    def _resort(self, cam) -> None:
        """Back to front within a few shown tiles per frame (premultiplied over
        needs it; the bin sorts the tiles themselves)."""
        shown = [k for k in self._shown if self._tiles[k].tris is not None]
        if not shown:
            return
        shown.sort()
        for _ in range(min(RESORTS_PER_FRAME, len(shown))):
            self._resort_cursor = (self._resort_cursor + 1) % len(shown)
            tile = self._tiles[shown[self._resort_cursor]]
            p = tile.data.particles
            offset = tile.data.centre - cam
            distance = np.einsum("ij,ij->i", p[:, 0:3] + offset, p[:, 0:3] + offset)
            order = np.argsort(-distance).astype(np.uint32)
            index = (order[:, None] * 4 + _QUAD_TRIS).ravel()
            memoryview(tile.tris.modify_vertices()).cast("B")[:] = index.tobytes()
