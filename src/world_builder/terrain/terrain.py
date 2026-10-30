"""
Runtime terrain: LOD selection, background generation, upload, and drawing.

Usage::

    terrain = Terrain(parent=render, spec=PRESETS["earthlike"])
    # every frame, with the camera's float64 position in the world frame
    # (metres; the planet's centre is at terrain.position, the origin by
    # default):
    terrain.update(camera_world)

Several planets: give each its own ``position`` and one shared
:class:`ChunkExecutor` (``executor=``), and update them all with the same
camera. Positions and queries of one terrain are in its *planet frame*
(centre at the origin, same axes as the world frame); ``to_render`` takes
those.

Precision: Panda transforms are float32, which cannot place a vertex 600 km
from the origin to better than a few centimetres. The world therefore moves
around the camera: the terrain root hangs directly under ``render`` and is
moved every frame by ``origin - camera``, computed in float64, where
``origin`` is the camera position snapped to a 3D grid (4 km near the ground,
coarser with altitude, so orbital speeds do not re-place every chunk each
frame). Chunks sit under the root at ``centre - origin`` and only move when the
snapped origin does. Near the camera every float32 number involved is small.
Use :meth:`Terrain.to_render` to place other objects the same way.

Lighting: the sun's colour is given above any air. With its own atmosphere
(``atmosphere=``) the shader passes it through that air (reddened near the
terminator, gone in the planet's shadow); an airless body only has a
day/night line. The sky light fades out through twilight. The terrain is then
seen through every atmosphere of the world along the sight line.
"""

from __future__ import annotations

import functools
import logging
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BitMask32,
    CullBinManager,
    LPoint3f,
    LQuaternionf,
    LVecBase2f,
    LVecBase3f,
    NodePath,
    PTA_float,
    PTA_LVecBase3f,
    SamplerState,
    Shader,
    Texture,
    Vec3,
)

from world_builder import SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M
from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.depth import PASS_TAG, REFLECTION_CAMERA_MASK
from world_builder.ocean.ocean import bind_sea_defaults
from world_builder.ocean.swell import Swell
from world_builder.quality import RESTART, QualityLevel, QualitySettings, knob
from world_builder.terrain.biomes import BIOME_COLORS
from world_builder.terrain.chunkgen import ChunkData
from world_builder.terrain.collision import TerrainCollision
from world_builder.terrain.materials import (
    MaterialSet,
    load_materials,
    make_noise_texture,
    make_strata_texture,
    make_texture_array,
)
from world_builder.terrain.mesh import build_chunk_node
from world_builder.terrain.planet import Planet, PlanetSpec
from world_builder.terrain.quadtree import (
    LodSelector,
    NativeLodSelector,
    make_selector,
)
from world_builder.terrain.query import TerrainQuery
from world_builder.terrain.strata import HIGH as STRATA_HIGH
from world_builder.terrain.strata import LOW as STRATA_LOW
from world_builder.terrain.strata import strata_lut
from world_builder.terrain.workers import ChunkExecutor

SHADER_PATH = Path(__file__).parent / "shaders"


@functools.cache
def _shader(fragment: str) -> Shader:
    """:returns: the terrain's shader with this fragment stage: terrain.frag,
    or terrain_sea.frag (with the sea's near look)"""
    return Shader.load(
        Shader.SL_GLSL,
        vertex=SHADER_PATH / "terrain.vert",
        fragment=SHADER_PATH / fragment,
    )


def depth_only_shader() -> Shader:
    """:returns: the terrain's shader with a depth-only fragment stage, for
    passes that want its depth and nothing else (``terrain.vert`` kept, so
    the sea's lift and swell are right). Each terrain's root carries the
    tag ``PASS_TAG = "terrain"`` for a camera's tag state to swap it in."""
    return _shader("terrain_depth.frag")


def _ensure_terrain_bin() -> None:
    manager = CullBinManager.get_global_ptr()
    if manager.find_bin(TERRAIN_BIN) == -1:
        manager.add_bin(TERRAIN_BIN, CullBinManager.BT_front_to_back, TERRAIN_BIN_SORT)


#: The terrain root is re-based to the camera snapped to this grid, metres;
#: above ORIGIN_SNAP * 16 of altitude the grid grows to altitude / 16.
ORIGIN_SNAP = 4096.0
#: Chunks finer than the level whose chunks are about this wide, metres, hang
#: under a cell node: their ancestor at that level. A rebase then moves the
#: cells, not every chunk (it took 5-13 ms over ~1,500 nodes). The coarser
#: chunks, drawn only far away, hang under one cell at the planet's centre.
CELL_SIZE = 32_000.0
#: Slots the chunk table starts with (it doubles when they run out). A slot
#: holds one uploaded chunk's own shader values; the vertices name theirs.
CHUNK_TABLE_SLOTS = 2048
LOGGER = logging.getLogger(__name__)
#: The chunks' cull bin: front to back, before the opaque bin (20), so near
#: ground hides what lies behind it before that is shaded (early depth test).
#: Panda's opaque bin sorts by state, which chunks share, so its order was
#: arbitrary: front to back saves 1-1.6 ms a frame low over the sea.
TERRAIN_BIN = "terrain"
TERRAIN_BIN_SORT = 19
#: Chunks whose refinement lag :attr:`Terrain.refine_lags` keeps.
LAG_SAMPLES = 2000
#: Below the caps (``TerrainSettings.store_size``, ``node_cache_size``), a
#: terrain keeps this many times what its current selection needs (at least
#: MIN_RESIDENT): a planet seen only from afar shrinks to a few hundred chunks
#: instead of holding the ground it was last close to.
RESIDENT_FACTOR = 3
MIN_RESIDENT = 256
#: Biome transition softness (see terrain.frag uBlendDepth).
DEFAULT_BLEND_DEPTH = 1.0


@dataclass(frozen=True)
class TerrainSettings(QualitySettings):
    """What the terrain costs (:mod:`world_builder.quality`). All live but
    ``texture_size`` and ``workers``, which take effect at the next start.

    :ivar min_level: finest detail dropped: LOD levels below the finest
        (about 2 m vertex spacing at any radius), each doubling the spacing.
        Collision queries follow (they read the drawn grid), so the ground
        a ship hits stays the one on screen
    :ivar split_factor: how far detail reaches: a chunk splits nearer than
        this many of its own sizes
    :ivar detail_far: textures fade to the biomes' mean colours by this
        distance, metres, and are not sampled beyond
    :ivar anisotropy: the textures' anisotropic filtering (sharpness at
        grazing angles)
    :ivar textured: textures, or flat biome colours (cheapest)
    :ivar reflected: drawn in the ocean's reflection
    :ivar node_cache_size: uploaded chunks kept on the GPU, at most (at most
        ``store_size``)
    :ivar store_size: generated chunks kept in memory (~205 KB each), at most
    :ivar uploads_per_frame: chunks uploaded per frame, at most
    :ivar upload_ms_per_frame: and milliseconds spent on it
    :ivar select_every: frames between LOD passes, at most
    :ivar select_every_on_upload: frames between passes while uploading
    :ivar texture_size: biome texture size, texels (at most the textures'
        own and ``textures.yaml``'s ``max_size``)
    :ivar workers: generation processes for a terrain that makes its own
        pool (0 = one per physical core but one)
    """

    min_level: int = knob(
        0,
        "Finest detail dropped",
        unit="levels",
        range=(0, 8),
        help="Each level doubles the ground's finest vertex spacing (0 = about "
        "2 m): fewer chunks to generate and draw. Collisions follow the drawn "
        "ground.",
    )
    split_factor: float = knob(
        2.0,
        "Detail distance",
        unit="x",
        range=(1.25, 3.0),
        help="How far fine detail reaches: more chunks, generated and drawn.",
    )
    detail_far: float = knob(
        4000.0,
        "Texture distance",
        unit="m",
        range=(500.0, 10000.0),
        help="Textures fade to flat colours by this distance and are not "
        "sampled beyond.",
    )
    anisotropy: int = knob(
        8,
        "Texture filtering",
        unit="x",
        choices={"1x": 1, "2x": 2, "4x": 4, "8x": 8, "16x": 16},
        help="Anisotropic filtering: sharp ground at grazing angles.",
    )
    textured: bool = knob(True, "Textures", help="Off: flat biome colours.")
    reflected: bool = knob(
        True,
        "In reflections",
        help="Draw the terrain in the sea's reflection (about 1.1 ms).",
    )
    node_cache_size: int = knob(
        1200,
        "Chunks kept on the GPU",
        unit="chunks",
        range=(300, 4000),
        advanced=True,
    )
    store_size: int = knob(
        2000,
        "Chunks kept in memory",
        unit="chunks",
        range=(500, 6000),
        advanced=True,
        help="About 205 KB each.",
    )
    uploads_per_frame: int = knob(
        6,
        "Uploads per frame",
        unit="chunks",
        range=(1, 32),
        advanced=True,
    )
    upload_ms_per_frame: float = knob(
        4.0,
        "Upload time per frame",
        unit="ms",
        range=(0.5, 16.0),
        advanced=True,
    )
    select_every: int = knob(
        15,
        "LOD pass period",
        unit="frames",
        range=(1, 60),
        advanced=True,
    )
    select_every_on_upload: int = knob(
        3,
        "LOD pass period while loading",
        unit="frames",
        range=(1, 15),
        advanced=True,
    )
    texture_size: int = knob(
        2048,
        "Texture size",
        RESTART,
        unit="px",
        choices={"256": 256, "512": 512, "1024": 1024, "2048": 2048, "4096": 4096},
        help="Biome texture resolution (video memory, load time).",
    )
    workers: int = knob(
        0,
        "Generation processes",
        RESTART,
        range=(0, 64),
        advanced=True,
        help="0 = one per physical core but one.",
    )

    LABEL: ClassVar = "Terrain"
    PRESETS: ClassVar = {
        QualityLevel.LOW: {
            "min_level": 4,
            "split_factor": 1.5,
            "detail_far": 1500.0,
            "anisotropy": 2,
            "reflected": False,
            "node_cache_size": 800,
            "store_size": 1200,
            "uploads_per_frame": 4,
            "upload_ms_per_frame": 2.5,
            "texture_size": 512,
        },
        QualityLevel.MID: {
            "min_level": 2,
            "split_factor": 1.75,
            "detail_far": 2500.0,
            "anisotropy": 4,
            "node_cache_size": 1000,
            "store_size": 1600,
            "upload_ms_per_frame": 3.0,
            "texture_size": 1024,
        },
        QualityLevel.ULTRA: {
            "split_factor": 2.5,
            "detail_far": 6000.0,
            "anisotropy": 16,
            "node_cache_size": 1800,
            "store_size": 3000,
            "uploads_per_frame": 8,
            "upload_ms_per_frame": 6.0,
        },
    }


class Terrain:
    """Draws one planet's terrain around a moving camera.

    :ivar root: the node every drawn chunk hangs under (shader inputs live here)
    :ivar position: the planet centre in the world frame, float64 metres
    :ivar spec: the current (resolved) planet spec; change it with
        :meth:`set_spec`
    :ivar planet: the :class:`Planet` evaluating it
    :ivar grid: its chunk grid
    :ivar query: collision and height queries on the drawn triangles
    :ivar collision: the collision node for a game's traverser, once
        :meth:`enable_collision` is called (else None)
    :ivar materials: the biome textures (share them with other terrains)
    :ivar executor: the generation pool
    :ivar stats: counters and timings for an on-screen readout
    :ivar refine_lags: (level, seconds from the chunk first wanted by an LOD
        pass to it drawn), for the latest :data:`LAG_SAMPLES` chunks drawn
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = TerrainSettings

    def __init__(
        self,
        parent: NodePath,
        spec: PlanetSpec,
        backend: str = "process",
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        max_distance: float | None = None,
        sun_direction: ArrayLike = (0.4, 0.5, 0.6),
        sun_color: ArrayLike = (2.4, 2.25, 2.0),
        ambient: ArrayLike = (0.35, 0.45, 0.65),
        exposure: float = 0.9,
        materials: MaterialSet | None = None,
        textures: tuple[Texture, Texture] | None = None,
        tonemap: bool = True,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        executor: ChunkExecutor | None = None,
        settings: TerrainSettings | None = None,
    ):
        """
        :param parent: node to hang the terrain root under; must not be offset
            by large distances (use ``render``, not the game root)
        :param spec: the planet
        :param backend: chunk generation backend, see :class:`ChunkExecutor`
            (ignored with *executor*)
        :param atmospheres: the world's atmospheres: the terrain is seen
            through all of them (aerial perspective); None = no air anywhere
        :param atmosphere: index of this planet's own atmosphere in
            *atmospheres*: its sunlight passes through it (reddened low, none
            in the planet's shadow); None = airless
        :param max_distance: terrain beyond this 3D distance is not drawn;
            None = out to the horizon
        :param sun_direction: toward the sun, planet frame
        :param sun_color: direct light above any air, linear (may exceed 1)
        :param ambient: sky light, linear (near zero on an airless body); the
            shader fades it out through twilight
        :param exposure: pre-tonemap exposure, as the ocean's
        :param materials: biome textures; loaded from the textures folder if
            None (at ``settings.texture_size``)
        :param textures: the biome texture arrays already on the GPU (albedo
            and height, normal-roughness-ao: as ``make_texture_array`` makes
            them from *materials*), shared with other terrains; None = made
            here
        :param tonemap: the ACES tonemap (True), or a plain display encoding,
            next to models lit by Panda3D's fixed-function lights
        :param position: the planet centre in the world frame, metres
            (float64; settable later through :attr:`position`)
        :param orientation: world from planet frame, a unit quaternion (w, x,
            y, z) (:meth:`set_orientation`): a body that turns
        :param executor: a generation pool shared with other terrains (this
            terrain takes its own channel and does not shut it down); None
            creates a private one from *backend* and ``settings.workers``
        :param settings: what the terrain costs (:class:`TerrainSettings`;
            :meth:`set_settings` changes it); None = the HIGH preset
        """
        settings = settings or TerrainSettings()
        self._settings = settings
        self.root = parent.attach_new_node("terrain")
        self.root.set_shader(_shader("terrain.frag"))
        _ensure_terrain_bin()
        self.root.set_bin(TERRAIN_BIN, 0)
        self.root.set_tag(PASS_TAG, "terrain")  # see depth_only_shader
        # The chunks with water hang here: a Sea gives them the shader with
        # its near look (set_sea_near), which the others never pay for.
        self.sea_root = self.root.attach_new_node("terrain_sea")
        #: How far the sea's swell may raise the water, metres (a Sea sets
        #: it): the chunks with water are culled by bounds this much wider
        self.sea_swell = 0.0
        #: The sea's swell (set_swell); None = calm water at sea level
        self.swell: Swell | None = None
        self.max_distance = max_distance
        self.materials = materials or load_materials(max_size=settings.texture_size)
        self.blend_depth = DEFAULT_BLEND_DEPTH
        self.min_level = settings.min_level
        self._set_static_inputs(sun_direction, sun_color, ambient, exposure, textures)
        self.root.set_shader_input("uTonemap", int(tonemap))
        bind_defaults(self.root)
        if atmospheres is not None:
            atmospheres.bind(self.root)
        self.root.set_shader_input(
            "uOwnAtmosphere", -1 if atmosphere is None else int(atmosphere)
        )

        self.position = np.asarray(position, dtype=np.float64)
        self.set_orientation(orientation)
        self._owns_executor = executor is None
        self.executor = executor or ChunkExecutor(
            backend=backend, workers=settings.workers or None
        )
        self._channel = f"terrain-{id(self):x}"
        self.planet: Planet | None = None
        self.query: TerrainQuery | None = None
        self.selector: LodSelector | NativeLodSelector | None = None
        self.store: OrderedDict[tuple, ChunkData] = OrderedDict()
        # (h_min, h_max) per stored chunk: one tuple per chunk, which the
        # selector's geometry cache recognises by identity.
        self._bounds: dict[tuple, tuple[float, float]] = {}
        self.nodes: OrderedDict[tuple, NodePath] = OrderedDict()
        # (cell key or None for the planet's, wet) -> (node, its centre)
        self._cells: dict[tuple, tuple[NodePath, np.ndarray]] = {}
        self.shown: set[tuple] = set()
        self._keep: set[tuple] = set()  # drawn chunks and their ancestors
        self._origin = None
        self._snap = ORIGIN_SNAP
        self._camera = np.zeros(3)
        self._render_camera = np.zeros(3)
        self._forward = None
        self._last_select_pos = None
        self._last_select_frame = 0
        self._unshown = 0  # chunks uploaded since the last LOD pass
        self._frame = 0
        self._wanted: list[tuple] = []
        self._loading = True  # the last LOD pass still wanted chunks
        self.collision: TerrainCollision | None = None
        self.refine_lags: deque[tuple[int, float]] = deque(maxlen=LAG_SAMPLES)
        self._wanted_since: dict[tuple, float] = {}
        self.stats = {
            "shown": 0,
            "stored": 0,
            "uploaded": 0,
            "pending": 0,
            "wanted": 0,
            "gen_ms": 0.0,
            "upload_ms": 0.0,
            "select_ms": 0.0,
            "finest": 0,
            "generated": 0,
        }
        self.set_spec(spec)
        self._apply_settings(settings)

    # ── settings ───────────────────────────────────────────────────────────────

    @property
    def settings(self) -> TerrainSettings:
        """What the terrain costs (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: TerrainSettings) -> None:
        """Change what the terrain costs: live, but ``texture_size`` and
        ``workers``, which only a new terrain takes (logged). Lower caches
        evict on the next update, never a chunk the drawn selection needs.
        """
        changes = self._settings.changes(settings)
        if not changes:
            return
        self._settings = settings
        for name, mode in changes.items():
            if mode == RESTART:
                LOGGER.info(f"terrain setting {name} takes effect at the next start")
        self._apply_settings(settings)

    def _apply_settings(self, settings: TerrainSettings) -> None:
        # The finest level drawn; collision queries read the same level's
        # grid, so the physical ground stays the one on screen.
        self.min_level = min(settings.min_level, self.grid.max_level)
        self.selector.min_level = self.min_level
        self.query.level = self.min_level
        self.selector.split_factor = settings.split_factor
        self._last_select_pos = None
        self.root.set_shader_input("uDetailFar", float(settings.detail_far))
        for texture in self._textures:
            texture.set_anisotropic_degree(settings.anisotropy)
        self.root.set_shader_input("uTextured", int(settings.textured))
        # The reflection pass draws the terrain a second time; it clips at
        # the water plane (the vertex shader writes gl_ClipDistance).
        if settings.reflected:
            self.root.show(REFLECTION_CAMERA_MASK)
        else:
            self.root.hide(REFLECTION_CAMERA_MASK)

    # ── public API ─────────────────────────────────────────────────────────────

    def set_spec(self, spec: PlanetSpec) -> None:
        """Switch to another planet (or seed); regenerates everything, coarse first.

        :param spec: the new planet
        """
        version = self.executor.set_spec(spec, self._channel)
        self.version = version
        self.planet = spec.evaluator()
        self.spec = self.planet.spec
        self.grid = self.planet.grid
        self.root.set_shader_input("uMaxLevel", self.grid.max_level)
        self.root.set_shader_input("uSeaSurface", int(self.spec.has_ocean))
        # A planet's day and night follow its vertical; a rock is lit as an
        # object, each face as it faces the sun (RockSpec.radial_light)
        self.root.set_shader_input(
            "uRadialLight", int(getattr(self.spec, "radial_light", True))
        )
        # The rock strata's tints, by altitude (none: a white texel)
        strata = getattr(self.planet, "strata", None)
        lut = None if strata is None else strata_lut(strata)
        self.root.set_shader_input("uStrata", make_strata_texture(lut))
        self.root.set_shader_input("uStrataRange", LVecBase2f(STRATA_LOW, STRATA_HIGH))
        for node in self.nodes.values():
            node.remove_node()
        self.nodes.clear()
        self._slots = {}
        self._free_slots = list(range(len(self._chunk_values[0]) - 1, -1, -1))
        for cell, _ in self._cells.values():
            cell.remove_node()
        self._cells.clear()
        size0 = self.grid.chunk_size(0)
        self._cell_level = min(
            self.grid.max_level, max(0, round(math.log2(CELL_SIZE / size0)))
        )
        self.store.clear()
        self._bounds.clear()
        self.shown.clear()
        self._keep = set()
        # The native pass when the library is built; it keeps its own copy
        # of what is stored (bounds) and uploaded (ready), told below.
        self.selector = make_selector(
            self.grid,
            max_distance=self.max_distance,
            horizon_floor=0.0 if self.spec.has_ocean else self.planet.height_bounds[0],
            unknown_bounds=self.planet.height_bounds,
            min_level=self.min_level,
            split_factor=self._settings.split_factor,
        )
        # Chunks the query generates (generate=True) join the store like the
        # workers' do, bounds included.
        self.query = TerrainQuery(
            self.planet,
            self.store,
            version=version,
            level=self.min_level,
            store_chunk=self._store,
        )
        self.query.swell = getattr(self, "swell", None)
        if self.collision is not None:
            self.collision.reset()
        self._last_select_pos = None
        self._loading = True
        self.stats["finest"] = self.grid.max_level

    def enable_collision(
        self,
        into_mask: BitMask32,
        name: str = "terrain",
        owner_tag: str = "owner",
        max_distance: float = 100_000.0,
        include_ocean: bool | None = None,
    ) -> TerrainCollision:
        """Make the drawn terrain solid to a game's ``CollisionTraverser``.

        Adds one ``CollisionNode`` (into-only) beside :attr:`root`, holding a
        native solid that answers sphere and segment tests against the drawn
        triangles of every chunk within *max_distance* of the camera (and the
        sea surface, see *include_ocean*). Each collider gets at
        most one entry per frame, with the surface point and normal. The
        node's *owner_tag* python tag is a
        :class:`~world_builder.terrain.collision.TerrainSurface`: its
        ``material_at(point)`` names the ground at a hit.

        :param into_mask: the node's into collide mask
        :param name: the node's name (the ``%in`` of event patterns)
        :param owner_tag: the python tag the owner is stored under
        :param max_distance: chunks farther from the camera are not solid,
            metres
        :param include_ocean: the sea surface is solid (else the seabed
            under it); None = if the planet has an ocean, as
            :class:`TerrainQuery` does
        :returns: the collision (also :attr:`collision`)
        :raises NativeCollisionError: the native library is not built for
            the running Panda3D (``invoke native``)
        """
        self.disable_collision()
        self.collision = TerrainCollision(
            self, into_mask, name, owner_tag, max_distance, include_ocean
        )
        if self._origin is not None:
            self.collision.follow(self.root, self._origin, rebased=True)
            self.collision.update(self.shown, self.store, self._camera)
        return self.collision

    def disable_collision(self) -> None:
        """Remove the collision node, if any."""
        if self.collision is not None:
            self.collision.destroy()
            self.collision = None

    def update(
        self,
        camera_world: ArrayLike,
        camera_render: ArrayLike = (0.0, 0.0, 0.0),
        camera_forward: ArrayLike | None = None,
    ) -> None:
        """Per-frame update: re-base the root, take finished chunks, upload a
        few, and run an LOD pass when the camera has moved enough (or new
        chunks wait to be shown).

        :param camera_world: camera position in the world frame, metres,
            float64 (the planet frame if :attr:`position` is the origin)
        :param camera_render: where the camera is in the render frame (the
            game keeps it at the origin)
        :param camera_forward: the view direction (world frame), so ground in
            view is generated before ground behind; None = no preference
        """
        self._frame += 1
        cam = self.to_planet(camera_world)
        render_cam = np.asarray(camera_render, dtype=np.float64)
        self._camera = cam.copy()
        self._render_camera = render_cam.copy()
        if camera_forward is not None and self._rotated:
            camera_forward = self._rotation.T @ np.asarray(camera_forward, dtype=float)
        self._forward = camera_forward
        self._rebase(cam, render_cam)

        finished = self.executor.collect(self._channel)
        for chunk in finished:
            self._store(chunk)
        # Once per frame (and per pass), not per chunk: it scans the store.
        self._evict_store()
        if finished:
            self.stats["generated"] += len(finished)
            gen = [c.gen_ms for c in finished]
            self.stats["gen_ms"] = 0.9 * self.stats["gen_ms"] + 0.1 * float(
                np.mean(gen)
            )

        self._unshown += self._upload()
        # Re-select after moving a quarter of a leaf (29 m at 600 km) at the
        # default split factor of 2, or a 250th of the altitude: well inside
        # the selector's 15 % hysteresis band (a leaf splits at split_factor
        # leaf sizes), so no split or merge waits visibly, and fast flight
        # does not run a pass every frame.
        settings = self._settings
        threshold = max(
            settings.split_factor * self.grid.chunk_size(0) / 8.0,
            0.004 * self.altitude(cam),
        )
        moved = (
            self._last_select_pos is None
            or np.linalg.norm(cam - self._last_select_pos) > threshold
        )
        # New uploads only need a pass to be shown, which can wait a frame or
        # two: while refining, uploads land every frame, and a pass per
        # frame was most of the frame. Uploads wait for it whichever frame
        # they landed in.
        since = self._frame - self._last_select_frame
        if (
            moved
            or (self._unshown and since >= settings.select_every_on_upload)
            or since >= settings.select_every
        ):
            self._select(cam)

        frame = self._frame_inputs
        frame["uCamPos"][0] = LVecBase3f(*render_cam)
        frame["uPlanetCentre"][0] = LVecBase3f(*(render_cam - self._rotation @ cam))
        self.stats.update(
            shown=len(self.shown),
            stored=len(self.store),
            uploaded=len(self.nodes),
            pending=self.executor.pending_for(self._channel),
            wanted=len(self._wanted),
        )

    @property
    def loading(self) -> bool:
        """Whether the view still waits for ground: the last LOD pass found
        chunks it needs that are not drawn yet (prefetch does not count).
        True until the first pass and after :meth:`set_spec`.

        The signal for a level's loading phase: ground that is not drawn is
        not solid either (:meth:`enable_collision` answers for drawn chunks
        only). In flight it turns on briefly whenever detail refines, so wait
        for it to stay False for a moment rather than for one frame.
        """
        return self._loading

    @property
    def radius(self) -> float:
        """Sea-level radius, metres."""
        return self.spec.radius

    def set_orientation(self, orientation: ArrayLike) -> None:
        """Turn the body: world from planet frame, a unit quaternion (w, x, y,
        z). A turning body (an asteroid) sets it every frame; chunks, queries
        and collision stay in the planet frame."""
        q = np.asarray(orientation, dtype=np.float64)
        q = q / np.linalg.norm(q)
        self.orientation = q
        w, x, y, z = q
        self._rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        self._rotated = not np.array_equal(q, [1.0, 0.0, 0.0, 0.0])

    @property
    def rotation(self) -> np.ndarray:
        """(3, 3) world from planet frame"""
        return self._rotation

    def to_planet(self, point_world: ArrayLike) -> np.ndarray:
        """:returns: a world-frame point in this planet's frame (float64)"""
        offset = np.asarray(point_world, dtype=np.float64) - self.position
        return offset @ self._rotation if self._rotated else offset

    def altitude(self, camera_planet: ArrayLike) -> float:
        """:returns: metres above sea level of a planet-frame position"""
        return float(np.linalg.norm(camera_planet)) - self.spec.radius

    def to_render(self, point_planet: ArrayLike) -> np.ndarray:
        """Where a planet-frame point is in the render frame this frame.

        Computed in float64 against the camera, so near the camera it is
        exact; give the result to ``node.set_pos(render, ...)``.

        :param point_planet: (3,) or (N, 3) metres
        :returns: same shape, render-frame metres
        """
        relative = np.asarray(point_planet, dtype=np.float64) - self._camera
        if self._rotated:
            relative = relative @ self._rotation.T
        return relative + self._render_camera

    def set_swell(self, swell: Swell | None) -> None:
        """The sea's swell now (a :class:`world_builder.ocean.Sea` gives it
        every frame): queries and collisions follow the water it raises, as
        the shader draws it. None = calm water at sea level."""
        self.swell = swell
        self.query.swell = swell
        if self.collision is not None:
            self.collision.set_swell(swell)

    def set_sea_near(self, on: bool) -> None:
        """Draw the chunks with water with the sea's near look
        (:class:`world_builder.ocean.Sea` turns it on below its altitude
        band), or with the far look only (the plain shader)."""
        if on != getattr(self, "_sea_near", False):
            self._sea_near = on
            if on:
                self.sea_root.set_shader(_shader("terrain_sea.frag"))
            else:
                self.sea_root.clear_shader()  # the root's

    def set_shallow_water(self, color: ArrayLike, depth: float) -> None:
        """The sea's tint over shallow seabed, near and far (a
        :class:`world_builder.ocean.Sea` sets its own). Defaults to the shared
        :data:`world_builder.SHALLOW_WATER_COLOR` and
        :data:`world_builder.SHALLOW_WATER_DEPTH_M`.

        :param color: linear (r, g, b)
        :param depth: metres over which the deep colour takes over (e-folding)
        """
        self.root.set_shader_input("uShallowWater", LVecBase3f(*color))
        self.root.set_shader_input("uShallowDepth", float(depth))

    def set_sun(
        self, direction: ArrayLike, color: ArrayLike, ambient: ArrayLike
    ) -> None:
        """Relight the terrain (no rebuild).

        :param direction: toward the sun, planet frame (the render frame's
            axes are the planet's)
        :param color: direct light above any air, linear (reddened by the
            planet's own atmosphere, if any, in the shader)
        :param ambient: sky light, linear (near zero on an airless body)
        """
        frame = self._frame_inputs
        frame["uSunDir"][0] = Vec3(*direction).normalized()
        frame["uSunColor"][0] = LVecBase3f(*color)
        frame["uAmbient"][0] = LVecBase3f(*ambient)

    def set_debug_mode(self, mode: int) -> None:
        """Debug view: 0 shaded, 1 LOD level, 2 biomes (flat colours, lit),
        3 slope, 4 normals, 5 skirts (magenta, over LOD-tinted shading)."""
        self.root.set_shader_input("uDebugMode", int(mode))

    def set_blend_depth(self, depth: float) -> None:
        """Biome transition softness: ~0.1 crisp height-cut edges, >= 1 smooth."""
        self.blend_depth = depth
        self.root.set_shader_input("uBlendDepth", float(depth))

    def set_wireframe(self, on: bool) -> None:
        """Toggle wireframe rendering of every chunk."""
        if on:
            self.root.set_render_mode_wireframe()
        else:
            self.root.clear_render_mode()

    def set_max_distance(self, max_distance: float | None) -> None:
        """Draw distance cap; None = out to the horizon."""
        self.max_distance = max_distance
        self.selector.max_distance = max_distance
        self._last_select_pos = None

    def destroy(self) -> None:
        """Remove the geometry, and stop the workers if they are this
        terrain's own."""
        self.disable_collision()
        if self._owns_executor:
            self.executor.shutdown()
        self.root.remove_node()

    # ── internals ──────────────────────────────────────────────────────────────

    def _set_static_inputs(self, sun_direction, sun_color, ambient, exposure, textures):
        root = self.root
        m = self.materials
        if textures is not None:
            albedo, normal = textures
            self._textures = ()  # their owner sets their filtering
        else:
            albedo = make_texture_array("terrain_albedo_h", m.albedo_height, True)
            normal = make_texture_array("terrain_normal_ra", m.normal_rough_ao, False)
            self._textures = (albedo, normal)  # their anisotropy is a setting
        root.set_shader_input("uAlbedoH", albedo)
        root.set_shader_input("uNormRA", normal)
        root.set_shader_input("uNoise", make_noise_texture())
        tile = PTA_float.empty_array(len(m.tile_m))
        mean = PTA_LVecBase3f.empty_array(len(m.tile_m))
        flat = PTA_LVecBase3f.empty_array(len(m.tile_m))
        for i in range(len(m.tile_m)):
            tile[i] = float(m.tile_m[i])
            mean[i] = LVecBase3f(*m.mean_albedo[i])
            flat[i] = LVecBase3f(*(BIOME_COLORS[i] ** 2.2))
        root.set_shader_input("uTileM", tile)
        root.set_shader_input("uMeanAlbedo", mean)
        root.set_shader_input("uFlatAlbedo", flat)
        root.set_shader_input("uTextured", 1)
        root.set_shader_input("uDetailFar", 4000.0)
        root.set_shader_input("uBlendDepth", self.blend_depth)
        root.set_shader_input("uExposure", float(exposure))
        root.set_shader_input("uDebugMode", 0)
        root.set_shader_input("uMaxLevel", 0)
        self._make_chunk_table(CHUNK_TABLE_SLOTS)
        # Inputs that change every frame are arrays bound once and written in
        # place: setting a new value on the root would make a new render
        # state per input per frame, for Panda to compose and collect.
        self._frame_inputs = {}
        for name, value in (
            ("uCamPos", (0.0, 0.0, 0.0)),
            ("uPlanetCentre", (0.0, 0.0, -1.0)),
            ("uSunDir", (0.0, 0.0, 1.0)),
            ("uSunColor", (1.0, 1.0, 1.0)),
            ("uAmbient", (0.0, 0.0, 0.0)),
        ):
            array = PTA_LVecBase3f.empty_array(1)
            array[0] = LVecBase3f(*value)
            self._frame_inputs[name] = array
            root.set_shader_input(name, array)
        # The sea (set_spec turns it on for a planet with one): its seabed
        # lifted to the water and drawn as such (the far look; a Sea adds the
        # near one).
        root.set_shader_input("uSeaSurface", 0)
        self.set_shallow_water(SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M)
        bind_sea_defaults(root)
        self.set_sun(sun_direction, sun_color, ambient)

    def _rebase(self, cam: np.ndarray, render_cam: np.ndarray) -> None:
        altitude = max(self.altitude(cam), 0.0)
        snap = ORIGIN_SNAP
        while snap * 16.0 < altitude:
            snap *= 2.0
        self._snap = snap
        origin = np.round(cam / snap) * snap
        rebased = self._origin is None or bool(np.any(origin != self._origin))
        if rebased:
            self._origin = origin
            for name, (cell, centre) in list(self._cells.items()):
                if cell.get_num_children() == 0:  # made again when needed
                    cell.remove_node()
                    del self._cells[name]
                else:
                    cell.set_pos(*(centre - origin))
        if self._rotated:
            self.root.set_pos_quat(
                LPoint3f(*(render_cam - self._rotation @ (cam - origin))),
                LQuaternionf(*self.orientation),
            )
        else:
            self.root.set_pos(*(render_cam - (cam - origin)))
        if self.collision is not None:
            self.collision.follow(self.root, origin, rebased)

    def _cell(self, key, wet: bool) -> tuple[NodePath, np.ndarray]:
        """:returns: the cell node a chunk hangs under (made if new), and the
        cell's centre, planet frame. The chunk's position is relative to it:
        float32 offsets of at most ~CELL_SIZE."""
        level, face, i, j = key
        top = self._cell_level
        shift = top - level
        cell_key = (top, face, i >> shift, j >> shift) if level < top else None
        name = (cell_key, wet)
        entry = self._cells.get(name)
        if entry is None:
            centre = (
                np.zeros(3)
                if cell_key is None
                else self.grid.centre_dir(cell_key) * self.spec.radius
            )
            cell = (self.sea_root if wet else self.root).attach_new_node("cell")
            if self._origin is not None:
                cell.set_pos(*(centre - self._origin))
            entry = self._cells[name] = (cell, centre)
        return entry

    def _store(self, chunk: ChunkData) -> None:
        """Keep a chunk; the store is trimmed by the next :meth:`update`."""
        self.store[chunk.key] = chunk
        # The ground's bounds, seabed included, also where the sea is drawn
        # over it: deep water stays coarse (its surface has no relief to
        # resolve), the shallows refine like the land (the coast, the foam).
        # Culling uses the chunk's own box, which holds the sea surface.
        bounds = (chunk.h_min, chunk.h_max)
        self._bounds[chunk.key] = bounds
        self.selector.set_bounds(chunk.key, bounds)
        self.store.move_to_end(chunk.key)

    def _resident(self, cap: int) -> int:
        """:returns: how many chunks to keep, given what the selection needs"""
        return min(cap, max(MIN_RESIDENT, RESIDENT_FACTOR * len(self._keep)))

    def _evict_store(self) -> None:
        """Drop least recently used chunks past the resident limit, never one
        the selection depends on or one waiting for upload."""
        excess = len(self.store) - self._resident(self._settings.store_size)
        if excess <= 0:
            return
        keep, wanted = self._keep, set(self._wanted)
        for _ in range(len(self.store)):
            if excess <= 0:
                break
            old = next(iter(self.store))
            if old in keep or old in wanted:
                # In use: the most recently used now, so the oldest stay in
                # front (walking past every kept chunk cost up to 0.6 ms).
                self.store.move_to_end(old)
                continue
            del self.store[old]
            del self._bounds[old]
            self.selector.set_bounds(old, None)
            if old in self.nodes:
                self._drop_node(old)
            excess -= 1

    def _upload(self) -> int:
        start = time.perf_counter()
        count = 0
        for key in self._wanted:
            if count >= self._settings.uploads_per_frame:
                break
            if (time.perf_counter() - start) * 1e3 > self._settings.upload_ms_per_frame:
                break
            chunk = self.store.get(key)
            if chunk is None or key in self.nodes:
                continue
            wet = bool(self.spec.has_ocean and chunk.h_min < 0.0)
            slot = self._take_slot(key, chunk)
            node = NodePath(
                build_chunk_node(chunk, self.sea_swell if wet else 0.0, slot)
            )
            node.set_python_tag("wet", wet)
            if wet and chunk.h_max < 0.0:
                # All water: the reflection clips every vertex of it (its
                # surface is not in its own mirror). Its bounds reach above
                # the mirror plane, so the clip plane would not cull it.
                node.hide(REFLECTION_CAMERA_MASK)
            node.set_pos(*(chunk.centre - self._cell(key, wet)[1]))
            self.nodes[key] = node
            self.selector.set_ready(key, True)
            count += 1
        if count:
            self._write_chunk_table(self._new_slots)
            self._new_slots = []
            per_chunk = (time.perf_counter() - start) * 1e3 / count
            self.stats["upload_ms"] = 0.8 * self.stats["upload_ms"] + 0.2 * per_chunk
        self._evict_nodes()
        return count

    def _make_chunk_table(self, slots: int) -> None:
        """The chunk table: per slot, an uploaded chunk's own shader values,
        three float texels (terrain.vert: centre and swell band; uv origin,
        uv step and skirt depth; level and face). One texture for every
        chunk instead of inputs per chunk node, so Panda draws them all in
        one state (per-chunk inputs cost ~2.5 ms a frame low over the
        planet). Kept rows (old values) when it grows."""
        old = getattr(self, "_chunk_values", None)
        values = np.zeros((4, slots, 4), dtype=np.float32)  # 4 rows: a power of two
        if old is not None:
            values[:, : old.shape[1]] = old
        self._chunk_values = values
        texture = Texture("terrain_chunks")
        texture.setup_2d_texture(slots, 4, Texture.T_float, Texture.F_rgba32)
        texture.set_compression(Texture.CM_off)
        texture.set_minfilter(SamplerState.FT_nearest)
        texture.set_magfilter(SamplerState.FT_nearest)
        texture.set_wrap_u(SamplerState.WM_clamp)
        texture.set_wrap_v(SamplerState.WM_clamp)
        self._chunk_texture = texture
        self._write_chunk_table()
        self.root.set_shader_input("uChunkTable", texture)
        if old is None:
            self._slots: dict[tuple, int] = {}
            self._free_slots = list(range(slots - 1, -1, -1))
            self._new_slots: list[int] = []
        else:
            self._free_slots = list(range(slots - 1, old.shape[1] - 1, -1)) + (
                self._free_slots
            )

    def _write_chunk_table(self, slots: list[int] | None = None) -> None:
        """Copy the table into the texture: every slot, or just these."""
        values = self._chunk_values
        ram = np.frombuffer(
            memoryview(self._chunk_texture.modify_ram_image()), np.float32
        ).reshape(values.shape)
        # Panda keeps an RGBA texture's RAM image as BGRA.
        if slots is None:
            ram[:] = values[..., [2, 1, 0, 3]]
        else:
            ram[:, slots] = values[:, slots][..., [2, 1, 0, 3]]

    def _take_slot(self, key, chunk: ChunkData) -> int:
        """A free slot of the chunk table, filled with the chunk's values
        (written to the texture once per upload pass)."""
        if not self._free_slots:
            self._make_chunk_table(2 * self._chunk_values.shape[1])
        slot = self._free_slots.pop()
        self._slots[key] = slot
        self._new_slots.append(slot)
        level, face = key[0], key[1]
        u0, v0 = self.grid.uv_origin(key)
        values = self._chunk_values
        values[0, slot] = (*chunk.centre, 0.25 / self.grid.spacing(level))
        values[1, slot] = (u0, v0, (1 << level) * self.grid.uv_unit, chunk.skirt_depth)
        values[2, slot] = (level, face, 0.0, 0.0)
        return slot

    def _drop_node(self, key) -> None:
        """Remove an uploaded chunk's node and free its slot."""
        self.nodes.pop(key).remove_node()
        self._free_slots.append(self._slots.pop(key))
        self.selector.set_ready(key, False)

    def _evict_nodes(self) -> None:
        cap = min(self._settings.node_cache_size, self._settings.store_size)
        excess = len(self.nodes) - self._resident(cap)
        if excess <= 0:
            return
        # Wanted chunks were just uploaded for the next selection to use.
        keep, wanted = self._keep, set(self._wanted)
        for _ in range(len(self.nodes)):
            if excess <= 0:
                break
            key = next(iter(self.nodes))
            if key in keep or key in wanted:
                self.nodes.move_to_end(key)  # in use (see _evict_store)
            else:
                self._drop_node(key)
                excess -= 1

    def _select(self, cam: np.ndarray) -> None:
        start = time.perf_counter()
        self._last_select_pos = cam.copy()
        self._last_select_frame = self._frame
        self._unshown = 0

        selection = self.selector.select(
            cam,
            ready=self.nodes.__contains__,
            bounds=self._bounds.get,
            forward=self._forward,
        )
        draw = set(selection.draw)
        for key in self.shown - draw:
            node = self.nodes.get(key)
            if node is not None:
                node.detach_node()
        now = time.perf_counter()
        for key in draw - self.shown:
            node = self.nodes[key]
            node.reparent_to(self._cell(key, node.get_python_tag("wet"))[0])
            since = self._wanted_since.pop(key, None)
            if since is not None:
                self.refine_lags.append((key[0], now - since))
        for key in selection.wanted:
            self._wanted_since.setdefault(key, now)
        if len(self._wanted_since) > 4 * LAG_SAMPLES:  # never drawn: forget
            self._wanted_since = {
                k: t for k, t in self._wanted_since.items() if now - t < 30.0
            }
        # A split node's children must stay ready even when they are not drawn
        # (their own children are, or they are culled): the selector splits a
        # node only while all its visible children are ready. Evicting one as
        # "least recently drawn" collapsed its whole subtree to a coarse parent
        # for a few frames, then refined it again. Kept chunks are never
        # evicted, so their place in the LRU order matters only once they are
        # let go: they become the most recently used then (the order they
        # would have had if marked on every pass, for a fraction of the cost).
        keep = selection.keep
        for key in self._keep - keep:
            if key in self.nodes:
                self.nodes.move_to_end(key)
            if key in self.store:
                self.store.move_to_end(key)
        self._keep = keep
        self.shown = draw
        if self.collision is not None:
            self.collision.update(draw, self.store, cam)
        self._wanted = selection.wanted + selection.prefetch
        self._loading = bool(selection.wanted)
        self.executor.request(
            [k for k in self._wanted if k not in self.store], self._channel
        )
        self._evict_store()
        self._evict_nodes()
        self.stats["select_ms"] = (time.perf_counter() - start) * 1e3
        self.stats["finest"] = min((k[0] for k in draw), default=self.grid.max_level)
