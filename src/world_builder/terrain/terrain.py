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

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BitMask32,
    LVecBase3f,
    LVecBase4f,
    NodePath,
    PTA_float,
    PTA_LVecBase3f,
    Shader,
    Vec3,
)

from world_builder import SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M
from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.depth import REFLECTION_CAMERA_MASK
from world_builder.quality import RESTART, QualityLevel, QualitySettings, knob
from world_builder.terrain.biomes import BIOME_COLORS
from world_builder.terrain.chunkgen import ChunkData
from world_builder.terrain.collision import TerrainCollision
from world_builder.terrain.materials import (
    MaterialSet,
    load_materials,
    make_noise_texture,
    make_texture_array,
)
from world_builder.terrain.mesh import build_chunk_node
from world_builder.terrain.planet import Planet, PlanetSpec
from world_builder.terrain.quadtree import UNKNOWN_BOUNDS, LodSelector
from world_builder.terrain.query import TerrainQuery
from world_builder.terrain.workers import ChunkExecutor

SHADER_PATH = Path(__file__).parent / "shaders"
#: The terrain root is re-based to the camera snapped to this grid, metres;
#: above ORIGIN_SNAP * 16 of altitude the grid grows to altitude / 16.
ORIGIN_SNAP = 4096.0
LOGGER = logging.getLogger(__name__)
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
        position: ArrayLike = (0.0, 0.0, 0.0),
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
        :param position: the planet centre in the world frame, metres
            (float64; settable later through :attr:`position`)
        :param executor: a generation pool shared with other terrains (this
            terrain takes its own channel and does not shut it down); None
            creates a private one from *backend* and ``settings.workers``
        :param settings: what the terrain costs (:class:`TerrainSettings`;
            :meth:`set_settings` changes it); None = the HIGH preset
        """
        settings = settings or TerrainSettings()
        self._settings = settings
        self.root = parent.attach_new_node("terrain")
        self.root.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "terrain.vert",
                fragment=SHADER_PATH / "terrain.frag",
            )
        )
        self.max_distance = max_distance
        self.materials = materials or load_materials(max_size=settings.texture_size)
        self.blend_depth = DEFAULT_BLEND_DEPTH
        self.min_level = settings.min_level
        self._set_static_inputs(sun_direction, sun_color, ambient, exposure)
        bind_defaults(self.root)
        if atmospheres is not None:
            atmospheres.bind(self.root)
        self.root.set_shader_input(
            "uOwnAtmosphere", -1 if atmosphere is None else int(atmosphere)
        )

        self.position = np.asarray(position, dtype=np.float64)
        self._owns_executor = executor is None
        self.executor = executor or ChunkExecutor(
            backend=backend, workers=settings.workers or None
        )
        self._channel = f"terrain-{id(self):x}"
        self.planet: Planet | None = None
        self.query: TerrainQuery | None = None
        self.selector: LodSelector | None = None
        self.store: OrderedDict[tuple, ChunkData] = OrderedDict()
        # (h_min, h_max) per stored chunk: one tuple per chunk, which the
        # selector's geometry cache recognises by identity.
        self._bounds: dict[tuple, tuple[float, float]] = {}
        self.nodes: OrderedDict[tuple, NodePath] = OrderedDict()
        self.shown: set[tuple] = set()
        self._keep: set[tuple] = set()  # drawn chunks and their ancestors
        self._origin = None
        self._snap = ORIGIN_SNAP
        self._camera = np.zeros(3)
        self._render_camera = np.zeros(3)
        self._last_select_pos = None
        self._last_select_frame = 0
        self._frame = 0
        self._wanted: list[tuple] = []
        self.collision: TerrainCollision | None = None
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
        self.planet = Planet(spec)
        self.spec = self.planet.spec
        self.grid = self.planet.grid
        self.root.set_shader_input("uMaxLevel", self.grid.max_level)
        for node in self.nodes.values():
            node.remove_node()
        self.nodes.clear()
        self.store.clear()
        self._bounds.clear()
        self.shown.clear()
        self._keep = set()
        self.selector = LodSelector(
            self.grid,
            max_distance=self.max_distance,
            horizon_floor=0.0 if self.spec.has_ocean else UNKNOWN_BOUNDS[0],
            min_level=self.min_level,
            split_factor=self._settings.split_factor,
        )
        self.query = TerrainQuery(
            self.planet, self.store, version=version, level=self.min_level
        )
        if self.collision is not None:
            self.collision.reset()
        self._last_select_pos = None
        self.stats["finest"] = self.grid.max_level

    def enable_collision(
        self,
        into_mask: BitMask32,
        name: str = "terrain",
        owner_tag: str = "owner",
        max_distance: float = 100_000.0,
    ) -> TerrainCollision:
        """Make the drawn terrain solid to a game's ``CollisionTraverser``.

        Adds one ``CollisionNode`` (into-only) beside :attr:`root`, holding a
        native solid that answers sphere and segment tests against the drawn
        triangles of every chunk within *max_distance* of the camera (sea
        surface included if the planet has an ocean). Each collider gets at
        most one entry per frame, with the surface point and normal. The
        node's *owner_tag* python tag is a
        :class:`~world_builder.terrain.collision.TerrainSurface`: its
        ``material_at(point)`` names the ground at a hit.

        :param into_mask: the node's into collide mask
        :param name: the node's name (the ``%in`` of event patterns)
        :param owner_tag: the python tag the owner is stored under
        :param max_distance: chunks farther from the camera are not solid,
            metres
        :returns: the collision (also :attr:`collision`)
        :raises NativeCollisionError: the native library is not built for
            the running Panda3D (``scripts/build_native.sh``)
        """
        self.disable_collision()
        self.collision = TerrainCollision(
            self, into_mask, name, owner_tag, max_distance
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
        self, camera_world: ArrayLike, camera_render: ArrayLike = (0.0, 0.0, 0.0)
    ) -> None:
        """Per-frame update: re-base the root, take finished chunks, upload a
        few, and run an LOD pass when the camera has moved enough (or new
        chunks wait to be shown).

        :param camera_world: camera position in the world frame, metres,
            float64 (the planet frame if :attr:`position` is the origin)
        :param camera_render: where the camera is in the render frame (the
            game keeps it at the origin)
        """
        self._frame += 1
        cam = np.asarray(camera_world, dtype=np.float64) - self.position
        render_cam = np.asarray(camera_render, dtype=np.float64)
        self._camera = cam.copy()
        self._render_camera = render_cam.copy()
        self._rebase(cam, render_cam)

        finished = self.executor.collect(self._channel)
        for chunk in finished:
            self._store(chunk)
        if finished:
            self.stats["generated"] += len(finished)
            gen = [c.gen_ms for c in finished]
            self.stats["gen_ms"] = 0.9 * self.stats["gen_ms"] + 0.1 * float(
                np.mean(gen)
            )

        uploaded = self._upload()
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
        # frame was most of the frame.
        since = self._frame - self._last_select_frame
        if (
            moved
            or (uploaded and since >= settings.select_every_on_upload)
            or since >= settings.select_every
        ):
            self._select(cam)

        frame = self._frame_inputs
        frame["uCamPos"][0] = LVecBase3f(*render_cam)
        frame["uPlanetCentre"][0] = LVecBase3f(*(render_cam - cam))
        self.stats.update(
            shown=len(self.shown),
            stored=len(self.store),
            uploaded=len(self.nodes),
            pending=self.executor.pending_for(self._channel),
            wanted=len(self._wanted),
        )

    @property
    def radius(self) -> float:
        """Sea-level radius, metres."""
        return self.spec.radius

    def to_planet(self, point_world: ArrayLike) -> np.ndarray:
        """:returns: a world-frame point in this planet's frame (float64)"""
        return np.asarray(point_world, dtype=np.float64) - self.position

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
        return (
            np.asarray(point_planet, dtype=np.float64)
            - self._camera
            + self._render_camera
        )

    def set_sea_surface(self, on: bool) -> None:
        """Draw the sea as flat water at sea level (seen from high up, where
        the swell ocean is not shown), instead of the bare seabed.

        Seabed vertices are lifted to sea level and shaded as water, tinted by
        depth, reflecting the sky light at grazing angles.

        :param on: water on; off (the default) draws the seabed for the swell
            ocean to cover
        """
        self.root.set_shader_input("uSeaSurface", 1 if on else 0)

    def set_shallow_water(self, color: ArrayLike, depth: float) -> None:
        """The sea surface's tint over shallow seabed: give it the swell
        ocean's (``Ocean(shallow_color=..., shallow_depth=...)``), so the coast
        keeps its colour across their handover. Defaults to the shared
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

    def _set_static_inputs(self, sun_direction, sun_color, ambient, exposure):
        root = self.root
        m = self.materials
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
        root.set_shader_input("uLevel", 0)
        root.set_shader_input("uMaxLevel", 0)
        root.set_shader_input("uFace", 0)
        root.set_shader_input("uChunk", LVecBase4f(0.0, 0.0, 1.0, 0.0))
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
        root.set_shader_input("uSeaSurface", 0)
        self.set_shallow_water(SHALLOW_WATER_COLOR, SHALLOW_WATER_DEPTH_M)
        root.set_shader_input("uDepthBias", 6.0)
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
            for key, node in self.nodes.items():
                self._place(key, node)
        self.root.set_pos(*(render_cam - (cam - origin)))
        if self.collision is not None:
            self.collision.follow(self.root, origin, rebased)

    def _place(self, key, node: NodePath) -> None:
        node.set_pos(*(node.get_python_tag("centre") - self._origin))

    def _store(self, chunk: ChunkData) -> None:
        self.store[chunk.key] = chunk
        self._bounds[chunk.key] = (chunk.h_min, chunk.h_max)
        self.store.move_to_end(chunk.key)
        self._evict_store()

    def _resident(self, cap: int) -> int:
        """:returns: how many chunks to keep, given what the selection needs"""
        return min(cap, max(MIN_RESIDENT, RESIDENT_FACTOR * len(self._keep)))

    def _evict_store(self) -> None:
        """Drop least recently used chunks past the resident limit, never one
        the selection depends on or one waiting for upload."""
        excess = len(self.store) - self._resident(self._settings.store_size)
        if excess <= 0:
            return
        protected = self._keep | set(self._wanted)
        for old in list(self.store):
            if excess <= 0:
                break
            if old in protected:
                continue
            del self.store[old]
            del self._bounds[old]
            if old in self.nodes:
                self.nodes.pop(old).remove_node()
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
            node = NodePath(build_chunk_node(chunk))
            node.set_python_tag("centre", chunk.centre)
            u0, v0 = self.grid.uv_origin(key)
            node.set_shader_input("uLevel", key[0])
            node.set_shader_input("uFace", key[1])
            node.set_shader_input(
                "uChunk",
                LVecBase4f(
                    u0, v0, (1 << key[0]) * self.grid.uv_unit, chunk.skirt_depth
                ),
            )
            self._place(key, node)
            self.nodes[key] = node
            count += 1
        if count:
            per_chunk = (time.perf_counter() - start) * 1e3 / count
            self.stats["upload_ms"] = 0.8 * self.stats["upload_ms"] + 0.2 * per_chunk
        self._evict_nodes()
        return count

    def _evict_nodes(self) -> None:
        cap = min(self._settings.node_cache_size, self._settings.store_size)
        excess = len(self.nodes) - self._resident(cap)
        if excess <= 0:
            return
        # Wanted chunks were just uploaded for the next selection to use.
        wanted = set(self._wanted)
        for key in list(self.nodes):
            if excess <= 0:
                break
            if key not in self._keep and key not in wanted:
                self.nodes.pop(key).remove_node()
                excess -= 1

    def _select(self, cam: np.ndarray) -> None:
        start = time.perf_counter()
        self._last_select_pos = cam.copy()
        self._last_select_frame = self._frame

        selection = self.selector.select(
            cam,
            ready=self.nodes.__contains__,
            bounds=self._bounds.get,
        )
        draw = set(selection.draw)
        for key in self.shown - draw:
            node = self.nodes.get(key)
            if node is not None:
                node.detach_node()
        for key in draw - self.shown:
            self.nodes[key].reparent_to(self.root)
        # A split node's children must stay ready even when they are not drawn
        # (their own children are, or they are culled): the selector splits a
        # node only while all its visible children are ready. Evicting one as
        # "least recently drawn" collapsed its whole subtree to a coarse parent
        # for a few frames, then refined it again.
        keep = selection.keep
        for key in keep:
            if key in self.nodes:
                self.nodes.move_to_end(key)
            if key in self.store:
                self.store.move_to_end(key)
        self._keep = keep
        self.shown = draw
        if self.collision is not None:
            self.collision.update(draw, self.store, cam)
        self._wanted = selection.wanted + selection.prefetch
        self.executor.request(
            [k for k in self._wanted if k not in self.store], self._channel
        )
        self._evict_store()
        self._evict_nodes()
        self.stats["select_ms"] = (time.perf_counter() - start) * 1e3
        self.stats["finest"] = min((k[0] for k in draw), default=self.grid.max_level)
