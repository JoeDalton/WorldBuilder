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

import time
from collections import OrderedDict
from pathlib import Path

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
from world_builder.terrain.biomes import BIOME_COLORS
from world_builder.terrain.chunkgen import ChunkData
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
#: Generated chunks kept in memory (about 205 KB each), at most.
STORE_SIZE = 2000
#: Uploaded chunks kept on the GPU, shown or not, at most.
NODE_CACHE_SIZE = 1200
#: Below the caps, a terrain keeps this many times what its current selection
#: needs (at least MIN_RESIDENT): a planet seen only from afar shrinks to a
#: few hundred chunks instead of holding the ground it was last close to.
RESIDENT_FACTOR = 3
MIN_RESIDENT = 256
#: An LOD pass runs at least this often (frames), and after uploads at most
#: this often (it always runs at once when the camera has moved enough).
SELECT_EVERY = 15
SELECT_EVERY_ON_UPLOAD = 3
#: Upload budget per frame.
UPLOADS_PER_FRAME = 6
UPLOAD_MS_PER_FRAME = 4.0
#: Biome transition softness (see terrain.frag uBlendDepth).
DEFAULT_BLEND_DEPTH = 1.0
#: Every bit the ocean's reflection camera may carry (its mask is bits 0-19
#: less the ocean's own bit 1). Panda draws a node for a camera while any of
#: the camera's bits is still shown on it, so all of them are hidden at once;
#: the main cameras (all bits, or the split-depth bits 21 and 22) still see it.
REFLECTION_CAMERA_BITS = BitMask32(0xFFFFF)


class Terrain:
    """Draws one planet's terrain around a moving camera.

    :ivar root: the node every drawn chunk hangs under (shader inputs live here)
    :ivar position: the planet centre in the world frame, float64 metres
    :ivar spec: the current (resolved) planet spec; change it with
        :meth:`set_spec`
    :ivar planet: the :class:`Planet` evaluating it
    :ivar grid: its chunk grid
    :ivar query: collision and height queries on the drawn triangles
    :ivar materials: the biome textures (share them with other terrains)
    :ivar executor: the generation pool
    :ivar stats: counters and timings for an on-screen readout
    """

    def __init__(
        self,
        parent: NodePath,
        spec: PlanetSpec,
        backend: str = "process",
        workers: int | None = None,
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        max_distance: float | None = None,
        sun_direction: ArrayLike = (0.4, 0.5, 0.6),
        sun_color: ArrayLike = (2.4, 2.25, 2.0),
        ambient: ArrayLike = (0.35, 0.45, 0.65),
        exposure: float = 0.9,
        materials: MaterialSet | None = None,
        min_level: int = 0,
        position: ArrayLike = (0.0, 0.0, 0.0),
        executor: ChunkExecutor | None = None,
    ):
        """
        :param parent: node to hang the terrain root under; must not be offset
            by large distances (use ``render``, not the game root)
        :param spec: the planet
        :param backend: chunk generation backend, see :class:`ChunkExecutor`
            (ignored with *executor*)
        :param workers: generation worker count; None =
            :func:`workers.default_workers` (ignored with *executor*)
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
        :param materials: biome textures; loaded from the textures folder if None
        :param min_level: finest LOD level drawn (0 = full detail, about 2 m
            vertex spacing; each level up doubles the spacing)
        :param position: the planet centre in the world frame, metres
            (float64; settable later through :attr:`position`)
        :param executor: a generation pool shared with other terrains (this
            terrain takes its own channel and does not shut it down); None
            creates a private one from *backend* and *workers*
        """
        self.root = parent.attach_new_node("terrain")
        self.root.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "terrain.vert",
                fragment=SHADER_PATH / "terrain.frag",
            )
        )
        self.max_distance = max_distance
        self.materials = materials or load_materials()
        self.blend_depth = DEFAULT_BLEND_DEPTH
        self.min_level = min_level
        self._set_static_inputs(sun_direction, sun_color, ambient, exposure)
        bind_defaults(self.root)
        if atmospheres is not None:
            atmospheres.bind(self.root)
        self.root.set_shader_input(
            "uOwnAtmosphere", -1 if atmosphere is None else int(atmosphere)
        )

        self.position = np.asarray(position, dtype=np.float64)
        self._owns_executor = executor is None
        self.executor = executor or ChunkExecutor(backend=backend, workers=workers)
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
        )
        self.query = TerrainQuery(
            self.planet, self.store, version=version, level=self.min_level
        )
        self._last_select_pos = None
        self.stats["finest"] = self.grid.max_level

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
        # Re-select after moving a quarter of a leaf (29 m at 600 km), or a
        # 250th of the altitude: well inside the selector's 15 % hysteresis
        # band (a leaf splits at 2 leaf sizes), so no split or merge waits
        # visibly, and fast flight does not run a pass every frame.
        threshold = max(self.grid.chunk_size(0) / 4.0, 0.004 * self.altitude(cam))
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
            or (uploaded and since >= SELECT_EVERY_ON_UPLOAD)
            or since >= SELECT_EVERY
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

    def set_textured(self, textured: bool) -> None:
        """Toggle textures (off: flat biome colours)."""
        self.root.set_shader_input("uTextured", 1 if textured else 0)

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

    def set_reflected(self, reflected: bool) -> None:
        """Whether the ocean's reflection camera draws the terrain (on by
        default; it draws the terrain a second time).

        The reflection pass clips at the water plane: the vertex shader writes
        ``gl_ClipDistance`` from Panda's clip plane for it.
        """
        if reflected:
            self.root.show(REFLECTION_CAMERA_BITS)
        else:
            self.root.hide(REFLECTION_CAMERA_BITS)

    def set_min_level(self, level: int) -> None:
        """Finest LOD level drawn: 0 = full detail; each step up drops one level.

        Collision queries follow: they read the same level's grid, so the
        physical ground stays the one on screen.
        """
        self.min_level = max(0, min(int(level), self.grid.max_level))
        self.selector.min_level = self.min_level
        self.query.level = self.min_level
        self._last_select_pos = None

    def set_max_distance(self, max_distance: float | None) -> None:
        """Draw distance cap; None = out to the horizon."""
        self.max_distance = max_distance
        self.selector.max_distance = max_distance
        self._last_select_pos = None

    def destroy(self) -> None:
        """Remove the geometry, and stop the workers if they are this
        terrain's own."""
        if self._owns_executor:
            self.executor.shutdown()
        self.root.remove_node()

    # ── internals ──────────────────────────────────────────────────────────────

    def _set_static_inputs(self, sun_direction, sun_color, ambient, exposure):
        root = self.root
        m = self.materials
        root.set_shader_input(
            "uAlbedoH", make_texture_array("terrain_albedo_h", m.albedo_height, True)
        )
        root.set_shader_input(
            "uNormRA", make_texture_array("terrain_normal_ra", m.normal_rough_ao, False)
        )
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
        if self._origin is None or np.any(origin != self._origin):
            self._origin = origin
            for key, node in self.nodes.items():
                self._place(key, node)
        self.root.set_pos(*(render_cam - (cam - origin)))

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
        excess = len(self.store) - self._resident(STORE_SIZE)
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
            if count >= UPLOADS_PER_FRAME:
                break
            if (time.perf_counter() - start) * 1e3 > UPLOAD_MS_PER_FRAME:
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
        excess = len(self.nodes) - self._resident(NODE_CACHE_SIZE)
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
        self._wanted = selection.wanted + selection.prefetch
        self.executor.request(
            [k for k in self._wanted if k not in self.store], self._channel
        )
        self._evict_store()
        self._evict_nodes()
        self.stats["select_ms"] = (time.perf_counter() - start) * 1e3
        self.stats["finest"] = min((k[0] for k in draw), default=self.grid.max_level)
