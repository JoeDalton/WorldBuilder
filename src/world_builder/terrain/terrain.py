"""
Runtime terrain: LOD selection, background generation, upload, and drawing.

Usage::

    terrain = Terrain(parent=render, spec=PRESETS["earthlike"])
    # every frame, with the camera's float64 position in the planet frame
    # (planet centre at the origin, metres):
    terrain.update(camera_planet)

Precision: Panda transforms are float32, which cannot place a vertex 600 km
from the origin to better than a few centimetres. The world therefore moves
around the camera: the terrain root hangs directly under ``render`` and is
moved every frame by ``origin - camera``, computed in float64, where
``origin`` is the camera position snapped to a 3D grid (4 km near the ground,
coarser with altitude, so orbital speeds do not re-place every chunk each
frame). Chunks sit under the root at ``centre - origin`` and only move when the
snapped origin does. Near the camera every float32 number involved is small.
Use :meth:`Terrain.to_render` to place other objects the same way.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
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
#: Height of the haze shell above sea level, metres: the path length of a
#: line of sight inside it is what fades terrain to the haze colour.
HAZE_HEIGHT = 12000.0
#: Generated chunks kept in memory (about 170 KB each).
STORE_SIZE = 2000
#: Uploaded chunks kept on the GPU, shown or not.
NODE_CACHE_SIZE = 1200
#: Upload budget per frame.
UPLOADS_PER_FRAME = 6
UPLOAD_MS_PER_FRAME = 4.0
#: Biome transition softness (see terrain.frag uBlendDepth).
DEFAULT_BLEND_DEPTH = 1.0
#: Every camera bit of the ocean's reflection camera (bits 0-19).
REFLECTION_CAMERA_BITS = BitMask32(0xFFFFF)


class Terrain:
    """Draws one planet's terrain around a moving camera."""

    def __init__(
        self,
        parent: NodePath,
        spec: PlanetSpec,
        backend: str = "process",
        workers: int | None = None,
        haze_color=(0.55, 0.65, 0.8),
        haze_distance: float = 150000.0,
        haze_height: float = HAZE_HEIGHT,
        max_distance: float | None = None,
        sun_direction=(0.4, 0.5, 0.6),
        sun_color=(2.4, 2.25, 2.0),
        ambient=(0.35, 0.45, 0.65),
        exposure: float = 0.9,
        materials: MaterialSet | None = None,
        min_level: int = 0,
    ):
        """
        :param parent: node to hang the terrain root under; must not be offset
            by large distances (use ``render``, not the game root)
        :param spec: the planet
        :param backend: chunk generation backend, see :class:`ChunkExecutor`
        :param workers: generation worker count
        :param haze_color: colour distant terrain fades to (display space)
        :param haze_distance: metres of line of sight inside the haze shell
            over which terrain reaches the haze fully
        :param haze_height: top of the haze shell above sea level, metres
        :param max_distance: terrain beyond this 3D distance is not drawn;
            None = out to the horizon
        :param sun_direction: toward the sun
        :param sun_color: direct light, linear (may exceed 1)
        :param ambient: sky light, linear
        :param exposure: pre-tonemap exposure, as the ocean's
        :param materials: biome textures; loaded from the textures folder if None
        :param min_level: finest LOD level drawn (0 = full detail, about 2 m
            vertex spacing; each level up doubles the spacing)
        """
        self.root = parent.attach_new_node("terrain")
        self.root.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "terrain.vert",
                fragment=SHADER_PATH / "terrain.frag",
            )
        )
        self.haze_distance = haze_distance
        self.haze_height = haze_height
        self.max_distance = max_distance
        self.materials = materials or load_materials()
        self.blend_depth = DEFAULT_BLEND_DEPTH
        self.min_level = min_level
        self._set_static_inputs(haze_color, sun_direction, sun_color, ambient, exposure)

        self.executor = ChunkExecutor(backend=backend, workers=workers)
        self.planet: Planet | None = None
        self.query: TerrainQuery | None = None
        self.selector: LodSelector | None = None
        self.store: OrderedDict[tuple, ChunkData] = OrderedDict()
        self.nodes: OrderedDict[tuple, NodePath] = OrderedDict()
        self.shown: set[tuple] = set()
        self._keep: set[tuple] = set()  # drawn chunks and their ancestors
        self._origin = None
        self._snap = ORIGIN_SNAP
        self._camera = np.zeros(3)
        self._render_camera = np.zeros(3)
        self._last_select_pos = None
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
        }
        self.set_spec(spec)

    # ── public API ─────────────────────────────────────────────────────────────

    def set_spec(self, spec: PlanetSpec) -> None:
        """Switch to another planet (or seed); regenerates everything, coarse first.

        :param spec: the new planet
        """
        version = self.executor.set_spec(spec)
        self.version = version
        self.planet = Planet(spec)
        self.spec = self.planet.spec
        self.grid = self.planet.grid
        self.root.set_shader_input("uMaxLevel", self.grid.max_level)
        for node in self.nodes.values():
            node.remove_node()
        self.nodes.clear()
        self.store.clear()
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

    def update(self, camera_planet, camera_render=(0.0, 0.0, 0.0)) -> None:
        """Per-frame update.

        :param camera_planet: camera position in the planet frame (centre at
            the origin), metres, float64
        :param camera_render: where the camera is in the render frame (the
            game keeps it at the origin)
        """
        self._frame += 1
        cam = np.asarray(camera_planet, dtype=np.float64)
        render_cam = np.asarray(camera_render, dtype=np.float64)
        self._camera = cam.copy()
        self._render_camera = render_cam.copy()
        self._rebase(cam, render_cam)

        finished = self.executor.collect()
        for chunk in finished:
            self._store(chunk)
        if finished:
            gen = [c.gen_ms for c in finished]
            self.stats["gen_ms"] = 0.9 * self.stats["gen_ms"] + 0.1 * float(
                np.mean(gen)
            )

        uploaded = self._upload()
        # Re-select after moving an eighth of a leaf, or a 500th of the
        # altitude: chunk sizes on screen change no faster than that.
        threshold = max(self.grid.chunk_size(0) / 8.0, 0.002 * self.altitude(cam))
        moved = (
            self._last_select_pos is None
            or np.linalg.norm(cam - self._last_select_pos) > threshold
        )
        if moved or uploaded or self._frame % 15 == 0:
            self._select(cam)

        rho = float(np.linalg.norm(cam))
        shell = self.spec.radius + self.haze_height
        root = self.root
        root.set_shader_input("uCamPos", LVecBase3f(*render_cam))
        root.set_shader_input("uPlanetCentre", LVecBase3f(*(render_cam - cam)))
        root.set_shader_input("uCamFromCentre", LVecBase3f(*cam))
        # In float64: as a float32 difference of squares it would be noise.
        root.set_shader_input("uCamShellC", float((rho - shell) * (rho + shell)))
        self.stats.update(
            shown=len(self.shown),
            stored=len(self.store),
            uploaded=len(self.nodes),
            pending=self.executor.pending,
            wanted=len(self._wanted),
        )

    @property
    def radius(self) -> float:
        """Sea-level radius, metres."""
        return self.spec.radius

    def altitude(self, camera_planet) -> float:
        """:returns: metres above sea level of a planet-frame position"""
        return float(np.linalg.norm(camera_planet)) - self.spec.radius

    def to_render(self, point_planet) -> np.ndarray:
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
        the swell ocean is not shown), instead of the bare seabed."""
        self.root.set_shader_input("uSeaSurface", 1 if on else 0)

    def set_sun(self, direction, color, ambient, haze_color=None) -> None:
        """Relight the terrain (no rebuild).

        :param direction: toward the sun, planet frame (the render frame's
            axes are the planet's)
        """
        self.root.set_shader_input("uSunDir", Vec3(*direction).normalized())
        self.root.set_shader_input("uSunColor", LVecBase3f(*color))
        self.root.set_shader_input("uAmbient", LVecBase3f(*ambient))
        if haze_color is not None:
            self.root.set_shader_input("uHazeColor", LVecBase3f(*haze_color))

    def set_debug_mode(self, mode: int) -> None:
        """0 shaded, 1 LOD level, 2 biomes, 3 slope, 4 normals, 5 skirts."""
        self.root.set_shader_input("uDebugMode", int(mode))

    def set_textured(self, textured: bool) -> None:
        """Toggle textures (off: flat biome colours)."""
        self.root.set_shader_input("uTextured", 1 if textured else 0)

    def set_blend_depth(self, depth: float) -> None:
        """Biome transition softness: ~0.1 crisp height-cut edges, >= 1 smooth."""
        self.blend_depth = depth
        self.root.set_shader_input("uBlendDepth", float(depth))

    def set_wireframe(self, on: bool) -> None:
        if on:
            self.root.set_render_mode_wireframe()
        else:
            self.root.clear_render_mode()

    def set_reflected(self, reflected: bool) -> None:
        """Whether the ocean's reflection camera draws the terrain (doubles cost)."""
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
        """Stop the workers and remove the geometry."""
        self.executor.shutdown()
        self.root.remove_node()

    # ── internals ──────────────────────────────────────────────────────────────

    def _set_static_inputs(
        self, haze_color, sun_direction, sun_color, ambient, exposure
    ):
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
        root.set_shader_input("uHazeColor", LVecBase3f(*haze_color))
        root.set_shader_input("uHazeDistance", float(self.haze_distance))
        root.set_shader_input("uDebugMode", 0)
        root.set_shader_input("uLevel", 0)
        root.set_shader_input("uMaxLevel", 0)
        root.set_shader_input("uFace", 0)
        root.set_shader_input("uChunk", LVecBase4f(0.0, 0.0, 1.0, 0.0))
        root.set_shader_input("uCamPos", LVecBase3f(0.0, 0.0, 0.0))
        root.set_shader_input("uPlanetCentre", LVecBase3f(0.0, 0.0, -1.0))
        root.set_shader_input("uCamFromCentre", LVecBase3f(0.0, 0.0, 1.0))
        root.set_shader_input("uCamShellC", -1.0)
        root.set_shader_input("uSeaSurface", 0)
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
        self.store.move_to_end(chunk.key)
        excess = len(self.store) - STORE_SIZE
        for old in list(self.store):
            if excess <= 0:
                break
            if old in self._keep:
                continue
            del self.store[old]
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
        excess = len(self.nodes) - NODE_CACHE_SIZE
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

        def bounds(key):
            chunk = self.store.get(key)
            return None if chunk is None else (chunk.h_min, chunk.h_max)

        selection = self.selector.select(
            cam,
            ready=self.nodes.__contains__,
            bounds=bounds,
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
        self.executor.request([k for k in self._wanted if k not in self.store])
        self.stats["select_ms"] = (time.perf_counter() - start) * 1e3
        self.stats["finest"] = min((k[0] for k in draw), default=self.grid.max_level)
