"""
The trees, bushes and rocks drawn: the plants :mod:`world_builder.terrain.forest`
placed on each chunk, as the tree assets of :mod:`world_builder.datafiles`.

Two looks, by each plant's distance from the camera:

- **Meshes** (within ``VegetationSettings.mesh_range``): each asset's model,
  one instanced draw per asset for every plant near the camera (their bases,
  turns, scales and tints in a buffer texture: ``tree_mesh.vert``), rebuilt
  after the LOD passes that change the chunks near the camera.
- **Impostors** (out to ``tree_range``): one point per plant (a sprite as
  wide as its asset's bounding sphere), each chunk's in a node of its own
  under the chunk's (culled with it), drawn from the asset's baked views
  (``scripts/bake_tree_impostors.py``: the one nearest the view's direction,
  lit by its baked normals: ``tree_impostor.frag``). Rocks and bushes have
  none: past the meshes they are the ground's.

The two cross over a dithered band, plant by plant. Past ``tree_range`` the
far forest (``vegetation_far.glsl``) stands for them, faded in over the
impostors' last band.
"""

from __future__ import annotations

import functools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from panda3d.core import (
    BitMask32,
    BoundingBox,
    CullBinManager,
    Filename,
    Geom,
    GeomEnums,
    GeomNode,
    GeomPoints,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    InternalName,
    LVecBase4f,
    NodePath,
    OmniBoundingVolume,
    Point3,
    PTA_float,
    PTA_LVecBase4f,
    SamplerState,
    Shader,
    ShaderAttrib,
    TexGenAttrib,
    Texture,
    TexturePool,
    TextureStage,
)

from world_builder.datafiles import TREES_PATH, tree_assets
from world_builder.terrain import forest

SHADER_PATH = Path(__file__).parent / "shaders"
IMPOSTORS_PATH = TREES_PATH / "impostors"
#: The rows (texels) of one plant in the meshes' buffer texture.
TEXELS = 2
#: The chunks within this much more than the meshes' range give theirs.
MESH_MARGIN = 60.0
#: The trees' cull bin: front to back, before the ground's (19).
TREE_BIN = "trees"
TREE_BIN_SORT = 18


@functools.cache
def _shader(name: str) -> Shader:
    return Shader.load(
        Shader.SL_GLSL,
        vertex=SHADER_PATH / f"{name}.vert",
        fragment=SHADER_PATH / f"{name}.frag",
    )


@functools.cache
def point_format() -> GeomVertexFormat:
    """One impostor: its base (from its chunk's centre) and its plant
    (turn, scale, asset, tint)."""
    array = GeomVertexArrayFormat()
    array.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    array.add_column(InternalName.make("plant"), 4, Geom.NT_float32, Geom.C_other)
    return GeomVertexFormat.register_format(GeomVertexFormat(array))


def _main_cameras() -> BitMask32:
    from world_builder.depth import (
        CASTER_BIT,
        HIDE_FROM_CLOUD_PASS,
        REFLECTION_CAMERA_MASK,
    )

    mask = BitMask32.all_on()
    mask &= ~REFLECTION_CAMERA_MASK
    mask &= ~HIDE_FROM_CLOUD_PASS
    mask &= ~BitMask32.bit(CASTER_BIT)
    return mask


class Trees:
    """One terrain's trees, bushes and rocks (``terrain.trees``): made by
    its :class:`~world_builder.terrain.terrain.Terrain`, which calls
    :meth:`update` after its LOD passes and :meth:`drop` for a chunk it lets
    go. Its settings are the vegetation's (``terrain.vegetation.settings``)."""

    def __init__(self, terrain: Any):
        self._terrain = terrain
        manager = CullBinManager.get_global_ptr()
        if manager.find_bin(TREE_BIN) == -1:
            manager.add_bin(TREE_BIN, CullBinManager.BT_front_to_back, TREE_BIN_SORT)
        self._root = terrain.root.attach_new_node("trees")
        self._root.hide(BitMask32.all_on())
        self._root.show(_main_cameras())
        self._pixel_scale = PTA_float.empty_array(1)
        self.set_pixel_angle(math.radians(70.0) / 1440.0)
        self._meshes: list[NodePath | None] = []
        self._buffers: list[Texture | None] = []
        self._points: dict[tuple, NodePath] = {}
        self._counts: dict[tuple, int] = {}
        self.stats = {
            "meshes": 0,
            "impostors": 0,
            "chunks": 0,
            "fill_ms": 0.0,
            "points_ms": 0.0,
            "made": 0,
        }
        self._load_assets()
        self._load_impostors()
        self.apply_settings()

    # ── settings ───────────────────────────────────────────────────────────

    @property
    def settings(self):
        return self._terrain.vegetation.settings

    @property
    def active(self) -> bool:
        return self._terrain.vegetation.active and self.settings.tree_range > 0.0

    def apply_settings(self) -> None:
        s = self.settings
        # On the terrain's root: the impostors hang under its chunks, and the
        # far forest fades in over their band.
        root = self._terrain.root
        on = self.active
        root.set_shader_input("uMeshRange", float(s.mesh_range) if on else 0.0)
        root.set_shader_input("uTreeRange", float(s.tree_range) if on else 0.0)
        root.set_shader_input("uPlanetRadius", float(self._terrain.radius))
        if not on:
            self.clear()
        else:
            # The impostors' nodes again at the next pass (their range).
            for node in self._points.values():
                node.remove_node()
            self._points.clear()
            self._terrain._last_select_pos = None

    def set_pixel_angle(self, radians: float) -> None:
        """The angle one pixel of the view spans (vertical fov / its
        height): the impostors' size on screen."""
        self._pixel_scale[0] = 1.0 / max(radians, 1e-9)

    # ── assets ─────────────────────────────────────────────────────────────

    def _load_assets(self) -> None:
        by_name = {a.name: a for a in tree_assets()}
        shader = _shader("tree_mesh")
        for name in forest.ASSETS:
            model = by_name[name].load()
            model.flatten_strong()
            node = NodePath(f"trees_{name}")
            model.reparent_to(node)
            node.node().set_bounds(OmniBoundingVolume())
            node.node().set_final(True)
            node.set_shader(shader, 20)
            node.set_bin(TREE_BIN, 0, 1)
            node.set_two_sided(True)
            buffer = Texture(f"trees_{name}")
            buffer.setup_buffer_texture(
                TEXELS, Texture.T_float, Texture.F_rgba32, GeomEnums.UH_dynamic
            )
            node.set_shader_input("uPlants", buffer)
            node.set_instance_count(0)
            self._meshes.append(node)
            self._buffers.append(buffer)

    def _load_impostors(self) -> None:
        meta = json.loads((IMPOSTORS_PATH / "impostors.json").read_text())
        self._frames = int(meta["frames"])
        layers = {e["name"]: e for e in meta["assets"]}
        table = PTA_LVecBase4f.empty_array(len(forest.ASSETS))
        for k, name in enumerate(forest.ASSETS):
            e = layers.get(name)
            table[k] = (
                LVecBase4f(e["layer"], e["diameter"], e["centre"][2], 1.0)
                if e is not None
                else LVecBase4f(-1.0, 0.0, 0.0, 0.0)
            )
        self._has_impostor = np.array(
            [name in layers for name in forest.ASSETS], dtype=bool
        )
        arrays = []
        for kind in ("albedo", "normal"):
            tex = TexturePool.load_2d_texture_array(
                Filename.from_os_specific(str(IMPOSTORS_PATH / f"{kind}_##.png"))
            )
            tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
            tex.set_magfilter(SamplerState.FT_linear)
            tex.set_wrap_u(SamplerState.WM_clamp)
            tex.set_wrap_v(SamplerState.WM_clamp)
            arrays.append(tex)
        attrib = ShaderAttrib.make(_shader("tree_impostor"))
        attrib = attrib.set_flag(ShaderAttrib.F_shader_point_size, True)
        self._point_state = attrib
        root = self._terrain.root
        root.set_shader_input("uImpostorAlbedo", arrays[0])
        root.set_shader_input("uImpostorNormal", arrays[1])
        root.set_shader_input("uImpostors", table)
        root.set_shader_input("uFrames", self._frames)
        root.set_shader_input("uPixelScale", self._pixel_scale)

    # ── per LOD pass ───────────────────────────────────────────────────────

    def update(self, draw: set, camera: np.ndarray, origin: np.ndarray) -> None:
        """The plants of the drawn chunks: the impostors' nodes (made for
        chunks new in range, dropped with their chunks), and the meshes'
        instances (the plants within their range, refilled).

        :param draw: the drawn chunk keys
        :param camera: the camera, planet frame
        :param origin: the terrain root's origin (planet frame): the meshes'
            bases are from it
        """
        if not self.active:
            return
        start = time.perf_counter()
        made = 0
        terrain = self._terrain
        s = self.settings
        near: list[tuple] = []
        for key in draw:
            chunk = terrain.store.get(key)
            plants = getattr(chunk, "forest", None)
            if plants is None or key not in terrain.nodes:
                continue
            low = chunk.centre + chunk.aabb_min
            high = chunk.centre + chunk.aabb_max
            gap = np.maximum(np.maximum(low - camera, camera - high), 0.0)
            distance = float(np.sqrt(gap @ gap))
            if distance < s.mesh_range + MESH_MARGIN:
                near.append(key)
            if distance < s.tree_range and key not in self._points:
                node = self._point_node(key, chunk, plants)
                made += 1
                if node is not None:
                    self._points[key] = node
        for key in [k for k in self._points if k not in draw]:
            self._points.pop(key).remove_node()
        # Every pass: the plants within reach follow the camera (a pass runs
        # every ~30 m of travel near the ground, and when the origin moves).
        mid = time.perf_counter()
        self._fill_meshes(near, origin, camera)
        self.stats["fill_ms"] = (time.perf_counter() - mid) * 1e3
        self.stats["points_ms"] = (mid - start) * 1e3
        self.stats["made"] = made
        self.stats["chunks"] = len(self._points)
        self.stats["impostors"] = sum(self._counts.get(k, 0) for k in self._points)

    def drop(self, key: tuple) -> None:
        node = self._points.pop(key, None)
        if node is not None:
            node.remove_node()
        self._counts.pop(key, None)

    def clear(self) -> None:
        for node in self._points.values():
            node.remove_node()
        self._points.clear()
        self._counts.clear()
        for node in self._meshes:
            if node is not None:
                node.set_instance_count(0)
                node.detach_node()
        self.stats.update(meshes=0, impostors=0, chunks=0)

    # ── internals ──────────────────────────────────────────────────────────

    def _fill_meshes(
        self, near: list[tuple], origin: np.ndarray, camera: np.ndarray
    ) -> None:
        """Every plant within the meshes' range (and MESH_MARGIN: the camera
        moves until the next pass) into its asset's buffer: (base from the
        origin, turn), (scale, tint, 0, 0)."""
        terrain = self._terrain
        reach = self.settings.mesh_range + MESH_MARGIN
        rows = []
        for key in near:
            chunk = terrain.store[key]
            plants = chunk.forest
            base = plants[:, :3].astype(np.float64) + chunk.centre
            d = base - camera
            mine = (d * d).sum(axis=1) < reach * reach
            if mine.any():
                rows.append(
                    np.concatenate([base[mine] - origin, plants[mine, 3:]], axis=1)
                )
        all_rows = np.concatenate(rows) if rows else np.zeros((0, 7))
        assets = all_rows[:, 5].astype(np.int64)
        count = 0
        for k, node in enumerate(self._meshes):
            mine = all_rows[assets == k]
            n = len(mine)
            buffer = self._buffers[k]
            if n == 0:
                node.set_instance_count(0)
                node.detach_node()
                continue
            data = np.zeros((n, TEXELS, 4), dtype=np.float32)
            data[:, 0, :3] = mine[:, :3]
            data[:, 0, 3] = mine[:, 3]
            data[:, 1, 0] = mine[:, 4]
            data[:, 1, 1] = mine[:, 6]
            buffer.setup_buffer_texture(
                n * TEXELS, Texture.T_float, Texture.F_rgba32, GeomEnums.UH_dynamic
            )
            # A buffer texture's RAM image goes to the GPU as it is (RGBA).
            buffer.set_ram_image(data.tobytes())
            node.set_instance_count(n)
            if node.get_parent().is_empty():
                node.reparent_to(self._root)
            count += n
        self.stats["meshes"] = count

    def _point_node(
        self, key: tuple, chunk: Any, plants: np.ndarray
    ) -> NodePath | None:
        mine = plants[self._has_impostor[plants[:, 5].astype(np.int64)]]
        self._counts[key] = len(mine)
        if not len(mine):
            return None
        vdata = GeomVertexData("impostors", point_format(), Geom.UH_static)
        vdata.unclean_set_num_rows(len(mine))
        packed = np.ascontiguousarray(mine[:, [0, 1, 2, 3, 4, 5, 6]], dtype=np.float32)
        memoryview(vdata.modify_array(0)).cast("B")[:] = packed.tobytes()
        points = GeomPoints(Geom.UH_static)
        points.add_next_vertices(len(mine))
        geom = Geom(vdata)
        geom.add_primitive(points)
        lo, hi = mine[:, :3].min(axis=0), mine[:, :3].max(axis=0)
        bounds = BoundingBox(Point3(*(lo - 30.0)), Point3(*(hi + 40.0)))
        geom.set_bounds(bounds)
        node = GeomNode("impostors")
        node.add_geom(geom)
        node.set_bounds(bounds)
        node.set_final(True)
        path = self._terrain.nodes[key].attach_new_node(node)
        path.hide(BitMask32.all_on())
        path.show(_main_cameras())
        path.set_attrib(self._point_state, 20)  # the root's inputs
        # Front to back, before the ground: the ground they hide is not shaded.
        path.set_bin(TREE_BIN, 0, 1)
        # gl_PointCoord (a compatibility context needs point sprites on).
        path.set_tex_gen(TextureStage.get_default(), TexGenAttrib.M_point_sprite)
        return path
