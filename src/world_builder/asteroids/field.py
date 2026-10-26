"""
An asteroid field drawn in Panda: every rock around the camera, by hardware
instancing.

- **Prototypes** are built by worker processes (:mod:`prototype`, on a
  ``ChunkExecutor`` channel): each level of detail of each prototype is one
  GeomNode, drawn once per rock that shows it. ``loading`` stays True until
  they have all arrived.
- **Rocks** (:mod:`population`) hold slots in the instance table, a float
  texture of four texels per slot (``asteroid.vert``), written when they come
  into range and uploaded at most every few frames. A rock is drawn only once
  its row is on the GPU.
- **Per frame**, the visible pass picks the rocks drawn and their level of
  detail; their slots go into the index texture, grouped per draw, and each
  draw gets its count (``set_instance_count``) and offset. Nothing per rock
  goes through the scene graph.
- **Spin** is computed in the vertex shader at the game's time (``clock``,
  never Panda's): the table holds each rock's orientation at an epoch, moved
  every :data:`EPOCH_S` seconds so float32 time stays precise.
- **Frames.** The field sits at ``position`` with ``orientation`` (world from
  field) in the world. It hangs under render and is placed relative to the
  camera, as the terrain is: render's axes are the world's.
- **Light.** The terrain's materials (its rock and snow layers) and lighting
  (``object_light.glsl``), then the world's tonemap, or a plain display
  encoding next to models lit by Panda's own lights (``tonemap=False``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    BitMask32,
    GeomNode,
    LMatrix4f,
    LVecBase3f,
    NodePath,
    OmniBoundingVolume,
    PTA_float,
    PTA_int,
    PTA_LVecBase3f,
    SamplerState,
    Shader,
    Texture,
)

from world_builder.asteroids.mesh import build_geom, normal_map_texture
from world_builder.asteroids.motion import rotate
from world_builder.asteroids.population import SLOT_FLOATS, Population
from world_builder.asteroids.prototype import PrototypeJob, generate_prototype
from world_builder.asteroids.scatter import Scatter
from world_builder.asteroids.settings import AsteroidSettings
from world_builder.asteroids.spec import FieldSpec, KeepOut
from world_builder.depth import REFLECTION_CAMERA_MASK
from world_builder.terrain.materials import (
    MaterialSet,
    load_materials,
    make_texture_array,
)
from world_builder.terrain.workers import ChunkExecutor

SHADER_PATH = Path(__file__).parent / "shaders"
#: Texels per row of the instance and index textures (asteroid.vert).
TABLE_WIDTH = 4096
#: Size classes the shader holds (its uniform arrays).
MAX_CLASSES = 16
#: The epoch of the table's orientations moves every this many seconds.
EPOCH_S = 1024.0
#: Frames between uploads of the instance table, at most (new rocks are at
#: the edge of their drawing distance: a few frames late is unseen).
UPLOAD_EVERY = 3
#: Without a camera (headless), the pixel scale ranges are set for.
HEADLESS_PIXELS_PER_RADIAN = 900.0 / math.radians(40.0)
#: Past the budget, the smallest size drawn grows by this factor per frame;
#: well under it, it shrinks back.
BUDGET_STEP = 1.1

_LODS = {16: (4, 8, 16), 32: (4, 8, 16, 32), 64: (4, 8, 16, 32, 64)}


def _rotation(quat: ArrayLike) -> np.ndarray:
    """:returns: the 3x3 matrix of a unit quaternion (w, x, y, z)"""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return np.stack([rotate(q, axis) for axis in np.eye(3)], axis=1)


class AsteroidField:
    """An asteroid field, drawn around the camera.

    :ivar spec: the field
    :ivar scatter: its rocks
    :ivar population: the rocks in range, in slots
    :ivar root: the node everything hangs under
    """

    SETTINGS_TYPE: ClassVar = AsteroidSettings

    def __init__(
        self,
        app: Any,
        spec: FieldSpec,
        clock: Callable[[], float],
        settings: AsteroidSettings | None = None,
        keep_outs: Sequence[KeepOut] = (),
        executor: ChunkExecutor | None = None,
        backend: str = "process",
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        sun_direction: ArrayLike = (0.4, 0.5, 0.6),
        sun_color: ArrayLike = (2.4, 2.25, 2.0),
        ambient: ArrayLike = (0.03, 0.03, 0.035),
        exposure: float = 0.9,
        tonemap: bool = True,
        materials: MaterialSet | None = None,
        parent: NodePath | None = None,
        far: float = 600_000.0,
        name: str = "asteroids",
    ):
        """
        :param app: the ShowBase (``win``, ``cam`` for the default view)
        :param spec: the field
        :param clock: the game's time, seconds: the rocks turn with it
        :param settings: what it costs; None = HIGH
        :param keep_outs: room no rock enters, added to the spec's
        :param executor: a generation pool shared with others (this field
            takes its own channel); None = its own
        :param backend: its own pool's backend (see :class:`ChunkExecutor`)
        :param position: the field's centre in the world, float64 metres
        :param orientation: world from field, unit quaternion (w, x, y, z)
        :param sun_direction: toward the sun, world frame
        :param sun_color: direct light, linear
        :param ambient: sky light, linear
        :param exposure: pre-tonemap exposure
        :param tonemap: the world's ACES tonemap (True), or a plain display
            encoding, to match models lit by Panda's fixed-function lights
        :param materials: the terrain's textures (rock and snow are used);
            None = loaded at ``settings.texture_size``
        :param parent: what to hang under (default render; not a node offset
            by large distances)
        :param far: rocks beyond this are never drawn (the camera's far)
        :param name: the root's name, and the executor channel's
        """
        self.app = app
        self.spec = spec
        self._clock = clock
        self._settings = settings or AsteroidSettings()
        self.scatter = Scatter(spec, tuple(keep_outs))
        if len(self.scatter.classes) > MAX_CLASSES:
            raise ValueError(f"more than {MAX_CLASSES} size classes")
        self.population = Population(self.scatter)
        self.far = float(far)
        self.position = np.asarray(position, dtype=np.float64)
        self.orientation = np.asarray(orientation, dtype=np.float64)
        self._rotation_matrix = _rotation(self.orientation)
        self._owns_executor = executor is None
        self.executor = executor or ChunkExecutor(backend=backend)
        self._channel = f"{name}:{id(self)}"
        parent = parent if parent is not None else app.render
        self.root = parent.attach_new_node(name)
        self.root.hide(REFLECTION_CAMERA_MASK)
        self.root.set_shader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=SHADER_PATH / "asteroid.vert",
                fragment=SHADER_PATH / "asteroid.frag",
            )
        )
        self._min_pixels = self._settings.min_pixels
        self._epoch = None
        self._ready = np.zeros(self.population.capacity, dtype=bool)
        self._pending: list[np.ndarray] = []
        self._frames_since_upload = UPLOAD_EVERY
        self._prototypes: dict[int, Any] = {}
        self._draws: list[tuple[NodePath, Any, int]] = []  # node, offset PTA, count
        self._errors = None
        self._last_visible = None
        self.collision = None
        self.stats: dict[str, float] = {}
        self._set_inputs(
            sun_direction, sun_color, ambient, exposure, tonemap, materials
        )
        self._make_table(self.population.capacity)
        self._make_index(1024)
        self._request_prototypes()

    # ── public ────────────────────────────────────────────────────────────

    @property
    def settings(self) -> AsteroidSettings:
        return self._settings

    @property
    def loading(self) -> bool:
        """True until every prototype is built (nothing is drawn before)."""
        return self._errors is None

    def set_settings(self, settings: AsteroidSettings) -> None:
        """Apply new settings: the meshes and normal maps are rebuilt if
        their resolution changed (the textures' size only at the next
        start)."""
        changes = self._settings.changes(settings)
        self._settings = settings
        self._min_pixels = settings.min_pixels
        self.root.set_shader_input("uDetailFar", float(settings.detail_far))
        if "max_lod" in changes or "normal_size" in changes:
            self._clear_draws()
            self._request_prototypes()

    def set_sun(
        self, direction: ArrayLike, color: ArrayLike, ambient: ArrayLike
    ) -> None:
        """:param direction: toward the sun, world frame (normalised here)
        :param color: direct light, linear
        :param ambient: sky light, linear"""
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        self._frame["uSunDir"][0] = LVecBase3f(*d)
        self._frame["uSunColor"][0] = LVecBase3f(*color)
        self._frame["uAmbient"][0] = LVecBase3f(*ambient)

    def set_frame(self, position: ArrayLike, orientation: ArrayLike) -> None:
        """Move the field: its centre (world) and orientation (world from
        field, unit quaternion)."""
        self.position = np.asarray(position, dtype=np.float64)
        self.orientation = np.asarray(orientation, dtype=np.float64)
        self._rotation_matrix = _rotation(self.orientation)
        self._set_rotation()

    @property
    def rotation(self) -> np.ndarray:
        """(3, 3) world from field"""
        return self._rotation_matrix

    def to_field(self, point_world: ArrayLike) -> np.ndarray:
        """:returns: a world point in the field frame"""
        r = self.rotation
        return r.T @ (np.asarray(point_world, dtype=np.float64) - self.position)

    def enable_collision(
        self,
        into_mask: BitMask32,
        name: str = "asteroids",
        owner_tag: str = "owner",
        max_distance: float = 20_000.0,
        samples: int = 5,
    ) -> NodePath:
        """Make the rocks solid to a game's CollisionTraverser (over render):
        one into-only CollisionNode beside the root
        (:mod:`world_builder.asteroids.collision`).

        :param into_mask: the node's into-mask
        :param name: the node's name (a game's event patterns use it)
        :param owner_tag: its python tag holding the
            :class:`~world_builder.asteroids.collision.AsteroidSurface`
        :param max_distance: rocks farther than this from the camera are not
            solid
        :param samples: sphere footprint samples per side (odd)
        :returns: the node
        :raises NativeCollisionError: the native library is missing
        """
        from world_builder.asteroids.collision import AsteroidCollision

        self.disable_collision()
        self.collision = AsteroidCollision(
            self, into_mask, name, owner_tag, max_distance, samples
        )
        if self._errors is not None:
            self.collision.set_prototypes(self._prototype_list())
        return self.collision.node

    def disable_collision(self) -> None:
        """The rocks are no longer solid."""
        if self.collision is not None:
            self.collision.destroy()
            self.collision = None

    def update(
        self,
        camera_world: ArrayLike,
        camera_render: ArrayLike = (0.0, 0.0, 0.0),
        cameras: Sequence[NodePath] | None = None,
    ) -> None:
        """Per frame, once the camera is placed.

        :param camera_world: the camera in the world, float64 metres
        :param camera_render: the camera in render (render's axes are the
            world's)
        :param cameras: the views the rocks may be seen in (camera
            NodePaths), the first giving the pixel scale; None = the app's.
            Without any (headless), nothing is drawn; the rocks are still
            held and solid
        """
        start = time.perf_counter()
        self._take_prototypes()
        if cameras is None:
            cameras = [getattr(self.app, "cam", None)]
        cameras = [view for view in cameras if view is not None]
        camera = self.to_field(camera_world)
        if cameras and self.app.win is not None:
            vfov = cameras[0].node().get_lens().get_vfov()
            pixels_per_radian = self.app.win.get_y_size() / math.radians(vfov)
        else:  # headless: rocks held and solid as on a 900-pixel 40-degree view
            pixels_per_radian = HEADLESS_PIXELS_PER_RADIAN
        # Rocks in and out of range.
        ranges = self.population.ranges(pixels_per_radian, self._min_pixels, self.far)
        added, removed = self.population.update(camera, ranges)
        if self.population.capacity > len(self._ready):
            self._make_table(self.population.capacity)
        self._ready[removed] = False
        if len(added):
            self._pending.append(added)
        # The game's time, and the table's epoch.
        now = float(self._clock())
        epoch = math.floor(now / EPOCH_S) * EPOCH_S
        if epoch != self._epoch:
            # Every row's orientation moves to the new epoch, uploaded now:
            # the rocks keep being drawn.
            self._epoch = epoch
            live = np.flatnonzero(
                self.population.live & self._ready[: self.population.capacity]
            )
            self._table_values[live] = self.population.table_rows(live, epoch)
            self._frames_since_upload = UPLOAD_EVERY
            self._pending.append(np.zeros(0, dtype=np.int64))
        self._upload_table()
        frame = self._frame
        frame["uTime"][0] = now - self._epoch
        frame["uPixelsPerRadian"][0] = pixels_per_radian
        frame["uMinPixels"][0] = self._min_pixels
        frame["uCamRender"][0] = LVecBase3f(*camera_render)
        for k, c in enumerate(self.scatter.classes):
            cell = np.floor(camera / c.cell)
            frame["uCamCell"][k] = LVecBase3f(*cell)
            frame["uCamLocal"][k] = LVecBase3f(*(camera - cell * c.cell))
        visible_ms = 0.0
        if self._errors is not None and self.collision is not None:
            self.collision.update(camera, camera_render, now)
        if self._errors is not None and cameras:
            t = time.perf_counter()
            planes = np.array([self._planes(view) for view in cameras])
            visible = self.population.visible(
                camera,
                planes,
                pixels_per_radian,
                self._min_pixels,
                self._settings.lod_error_px,
                self._errors,
                self._ready[: self.population.capacity],
            )
            self._draw(visible)
            self._budget(visible.in_range)
            self._last_visible = visible
            visible_ms = (time.perf_counter() - t) * 1e3
        self.stats = {
            "rocks": self.population.count,
            "drawn": 0 if self._last_visible is None else len(self._last_visible.slots),
            "in_range": 0
            if self._last_visible is None
            else self._last_visible.in_range,
            "min_pixels": self._min_pixels,
            "visible_ms": visible_ms,
            "update_ms": (time.perf_counter() - start) * 1e3,
        }

    def destroy(self) -> None:
        """Remove the field's nodes; shut its own pool down."""
        self.disable_collision()
        self.executor.set_spec(None, self._channel)
        if self._owns_executor:
            self.executor.shutdown()
        self.root.remove_node()

    # ── prototypes ────────────────────────────────────────────────────────

    def _request_prototypes(self) -> None:
        s = self._settings
        job = PrototypeJob(
            self.spec.shapes, self.spec.seed, _LODS[s.max_lod], s.normal_size
        )
        self._prototypes = {}
        self._errors = None
        self.executor.set_spec(job, self._channel, generator=generate_prototype)
        self.executor.request(list(range(self.spec.shapes.prototypes)), self._channel)

    def _take_prototypes(self) -> None:
        if self._errors is not None:
            return
        for prototype in self.executor.collect(self._channel):
            self._prototypes[prototype.key] = prototype
        wanted = self.spec.shapes.prototypes
        if len(self._prototypes) < wanted:
            # Requested keys stay wanted until built: ask again only when
            # nothing is in flight (an inline executor runs a few per call)
            if self.executor.pending_for(self._channel) == 0:
                missing = [k for k in range(wanted) if k not in self._prototypes]
                self.executor.request(missing, self._channel)
            return
        prototypes = self._prototype_list()
        self.root.set_shader_input("uRockNormals", normal_map_texture(prototypes))
        if self.collision is not None:
            self.collision.set_prototypes(prototypes)
        levels = len(prototypes[0].meshes)
        for p in prototypes:
            for level, mesh in enumerate(p.meshes):
                node = GeomNode(f"rock{p.key}_lod{level}")
                node.add_geom(build_geom(mesh))
                node.set_bounds(OmniBoundingVolume())
                node.set_final(True)
                path = self.root.attach_new_node(node)
                offset = PTA_int.empty_array(1)
                path.set_shader_input("uOffset", offset)
                path.stash()
                self._draws.append((path, offset, 0))
        self._errors = np.array(
            [[m.error_to_finest for m in p.meshes] for p in prototypes]
        ).reshape(wanted, levels)

    def _prototype_list(self) -> list:
        return [self._prototypes[k] for k in range(self.spec.shapes.prototypes)]

    def _clear_draws(self) -> None:
        for path, _, _ in self._draws:
            path.remove_node()
        self._draws = []

    # ── per frame ─────────────────────────────────────────────────────────

    def _planes(self, view: NodePath) -> np.ndarray:
        """:returns: (4, 3) the view's side planes' inward normals, field
        frame"""
        lens = view.node().get_lens()
        h, v = (math.radians(0.5 * a) for a in lens.get_fov())
        local = np.array(
            [
                (math.cos(h), math.sin(h), 0.0),
                (-math.cos(h), math.sin(h), 0.0),
                (0.0, math.sin(v), math.cos(v)),
                (0.0, math.sin(v), -math.cos(v)),
            ]
        )
        quat = view.get_quat(self.app.render)
        to_render = _rotation((quat.get_r(), quat.get_i(), quat.get_j(), quat.get_k()))
        to_field = self._rotation_matrix.T
        return local @ (to_field @ to_render).T

    def _draw(self, visible) -> None:
        count = len(visible.slots)
        if count > self._index_capacity:
            self._make_index(max(count, 2 * self._index_capacity))
        if count:
            self._index_values[:count] = visible.slots
            self._index.set_ram_image(self._index_values.tobytes())
        offsets = np.concatenate(([0], np.cumsum(visible.counts)[:-1]))
        for i, (path, offset, shown) in enumerate(self._draws):
            n = int(visible.counts[i])
            offset[0] = int(offsets[i])
            if n == shown:
                continue
            if n == 0:
                path.stash()
            else:
                if shown == 0:
                    path.unstash()
                path.set_instance_count(n)
            self._draws[i] = (path, offset, n)

    def _budget(self, in_range: int) -> None:
        budget = self._settings.instance_budget
        if in_range > budget:
            self._min_pixels *= BUDGET_STEP
        elif in_range < 0.8 * budget and self._min_pixels > self._settings.min_pixels:
            self._min_pixels = max(
                self._settings.min_pixels, self._min_pixels / BUDGET_STEP
            )

    # ── textures ──────────────────────────────────────────────────────────

    def _make_table(self, capacity: int) -> None:
        rows = math.ceil(capacity * 4 / TABLE_WIDTH)
        texture = Texture("asteroid_instances")
        texture.setup_2d_texture(TABLE_WIDTH, rows, Texture.T_float, Texture.F_rgba32)
        _nearest(texture)
        old = getattr(self, "_table_values", None)
        self._table_values = np.zeros(
            (rows * TABLE_WIDTH // 4, SLOT_FLOATS), np.float32
        )
        ready = np.zeros(capacity, dtype=bool)
        if old is not None:
            self._table_values[: len(old)] = old
            ready[: len(self._ready)] = self._ready
        self._ready = ready
        self._table = texture
        self.root.set_shader_input("uInstances", texture)
        self._frames_since_upload = UPLOAD_EVERY  # upload it whole now

    def _upload_table(self) -> None:
        self._frames_since_upload += 1
        if not self._pending or self._frames_since_upload < UPLOAD_EVERY:
            return
        slots = np.concatenate(self._pending)
        slots = slots[self.population.live[slots]]
        self._pending = []
        if len(slots):
            self._table_values[slots] = self.population.table_rows(slots, self._epoch)
        # Panda keeps an RGBA texture's RAM image as BGRA.
        texels = self._table_values.reshape(-1, 4)[:, [2, 1, 0, 3]]
        self._table.set_ram_image(np.ascontiguousarray(texels).tobytes())
        self._ready[slots] = True
        self._frames_since_upload = 0

    def _make_index(self, capacity: int) -> None:
        rows = math.ceil(capacity / TABLE_WIDTH)
        texture = Texture("asteroid_index")
        texture.setup_2d_texture(TABLE_WIDTH, rows, Texture.T_float, Texture.F_r32)
        _nearest(texture)
        self._index = texture
        self._index_capacity = rows * TABLE_WIDTH
        self._index_values = np.zeros(self._index_capacity, dtype=np.float32)
        texture.set_ram_image(self._index_values.tobytes())
        self.root.set_shader_input("uIndex", texture)

    # ── inputs ────────────────────────────────────────────────────────────

    def _set_inputs(
        self, sun_direction, sun_color, ambient, exposure, tonemap, materials
    ):
        root = self.root
        s = self._settings
        m = (
            materials
            if materials is not None
            else load_materials(max_size=s.texture_size)
        )
        # Snow and rock, the two first biome layers.
        root.set_shader_input(
            "uAlbedoH", make_texture_array("rock_albedo_h", m.albedo_height[:2], True)
        )
        root.set_shader_input(
            "uNormRA",
            make_texture_array("rock_normal_ra", m.normal_rough_ao[:2], False),
        )
        tile = PTA_float.empty_array(6)
        mean = PTA_LVecBase3f.empty_array(6)
        for i in range(6):
            tile[i] = float(m.tile_m[min(i, 1)])
            mean[i] = LVecBase3f(*m.mean_albedo[min(i, 1)])
        root.set_shader_input("uTileM", tile)
        root.set_shader_input("uMeanAlbedo", mean)
        root.set_shader_input("uBlendDepth", 0.3)
        root.set_shader_input("uDetailFar", float(s.detail_far))
        root.set_shader_input("uExposure", float(exposure))
        root.set_shader_input("uTonemap", 1 if tonemap else 0)
        cells = PTA_float.empty_array(MAX_CLASSES)
        for k, c in enumerate(self.scatter.classes):
            cells[k] = c.cell
        root.set_shader_input("uCellSize", cells)
        # Inputs that change every frame: arrays bound once, written in
        # place (a new value would make a new render state each frame).
        self._frame = {}
        for name, size in (
            ("uCamRender", 1),
            ("uCamCell", MAX_CLASSES),
            ("uCamLocal", MAX_CLASSES),
            ("uSunDir", 1),
            ("uSunColor", 1),
            ("uAmbient", 1),
        ):
            array = PTA_LVecBase3f.empty_array(size)
            self._frame[name] = array
            root.set_shader_input(name, array)
        for name in ("uTime", "uPixelsPerRadian", "uMinPixels"):
            array = PTA_float.empty_array(1)
            self._frame[name] = array
            root.set_shader_input(name, array)
        self.set_sun(sun_direction, sun_color, ambient)
        self._set_rotation()

    def _set_rotation(self) -> None:
        r = self._rotation_matrix
        # Panda's matrices act on row vectors: its rows are GLSL's columns.
        # A mat3 input is given as a 4x4 (its upper-left 3x3 is taken).
        m = np.eye(4)
        m[:3, :3] = r.T
        self.root.set_shader_input("uFieldRot", LMatrix4f(*m.ravel()))


def _nearest(texture: Texture) -> None:
    texture.set_minfilter(SamplerState.FT_nearest)
    texture.set_magfilter(SamplerState.FT_nearest)
    texture.set_wrap_u(SamplerState.WM_clamp)
    texture.set_wrap_v(SamplerState.WM_clamp)
    texture.set_compression(Texture.CM_off)
    texture.set_keep_ram_image(True)
