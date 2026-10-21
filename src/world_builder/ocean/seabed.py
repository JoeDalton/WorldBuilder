"""
The seabed under the ocean: sea-floor heights baked into textures around the
camera, for the water's colour by depth, its shoreline foam and its swell.

The ocean knows nothing of the terrain. An app hands it a *sampler*: a function
from points of the ocean's plane, in the app's own world, to the sea floor's
height under them. :class:`SeabedMap` calls it on a worker thread over a few
square cascades centred on the camera (fine and small to coarse and wide),
and uploads each as a float texture.

Where the plane sits in that world (:meth:`SeabedMap.set_frame`) may change
every frame: a camera-following tangent plane on a round planet turns as the
camera flies. Each cascade remembers the frame it was baked in, and every
frame :meth:`SeabedMap.update` maps the current plane's xy into each cascade's
texture coordinates, so a bake stays put on the ground however the frame moves
until the camera leaves its middle and it is baked again.

Outside every cascade, or with no sampler, the sea is deep: the water looks as
it does without a seabed.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from panda3d.core import (
    LVecBase3f,
    NodePath,
    PTA_float,
    PTA_LVecBase3f,
    SamplerState,
    Texture,
)

#: Seabed sampler: (N, 3) float64 points of the ocean's plane in the app's
#: world, and the texel size in metres (a resolution hint), to (N,) heights of
#: the sea floor in metres, negative under water.
Sampler = Callable[[np.ndarray, float], np.ndarray]

#: Cascades the shaders can read (seabed.glsl's uSeabedTex0..2).
MAX_CASCADES = 3
#: Height reported where no cascade covers the water: deep sea.
DEEP_SEA_M = -10000.0
#: A cascade is baked again once the camera is this fraction of its extent
#: from the cascade's centre.
REBAKE_FRACTION = 0.25
#: And refreshed this often anyway, so the terrain's later refinement shows.
MAX_AGE_S = 5.0
#: But not more often than this: the bakes' budget. A fast camera outruns
#: its finest cascade for a moment, and the next coarser covers meanwhile.
MIN_INTERVAL_S = 0.5


@dataclass(frozen=True)
class Cascade:
    """One square around the camera.

    :ivar extent: its side, metres
    :ivar resolution: texels per side
    """

    extent: float
    resolution: int

    @property
    def texel(self) -> float:
        """:returns: the texel size, metres"""
        return self.extent / self.resolution


@dataclass
class _Bake:
    """A cascade as baked: its frame, and its square's corner in it."""

    origin: np.ndarray  # the plane's origin in the world, at bake time
    east: np.ndarray  # its x axis
    north: np.ndarray  # its y axis
    x0: float  # the square's corner, in that frame's xy
    y0: float
    heights: np.ndarray | None = None  # (res, res) float32, rows by y
    when: float = 0.0


class SeabedMap:
    """Seabed textures around the camera, baked from an app's sampler.

    Shader inputs it sets (read by seabed.glsl): ``uSeabedCount``,
    ``uSeabedTex0..2``, ``uSeabedTexel[3]`` (metres), ``uSeabedU[3]`` and
    ``uSeabedV[3]`` (the current plane's (x, y, 1) to each cascade's texture
    u and v). The mappings change
    every frame, so they are arrays bound once and written in place (a new
    value per frame would make a new render state per frame); the count and
    the textures only change when a bake lands.
    """

    def __init__(self, node: NodePath, cascades: Sequence[Cascade]):
        """
        :param node: where the shader inputs go (the ocean's surface)
        :param cascades: finest first, at most :data:`MAX_CASCADES`
        """
        self.node = node
        self.sampler: Sampler | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seabed")
        self._pending: dict[int, Future] = {}
        self._origin = np.zeros(3)
        self._east = np.array([1.0, 0.0, 0.0])
        self._north = np.array([0.0, 1.0, 0.0])
        self._texel = PTA_float.empty_array(MAX_CASCADES)
        node.set_shader_input("uSeabedTexel", self._texel)
        self._u = PTA_LVecBase3f.empty_array(MAX_CASCADES)
        self._v = PTA_LVecBase3f.empty_array(MAX_CASCADES)
        node.set_shader_input("uSeabedU", self._u)
        node.set_shader_input("uSeabedV", self._v)
        self._count = -1
        self._textures = [Texture(f"seabed{k}") for k in range(MAX_CASCADES)]
        for tex in self._textures:
            tex.set_wrap_u(SamplerState.WM_clamp)
            tex.set_wrap_v(SamplerState.WM_clamp)
            tex.set_minfilter(SamplerState.FT_linear)
            tex.set_magfilter(SamplerState.FT_linear)
            self._fill(tex, np.full((1, 1), DEEP_SEA_M, np.float32))
            node.set_shader_input(tex.get_name().replace("seabed", "uSeabedTex"), tex)
        self.set_cascades(cascades)

    # ── public ─────────────────────────────────────────────────────────────────

    def set_sampler(self, sampler: Sampler | None) -> None:
        """Start (or stop, with None) reading the seabed; drops the bakes."""
        self.sampler = sampler
        self._reset()

    def set_cascades(self, cascades: Sequence[Cascade]) -> None:
        """Change the cascades (a quality setting); drops the bakes."""
        if len(cascades) > MAX_CASCADES:
            raise ValueError(f"at most {MAX_CASCADES} seabed cascades")
        self.cascades = tuple(cascades)
        for k, cascade in enumerate(self.cascades):
            self._texel[k] = cascade.texel
        self._reset()

    def set_frame(self, origin, east, north) -> None:
        """Where the ocean's plane is in the sampler's world (float64).

        :param origin: the plane's (0, 0)
        :param east: its unit x axis
        :param north: its unit y axis
        """
        self._origin = np.asarray(origin, dtype=np.float64)
        self._east = np.asarray(east, dtype=np.float64)
        self._north = np.asarray(north, dtype=np.float64)

    def update(self, camera_xy: tuple[float, float]) -> None:
        """Per frame: take finished bakes, start the ones due, and map the
        current plane into each cascade.

        :param camera_xy: the camera over the plane, in its xy
        """
        for k, future in list(self._pending.items()):
            if future.done():
                del self._pending[k]
                bake = future.result()
                if bake is not None and k < len(self._bakes):
                    self._bakes[k] = bake
                    self._fill(self._textures[k], bake.heights)
        if self.sampler is not None:
            self._schedule(camera_xy)
        count = 0
        for k, bake in enumerate(self._bakes):
            if bake is None or bake.heights is None:
                break  # the shader reads cascades 0 .. count - 1, finest first
            self._u[k], self._v[k] = self._mapping(bake, self.cascades[k].extent)
            count += 1
        if count != self._count:
            self._count = count
            self.node.set_shader_input("uSeabedCount", count)

    def heights_at(self, xy: np.ndarray) -> np.ndarray:
        """The baked seabed under points of the current plane, as the shader
        reads it (finest covering cascade, nearest texel; deep elsewhere).
        For tests and debugging.

        :param xy: (N, 2) the plane's xy
        :returns: (N,) heights, metres
        """
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        out = np.full(len(xy), DEEP_SEA_M)
        todo = np.ones(len(xy), bool)
        for k, bake in enumerate(self._bakes):
            if bake is None or bake.heights is None:
                break
            u, v = self._mapping(bake, self.cascades[k].extent)
            ones = np.ones(len(xy))
            uu = np.column_stack([xy, ones]) @ np.array(tuple(u))
            vv = np.column_stack([xy, ones]) @ np.array(tuple(v))
            inside = todo & (uu >= 0) & (uu < 1) & (vv >= 0) & (vv < 1)
            res = bake.heights.shape[0]
            i = np.clip((vv * res).astype(int), 0, res - 1)
            j = np.clip((uu * res).astype(int), 0, res - 1)
            out[inside] = bake.heights[i[inside], j[inside]]
            todo &= ~inside
        return out

    def destroy(self) -> None:
        """Stop the worker (pending bakes are dropped)."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ── internals ──────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        for future in self._pending.values():
            future.cancel()
        self._pending = {}
        self._bakes: list[_Bake | None] = [None] * len(self.cascades)

    def _plane_point(self, x: float, y: float) -> np.ndarray:
        return self._origin + x * self._east + y * self._north

    def _schedule(self, camera_xy: tuple[float, float]) -> None:
        """Start the most urgent due bake, if the worker is free: one at a
        time, finest first, so a fast camera keeps its near cascade fresh."""
        if self._pending:
            return
        now = time.monotonic()
        camera = self._plane_point(*camera_xy)
        for k, cascade in enumerate(self.cascades):
            bake = self._bakes[k]
            if bake is not None and now - bake.when < MIN_INTERVAL_S:
                continue
            due = bake is None or now - bake.when > MAX_AGE_S
            if not due:
                d = camera - bake.origin
                cx = d @ bake.east - (bake.x0 + cascade.extent / 2.0)
                cy = d @ bake.north - (bake.y0 + cascade.extent / 2.0)
                due = max(abs(cx), abs(cy)) > REBAKE_FRACTION * cascade.extent
            if due:
                # The corner on the texel lattice of the plane's xy: while the
                # plane stays put on the ground (as the demo's chart keeps
                # it), a new bake samples the same ground points, and the
                # shoreline does not shift when it lands.
                t = cascade.texel
                x0 = math.floor((camera_xy[0] - cascade.extent / 2.0) / t) * t
                y0 = math.floor((camera_xy[1] - cascade.extent / 2.0) / t) * t
                job = _Bake(
                    self._origin.copy(), self._east.copy(), self._north.copy(), x0, y0
                )
                self._pending[k] = self._executor.submit(
                    _bake, self.sampler, job, cascade
                )
                return

    def _mapping(self, bake: _Bake, extent: float) -> tuple[LVecBase3f, LVecBase3f]:
        """The current plane's (x, y, 1) to the bake's texture (u, v): its
        point in the world, then in the bake's frame (float64, so far frames
        keep their precision)."""
        d = self._origin - bake.origin
        rows = []
        for axis, corner in ((bake.east, bake.x0), (bake.north, bake.y0)):
            rows.append(
                LVecBase3f(
                    float(self._east @ axis) / extent,
                    float(self._north @ axis) / extent,
                    float((d @ axis - corner) / extent),
                )
            )
        return rows[0], rows[1]

    @staticmethod
    def _fill(tex: Texture, heights: np.ndarray) -> None:
        res_y, res_x = heights.shape
        if tex.get_x_size() != res_x or tex.get_y_size() != res_y:
            tex.setup_2d_texture(res_x, res_y, Texture.T_float, Texture.F_r32)
        tex.set_ram_image(np.ascontiguousarray(heights, dtype=np.float32).tobytes())


def _bake(sampler: Sampler, job: _Bake, cascade: Cascade) -> _Bake | None:
    """Worker thread: sample the seabed at the cascade's texel centres."""
    res, t = cascade.resolution, cascade.texel
    centres = (np.arange(res) + 0.5) * t
    gx, gy = np.meshgrid(job.x0 + centres, job.y0 + centres)
    points = job.origin + gx.reshape(-1, 1) * job.east + gy.reshape(-1, 1) * job.north
    try:
        heights = np.asarray(sampler(points, t), dtype=np.float32).reshape(res, res)
    except Exception as error:  # a sampler's failure must not stop the ocean
        print(f"seabed bake failed: {error!r}")
        return None
    job.heights = heights
    job.when = time.monotonic()
    return job
