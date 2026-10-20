"""
Ocean — camera-locked reflective ocean surface with planar reflections.

The surface is either a single huge flat quad (default; all wave detail is
computed per-pixel in ocean.frag) or, with ``geometric_swell=True``, a dense
grid displaced by the swell in ocean.vert and ringed by flat border cells.
Either mesh is re-centred under the camera every frame.  Reflections come
from an offscreen buffer rendered by a camera mirrored across Z = 0.

Usage:
    from world_builder.ocean import Ocean
    self.ocean = Ocean(game=game)

``Ocean`` registers its own no-argument :meth:`Ocean.update` in
``game.method_lists``, so no manual per-frame call is needed.

It reads from ``game``: ``root_node``, ``method_lists``, ``game_time``
(``get_current_time()``) and ``app`` (the ShowBase: ``camera``, ``camLens``,
``win``, ``graphicsEngine``). Two settings are read when the app has them,
with defaults otherwise: ``app.graphics_settings.config["render"]
["reflection_scale"]`` and ``app.graphics_manager.get_render_size()``.

The ocean has no collision of its own: physics is the game's business. Its
surface is the plane z = 0, exact at the camera, where the droop is zero.
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from panda3d.core import (
    BitMask32,
    Camera,
    ClipPlaneAttrib,
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LMatrix4f,
    LPlane,
    LVecBase2f,
    LVecBase3f,
    LVecBase4f,
    PlaneNode,
    PTA_LVecBase2f,
    Shader,
    Texture,
)

from world_builder import PLANET_RADIUS_M
from world_builder.atmosphere import bind_defaults
from world_builder.utils import compute_next_power_of_2

_SHADERS = Path(__file__).parent / "shaders"
#: Reflection resolution, as a fraction of the render size, when the app has no
#: graphics settings to read it from.
DEFAULT_REFLECTION_SCALE = 0.5

# The surface must reach past the horizon from the highest eye, so its own bulge
# hides its edge: sqrt(2*R*altitude) is ~505 km at 20 km, plus a margin.
_MAX_EYE_ALTITUDE_M = 20000.0
_HORIZON_MARGIN = 1.1

# Camera mask bit used to hide the ocean from the reflection camera
_OCEAN_BIT = BitMask32.bit(1)

# Max wave iterations the shader's iWaveDirs[] array can hold (must match the
# MAX_WAVE_ITER #define in ocean.frag).
_MAX_WAVE_ITER = 64


def compute_wave_dirs(wind_dir: LVecBase2f, wind_strength: float) -> PTA_LVecBase2f:
    """Precompute the per-iteration wave directions the fragment shader uses.

    These depend only on the iteration index and the wind, not on the pixel, so
    we build them once on the CPU and pass them as a uniform array instead of
    recomputing sin/cos/normalize/mix per pixel per iteration in the shader.
    Mirrors the original in-shader formula exactly.

    :param wind_dir: Wind direction in the XY plane (normalised).
    :param wind_strength: How strongly directions align to the wind; 0 =
        random, 1 = fully aligned.
    :return: One normalised wave direction per shader iteration.
    """
    pta = PTA_LVecBase2f()
    it = 0.0
    ws = wind_strength
    for _ in range(_MAX_WAVE_ITER):
        rx, ry = math.sin(it), math.cos(it)
        # mix(rndDir, wind_dir, ws)
        mx = rx * (1.0 - ws) + wind_dir.x * ws
        my = ry * (1.0 - ws) + wind_dir.y * ws
        length = max(math.hypot(mx, my), 1e-5)
        pta.pushBack(LVecBase2f(mx / length, my / length))
        it += 1232.399963
    return pta


def make_plane_mesh(size: float) -> GeomNode:
    """
    Single flat XY quad of the given world size, centred on the origin.

    The ocean surface has no vertical displacement — all wave detail is
    computed per-pixel in the fragment shader from the interpolated world
    position.  Across a flat triangle that interpolation is exact, so a single
    quad is pixel-identical to any tessellation: no subdivision is needed, and
    the vertex carries only its position (no unused normal/texcoord).

    :param size: Edge length of the quad, in world units.
    :return: A :class:`GeomNode` holding the flat quad.
    """
    fmt = GeomVertexFormat.getV3()
    vdata = GeomVertexData("ocean_plane", fmt, Geom.UHStatic)
    vdata.setNumRows(4)

    vw = GeomVertexWriter(vdata, "vertex")
    half = size * 0.5
    for x, y in ((-half, -half), (half, -half), (half, half), (-half, half)):
        vw.addData3f(x, y, 0.0)

    tris = GeomTriangles(Geom.UHStatic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(0, 2, 3)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("ocean_plane")
    node.addGeom(geom)
    return node


def border_coords(grid_half: float, outer_half: float, rings: int) -> np.ndarray:
    """
    Radii of the border rings between the dense grid and the surface's edge.

    Geometrically spaced, so the growth factor is derived rather than chosen: the
    last ring lands exactly on outer_half whatever the ring count.  Geometric
    rather than uniform because what has to be resolved is the planet's CURVE, and
    the error in approximating it by flat cells is ``spacing^2 / (8R)`` — so a ring
    schedule that grows with radius keeps the sag a roughly constant fraction of
    the distance it is seen at, instead of spending vertices out where nothing can
    be resolved anyway.

    :param grid_half: Half-size of the dense inner grid, in world units.
    :param outer_half: Half-size of the whole surface, in world units.
    :param rings: Number of border rings.
    :return: Ring radii, ascending, ending at outer_half.
    """
    growth = (outer_half / grid_half) ** (1.0 / rings)
    return grid_half * growth ** np.arange(1, rings + 1, dtype=np.float64)


def make_swell_grid_mesh(
    grid_half: float,
    subdivs: int,
    outer_half: float,
    curvature: float = 0.0,
    border_rings: int = 28,
    exact_sphere: bool = False,
) -> GeomNode:
    """
    Mesh for the geometric-swell mode: a dense uniform grid of half-size
    grid_half (displaced by the swell in the vertex shader, tapered to zero
    before its edge), ringed by geometrically-spaced border rings out to
    outer_half.  Rings rather than one border quad so the surface can curve;
    28 keeps the curve's sag under 1 arcmin.  Curvature is baked into z, which
    is exact because the mesh is recentred on the camera every frame.

    :param grid_half: Half-size of the dense inner grid, in world units.
    :param subdivs: Number of subdivisions across the dense inner grid.
    :param outer_half: Half-size of the outer border, in world units.
    :param curvature: 1 / (2 * planet radius); 0 for a flat surface.
    :param border_rings: Number of geometrically-spaced border rings.
    :param exact_sphere: Bake the sphere of that radius instead of the
        paraboloid d^2 / 2R. They part by d^4 / 8R^3 (58 m at 100 km on a
        600 km planet), so an ocean meeting real spherical terrain needs it;
        one meeting the other drooped objects (clouds) must keep the
        paraboloid.
    :return: A :class:`GeomNode` holding the grid mesh.
    """
    # Per-axis coordinates: the border rings mirrored below the grid, the uniform
    # inner span, then the border rings above it. Vectorised: the default mesh is
    # ~324k verts.
    border = border_coords(grid_half, outer_half, border_rings)
    inner = -grid_half + 2.0 * grid_half * np.arange(subdivs + 1) / subdivs
    coords = np.concatenate([-border[::-1], inner, border]).astype(np.float32)
    m = coords.size

    # Vertex positions (m*m, 3): vertex j*m + i sits at (coords[i], coords[j]).
    verts = np.zeros((m * m, 3), dtype=np.float32)
    verts[:, 0] = np.tile(coords, m)
    verts[:, 1] = np.repeat(coords, m)
    # Droop from the true radius, so the grid samples a paraboloid of revolution
    # (or the sphere itself).
    r2 = verts[:, 0].astype(np.float64) ** 2 + verts[:, 1].astype(np.float64) ** 2
    if exact_sphere and curvature > 0.0:
        radius = 1.0 / (2.0 * curvature)
        verts[:, 2] = np.sqrt(np.maximum(radius * radius - r2, 0.0)) - radius
    else:
        verts[:, 2] = -curvature * r2

    fmt = GeomVertexFormat.getV3()
    vdata = GeomVertexData("ocean_grid", fmt, Geom.UHStatic)
    vdata.setNumRows(m * m)
    memoryview(vdata.modifyArray(0)).cast("B")[: verts.nbytes] = verts.tobytes()

    # Two triangles per cell:
    #   (v0, v0+1, v0+m) and (v0+1, v0+m+1, v0+m), with v0 = j*m + i.
    jj, ii = np.meshgrid(np.arange(m - 1), np.arange(m - 1), indexing="ij")
    v0 = (jj * m + ii).ravel().astype(np.uint32)
    tri = np.empty((v0.size, 6), dtype=np.uint32)
    tri[:, 0] = v0
    tri[:, 1] = v0 + 1
    tri[:, 2] = v0 + m
    tri[:, 3] = v0 + 1
    tri[:, 4] = v0 + m + 1
    tri[:, 5] = v0 + m
    indices = tri.reshape(-1)

    tris = GeomTriangles(Geom.UHStatic)
    # >65535 verts at the default subdivisions, so indices must be 32-bit.
    tris.setIndexType(GeomEnums.NT_uint32)
    tris.addNextVertices(indices.size)
    memoryview(tris.modifyVertices()).cast("B")[: indices.nbytes] = indices.tobytes()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("ocean_grid")
    node.addGeom(geom)
    return node


class Ocean:
    def __init__(
        self,
        game: Any,
        water_color: LVecBase3f = LVecBase3f(0.02, 0.06, 0.14),
        ripple_strength: float = 0.02,
        wind_dir: LVecBase2f = LVecBase2f(1.0, 0.0),
        wind_strength: float = 0.5,
        wave_iterations: int = 36,
        geometric_swell: bool = False,
        swell_grid_subdivs: int = 512,
        swell_grid_half: float = 4000.0,
        swell_amplitude: float = 20.0,
        swell_scale: float = 0.004,
        swell_drift: float = 0.1,
        fm_depth: float = 0.2,
        wave_fade_near: float = 300.0,
        wave_fade_far: float = 5000.0,
        wave_fade_k2: float = 0.001,
        haze_color: LVecBase3f = LVecBase3f(0.2, 0.2, 0.4),
        haze_distance: float = 150000.0,
        planet_radius: float = PLANET_RADIUS_M,
        exact_sphere: bool = False,
        vert_shader: Path = _SHADERS / "ocean.vert",
        frag_shader: Path = _SHADERS / "ocean.frag",
    ):
        """
        Build the ocean surface, its shared reflection buffer, and shader.

        :param game: The game instance.
        :param water_color: Deep-water scatter colour.
        :param ripple_strength: Normal-based UV perturbation strength.
        :param wind_dir: Wind direction in the XY plane (normalised).
        :param wind_strength: How strongly the waves align to the wind; 0 =
            random, 1 = fully aligned.
        :param wave_iterations: Wave-detail quality knob; higher is sharper but
            costlier (e.g. 16 or 36).
        :param geometric_swell: Prototype mode that geometrically displaces the
            surface by the swell rather than only shading it.
        :param swell_grid_subdivs: Tessellation of the dense centre grid used by
            the geometric-swell mode.
        :param swell_grid_half: Half-size of the dense grid, in world units.
        :param swell_amplitude: Vertical swell displacement, in world units.
        :param swell_scale: Swell spatial frequency, shared by the vertex and
            fragment shaders; a long wavelength so the geometric-swell mesh
            samples it without aliasing.
        :param swell_drift: Swell scroll speed along the wind, shared by the
            vertex and fragment shaders.
        :param fm_depth: Small-wave frequency-modulation depth, in radians; the
            swell field modulates the wave phase to break the dominant octave's
            tiling (0 = off). Fragment-only; does not affect the geometric-swell
            displacement.
        :param wave_fade_near: Distance, in world units, below which small waves
            are at full detail.
        :param wave_fade_far: Distance, in world units, beyond which small waves
            are fully suppressed.
        :param wave_fade_k2: Exponential decay rate for the iteration count
            (iter = wave_iterations * exp(-k2 * dist)).
        :param haze_color: Colour distant water fades toward. Should match what
            the cloud field fades to, or sea and sky disagree at the horizon.
        :param haze_distance: Metres over which water reaches haze_color fully.
        :param planet_radius: Radius the surface curves with (its droop is
            d^2 / 2R) and whose horizon it must reach, in metres. Must match
            the clouds' and the terrain's.
        :param exact_sphere: Curve the geometric-swell surface on the exact
            sphere rather than the d^2 / 2R paraboloid (see
            :func:`make_swell_grid_mesh`): for real spherical terrain.
        :param vert_shader: Path to ocean.vert.
        :param frag_shader: Path to ocean.frag.
        """
        self.game = game
        self.id = uuid.uuid4()
        self.base_node = self.game.root_node.attachNewNode("ocean")

        # ── Reflection buffer ─────────────────────────────────────────────────
        settings = getattr(self.game.app, "graphics_settings", None)
        refl_scale = (
            settings.config["render"]["reflection_scale"]
            if settings is not None
            else DEFAULT_REFLECTION_SCALE
        )
        self.refl_tex, uv_scale = self.make_reflection_buffer(refl_scale, water_color)

        # ── Shader ────────────────────────────────────────────────────────────
        shader = Shader.load(Shader.SL_GLSL, vertex=vert_shader, fragment=frag_shader)

        # ── Camera-locked surface ─────────────────────────────────────────────
        # Follows the camera in XY each frame so the view is always covered to the
        # horizon. The reflection mirror (and a game's collision plane) stay flat
        # at z=0; both are exact at the camera, where the droop is zero.
        horizon = math.sqrt(2.0 * planet_radius * _MAX_EYE_ALTITUDE_M)
        outer_half = horizon * _HORIZON_MARGIN
        if geometric_swell:
            mesh = make_swell_grid_mesh(
                swell_grid_half,
                swell_grid_subdivs,
                outer_half,
                curvature=1.0 / (2.0 * planet_radius),
                exact_sphere=exact_sphere,
            )
        else:
            # Four vertices cannot curve, so this path stays flat.
            mesh = make_plane_mesh(outer_half * 2.0)
        self.ocean_node = self.base_node.attachNewNode(mesh)
        self.ocean_node.setShader(shader)
        self.ocean_node.setShaderInput("iTime", 0.0)
        self.ocean_node.setShaderInput("iCameraPos", LVecBase3f(0, 0, 20))
        self.ocean_node.setShaderInput("uModelMatrix", LMatrix4f.identMat())
        self.ocean_node.setShaderInput("iWaterColor", water_color)
        self.ocean_node.setShaderInput("iReflectionTex", self.refl_tex)
        self.ocean_node.setShaderInput("iRippleStrength", ripple_strength)
        self.ocean_node.setShaderInput("iWindDir", wind_dir)
        self.ocean_node.setShaderInput("iWindStrength", wind_strength)
        self.ocean_node.setShaderInput(
            "iIterationsNormal", min(int(wave_iterations), _MAX_WAVE_ITER)
        )
        self.ocean_node.setShaderInput(
            "iWaveDirs", compute_wave_dirs(wind_dir, wind_strength)
        )
        self.ocean_node.setShaderInput("uReflMVP", LMatrix4f.identMat())
        self.ocean_node.setShaderInput("uReflUVScale", uv_scale)
        # Geometric-swell prototype: vertex-shader displacement (off by default).
        self.ocean_node.setShaderInput("uGeometricSwell", 1 if geometric_swell else 0)
        self.ocean_node.setShaderInput("uSwellAmplitude", float(swell_amplitude))
        self.ocean_node.setShaderInput("uSwellGridHalf", float(swell_grid_half))
        # Swell frequency/drift shared by the vertex and fragment shaders.
        self.ocean_node.setShaderInput("iSwellScale", float(swell_scale))
        self.ocean_node.setShaderInput("iSwellDrift", float(swell_drift))
        # Swell-driven frequency modulation of the small waves (frag-only).
        self.ocean_node.setShaderInput("uFmDepth", float(fm_depth))
        self.ocean_node.setShaderInput("uDebugMode", 0)
        self.ocean_node.setShaderInput("uWaveOff", 0)
        self.ocean_node.setShaderInput("uExposure", 0.9)
        self.ocean_node.setShaderInput("uWaveFadeNear", float(wave_fade_near))
        self.ocean_node.setShaderInput("uWaveFadeFar", float(wave_fade_far))
        self.ocean_node.setShaderInput("uWaveFadeK2", float(wave_fade_k2))
        # Aerial perspective, to meet the cloud haze at the horizon.
        self.ocean_node.setShaderInput("uHazeColor", haze_color)
        self.ocean_node.setShaderInput("uHazeDistance", float(haze_distance))
        # No atmosphere unless a world binds its own (atmosphere.Atmospheres):
        # then they replace the haze. The base frame's rotation and the sun
        # are the binder's to keep current (set_atmosphere_frame).
        bind_defaults(self.ocean_node)
        self.ocean_node.setShaderInput("uBaseToWorld", LMatrix4f.identMat())
        self.ocean_node.setShaderInput("uAtmoSunDir", LVecBase3f(0.0, 0.0, 1.0))
        # Hide the ocean from the reflection camera
        self.ocean_node.hide(_OCEAN_BIT)

        self.game.method_lists[self.id] = [self.update]

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self):
        """Per-frame update (self-registered in ``game.method_lists``)."""
        current_time = self.game.game_time.get_current_time()
        camera_pos = self.game.app.camera.getPos(self.base_node)
        self.mirror_camera()

        # Reflection view-projection, for the shaders' reflection lookup
        view = LMatrix4f()
        view.invertFrom(self.refl_cam.getMat(self.base_node))
        proj = self.refl_cam_node.getLens().getProjectionMat()
        mvp = view * proj

        # The valid region of the reflection texture is buffer_size /
        # texture_size. The texture is padded to a power of two only on some
        # pipelines (the default one pads, simplepbr does not), and its real size
        # is known only once the buffer is realized, so refresh uReflUVScale from
        # the actual dimensions on the first realized frame.
        if not hasattr(self, "_uv_scale_set"):
            tex_w = self.refl_tex.getXSize()
            tex_h = self.refl_tex.getYSize()
            if tex_w > 0 and tex_h > 0:
                self._uv_scale_set = True
                uv_scale = LVecBase2f(
                    self.refl_buffer.getXSize() / tex_w,
                    self.refl_buffer.getYSize() / tex_h,
                )
                self.ocean_node.setShaderInput("uReflUVScale", uv_scale)

        # Keep the plane centred under the camera so the view is always covered.
        # The wave pattern is anchored in world space (shader reads world
        # position), so sliding the plane introduces no motion artifacts.
        self.ocean_node.setPos(camera_pos.x, camera_pos.y, 0.0)
        self.ocean_node.setShaderInput(
            "uModelMatrix", self.ocean_node.getMat(self.base_node)
        )
        self.ocean_node.setShaderInput("iTime", current_time)
        self.ocean_node.setShaderInput("iCameraPos", camera_pos)
        self.ocean_node.setShaderInput("uReflMVP", mvp)

    def set_atmosphere_frame(self, base_to_world, sun_direction) -> None:
        """For a bound atmosphere: how the base node's axes sit in the world
        (the atmospheres work in world axes) and where the sun is.

        :param base_to_world: the base node's transform relative to render
            (an LMatrix4f, e.g. ``base_node.get_mat(render)``; only its
            rotation is used)
        :param sun_direction: toward the sun, world axes
        """
        self.ocean_node.setShaderInput("uBaseToWorld", base_to_world)
        self.ocean_node.setShaderInput("uAtmoSunDir", LVecBase3f(*sun_direction))

    def set_wave_iterations(self, iterations: int):
        """Adjust wave-detail quality at runtime (e.g. from a settings menu).

        Higher = sharper waves up close but more expensive per pixel; lower =
        smoother and cheaper.  Far/grazing water already uses fewer iterations.

        :param iterations: Wave-detail iteration count, clamped to the shader's
            maximum.
        """
        self.ocean_node.setShaderInput(
            "iIterationsNormal", min(int(iterations), _MAX_WAVE_ITER)
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def make_reflection_buffer(
        self, refl_scale: float, water_color: LVecBase3f
    ) -> tuple[Texture, LVecBase2f]:
        """
        Create the offscreen buffer and mirrored camera for planar reflections.

        :param refl_scale: Reflection resolution as a fraction of the 3D render
            size (< 1 renders reflections cheaper than the main view).
        :param water_color: Colour the buffer is cleared to, seen where the
            reflection camera renders nothing.
        :return: The reflection :class:`Texture` and the provisional UV scale
            (buffer size / padded texture size) for the first frame.
        """
        # Size off the 3D render resolution (which may be a downscale of the
        # window when render-scale is active) rather than the window itself, so
        # the reflection cost scales with the chosen internal resolution.
        manager = getattr(self.game.app, "graphics_manager", None)
        if manager is not None:
            render_w, render_h = manager.get_render_size()
        else:
            render_w, render_h = (
                self.game.app.win.get_x_size(),
                self.game.app.win.get_y_size(),
            )
        buf_w = max(1, int(render_w * refl_scale))
        buf_h = max(1, int(render_h * refl_scale))

        refl_tex = Texture("refl_tex")
        refl_tex.setWrapU(Texture.WMClamp)
        refl_tex.setWrapV(Texture.WMClamp)

        self.refl_buffer = self.game.app.win.makeTextureBuffer(
            "refl_buffer", buf_w, buf_h, refl_tex
        )
        self.refl_buffer.setSort(-100)
        # Clear the reflection buffer to the water colour, so wherever the
        # reflection camera renders nothing (no skybox/geometry) the reflection
        # blends into the water.
        self.refl_buffer.setClearColor(LVecBase4f(*water_color, 1))

        self.refl_cam_node = Camera("refl_cam")
        # Use a wider lens than the main camera to avoid frustum culling
        # artifacts when turning quickly — the extra margin ensures reflected
        # objects are never culled at the edges of the reflection buffer.
        refl_lens = self.game.app.camLens.makeCopy()
        fov = self.game.app.camLens.getFov()
        refl_lens.setFov(fov.x * 1.4, fov.y * 1.4)
        # The reflection camera sits ~2*altitude further from the skybox's far
        # side than the main camera, so with the same far plane the skybox could
        # be clipped at the reflected zenith. Push the reflection far plane well
        # past it so it always renders in full.
        refl_lens.setFar(self.game.app.camLens.getFar() * 4.0)
        self.refl_cam_node.setLens(refl_lens)
        self.refl_cam = self.base_node.attachNewNode(self.refl_cam_node)

        # Use only the lower 20 bits, excluding bit 1 (_OCEAN_BIT)
        self.refl_cam_node.setCameraMask(BitMask32(0xFFFFF & ~2))

        # TODO(perf): the reflection re-renders all scene geometry. Objects that
        # contribute little (small, distant) could be excluded with another
        # camera-mask bit as a "do not reflect" flag, as _OCEAN_BIT does for the
        # ocean. Worth it with many minor objects, not for one large hero object.

        self.refl_buffer.makeDisplayRegion(0, 1, 0, 1).setCamera(self.refl_cam)

        clip_plane = PlaneNode("water_clip")
        clip_plane.setPlane(LPlane(0, 0, 1, 0))
        self.clip_np = self.base_node.attachNewNode(clip_plane)
        clip_attrib = ClipPlaneAttrib.makeAllOff().addOnPlane(self.clip_np)
        self.refl_cam_node.setInitialState(
            self.refl_cam_node.getInitialState().addAttrib(clip_attrib)
        )

        # Provisional UV scale assuming power-of-two padding; update() corrects it
        # from the real texture size on the first realized frame.
        uv_scale = LVecBase2f(
            buf_w / compute_next_power_of_2(buf_w),
            buf_h / compute_next_power_of_2(buf_h),
        )
        return refl_tex, uv_scale

    def mirror_camera(self):
        """Mirror the reflection camera across the water plane (Z = 0)."""
        pos = self.game.app.camera.getPos(self.base_node)
        hpr = self.game.app.camera.getHpr(self.base_node)
        self.refl_cam.setPos(self.base_node, pos.x, pos.y, -pos.z)
        self.refl_cam.setHpr(self.base_node, hpr.x, -hpr.y, -hpr.z)

    def clean(self):
        """
        Cleans the Ocean object
        """
        if self.game.method_lists:
            self.game.method_lists.pop(self.id, None)
        # Destroy the offscreen reflection buffer; removing base_node alone
        # leaves the buffer registered with the graphics engine, so it keeps
        # rendering its camera every frame (a leak across scene reloads).
        self.game.app.graphicsEngine.removeWindow(self.refl_buffer)
        self.base_node.removeNode()
        self.game = None
