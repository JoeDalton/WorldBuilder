"""
The mountains' shadow map (world_builder.terrain.shadow_map): none on a flat
planet, a peak's shadow as long as its height over the sun's tangent, the
sweep's ceiling against a brute-force march, more sun higher up, the
ShadowMap's bake in a pool and its re-bake for a new sun, and shadows.glsl's
terrainShadow read back against the CPU twin.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from panda3d.core import NodePath

from world_builder.clouds.weather import sample_cube
from world_builder.shadows import SUN_ANGULAR_RADIUS
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.shadow_map import (
    ShadowMap,
    ceiling_cube,
    terrain_visibility,
)
from world_builder.terrain.workers import ChunkExecutor

R = 600e3
SIZE = 512
#: A flat-topped peak: 3 km high, 6 km in radius, over the direction PEAK.
PEAK = np.array([0.3, 0.2, 0.93]) / np.linalg.norm([0.3, 0.2, 0.93])
TOP, WIDE = 3000.0, 6e3


def _peak(dirs):
    across = np.arccos(np.clip(dirs @ PEAK, -1.0, 1.0)) * R
    return TOP * np.clip((WIDE + 1e3 - across) / 1e3, 0.0, 1.0)


def _sun(elevation):
    """A sun *elevation* degrees over the peak's horizon, toward its east."""
    east = np.cross([0.0, 0.0, 1.0], PEAK)
    east /= np.linalg.norm(east)
    el = math.radians(elevation)
    return math.sin(el) * PEAK + math.cos(el) * east, east


def _behind(distance, east):
    """The direction *distance* metres from the peak's centre, away from the
    sun."""
    angle = distance / R
    return math.cos(angle) * PEAK - math.sin(angle) * east


@pytest.fixture(scope="module")
def heights():
    return _peak(cube_directions(SIZE)).astype(np.float32)


@pytest.fixture(scope="module")
def baked(heights):
    sun, east = _sun(10.0)
    return ceiling_cube(heights, R, sun), sun, east


def _visibility(cube, dirs, altitude=0.0):
    ceiling, distance = np.moveaxis(sample_cube(cube, np.atleast_2d(dirs)), -1, 0)
    return terrain_visibility(ceiling, distance, altitude, SUN_ANGULAR_RADIUS)


def test_a_flat_planet_casts_no_shadow():
    sun, _ = _sun(3.0)
    cube = ceiling_cube(np.zeros((6, 64, 64), np.float32), R, sun)
    dirs = cube_directions(64).reshape(-1, 3)
    up = dirs @ (sun / np.linalg.norm(sun))
    lit = _visibility(cube, dirs[up > math.sin(math.radians(1.0))])
    np.testing.assert_allclose(lit, 1.0)


def test_a_peak_casts_a_shadow_its_height_over_the_suns_tangent(baked):
    """10 degrees up: 3 km cast ~17 km beyond the peak's edge (a little
    less: the ground curves away). Nearer than the bake's step (two texels,
    ~4.7 km here) the peak is not seen: part E's."""
    cube, _, east = baked
    length = TOP / math.tan(math.radians(10.0))
    edge = WIDE + 1e3
    shadowed = [_visibility(cube, _behind(edge + d, east))[0] for d in (6e3, 12e3)]
    lit = _visibility(cube, _behind(edge + length + 4e3, east))[0]
    assert max(shadowed) < 0.05
    assert lit > 0.95
    sunward = _visibility(cube, math.cos(0.05) * PEAK + math.sin(0.05) * east)[0]
    assert sunward == 1.0


def test_the_sweep_is_a_march_toward_the_sun(baked, heights):
    """Past its first step, the ceiling is the lowest height from which a
    ray toward the sun clears the ground the map was given (its texels,
    bilinear), found by marching it."""
    cube, sun, east = baked
    ground = heights[..., None]
    unit = sun / np.linalg.norm(sun)
    t = np.linspace(100.0, 80e3, 8000)
    for distance in (14e3, 20e3, 25e3):
        p = _behind(distance, east)

        def clears(a):
            q = p[None] * (R + a) + unit[None] * t[:, None]
            r = np.linalg.norm(q, axis=1)
            return np.all(r >= R + sample_cube(ground, q)[:, 0])

        lo, hi = -5000.0, 10000.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if clears(mid) else (mid, hi)
        baked_ceiling = sample_cube(cube, p[None])[0, 0]
        # Within the grid's resolution (2.3 km texels here; ~40 m at 1024),
        # and not growing along the shadow.
        assert baked_ceiling == pytest.approx(hi, abs=150.0)


def test_more_sun_higher_up(baked):
    """Shadow on a heightfield only lessens with altitude."""
    cube, _, east = baked
    p = _behind(12e3, east)
    altitudes = np.linspace(0.0, 4000.0, 81)
    seen = _visibility(np.broadcast_to(cube, cube.shape), p, altitudes)
    assert np.all(np.diff(seen) >= 0.0)
    assert seen[0] < 0.05 and seen[-1] == 1.0


def test_a_shadow_map_bakes_in_the_pool_and_again_for_a_new_sun():
    """ShadowMap: heights per face then the ceiling, in the pool; off
    until baked, then current; a new sun is baked again; a new size starts
    over; the CPU twin reads the map."""
    spec = PRESETS["moon"]
    terrain = SimpleNamespace(
        spec=spec,
        radius=spec.radius,
        rotation=np.eye(3),
        to_planet=lambda points: np.asarray(points, float),
    )
    executor = ChunkExecutor(backend="inline")
    try:
        shadow = ShadowMap(terrain, executor, 32, (0.0, 0.3, 1.0))
        node = NodePath("ground")
        shadow.bind(node)
        assert shadow.loading and not shadow.current
        for _ in range(3):
            shadow.update()
        assert not shadow.loading and shadow.current
        assert shadow.cube.shape == (6, 32, 32, 2)
        assert node.get_shader_input("uTerrainShadow").get_texture().get_x_size() == 32
        assert shadow._on[0] == 1
        first = shadow.cube
        shadow.set_sun((1.0, 0.0, 0.2))
        assert not shadow.current
        for _ in range(3):
            shadow.update()
        assert shadow.current and shadow.cube is not first
        points = np.array([[0.0, 0.0, spec.radius + 5e4], [spec.radius, 0.0, 0.0]])
        assert shadow.visibility(points, SUN_ANGULAR_RADIUS).shape == (2,)
        assert shadow.visibility(points[:1], SUN_ANGULAR_RADIUS)[0] == 1.0
        shadow.set_size(0)
        assert shadow.current and not shadow.loading and shadow._on[0] == 0
    finally:
        executor.shutdown()


# ── terrainShadow read back ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def window(showbase):
    """The session's ShowBase with an offscreen window; skipped where none
    can be opened."""
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


def test_the_shader_terrain_shadow_is_the_cpus(window, baked):
    """terrainShadow (the half-float cube in km, filtered by the GPU)
    matches terrain_visibility across the peak's shadow and its penumbra."""
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomPoints,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        OmniBoundingVolume,
        Shader,
        Texture,
    )

    from world_builder.shadows import Shadows, bind_defaults

    cube, _, east = baked
    size = 32
    rng = np.random.default_rng(3)
    distance = rng.uniform(8e3, 40e3, size * size)
    sideways = rng.uniform(-8e3, 8e3, size * size)
    north = np.cross(PEAK, east)
    dirs = (
        np.array([_behind(d, east) for d in distance]) + (sideways[:, None] / R) * north
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    altitude = rng.uniform(0.0, 3000.0, size * size)
    # What the GPU reads: the cube in half floats, km.
    stored = (cube / 1e3).astype(np.float16).astype(np.float64) * 1e3
    expected = _visibility(stored, dirs, altitude)
    assert expected.min() < 0.05 and expected.max() == 1.0
    assert ((expected > 0.05) & (expected < 0.95)).sum() > 10  # penumbrae

    vdata = GeomVertexData("points", GeomVertexFormat.get_v3n3(), Geom.UH_static)
    position = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    for d, a in zip(dirs, altitude):
        position.add_data3f(*(float(v) for v in d))
        normal.add_data3f(float(a), 0.0, 0.0)
    prim = GeomPoints(Geom.UH_static)
    prim.add_next_vertices(len(dirs))
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("terrain_shadow_points")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    source = (
        (Path(__file__).parents[1] / "src/world_builder/shadows/shaders/shadows.glsl")
        .read_text()
        .replace("#pragma once", "")
    )
    vertex = f"""#version 330
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
flat out float vVisible;
{source}
void main() {{
    vVisible = terrainShadow(normalize(p3d_Vertex.xyz), p3d_Normal.x);
    int x = gl_VertexID % {size}, y = gl_VertexID / {size};
    gl_Position = vec4((vec2(x, y) + 0.5) / {size}.0 * 2.0 - 1.0, 0.0, 1.0);
}}
"""
    fragment = """#version 330
flat in float vVisible;
out vec4 fragColor;
void main() {
    float v = floor(clamp(vVisible, 0.0, 1.0) * 16711680.0 + 0.5);
    float r = floor(v / 65536.0);
    float g = floor((v - r * 65536.0) / 256.0);
    fragColor = vec4(r, g, v - r * 65536.0 - g * 256.0, 255.0) / 255.0;
}
"""
    root = NodePath("terrain_shadow_check")
    points_np = root.attach_new_node(node)
    points_np.set_shader(Shader.make(Shader.SL_GLSL, vertex, fragment))
    bind_defaults(points_np)
    Shadows().bind(points_np)
    shadow = ShadowMap.__new__(ShadowMap)  # its texture alone, from the cube
    shadow._texture = Texture("terrain_shadow")
    shadow._on = np.zeros(1, dtype=int)
    shadow._upload(cube)
    points_np.set_shader_inputs(uTerrainShadow=shadow._texture, uTerrainShadowOn=1)
    texture = Texture("terrain_shadow_check")
    buffer = window.win.make_texture_buffer(
        "terrain_shadow_check", size, size, texture, True
    )
    try:
        camera = window.makeCamera(buffer)
        camera.node().set_scene(root)
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGB").get_data(), np.uint8)
        rgb = ram.reshape(size, size, 3).astype(np.int64)
        got = (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]) / 16711680.0
        # The GPU's filtering weights are 8-bit: steep penumbrae differ a
        # little.
        error = np.abs(got.ravel() - expected)
        assert np.median(error) < 1e-3
        assert np.percentile(error, 99) < 0.05
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()
