"""
The planet's clouds' shadow on what is under them (cloud_field.glsl's
cloudShadowDepth, noise.cloud_shadow_on_sphere): none without cloud or above
the slab, the slab's column under an overhead sun, a longer chord under a low
one, more with more cover; and the shader read back against the CPU twin.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from panda3d.core import NodePath, Vec3

from world_builder.atmosphere import bind_defaults
from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.noise import (
    CLOUD_SHADOW_STEP_SCALE,
    DECK_LIGHT_COS_MIN,
    LAYER_VEC4S,
    MAX_LAYERS,
    NOISE_FEATURE,
    NOISE_SIZE,
    build_noise_texture,
    cloud_shadow_on_sphere,
    far_cloud_depth,
    fbm_sigma,
    ground_shadow_table,
    pack_layer_params,
    sphere_column_peaks,
    with_threshold,
)
from world_builder.clouds.sphere import SHADE_CACHE_SIZE, _noise
from world_builder.clouds.weather import (
    WeatherSpec,
    build_weather_texture,
    coverage_to_shift,
    sample_cube,
    shift_table,
    weather_cube,
)
from world_builder.shadows import bind_defaults as bind_shadow_defaults
from world_builder.terrain.chart import TangentFrame
from world_builder.terrain.planet import PRESETS as PLANETS

R = 600e3
RAW = replace(PRESETS[CloudType.CUMULUS].field, slab=(2200.0, 1000.0))
OPTICS = PRESETS[CloudType.CUMULUS].optics
SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"


@pytest.fixture(scope="module")
def peaks():
    return sphere_column_peaks(_noise(7), RAW, R)


@pytest.fixture(scope="module")
def field(peaks):
    return with_threshold(RAW, peaks, fbm_sigma(_noise(7), RAW))


@pytest.fixture(scope="module")
def table(peaks, field):
    return shift_table(peaks, field.threshold[0])


def _shadow(field, points, sun, steps=8, shift=0.0, optics=OPTICS):
    return cloud_shadow_on_sphere(
        _noise(7),
        field,
        optics,
        np.asarray(points, dtype=np.float64),
        sun,
        R,
        steps,
        shift_at=lambda p: np.full(len(p), shift),
    )


def _ground(n=400, seed=1):
    """Random points 10 m over sea level, and their verticals."""
    rng = np.random.default_rng(seed)
    up = rng.normal(size=(n, 3))
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    return up * (R + 10.0), up


def test_no_cloud_no_shadow(field, table):
    """A clear sky, a point over the slab, or no steps: no shadow."""
    points, up = _ground()
    clear = coverage_to_shift(0.0, table) - 1.0
    assert np.all(_shadow(field, points, up, shift=clear) == 0.0)
    overcast = coverage_to_shift(1.0, table)
    above = up * (R + sum(field.slab) + 50.0)
    assert np.all(_shadow(field, above, up, shift=overcast) == 0.0)
    assert np.all(_shadow(field, points, up, steps=0, shift=overcast) == 0.0)


def _column(field, slant=1.0):
    """The depth of a full-cover column: the slab's base fill integrated
    over its thickness, along a path *slant* times as long."""
    h = (np.arange(100000) + 0.5) / 100000
    fill = (1.0 - np.exp2(-field.base_ramp * h)).mean()
    return fill * field.slab[1] * slant * field.density * OPTICS.shadow_strength


# Steps of a metre, as many as the chord takes: the march as an integral.
FINE = replace(OPTICS, sun_step_length=1.0)


def test_an_overhead_sun_through_a_full_slab_is_its_column(field):
    """Every column full of cloud (a huge shift): under an overhead sun the
    depth is the slab's own; the march's own steps (four through 1 km)
    come within two percent (the base's ramp, by the midpoint rule)."""
    points, up = _ground(50)
    full = 100.0
    fine = _shadow(field, points, up, steps=4096, shift=full, optics=FINE)
    np.testing.assert_allclose(fine, _column(field), rtol=1e-4)
    coarse = _shadow(field, points, up, steps=8, shift=full)
    np.testing.assert_allclose(coarse, _column(field), rtol=0.02)


def _full_chord(field, point, sun):
    """Brute force: a full slab's depth along the ray, by centimetre steps
    of its base fill (no shared code: where the ray is in the shell, by
    its radius)."""
    t = np.arange(0.0, 20000.0, 0.01) + 0.005
    radii = np.linalg.norm(point + t[:, None] * sun, axis=1)
    h = (radii - R - field.slab[0]) / field.slab[1]
    inside = (h > 0.0) & (h < 1.0)
    fill = (1.0 - np.exp2(-field.base_ramp * h)) * inside
    return fill.sum() * 0.01 * field.density * OPTICS.shadow_strength


@pytest.mark.parametrize("elevation", [60.0, 30.0, 15.0])
def test_a_low_sun_crosses_a_longer_chord(field, elevation):
    """A full slab's depth under a low sun is the chord's (longer than the
    column, shorter than flat ground's 1 / sine: the shell curves away), and
    the steps lengthen with it rather than multiply."""
    frame = TangentFrame.at(10.0, 20.0)
    el = math.radians(elevation)
    sun = math.sin(el) * frame.up + math.cos(el) * frame.east
    point = frame.up * (R + 10.0)
    depth = _shadow(field, point[None], sun, steps=8192, shift=100.0, optics=FINE)[0]
    assert depth == pytest.approx(_full_chord(field, point, sun), rel=1e-3)
    assert _column(field) < depth < _column(field, 1.0 / math.sin(el))
    capped = _shadow(field, point[None], sun, steps=8, shift=100.0)[0]
    assert capped == pytest.approx(depth, rel=0.03)  # 8 longer steps


def test_more_cover_more_shadow(field, table):
    """Where the weather adds cloud, the shadow only deepens."""
    points, up = _ground(400, seed=4)
    tilt = np.cross(up, [0.0, 0.0, 1.0])
    tilt /= np.linalg.norm(tilt, axis=1, keepdims=True)
    sun = 0.8 * up + 0.6 * tilt
    depths = [
        _shadow(field, points, sun, shift=coverage_to_shift(c, table))
        for c in (0.1, 0.3, 0.6, 0.9)
    ]
    for less, more in zip(depths, depths[1:]):
        assert np.all(more >= less - 1e-12)
    assert depths[0].mean() < depths[-1].mean()
    assert 0.0 < (depths[1] > 0).mean() < 1.0  # broken cover: some sun, some shade


# ── cloudShadowDepth read back ─────────────────────────────────────────────────


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


def test_the_shader_cloud_shadow_is_the_cpus(window, field, table):
    """cloudShadowDepth (float32, the noise and the weather through the
    GPU's filtering) matches cloud_shadow_on_sphere under a real weather
    map, from overhead to a low sun."""
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomPoints,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        LVecBase4f,
        OmniBoundingVolume,
        PTA_LVecBase4f,
        Shader,
        Texture,
    )

    size = 32
    cube = weather_cube(WeatherSpec(PLANETS["earthlike"]), table)
    frame = TangentFrame.at(5.0, 30.0)
    rng = np.random.default_rng(6)
    offsets = rng.uniform(-40e3, 40e3, size=(size * size, 2))
    up = frame.up + (offsets[:, :1] * frame.east + offsets[:, 1:] * frame.north) / R
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    points = (up * (R + 20.0)).astype(np.float32).astype(np.float64)
    el = np.radians(rng.uniform(12.0, 90.0, size * size))[:, None]
    sun = np.sin(el) * up + np.cos(el) * frame.east
    sun = (sun / np.linalg.norm(sun, axis=1, keepdims=True)).astype(np.float32)
    steps = 8
    expected = np.exp2(
        -cloud_shadow_on_sphere(
            _noise(7),
            field,
            OPTICS,
            points,
            sun.astype(np.float64),
            R,
            steps,
            shift_at=lambda p: sample_cube(cube[..., 1], p),
        )
    )
    assert expected.min() < 0.5 and expected.max() == 1.0  # shade and sun

    vdata = GeomVertexData("points", GeomVertexFormat.get_v3n3(), Geom.UH_static)
    position = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    for p, d in zip(points, sun):
        position.add_data3f(*(float(v) for v in p))
        normal.add_data3f(*(float(v) for v in d))
    prim = GeomPoints(Geom.UH_static)
    prim.add_next_vertices(len(points))
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("cloud_shadow_points")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    # cloud_field.glsl's tint helpers need the air's (atmoSunlight), which
    # needs the shadows': pasted in order, their includes resolved here.
    atmosphere = SHADERS.parents[1] / "atmosphere/shaders/atmosphere.glsl"
    shadows = SHADERS.parents[1] / "shadows/shaders/shadows.glsl"
    include = "\n".join(
        line
        for path in (shadows, atmosphere, SHADERS / "cloud_field.glsl")
        for line in path.read_text().splitlines()
        if not line.startswith("#pragma")
    )
    vertex = f"""#version 330
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
flat out float vVisible;
{include}
void main() {{
    float depth = cloudShadowDepth(getField(0), p3d_Vertex.xyz, p3d_Normal, {steps});
    vVisible = exp2(-depth);
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
    params = PTA_LVecBase4f.empty_array(MAX_LAYERS * LAYER_VEC4S)
    packed = pack_layer_params(
        field,
        extinction=field.density,
        wrap_radius=1e12,
        wrap_fade_band=0.0,
        octave_mean=float(_noise(7).mean()),
        optics=OPTICS,
        octaves=2,
    )
    for i, row in enumerate(packed):
        params[i] = LVecBase4f(*(float(v) for v in row))
    root = NodePath("cloud_shadow_check")
    points_np = root.attach_new_node(node)
    points_np.set_shader(Shader.make(Shader.SL_GLSL, vertex, fragment))
    bind_defaults(points_np)
    bind_shadow_defaults(points_np)
    points_np.set_shader_inputs(
        cloudNoise=build_noise_texture(_noise(7)),
        noiseSize=float(NOISE_SIZE),
        noiseOffset=Vec3(0.0, 0.0, 0.0),
        sunDir=Vec3(0.0, 0.0, 1.0),
        layerParams=params,
        uSpherical=1,
        planetRadius=R,
        weatherMap=build_weather_texture(cube),
        uCloudDebug=0,
    )
    texture = Texture("cloud_shadow_check")
    buffer = window.win.make_texture_buffer(
        "cloud_shadow_check", size, size, texture, True
    )
    try:
        camera = window.makeCamera(buffer)
        camera.node().set_scene(root)
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGB").get_data(), np.uint8)
        rgb = ram.reshape(size, size, 3).astype(np.int64)
        got = (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]) / 16711680.0
        error = np.abs(got.ravel() - expected)
        # The GPU filters the noise and the weather with 8-bit weights: a
        # little off near a cloud's edge, where the smoothstep is steep.
        assert np.median(error) < 2e-3
        assert np.percentile(error, 99) < 0.05
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()


# ── the far shadow: its mean ───────────────────────────────────────────────────


def test_the_mean_shadow_darkens_with_cover_and_a_low_sun(field, table):
    """ground_shadow_table: no cloud, full sun; more cover or a lower sun,
    more shadow; none of the sun below the horizon; and each entry the
    mean of the march it stands for."""
    mean = ground_shadow_table(_noise(7), field, OPTICS, R, table, columns=512)
    lit = np.linspace(DECK_LIGHT_COS_MIN, 1.0, mean.shape[1]) > 0.0
    assert mean.shape == (17, 32)
    np.testing.assert_allclose(mean[0, lit], 1.0)
    assert np.all(mean[:, ~lit] == 0.0)
    assert np.all(np.diff(mean[:, lit], axis=0) <= 1e-6)  # more cover
    assert np.all(np.diff(mean[1:, lit], axis=1) >= -0.02)  # a higher sun
    assert mean[8, -1] < 0.9 and mean[-1, lit].max() < 0.1
    # The noon column at half cover, against the twin over other points.
    points, up = _ground(2000, seed=8)
    noon = np.exp2(-_shadow(field, points, up, shift=coverage_to_shift(0.5, table)))
    assert mean[8, -1] == pytest.approx(noon.mean(), abs=0.03)


def test_the_shader_knows_the_twins_constants():
    """cloud_field.glsl's NOISE_FEATURE (the far fade's scale) and
    CLOUD_SHADOW_STEP_SCALE (the march's steps) are noise.py's."""
    source = (SHADERS / "cloud_field.glsl").read_text()
    assert f"const float NOISE_FEATURE = {NOISE_FEATURE:.1f};" in source
    scale = f"const float CLOUD_SHADOW_STEP_SCALE = {CLOUD_SHADOW_STEP_SCALE:.1f};"
    assert scale in source


# ── SphericalClouds: the binding and the CPU side ─────────────────────────────


def test_clouds_shade_a_bound_node_once_ready_and_the_cpu_alike(showbase):
    """A node bound before the clouds' set-up gets no shadow (layer -1),
    then the deck's layer, the weather and the far table; the CPU shade is
    the twin's at the point's cell, cached until the sun changes, and
    removing the clouds takes the shadow off the node."""
    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.sphere import SHADE_CELL, SphericalClouds
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    clouds = SphericalClouds(
        showbase.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(shells=2),
        planet=PLANETS["forest"],
    )
    ground = NodePath("ground")
    try:
        clouds.bind_shadow_inputs(ground)
        bound = ground.get_shader_input("uCloudShadowLayer")
        assert clouds._shadow_layer[0] == -1  # the set-up is not done
        up = TangentFrame.at(20.0, 40.0).up
        points = up * (R + 30.0) + np.array([[0.0, 0.0, 0.0], [3e3, 0.0, 0.0]])
        sun = np.array([0.4, 0.3, 0.866])
        assert np.all(clouds.sun_visibility(points, sun, 8) == 1.0)  # not ready
        clouds.update(up * (R + 1000.0))
        assert clouds.weather is not None
        assert clouds._shadow_layer[0] == clouds._deck_layer >= 0
        assert ground.get_shader_input("uCloudShadowLayer") == bound  # in place
        light = ground.get_shader_input("deckLight").get_texture()
        assert light.get_num_components() == 3
        assert ground.get_shader_input("weatherMap").get_texture() is not None

        got = clouds.sun_visibility(points, sun, 8)
        cells = np.round(points / SHADE_CELL) * SHADE_CELL
        expected = np.exp2(
            -cloud_shadow_on_sphere(
                _noise(clouds._seed),
                clouds._deck_field,
                clouds._deck_optics,
                points,
                sun / np.linalg.norm(sun),
                R,
                8,
                shift_at=lambda p: sample_cube(clouds.weather[..., 1], p),
            )
        )
        np.testing.assert_allclose(got, expected, atol=0.05)
        assert len(clouds._shades) == 2
        clouds.sun_visibility(points + 3.0, sun, 8)  # the same cells
        assert len(clouds._shades) == 2
        assert np.abs(cells - points).max() <= SHADE_CELL / 2
        clouds.sun_visibility(points, -sun + 2 * up, 8)  # a new sun
        assert len(clouds._shades) == 2
        # A full cache is dropped before a call's lookup, not in it: a call
        # with a cell kept and a new one.
        clouds._shades.update({(k, 0, 0): 1.0 for k in range(SHADE_CACHE_SIZE)})
        moved = points + [[0.0, 0.0, 0.0], [0.0, 4.0 * SHADE_CELL, 0.0]]
        assert clouds.sun_visibility(moved, -sun + 2 * up, 8).shape == (2,)
        assert len(clouds._shades) == 2
    finally:
        clouds.remove()
        executor.shutdown()
    assert clouds._shadow_layer[0] == -1


# ── the cloud shadow maps (part E) ─────────────────────────────────────────────


def test_a_cloud_shadow_map_is_the_march_from_the_base(window):
    """Each texel of the near map is cloudShadowDepth from where its ray
    enters the slab's base (its chart node on the base's sphere), as the
    CPU twin marches it; drawn again only when the sun turns or the camera
    moves a quarter of it."""
    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.shadow_map import NEAR_SPACING, CloudShadowMaps
    from world_builder.clouds.sphere import SphericalClouds
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    clouds = SphericalClouds(
        window.render,
        R,
        executor,
        altitude=2200.0,
        settings=CloudSettings(shells=2),
        planet=PLANETS["forest"],
    )
    size, steps = 128, 8
    maps = CloudShadowMaps(window.win, clouds, size, to_ram=True)
    try:
        clouds.update(TangentFrame.at(20.0, 40.0).up * (R + 1000.0))
        assert clouds.weather is not None
        # The cloudiest of a few places: a map with cloud in it.
        places = [(lat, lon) for lat in range(-60, 61, 20) for lon in range(0, 360, 30)]
        cover = [
            sample_cube(clouds.weather[..., 0], TangentFrame.at(*place).up[None])[0]
            for place in places
        ]
        frame = TangentFrame.at(*places[int(np.argmax(cover))])
        camera = frame.up * (R + 1000.0)
        clouds.update(camera)
        sun = 0.8 * frame.up + 0.6 * frame.east
        clouds.set_sun(sun)
        maps.update(camera, sun / np.linalg.norm(sun), steps)
        assert maps._on[0] == 1 and maps.near.drawn is not None
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        texture = maps.near.texture
        dtype = np.float32 if texture.get_component_width() == 4 else np.float16
        ram = np.frombuffer(texture.get_ram_image_as("RGBA"), dtype)
        got = ram.reshape(size, size, 4)[..., 0].astype(np.float64)

        nodes = (np.arange(size) - 0.5 * size + 0.5) * NEAR_SPACING
        x, y = np.meshgrid(nodes, nodes)
        base = R + clouds._deck_field.slab[0]
        angle = np.hypot(x, y) / base
        toward = np.stack([x, y], -1) / np.maximum(np.hypot(x, y), 1e-9)[..., None]
        dirs = np.cos(angle)[..., None] * frame.up + np.sin(angle)[..., None] * (
            toward[..., :1] * frame.east + toward[..., 1:] * frame.north
        )
        expected = cloud_shadow_on_sphere(
            _noise(7),
            clouds._deck_field,
            clouds._deck_optics,
            dirs.reshape(-1, 3) * base,
            sun / np.linalg.norm(sun),
            R,
            steps,
            shift_at=lambda p: sample_cube(clouds.weather[..., 1], p),
        ).reshape(size, size)
        assert expected.max() > 1.0  # some cloud in it
        shade_error = np.abs(np.exp2(-got) - np.exp2(-expected))
        assert np.median(shade_error) < 2e-3
        assert np.percentile(shade_error, 99) < 0.05

        # Still: not drawn again; a turned sun or a camera moved far: again.
        near, far = maps.near.drawn, maps.far.drawn
        maps.update(camera, sun / np.linalg.norm(sun), steps)
        assert maps.near.drawn is near and maps.far.drawn is far
        turned = sun + 0.1 * frame.north
        maps.update(camera, turned / np.linalg.norm(turned), steps)
        assert maps.near.drawn is not near and maps.far.drawn is not far
        near, far = maps.near.drawn, maps.far.drawn
        step = 0.3 * size * NEAR_SPACING / R
        moved = (frame.up + step * frame.east) * (R + 1000.0)
        maps.update(moved, turned / np.linalg.norm(turned), steps)
        assert maps.near.drawn is not near and maps.far.drawn is far
    finally:
        maps.destroy()
        clouds.remove()
        executor.shutdown()


# ── the far segment (part F) ───────────────────────────────────────────────────


def _far(field, points, sun, reach, shift=0.0, optics=OPTICS):
    return far_cloud_depth(
        _noise(7),
        field,
        optics,
        np.asarray(points, dtype=np.float64),
        sun,
        R,
        reach,
        shift_at=lambda p: np.full(len(p), shift),
    )


def _in_slab(field, n=200, seed=5, height=0.5):
    rng = np.random.default_rng(seed)
    up = rng.normal(size=(n, 3))
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    return up * (R + field.slab[0] + height * field.slab[1]), up


def _low_sun(up, elevation):
    level = np.cross(up, [0.0, 0.0, 1.0])
    level /= np.linalg.norm(level, axis=1, keepdims=True)
    el = math.radians(elevation)
    return math.sin(el) * up + math.cos(el) * level


def test_the_far_segment_is_past_the_clouds_own_march(field, table):
    """Nothing within the cloud's own march, nothing past the slab's top
    (a high sun leaves it at once), and under a low sun more cloud the
    farther it looks (on the whole: a longer reach moves the steps)."""
    points, up = _in_slab(field)
    near = OPTICS.sun_steps * OPTICS.sun_step_length
    overcast = coverage_to_shift(1.0, table)
    assert np.all(_far(field, points, up, near, overcast) == 0.0)
    assert np.all(_far(field, points, up, 4000.0, overcast) == 0.0)  # straight up
    sun = _low_sun(up, 2.0)  # in the slab for ~14 km from its middle
    broken = coverage_to_shift(0.5, table)
    depths = [_far(field, points, sun, reach, broken) for reach in (2e3, 4e3, 8e3)]
    means = [depth.mean() for depth in depths]
    assert 0.0 < means[0] < means[1] < means[2]


def test_the_far_segment_through_a_full_slab_is_its_integral(field):
    """Every column full of cloud: the depth is the slab's base fill
    integrated along the ray from the cloud's own march to the reach (or
    the slab's top), the steps coarse but the fill smooth."""
    points, up = _in_slab(field, n=5)
    sun = _low_sun(up, 3.0)
    near = OPTICS.sun_steps * OPTICS.sun_step_length
    got = _far(field, points, sun, 6000.0, 100.0)
    for p, d, value in zip(points, sun, got):
        t = np.arange(near, 6000.0, 1.0) + 0.5
        radii = np.linalg.norm(p + t[:, None] * d, axis=1)
        h = (radii - R - field.slab[0]) / field.slab[1]
        fill = np.where((h > 0) & (h < 1), 1.0 - np.exp2(-field.base_ramp * h), 0.0)
        expected = fill.sum() * field.density * OPTICS.shadow_strength
        assert value == pytest.approx(expected, rel=0.03)


def test_the_shader_far_segment_is_the_cpus(window, field, table):
    """cloudFarDepth read back against far_cloud_depth, under a real
    weather map and a low sun."""
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomPoints,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        LVecBase4f,
        OmniBoundingVolume,
        PTA_LVecBase4f,
        Shader,
        Texture,
    )

    size = 32
    cube = weather_cube(WeatherSpec(PLANETS["earthlike"]), table)
    points, up = _in_slab(field, n=size * size, seed=9, height=0.4)
    points = points.astype(np.float32).astype(np.float64)
    sun = _low_sun(up, 6.0).astype(np.float32)
    reach = 5000.0
    expected = np.exp2(
        -far_cloud_depth(
            _noise(7),
            field,
            OPTICS,
            points,
            sun.astype(np.float64),
            R,
            reach,
            shift_at=lambda p: sample_cube(cube[..., 1], p),
        )
    )
    assert expected.min() < 0.9

    vdata = GeomVertexData("points", GeomVertexFormat.get_v3n3(), Geom.UH_static)
    position = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    for p, d in zip(points, sun):
        position.add_data3f(*(float(v) for v in p))
        normal.add_data3f(*(float(v) for v in d))
    prim = GeomPoints(Geom.UH_static)
    prim.add_next_vertices(len(points))
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("far_points")
    node.add_geom(geom)
    node.set_bounds(OmniBoundingVolume())
    node.set_final(True)
    atmosphere = SHADERS.parents[1] / "atmosphere/shaders/atmosphere.glsl"
    shadows = SHADERS.parents[1] / "shadows/shaders/shadows.glsl"
    include = "\n".join(
        line
        for path in (shadows, atmosphere, SHADERS / "cloud_field.glsl")
        for line in path.read_text().splitlines()
        if not line.startswith("#pragma")
    )
    vertex = f"""#version 330
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
flat out float vVisible;
{include}
void main() {{
    vVisible = exp2(-cloudFarDepth(getField(0), p3d_Vertex.xyz, p3d_Normal, {reach}));
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
    params = PTA_LVecBase4f.empty_array(MAX_LAYERS * LAYER_VEC4S)
    packed = pack_layer_params(
        field,
        extinction=field.density,
        wrap_radius=1e12,
        wrap_fade_band=0.0,
        octave_mean=float(_noise(7).mean()),
        optics=OPTICS,
        octaves=2,
    )
    for i, row in enumerate(packed):
        params[i] = LVecBase4f(*(float(v) for v in row))
    root = NodePath("far_check")
    points_np = root.attach_new_node(node)
    points_np.set_shader(Shader.make(Shader.SL_GLSL, vertex, fragment))
    bind_defaults(points_np)
    bind_shadow_defaults(points_np)
    points_np.set_shader_inputs(
        cloudNoise=build_noise_texture(_noise(7)),
        noiseSize=float(NOISE_SIZE),
        noiseOffset=Vec3(0.0, 0.0, 0.0),
        sunDir=Vec3(0.0, 0.0, 1.0),
        layerParams=params,
        uSpherical=1,
        planetRadius=R,
        weatherMap=build_weather_texture(cube),
        uCloudDebug=0,
    )
    texture = Texture("far_check")
    buffer = window.win.make_texture_buffer("far_check", size, size, texture, True)
    try:
        camera = window.makeCamera(buffer)
        camera.node().set_scene(root)
        window.graphicsEngine.render_frame()
        window.graphicsEngine.render_frame()
        ram = np.frombuffer(texture.get_ram_image_as("RGB").get_data(), np.uint8)
        rgb = ram.reshape(size, size, 3).astype(np.int64)
        got = (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]) / 16711680.0
        error = np.abs(got.ravel() - expected)
        assert np.median(error) < 2e-3
        assert np.percentile(error, 99) < 0.05
    finally:
        window.graphicsEngine.remove_window(buffer)
        root.remove_node()
