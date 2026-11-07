"""
The atmospheres' precomputed tables (world_builder.atmosphere.tables,
RENDERING_PLAN.md part B): their mappings, their values against the CPU twin,
and the GLSL's reads of them (offscreen readback; skipped without a GL
context).
"""

from __future__ import annotations

import numpy as np
import pytest
from panda3d.core import (
    CardMaker,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    LVecBase4f,
    NodePath,
    PTA_LVecBase4f,
    Shader,
    Texture,
    WindowProperties,
)

from world_builder.atmosphere import (
    MAX_ATMOSPHERES,
    SHADER_PATH,
    AtmosphereParams,
    Atmospheres,
)
from world_builder.atmosphere.tables import (
    MULTIPLE_SIZE,
    TRANSMITTANCE_SIZE,
    inscatter_reference,
    multiple_table,
    transmittance_r_mu,
    transmittance_table,
    transmittance_uv,
)

R = 600e3  # the demo's planet: its sunset_radius curvature is ~10.6
EARTH = 6371e3


def _bilinear(table: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """GL's linear filter of a (rows, columns, c) *table* at (u, v) in [0, 1],
    clamped."""
    rows, columns = table.shape[:2]
    x = np.clip(u * columns - 0.5, 0, columns - 1)
    y = np.clip(v * rows - 0.5, 0, rows - 1)
    x0 = np.minimum(np.floor(x).astype(int), columns - 2)
    y0 = np.minimum(np.floor(y).astype(int), rows - 2)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    top = table[y0, x0] * (1 - fx) + table[y0, x0 + 1] * fx
    bottom = table[y0 + 1, x0] * (1 - fx) + table[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def _random_rays(radius: float, top: float, n: int, seed: int = 1):
    """(r, mu) in the air, over the ground's horizon (rays the table holds)."""
    rng = np.random.default_rng(seed)
    r = radius + (top - radius) * rng.uniform(0.0, 1.0, n) ** 2
    horizon = -np.sqrt(np.maximum(1.0 - (radius / r) ** 2, 0.0))
    mu = horizon + (1.0 - horizon) * rng.uniform(0.0, 1.0, n) ** 1.5
    return r, mu


def test_the_transmittance_mapping_round_trips():
    """(r, mu) to (u, v) and back, from the ground to the top and from the
    zenith to the horizon."""
    top = R + 60e3
    r, mu = _random_rays(R, top, 500)
    u, v = transmittance_uv(r, mu, R, top)
    assert np.all((u > 0) & (u < 1) & (v > 0) & (v < 1))
    back_r, back_mu = transmittance_r_mu(u, v, R, top)
    np.testing.assert_allclose(back_r, r, rtol=1e-9)
    np.testing.assert_allclose(back_mu, mu, atol=1e-6)


@pytest.mark.parametrize("radius", [R, EARTH])
def test_the_transmittance_table_holds_the_twin_between_its_texels(radius):
    """The table filtered as GL does at random rays: the CPU twin's
    transmittance_to_sun (atmoDepthToward with the sunset_radius curvature)
    within 3 % of the light or 5e-3, the horizon's steep tail included (the
    worst: grazing rays a few metres over a small planet's ground, its
    curvature x10.6 steepest there, and rays grazing the ozone layer, where
    the twin filters the ozone's depth and the table its transmittance;
    99 % of rays are within 1.2 %)."""
    params = AtmosphereParams()
    atmospheres = Atmospheres()
    atmospheres.add(radius, params)
    top = radius + params.height
    table = transmittance_table(radius, params)
    assert table.shape == (TRANSMITTANCE_SIZE[1], TRANSMITTANCE_SIZE[0], 4)
    r, mu = _random_rays(radius, top, 2000)
    u, v = transmittance_uv(r, mu, radius, top)
    got = _bilinear(table[..., :3], u, v)
    points = np.zeros((len(r), 3))
    points[:, 2] = r
    want = np.array(
        [
            atmospheres.transmittance_to_sun(
                0, p, (np.sqrt(max(0.0, 1 - m * m)), 0.0, m)
            )
            for p, m in zip(points, mu)
        ]
    )
    error = np.abs(got - want)
    assert np.all(error <= np.maximum(0.03 * want, 5e-3))
    assert np.percentile(error / np.maximum(want, 1e-2), 99) < 0.012


@pytest.mark.parametrize("radius", [R, EARTH])
def test_the_ozone_depth_straight_up_is_the_tents_area(radius):
    """Straight up from the ground, the ozone's optical depth is its tent's
    area (peak x half-width x 2 / 2 = the half-width), at any curvature;
    longer toward the horizon; the table's alpha holds it in km."""
    from world_builder.atmosphere.tables import ozone_toward

    params = AtmosphereParams()
    up = float(ozone_toward(radius, 1.0, radius, params))
    assert up == pytest.approx(params.ozone_width, rel=5e-3)
    assert float(ozone_toward(radius, 0.05, radius, params)) > 5.0 * up
    table = transmittance_table(radius, params)
    assert table[0, 0, 3] == pytest.approx(params.ozone_width / 1e3, rel=5e-3)


def test_ozone_keeps_the_twilight_zenith_blue():
    """With the sun 4 degrees down, the zenith from the ground is blue over
    green with the ozone layer, about grey without it (the olive twilight
    it fixes), and noon's sky barely changes (< 5 % in blue)."""
    from dataclasses import replace

    params = AtmosphereParams()
    none = replace(params, ozone=(0.0, 0.0, 0.0))
    eye = (0.0, 0.0, R + 1300.0)
    dusk = (np.cos(np.radians(-4.0)), 0.0, np.sin(np.radians(-4.0)))
    with_ozone = inscatter_reference(R, params, eye, (0, 0, 1), 1e9, dusk)[1]
    without = inscatter_reference(R, none, eye, (0, 0, 1), 1e9, dusk)[1]
    assert with_ozone[2] > with_ozone[1]
    # green over blue: ~1 without (grey-olive), well under with
    assert with_ozone[1] / with_ozone[2] < 0.75 * without[1] / without[2]
    noon = (0.0, 0.0, 1.0)
    a = inscatter_reference(R, params, eye, (0.6, 0, 0.8), 1e9, noon)[1]
    b = inscatter_reference(R, none, eye, (0.6, 0, 0.8), 1e9, noon)[1]
    assert a[2] == pytest.approx(b[2], rel=0.05)


def test_the_multiple_scattering_table_lights_the_planets_shadow():
    """Psi_ms: none with the sun straight below; growing with the sun's
    height; still there in the planet's shadow just past the terminator (the
    dusk sky's blue-grey, no longer a flat share); blue in the air up high;
    the ground's albedo brightens it by day, not at night."""
    from dataclasses import replace

    params = AtmosphereParams()
    table = multiple_table(R, params)
    assert table.shape == (MULTIPLE_SIZE[1], MULTIPLE_SIZE[0], 3)
    assert not table.flags.writeable  # cached: shared by every caller
    assert np.all(table >= 0.0)
    np.testing.assert_allclose(table[:, 0], 0.0, atol=1e-6)  # sun at the nadir
    luminance = table.sum(axis=-1)
    assert np.all(np.diff(luminance[0, 12:]) > 0.0)  # brighter as the sun rises
    # the sun 8 degrees below the horizon at the ground: still lit from above
    column = int(round((np.sin(np.radians(-8.0)) * 0.5 + 0.5) * (MULTIPLE_SIZE[0] - 1)))
    assert luminance[0, column] > 0.0
    # what it adds to a view ray up high (times the air's scattering): blue
    added = table[-1, -1] * np.asarray(params.rayleigh)
    assert added[2] > 2.0 * added[0]
    dark = multiple_table(R, replace(params, ground_albedo=0.0))
    assert np.all(table[:, -1].sum(-1) > dark[:, -1].sum(-1))
    np.testing.assert_allclose(table[:, :8], dark[:, :8], atol=1e-9)


def test_the_reference_in_scatter_is_the_marches_limit():
    """inscatter_reference converges (2000 against 8000 steps, 0.5 %), and its
    transmittance is the air's optical depth's."""
    params = AtmosphereParams()
    origin = (0.0, 0.0, R + 1300.0)
    sun = (0.8, 0.0, 0.6)
    for direction in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.05), (-1.0, 0.3, 0.1)):
        t, s = inscatter_reference(R, params, origin, direction, 1e9, sun)
        t2, s2 = inscatter_reference(R, params, origin, direction, 1e9, sun, steps=8000)
        np.testing.assert_allclose(s, s2, rtol=5e-3)
        np.testing.assert_allclose(t, t2, rtol=1e-4)
        assert np.all(s > 0.0) and np.all((t > 0.0) & (t < 1.0))


# ── the GLSL's reads, read back ───────────────────────────────────────────────


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


def _evaluate(window, tmp_path, atmospheres, expression, inputs, prelude=""):
    """*expression* (a vec3 GLSL expression of ``vec4 q``, an input) for each
    of *inputs* ((n, 4), n <= 256), drawn in a float target, read back.
    *prelude*: GLSL put before ``main`` (helpers, uniforms)."""
    import zlib

    n = len(inputs)
    # A file per shader: Panda caches Shader.load by its files' names.
    key = zlib.crc32(f"{prelude}{expression}{n}".encode())
    fragment = tmp_path / f"evaluate_{key:08x}.frag"
    fragment.write_text(
        f"""#version 330
#pragma include "{SHADER_PATH}/atmosphere.glsl"
{prelude}
uniform vec4 uInputs[{n}];
out vec4 colour;
void main() {{
    vec4 q = uInputs[int(gl_FragCoord.x)];
    colour = vec4({expression}, 1.0);
}}
"""
    )
    vertex = tmp_path / "evaluate.vert"
    vertex.write_text(
        """#version 330
in vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;
void main() { gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex; }
"""
    )
    maker = CardMaker("evaluate")
    maker.set_frame(-1.0, 1.0, -1.0, 1.0)
    root = NodePath("evaluate")
    card = root.attach_new_node(maker.generate())
    card.set_shader(
        Shader.load(Shader.SL_GLSL, vertex=str(vertex), fragment=str(fragment))
    )
    from world_builder.shadows import bind_defaults as shadow_defaults

    shadow_defaults(card)
    atmospheres.bind(card)
    values = PTA_LVecBase4f.empty_array(n)
    for i, row in enumerate(inputs):
        values[i] = LVecBase4f(*(float(x) for x in row))
    card.set_shader_input("uInputs", values)
    target = Texture("evaluate")
    props = FrameBufferProperties()
    props.set_rgba_bits(32, 32, 32, 32)
    props.set_float_color(True)
    engine = window.graphicsEngine
    buffer = engine.make_output(
        window.pipe,
        "evaluate",
        -100,
        props,
        WindowProperties.size(n, 1),
        GraphicsPipe.BF_refuse_window,
        window.win.get_gsg(),
        window.win,
    )
    buffer.add_render_texture(target, GraphicsOutput.RTM_copy_ram)
    try:
        camera = window.makeCamera2d(buffer)
        camera.reparent_to(root)
        engine.render_frame()
        engine.render_frame()
        image = np.frombuffer(target.get_ram_image(), np.float32)
        return image.reshape(-1, 4)[:n, [2, 1, 0]]  # BGRA
    finally:
        engine.remove_window(buffer)
        root.remove_node()


def test_the_shader_reads_the_transmittance_table_as_the_twin(window, tmp_path):
    """atmoTransmittance for both atmospheres (a small planet, Earth): the CPU
    twin's transmittance_to_sun within 3 % or 5e-3 (the table's, above, with
    half floats); 0 into the ground."""
    atmospheres = Atmospheres()
    atmospheres.add(R, AtmosphereParams(), (0.0, 0.0, 0.0))
    atmospheres.add(EARTH, AtmosphereParams(mie=8e-6), (1e9, 0.0, 0.0))
    rows, want = [], []
    for index, radius in enumerate((R, EARTH)):
        top = radius + 60e3
        r, mu = _random_rays(radius, top, 100, seed=index + 3)
        for ri, mi in zip(r, mu):
            rows.append((index, ri, mi, 0.0))
            centre = atmospheres._bodies[index][2]
            want.append(
                atmospheres.transmittance_to_sun(
                    index,
                    centre + (0.0, 0.0, ri),
                    (np.sqrt(max(0.0, 1 - mi * mi)), 0.0, mi),
                )
            )
    # into the ground: from 1 km up, 30 degrees down
    rows.append((0, R + 1e3, -0.5, 0.0))
    want.append(np.zeros(3))
    got = _evaluate(
        window,
        tmp_path,
        atmospheres,
        "atmoTransmittance(int(q.x + 0.5), q.y, q.z)",
        np.array(rows),
    )
    want = np.array(want)
    error = np.abs(got - want)
    bound = np.maximum(0.03 * want, 5e-3)
    worst = int(np.argmax((error - bound).max(axis=1)))
    assert np.all(error <= bound), (rows[worst], got[worst], want[worst])
    assert MAX_ATMOSPHERES == 2


_ALONG = """
uniform vec3 uSun;
vec3 inscatterOf(vec4 q) {
    vec3 transmittance, inscatter;
    atmosphereAlong(normalize(q.xyz), q.w, uSun, 16, transmittance, inscatter);
    return inscatter;
}
vec3 transmittanceOf(vec4 q) {
    vec3 transmittance, inscatter;
    atmosphereAlong(normalize(q.xyz), q.w, uSun, 16, transmittance, inscatter);
    return transmittance;
}
"""


@pytest.mark.parametrize("elevation", [60.0, 8.0, -4.0])
def test_the_shaders_air_is_the_reference(window, tmp_path, elevation):
    """atmosphereAlong at 16 samples (the table's sunlight, the table's
    multiple scattering) against inscatter_reference's fine march, from
    1.3 km up, at noon, at a low sun and in twilight, over the sky's
    directions and toward the ground: within 6 % of the colour or 2 % of
    the noon sky's brightness (measured: 3 % at most, twilight 1 %; the
    shader's own march: 16 midpoint samples, the lit share of each)."""
    from math import cos, radians, sin

    params = AtmosphereParams()
    atmospheres = Atmospheres()
    atmospheres.add(R, params)
    eye = np.array([0.0, 0.0, R + 1300.0])
    atmospheres.update(eye)
    e = radians(elevation)
    sun = np.array([cos(e), 0.0, sin(e)])
    rng = np.random.default_rng(5)
    dirs = rng.normal(size=(60, 3))
    dirs[:, 2] = np.abs(dirs[:, 2]) * 0.6  # mostly the sky, some low
    dirs = np.concatenate([dirs, [(1, 0, 0.02), (-1, 0, 0.02), (0, 0, 1)]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    distances = np.full(len(dirs), 1e9)
    distances[:10] = rng.uniform(2e3, 40e3, 10)  # to things in the air
    rows = np.concatenate([dirs, distances[:, None]], axis=1)
    want = np.array(
        [
            inscatter_reference(R, params, eye, d, dist, sun)[1]
            for d, dist in zip(dirs, distances)
        ]
    )

    def run(expression):
        return _evaluate_with_sun(
            window, tmp_path, atmospheres, expression, rows, _ALONG, sun
        )

    got = run("inscatterOf(q)")
    scale = 0.02 * inscatter_reference(R, params, eye, (0, 0, 1), 1e9, (0, 0, 1))[1]
    error = np.abs(got - want)
    bound = np.maximum(0.06 * want, scale)
    # Within ~10 degrees of the sun, Mie's forward peak at 16 midpoint samples
    # (the shader's own march, not the tables): up to 15 % at the disc.
    near_sun = rows[:, :3] @ sun > 0.985
    bound[near_sun] = np.maximum(0.2 * want, scale)[near_sun]
    worst = int(np.argmax((error - bound).max(axis=1)))
    assert np.all(error <= bound), (rows[worst], got[worst], want[worst])
    transmittance = run("transmittanceOf(q)")
    want_t = np.array(
        [
            inscatter_reference(R, params, eye, d, dist, sun)[0]
            for d, dist in zip(dirs, distances)
        ]
    )
    np.testing.assert_allclose(transmittance, want_t, rtol=0.03, atol=2e-3)


def _evaluate_with_sun(window, tmp_path, atmospheres, expression, rows, prelude, sun):
    """_evaluate with ``uSun`` set (bound with the atmospheres' inputs)."""
    original = atmospheres.bind

    def bind(node):
        original(node)
        node.set_shader_input("uSun", tuple(float(v) for v in sun))

    atmospheres.bind = bind
    try:
        return _evaluate(window, tmp_path, atmospheres, expression, rows, prelude)
    finally:
        atmospheres.bind = original
