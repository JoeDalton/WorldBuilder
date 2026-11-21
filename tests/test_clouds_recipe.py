"""
The recipe genera (noise.GenusField's recipe, genusRecipeDensity in
cloud_field.glsl): the Worley cells, the recipe's packing into a genus's rows
14-15, its CPU mirror (noise.recipe_density_on_sphere), its coverage
calibration (noise.recipe_column_peaks, field_setup.recipe_field), coverage
cube B, and a GPU smoke test.
"""

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.genera import GENUS_PRESETS
from world_builder.clouds.noise import (
    COVERAGE_SOURCES,
    GENUS_VEC4S,
    LAYER_VEC4S,
    RECIPE_CELLS,
    RECIPE_EROSION_OFFSET,
    RECIPE_EROSION_SCALE,
    RECIPE_LENS,
    RECIPE_ON,
    RECIPE_RESOLVED,
    RECIPE_WORLEY,
    WORLEY_CELLS,
    GenusField,
    cell_means,
    cell_spreads,
    pack_genus_params,
    pack_layer_params,
    pack_recipe_cells,
    pack_recipe_frame,
    recipe_density_on_sphere,
    recipe_flags,
    recipe_position,
    recipe_value,
    sample_cells,
    worley_points,
    worley_volume,
)

R = 600e3
SHADERS = Path(__file__).parents[1] / "src/world_builder/clouds/shaders"


@pytest.fixture(scope="module")
def cells():
    return worley_volume()


def _genus(field=None, **recipe):
    field = field or replace(PRESETS[CloudType.STRATUS].field, threshold=(0.45, 0.55))
    return GenusField(
        "rcp", field, PRESETS[CloudType.STRATUS].optics, ("A", 1), 1.0, **recipe
    )


# ── The Worley volume ────────────────────────────────────────────────────────


def test_worley_cells_wrap_continuously_across_the_volume_edge(cells):
    """A tileable volume: across its edge (last texel to first, wrapped) F1
    steps no more than between any two neighbours inside it."""
    f1 = cells[..., 0]
    for axis in range(3):
        inside = np.abs(np.diff(f1, axis=axis)).max()
        first = np.take(f1, 0, axis=axis)
        last = np.take(f1, -1, axis=axis)
        assert np.abs(first - last).max() <= inside * 1.001 + 1e-6


def test_worley_distances_are_in_cells_f1_under_f2(cells):
    assert cells.shape == (64, 64, 64, 2) and cells.dtype == np.float32
    f1, f2 = cells[..., 0], cells[..., 1]
    assert f1.min() >= 0.0 and np.all(f2 >= f1)
    assert f1.min() < 0.1  # texels by a point
    assert 0.6 < f1.max() < 1.2  # about [0, 1]: one point a cell
    assert 0.3 < f1.mean() < 0.6


def test_worley_volume_size_must_hold_whole_cells():
    with pytest.raises(ValueError):
        worley_volume(size=20, cells=8)


# ── Packing ──────────────────────────────────────────────────────────────────


def test_a_genus_without_the_recipe_packs_zeros_in_rows_14_and_15():
    """Neutral defaults, and any recipe knob on a genus without the recipe:
    rows 0-13 the field's and coverage row, 14-15 (and row 10's x) zero."""
    plain = _genus()
    knobs = _genus(noise_kind="worley", cell_fill=0.7, stretch=3.0, erosion=(1.0, 2.0))
    expected = pack_layer_params(plain.field, 0.02, optics=plain.optics, octaves=4)
    for genus in (plain, knobs):
        rows = pack_genus_params(genus, 0.02, octaves=4, shift_row=2)
        np.testing.assert_array_equal(rows[:LAYER_VEC4S], expected)
        assert tuple(rows[LAYER_VEC4S]) == (COVERAGE_SOURCES["A"], 1.0, 2.0, 1.0)
        assert not rows[14:].any() and rows[10, 0] == 0.0


def test_a_recipe_genus_packs_its_knobs_in_rows_14_and_15():
    genus = _genus(
        recipe=True, noise_kind="worley", cell_size=1200.0, cell_fill=0.75,
        stretch=3.0, vertical_stretch=0.5, profile=(0.2, 0.3, True),
        erosion=(-1.5, 2.0), erosion_base=0.4,
    )  # fmt: skip
    rows = pack_genus_params(genus, 0.02)
    flags = RECIPE_ON | RECIPE_WORLEY | RECIPE_LENS
    np.testing.assert_allclose(rows[14], (flags, 1200.0, 0.75, 3.0))
    np.testing.assert_allclose(rows[15], (0.5, 0.2, 0.3, -1.5))
    assert rows[10, 0] == 2.0  # the erosion's skin
    assert rows[12, 3] == pytest.approx(0.4)  # its base (no cell octave's power)
    # The field's other rows as without the recipe
    plain = pack_genus_params(_genus(), 0.02)
    rows[10, 0] = 0.0
    rows[12, 3] = plain[12, 3]
    np.testing.assert_array_equal(rows[:14], plain[:14])
    # A value-noise recipe: no cells bit, whatever the fill
    value = pack_genus_params(_genus(recipe=True, cell_fill=0.5), 0.02)
    assert value[14, 0] == RECIPE_ON


def test_the_recipe_frame_is_the_unit_east_and_the_cells_mean():
    np.testing.assert_allclose(pack_recipe_frame((0.0, 2.0, 0.0), 0.4), (0, 1, 0, 0.4))


def test_recipe_knobs_are_checked():
    with pytest.raises(ValueError):
        _genus(noise_kind="perlin")
    with pytest.raises(ValueError):
        _genus(stretch=0.5)
    with pytest.raises(ValueError):
        _genus(cell_fill=1.5)
    with pytest.raises(ValueError):
        _genus(erosion_base=1.5)
    with pytest.raises(ValueError):  # row 12's w is the erosion's base
        _genus(replace(_flat(), cell_weight=0.3), recipe=True)


# ── The CPU mirror ───────────────────────────────────────────────────────────


def _flat(**overrides):
    """A field for hand-computed points: no offset, no lift, a slab 1-7 km."""
    field = replace(
        PRESETS[CloudType.STRATUS].field,
        slab=(1000.0, 6000.0), offset=(0.0, 0.0, 0.0), top_erosion=0.0,
        noise_scale=(0.001, 0.001, 0.001), threshold=(0.5, 0.6),
    )  # fmt: skip
    return replace(field, **overrides)


def test_a_worley_genus_is_densest_at_a_feature_point(cells):
    """At a feature point (planet frame = recipe coordinates: no offset, no
    stretch) F1 is ~0, the base value ~1 (all cells), the density 1; halfway
    to the next cell's point the value is lower."""
    from world_builder.clouds.field_setup import _noise

    volume = _noise(7)
    L = 1500.0
    genus = _genus(
        _flat(), recipe=True, noise_kind="worley", cell_size=L, cell_fill=1.0
    )
    near = np.array([0.0, 0.0, R + 4000.0]) / L
    home = np.floor(near).astype(int)
    point = home + (
        worley_points()[tuple((home % WORLEY_CELLS)[::-1])] - home % WORLEY_CELLS
    )
    p = point * L
    assert 1000.0 < np.linalg.norm(p) - R < 7000.0
    np.testing.assert_allclose(recipe_position(genus, p, R), p)
    f1 = float(sample_cells(cells, point))
    assert f1 < 0.12
    value = recipe_value(volume, cells, genus, p)
    assert value == pytest.approx(1.0 - f1, abs=1e-6)
    plane = np.zeros((6, 2, 2))
    assert recipe_density_on_sphere(volume, cells, plane, genus, p, R) == pytest.approx(
        1.0
    )
    # Toward the next cell's point, halfway: farther from both points
    other = point + np.array([1.0, 0.0, 0.0])
    between = recipe_value(volume, cells, genus, 0.5 * (p + other * L))
    assert between < value - 0.2


@pytest.mark.parametrize(
    "profile, h, expected",
    [
        ((0.2, 0.3, False), 0.1, 0.5),  # half way up the soft bottom
        ((0.2, 0.3, False), 0.85, 0.5),  # half way down the soft top
        ((0.2, 0.3, False), 0.5, 1.0),
        ((0.0, 0.0, True), 0.25, 0.75),  # the lens, 4h(1 - h)
        ((0.0, 0.0, True), 0.5, 1.0),
    ],
)
def test_the_profile_scales_a_solid_core(cells, profile, h, expected):
    from world_builder.clouds.field_setup import _noise

    genus = _genus(_flat(threshold=(-10.0, -9.0)), recipe=True, profile=profile)
    p = np.array([0.3, 0.5, 0.8])
    p = p / np.linalg.norm(p) * (R + 1000.0 + h * 6000.0)
    density = recipe_density_on_sphere(
        _noise(7), cells, np.zeros((6, 2, 2)), genus, p, R
    )
    assert density == pytest.approx(expected, abs=1e-6)


def test_erosion_spares_the_core_and_moves_the_edge(cells):
    from world_builder.clouds.field_setup import _noise

    volume = _noise(7)
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(400, 3))
    p = dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * (R + 4000.0)
    plane = np.zeros((6, 2, 2))
    core = _genus(_flat(threshold=(-10.0, -9.0)), recipe=True)
    eroded = replace(core, erosion=(2.0, 1.0))
    np.testing.assert_array_equal(
        recipe_density_on_sphere(volume, cells, plane, eroded, p, R), 1.0
    )
    # The edge: the window around the points' values
    value = recipe_value(volume, cells, core, recipe_position(core, p, R))
    lo = float(np.median(value)) - 0.02
    edge = replace(core, field=replace(core.field, threshold=(lo, lo + 0.04)))
    plain = recipe_density_on_sphere(volume, cells, plane, edge, p, R)
    for amount in (2.0, -2.0):  # billowy, wispy
        moved = recipe_density_on_sphere(
            volume, cells, plane, replace(edge, erosion=(amount, 1.0)), p, R
        )
        assert np.abs(moved - plain).max() > 0.2


def test_the_erosion_base_keeps_the_erosion_near_the_base(cells):
    """erosion_base: the erosion whole at the slab's base, none above that
    share of it (a stratus's fractus: ragged low down, its outline above
    as it is)."""
    from world_builder.clouds.field_setup import _noise
    from world_builder.clouds.noise import recipe_erosion_weight

    genus = _genus(_flat(), recipe=True, erosion=(-2.0, 1.0), erosion_base=0.4)
    np.testing.assert_allclose(
        recipe_erosion_weight(genus, np.array([0.0, 0.2, 0.4, 0.8])),
        (1.0, 0.5, 0.0, 0.0),
    )
    assert np.all(recipe_erosion_weight(replace(genus, erosion_base=0.0), 0.7) == 1)
    volume = _noise(7)
    rng = np.random.default_rng(4)
    dirs = rng.normal(size=(400, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    plane = np.zeros((6, 2, 2))
    for h, moves in ((0.05, True), (0.6, False)):
        p = dirs * (R + 1000.0 + h * 6000.0)
        value = recipe_value(volume, cells, genus, recipe_position(genus, p, R))
        lo = float(np.median(value)) - 0.02
        edge = replace(genus, field=replace(genus.field, threshold=(lo, lo + 0.04)))
        plain = recipe_density_on_sphere(
            volume, cells, plane, replace(edge, erosion=(0.0, 1.0)), p, R
        )
        moved = recipe_density_on_sphere(volume, cells, plane, edge, p, R)
        assert (np.abs(moved - plain).max() > 0.2) == moves, h


def test_the_stretch_squeezes_along_east_and_the_height():
    genus = _genus(_flat(), recipe=True, stretch=4.0, vertical_stretch=2.0,
                   east=(0.0, 1.0, 0.0))  # fmt: skip
    up = np.array([1.0, 0.0, 0.0])
    p = up * (R + 2000.0)
    q = recipe_position(genus, p, R)
    along = recipe_position(genus, p + np.array([0.0, 400.0, 0.0]), R) - q
    across = recipe_position(genus, p + np.array([0.0, 0.0, 400.0]), R) - q
    higher = recipe_position(genus, p + up * 400.0, R) - q
    assert along[1] == pytest.approx(100.0, rel=1e-3)
    assert across[2] == pytest.approx(400.0, rel=1e-3)
    assert higher[0] == pytest.approx(200.0, rel=1e-3)
    # About the slab's base: there the height holds
    base = up * (R + 1000.0)
    np.testing.assert_allclose(recipe_position(genus, base, R), base, atol=1e-6)


# ── Coverage calibration ─────────────────────────────────────────────────────


def _cells_genus():
    """A synthetic recipe genus: Worley cells 1.5 km, a slab 1-2 km."""
    field = replace(PRESETS[CloudType.STRATUS].field, slab=(1000.0, 1000.0))
    return _genus(
        field, recipe=True, noise_kind="worley", cell_size=1500.0, cell_fill=0.8,
        stretch=2.0, east=(0.0, 1.0, 0.0), profile=(0.1, 0.3, False),
    )  # fmt: skip


@pytest.mark.parametrize("coverage", [0.3, 0.7])
def test_a_recipe_genus_coverage_is_what_its_shift_table_asks(cells, coverage):
    """The recipe's own column peaks calibrate its shift: the share of
    columns (other than the calibration's) holding cloud anywhere in the
    slab is within 5 % of the coverage asked."""
    from world_builder.clouds.field_setup import _noise, recipe_field
    from world_builder.clouds.weather import coverage_to_shift, shift_table

    genus, peaks = recipe_field(_cells_genus(), 7, R)
    lo, hi = genus.field.threshold
    assert hi > lo
    shift = float(coverage_to_shift(coverage, shift_table(peaks, lo)))
    rng = np.random.default_rng(5)
    dirs = rng.normal(size=(3000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    h = (np.arange(64) + 0.5) / 64
    points = dirs[:, None, :] * (R + 1000.0 + h * 1000.0)[None, :, None]
    plane = np.full((6, 2, 2), shift)
    density = recipe_density_on_sphere(_noise(7), cells, plane, genus, points, R)
    measured = float(np.mean((density > 0.0).any(axis=1)))
    assert measured == pytest.approx(coverage, abs=0.05)


# ── The shaders ──────────────────────────────────────────────────────────────


def test_the_shaders_share_the_recipe_constants():
    field = (SHADERS / "cloud_field.glsl").read_text()
    assert "uniform sampler3D cloudCells;" in field
    assert "uniform samplerCube genusCoverB;" in field
    assert float(re.search(r"const float WORLEY_CELLS = ([\d.]+);", field)[1]) == (
        WORLEY_CELLS
    )
    assert float(
        re.search(r"const float RECIPE_EROSION_SCALE = ([\d.]+);", field)[1]
    ) == (RECIPE_EROSION_SCALE)
    offset = re.search(r"const vec3 RECIPE_EROSION_OFFSET = vec3\(([^)]*)\);", field)[1]
    assert tuple(float(v) for v in offset.split(",")) == RECIPE_EROSION_OFFSET
    resolved = re.search(r"const vec2 RECIPE_RESOLVED = vec2\(([^)]*)\);", field)[1]
    assert tuple(float(v) for v in resolved.split(",")) == RECIPE_RESOLVED
    for name, value in (
        ("ON", RECIPE_ON),
        ("WORLEY", RECIPE_WORLEY),
        ("LENS", RECIPE_LENS),
        ("CELLS", RECIPE_CELLS),
    ):
        assert int(re.search(rf"const int RECIPE_{name} = (\d+);", field)[1]) == value
    # The recipe in the summed depths (the bakes'), and the view's
    levels = field[field.index("float skyShadowLevels(") :]
    levels = levels[: levels.index("\n}\n")]
    assert "#ifdef CLOUD_RECIPE_SHADOWS" in levels and "genusRecipeDensity(" in levels
    march = (SHADERS / "cloud_march.glsl").read_text()
    density = march[march.index("float genusDensity(") :]
    assert "genusRecipeDensity(" in density[: density.index("\n}\n")]
    # Compiled only where it is needed: the view's variant with recipe
    # genera, the summed depths' bakes; not the shared shadow march.
    assert "#define CLOUD_RECIPES" in (SHADERS / "cloud_march_recipes.frag").read_text()
    assert "CLOUD_RECIPES" not in (SHADERS / "cloud_march.frag").read_text()
    for name in ("cloud_light_bake.frag", "cloud_shadow_map.frag"):
        assert "#define CLOUD_RECIPE_SHADOWS" in (SHADERS / name).read_text()


# ── A recipe genus on the GPU (test-only presets) ────────────────────────────

CELLS_A = "recipe_cells"
VEIL_B = "recipe_veil"


@pytest.fixture
def recipe_genera(monkeypatch):
    """Two synthetic recipe genera, a Worley one on cube A's second channel
    and a stretched value-noise lens on cube B's first, each a feature of
    its own where the stratus deck is, inside the convective slab."""
    from world_builder.clouds.genera import (
        DEFAULT_FEATURES,
        Feature,
        GeneraSpec,
        GenusPreset,
    )
    from world_builder.clouds.weather import WeatherSpec
    from world_builder.terrain.planet import PlanetSpec

    monkeypatch.setitem(
        GENUS_PRESETS,
        CELLS_A,
        GenusPreset(
            CloudType.STRATUS, ("A", 1), far_shadow=1.0, slab=(3000.0, 1000.0),
            recipe=dict(noise_kind="worley", cell_size=1500.0, cell_fill=0.8,
                        profile=(0.1, 0.3, False), erosion=(1.0, 1.0)),
        ),
    )  # fmt: skip
    monkeypatch.setitem(
        GENUS_PRESETS,
        VEIL_B,
        GenusPreset(
            CloudType.STRATUS, ("B", 0), far_shadow=1.0, slab=(4500.0, 800.0),
            recipe=dict(stretch=3.0, profile=(0.0, 0.0, True), erosion=(-1.0, 1.0)),
            field=dict(edge_softness=3.0),
        ),
    )  # fmt: skip
    spec = PlanetSpec(name="clouds", radius=R)
    deck = next(f for f in DEFAULT_FEATURES if f.genus == "stratus")
    features = DEFAULT_FEATURES + tuple(
        replace(deck, genus=name) for name in (CELLS_A, VEIL_B)
    )
    assert all(isinstance(f, Feature) for f in features)
    return GeneraSpec(
        WeatherSpec(spec, seed=11), features=features,
        genera=("convective", "stratus", CELLS_A, VEIL_B),
    )  # fmt: skip


def test_cube_b_holds_its_genus_coverage(recipe_genera):
    """Each genus's channel holds its own features' coverage
    (genera.genus_coverage), the stratus's in genus_planes' third too."""
    from world_builder.clouds.genera import coverage_cube, genus_coverage, genus_planes

    planes = genus_planes(recipe_genera, 128)
    cube_a = coverage_cube(recipe_genera, "A", 128)
    cube_b = coverage_cube(recipe_genera, "B", 128)
    np.testing.assert_array_equal(cube_a[..., 0], planes[..., 2])
    own = genus_coverage(recipe_genera, CELLS_A, 128)
    np.testing.assert_array_equal(cube_a[..., 1], own)
    own = genus_coverage(recipe_genera, VEIL_B, 128)
    np.testing.assert_array_equal(cube_b[..., 0], own)
    assert cube_a[..., 1].max() > 0.5 and cube_b[..., 0].max() > 0.5
    assert not cube_b[..., 1:].any()


def test_a_preset_recipe_reaches_its_genus_field(recipe_genera, showbase):
    """The march builds a preset's recipe (its knobs, its field's
    overrides) into the genus's field, its threshold the recipe's own."""
    from world_builder.clouds.field_setup import _cells
    from world_builder.clouds.march import CloudMarch
    from world_builder.terrain.chart import TangentFrame
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    try:
        march = CloudMarch(showbase.render, R, executor, altitude=2200.0,
                           genera=recipe_genera)  # fmt: skip
        cells_genus = march._genera[2].recipe
        veil = march._genera[3].recipe
        assert march._genera[0].recipe is None  # the convective clouds
        assert march._genera[1].recipe.noise_kind == "value"  # the stratus
        assert (
            cells_genus.recipe and cells_genus.worley and cells_genus.cell_fill == 0.8
        )
        assert veil.stretch == 3.0 and veil.field.edge_softness == 3.0
        np.testing.assert_allclose(
            veil.east, TangentFrame.at(*recipe_genera.focus).east
        )
        rows = np.array(
            [tuple(march._layer_params[i]) for i in range(11 * GENUS_VEC4S)]
        )
        rows = rows.reshape(11, GENUS_VEC4S, 4)
        assert rows[2, 14, 0] == RECIPE_ON | RECIPE_WORLEY
        assert rows[3, 14, 0] == RECIPE_ON | RECIPE_LENS and rows[3, 14, 3] == 3.0
        assert tuple(rows[3, 13, :2]) == (COVERAGE_SOURCES["B"], 0.0)
        assert not rows[0, 14:].any()  # the convective clouds: no recipe
        assert rows[1, 14, 0] == RECIPE_ON  # the stratus: value noise
        # The deck: no recipe; its row 14 the closed cells' mean F2 - F1, then
        # the spreads of F1 and F2 - F1 (the march's far edges' widening)
        np.testing.assert_allclose(
            rows[10, 14],
            (cell_means(_cells())[1], *cell_spreads(_cells()), 0),
            rtol=1e-6,
        )
        np.testing.assert_allclose(rows[10, 15, :3], veil.east, atol=1e-6)
        assert rows[3, 0, 3] == pytest.approx(veil.field.threshold[0])
        march.remove()
    finally:
        executor.shutdown()


def _pixels(texture):
    """A float texture's RAM image as (n, components) float32."""
    from panda3d.core import Texture

    dtype = {Texture.T_float: np.float32, Texture.T_half_float: np.float16}[
        texture.get_component_type()
    ]
    image = np.frombuffer(texture.get_ram_image(), dtype).astype(np.float32)
    return image.reshape(-1, texture.get_num_components())


@pytest.fixture
def window(showbase):
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(64, 64))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


def test_recipe_genera_render(window, recipe_genera):
    """Two recipe genera (cube A's second channel, cube B's first) with the
    convective clouds and the stratus: looking up from under the stratus
    features the march draws cloud, nothing that is not finite."""
    from panda3d.core import Point3

    from world_builder.clouds.cloud import CloudSettings
    from world_builder.clouds.planet import PlanetClouds
    from world_builder.depth import SplitDepthCameras
    from world_builder.terrain.chart import TangentFrame
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=64)
    clouds = PlanetClouds(
        window.render, R, executor, altitude=2200.0,
        settings=CloudSettings(octave_bands=2), genera=recipe_genera,
    )  # fmt: skip
    depth = SplitDepthCameras(window, near=1.0)
    camera = window.camera
    pos, hpr = camera.get_pos(), camera.get_hpr()
    try:
        march = clouds.volume
        assert march.genera_count == 4
        clouds.use_split_depth(window, depth)
        march._finish()
        for name in ("genusCoverA", "genusCoverB", "cloudCells"):
            assert name in march._textures or name in march._inputs
        frame = TangentFrame.at(-2.86, 0.0)
        here = frame.up * (R + 600.0)
        camera.set_pos(0.0, 0.0, 0.0)
        camera.look_at(Point3(*frame.up))
        clouds.set_sun(frame.up + 0.5 * frame.east)
        (layer,) = march._layers
        from panda3d.core import Shader

        shader = layer.card.get_shader()
        assert "cloud_march_recipes.frag" in str(
            shader.get_filename(Shader.ST_fragment)
        )
        gsg = window.win.get_gsg()
        alphas = []
        for _ in range(6):
            depth.update(*SplitDepthCameras.planet_ranges(R, R + 600.0))
            clouds.update(here)
            window.graphicsEngine.render_frame()
            window.graphicsEngine.extract_texture_data(layer.colour, gsg)
            image = _pixels(layer.colour)
            assert np.all(np.isfinite(image))
            alphas.append(image[:, 3].max())
        assert max(alphas) > 0.05
        # The recipe genera alone (the others hidden) still draw cloud
        for genus in (0, 1):
            march.set_genus_visible(genus, False)
        alphas = []
        for _ in range(6):
            clouds.update(here)
            window.graphicsEngine.render_frame()
            window.graphicsEngine.extract_texture_data(layer.colour, gsg)
            image = _pixels(layer.colour)
            assert np.all(np.isfinite(image))
            alphas.append(image[:, 3].max())
        assert max(alphas) > 0.05
    finally:
        camera.set_pos(pos)
        camera.set_hpr(hpr)
        clouds.remove()
        depth.destroy()
        executor.shutdown()


# ── Closed cells ─────────────────────────────────────────────────────────────


def test_closed_cells_are_zero_on_the_walls_and_highest_at_the_points(cells):
    """noise_kind "cells": F2 - F1, 0 where two points are equally near (the
    wall between their cells), highest at a point; its own flag and mean."""
    from world_builder.clouds.noise import sample_cell_edges

    edges = cells[..., 1] - cells[..., 0]
    assert edges.min() < 0.02 and 0.01 < float((edges < 0.05).mean()) < 0.3
    points = worley_points().reshape(-1, 3)
    for a in points[:: len(points) // 12]:
        # A peak at the point (a cone, rounded by the texels): lower a
        # quarter of a cell to each side on average
        at = float(sample_cell_edges(cells, a[None])[0])
        sides = np.concatenate([a + np.eye(3) * 0.25, a - np.eye(3) * 0.25])
        assert float(sample_cell_edges(cells, sides).mean()) < at - 0.05
    genus = _genus(_flat(), recipe=True, noise_kind="cells", cell_fill=0.8)
    assert recipe_flags(genus) == RECIPE_ON | RECIPE_WORLEY | RECIPE_CELLS
    f1, edges = cell_means(cells)
    assert edges == pytest.approx(float((cells[..., 1] - cells[..., 0]).mean()))
    assert f1 == pytest.approx(float(cells[..., 0].mean()))
    # Far away (unresolved) the cells fade to their mean
    m = np.array([[1234.0, 567.0, 89.0]])
    far = recipe_value(np.full((4, 4, 4), 0.5), cells, genus, m, footprint=1e6)
    fbm = 0.5 * sum(genus.field.octave_weights)  # each octave its mean
    assert float(far[0]) == pytest.approx(0.2 * fbm + 0.8 * edges, abs=1e-6)
    np.testing.assert_allclose(pack_recipe_cells(0.25), (0.25, 0, 0, 0))
    np.testing.assert_allclose(pack_recipe_cells(0.25, 0.1, 0.2), (0.25, 0.1, 0.2, 0))
    f1_sigma, edges_sigma = cell_spreads(cells)
    assert f1_sigma == pytest.approx(float(cells[..., 0].std()))
    assert edges_sigma == pytest.approx(float((cells[..., 1] - cells[..., 0]).std()))


# ── The presets: stratus and stratocumulus ───────────────────────────────────


def _preset(name):
    """A preset genus as the march builds it (field_setup.genus_field)."""
    from world_builder.clouds.field_setup import genus_field
    from world_builder.clouds.genera import GeneraSpec, genus_specs
    from world_builder.clouds.weather import WeatherSpec
    from world_builder.terrain.planet import PlanetSpec

    spec = GeneraSpec(WeatherSpec(PlanetSpec(name="clouds", radius=R)))
    (base, thickness) = next(s[2:4] for s in genus_specs(spec) if s[0] == name)
    return genus_field(name, base, thickness, 7, R)


STRATIFORM = ["stratus", "stratocumulus", "nimbostratus", "altostratus", "altocumulus"]


@pytest.mark.parametrize("name", STRATIFORM)
def test_the_stratiform_presets_are_recipe_genera(name):
    preset, field, peaks, genus = _preset(name)
    assert genus is not None and genus.recipe and genus.field == field
    assert genus.coverage == GENUS_PRESETS[name].coverage
    assert field.threshold[1] > field.threshold[0]
    if name == "stratus":
        # A low-frequency fbm sheet, three times longer along east, a soft
        # top, wispy erosion low down only
        assert genus.noise_kind == "value" and genus.stretch == pytest.approx(3.0)
        assert field.feature_size[0] >= 10e3 and field.slab[0] == 900.0
        assert genus.profile[1] == pytest.approx(0.3) and genus.erosion[0] < 0
        assert 0.0 < genus.erosion_base < 1.0
    elif name == "stratocumulus":
        # Closed cells 1.5 km wide over 0.8-1.8 km, domed (the threshold
        # rising to the top), billowy
        assert genus.noise_kind == "cells" and genus.worley
        assert genus.cell_size == 1500.0 and 0.6 <= genus.cell_fill <= 0.8
        assert field.slab == (800.0, 1000.0) and field.top_erosion > 0.2
        assert genus.profile[0] < 0.1 and genus.erosion[0] > 0
    elif name == "nimbostratus":
        # A thick top-hat over 0.6-5 km spanning the low and middle étages:
        # fbm of 20 km features or more, three times longer along east,
        # wispy erosion (a ragged base) low down only, its ends soft but
        # its body full; denser than the stratus (dark, the sun blotted out)
        assert genus.noise_kind == "value" and not genus.worley
        assert field.slab == (600.0, 4400.0) and field.feature_size[0] >= 20e3
        assert genus.stretch == pytest.approx(3.0)
        assert genus.erosion[0] < 0 and 0.0 < genus.erosion_base <= 0.25
        assert max(genus.profile[:2]) <= 0.2 and not genus.profile[2]
        assert field.density > 1.5 * PRESETS[CloudType.STRATUS].field.density
    elif name == "altocumulus":
        # A thin lens at 3.5-3.8 km (inside the nimbostratus's, under the
        # altostratus's base) of closed Worley cells: rounded patches with
        # small clear gaps, each 1-5 degrees across from the ground (WMO:
        # an element is about 1.2 cells, so cells of 250-500 m at 3.5 km),
        # flatter than wide, billowy erosion at a sixth of the cells.
        # Unstretched (stretched, the erosion's cells, read in the recipe's
        # squeezed coordinates, saw its rim into teeth), its last octave
        # dropped (the far levels' cut); coverage on cube B
        assert genus.noise_kind == "cells" and genus.worley
        assert 250.0 <= genus.cell_size <= 500.0 and genus.cell_fill >= 0.6
        assert field.slab == (3500.0, 300.0) and genus.coverage == ("B", 0)
        assert field.slab[1] < genus.cell_size
        assert genus.profile[2] and genus.erosion[0] > 0
        assert genus.stretch == 1.0 and field.octave_weights[3] == 0.0
        ns = GENUS_PRESETS["nimbostratus"].slab
        as_ = GENUS_PRESETS["altostratus"].slab
        assert ns[0] <= field.slab[0] and sum(field.slab) <= ns[0] + ns[1]
        assert sum(field.slab) < as_[0]
    else:
        # A sheet over 4.7-6 km (clear of the altocumulus's 3.5-3.8 km):
        # fbm of 15 km features or more, stretched 4-6 times along east,
        # soft at both ends, no erosion (its rim sawn); its last octave
        # dropped (the far levels' cut drew a contour in its wide window);
        # thin (ground glass: the sun shows through)
        assert genus.noise_kind == "value" and not genus.worley
        assert field.slab == (4700.0, 1300.0) and field.feature_size[0] >= 15e3
        assert 4.0 <= genus.stretch <= 6.0
        assert min(genus.profile[:2]) >= 0.2 and not genus.profile[2]
        assert genus.erosion[0] == 0.0 and field.octave_weights[3] == 0.0
        assert field.density < 0.5 * PRESETS[CloudType.STRATUS].field.density
        assert field.slab[0] > 4500.0  # the altocumulus: 3.5-3.8 km


def test_the_nimbostratus_and_altostratus_take_their_own_optics():
    """GenusPreset.optics overrides its type's optics (the stratus's): the
    nimbostratus darker (less sky fill, a weaker forward lobe: rain cloud,
    not silver lining) and its sun steps longer over its 4.4 km; the
    altostratus an ice sheet (more sky fill: grey-bluish; its forward lobe
    an ice cloud's)."""
    base = PRESETS[CloudType.STRATUS].optics
    ns, as_ = _preset("nimbostratus")[0].optics, _preset("altostratus")[0].optics
    assert GENUS_PRESETS["stratus"].optics is None
    assert _preset("stratus")[0].optics == base
    # Darker inside than the stratus (its sunward depth scaled less), its
    # broad faces lit by more fill (they are kilometres from the sun)
    assert ns.shadow_strength > base.shadow_strength
    assert ns.multiple_scattering > base.multiple_scattering
    assert ns.forward_gain < base.forward_gain
    assert ns.sun_step_length > base.sun_step_length
    assert as_.multiple_scattering > base.multiple_scattering
    assert as_.forward_anisotropy != base.forward_anisotropy
    # The genus's own optics are its GenusField's too
    assert _preset("nimbostratus")[3].optics == ns


@pytest.mark.parametrize("name", STRATIFORM)
@pytest.mark.parametrize("coverage", [0.3, 0.75, 0.9])
def test_a_preset_coverage_holds_with_its_erosion(cells, name, coverage):
    """The share of columns holding cloud, the erosion on (the altostratus
    has none), within 5 % of the coverage its shift table asks (the erosion
    calibrated into the column peaks: noise.recipe_column_peaks)."""
    from world_builder.clouds.field_setup import _noise
    from world_builder.clouds.weather import coverage_to_shift, shift_table

    _, field, peaks, genus = _preset(name)
    assert (genus.erosion[0] != 0.0) == (name != "altostratus")
    shift = float(coverage_to_shift(coverage, shift_table(peaks, field.threshold[0])))
    rng = np.random.default_rng(5)
    dirs = rng.normal(size=(3000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    h = (np.arange(64) + 0.5) / 64
    base, thickness = field.slab
    points = dirs[:, None, :] * (R + base + h * thickness)[None, :, None]
    plane = np.full((6, 2, 2), shift)
    density = recipe_density_on_sphere(_noise(7), cells, plane, genus, points, R)
    measured = float(np.mean((density > 0.0).any(axis=1)))
    assert measured == pytest.approx(coverage, abs=0.05)


# ── Far averages and the CPU twin from the recipe ────────────────────────────


@pytest.fixture(scope="module")
def stratocumulus_far():
    """The stratocumulus at its far level (3 octaves, the deck's optics) and
    its far table (noise.recipe_deck_light_table)."""
    from world_builder.clouds.field_setup import _noise
    from world_builder.clouds.noise import recipe_deck_light_table
    from world_builder.clouds.weather import shift_table

    _, field, peaks, genus = _preset("stratocumulus")
    table = shift_table(peaks, field.threshold[0])
    genus = replace(genus, optics=replace(genus.optics, max_optical_depth=6.0))
    far = recipe_deck_light_table(
        _noise(7), worley_volume(), genus, R, table, octaves=3, coverages=5,
        cosines=6, columns=512, ground_columns=256,
    )  # fmt: skip
    return genus, table, far


def test_a_recipe_far_table_is_its_own_density_mean(stratocumulus_far):
    """The ground's mean shadow in a recipe genus's far table is the mean of
    its CPU shadow (noise.recipe_shadow_on_sphere, the bakes' march) over
    the planet at that coverage and sun; clear sky lets the sun through,
    more cloud less of it; the lighting's channels in [0, 1]."""
    from world_builder.clouds.field_setup import _noise
    from world_builder.clouds.noise import (
        DECK_LIGHT_COS_MIN,
        _fibonacci_dirs,
        recipe_shadow_on_sphere,
    )
    from world_builder.clouds.weather import coverage_to_shift

    genus, table, far = stratocumulus_far
    assert far.shape == (5, 6, 4)
    assert np.all((far >= 0.0) & (far <= 1.0))
    mu = np.linspace(DECK_LIGHT_COS_MIN, 1.0, 6)
    noon = int(np.argmax(mu))
    assert far[0, noon, 2] == pytest.approx(1.0, abs=1e-3)  # no cloud
    assert far[4, noon, 2] < far[2, noon, 2] < far[0, noon, 2]
    up = _fibonacci_dirs(400)
    shift = float(coverage_to_shift(0.5, np.asarray(table)))
    depth = recipe_shadow_on_sphere(
        _noise(7), worley_volume(), genus, up * (R + 1.0), up, R, 8,
        shift_at=lambda p: np.full(len(p), shift), octaves=3,
    )  # fmt: skip
    assert far[2, noon, 2] == pytest.approx(float(np.exp2(-depth).mean()), abs=0.05)


def test_the_far_table_of_a_recipe_genus_is_not_its_fields(stratocumulus_far):
    """Its fbm's table (what Task 15 left) shades the closed cells as the
    fbm's blobs: the recipe's own differs (the ground's mean shadow by more
    than 0.1 somewhere: 0.2 at a quarter cover and a low sun)."""
    from world_builder.clouds.field_setup import _noise
    from world_builder.clouds.noise import (
        fbm_sigma,
        ground_shadow_table,
        sphere_column_peaks,
        with_threshold,
    )
    from world_builder.clouds.weather import shift_table

    genus, _, far = stratocumulus_far
    field = replace(genus.field, threshold=None)
    peaks = sphere_column_peaks(_noise(7), field, R, 3, columns=4096)
    field = with_threshold(field, peaks, fbm_sigma(_noise(7), field))
    own = ground_shadow_table(
        _noise(7), field, genus.optics, R, shift_table(peaks, field.threshold[0]),
        cosines=6, coverages=5, columns=256,
    )  # fmt: skip
    assert float(np.abs(own - far[..., 2]).max()) > 0.1


@pytest.fixture(scope="module")
def sheet_far():
    """The nimbostratus's, the altostratus's and the altocumulus's far
    tables (3 octaves, the deck's optics: noise.recipe_deck_light_table)."""
    from world_builder.clouds.field_setup import _noise
    from world_builder.clouds.noise import recipe_deck_light_table
    from world_builder.clouds.weather import shift_table

    out = {}
    for name in ("nimbostratus", "altostratus", "altocumulus"):
        _, field, peaks, genus = _preset(name)
        table = shift_table(peaks, field.threshold[0])
        genus = replace(genus, optics=replace(genus.optics, max_optical_depth=6.0))
        out[name] = recipe_deck_light_table(
            _noise(7), worley_volume(), genus, R, table, octaves=3, coverages=5,
            cosines=6, columns=512, ground_columns=256,
        )  # fmt: skip
    return out


def test_the_nimbostratus_blots_out_the_sun_the_altostratus_dims_it(sheet_far):
    """Under a full sheet at noon (the far table's ground shadow): the
    nimbostratus lets almost none of the sun through (WMO: "blots it
    out"); the altostratus, ground glass, a share of it (the sun seen
    dimly), far more than the nimbostratus; both clear with no cloud."""
    from world_builder.clouds.noise import DECK_LIGHT_COS_MIN

    noon = int(np.argmax(np.linspace(DECK_LIGHT_COS_MIN, 1.0, 6)))
    ns, as_ = sheet_far["nimbostratus"], sheet_far["altostratus"]
    for far in (ns, as_):
        assert far.shape == (5, 6, 4) and np.all((far >= 0.0) & (far <= 1.0))
        assert far[0, noon, 2] == pytest.approx(1.0, abs=1e-3)
        assert far[4, noon, 2] < far[2, noon, 2] < far[0, noon, 2]
    assert ns[4, noon, 2] < 0.02
    assert 0.08 < as_[4, noon, 2] < 0.6
    assert as_[4, noon, 2] > 5.0 * ns[4, noon, 2]


def test_the_altocumulus_layer_shades_the_ground_in_patches(
    sheet_far, stratocumulus_far
):
    """Under an altocumulus layer at noon (its far table's ground shadow):
    at three-quarter cover its gaps and thin elements let most of the sun
    through, more than the stratocumulus's or the nimbostratus's; a full
    layer (opacus) hides it more than the altostratus's ground glass (WMO:
    "with shading"); clear with no cloud, darker the more cloud."""
    from world_builder.clouds.noise import DECK_LIGHT_COS_MIN

    noon = int(np.argmax(np.linspace(DECK_LIGHT_COS_MIN, 1.0, 6)))
    ac, ns, as_ = (sheet_far[n] for n in ("altocumulus", "nimbostratus", "altostratus"))
    sc = stratocumulus_far[2]
    assert ac.shape == (5, 6, 4) and np.all((ac >= 0.0) & (ac <= 1.0))
    assert ac[0, noon, 2] == pytest.approx(1.0, abs=1e-3)
    assert ac[4, noon, 2] < ac[3, noon, 2] < ac[2, noon, 2] < ac[0, noon, 2]
    assert 0.45 < ac[3, noon, 2] < 0.8
    assert ac[3, noon, 2] > sc[3, noon, 2] + 0.1 and ac[3, noon, 2] > ns[3, noon, 2]
    assert ac[4, noon, 2] < 0.5 * as_[4, noon, 2]


def test_the_cpu_twin_casts_a_recipe_genus_shadow(showbase):
    """field_setup.genus_sun_depth: a recipe genus's caster marches its
    recipe (noise.recipe_shadow_on_sphere) through its far level, its shift
    from its coverage cube: under a full deck, in shadow; over the deck's
    top or with no coverage, none."""
    from types import SimpleNamespace

    from world_builder.clouds.field_setup import genus_sun_depth
    from world_builder.clouds.weather import shift_table

    _, field, peaks, genus = _preset("stratus")
    table = shift_table(peaks, field.threshold[0])
    up = np.array([0.0, 0.0, 1.0])
    points = np.array([up * (R + 10.0), up * (R + 3000.0)])

    def caster(coverage):
        cube = np.full((6, 4, 4, 4), coverage, dtype=np.float32)
        return SimpleNamespace(
            recipe=genus, weather=cube[..., :3], coverage_source=(cube, 0, table),
            _seed=7, radius=R, _deck_field=field, _deck_optics=genus.optics,
            _deck_octaves=4,
        )  # fmt: skip

    full = genus_sun_depth(caster(1.0), points, up, 8)
    assert full[0] > 1.0 and full[1] == 0.0
    assert genus_sun_depth(caster(0.0), points, up, 8)[0] == pytest.approx(0.0)
