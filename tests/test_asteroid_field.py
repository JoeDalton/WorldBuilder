"""Drawing a field (asteroids.population, asteroids.field): the rocks held
around the camera, the visible pass, the instance table, and the field
rendered offscreen."""

import math

import numpy as np
import pytest

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.motion import orientation
from world_builder.asteroids.population import (
    LOD_HYSTERESIS,
    Population,
    cell_keys,
    key_cells,
)
from world_builder.asteroids.scatter import Scatter
from world_builder.asteroids.spec import (
    PRESETS,
    Box,
    Density,
    FieldSpec,
    ShapeSpec,
    SpinSpec,
    Tumble,
)

PPR = 900 / math.radians(40.0)  # a 900-pixel-high view, 40 degrees


def _forward_view(heading: float = 0.0) -> np.ndarray:
    """(4, 3) side-plane normals of a view along +y (turned by heading,
    degrees, about z), 60 x 40 degrees."""
    h, v = math.radians(30.0), math.radians(20.0)
    local = np.array(
        [
            (math.cos(h), math.sin(h), 0.0),
            (-math.cos(h), math.sin(h), 0.0),
            (0.0, math.sin(v), math.cos(v)),
            (0.0, math.sin(v), -math.cos(v)),
        ]
    )
    a = math.radians(heading)
    turn = np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    )
    return local @ turn.T


def _small_field(**kwargs) -> FieldSpec:
    return FieldSpec(
        shape=Box((3000.0, 3000.0, 3000.0)),
        density=Density(20.0),
        sizes=Distribution.power_law(3.0, 50.0, 400.0),
        **kwargs,
    )


def test_cell_keys_round_trip():
    cells = np.array([[0, 0, 0], [-1, 2, -3], [1000, -999999, 12345]])
    np.testing.assert_array_equal(key_cells(cell_keys(cells)), cells)
    assert len(set(cell_keys(cells).tolist())) == 3
    with pytest.raises(ValueError):
        cell_keys([[1 << 21, 0, 0]])


def test_the_population_holds_the_rocks_in_range():
    scatter = Scatter(_small_field())
    pop = Population(scatter, capacity=16)
    camera = np.array([0.0, 0.0, 0.0])
    ranges = np.full(len(scatter.classes), 1500.0)
    added, removed = pop.update(camera, ranges)
    assert len(removed) == 0 and len(added) == pop.count > 0
    held = pop.rocks[pop.live]
    assert len(np.unique(held["id"])) == len(held)  # never twice
    # every rock within range of the camera is held
    near = scatter.rocks_near(camera, 1000.0)
    assert set(near["id"].tolist()) <= set(held["id"].tolist())
    # moving away frees the cells left behind and fills new ones
    camera2 = np.array([2500.0, 0.0, 0.0])
    added2, removed2 = pop.update(camera2, ranges)
    assert len(added2) and len(removed2)
    held2 = pop.rocks[pop.live]
    assert len(np.unique(held2["id"])) == len(held2)
    assert set(scatter.rocks_near(camera2, 1000.0)["id"].tolist()) <= set(
        held2["id"].tolist()
    )
    # freed slots are reused
    assert set(added2.tolist()) & set(removed2.tolist())
    # the same window again: nothing to do
    assert all(len(x) == 0 for x in pop.update(camera2, ranges))


def test_ranges_follow_the_pixel_size():
    pop = Population(Scatter(PRESETS["cube"]))
    ranges = pop.ranges(PPR, 2.0, 1e9)
    hi = np.array([c.hi for c in pop.scatter.classes])
    np.testing.assert_allclose(ranges, 2.0 * hi * PPR / 2.0 + hi)
    assert np.all(np.diff(ranges) > 0)
    assert np.all(pop.ranges(PPR, 2.0, 5000.0) <= 5000.0 + hi)


def _one_class_population(radius: float = 100.0) -> Population:
    spec = FieldSpec(
        shape=Box((4000.0, 4000.0, 4000.0)),
        density=Density(4.0),
        sizes=Distribution.constant(radius),
        shapes=ShapeSpec(prototypes=1),
    )
    pop = Population(Scatter(spec))
    pop.update(np.zeros(3), np.full(1, 10_000.0))
    return pop


def test_the_visible_pass_culls_behind_and_small():
    pop = _one_class_population()
    errors = np.array([[0.1, 0.05, 0.0]])
    rel = pop.rocks["centre"]
    lateral = np.maximum(np.abs(rel[:, 0]), np.abs(rel[:, 2]))
    ahead = pop.live & (rel[:, 1] > 300.0) & (lateral < 0.3 * rel[:, 1])
    behind = pop.live & (rel[:, 1] < -300.0) & (lateral < 0.3 * -rel[:, 1])
    visible = pop.visible(np.zeros(3), _forward_view()[None], PPR, 2.0, 1.0, errors)
    shown = set(visible.slots.tolist())
    assert np.flatnonzero(ahead).tolist() and set(np.flatnonzero(ahead)) <= shown
    assert not set(np.flatnonzero(behind)) & shown
    assert visible.counts.sum() == len(visible.slots)
    # a rear view (a mirror) shows those behind too
    both = np.stack((_forward_view(), _forward_view(180.0)))
    visible = pop.visible(np.zeros(3), both, PPR, 2.0, 1.0, errors)
    assert set(np.flatnonzero(behind)) <= set(visible.slots.tolist())
    # too small on screen: not drawn (diameter 200 m, 2 px at 1.6 km per
    # 160 px/rad... so at 1 px per rad nothing is)
    visible = pop.visible(np.zeros(3), both, 1.0, 2.0, 1.0, errors)
    assert len(visible.slots) == 0
    # rows not on the GPU yet are never drawn
    ready = np.zeros(pop.capacity, dtype=bool)
    assert len(pop.visible(np.zeros(3), both, PPR, 2.0, 1.0, errors, ready).slots) == 0


def test_nearer_rocks_get_finer_levels_with_hysteresis():
    pop = _one_class_population()
    errors = np.array([[0.1, 0.05, 0.0]])
    view = np.stack((_forward_view(), _forward_view(180.0)))
    ppr = 100.0  # a coarse view: the three levels within the 4 km held
    visible = pop.visible(np.zeros(3), view, ppr, 2.0, 1.0, errors)
    slots = visible.slots
    level = pop.lod[slots]
    dist = np.linalg.norm(pop.rocks["centre"][slots], axis=1)
    order = np.argsort(dist)
    # never coarser when nearer
    assert np.all(np.diff(level[order].astype(int)) <= 0)
    assert level.min() < level.max()
    # each level is the coarsest whose error fits
    scale = 100.0 * ppr / dist
    for s, lv, k in zip(slots, level, scale):
        assert errors[0, lv] * k <= 1.0
        assert lv == 0 or errors[0, lv - 1] * k > 1.0
    # a rock just past where it would coarsen keeps its level
    s, k = slots[0], scale[0]
    lv = pop.lod[s]
    if lv > 0:
        threshold = 1.0 / errors[0, lv - 1]  # where lv - 1 fits
        just_coarse = threshold / (0.5 * (1.0 + LOD_HYSTERESIS))
        pop.rocks["centre"][s] *= k / just_coarse
        pop.visible(np.zeros(3), view, ppr, 2.0, 1.0, errors)
        assert pop.lod[s] == lv


def test_buckets_group_the_slots():
    spec = _small_field(shapes=ShapeSpec(prototypes=3))
    pop = Population(Scatter(spec))
    pop.update(np.zeros(3), np.full(len(pop.scatter.classes), 10_000.0))
    errors = np.tile([0.1, 0.0], (3, 1))
    view = np.stack((_forward_view(), _forward_view(180.0)))
    visible = pop.visible(np.zeros(3), view, PPR, 2.0, 1.0, errors)
    start = 0
    for bucket, n in enumerate(visible.counts):
        group = visible.slots[start : start + n]
        assert np.all(pop.rocks["proto"][group] * 2 + pop.lod[group] == bucket)
        start += n


def _shader_orientation(rows: np.ndarray, t: np.float32) -> np.ndarray:
    """asteroid.vert's orientation, in float32 as the GPU computes it."""

    def turn(w, t):
        rate = np.linalg.norm(w, axis=1).astype(np.float32)
        half = np.float32(0.5) * rate * t
        axis = w / np.maximum(rate, np.float32(1e-12))[:, None]
        q = np.concatenate((np.cos(half)[:, None], axis * np.sin(half)[:, None]), 1)
        return np.where(rate[:, None] < 1e-12, [1, 0, 0, 0], q).astype(np.float32)

    def mul(a, b):
        w = a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], 1)
        v = a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] + np.cross(a[:, 1:], b[:, 1:])
        return np.concatenate((w[:, None], v), 1)

    spin = np.zeros((len(rows), 3), np.float32)
    spin[:, 2] = rows[:, 11]
    return mul(mul(turn(rows[:, 8:11], t), rows[:, 4:8]), turn(spin, t))


def test_table_rows_rebuild_each_rock():
    tumble = Tumble(0.5, Distribution.uniform(0.2, 0.8), Distribution.uniform(0.5, 1.5))
    spec = _small_field(spin=SpinSpec(Distribution.uniform(0.01, 0.2), tumble=tumble))
    pop = Population(Scatter(spec))
    pop.update(
        np.array([100.0, -200.0, 50.0]), np.full(len(pop.scatter.classes), 2500.0)
    )
    slots = np.flatnonzero(pop.live)
    rocks = pop.rocks[slots]
    for epoch in (0.0, 3072.0):
        rows = pop.table_rows(slots, epoch)
        bits = rows[:, 15].astype(np.int64)
        np.testing.assert_array_equal(bits >> 18, rocks["cls"])
        np.testing.assert_array_equal((bits >> 6) & 4095, rocks["proto"])
        np.testing.assert_allclose((bits & 63) / 63.0, rocks["ice"], atol=1 / 126)
        cell = np.array([c.cell for c in pop.scatter.classes])[rocks["cls"]]
        centre = rows[:, 12:15].astype(np.float64) * cell[:, None] + rows[:, 0:3]
        np.testing.assert_allclose(centre, rocks["centre"], atol=1e-3)
        np.testing.assert_allclose(rows[:, 3], rocks["radius"], rtol=1e-6)
        # the shader's turn from the epoch: the closed form at epoch + t,
        # in float32, a second before the next epoch too
        for t in (0.0, 1.5, 1023.0):
            expected = orientation(
                rocks["q0"],
                rocks["axis"],
                rocks["precession"],
                rocks["spin"],
                epoch + t,
            )
            got = _shader_orientation(rows, np.float32(t))
            same = np.abs(np.sum(got * expected, axis=1))
            assert np.all(same > 1.0 - 1e-5)


# ── on the GPU ──────────────────────────────────────────────────────────────


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


@pytest.fixture(scope="module")
def built_field(window):
    from world_builder.asteroids.field import AsteroidField
    from world_builder.asteroids.settings import AsteroidSettings
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=4)
    clock = {"t": 0.0}
    spec = FieldSpec(
        shape=Box((3000.0, 3000.0, 3000.0)),
        density=Density(10.0),
        sizes=Distribution.uniform(80.0, 150.0),
        shapes=ShapeSpec(prototypes=2),
        spin=SpinSpec(Distribution.constant(0.3), size_exponent=0.0),
    )
    field = AsteroidField(
        window,
        spec,
        clock=lambda: clock["t"],
        settings=AsteroidSettings(max_lod=16, normal_size=64, texture_size=256),
        executor=executor,
    )
    for _ in range(10):
        field.update((0.0, 0.0, 0.0))
        if not field.loading:
            break
    assert not field.loading
    yield field, clock
    field.destroy()
    executor.shutdown()


def _look_at_rock(window, field):
    """Put the camera 4 radii in front of a rock, looking at it.

    :returns: the camera's position"""
    pop = field.population
    slot = int(np.flatnonzero(pop.live)[0])
    rock = pop.rocks[slot]
    camera = rock["centre"] - np.array([0.0, 4.0 * rock["radius"], 0.0])
    window.camera.set_pos(*camera)
    window.camera.look_at(*rock["centre"])
    return camera


def _frame(window, field, camera) -> np.ndarray:
    for _ in range(UPLOAD_FRAMES):
        field.update(camera, camera)
        window.graphicsEngine.render_frame()
    shot = window.win.get_screenshot()
    data = np.frombuffer(shot.get_ram_image_as("RGB"), dtype=np.uint8)
    return data.reshape(shot.get_y_size(), shot.get_x_size(), 3).astype(int)


UPLOAD_FRAMES = 5


def test_the_field_draws_its_rocks(window, built_field):
    field, _ = built_field
    window.set_background_color(0, 0, 0, 1)
    field.set_sun((0.0, -1.0, 0.3), (2.4, 2.25, 2.0), (0.03, 0.03, 0.035))
    camera = _look_at_rock(window, field)
    image = _frame(window, field, camera)
    centre = image[24:40, 24:40]
    assert centre.mean() > 10  # the rock, lit
    assert field.stats["drawn"] >= 1
    # its row on the GPU is the population's
    slot = int(np.flatnonzero(field.population.live)[0])
    np.testing.assert_allclose(
        field._table_values[slot], field.population.table_rows(np.array([slot]), 0.0)[0]
    )


def test_the_rocks_turn_with_the_given_clock_only(window, built_field):
    field, clock = built_field
    camera = _look_at_rock(window, field)
    clock["t"] = 0.0
    a = _frame(window, field, camera)
    b = _frame(window, field, camera)  # Panda's own clock moved on
    assert np.array_equal(a, b)
    clock["t"] = 3.0  # 0.9 rad later
    c = _frame(window, field, camera)
    assert not np.array_equal(a, c)
    clock["t"] = 0.0
