"""Drawing a field (asteroids.population, asteroids.field): the rocks held
around the camera, the visible pass, the instance table, and the field
rendered offscreen."""

import math
import os
from dataclasses import replace

import numpy as np
import pytest

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.motion import orientation
from world_builder.asteroids.population import (
    LOD_HYSTERESIS,
    Population,
)
from world_builder.asteroids.scatter import Scatter
from world_builder.asteroids.spec import (
    PRESETS,
    Box,
    Density,
    FieldSpec,
    MaterialSpec,
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


def _brute_ball(ball):
    """The ball's cells by its definition, cell by cell."""
    reach = int(ball.radius // ball.cell) + 2
    offsets = np.stack(
        np.meshgrid(*[np.arange(-reach, reach + 1)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    gap = np.maximum(np.abs(offsets) - 1, 0) * ball.cell
    cells = ball.home + offsets[np.sum(gap * gap, axis=1) <= ball.radius**2]
    inside = np.all((cells >= ball.first) & (cells <= ball.last), axis=1)
    return {tuple(c) for c in cells[inside].tolist()}


def test_a_window_is_the_ball_of_its_range_and_adds_only_new_cells():
    from world_builder.asteroids.population import _Ball, _difference

    first, last = np.array([-30, -30, -3]), np.array([30, 30, 3])
    old = _Ball([0, 0, 0], 950.0, 100.0, first, last)
    assert {tuple(c) for c in old.grid().tolist()} == _brute_ball(old)
    for new in (
        _Ball([1, 0, -1], 950.0, 100.0, first, last),  # moved a cell
        _Ball([0, 0, 0], 1700.0, 100.0, first, last),  # grown
        _Ball([25, 28, 2], 600.0, 100.0, first, last),  # elsewhere, clipped
        _Ball([0, 0, 0], 400.0, 100.0, first, last),  # shrunk: nothing new
    ):
        cells = _difference(new, old)
        assert {tuple(c) for c in cells.tolist()} == _brute_ball(new) - _brute_ball(old)
        assert len({tuple(c) for c in cells.tolist()}) == len(cells)  # each once
        held = _brute_ball(new)
        probe = np.array(sorted(held | _brute_ball(old)))
        np.testing.assert_array_equal(
            new.contains(probe), [tuple(c) in held for c in probe.tolist()]
        )


def test_windows_grow_within_the_quota():
    from world_builder.asteroids.population import _Ball, _count, _toward

    first, last = np.array([-60, -60, -2]), np.array([60, 60, 2])
    home = np.array([3, -4, 0])
    window, frames = None, 0
    while window is None or window.radius < 5000.0:
        before = window
        window = _toward(window, 5000.0, home, 100.0, first, last, 2000)
        frames += 1
        assert _count(window, before) <= max(2000, 27 * 5)
        assert window.contains(home[None, :])[0]
    assert frames > 3
    # moved a cell: the window follows at once, whatever it costs
    moved = _toward(window, 5000.0, home + [1, 0, 0], 100.0, first, last, 10)
    assert moved.radius == 5000.0
    # teleported: it restarts around the camera
    restart = _toward(window, 5000.0, np.array([55, 55, 0]), 100.0, first, last, 50)
    assert restart.radius < 5000.0
    assert restart.contains(np.array([[55, 55, 0]]))[0]
    # every point within the range of the camera's cell is in a cell held
    ball = _Ball(home, 777.0, 100.0, first, last)
    rng = np.random.default_rng(1)
    camera = (home + rng.uniform(0.0, 1.0, (200, 3))) * 100.0
    points = camera + rng.normal(size=(200, 3)) * 300.0
    near = np.linalg.norm(points - camera, axis=1) <= 777.0
    cells = np.floor(points[near] / 100.0).astype(np.int64)
    inside = np.all((cells >= first) & (cells <= last), axis=1)
    assert np.all(ball.contains(cells[inside]))


def test_a_gradual_fill_holds_the_same_rocks():
    scatter = Scatter(_small_field())
    camera = np.array([100.0, -200.0, 50.0])
    ranges = np.full(len(scatter.classes), 1500.0)
    whole = Population(scatter)
    whole.update(camera, ranges, cells_per_frame=10**9)
    assert whole.settled
    gradual = Population(scatter)
    frames = 0
    while not np.array_equal(gradual.limits(), ranges):
        gradual.update(camera, ranges, cells_per_frame=len(scatter.classes) * 30)
        frames += 1
        # settled (loaded) once each window is within 10 % of its range
        assert gradual.settled == bool(np.all(gradual.limits() >= 0.9 * ranges))
    assert frames > 2
    assert sorted(gradual.rocks[gradual.live]["id"]) == sorted(
        whole.rocks[whole.live]["id"]
    )


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


def test_no_rock_is_drawn_past_far():
    """The visible pass draws nothing past far (a sphere, whatever the
    window's shape) nor counts it against the budget."""
    pop = _one_class_population()
    camera = np.zeros(3)
    planes = np.zeros((0, 4, 3))  # no view: check the counts only
    errors = np.zeros((1, 1))
    every = pop.visible(camera, planes, PPR, 0.5, 1.0, errors)
    held = pop.rocks[pop.live]
    dist = np.linalg.norm(held["centre"], axis=1)
    near = pop.visible(camera, planes, PPR, 0.5, 1.0, errors, far=3000.0)
    assert every.in_range == len(held)
    assert near.in_range == int(np.count_nonzero(dist <= 3000.0)) < every.in_range


def test_the_native_visible_pass_is_numpys():
    """visible.c against the numpy pass: the same slots, buckets, counts,
    levels, rocks in range and big rocks near, frame after frame (the levels' hysteresis
    carried over), with two views, a far and a window still growing."""
    from world_builder.terrain import noise

    if noise.native_library() is None:
        if os.environ.get("WORLD_BUILDER_REQUIRE_NATIVE"):
            pytest.fail("the native library is required but not built")
        pytest.skip("native library not built")
    scatter = Scatter(_small_field(shapes=ShapeSpec(prototypes=3)))
    pops = [Population(scatter), Population(scatter)]
    ranges = np.full(len(scatter.classes), 2500.0)
    rng = np.random.default_rng(4)
    errors = np.sort(rng.uniform(0.0, 0.2, (3, 4)), axis=1)[:, ::-1].copy()
    errors[:, -1] = 0.0
    for frame in range(6):
        camera = np.array([100.0 * frame, -50.0 * frame, 20.0])
        for pop in pops:
            pop.update(camera, ranges, cells_per_frame=200 if frame < 3 else 10**9)
        ready = rng.uniform(size=pops[0].capacity) < 0.9
        planes = np.array(
            [
                [[0.8, 0.6, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.6, 0.8], [0.0, 0.6, -0.8]],
                [
                    [0.6, -0.8, 0.0],
                    [-0.6, -0.8, 0.0],
                    [0.0, -0.8, 0.6],
                    [0.0, -0.8, -0.6],
                ],
            ]
        )
        args = (camera, planes, PPR, 1.5, 0.4 + 0.3 * frame, errors, ready, 2200.0)
        near = {"near_radius": 100.0, "near_factor": 8.0}
        a = pops[0].visible(*args, native=True, **near)
        b = pops[1].visible(*args, native=False, **near)
        np.testing.assert_array_equal(a.near, b.near)
        np.testing.assert_array_equal(a.slots, b.slots)
        np.testing.assert_array_equal(a.counts, b.counts)
        assert a.in_range == b.in_range > 0
        np.testing.assert_array_equal(pops[0].lod, pops[1].lod)


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
        material=MaterialSpec(Distribution.constant(0.1)),
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


def test_far_rocks_keep_their_textured_colour(window, built_field):
    """Past the texture distance a rock is drawn in its textures' mean
    colour: the colour its textures average to, not paler by its ice (the
    mean colours mixed by weight made rocks of 10 % ice 1.5 times brighter
    and grey)."""
    field, _ = built_field
    window.set_background_color(0, 0, 0, 1)
    field.set_sun((0.0, -1.0, 0.3), (2.4, 2.25, 2.0), (0.03, 0.03, 0.035))
    camera = _look_at_rock(window, field)
    try:
        field.root.set_shader_input("uDetailFar", 1.0e7)
        textured = _frame(window, field, camera)
        field.root.set_shader_input("uDetailFar", 1.0)
        flat = _frame(window, field, camera)
    finally:
        field.root.set_shader_input("uDetailFar", float(field.settings.detail_far))
    rock = (textured.sum(axis=2) > 0) & (flat.sum(axis=2) > 0)
    assert rock.sum() > 50
    a, b = textured[rock].mean(axis=0), flat[rock].mean(axis=0)
    np.testing.assert_allclose(b, a, rtol=0.15)
    assert b.std() > 3  # brown, not grey


def test_the_rocks_held_keep_to_their_budget(showbase):
    """Past the held budget, the windows keep only bigger rocks: the rocks
    held come down under it, the class limits with them (headless)."""
    from world_builder.asteroids.field import AsteroidField
    from world_builder.asteroids.settings import AsteroidSettings
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=8)
    spec = _small_field(shapes=ShapeSpec(prototypes=2))
    settings = AsteroidSettings(
        max_lod=16, normal_size=64, texture_size=256, held_budget=20_000
    )
    field = AsteroidField(
        showbase, spec, clock=lambda: 0.0, settings=settings, executor=executor
    )
    try:
        for _ in range(10):
            field.update(np.zeros(3), cameras=[])
        unbounded = field.population.count
        limits = field.population.limits().copy()
        field.set_settings(replace(settings, held_budget=unbounded // 3))
        for _ in range(200):
            field.update(np.zeros(3), cameras=[])
        assert field.population.count <= unbounded // 3
        assert field.stats["held_pixels"] > settings.min_pixels
        assert np.all(field.population.limits() <= limits)
    finally:
        field.destroy()
        executor.shutdown()
