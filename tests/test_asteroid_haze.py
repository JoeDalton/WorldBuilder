"""The rocks not drawn, as a haze (asteroids.haze): its cross-section against
the scatter's rocks, its rays, and the shader against the numpy model."""

import math
from dataclasses import replace

import numpy as np
import pytest

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.haze import (
    FADE_MIDDLE,
    HazeModel,
    hull_triangles,
    intervals,
    optical_depth,
    phase,
    radiance,
)
from world_builder.asteroids.scatter import Scatter
from world_builder.asteroids.spec import (
    Annulus,
    Box,
    Density,
    FieldSpec,
    MaterialSpec,
    ShapeSpec,
    Sphere,
)
from world_builder.ocean.ocean import aces_tonemap


def test_a_lambert_sphere_is_dark_against_the_sun():
    assert phase(0.0) == pytest.approx(1.0)
    assert phase(math.pi) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(phase(np.linspace(0.0, math.pi, 50))) < 0.0)


@pytest.mark.parametrize(
    "spec, half",
    [
        (
            FieldSpec(
                shape=Box((15_000.0,) * 3),
                density=Density(2.0),
                sizes=Distribution.power_law(3.0, 50.0, 400.0),
            ),
            14_000.0,
        ),
        (  # the hard core removes a fifth: the haze must too
            FieldSpec(
                shape=Box((8000.0,) * 3),
                density=Density(30.0),
                sizes=Distribution.power_law(3.0, 40.0, 200.0),
                seed=5,
            ),
            7500.0,
        ),
    ],
)
def test_the_haze_and_the_drawn_rocks_add_up(spec, half):
    """Wherever the size cut falls, the haze (the rocks under it) and the
    scatter's rocks over it add up to the same cross-section, within 2 %."""
    rocks = Scatter(spec).rocks_in_box([-half] * 3, [half] * 3)
    volume = (2.0 * half) ** 3
    model = HazeModel(spec, cover=1.0)
    totals = []
    for cut in np.geomspace(spec.sizes.lo, spec.sizes.hi, 9):
        big = rocks["radius"][rocks["radius"] > cut]
        totals.append(model.extinction(cut) + np.sum(np.pi * big * big) / volume)
    totals = np.array(totals)
    assert (totals.max() - totals.min()) / totals.mean() < 0.02
    # all of it at the top: the rocks' own cross-section
    assert model.extinction(1e9) == pytest.approx(totals[-1])
    assert model.extinction(spec.sizes.lo) == 0.0


def test_rays_through_a_ring():
    ring = Annulus(1000.0, 2000.0, 100.0)
    # across the whole ring through its hole: two intervals
    spans = intervals(ring, [-3000.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(spans, [(1000.0, 2000.0), (4000.0, 5000.0)])
    # down through it, from above
    spans = intervals(ring, [1500.0, 0.0, 500.0], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(spans, [(450.0, 550.0)])
    # from inside, along it: from here
    (span,) = intervals(ring, [0.0, 1500.0, 0.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(span, (0.0, 500.0))
    assert intervals(ring, [0.0, 0.0, 500.0], [0.0, 0.0, 1.0]) == []
    np.testing.assert_allclose(
        intervals(Sphere(10.0), [0.0, 0.0, -20.0], [0.0, 0.0, 1.0]), [(10.0, 30.0)]
    )
    np.testing.assert_allclose(
        intervals(Box((1.0, 2.0, 3.0)), [0.0, -5.0, 0.0], [0.0, 1.0, 0.0]),
        [(3.0, 7.0)],
    )


@pytest.mark.parametrize(
    "shape",
    [Box((100.0, 200.0, 300.0)), Sphere(500.0), Annulus(1000.0, 2000.0, 100.0)],
)
def test_hulls_enclose_their_shape_facing_out(shape):
    tris, normals = hull_triangles(shape)
    lo, hi = shape.bounds()
    # every point of the shape is inside every face's plane
    for _ in range(3):
        points = np.random.default_rng(0).uniform(lo, hi, (2000, 3))
        points = points[shape.profile(points) > 0.0]
        side = np.einsum("pk,tk->pt", points, normals) - np.einsum(
            "tk,tk->t", tris[:, 0], normals
        )
        if isinstance(shape, Annulus):  # the hole's walls: only near them
            continue
        assert np.all(side <= 1e-6)
    # each normal is the face's own, outward
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    valid = np.linalg.norm(cross, axis=1) > 1e-9
    assert np.all(np.einsum("tk,tk->t", cross[valid], normals[valid]) != 0.0)


# ── the shader against the model ────────────────────────────────────────────

#: A dense ring, small enough for a test: its haze is easily read.
RING = FieldSpec(
    shape=Annulus(
        2.0e6,
        3.0e6,
        2000.0,
        edge=300.0,
        radial_edge=1.0e5,
        radial_profile=(1.0, 0.3, 1.0),
    ),
    density=Density(30.0, dust=0.05),
    sizes=Distribution.power_law(3.0, 50.0, 2000.0),
    shapes=ShapeSpec(prototypes=2),
    material=MaterialSpec(Distribution.uniform(0.6, 1.0)),
)
SIZE = 64


@pytest.fixture(scope="module")
def window(showbase):
    if showbase.win is None:
        try:
            showbase.openMainWindow(type="offscreen", size=(SIZE, SIZE))
        except Exception as error:  # no GL context here
            pytest.skip(f"no offscreen window: {error}")
    if showbase.win is None:
        pytest.skip("no offscreen window")
    return showbase


@pytest.fixture(scope="module")
def ring(window):
    from world_builder.asteroids.field import AsteroidField
    from world_builder.asteroids.settings import AsteroidSettings
    from world_builder.terrain.workers import ChunkExecutor

    executor = ChunkExecutor(backend="inline", inline_budget=8)
    field = AsteroidField(
        window,
        RING,
        clock=lambda: 0.0,
        settings=AsteroidSettings.preset("low"),
        executor=executor,
        sun_direction=(0.3, -0.2, 0.9),
    )
    lens = window.camLens
    saved = (lens.get_fov(), lens.get_near(), lens.get_far())
    lens.set_fov(40.0)
    yield field
    lens.set_fov(saved[0])
    lens.set_near_far(saved[1], saved[2])
    field.destroy()
    executor.shutdown()


def _shoot(window, field, camera_world, look, near=10.0):
    """Render with the camera at *camera_world* (render's origin), looking
    along *look*; :returns: the frame, and the frame without the haze (the
    rocks alone)."""
    window.set_background_color(0, 0, 0, 1)
    window.camLens.set_near_far(near, 1.0e9)
    window.camera.set_pos(0.0, 0.0, 0.0)
    window.camera.look_at(*look)
    frames = []
    for show in (True, False):
        if show:
            field.haze.node.show()
        else:
            field.haze.node.hide()
        for i in range(400):
            field.update(camera_world)
            window.graphicsEngine.render_frame()
            if not field.loading and i > 4:
                break
        shot = window.win.get_screenshot()
        data = np.frombuffer(shot.get_ram_image_as("RGB"), dtype=np.uint8)
        frames.append(data.reshape(SIZE, SIZE, 3)[::-1].astype(float))
    field.haze.node.show()
    return frames


def _expected(window, field, camera_world, x, y):
    """The model's pixel (x, y): the haze over black."""
    from panda3d.core import Point2, Point3

    lens = window.camLens
    near, far = Point3(), Point3()
    ndc = Point2((x + 0.5) / SIZE * 2.0 - 1.0, 1.0 - (y + 0.5) / SIZE * 2.0)
    lens.extrude(ndc, near, far)
    to_render = window.camera.get_mat(window.render)
    a = np.array(to_render.xform_point(near))
    b = np.array(to_render.xform_point(far))
    d = (b - a) / np.linalg.norm(b - a)
    t_near = float(np.linalg.norm(a))
    ppr = SIZE / math.radians(lens.get_vfov())
    total = 0.0
    for t0, t1 in intervals(field.spec.shape, camera_world, d):
        total += optical_depth(
            field.haze.model,
            camera_world,
            d,
            max(t0, t_near),
            t1,
            ppr,
            field._min_pixels,
            field.population.limits(field.far),
        )
    sun, colour, ambient = field.sun
    light = radiance(field.haze.albedo, sun, colour, ambient, d)
    return aces_tonemap(light, field.exposure) * (1.0 - math.exp(-total)) * 255.0


def _compare(window, field, camera_world, frame, rocks, tolerance=4.0):
    checked = 0
    for y in range(4, SIZE, 9):
        for x in range(4, SIZE, 9):
            if rocks[y, x].sum() > 0.0:  # a rock drawn there
                continue
            expected = _expected(window, field, camera_world, x, y)
            np.testing.assert_allclose(
                frame[y, x], expected, atol=tolerance, err_msg=f"pixel {x}, {y}"
            )
            checked += 1
    return checked


def test_the_shader_draws_the_models_haze_from_above(window, ring):
    """5,000 km over the ring: no rock drawn, the haze across its width."""
    camera = np.array([0.0, 2.5e6, 5.0e6])
    frame, rocks = _shoot(window, ring, camera, (0.0, 0.0, -1.0))
    assert rocks.max() == 0.0
    assert frame[SIZE // 2, SIZE // 2].mean() > 10.0  # readable
    assert _compare(window, ring, camera, frame, rocks) > 40


def test_rocks_appear_out_of_the_haze(window, ring):
    """20 km over the ring: the big rocks drawn, the haze of the others
    between them as the model has it."""
    camera = np.array([0.0, 2.5e6, 2.0e4])
    frame, rocks = _shoot(window, ring, camera, (0.3, 0.2, -1.0))
    assert ring.stats["drawn"] > 0
    assert _compare(window, ring, camera, frame, rocks) > 20


def test_rocks_fade_into_the_haze_at_far(window, ring):
    """A far just past the ring, 20 km under the camera: the rocks of its
    last fifth fade out, the haze takes them in, as the model has it."""
    camera = np.array([0.0, 2.5e6, 2.0e4])
    ring.far = 2.15e4
    try:
        frame, rocks = _shoot(window, ring, camera, (0.3, 0.2, -1.0))
        assert _compare(window, ring, camera, frame, rocks) > 20
    finally:
        ring.far = math.inf


def test_the_haze_starts_at_the_near_plane(window, ring):
    """A near plane inside the ring: only what lies beyond it (the far
    camera of a split starts at the split)."""
    camera = np.array([0.0, 2.5e6, 5.0e6])
    near = 5.0e6 - 200.0
    whole, _ = _shoot(window, ring, camera, (0.0, 0.0, -1.0))
    frame, rocks = _shoot(window, ring, camera, (0.0, 0.0, -1.0), near=near)
    assert frame[SIZE // 2, SIZE // 2].mean() < 0.9 * whole[SIZE // 2, SIZE // 2].mean()
    assert _compare(window, ring, camera, frame, rocks) > 40


def test_dust_is_haze_from_the_camera_on():
    """Dust is never drawn: its optical depth is its rate times the path
    through the profile, nearest included."""
    ring = Annulus(1000.0, 2000.0, 100.0)
    spec = FieldSpec(
        shape=ring, density=Density(0.0, dust=0.5), sizes=Distribution.constant(60.0)
    )
    model = HazeModel(spec)
    assert model.start(1000.0, 2.0, 1e9) == 0.0
    # down through the ring: 100 m of it
    tau = optical_depth(
        model,
        [1500.0, 0.0, 60.0],
        [0.0, 0.0, -1.0],
        10.0,
        110.0,
        1000.0,
        2.0,
        1e9,
        samples=512,
    )
    assert tau == pytest.approx(0.05, rel=1e-3)


def test_the_haze_takes_the_rocks_faded_out_by_far():
    """Past far every rock is haze; over its last fifth, as many as the
    shader fades out (linearly); nearer, only those too small."""
    from world_builder.asteroids.population import FAR_FADE

    model = HazeModel(RING)
    ppr, min_pixels, far = 1000.0, 2.0, 5.0e5
    every = model.extinction(RING.sizes.hi)
    small = model.extinction(model.r_cut(3.0e5, ppr, min_pixels))
    assert model.missing(3.0e5, ppr, min_pixels, far) == pytest.approx(small)
    assert model.missing(6.0e5, ppr, min_pixels, far) == pytest.approx(every)
    middle = (1.0 - 0.5 * FAR_FADE) * far
    under = model.extinction(model.r_cut(middle, ppr, min_pixels))
    assert model.missing(middle, ppr, min_pixels, far) == pytest.approx(
        0.5 * (under + every)
    )
    # no far: only the rocks too small
    assert model.missing(6.0e5, ppr, min_pixels, math.inf) == pytest.approx(
        model.extinction(model.r_cut(6.0e5, ppr, min_pixels))
    )


def test_each_class_past_its_limit_is_haze():
    """A window still growing (its class's limit near): that class's rocks
    past it are haze, the others' as before."""
    model = HazeModel(RING)
    scatter = Scatter(RING)
    np.testing.assert_allclose(model.class_lo, [c.lo for c in scatter.classes])
    np.testing.assert_allclose(model.class_hi, [c.hi for c in scatter.classes])
    ppr, min_pixels, t = 1000.0, 2.0, 2.0e4
    limits = np.full(len(model.class_lo), np.inf)
    base = model.missing(t, ppr, min_pixels, limits)
    limits[1] = 1.0e4  # class 1 held to 10 km only
    k = 1
    whole = model.cumulative(model.class_hi[k]) - model.cumulative(model.class_lo[k])
    cut = np.clip(model.r_cut(t, ppr, min_pixels), model.class_lo[k], model.class_hi[k])
    under = model.cumulative(cut) - model.cumulative(model.class_lo[k])
    expected = base + model.density * model.cover * (whole - under)
    assert model.missing(t, ppr, min_pixels, limits) == pytest.approx(expected)


def test_the_haze_veils_the_rocks(window, ring):
    """The haze between the camera and a rock veils it: its pixels move
    toward the haze's colour."""
    camera = np.array([0.0, 2.5e6, 2.0e4])
    try:
        veiled, rocks = _shoot(window, ring, camera, (0.3, 0.2, -1.0))
        ring.root.set_shader_input("uVeil", 0)
        bare, _ = _shoot(window, ring, camera, (0.3, 0.2, -1.0))
    finally:
        ring.root.set_shader_input("uVeil", 1)
    on_rock = rocks.sum(axis=2) > 30.0
    assert on_rock.sum() > 20
    # the haze's own colour (what it would be, opaque), about the view's
    sun, colour, ambient = ring.sun
    view = np.array([0.3, 0.2, -1.0]) / np.linalg.norm([0.3, 0.2, -1.0])
    light = radiance(ring.haze.albedo, sun, colour, ambient, view)
    haze = aces_tonemap(light, ring.exposure) * 255.0
    nearer = np.abs(veiled[on_rock] - haze).sum(axis=1)
    farther = np.abs(bare[on_rock] - haze).sum(axis=1)
    # (the haze's colour turns with the view: one is the view's centre's)
    assert np.mean(nearer <= farther + 2.0) > 0.95
    assert np.mean(farther - nearer) > 1.0


def test_the_haze_follows_the_drawing_scale():
    """The cut is the drawn rocks' fade's middle: at the distance where a
    rock covers FADE_MIDDLE * min_pixels, it is the cut."""
    model = HazeModel(replace(RING, density=Density(30.0)))  # no dust
    ppr, min_pixels = 1000.0, 2.0
    t = 1.0e5
    cut = model.r_cut(t, ppr, min_pixels)
    assert 2.0 * cut * ppr / t == pytest.approx(FADE_MIDDLE * min_pixels)
    assert model.start(ppr, min_pixels, 1e9) == pytest.approx(
        RING.sizes.lo * 2.0 * ppr / (FADE_MIDDLE * min_pixels)
    )
