"""BodyChart: the chart of a body of a world file, placed in the world."""

import numpy as np
import pytest

from world_builder.terrain.chart import BodyChart
from world_builder.world_file import (
    SHIPPED_PLANETS,
    WORLDS_PATH,
    GameWorld,
    parse_world,
)

THREE = WORLDS_PATH / "three_planets.yaml"


@pytest.fixture(scope="module")
def spec():
    return parse_world(THREE)


@pytest.fixture(scope="module")
def frame(spec) -> BodyChart:
    return spec.chart("planet")


def _matrix(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_the_reference_point_is_on_the_planets_sea_level(frame):
    reference = frame.position(0, 0, 0)
    assert np.linalg.norm(reference - frame.centre) == pytest.approx(frame.radius)
    assert frame.position(0, 0, 250) == pytest.approx(reference + 250 * frame.up)


def test_the_planet_may_be_anywhere():
    shifted = BodyChart(20.0, 30.0, 6e5, centre=(1e6, -2e6, 3e6))
    base = BodyChart(20.0, 30.0, 6e5, centre=(0, 0, 0))
    assert shifted.position(300, 400, 50) == pytest.approx(
        base.position(300, 400, 50) + [1e6, -2e6, 3e6]
    )
    assert shifted.up_at(shifted.position(300, 400, 50)) == pytest.approx(
        base.up_at(base.position(300, 400, 50))
    )


def test_the_chart_is_made_of_the_bodys_planet_and_position(spec):
    """Radius and origin come from the planet, the centre from where the file
    puts the body: another body, another chart."""
    planet, moon, _ = spec.bodies
    for body in (planet, moon):
        chart = spec.chart(body.name)
        assert chart.radius == body.spec.radius
        assert chart.centre == pytest.approx(body.position)
        assert np.linalg.norm(chart.position(0, 0, 0) - chart.centre) == (
            pytest.approx(body.spec.radius)
        )
    assert not np.allclose(
        spec.chart("planet").position(100, 200, 0),
        spec.chart("moon").position(100, 200, 0),
    )
    with pytest.raises(KeyError):
        spec.chart("nope")


def test_a_game_world_gives_its_files_charts(spec):
    game = GameWorld.__new__(GameWorld)
    game.spec = spec
    assert game.chart("moon").centre == pytest.approx(spec.chart("moon").centre)


def test_axes_are_orthonormal(frame):
    axes = np.array([frame.east, frame.north, frame.up])
    assert axes @ axes.T == pytest.approx(np.eye(3))


def test_chart_distances_follow_the_surface(frame):
    point = frame.position(6000, 8000, 0)
    assert np.linalg.norm(point - frame.centre) == pytest.approx(frame.radius)
    angle = np.arccos(frame.up_at(point) @ frame.up)
    assert angle * frame.radius == pytest.approx(10_000, rel=1e-6)
    assert (point @ frame.north) > 0 and (point @ frame.east) > 0


def test_altitude_is_along_the_local_vertical(frame):
    low, high = frame.position(3000, 2000, 0), frame.position(3000, 2000, 1200)
    assert np.linalg.norm(high - low) == pytest.approx(1200)
    assert (high - low) / 1200 == pytest.approx(frame.up_at(low))


@pytest.mark.parametrize(
    "toward",
    [(0, -3000, 1300), (-3000, -4000, 1300), (0, -5000, 1300), (3000, -4000, 1300)],
)
def test_orientation_is_level_and_faces_the_target(frame, toward):
    start = frame.position(0, -4000, 1300)
    quat = frame.orientation(start, frame.position(*toward))
    assert np.linalg.norm(quat) == pytest.approx(1.0) and quat[0] >= 0
    r = _matrix(quat)
    right, forward, up = r[:, 0], r[:, 1], r[:, 2]
    assert up == pytest.approx(frame.up_at(start))
    assert forward @ up == pytest.approx(0, abs=1e-9)
    assert np.cross(right, forward) == pytest.approx(up)
    ahead = frame.position(*toward) - start
    assert forward @ ahead > 0


def test_every_shipped_planet_has_a_chart_whose_vertical_is_radial():
    from world_builder.world_file import load_planet

    for name in SHIPPED_PLANETS:
        planet = load_planet(name)
        chart = BodyChart(
            planet.reference_lat, planet.reference_lon, planet.radius, (0, 0, 0)
        )
        assert chart.up_at(chart.position(0, 0, 0)) == pytest.approx(chart.up)
