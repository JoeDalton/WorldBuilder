"""Fronts and the genus rules (docs/literature/synoptic-weather-review.tex,
sections "Fronts" and "Genus rules"), and the weather's maps keeping their
shapes."""

import math

import numpy as np
import pytest

from world_builder.clouds.synoptic import Scales
from world_builder.clouds.synoptic.fronts import fronts
from world_builder.clouds.synoptic.rules import LAYERS, apply_systems
from world_builder.clouds.synoptic.systems import Kind, SynopticObject
from world_builder.terrain.chart import dirs_from_latlon
from world_builder.terrain.planet import PlanetSpec

EARTH = PlanetSpec("e", radius=6.371e6, day_length_h=23.934)
SCALES = Scales.from_planet(EARTH)
R = SCALES.cyclone_radius
LIFE = 5.0 * 86400.0


def _system(kind, lat, stage=0.4, radius=R, strength=25.0, variant="norwegian"):
    return SynopticObject(
        kind=kind,
        birth=-stage * LIFE,
        lifetime=LIFE,
        lat0=lat,
        lon0=0.0,
        radius=radius,
        strength=strength,
        variant=variant,
        heading=0.0,
        seed=7,
    )


def _points(obj, xy):
    """Directions at local (east, poleward) offsets, in the system's radii."""
    centre = dirs_from_latlon(obj.lat0, obj.lon0)[0]
    east = np.cross([0.0, 0.0, 1.0], centre)
    east /= np.linalg.norm(east)
    north = np.cross(centre, east)
    h = 1.0 if obj.lat0 >= 0.0 else -1.0
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64)) * obj.radius / SCALES.radius
    d = centre + xy[:, :1] * east + h * xy[:, 1:] * north
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _apply(obj, dirs, sea=0.0, temperature=15.0, base=0.5):
    n = len(dirs)
    return apply_systems(
        dirs,
        [obj],
        SCALES,
        0.0,
        base=np.full(n, base),
        sea=np.full(n, sea),
        temperature=np.full(n, temperature),
        elevation=np.full(n, -1000.0 if sea else 200.0),
    )


def test_warm_front_sequence():
    low = _system(Kind.LOW, 50.0)
    warm = fronts(low, 0.0, SCALES).warm / R  # (M, 2) in radii
    mid = len(warm) // 2
    tangent = warm[mid + 1] - warm[mid - 1]
    tangent /= np.linalg.norm(tangent)
    cold_side = np.array([-tangent[1], tangent[0]])  # left of the front
    xi = np.linspace(1.2, 0.02, 60)
    out = _apply(low, _points(low, warm[mid] + xi[:, None] * cold_side))
    order = ["cirrus", "cirrostratus", "altostratus", "nimbostratus"]
    stack = np.stack([out.layers[g] for g in order], axis=-1)
    strong = stack.max(axis=-1) > 0.1
    seq = np.argmax(stack, axis=-1)[strong]
    assert len(seq) > 20
    assert np.all(np.diff(seq) >= 0), seq
    assert set(seq) == {0, 1, 2, 3}


def test_cold_front_cb():
    low = _system(Kind.LOW, 50.0)
    cold = fronts(low, 0.0, SCALES).cold / R
    # The cold front runs from its tip back to the triple point.
    at = cold[int(0.7 * (len(cold) - 1))]
    out = _apply(low, _points(low, at))
    assert out.depth[0] > 0.8
    assert out.shelf[0] > 0.0
    assert out.convective[0] > 0.5


def test_high_core_clear_and_sc_deck():
    high = _system(Kind.HIGH, 30.0, stage=0.5, radius=1.5 * R, strength=15.0)
    core = _apply(high, _points(high, [0.0, 0.0]), sea=0.0, temperature=25.0)
    assert core.convective[0] < 0.15
    for g in LAYERS:
        assert core.layers[g][0] < 0.15, g
    flank = _apply(high, _points(high, [1.0, 0.0]), sea=1.0, temperature=16.0, base=0.0)
    assert flank.layers["stratocumulus"][0] > 0.5


def test_tc_eye_clear():
    tc = _system(Kind.TROPICAL, 15.0, stage=0.5, radius=SCALES.tc_radius, strength=60.0)
    eye = _apply(tc, _points(tc, [0.0, 0.0]), sea=1.0, temperature=28.0)
    assert eye.convective[0] < 0.1
    wall = _apply(tc, _points(tc, [0.1, 0.0]), sea=1.0, temperature=28.0)
    assert wall.depth[0] > 0.8


def test_dry_slot():
    """Behind the cold front, west of the triple point: the dry intrusion."""
    low = _system(Kind.LOW, 50.0)
    out = _apply(low, _points(low, [-0.3, -0.6]), sea=1.0)
    assert out.convective[0] < 0.2
    for g in LAYERS:
        assert out.layers[g][0] < 0.2, g


def test_southern_low_is_mirrored():
    """The same layout poleward in both hemispheres: the comma head
    poleward of the low."""
    north = _system(Kind.LOW, 50.0, stage=0.6)
    south = _system(Kind.LOW, -50.0, stage=0.6)
    head = [0.3, 0.6]
    a = _apply(north, _points(north, head)).layers["nimbostratus"][0]
    b = _apply(south, _points(south, head)).layers["nimbostratus"][0]
    assert a > 0.3 and a == pytest.approx(b, abs=1e-6)


def test_interfaces_unchanged():
    from planets import PRESETS

    from world_builder.clouds.genera import (
        CONVECTIVE,
        DEPTHS,
        GeneraSpec,
        coverage_cube,
        genus_cube,
        genus_planes,
    )
    from world_builder.clouds.weather import (
        SHIFT_TABLE_SIZE,
        WeatherSpec,
        sample_cube,
        weather_cube,
    )

    weather = WeatherSpec(PRESETS["earthlike"])
    table = np.linspace(-1.0, 1.0, SHIFT_TABLE_SIZE)
    assert weather_cube(weather, table, 32).shape == (6, 32, 32, 2)
    genera = GeneraSpec(weather, focus=(29.7, 29.8))
    tables = tuple(tuple(table) for _ in DEPTHS)
    assert genus_cube(genera, CONVECTIVE, tables, 32).shape == (6, 32, 32, 4)
    assert coverage_cube(genera, "A", 32).shape == (6, 32, 32, 4)
    # The showcase's features still show where they are.
    planes = genus_planes(genera, 128)
    from world_builder.clouds.genera import _feature_dirs

    for f, centre, _ in _feature_dirs(genera):
        channel = 2 if f.genus == "stratus" else 0
        if f.radius < 10e3:
            continue  # smaller than a few texels at this size
        value = sample_cube(planes[..., channel], centre[None])[0]
        assert value >= 0.9 * f.coverage, f


@pytest.mark.parametrize("end", ["start", "end"])
def test_warm_shield_fades_past_the_fronts_ends(end):
    """Past either end of the warm front the shield fades smoothly to
    nothing: no fan round the end point, no seam along its ray."""
    low = _system(Kind.LOW, 50.0)
    warm = fronts(low, 0.0, SCALES).warm / R
    a, b = (warm[0], warm[1]) if end == "start" else (warm[-1], warm[-2])
    outward = (a - b) / np.linalg.norm(a - b)
    cold_side = np.array([-outward[1], outward[0]]) * (1.0 if end == "end" else -1.0)
    steps = np.linspace(0.0, 1.0, 101)
    line = a + steps[:, None] * outward + 0.4 * cold_side  # in the As band
    out = _apply(low, _points(low, line), base=0.0)
    shield = np.maximum.reduce(
        [out.layers[g] for g in ("cirrus", "cirrostratus", "altostratus")]
    )
    assert np.max(np.abs(np.diff(shield))) < 0.08
    assert shield[-1] < 0.05


def test_system_layers_carry_the_mesoscale_texture():
    """A high's altocumulus ring is broken by the mesoscale texture, as the
    cumulus are: no perfect circles."""
    from planets import PRESETS

    from world_builder.clouds.synoptic.state import synoptic_state
    from world_builder.clouds.synoptic.systems import position
    from world_builder.clouds.weather import (
        SYNOPTIC_CHANNELS,
        WeatherSpec,
        sample_cube,
        synoptic_cube,
    )
    from world_builder.terrain.chart import TangentFrame

    weather = WeatherSpec(PRESETS["earthlike"])
    state = synoptic_state(weather)
    cube = synoptic_cube(weather)[..., SYNOPTIC_CHANNELS.index("altocumulus")]
    spreads = []
    for obj in state.systems:
        if obj.kind is not Kind.HIGH:
            continue
        lat, lon = position(obj, state.t)
        frame = TangentFrame.at(lat, lon)
        a = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        angle = 1.3 * obj.radius / weather.planet.radius
        ring = math.cos(angle) * frame.up + math.sin(angle) * (
            np.cos(a)[:, None] * frame.east + np.sin(a)[:, None] * frame.north
        )
        values = sample_cube(cube, ring)
        if values.mean() > 0.1:
            # Roughness: neighbouring samples differ (a stencil's ring
            # varies only slowly, where other systems overlap it).
            spreads.append(np.abs(np.diff(values)).mean() / values.mean())
    assert spreads
    assert np.median(spreads) > 0.05, spreads


@pytest.mark.parametrize("variant", ["norwegian", "sk"])
def test_no_seams_anywhere_round_a_low(variant):
    """Every genus is continuous over the whole system: no seams where a
    front's half-plane continues past its end. At 0.01 r a seam jumps by
    more than 0.2; the steepest feature drawn (the cold front's Cb line,
    0.08 r wide) changes by under 0.1 a sample."""
    low = _system(Kind.LOW, 50.0, stage=0.45, variant=variant)
    n = 801
    x, y = np.meshgrid(np.linspace(-4.0, 4.0, n), np.linspace(-4.0, 4.0, n))
    dirs = _points(low, np.stack([x.ravel(), y.ravel()], -1))
    out = _apply(low, dirs, sea=1.0, temperature=12.0, base=0.0)
    for name, cover in [("convective", out.convective)] + list(out.layers.items()):
        grid = cover.reshape(n, n)
        jump = max(
            np.abs(np.diff(grid, axis=0)).max(), np.abs(np.diff(grid, axis=1)).max()
        )
        assert jump < 0.2, (name, jump)
