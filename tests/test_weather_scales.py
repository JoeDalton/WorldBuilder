"""The synoptic weather's scales follow the planet's radius and rotation
(docs/literature/synoptic-weather-review.tex, "Planetary scales")."""

from dataclasses import replace

from world_builder.clouds.synoptic import Scales
from world_builder.terrain.planet import PlanetSpec

EARTH = PlanetSpec("e", radius=6.371e6, day_length_h=23.934)
JUP = PlanetSpec("j", radius=69.911e6, day_length_h=9.925)
KERBIN = PlanetSpec("k")  # 600 km, 6 h


def test_earth_anchors():
    s = Scales.from_planet(EARTH)
    assert 800e3 < s.deformation_radius < 1200e3
    assert 5.0 < s.wavenumber < 9.0
    assert 27.0 < s.hadley_edge < 33.0
    assert len(s.jet_lats) == 2 and 25 < s.jet_lats[0] < 35 and 40 < s.jet_lats[1] < 60
    assert 5.0 <= s.tc_band[0] and s.tc_band[1] <= 30.0


def test_faster_spin_smaller_storms_more_jets():
    slow = Scales.from_planet(EARTH)
    fast = Scales.from_planet(replace(EARTH, day_length_h=10.0))
    assert fast.deformation_radius < slow.deformation_radius
    assert len(fast.jet_lats) >= len(slow.jet_lats)
    assert fast.hadley_edge < slow.hadley_edge


def test_jupiter_many_jets_kerbin_broad_hadley():
    assert len(Scales.from_planet(JUP).jet_lats) >= 4
    k = Scales.from_planet(KERBIN)
    assert k.hadley_edge > 40.0 and k.wavenumber < 4.0


def test_belts_ordered():
    for p in (EARTH, JUP, KERBIN):
        s = Scales.from_planet(p)
        assert (
            0
            < s.tc_band[1]
            <= s.hadley_edge
            <= s.storm_track[0]
            < s.storm_track[1]
            <= 85
        )


def test_one_storm_band_per_eddy_jet():
    for p in (EARTH, JUP, KERBIN):
        s = Scales.from_planet(p)
        eddy = s.jet_lats[1:]
        assert len(s.storm_bands) == len(eddy)
        for (lo, hi), jet in zip(s.storm_bands, eddy):
            assert s.hadley_edge <= lo < jet < hi <= 85.0
        assert s.storm_track == (s.storm_bands[0][0], s.storm_bands[-1][1])
    jup = Scales.from_planet(JUP)
    assert len(jup.storm_bands) >= 4 and jup.storm_track[1] > 60.0
    earth = Scales.from_planet(EARTH)
    assert 35.0 <= earth.storm_track[0] <= 45.0 and 60.0 <= earth.storm_track[1] <= 70.0
