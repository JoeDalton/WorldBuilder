"""
The planets the tests draw on: the library keeps no presets, a world file
defines every planet (src/world_builder/worlds/). These are the specs the
tests were written around, ``earth``, ``desert`` and ``forest`` of them also
shipped as body files (checked in test_world_file.py).
"""

from world_builder.terrain.planet import ClimateParams, ElevationParams, PlanetSpec

PRESETS: dict[str, PlanetSpec] = {
    "earthlike": PlanetSpec(name="earthlike"),
    # Earth's radius and sidereal day, the earthlike world's climate: the
    # weather's scales come out Earth's (clouds.synoptic.Scales).
    "earth": PlanetSpec(name="earth", radius=6.371e6, day_length_h=23.934),
    "desert": PlanetSpec(
        name="desert",
        has_ocean=False,
        elevation=ElevationParams(
            ocean_fraction=0.3,
            mountain_height=2700.0,
            mountain_coverage=-0.1,
            hills_height=220.0,
            detail_height=25.0,
        ),
        climate=ClimateParams(
            temperature_mean=31.0,
            temperature_spread=6.0,
            temperature_latitude=20.0,
            humidity_mean=0.08,
            humidity_spread=0.1,
        ),
    ),
    "forest": PlanetSpec(
        name="forest",
        elevation=ElevationParams(
            ocean_fraction=0.35, mountain_height=3800.0, hills_height=420.0
        ),
        climate=ClimateParams(
            temperature_mean=17.0,
            temperature_spread=5.0,
            temperature_latitude=22.0,
            humidity_mean=0.78,
            humidity_spread=0.18,
        ),
    ),
    # A Mun-sized, airless ball of sand: dunes and crater-like basins (the
    # continental lows, dry), rock only on cliffs. No sea, no vegetation, no
    # climate: hot and bone dry everywhere, no snow even on the peaks.
    "moon": PlanetSpec(
        name="moon",
        radius=200e3,
        has_ocean=False,
        has_vegetation=False,
        elevation=ElevationParams(
            ocean_fraction=0.4,
            plates=0,
            dunes=False,
            tablelands=False,
            mountain_coverage=-0.1,
            hills_height=260.0,
            detail_height=30.0,
        ),
        climate=ClimateParams(
            temperature_mean=30.0,
            temperature_spread=0.0,
            temperature_latitude=0.0,
            lapse_rate=0.0,
            humidity_mean=0.0,
            humidity_spread=0.0,
        ),
    ),
    # A 20 km test ball for the vegetation: no clouds, gentle relief, and
    # forest, grassland and desert within a few kilometres of each other (a
    # short humidity wavelength over a warm climate). Small enough to load in
    # seconds, for benchmarks and visual checks.
    "tiny": PlanetSpec(
        name="tiny",
        radius=20e3,
        has_clouds=False,
        elevation=ElevationParams(
            ocean_fraction=0.25,
            plates=0,
            continent_curve=(
                (-1.0, -900.0),
                (-0.35, -600.0),
                (-0.12, -250.0),
                (-0.04, -20.0),
                (0.0, 0.0),
                (0.04, 8.0),
                (0.15, 40.0),
                (0.35, 120.0),
                (1.0, 300.0),
            ),
            mountain_height=1000.0,
            mountain_coverage=-0.25,
            dunes=False,
            tablelands=False,
            hills_wavelength=3000.0,
            hills_height=60.0,
            detail_wavelength=400.0,
            detail_height=8.0,
        ),
        climate=ClimateParams(
            temperature_mean=19.0,
            temperature_spread=4.0,
            temperature_latitude=6.0,
            lapse_rate=0.0065,
            humidity_mean=0.42,
            humidity_spread=0.4,
            temperature_wavelength=60e3,
            humidity_wavelength=24e3,
            humidity_octaves=3,
        ),
    ),
    "ice": PlanetSpec(
        name="ice",
        has_ocean=False,
        elevation=ElevationParams(ocean_fraction=0.4, mountain_coverage=0.1),
        climate=ClimateParams(
            temperature_mean=-9.0,
            temperature_spread=8.0,
            temperature_latitude=20.0,
            humidity_mean=0.4,
            humidity_spread=0.2,
        ),
    ),
}
