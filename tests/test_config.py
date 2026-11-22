"""
Config files (world_builder.config): the looks' fields read from and written
to YAML, every knob exposed, strictly; and the links between files.
"""

import math
from dataclasses import replace

import pytest
import yaml
from fields import PRESETS as FIELDS
from planets import PRESETS

from world_builder.asteroids.distributions import Distribution
from world_builder.asteroids.spec import Annulus, FieldSpec
from world_builder.atmosphere import AtmosphereParams
from world_builder.clouds.genera import GeneraSpec
from world_builder.clouds.synoptic.presets import Preset, WeatherPin
from world_builder.clouds.weather import WeatherSpec
from world_builder.config import (
    ConfigError,
    Context,
    dump_yaml,
    expand,
    from_config,
    to_config,
)
from world_builder.terrain.stamps import Paint, Placement, Stamp
from world_builder.world import CloudLook, OceanLook, SkyLook

PLANET = PRESETS["earthlike"]
PLACE = {"lat": 0, "lon": 0, "width": 1, "height": 1}


def through_yaml(cls, obj, **kwargs):
    """*obj* written as YAML text and read back."""
    text = dump_yaml(to_config(obj, **kwargs))
    return from_config(cls, yaml.safe_load(text), **kwargs)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_planet_preset_round_trips(name):
    spec = PRESETS[name]
    assert through_yaml(type(spec), spec) == spec.resolved()


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_every_field_preset_round_trips(name):
    assert through_yaml(FieldSpec, FIELDS[name]) == FIELDS[name]


@pytest.mark.parametrize("obj", [AtmosphereParams(), OceanLook(), SkyLook()])
def test_looks_round_trip(obj):
    assert through_yaml(type(obj), obj) == obj


def test_clouds_with_weather_pins_and_genera_round_trip():
    weather = WeatherSpec(
        PLANET,
        seed=4,
        pins=(WeatherPin(10.0, 20.0, Preset.HURRICANE, 30.0),),
        zonal=((0.0, 0.5), (60.0, 0.2)),
    )
    ctx = Context(planet=PLANET)
    for look in (
        CloudLook(altitude=2200.0, cirrus=True, weather=weather),
        CloudLook(genera=GeneraSpec(weather=WeatherSpec(PLANET, seed=11))),
        CloudLook(
            weather=weather, genera=GeneraSpec(weather=WeatherSpec(PLANET, seed=9))
        ),
    ):
        config = to_config(look, ctx=Context(planet=PLANET))
        assert "planet" not in str(config.get("weather", ""))
        back = from_config(CloudLook, yaml.safe_load(dump_yaml(config)), ctx=ctx)
        assert back == look


def test_stamps_and_paints_keep_their_images_relative_to_the_file(tmp_path):
    spec = replace(
        PLANET,
        stamps=(Stamp(str(tmp_path / "s" / "a.png"), Placement(1.0, 2.0, 3.0, 4.0)),),
        paints=(
            Paint(
                str(tmp_path / "s" / "m.png"),
                Placement(1.0, 2.0, 3.0, 4.0, 15.0),
                (("#808080", "rock"),),
            ),
        ),
    )
    config = to_config(spec, tmp_path)
    assert config["stamps"][0]["heightmap"] == "s/a.png"
    assert from_config(PLANET.__class__, config, tmp_path) == spec.resolved()


def test_numbers_may_be_written_as_yaml_reads_them_as_text():
    """PyYAML reads 1e6 (no dot, no sign) as a string."""
    assert yaml.safe_load("1e6") == "1e6"
    config = {"inner": "5e6", "outer": 1e7, "thickness": 4000}
    assert from_config(Annulus, config) == Annulus(5e6, 1e7, 4000.0)


def test_omitted_fields_take_their_defaults():
    assert from_config(OceanLook, {"swell_amplitude": 3}) == OceanLook(
        swell_amplitude=3.0
    )


@pytest.mark.parametrize(
    "cls,config,message",
    [
        (OceanLook, {"swel_amplitude": 1.0}, "unknown keys.*swel_amplitude"),
        (Annulus, {"inner": 1.0, "outer": 2.0}, "thickness: is required"),
        (Annulus, {"inner": 3.0, "outer": 2.0, "thickness": 1.0}, "inner"),
        (OceanLook, {"swell_amplitude": "big"}, "not a number"),
        (OceanLook, {"wind_dir": [1.0]}, "needs 2 values"),
        (CloudLook, {"cloud_type": "plasma"}, "not one of"),
        (PLANET.__class__, {"name": "x", "sea_bias": 1.0}, "computed"),
        (
            Stamp,
            {"heightmap": "a.png", "placement": PLACE, "blend": "melt"},
            "blend",
        ),
        (FieldSpec, {"shape": {"type": "cube"}}, "type 'cube'"),
        (FieldSpec, {"shape": {"inner": 1}}, "needs a type"),
        (
            FieldSpec,
            {
                "shape": {"type": "sphere", "radius": 1.0},
                "density": {"count_per_km3": 1.0},
                "sizes": {"kind": "zipf"},
            },
            "kind 'zipf'",
        ),
    ],
)
def test_errors_say_where(cls, config, message):
    with pytest.raises(ConfigError, match=message):
        from_config(cls, config)


def test_errors_are_value_errors():
    assert issubclass(ConfigError, ValueError)


def test_a_union_is_chosen_by_its_type_key():
    field = from_config(
        FieldSpec,
        {
            "shape": {"type": "box", "half_size": [1, 2, 3]},
            "density": {"count_per_km3": 1.0},
            "sizes": {"kind": "uniform", "lo": 1, "hi": 2},
            "spin": {
                "rate": {"kind": "uniform", "lo": 0.1, "hi": 0.2},
                "axis": {"type": "around_axis", "axis": [0, 0, 1], "kappa": 4},
            },
        },
    )
    assert field.shape.half_size == (1.0, 2.0, 3.0)
    assert field.spin.axis.kappa == 4.0
    assert field.sizes == Distribution.uniform(1.0, 2.0)


# ── links ────────────────────────────────────────────────────────────────────


def test_a_file_link_is_the_file_with_the_keys_beside_it_on_top(tmp_path):
    (tmp_path / "sea.yaml").write_text("swell_amplitude: 5\nfoam_depth: 1\n")
    data = {"ocean": {"file": "sea.yaml", "foam_depth": 2}}
    ocean = from_config(OceanLook, expand(data, tmp_path)["ocean"], tmp_path)
    assert (ocean.swell_amplitude, ocean.foam_depth) == (5.0, 2.0)


def test_links_nest(tmp_path):
    (tmp_path / "a.yaml").write_text("x: {file: b.yaml}\n")
    (tmp_path / "b.yaml").write_text("y: 1\n")
    assert expand({"file": "a.yaml"}, tmp_path) == {"x": {"y": 1}}


@pytest.mark.parametrize("name", ["../a.yaml", "sub/a.yaml", "/etc/a.yaml", ""])
def test_a_link_is_a_file_of_the_same_folder(tmp_path, name):
    with pytest.raises(ConfigError, match="plain file name"):
        expand({"file": name}, tmp_path)


def test_a_missing_link_and_a_circle_are_errors(tmp_path):
    with pytest.raises(ConfigError, match="gone.yaml"):
        expand({"file": "gone.yaml"}, tmp_path)
    (tmp_path / "a.yaml").write_text("x: {file: b.yaml}\n")
    (tmp_path / "b.yaml").write_text("x: {file: a.yaml}\n")
    with pytest.raises(ConfigError, match="circle"):
        expand({"file": "a.yaml"}, tmp_path)


def test_distribution_round_trips_every_kind():
    for d in (
        Distribution.uniform(1.0, 2.0),
        Distribution.normal(1.5, 0.2, 1.0, 2.0),
        Distribution.lognormal(1.5, 0.2, 1.0, 2.0),
        Distribution.power_law(3.0, 1.0, 2.0),
        Distribution.u_shaped(0.5, 1.0, 2.0, log=False),
        Distribution.table([1.0, 2.0, 3.0], [1.0, 0.5, 0.1]),
    ):
        assert from_config(Distribution, to_config(d)) == d
    assert math.isfinite(Distribution.power_law(3.0, 1.0, 2.0).hi)
