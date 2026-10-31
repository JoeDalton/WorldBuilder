"""
The quality vs performance settings (world_builder.quality, .settings):
the shared level enum, every section's knobs and presets, and the lenient
round trip a game's settings file goes through. Pure Python: no window.
"""

import json
import logging
import math
from dataclasses import fields, replace

import pytest

from world_builder.atmosphere import AtmosphereSettings
from world_builder.clouds.cloud import PRESETS, CloudSettings, CloudType, with_settings
from world_builder.quality import (
    APPLY_MODES,
    Knob,
    QualityLevel,
    knob,
)
from world_builder.settings import FORMAT_VERSION, SECTIONS, WorldSettings
from world_builder.terrain.terrain import TerrainSettings

LEVELS = list(QualityLevel)
#: Per section, the fields whose cost must not fall from LOW to ULTRA (bools:
#: off counts as cheaper).
COST_FIELDS = {
    "terrain": [
        "split_factor",
        "detail_far",
        "anisotropy",
        "reflected",
        "node_cache_size",
        "store_size",
        "uploads_per_frame",
        "upload_ms_per_frame",
        "texture_size",
    ],
    "atmosphere": ["steps", "vertex_steps"],
    "sky": ["star_fraction", "glow_size"],
    "clouds": [
        "count_scale",
        "march_scale",
        "overlap_near",
        "overlap_far",
        "reflected",
    ],
    "ocean": [
        "wave_iterations",
        "rough_fresnel",
        "reflection_blur",
        "reflection_scale",
        "foam_noise",
    ],
    "asteroids": [
        "max_lod",
        "normal_size",
        "instance_budget",
        "held_budget",
        "detail_far",
        "texture_size",
        "max_terrain_rocks",
    ],
}


def test_the_levels_go_cheapest_first():
    """A menu's slider stops rely on "drag right costs more"."""
    assert [q.value for q in QualityLevel] == ["low", "mid", "high", "ultra"]
    assert [q.label for q in QualityLevel] == ["Low", "Mid", "High", "Ultra"]


@pytest.mark.parametrize("value", ["High", " low ", QualityLevel.MID, "ULTRA"])
def test_a_level_parses_leniently(value):
    assert isinstance(QualityLevel.parse(value), QualityLevel)


def test_an_unknown_level_is_high_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert QualityLevel.parse("fancy") is QualityLevel.HIGH
    assert "fancy" in caplog.text
    assert QualityLevel.parse(None) is QualityLevel.HIGH


@pytest.mark.parametrize("section", list(SECTIONS))
def test_every_field_is_a_described_knob(section):
    kind = SECTIONS[section]
    for f in fields(kind):
        k = f.metadata.get("knob")
        assert isinstance(k, Knob), f.name
        assert k.label and k.apply in APPLY_MODES
        assert not (k.range and k.choices), f.name


@pytest.mark.parametrize("section", list(SECTIONS))
def test_the_defaults_are_the_high_preset(section):
    kind = SECTIONS[section]
    assert kind() == kind.preset(QualityLevel.HIGH) == kind.preset("high")
    assert kind.level_of(kind()) is QualityLevel.HIGH


@pytest.mark.parametrize("section", list(SECTIONS))
@pytest.mark.parametrize("level", LEVELS)
def test_presets_are_within_their_knobs(section, level):
    """A preset builds (the constructor is strict), and its values are in
    their ranges and choices."""
    settings = SECTIONS[section].preset(level)
    for name, k in settings.knobs():
        value = getattr(settings, name)
        if k.range:
            assert k.range[0] <= value <= k.range[1], name
        if k.choices:
            assert value in k.choices.values(), name


@pytest.mark.parametrize("section", list(SECTIONS))
def test_presets_are_distinct(section):
    """A preset must be recognisable from its settings (a menu shows which
    one a part is at, and cycling must move on)."""
    kind = SECTIONS[section]
    presets = [kind.preset(level) for level in LEVELS]
    assert len(set(presets)) == len(presets)
    assert [kind.level_of(p) for p in presets] == LEVELS


@pytest.mark.parametrize("section", list(SECTIONS))
def test_presets_order_their_cost(section):
    presets = [SECTIONS[section].preset(level) for level in LEVELS]
    for name in COST_FIELDS[section]:
        values = [getattr(p, name) for p in presets]
        assert values == sorted(values), (name, values)


def test_high_clouds_are_the_authored_types():
    spec = PRESETS[CloudType.CUMULUS]
    assert with_settings(spec, CloudSettings.preset(QualityLevel.HIGH)) is spec


def test_lower_presets_drop_terrain_detail():
    """MID drops one level of the finest LOD, LOW two more."""
    dropped = [TerrainSettings.preset(level).min_level for level in LEVELS]
    assert dropped == [4, 2, 0, 0]


def test_the_constructor_is_strict():
    with pytest.raises(ValueError):
        AtmosphereSettings(steps=100)  # out of range
    with pytest.raises(ValueError):
        AtmosphereSettings(steps=2.5)  # not whole
    with pytest.raises(ValueError):
        CloudSettings(shell_octaves=(9, 9))  # not a choice
    with pytest.raises(ValueError):
        CloudSettings(reflected=1)  # not a bool
    assert isinstance(AtmosphereSettings(steps=6).steps, int)
    assert CloudSettings(count_scale=1).count_scale == 1.0  # an int for a float


def test_changes_name_how_each_applies():
    base = CloudSettings()
    new = replace(base, count_scale=0.5, march_scale=0.5)
    assert base.changes(new) == {"count_scale": "rebuild", "march_scale": "live"}
    assert base.changes(base) == {}


def test_loading_is_lenient(caplog):
    base = AtmosphereSettings()
    with caplog.at_level(logging.WARNING):
        loaded = AtmosphereSettings.from_overrides(
            base, {"steps": 99, "vertex_steps": "lots", "shiny": True}, "atmosphere"
        )
    assert loaded.steps == 16  # clamped
    assert loaded.vertex_steps == base.vertex_steps  # kept
    assert "atmosphere.shiny" in caplog.text
    assert AtmosphereSettings.from_overrides(base, "nonsense") == base
    assert AtmosphereSettings.from_overrides(base, None) == base


def test_a_choice_is_saved_and_loaded_by_its_label():
    s = CloudSettings.preset("low")
    data = s.overrides(CloudSettings())
    assert data["shell_octaves"] == "4-3-3-2"
    assert CloudSettings.from_overrides(CloudSettings(), data) == s


@pytest.mark.parametrize("level", LEVELS)
def test_world_presets_round_trip(level):
    settings = WorldSettings.preset(level)
    data = settings.to_dict()
    assert data == {"version": FORMAT_VERSION, "preset": level.value, "overrides": {}}
    assert not settings.is_custom()
    assert WorldSettings.from_dict(json.loads(json.dumps(data))) == settings


def test_a_custom_world_round_trips_through_json():
    settings = (
        WorldSettings.preset("mid")
        .with_value("terrain", "min_level", 3)
        .with_value("ocean", "wave_iterations", 20)
        .with_value("clouds", "shell_octaves", "4")
        .with_value("sky", "star_fraction", 0.25)
    )
    assert settings.is_custom()
    data = json.loads(json.dumps(settings.to_dict()))
    assert data["preset"] == "mid"
    assert set(data["overrides"]) == {"terrain", "ocean", "clouds", "sky"}
    assert WorldSettings.from_dict(data) == settings


@pytest.mark.parametrize(
    "data",
    [
        None,
        "high",
        {"preset": "bogus"},
        {"preset": "low", "overrides": "nope"},
        {"preset": "low", "overrides": {"planets": {"x": 1}}},
        {"overrides": {"terrain": {"anisotropy": "3x", "detail_far": "far"}}},
        {"overrides": {"terrain": {"min_level": 1e9, "textured": "yes"}}},
    ],
)
def test_bad_world_data_never_raises(data, caplog):
    with caplog.at_level(logging.WARNING):
        settings = WorldSettings.from_dict(data)
    assert isinstance(settings, WorldSettings)


def test_world_knobs_list_every_field_once():
    rows = list(WorldSettings().knobs())
    expected = sum(len(fields(kind)) for kind in SECTIONS.values())
    assert len(rows) == expected
    assert len({(section, name) for section, name, _, _ in rows}) == expected


def test_a_menu_slider_is_clamped():
    settings = WorldSettings().with_value("terrain", "detail_far", 1e9)
    assert settings.terrain.detail_far == 10000.0
    assert WorldSettings().with_value("nowhere", "x", 1) == WorldSettings()


def test_a_knob_field_carries_its_metadata():
    f = knob(3, "Things", unit="m", range=(1, 5))
    k = f.metadata["knob"]
    assert f.default == 3 and k.unit == "m" and k.apply == "live"
    with pytest.raises(ValueError):
        Knob("x", apply="sometimes")


def test_atmosphere_steps_cost_samples_per_pixel():
    """The air's steps are the per-pixel march: LOW halves HIGH's."""
    low, high = (AtmosphereSettings.preset(level).steps for level in ("low", "high"))
    assert math.isclose(low / high, 0.5)


# ── Discoverable by a game, without naming a knob ──────────────────────────────


def test_the_schema_is_plain_data_for_every_knob():
    schema = WorldSettings.schema()
    assert json.loads(json.dumps(schema)) == schema
    assert schema["version"] == FORMAT_VERSION
    assert [lv["id"] for lv in schema["levels"]] == [lv.value for lv in LEVELS]
    assert [sec["id"] for sec in schema["sections"]] == list(SECTIONS)
    for section in schema["sections"]:
        assert section["label"]
        kind = SECTIONS[section["id"]]
        assert [k["id"] for k in section["knobs"]] == [f.name for f in fields(kind)]
        for k in section["knobs"]:
            assert k["kind"] in ("bool", "int", "float", "choice")
            assert (k["kind"] == "choice") == (k["choices"] is not None)
            assert set(k["presets"]) == {lv.value for lv in LEVELS}
            assert k["presets"]["high"] == k["default"]


def test_a_menu_built_from_the_schema_round_trips():
    """What a game does without naming any knob: walk the schema, set every
    knob to its LOW value as the schema gives it, save, load."""
    settings = WorldSettings()
    for section in WorldSettings.schema()["sections"]:
        for k in section["knobs"]:
            settings = settings.with_value(section["id"], k["id"], k["presets"]["low"])
    assert WorldSettings.from_dict(settings.to_dict()) == settings
    low = WorldSettings.preset("low")
    for name in SECTIONS:
        assert getattr(settings, name) == getattr(low, name)


def test_a_renamed_knob_reads_its_old_id(monkeypatch):
    monkeypatch.setattr(AtmosphereSettings, "RENAMED", {"samples": "steps"})
    loaded = AtmosphereSettings.from_overrides(AtmosphereSettings(), {"samples": 6})
    assert loaded.steps == 6


def test_data_without_a_version_still_loads():
    data = {"preset": "low", "overrides": {"terrain": {"detail_far": 2000.0}}}
    assert WorldSettings.from_dict(data).terrain.detail_far == 2000.0


class _Part:
    def __init__(self, kind):
        self.SETTINGS_TYPE = kind
        self.got = None

    def set_settings(self, settings):
        self.got = settings


def test_apply_finds_each_objects_section_by_its_type():
    world = WorldSettings.preset("mid")
    parts = [_Part(kind) for kind in SECTIONS.values()]
    world.apply(*parts)
    for part, name in zip(parts, SECTIONS):
        assert part.got == getattr(world, name)
    assert world.settings_for(parts[0]) == world.terrain
    with pytest.raises(TypeError):
        world.apply(object())


def test_every_world_builder_object_declares_its_settings():
    from world_builder.atmosphere import Atmospheres
    from world_builder.clouds.cirrus import CloudSheet
    from world_builder.clouds.planet import PlanetClouds
    from world_builder.clouds.sphere import CloudVolume
    from world_builder.ocean import Sea
    from world_builder.sky import Sky
    from world_builder.terrain.terrain import Terrain

    world = WorldSettings.preset("low")
    for kind, section in (
        (Terrain, "terrain"),
        (Atmospheres, "atmosphere"),
        (Sky, "sky"),
        (PlanetClouds, "clouds"),
        (CloudVolume, "clouds"),
        (CloudSheet, "clouds"),
        (Sea, "ocean"),
    ):
        assert world.settings_for(kind) == getattr(world, section)
