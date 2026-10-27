"""
Every quality vs performance knob of a world, in one object a game's
settings menu can show and save (:mod:`world_builder.quality`), without the
game naming any knob or WorldBuilder class.

Usage::

    from world_builder.settings import WorldSettings

    settings = WorldSettings.from_dict(config.get("world"))  # lenient
    # Build each object with its section, found by its type:
    terrain = Terrain(render, spec, settings=settings.settings_for(Terrain))
    ...
    # A menu, from plain data: the levels, then each section's knobs.
    schema = WorldSettings.schema()
    for section in schema["sections"]:
        for knob in section["knobs"]:
            ...  # knob["label"], ["kind"], ["range"] / ["choices"], ["apply"]
    settings = settings.with_value(section["id"], knob["id"], value)
    settings.apply(terrain, atmospheres, sky, clouds, ocean)  # each its own
    config["world"] = settings.to_dict()  # {"version", "preset", "overrides"}

A preset (:meth:`WorldSettings.preset`) sets every section to the same
level; changing one knob afterwards makes it "custom" (:meth:`is_custom`),
saved as the preset plus that override.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from world_builder.asteroids.settings import AsteroidSettings
from world_builder.atmosphere import AtmosphereSettings
from world_builder.clouds.cloud import CloudSettings
from world_builder.ocean.ocean import OceanSettings
from world_builder.quality import Knob, QualityLevel, QualitySettings
from world_builder.shadows import ShadowSettings
from world_builder.sky.sky import SkySettings
from world_builder.terrain.terrain import TerrainSettings

LOGGER = logging.getLogger(__name__)

#: The saved format's version (:meth:`WorldSettings.to_dict`). Data without
#: one, or from an older version, is read the same: knob ids only change
#: through ``QualitySettings.RENAMED``.
FORMAT_VERSION = 1

#: The sections, in menu order: name -> settings class.
SECTIONS: dict[str, type[QualitySettings]] = {
    "terrain": TerrainSettings,
    "atmosphere": AtmosphereSettings,
    "sky": SkySettings,
    "clouds": CloudSettings,
    "ocean": OceanSettings,
    "asteroids": AsteroidSettings,
    "shadows": ShadowSettings,
}


@dataclass(frozen=True)
class WorldSettings:
    """What a whole world costs: a preset level and each part's settings.

    :ivar level: the preset the sections started from (:meth:`is_custom`
        tells whether they still are it)
    """

    level: QualityLevel = QualityLevel.HIGH
    terrain: TerrainSettings = field(default_factory=TerrainSettings)
    atmosphere: AtmosphereSettings = field(default_factory=AtmosphereSettings)
    sky: SkySettings = field(default_factory=SkySettings)
    clouds: CloudSettings = field(default_factory=CloudSettings)
    ocean: OceanSettings = field(default_factory=OceanSettings)
    asteroids: AsteroidSettings = field(default_factory=AsteroidSettings)
    shadows: ShadowSettings = field(default_factory=ShadowSettings)

    @classmethod
    def preset(cls, level: QualityLevel | str) -> WorldSettings:
        """:returns: every section at one level (or its name)"""
        level = QualityLevel.parse(level)
        return cls(
            level=level,
            **{name: kind.preset(level) for name, kind in SECTIONS.items()},
        )

    @staticmethod
    def schema() -> dict[str, Any]:
        """The whole world's knobs as plain data (JSON-ready), for a game to
        build its menu from without naming any.

        :returns: ``version``; ``levels`` ([{``id``, ``label``}], cheapest
            first); ``sections`` ([{``id``, ``label``, ``knobs``}], each knob
            as :meth:`quality.QualitySettings.schema` describes it)
        """
        return {
            "version": FORMAT_VERSION,
            "levels": [{"id": lv.value, "label": lv.label} for lv in QualityLevel],
            "sections": [
                {"id": name, "label": kind.LABEL, "knobs": kind.schema()}
                for name, kind in SECTIONS.items()
            ],
        }

    def settings_for(self, target: Any) -> QualitySettings:
        """The section a WorldBuilder object (or class) takes, found by its
        ``SETTINGS_TYPE``: to build it with (``settings=``).

        :raises TypeError: for an object that takes no world settings
        """
        kind = getattr(target, "SETTINGS_TYPE", None)
        for name, section in SECTIONS.items():
            if kind is section:
                return getattr(self, name)
        raise TypeError(f"{target!r} takes no world settings")

    def apply(self, *objects: Any) -> None:
        """Give each object its section (``set_settings``), whatever its
        type: a game hands over the WorldBuilder objects it made, and need
        not know which section each takes. A section that did not change
        costs nothing.

        :raises TypeError: for an object that takes no world settings
        """
        for target in objects:
            settings = self.settings_for(target)  # first: it names the error
            target.set_settings(settings)

    def is_custom(self) -> bool:
        """:returns: whether a knob differs from :attr:`level`'s preset"""
        return self != WorldSettings.preset(self.level)

    def knobs(self) -> Iterator[tuple[str, str, Knob, Any]]:
        """:returns: (section, field name, its knob, its value), section by
        section in :data:`SECTIONS` order: a menu's rows"""
        for section in SECTIONS:
            settings = getattr(self, section)
            for name, knob in settings.knobs():
                yield section, name, knob, getattr(settings, name)

    def with_value(self, section: str, name: str, value: Any) -> WorldSettings:
        """:returns: these settings with one knob changed, leniently (a
        menu's slider: clamped to its range, or kept with a warning)"""
        if section not in SECTIONS:
            LOGGER.warning(f"unknown settings section {section!r}, ignored")
            return self
        current = getattr(self, section)
        changed = type(current).from_overrides(current, {name: value}, section)
        return dataclasses.replace(self, **{section: changed})

    def section_level(self, section: str) -> QualityLevel | None:
        """:returns: the preset one section is; None = custom"""
        settings = getattr(self, section)
        return type(settings).level_of(settings)

    def with_section_level(
        self, section: str, level: QualityLevel | str
    ) -> WorldSettings:
        """:returns: these settings with one section at a preset (a menu's
        per-part preset selector)"""
        return dataclasses.replace(self, **{section: SECTIONS[section].preset(level)})

    def with_level(self, level: QualityLevel | str) -> WorldSettings:
        """:returns: the preset *level* (a menu's preset selector)"""
        return WorldSettings.preset(level)

    def to_dict(self) -> dict[str, Any]:
        """:returns: plain data for a config file: the preset's name and the
        knobs that differ from it, by section (choices by label)"""
        base = WorldSettings.preset(self.level)
        overrides = {}
        for section in SECTIONS:
            changed = getattr(self, section).overrides(getattr(base, section))
            if changed:
                overrides[section] = changed
        return {
            "version": FORMAT_VERSION,
            "preset": self.level.value,
            "overrides": overrides,
        }

    @classmethod
    def from_dict(cls, data: Any) -> WorldSettings:
        """Settings from :meth:`to_dict`'s data, leniently: an unknown preset
        is HIGH, an unknown section or knob is skipped, a bad value keeps the
        preset's, an out-of-range one is clamped (each with a warning;
        never an exception).

        :param data: {"version": n, "preset": name, "overrides": {section:
            {knob: value}}} (as :meth:`to_dict` writes it)
        """
        if data is None:
            return cls.preset(QualityLevel.HIGH)
        if not isinstance(data, Mapping):
            LOGGER.warning("world settings: not a mapping, using high")
            return cls.preset(QualityLevel.HIGH)
        settings = cls.preset(QualityLevel.parse(data.get("preset")))
        overrides = data.get("overrides") or {}
        if not isinstance(overrides, Mapping):
            LOGGER.warning("world settings overrides: not a mapping, ignored")
            return settings
        sections = {}
        for section, values in overrides.items():
            kind = SECTIONS.get(section)
            if kind is None:
                LOGGER.warning(f"unknown settings section {section!r}, ignored")
                continue
            sections[section] = kind.from_overrides(
                getattr(settings, section), values, section
            )
        return dataclasses.replace(settings, **sections)
