"""
Quality vs performance settings: one shape for every part of a world.

Each drawn part has a frozen settings dataclass (``TerrainSettings``,
``AtmosphereSettings``, ``SkySettings``, ``CloudSettings``,
``OceanSettings``), bundled in :class:`world_builder.settings.WorldSettings`.
Every field is one knob a game's settings menu can show on its own: it
carries a :class:`Knob` (its label, range or choices, unit, and how a change
applies). Each class also has LOW / MID / HIGH / ULTRA presets
(:class:`QualityLevel`); its field defaults are the HIGH ones.

The objects take ``settings=`` and change with ``set_settings()``:
:data:`LIVE` fields apply at once, :data:`REBUILD` ones make the object
rebuild a part of itself (which may show, e.g. clouds refilling), and
:data:`RESTART` ones only at the next construction.

Building a settings object by hand is strict: a bad value raises.
Reading one from data (:meth:`QualitySettings.from_overrides`, a saved
config) is lenient: out-of-range values are clamped, bad ones keep the base
value, unknown names are skipped, each with a warning, never an exception.

A game should not name the knobs: :meth:`QualitySettings.schema` (and
:meth:`world_builder.settings.WorldSettings.schema` for a whole world)
describes every one as plain data — id, label, kind, range or choices,
unit, how it applies, default and per-level values — to build a menu
from, so knobs WorldBuilder adds later appear in it unchanged. The ids are
what saved settings are keyed by; a renamed knob keeps reading its old id
(:attr:`QualitySettings.RENAMED`).
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, TypeVar

LOGGER = logging.getLogger(__name__)

#: Applies at once.
LIVE = "live"
#: The object rebuilds a part of itself (a short refill may show).
REBUILD = "rebuild"
#: Only at the next construction (e.g. worker processes, texture loading).
RESTART = "restart"
APPLY_MODES = (LIVE, REBUILD, RESTART)


class QualityLevel(Enum):
    """The presets every settings class has, cheapest first."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"
    ULTRA = "ultra"

    @property
    def label(self) -> str:
        """:returns: its name in a menu ("Low", ...)"""
        return self.value.title()

    @classmethod
    def parse(cls, value: Any, default: QualityLevel | None = None) -> QualityLevel:
        """A level from a config value, leniently.

        :param value: a level, or its name in any case ("High", " low")
        :param default: what an unknown value gives; None = HIGH
        :returns: the level; *default* (with a warning) for anything else
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            fallback = cls.HIGH if default is None else default
            if value is not None:
                LOGGER.warning(
                    f"unknown quality level {value!r}; using {fallback.value}"
                )
            return fallback


@dataclass(frozen=True)
class Knob:
    """What a settings menu needs to show one field.

    :ivar label: its name in a menu
    :ivar apply: :data:`LIVE`, :data:`REBUILD` or :data:`RESTART`
    :ivar unit: shown after the value ("m", "px", ...); "" = none
    :ivar range: (lowest, highest) for a number; None = none
    :ivar choices: menu label -> value, for a field with a fixed set of
        values (saved by label); None = free
    :ivar advanced: a menu may hide it behind an "advanced" switch
    :ivar help: a sentence on what it trades
    :ivar kept: a choice rather than a level of quality (which renderer):
        moving to a preset leaves it as it is
        (:meth:`QualitySettings.at_level`), and it does not make the
        settings custom (:meth:`QualitySettings.level_of`)
    """

    label: str
    apply: str = LIVE
    unit: str = ""
    range: tuple[float, float] | None = None
    choices: Mapping[str, Any] | None = None
    advanced: bool = False
    help: str = ""
    kept: bool = False

    def __post_init__(self):
        if self.apply not in APPLY_MODES:
            raise ValueError(f"unknown apply mode {self.apply!r}")

    def label_of(self, value: Any) -> str | None:
        """:returns: the choice label of *value*; None if it is not one"""
        for label, choice in (self.choices or {}).items():
            if choice == value:
                return label
        return None


def knob(default: Any, label: str, apply: str = LIVE, **meta) -> Any:
    """A settings dataclass field with its :class:`Knob`.

    :param default: its HIGH value
    :param label: its name in a menu
    :param apply: how a change applies
    :param meta: the other :class:`Knob` fields
    """
    return field(default=default, metadata={"knob": Knob(label, apply, **meta)})


S = TypeVar("S", bound="QualitySettings")


class QualitySettings:
    """What every settings dataclass shares (a mixin for frozen dataclasses
    whose fields are all :func:`knob` fields).

    :cvar LABEL: the section's name in a menu
    :cvar PRESETS: per level, the fields that differ from the defaults (the
        HIGH values)
    :cvar RENAMED: old field name -> new, read from saved data so a renamed
        knob keeps its saved value
    """

    LABEL: ClassVar[str] = ""
    PRESETS: ClassVar[dict[QualityLevel, dict[str, Any]]] = {}
    RENAMED: ClassVar[dict[str, str]] = {}

    def __post_init__(self):
        hints = typing.get_type_hints(type(self))
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            fixed, problem = _check(f.metadata["knob"], hints[f.name], value)
            if problem is not None or (
                isinstance(fixed, (int, float)) and fixed != value
            ):
                raise ValueError(
                    f"{type(self).__name__}.{f.name} = {value!r}: "
                    f"{problem or 'out of range'}"
                )
            if fixed is not value:  # an int given for a float
                object.__setattr__(self, f.name, fixed)

    @classmethod
    def preset(cls: type[S], level: QualityLevel | str) -> S:
        """:returns: the settings of a preset (a level, or its name)"""
        return cls(**cls.PRESETS.get(QualityLevel.parse(level), {}))

    @classmethod
    def knobs(cls) -> list[tuple[str, Knob]]:
        """:returns: (field name, its knob), in field order: a menu's rows"""
        return [(f.name, f.metadata["knob"]) for f in dataclasses.fields(cls)]

    @classmethod
    def schema(cls) -> list[dict[str, Any]]:
        """Every knob as plain data (JSON-ready), in field order: what a menu
        is built from without naming any knob.

        :returns: per knob: ``id``, ``label``, ``kind`` ("bool", "int",
            "float" or "choice"), ``unit``, ``range`` ([lowest, highest] or
            None), ``step`` (1 for whole numbers, else None), ``choices``
            (labels, or None), ``apply`` (:data:`LIVE`, :data:`REBUILD`,
            :data:`RESTART`), ``advanced``, ``help``, ``default`` and
            ``presets`` (level name -> value), values as saved (choices by
            label)
        """
        hints = typing.get_type_hints(cls)
        presets = {level: cls.preset(level) for level in QualityLevel}
        default = cls()
        out = []
        for name, k in cls.knobs():
            kind = "choice" if k.choices is not None else hints[name].__name__
            out.append(
                {
                    "id": name,
                    "label": k.label,
                    "kind": kind,
                    "unit": k.unit,
                    "range": list(k.range) if k.range is not None else None,
                    "step": 1 if kind == "int" else None,
                    "choices": list(k.choices) if k.choices is not None else None,
                    "apply": k.apply,
                    "advanced": k.advanced,
                    "help": k.help,
                    "kept": k.kept,
                    "default": _encode(k, getattr(default, name)),
                    "presets": {
                        level.value: _encode(k, getattr(p, name))
                        for level, p in presets.items()
                    },
                }
            )
        return out

    def at_level(self: S, level: QualityLevel | str) -> S:
        """:returns: the preset *level* with these settings' kept knobs (the
        choices that are not quality: :attr:`Knob.kept`)"""
        kept = {name: getattr(self, name) for name, k in self.knobs() if k.kept}
        return dataclasses.replace(type(self).preset(level), **kept)

    @classmethod
    def level_of(cls: type[S], settings: S) -> QualityLevel | None:
        """:returns: the preset *settings* equals, whatever its kept knobs;
        None = custom"""
        for level in QualityLevel:
            if settings.at_level(level) == settings:
                return level
        return None

    def changes(self: S, new: S) -> dict[str, str]:
        """:returns: the fields *new* changes, with how each applies"""
        return {
            name: k.apply
            for name, k in self.knobs()
            if getattr(self, name) != getattr(new, name)
        }

    def overrides(self: S, base: S) -> dict[str, Any]:
        """:returns: the fields that differ from *base*, as plain data
        (choices by label, tuples as lists): what a config file stores"""
        out = {}
        for name, k in self.knobs():
            value = getattr(self, name)
            if value != getattr(base, name):
                out[name] = _encode(k, value)
        return out

    @classmethod
    def from_overrides(cls: type[S], base: S, data: Any, where: str = "") -> S:
        """*base* with the fields *data* sets, leniently (see the module).

        :param base: the settings to start from (usually a preset)
        :param data: field name -> value (choices by label)
        :param where: names the section in warnings
        :returns: the settings; *base* where *data* is unusable
        """
        prefix = f"{where}." if where else ""
        if data is None:
            return base
        if not isinstance(data, Mapping):
            LOGGER.warning(f"settings {where or cls.__name__}: not a mapping, ignored")
            return base
        hints = typing.get_type_hints(cls)
        knobs = dict(cls.knobs())
        values = {}
        for name, raw in data.items():
            name = cls.RENAMED.get(name, name)
            k = knobs.get(name)
            if k is None:
                LOGGER.warning(f"unknown setting {prefix}{name}, ignored")
                continue
            value = raw
            if k.choices is not None and isinstance(raw, str):
                value = k.choices.get(raw, raw)
            fixed, problem = _check(k, hints[name], value)
            if problem is not None:
                LOGGER.warning(
                    f"setting {prefix}{name} = {raw!r}: {problem}; "
                    f"keeping {_encode(k, getattr(base, name))!r}"
                )
                continue
            if fixed != value:
                LOGGER.warning(f"setting {prefix}{name} = {raw!r} clamped to {fixed!r}")
            values[name] = fixed
        return dataclasses.replace(base, **values)

    def with_value(self: S, name: str, value: Any) -> S:
        """:returns: these settings with one field changed, leniently (a
        menu's slider or choice)"""
        return type(self).from_overrides(self, {name: value})


def _encode(k: Knob, value: Any) -> Any:
    """A field's value as plain data: its choice label, tuples as lists."""
    label = k.label_of(value)
    if label is not None:
        return label
    if isinstance(value, tuple):
        return list(value)
    return value


def _check(k: Knob, kind: Any, value: Any) -> tuple[Any, str | None]:
    """A value checked against its field.

    :returns: (the value, coerced to the field's type and clamped to its
        range; a problem, or None)
    """
    if k.choices is not None:
        if any(choice == value for choice in k.choices.values()):
            return value, None
        return value, f"not one of {list(k.choices)}"
    if kind is bool:
        return (value, None) if isinstance(value, bool) else (value, "not a boolean")
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value, "not a whole number"
        if value != int(value):
            return value, "not a whole number"
        value = int(value)
    elif kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value, "not a number"
        value = float(value)
    if k.range is not None:
        lo, hi = k.range
        value = type(value)(min(max(value, lo), hi))
    return value, None
