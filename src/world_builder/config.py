"""
Config files: the library's frozen dataclasses, read from and written to YAML.

A world file (:mod:`world_builder.world_file`) is made of these. Every field
of every look is a key of the same name (so every knob is exposed, and a
new field of a look is a new key with no code here); the one-line rules::

    key: value                    a number, string, bool or null
    key: [a, b, c]                a tuple (lists become tuples, recursively)
    key: {field: value, ...}      a dataclass (omitted fields take its default;
                                  an unknown key or a missing required one is
                                  an error naming where it is)
    key: {file: other.yaml}       the content of other.yaml, which sits in the
                                  same folder as the world file; keys beside
                                  ``file`` replace the file's own

The few types that are not plain dataclasses have a convention:

- a union of dataclasses (a field shape, a spin axis) is chosen by a ``type``
  key: ``box``, ``sphere``, ``annulus``, ``isotropic``, ``around_axis``;
- a :class:`Distribution` is ``{kind: power_law, q: 3.0, lo: 50, hi: 2000}``
  (the factory of that name, with its arguments);
- an Enum is its value (``cloud_type: cumulus``);
- ``sky.stars`` is the arguments of a :class:`Starfield`, or null for the
  default one; ``nebula.palette`` is a list of colours or a palette name;
- image paths (a stamp's ``heightmap``, a paint's ``mask``) are relative to
  the world file's folder;
- what a body gives its parts is not written in them: the weather's planet,
  and the genera's weather (the cloud's own, else the default for its seed).

``from_config`` reads, ``to_config`` writes the same thing back
(``from_config(to_config(x)) == x``), and ``dump_yaml`` prints it.
"""

from __future__ import annotations

import enum
import os
import re
import types
import typing
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from world_builder.asteroids.distributions import Distribution
from world_builder.clouds.synoptic.presets import WeatherPin
from world_builder.clouds.weather import WeatherSpec
from world_builder.sky.starfield import NEBULA_PALETTES, Starfield
from world_builder.terrain.planet import PlanetSpec
from world_builder.terrain.stamps import DensityPaint, Paint, Stamp


class ConfigError(ValueError):
    """A config that is not valid; the message says where."""


# ── what the types do not say ────────────────────────────────────────────────

_PAIRS = typing.Optional[tuple[tuple[float, float], ...]]  # noqa: UP045
#: Fields annotated with a bare ``tuple``: what their elements are.
_FIELD_TYPES: dict[tuple[str, str], Any] = {
    ("PlanetSpec", "stamps"): tuple[Stamp, ...],
    ("PlanetSpec", "paints"): tuple[Paint, ...],
    ("PlanetSpec", "density_paints"): tuple[DensityPaint, ...],
    ("WeatherSpec", "pins"): tuple[WeatherPin, ...],
    ("WeatherSpec", "zonal"): _PAIRS,
    ("WeatherSpec", "jets"): _PAIRS,
    ("WeatherSpec", "cirrus_zonal"): _PAIRS,
}
#: Image paths: relative to the world file's folder in a file.
_PATHS = {("Stamp", "heightmap"), ("Paint", "mask"), ("DensityPaint", "mask")}
#: Fields a body hands to its parts: never in a file.
_INJECTED = {("WeatherSpec", "planet"), ("GeneraSpec", "weather")}
#: Fields computed from the others: never in a file.
_DERIVED = {("PlanetSpec", "sea_bias")}
#: Fields a dataclass cannot give a type for (read by hand, below).
_HAND_READ = {("SkyLook", "stars")}


@dataclass
class Context:
    """What a read knows besides the text: the world file's folder, and the
    body being read (its planet and weather go to the parts that need them).

    :ivar base_dir: the world file's folder (links and image paths)
    :ivar planet: the planet of the body being read
    :ivar weather: the weather of its clouds, once read
    :ivar weather_seed: the seed of the weather its clouds make when none is
        given
    """

    base_dir: Path | None = None
    planet: PlanetSpec | None = None
    weather: WeatherSpec | None = None
    weather_seed: int = 11

    def injected_weather(self) -> WeatherSpec:
        """:returns: the weather the genera use when not given their own"""
        if self.weather is not None:
            return self.weather
        if self.planet is None:
            raise ConfigError("weather needs a planet (it is read inside a body)")
        return WeatherSpec(self.planet, seed=self.weather_seed)


# ── links ────────────────────────────────────────────────────────────────────


def expand(data: Any, base_dir: Path | None, stack: tuple[str, ...] = ()) -> Any:
    """Replace every ``{file: name.yaml, ...}`` by that file's content, the
    other keys on top, throughout *data*.

    :param base_dir: the folder the files are in (the world file's)
    :raises ConfigError: a file that is not a plain name in that folder, is
        missing, is not a mapping, or links back to itself
    """
    if isinstance(data, Mapping):
        if "file" in data:
            name = data["file"]
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name in (".", "..")
            ):
                raise ConfigError(
                    f"file: {name!r} must be a plain file name in the folder "
                    "of the world file"
                )
            if base_dir is None:
                raise ConfigError(f"file: {name} needs a folder to look in")
            if name in stack:
                raise ConfigError(f"files link in a circle: {[*stack, name]}")
            try:
                loaded = yaml.safe_load((base_dir / name).read_text())
            except OSError as error:
                raise ConfigError(f"file: {name}: {error.strerror}") from None
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"file: {name} must hold a mapping")
            merged = {**loaded, **{k: v for k, v in data.items() if k != "file"}}
            return expand(merged, base_dir, (*stack, name))
        return {k: expand(v, base_dir, stack) for k, v in data.items()}
    if isinstance(data, list):
        return [expand(v, base_dir, stack) for v in data]
    return data


def load_document(path: Path | str) -> dict:
    """:returns: the YAML mapping in *path*, its links expanded"""
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: must hold a mapping")
    return expand(data, path.parent, (path.name,))


# ── reading ──────────────────────────────────────────────────────────────────


def from_config(
    cls: Any, data: Any, base_dir: Path | str | None = None, ctx: Context | None = None
) -> Any:
    """Build a *cls* from its config.

    :param cls: a dataclass (or any type this module reads)
    :param data: the parsed YAML
    :param base_dir: the world file's folder (links, image paths)
    :param ctx: what the surrounding body gives (see :class:`Context`)
    :raises ConfigError: on an unknown key, a missing or ill-typed value, or
        anything the dataclass itself refuses
    """
    ctx = ctx or Context(Path(base_dir) if base_dir is not None else None)
    return build(cls, expand(data, ctx.base_dir), ctx, cls.__name__)


def build(tp: Any, value: Any, ctx: Context, path: str) -> Any:
    """Read *value* as a *tp* (the recursion of :func:`from_config`; *path*
    names the place for errors)."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            if len(members) < len(typing.get_args(tp)):
                return None
            raise ConfigError(f"{path}: needs a value")
        if len(members) == 1:
            return build(members[0], value, ctx, path)
        return _build_union(members, value, ctx, path)
    if tp is Any:
        return value
    if origin is tuple:
        return _build_tuple(typing.get_args(tp), value, ctx, path)
    if tp is tuple:
        return _plain(value, path)
    if not isinstance(tp, type):
        raise ConfigError(f"{path}: cannot read a {tp}")
    if isinstance(value, tp) and not isinstance(value, bool | int | float | str):
        return value  # already built (a body's planet)
    if issubclass(tp, enum.Enum):
        try:
            return tp(value)
        except ValueError:
            options = [m.value for m in tp]
            raise ConfigError(f"{path}: {value!r} is not one of {options}") from None
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: {value!r} is not true or false")
        return value
    if tp is int:
        return _int(value, path)
    if tp is float:
        return _float(value, path)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: {value!r} is not text")
        return value
    if tp is Distribution:
        return _build_distribution(value, path)
    if is_dataclass(tp):
        return _build_dataclass(tp, value, ctx, path)
    raise ConfigError(f"{path}: cannot read a {tp.__name__}")


def _float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{path}: {value!r} is not a number")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):  # PyYAML reads 1e6 (no dot, no sign) as text
        try:
            return float(value)
        except ValueError:
            pass
    raise ConfigError(f"{path}: {value!r} is not a number")


def _int(value: Any, path: str) -> int:
    number = _float(value, path)
    if number != int(number):
        raise ConfigError(f"{path}: {value!r} is not a whole number")
    return int(number)


def _plain(value: Any, path: str) -> Any:
    """Lists to tuples, recursively (specs are hashable cache keys)."""
    if isinstance(value, list | tuple):
        return tuple(_plain(v, path) for v in value)
    return value


def _build_tuple(args: tuple, value: Any, ctx: Context, path: str) -> tuple:
    if not isinstance(value, list | tuple):
        raise ConfigError(f"{path}: {value!r} is not a list")
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(
            build(args[0], v, ctx, f"{path}[{i}]") for i, v in enumerate(value)
        )
    if len(args) != len(value):
        raise ConfigError(f"{path}: needs {len(args)} values, not {len(value)}")
    return tuple(
        build(a, v, ctx, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value))
    )


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _build_union(members: list, value: Any, ctx: Context, path: str) -> Any:
    tags = {_snake(m.__name__): m for m in members if is_dataclass(m)}
    if len(tags) != len(members):
        raise ConfigError(f"{path}: cannot read a union of {members}")
    if not isinstance(value, Mapping) or "type" not in value:
        raise ConfigError(f"{path}: needs a type, one of {sorted(tags)}")
    kind = value["type"]
    if kind not in tags:
        raise ConfigError(f"{path}: type {kind!r} is not one of {sorted(tags)}")
    rest = {k: v for k, v in value.items() if k != "type"}
    return build(tags[kind], rest, ctx, path)


def _factories() -> dict[str, tuple[Any, tuple[str, ...]]]:
    d = Distribution
    return {
        "uniform": (d.uniform, ("lo", "hi")),
        "normal": (d.normal, ("mean", "sd", "lo", "hi")),
        "lognormal": (d.lognormal, ("median", "sigma", "lo", "hi")),
        "power_law": (d.power_law, ("q", "lo", "hi")),
        "u_shaped": (d.u_shaped, ("a", "lo", "hi", "log")),
        "table": (d.table, ("xs", "pdf")),
    }


def _build_distribution(value: Any, path: str) -> Distribution:
    if not isinstance(value, Mapping) or "kind" not in value:
        raise ConfigError(f"{path}: needs a kind, one of {sorted(_factories())}")
    kind = value["kind"]
    factories = _factories()
    if kind not in factories:
        raise ConfigError(f"{path}: kind {kind!r} is not one of {sorted(factories)}")
    factory, names = factories[kind]
    unknown = set(value) - set(names) - {"kind"}
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)} for {kind}")
    required = names[:-1] if kind == "u_shaped" else names
    missing = [n for n in required if n not in value]
    if missing:
        raise ConfigError(f"{path}: a {kind} needs {missing}")
    args = {}
    for name in names:
        if name in value:
            v = value[name]
            if name == "log":
                args[name] = bool(v)
            elif name in ("xs", "pdf"):
                args[name] = [_float(x, f"{path}.{name}") for x in v]
            else:
                args[name] = _float(v, f"{path}.{name}")
    try:
        return factory(**args)
    except (ValueError, IndexError) as error:
        raise ConfigError(f"{path}: {error}") from None


def _stars(value: Any, path: str) -> Starfield | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: stars are a mapping of a Starfield's arguments")
    known = {
        "seed": int,
        "stars": int,
        "galactic_pole": None,
        "glow_size": int,
        "nebula_scale": float,
        "nebula_coverage": float,
        "red": float,
        "blue": float,
    }
    unknown = set(value) - set(known)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    args = {}
    for k, v in value.items():
        if k == "galactic_pole":
            args[k] = tuple(_float(x, f"{path}.{k}") for x in v)
        elif known[k] is int:
            args[k] = _int(v, f"{path}.{k}")
        else:
            args[k] = _float(v, f"{path}.{k}")
    stars = Starfield(**args)
    stars.config = args  # what it was made from, to write it back
    return stars


def _build_dataclass(cls: type, value: Any, ctx: Context, path: str) -> Any:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: needs a mapping of {cls.__name__}'s fields")
    name = cls.__name__
    value = dict(value)
    if name == "NebulaStyle" and isinstance(value.get("palette"), str):
        palette = value["palette"]
        if palette not in NEBULA_PALETTES:
            raise ConfigError(
                f"{path}.palette: {palette!r} is not one of {sorted(NEBULA_PALETTES)}"
            )
        value["palette"] = NEBULA_PALETTES[palette]
    if name == "CloudLook":
        ctx.weather = None
        ctx.weather_seed = _int(value.get("weather_seed", 11), f"{path}.weather_seed")
    hints = typing.get_type_hints(cls)
    known = {f.name: f for f in fields(cls) if f.init}
    unknown = set(value) - set(known)
    if unknown:
        raise ConfigError(
            f"{path}: unknown keys {sorted(unknown)} (a {name} has {sorted(known)})"
        )
    kwargs = {}
    for fname, f in known.items():
        key = (name, fname)
        where = f"{path}.{fname}"
        if key in _DERIVED:
            if fname in value:
                raise ConfigError(f"{where}: is computed, not given")
            continue
        if key in _INJECTED:
            if fname in value and not (name == "GeneraSpec"):
                raise ConfigError(f"{where}: comes from the body, not the file")
            if fname not in value:
                kwargs[fname] = _injected(key, ctx, where)
                continue
        if fname in value:
            if key in _HAND_READ:
                kwargs[fname] = _stars(value[fname], where)
                continue
            tp = _FIELD_TYPES.get(key, hints[fname])
            read = build(tp, value[fname], ctx, where)
            if key in _PATHS:
                read = _resolve_path(read, ctx)
            kwargs[fname] = read
            if name == "CloudLook" and fname == "weather":
                ctx.weather = read
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"{where}: is required")
    try:
        obj = cls(**kwargs)
    except (ValueError, TypeError) as error:
        raise ConfigError(f"{path}: {error}") from None
    return obj


def _injected(key: tuple[str, str], ctx: Context, where: str) -> Any:
    if key == ("WeatherSpec", "planet"):
        if ctx.planet is None:
            raise ConfigError(f"{where}: weather is read inside a body, with a planet")
        return ctx.planet
    return ctx.injected_weather()


def _resolve_path(value: str, ctx: Context) -> str:
    if ctx.base_dir is None:
        return value
    return str((ctx.base_dir / value).resolve())


# ── writing ──────────────────────────────────────────────────────────────────


def to_config(obj: Any, base_dir: Path | str | None = None, ctx: Context | None = None):
    """The config of *obj*: what :func:`from_config` reads back to an equal
    object. Plain dicts, lists, numbers, strings, bools and None.

    :param base_dir: image paths are written relative to this folder
    :param ctx: the body *obj* is in (its planet and weather are not written)
    """
    ctx = ctx or Context(Path(base_dir) if base_dir is not None else None)
    return _dump(obj, ctx, "")


def _dump(obj: Any, ctx: Context, path: str) -> Any:
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, int | float):
        return obj
    if isinstance(obj, list | tuple):
        return [_dump(v, ctx, path) for v in obj]
    if isinstance(obj, dict):
        return {k: _dump(v, ctx, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, Distribution):
        return _dump_distribution(obj)
    if is_dataclass(obj):
        return _dump_dataclass(obj, ctx, path)
    raise ConfigError(f"{path}: cannot write a {type(obj).__name__}")


def _dump_distribution(d: Distribution) -> dict:
    p = d.params
    out: dict[str, Any] = {"kind": d.kind}
    if d.kind == "uniform":
        out |= {"lo": d.lo, "hi": d.hi}
    elif d.kind == "normal":
        out |= {"mean": p[0], "sd": p[1], "lo": d.lo, "hi": d.hi}
    elif d.kind == "lognormal":
        out |= {"median": p[0], "sigma": p[1], "lo": d.lo, "hi": d.hi}
    elif d.kind == "power_law":
        out |= {"q": p[0], "lo": d.lo, "hi": d.hi}
    elif d.kind == "u_shaped":
        out |= {"a": p[0], "lo": d.lo, "hi": d.hi, "log": bool(p[1])}
    else:
        out |= {"xs": list(p[0]), "pdf": list(p[1])}
    return out


def _union_tagged(hint: Any) -> bool:
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        members = [a for a in typing.get_args(hint) if is_dataclass(a)]
        return len(members) > 1
    return False


def _dump_dataclass(obj: Any, ctx: Context, path: str) -> dict:
    cls = type(obj)
    name = cls.__name__
    hints = typing.get_type_hints(cls)
    out: dict[str, Any] = {}
    if name == "CloudLook":
        ctx.weather, ctx.weather_seed = None, obj.weather_seed
    for f in fields(cls):
        if not f.init:
            continue
        key = (name, f.name)
        if key in _DERIVED:
            continue
        value = getattr(obj, f.name)
        if key == ("WeatherSpec", "planet"):
            continue
        if key == ("GeneraSpec", "weather") and value == ctx.injected_weather():
            continue
        if name == "CloudLook" and f.name == "weather" and value is not None:
            ctx.weather = value
        if key in _HAND_READ:
            if value is not None and not hasattr(value, "config"):
                raise ConfigError(
                    f"{path}.{f.name}: only a Starfield read from a config can be "
                    "written"
                )
            out[f.name] = None if value is None else _dump(value.config, ctx, path)
            continue
        dumped = _dump(value, ctx, f"{path}.{f.name}")
        if key in _PATHS:
            dumped = _relative(dumped, ctx)
        out[f.name] = dumped
        if _union_tagged(hints[f.name]):
            out[f.name] = {"type": _snake(type(value).__name__), **dumped}
    return out


def _relative(value: str, ctx: Context) -> str:
    if ctx.base_dir is None:
        return value
    return os.path.relpath(value, ctx.base_dir)


# ── printing ─────────────────────────────────────────────────────────────────


class _Dumper(yaml.SafeDumper):
    """Short lists of numbers on one line, one key per line otherwise."""

    def represent_sequence(self, tag, sequence, flow_style=None):
        flat = all(not isinstance(v, list | dict) for v in sequence)
        return super().represent_sequence(tag, sequence, flow_style=flat)


def dump_yaml(config: Any) -> str:
    """:returns: *config* (from :func:`to_config`) as YAML text"""
    return yaml.dump(
        config, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=88
    )
