"""
World files: a whole world (its planets, moons, rings and asteroid fields, its
sun and sky) in YAML, loaded in one line::

    from world_builder.world_file import load_world

    world = load_world(app, "my_world.yaml")      # once
    ...
    world.update(camera_position_world)            # every frame

The file defines the world and nothing about how it is run: no quality, no
camera, no clock (they are arguments of :func:`load_world` and of the game).
It has every knob of every object, and a part of it may live in another file
of the same folder (``{name: my_planet, file: my_planet.yaml}``), so a world
reads as a list of bodies and a body can be reused (see :mod:`config` for the
rules, and ``docs/source/world_file.md`` for the keys)::

    sun: {direction: [..], color: [..], angular_radius: ..}
    sky: {...}                      # SkyLook: every field
    space_ambient: [r, g, b]
    bodies:
      - name: planet
        position: [x, y, z]
        density: 5514                 # or mass: kg
        planet: {...}                 # PlanetSpec: every field, stamps, paints
        atmosphere: {...}             # AtmosphereParams, or null for airless
        clouds: {..., weather: {...}, genera: {...}}   # CloudLook, or null
        ocean: {...}                  # OceanLook, or null
        ambient: [r, g, b]
        exposure: ..
    fields:
      - name: ring
        position: [x, y, z]
        axis: [x, y, z]               # its z axis (a ring's normal); or
                                      # orientation: [w, x, y, z]
        spec: {...}                   # FieldSpec: shape, density, sizes, ...
        ambient: .., exposure: .., far: .., haze: .., keep_outs: [..]

Positions are metres in the world frame; the game places its camera there.
"""

from __future__ import annotations

import functools
import logging
import math
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from world_builder.asteroids.spec import FieldSpec, KeepOut
from world_builder.config import (
    ConfigError,
    Context,
    build,
    dump_yaml,
    load_document,
    to_config,
)
from world_builder.depth import ReversedDepthCamera
from world_builder.settings import WorldSettings
from world_builder.terrain.planet import PlanetSpec
from world_builder.world import (
    COLLISION_MASK,
    BodyLook,
    FieldLook,
    SkyLook,
    SunLook,
    World,
)

#: The camera's far plane before the first frame: the sea's reflection camera
#: copies the lens when it is built, so it is set before the world.
INITIAL_FAR_M = 600000.0
#: Sleep this long after a loading step that still loads (:meth:`GameWorld.load_steps`)
LOADING_PAUSE_S = 0.005

__all__ = [
    "SHIPPED_FIELDS",
    "SHIPPED_PLANETS",
    "WORLDS_PATH",
    "GameWorld",
    "WorldSpec",
    "load_field",
    "load_planet",
    "load_world",
    "parse_world",
    "with_keep_outs",
    "world_to_yaml",
]

#: The shipped world files (and the bodies they link).
WORLDS_PATH = Path(__file__).parent / "worlds"
#: The shipped body files, for tools that draw one planet.
SHIPPED_PLANETS = ("kerbin", "earth", "desert", "forest", "moon", "ice")
#: The shipped field files, for tools that draw one field.
SHIPPED_FIELDS = ("asteroid_cube", "asteroid_icy_ball", "ring")

_TOP = {"sun", "sky", "space_ambient", "bodies", "fields"}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorldSpec:
    """What a world file defines.

    :ivar bodies: its planets and moons
    :ivar fields: its asteroid fields and rings
    :ivar sun: its sun (None: no sun, no direct light; the file has no
        ``sun`` or ``sun: null``)
    :ivar sky: its sky (None: no skybox; the file has no ``sky`` or ``sky: null``)
    :ivar space_ambient: the sky light away from every body, linear (None:
        none; a file without ``space_ambient``)
    """

    bodies: tuple[BodyLook, ...] = ()
    fields: tuple[FieldLook, ...] = ()
    sun: SunLook | None = None
    sky: SkyLook | None = None
    space_ambient: tuple[float, float, float] | None = None


def parse_world(source: Path | str | dict) -> WorldSpec:
    """Read a world file (no window needed).

    :param source: the YAML file, or its parsed mapping (links are then
        looked for in the current folder)
    :raises ConfigError: (a ValueError) on anything that is not valid, with
        where it is
    """
    if isinstance(source, dict):
        base = Path.cwd()
        from world_builder.config import expand

        data = expand(source, base)
    else:
        base = Path(source).parent
        data = load_document(source)
    unknown = set(data) - _TOP
    if unknown:
        raise ConfigError(
            f"unknown keys {sorted(unknown)} (a world has {sorted(_TOP)})"
        )
    ctx = Context(base)
    kwargs: dict[str, Any] = {}
    if "sun" in data:
        kwargs["sun"] = build(SunLook | None, data["sun"], ctx, "sun")
    if "sky" in data:
        kwargs["sky"] = build(SkyLook | None, data["sky"], ctx, "sky")
    if "space_ambient" in data:
        kwargs["space_ambient"] = build(
            tuple[float, float, float] | None,
            data["space_ambient"],
            ctx,
            "space_ambient",
        )
    bodies = data.get("bodies") or []
    fields_ = data.get("fields") or []
    for kind, items in (("bodies", bodies), ("fields", fields_)):
        if not isinstance(items, list):
            raise ConfigError(f"{kind}: must be a list")
    return WorldSpec(
        bodies=tuple(_body(b, base, f"bodies[{i}]") for i, b in enumerate(bodies)),
        fields=tuple(_field(f, base, f"fields[{i}]") for i, f in enumerate(fields_)),
        **kwargs,
    )


def load_planet(source: Path | str) -> PlanetSpec:
    """The planet of a body file (``planet:`` block) or of a world file's
    first body (no window needed). Read once per file version: the specs are
    frozen, so callers share them.

    :param source: a path, or the name of a body file of the shipped worlds
        folder (``"desert"``: see :data:`SHIPPED_PLANETS`)
    :raises ConfigError: if the file has no planet, or it is not valid
    """
    return _planet(*_locate(source))


def load_field(source: Path | str) -> FieldSpec:
    """The spec of a field file (``spec:`` block) or of a world file's first
    field (no window needed). Read once per file version, as
    :func:`load_planet`.

    :param source: a path, or the name of a field file of the shipped worlds
        folder (``"ring"``: see :data:`SHIPPED_FIELDS`)
    :raises ConfigError: if the file has no field, or it is not valid
    """
    return _field_spec(*_locate(source))


def _locate(source: Path | str) -> tuple[str, int]:
    """:returns: the file's path and modification time, for the caches"""
    path = Path(source)
    if not path.suffix and not path.exists():
        path = WORLDS_PATH / f"{source}.yaml"
    return str(path), path.stat().st_mtime_ns


@functools.lru_cache(maxsize=64)
def _planet(path: str, mtime: int) -> PlanetSpec:
    data = load_document(path)
    if "planet" not in data and data.get("bodies"):
        data = data["bodies"][0]
    if "planet" not in data:
        raise ConfigError(f"{path}: has no planet")
    return build(PlanetSpec, data["planet"], Context(Path(path).parent), "planet")


@functools.lru_cache(maxsize=64)
def _field_spec(path: str, mtime: int) -> FieldSpec:
    data = load_document(path)
    if "spec" not in data and data.get("fields"):
        data = data["fields"][0]
    if "spec" not in data:
        raise ConfigError(f"{path}: has no field")
    return build(FieldSpec, data["spec"], Context(Path(path).parent), "spec")


def _body(data: Any, base: Path, path: str) -> BodyLook:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: needs a mapping")
    data = dict(data)
    ctx = Context(base)
    if "planet" not in data:
        raise ConfigError(f"{path}.planet: is required")
    planet = build(PlanetSpec, data.pop("planet"), ctx, f"{path}.planet")
    ctx.planet = planet
    mass = data.pop("mass", None)
    density = data.pop("density", None)
    if (mass is None) == (density is None):
        raise ConfigError(f"{path}: give exactly one of mass (kg) and density (kg/m3)")
    if mass is None:
        radius = planet.radius
        mass = _number(density, f"{path}.density") * 4.0 / 3.0 * math.pi * radius**3
    data["mass"] = _number(mass, f"{path}.mass")
    data["spec"] = planet
    return build(BodyLook, data, ctx, path)


def _number(value: Any, path: str) -> float:
    return build(float, value, Context(), path)


def _field(data: Any, base: Path, path: str) -> FieldLook:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: needs a mapping")
    data = dict(data)
    if "axis" in data:
        if "orientation" in data:
            raise ConfigError(f"{path}: give axis or orientation, not both")
        axis = build(tuple[float, float, float], data.pop("axis"), Context(), path)
        data["orientation"] = _z_to(axis, f"{path}.axis")
    return build(FieldLook, data, Context(base), path)


def _z_to(axis: tuple[float, float, float], path: str) -> tuple[float, ...]:
    """:returns: the unit quaternion (w, x, y, z) turning +z onto *axis* by
    the shortest way"""
    a = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(a))
    if norm == 0.0:
        raise ConfigError(f"{path}: cannot be the zero vector")
    a /= norm
    if a[2] < -1.0 + 1e-12:  # straight down: half a turn about x
        return (0.0, 1.0, 0.0, 0.0)
    q = np.array([1.0 + a[2], -a[1], a[0], 0.0])
    q /= np.linalg.norm(q)
    return tuple(float(x) for x in q)


def world_to_yaml(spec: WorldSpec) -> str:
    """:returns: *spec* as one self-contained world file (no links; image
    paths absolute). Mostly to read what a world is made of."""
    out: dict[str, Any] = {}
    if spec.sun is not None:
        out["sun"] = to_config(spec.sun)
    if spec.sky is not None:
        out["sky"] = to_config(spec.sky)
    if spec.space_ambient is not None:
        out["space_ambient"] = list(spec.space_ambient)
    out |= {"bodies": [], "fields": []}
    for body in spec.bodies:
        ctx = Context(planet=body.spec)
        dumped = to_config(body, ctx=ctx)
        dumped["planet"] = dumped.pop("spec")
        out["bodies"].append(dumped)
    out["fields"] = [to_config(f) for f in spec.fields]
    return dump_yaml(out)


# ── the loaded world ─────────────────────────────────────────────────────────


class GameWorld:
    """A world loaded from a file: the :class:`World` and its depth camera,
    updated in one call.

    :ivar spec: what the file defined
    :ivar world: the :class:`World`: bodies, fields, sky, lights, gravity,
        collisions (see its docs)
    :ivar views: more cameras (camera NodePaths: a rear-view mirror's) the
        asteroid fields and rings are drawn for, besides the game's own; the
        game adds and removes them as they are shown (the list is read on
        every :meth:`update`)
    :ivar depth: the reversed camera
        (:class:`~world_builder.depth.ReversedDepthCamera`); None in an app
        without a window (``window-type none``: a game's headless runs),
        which draws nothing
    """

    def __init__(self, app: Any, spec: WorldSpec, world: World) -> None:
        self.app = app
        self.spec = spec
        self.world = world
        self.views: list[Any] = []
        self.depth: ReversedDepthCamera | None = None
        if getattr(app, "win", None) is None:
            LOGGER.info("no window: the world is built without its depth camera")
            return
        self.depth = ReversedDepthCamera(app, near=1.0, render=world.settings.render)
        if not self.depth.float_depth:
            LOGGER.warning(
                "the window has no float depth buffer: depth precision is a "
                "24-bit range's (open it with depth.float_depth_fbprops())"
            )
        world.use_depth_camera(self.depth)

    def body(self, name: str):
        """:returns: the built body called *name* (:meth:`World.body`)"""
        return self.world.body(name)

    def field(self, name: str):
        """:returns: the built field called *name* (:meth:`World.field`)"""
        return self.world.field(name)

    def update(self, origin_world: Any) -> None:
        """Draw the world for a frame: call it once per frame, after moving
        the camera.

        :param origin_world: the camera's position in the world frame, metres
            (float64; the camera itself stays at render's origin)

        The asteroid fields are drawn for :attr:`views` too.
        """
        views = tuple(self.views)
        if self.depth is None:
            self.world.update(origin_world, views=views)
            return
        self.depth.update(self.world.far_distance(origin_world))
        self.world.update(origin_world, far=self.depth.far, views=views)

    def load_steps(
        self,
        step: Callable[[], None],
        quiet_s: float = 0.5,
        timeout_s: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        pause: Callable[[float], None] = time.sleep,
    ) -> Generator[None, None, bool]:
        """Hold a game's loading phase until the world has loaded around the
        camera (ground not drawn yet is not solid): *step* once per
        iteration, until :attr:`World.loading` has stayed False for *quiet_s*
        (it flickers in flight, one quiet frame is not enough).

        The generator yields once per step, for the game to advance a loading
        animation by, and returns whether the world loaded (True) or
        *timeout_s* ran out (False, logged: the game may start anyway)::

            for _ in game_world.load_steps(step):
                render_loading_frame()

        :param step: advance the game by a frame: the game loop is paused
            while it loads, so this is what runs a frame (it moves render's
            origin and calls :meth:`update`)
        :param quiet_s: how long :attr:`World.loading` must stay False, seconds
        :param timeout_s: give up after this long, seconds
        :param clock: real time, seconds
        :param pause: sleeps, seconds: a few milliseconds after each step
            that still loads, as the workers' results come back through a
            thread of this process, which a loop that never sleeps starves
        """
        start = clock()
        quiet_since = None
        while True:
            step()
            now = clock()
            if self.world.loading:
                quiet_since = None
                pause(LOADING_PAUSE_S)
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= quiet_s:
                return True
            if now - start >= timeout_s:
                LOGGER.warning(
                    f"world still loading after {timeout_s:.0f} s: giving up waiting"
                )
                return False
            yield

    def destroy(self) -> None:
        """Remove the world and stop its workers."""
        if self.depth is not None:
            self.depth.destroy()
        self.world.destroy()


def with_keep_outs(
    spec: WorldSpec, keep_outs: Mapping[str, Sequence[KeepOut]]
) -> WorldSpec:
    """*spec* with more room kept clear in some of its fields.

    :param keep_outs: field name -> the room to keep clear there, added to the
        field's own (the file's)
    :raises ConfigError: a name that is not one of the fields
    """
    names = {look.name for look in spec.fields}
    unknown = sorted(set(keep_outs) - names)
    if unknown:
        raise ConfigError(
            f"keep_outs: no field called {unknown} (the world has {sorted(names)})"
        )
    fields = tuple(
        replace(look, keep_outs=(*look.keep_outs, *keep_outs[look.name]))
        if look.name in keep_outs
        else look
        for look in spec.fields
    )
    return replace(spec, fields=fields)


def load_world(
    app: Any,
    source: Path | str | WorldSpec,
    *,
    keep_outs: Mapping[str, Sequence[KeepOut]] | None = None,
    settings: WorldSettings | None = None,
    backend: str = "process",
    clock: Any = None,
    collision_mask: Any = COLLISION_MASK,
    **world_args: Any,
) -> GameWorld:
    """Build the world a file defines around a game's camera.

    :param app: the ShowBase
    :param source: the world file (YAML), or a :class:`WorldSpec` (what
        :func:`parse_world` returns, changed or built by the game)
    :param keep_outs: room a level keeps clear in some fields, by field name,
        added to the file's (see :func:`with_keep_outs`)
    :param settings: what every part costs; None = HIGH
    :param backend: where chunks and cloud tiles are generated
        (``"process"``, ``"thread"`` or ``"inline"``)
    :param clock: game time in seconds, a callable; None = Panda's frame time
    :param collision_mask: the into mask of the collision nodes of every
        terrain: bodies' ground and sea, fields' rocks, all solid from the
        start (named ``"terrain"``; None = nothing is solid). The game only
        gives its mask
    :param world_args: any other argument of :class:`World` that is the
        game's (``lights``, ``light_scale``, ``light_period``)
    :returns: the loaded world
    :raises ConfigError: a keep-out for a field the world does not have
    :raises NativeCollisionError: the native collision library is missing
        (``invoke native``)
    """
    spec = source if isinstance(source, WorldSpec) else parse_world(source)
    if keep_outs:
        spec = with_keep_outs(spec, keep_outs)
    # Before the world is built: the sea's reflection camera copies the lens.
    # (An app without a window has none.)
    if getattr(app, "camLens", None) is not None:
        app.camLens.set_near_far(1.0, INITIAL_FAR_M)
    world = World(
        app,
        spec.bodies,
        sun=spec.sun,
        sky=spec.sky,
        settings=settings,
        clock=clock,
        collision_mask=collision_mask,
        backend=backend,
        space_ambient=spec.space_ambient,
        fields=spec.fields,
        **world_args,
    )
    return GameWorld(app, spec, world)
