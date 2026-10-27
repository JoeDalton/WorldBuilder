"""
Big rocks near the camera drawn as terrain: the terrain's level of detail
(and its collision) instead of an instanced mesh, which is only as fine as a
prototype's finest level and its normal map.

- **When the terrain is finer.** An instanced rock carries its prototype's
  full detail in its normal map (``normal_size`` texels a face side: detail
  down to two texels, pi R / normal_size). The terrain at a distance d draws
  tiles of about d / split_factor, band-limited to four of their 64
  spacings: d / (16 split_factor). It is the finer within
  :func:`show_distance` = 16 split_factor pi R / normal_size of the surface
  (0.8 R at the defaults); farther, the instance looks better.
- **Which.** Rocks of at least ``terrain_min_radius``, nearest to their
  surface first (relative to that distance), at most ``max_terrain_rocks``
  at a time. A rock's terrain loads, hidden, from :data:`PRELOAD` times that
  distance; it is shown within it once its ground for the camera is drawn
  (the instance hidden, the collision passed to the terrain's own solid),
  hidden again a little farther (:data:`HIDE_FACTOR`), and released past the
  preload distance (:data:`RELEASE_FACTOR`).
- **How.** A pool of :class:`~world_builder.terrain.terrain.Terrain`, made as
  needed and reused (``set_spec`` with the rock's
  :class:`~world_builder.asteroids.rock.RockSpec`), all on the field's worker
  pool and texture arrays.
- **Motion.** Every frame each terrain sits at its rock's centre, turned as
  the rock is at the field's time (``motion.orientation``, the same as the
  shader's), so the swap does not jump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from world_builder.asteroids.motion import orientation, quat_multiply
from world_builder.asteroids.rock import RockSpec
from world_builder.terrain.terrain import Terrain, TerrainSettings

if TYPE_CHECKING:
    from world_builder.asteroids.field import AsteroidField

#: A rock's terrain loads from this many times its show distance.
PRELOAD = 2.0
#: Shown terrain goes back to the instance this much past the show distance.
HIDE_FACTOR = 1.15
#: Released this much farther than its preload distance.
RELEASE_FACTOR = 1.3
#: A chunk's band limit in tile sizes is 4 / QUADS (cubesphere.QUADS = 64).
_TILE_BANDS = 16.0
#: What a rock's terrain keeps: a rock is a few hundred chunks at most.
ROCK_TERRAIN_SETTINGS = TerrainSettings(node_cache_size=400, store_size=600)


def show_distance(
    radius: ArrayLike, normal_size: int, split_factor: float
) -> np.ndarray:
    """:returns: how near a rock's surface its terrain is finer than its
    normal-mapped instance, metres"""
    return _TILE_BANDS * split_factor * np.pi * np.asarray(radius) / normal_size


@dataclass
class _Promoted:
    """A rock with a terrain.

    :ivar slot: its slot in the field's population
    :ivar rock_id: its id (a slot is reused by other rocks)
    :ivar terrain: its terrain
    :ivar shown: whether the terrain has taken over from the instance
    """

    slot: int
    rock_id: int
    terrain: Terrain
    shown: bool = False


class TerrainRocks:
    """A field's rocks drawn as terrain.

    :ivar promoted: per slot of the population, whether it is drawn as
        terrain (its instance hidden, its collision the terrain's)
    """

    def __init__(self, field: AsteroidField):
        self._field = field
        self._pool: list[Terrain] = []
        self._rocks: dict[int, _Promoted] = {}  # by slot
        self.promoted = np.zeros(0, dtype=bool)
        self._collision = None  # (into_mask, name, owner_tag, max_distance)

    # ── public ────────────────────────────────────────────────────────────

    @property
    def terrains(self) -> list[Terrain]:
        """The rocks' terrains now in use (shown or loading)."""
        return [p.terrain for p in self._rocks.values()]

    def near_factor(self) -> float:
        """:returns: how many radii from its centre a big rock may be drawn
        as terrain (its release distance), with a quarter more: the visible
        pass lists the rocks within it"""
        s = self._field.settings
        reach = show_distance(1.0, s.normal_size, ROCK_TERRAIN_SETTINGS.split_factor)
        return 1.0 + 1.25 * RELEASE_FACTOR * PRELOAD * float(reach)

    def enable_collision(
        self, into_mask, name: str, owner_tag: str, max_distance: float
    ) -> None:
        """The shown terrains are solid as the field is (the same node name,
        mask and owner tag)."""
        self._collision = (into_mask, name, owner_tag, max_distance)
        for p in self._rocks.values():
            if p.shown:
                self._solid(p.terrain)

    def disable_collision(self) -> None:
        self._collision = None
        for terrain in self._pool:
            terrain.disable_collision()

    def set_sun(
        self, direction: ArrayLike, color: ArrayLike, ambient: ArrayLike
    ) -> None:
        for terrain in self._pool:
            terrain.set_sun(direction, color, ambient)

    def update(
        self,
        camera: np.ndarray,
        camera_world: np.ndarray,
        camera_render: ArrayLike,
        time: float,
        near: np.ndarray,
    ) -> np.ndarray:
        """Per frame, after the population's update.

        :param camera: the camera, field frame
        :param camera_world: and in the world
        :param camera_render: and in render
        :param time: the field's time
        :param near: the slots of the big rocks near the camera, from the
            visible pass (:attr:`Visible.near`, with :meth:`near_factor`;
            a frame old is fine: it reaches farther than they are used)
        :returns: :attr:`promoted`
        """
        field = self._field
        pop = field.population
        settings = field.settings
        if len(self.promoted) < pop.capacity:
            grown = np.zeros(pop.capacity, dtype=bool)
            grown[: len(self.promoted)] = self.promoted
            self.promoted = grown
        limit = settings.max_terrain_rocks if field.full_errors is not None else 0
        big = np.asarray(near, dtype=np.int64)
        big = big[big < pop.capacity]
        radius = pop.compact["radius"][big]
        big = big[pop.live[big] & (radius >= settings.terrain_min_radius)]
        radius = pop.compact["radius"][big]
        rel = pop.compact["centre"][big] - camera
        surface = np.sqrt(np.einsum("ij,ij->i", rel, rel)) - radius
        show = show_distance(
            radius, settings.normal_size, ROCK_TERRAIN_SETTINGS.split_factor
        )
        ratio = surface / (PRELOAD * show)  # under 1: preload
        # Those still within the release distance, nearest first; those held
        # stay, others come in within the preload distance
        near = np.flatnonzero(ratio < RELEASE_FACTOR)
        near = near[np.argsort(ratio[near], kind="stable")]
        ranked = zip(
            ratio[near].tolist(),
            big[near].tolist(),
            pop.rocks["id"][big[near]].tolist(),
            surface[near],
            show[near],
        )
        keep: dict[int, tuple[int, float, float]] = {}
        for r, slot, rock_id, near, reach in ranked:
            if r >= RELEASE_FACTOR or len(keep) >= limit:
                break
            current = self._rocks.get(slot)
            held = current is not None and current.rock_id == rock_id
            if held or r < 1.0:
                keep[slot] = (rock_id, float(near), float(reach))
        for slot in [s for s, p in self._rocks.items() if s not in keep]:
            self._release(slot)
        for slot, (rock_id, _, _) in keep.items():
            current = self._rocks.get(slot)
            if current is not None and current.rock_id != rock_id:
                self._release(slot)
            if slot not in self._rocks:
                self._promote(slot, rock_id)
        # Place, turn and update each terrain; show it where it is the finer
        rotation = field.rotation
        world_q = field.orientation / np.linalg.norm(field.orientation)
        for p in self._rocks.values():
            rock = pop.rocks[p.slot]
            q = orientation(
                rock["q0"], rock["axis"], rock["precession"], rock["spin"], time
            )
            p.terrain.position = field.position + rotation @ rock["centre"]
            p.terrain.set_orientation(quat_multiply(world_q, q))
            p.terrain.update(camera_world, camera_render)
            _, near, reach = keep[p.slot]
            if not p.shown and near < reach and not p.terrain.loading:
                self._show(p)
            elif p.shown and near > HIDE_FACTOR * reach:
                self._hide(p)
        return self.promoted

    def destroy(self) -> None:
        for terrain in self._pool:
            terrain.destroy()
        self._pool = []
        self._rocks = {}

    # ── internals ─────────────────────────────────────────────────────────

    def _spec(self, rock) -> RockSpec:
        spec = self._field.spec
        return RockSpec(
            shape=spec.shapes,
            seed=spec.seed,
            index=int(rock["proto"]),
            radius=float(rock["radius"]),
            ice=float(rock["ice"]),
            patch_scale=spec.material.patch_scale,
            detail_seed=int(rock["id"]) & 0x7FFFFFFF,
            lowest=float(self._field.lowest[int(rock["proto"])]),
        )

    def _promote(self, slot: int, rock_id: int) -> None:
        field = self._field
        spec = self._spec(field.population.rocks[slot])
        busy = self.terrains
        idle = [t for t in self._pool if t not in busy]
        if idle:
            terrain = idle[0]
            terrain.set_spec(spec)
        else:
            terrain = Terrain(
                field.root.get_parent(),
                spec,
                executor=field.executor,
                materials=field.materials,
                textures=field.rock_textures,
                tonemap=field.tonemap,
                exposure=field.exposure,
                sun_direction=field.sun[0],
                sun_color=field.sun[1],
                ambient=field.sun[2],
                settings=ROCK_TERRAIN_SETTINGS,
            )
            self._pool.append(terrain)
        terrain.root.hide()  # until it is the finer
        self._rocks[slot] = _Promoted(slot, rock_id, terrain)

    def _show(self, p: _Promoted) -> None:
        """The terrain takes over from the instance: drawn and solid."""
        p.shown = True
        p.terrain.root.show()
        self.promoted[p.slot] = True
        if self._collision is not None:
            self._solid(p.terrain)

    def _hide(self, p: _Promoted) -> None:
        """Back to the instance (the terrain stays loaded, hidden)."""
        p.shown = False
        p.terrain.root.hide()
        p.terrain.disable_collision()
        if p.slot < len(self.promoted):
            self.promoted[p.slot] = False

    def _release(self, slot: int) -> None:
        self._hide(self._rocks.pop(slot))

    def _solid(self, terrain: Terrain) -> None:
        into_mask, name, owner_tag, max_distance = self._collision
        if terrain.collision is None:
            terrain.enable_collision(into_mask, name, owner_tag, max_distance)
