"""
The vegetation of a terrain: its setting and its shader inputs
(``docs/source/vegetation.md``).

Where the plants stand is :mod:`~world_builder.terrain.forest`'s (in the
workers, per chunk: the plants and the vertices' far forest data). Two
looks draw them: the trees one by one near the camera
(:mod:`~world_builder.terrain.trees`: meshes, then impostors out to
``tree_range``), and past them the far forest, Bruneton and Neyret's
shader-map in the ground's shader (``shaders/vegetation_far.glsl``), from
tables baked from our trees (:mod:`~world_builder.terrain.far_forest`).
:class:`Vegetation` gives a terrain's root their inputs and its settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from panda3d.core import NodePath

from world_builder.quality import QualityLevel, QualitySettings, knob
from world_builder.terrain import far_forest


@dataclass(frozen=True)
class VegetationSettings(QualitySettings):
    """What the vegetation costs (:mod:`world_builder.quality`). All live.

    :ivar enabled: the trees and the forests at all
    :ivar mesh_range: the trees, bushes and rocks are drawn as their models
        out to here, metres (world_builder.terrain.trees)
    :ivar tree_range: and the trees as impostors out to here; past it, the
        far forest on the ground (0: the far forest everywhere)
    """

    enabled: bool = knob(True, "Vegetation", help="Trees and forests, near and far.")
    mesh_range: float = knob(
        200.0,
        "Tree detail distance",
        unit="m",
        range=(0.0, 2000.0),
        help="How far trees, bushes and rocks are drawn as their models "
        "(beyond: the trees as impostors, the bushes and rocks not at all).",
    )
    tree_range: float = knob(
        3500.0,
        "Tree distance",
        unit="m",
        range=(0.0, 12000.0),
        help="How far trees are drawn one by one (beyond: the forest drawn "
        "on the ground).",
    )

    LABEL: ClassVar = "Vegetation"
    #: Old knobs (the ray-marched canopy's, removed) are dropped on load.
    PRESETS: ClassVar = {
        QualityLevel.LOW: {"mesh_range": 80.0, "tree_range": 1200.0},
        QualityLevel.MID: {"mesh_range": 140.0, "tree_range": 2000.0},
        QualityLevel.ULTRA: {"mesh_range": 350.0, "tree_range": 6000.0},
    }


def bind_defaults(node: NodePath) -> None:
    """The vegetation's inputs, off (a terrain's root before its
    :class:`Vegetation` gives its own)."""
    node.set_shader_input("uVegOn", 0)
    node.set_shader_input("uTreeRange", 0.0)
    node.set_shader_input("uMeshRange", 0.0)
    far_forest.bind(node)
    node.set_shader_input("uForestGroups", 0)


class Vegetation:
    """One terrain's vegetation (``terrain.vegetation``): its settings and
    the far forest's inputs on the terrain's root."""

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``).
    SETTINGS_TYPE: ClassVar = VegetationSettings

    def __init__(self, terrain: Any, settings: VegetationSettings | None = None):
        self._terrain = terrain
        self._settings = settings or VegetationSettings()
        bind_defaults(terrain.root)
        self.reset()

    @property
    def settings(self) -> VegetationSettings:
        """What the vegetation draws (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: VegetationSettings) -> None:
        """Change what the vegetation draws (live)."""
        if settings != self._settings:
            self._settings = settings
            self._apply()
            trees = getattr(self._terrain, "trees", None)
            if trees is not None:
                trees.apply_settings()

    @property
    def active(self) -> bool:
        """Whether this planet grows anything and the vegetation is on."""
        return bool(self._settings.enabled and self._terrain.spec.has_vegetation)

    def reset(self) -> None:
        """Its inputs again (a new planet: :meth:`Terrain.set_spec`)."""
        self._apply()

    def _apply(self) -> None:
        root = self._terrain.root
        has_tables = far_forest.bind(root)
        root.set_shader_input("uVegOn", int(self.active and has_tables))
