"""
Data files shipped with WorldBuilder: the tree assets (``trees/``).

The trees are third-party models (CC-BY-4.0: ``trees/CREDITS.md`` names their
authors, to be credited wherever the game is shared), split into one Panda3D
``.bam`` per asset by ``scripts/build_tree_assets.py`` from their glTF sources
(``trees/sources``). Each asset stands on its base's centre at the origin, Z
up, in metres. ``trees/trees.json`` indexes them: name, kind (``tree_near``:
a conifer with branch cards; ``tree_far``: a background conifer, crossed
cards; ``tree``: a birch or a snowy pine; ``plant``; ``rock``), pack,
triangles, size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from panda3d.core import Filename, Loader, NodePath

DATAFILES_PATH = Path(__file__).parent
TREES_PATH = DATAFILES_PATH / "trees"


@dataclass(frozen=True)
class TreeAsset:
    """One tree, plant or rock (``trees/trees.json``).

    :ivar name: its file's stem
    :ivar kind: tree_near, tree_far, tree, plant or rock
    :ivar pack: the pack it comes from (forest, birch, snowy)
    :ivar file: its ``.bam``, in :data:`TREES_PATH`
    :ivar triangles: how many it draws
    :ivar size: (width, depth, height), metres
    """

    name: str
    kind: str
    pack: str
    file: str
    triangles: int
    size: tuple[float, float, float]

    def load(self) -> NodePath:
        """:returns: a new copy of its model (Panda3D's loader caches the file)"""
        path = Filename.from_os_specific(str(TREES_PATH / self.file))
        return NodePath(Loader.get_global_ptr().load_sync(path))


def tree_assets() -> list[TreeAsset]:
    """:returns: every tree asset, in the index's order"""
    index = json.loads((TREES_PATH / "trees.json").read_text())
    return [
        TreeAsset(
            name=e["name"],
            kind=e["kind"],
            pack=e["pack"],
            file=e["file"],
            triangles=int(e["triangles"]),
            size=tuple(e["size"]),
        )
        for e in index
    ]
