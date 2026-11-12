"""
Build the tree assets: split the tree packs (``src/world_builder/datafiles/
trees/sources``, glTF) into one Panda3D ``.bam`` per tree, plant or rock,
each standing on its base's centre at the origin (Z up, metres), and write
their index (``trees.json``: name, kind, pack, triangles, size).

Needs panda3d-gltf (a build-time tool: the game loads the ``.bam`` files with
Panda3D alone, :mod:`world_builder.datafiles`)::

    python scripts/build_tree_assets.py

The textures stay where they are (``sources/*/textures``): the ``.bam`` files
name them by relative path. The leaves' cards are drawn with a binary alpha
test (no sorting): their packs ask for blending.

The packs (all CC-BY-4.0: their credits are in ``trees/CREDITS.md``):

- ``forest`` (Low Poly Forest Tree Pack, 99.Miles): 8 background conifers of
  ~100 triangles on one 2048 atlas (crossed cards), 5 small plants of the
  same atlas, 4 nearer conifers (a trunk and a branch-card mesh, 300-650
  triangles), 9 rocks.
- ``birch`` (Birch Trees Pack, PS1 Low Poly, Vova Truskov): 3 birches of
  70-90 triangles on one 512 x 256 texture.
- ``snowy`` (Snowy Pine Trees Pack, PS1 Low Poly, the same author): 3 snowy
  pines of ~100 triangles on one 512 x 256 texture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gltf
from panda3d.core import (
    Filename,
    NodePath,
    TextureAttrib,
    TransparencyAttrib,
    loadPrcFileData,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from world_builder.datafiles import TREES_PATH  # noqa: E402

loadPrcFileData("", "bam-texture-mode relative")

SOURCES = TREES_PATH / "sources"


def _triangles(node: NodePath) -> int:
    count = 0
    for geom_node in node.find_all_matches("**/+GeomNode"):
        for geom in geom_node.node().get_geoms():
            for k in range(geom.get_num_primitives()):
                count += geom.get_primitive(k).decompose().get_num_primitives()
    return count


def _parts(model: NodePath) -> list[tuple[NodePath, tuple, tuple]]:
    out = []
    for part in model.find_all_matches("**/+GeomNode"):
        lo, hi = part.get_tight_bounds(model)
        out.append((part, tuple(lo), tuple(hi)))
    return out


def _groups(pack: str, parts: list) -> list[tuple[str, list]]:
    """(kind, parts) per asset: the forest pack's trunks with the branches
    over them; every other part alone."""
    groups = []
    if pack == "forest":
        trunks = [p for p in parts if p[0].get_name().startswith("Tree_Trunk")]
        branches = [p for p in parts if p[0].get_name().startswith("Tree_Branches")]
        for trunk in trunks:
            cx = 0.5 * (trunk[1][0] + trunk[2][0])
            cy = 0.5 * (trunk[1][1] + trunk[2][1])
            over = [
                b
                for b in branches
                if b[1][0] <= cx <= b[2][0] and b[1][1] <= cy <= b[2][1]
            ]
            groups.append(("tree_near", [trunk, *over]))
        for part in parts:
            name = part[0].get_name()
            if name.startswith("Background_Tree_Atlas"):
                height = part[2][2] - part[1][2]
                groups.append(("tree_far" if height > 12.0 else "plant", [part]))
            elif name.startswith("Rocks"):
                groups.append(("rock", [part]))
    else:
        groups = [("tree", [part]) for part in parts]
    return groups


def _asset(model: NodePath, parts: list, trunk_centre: bool) -> NodePath:
    """The parts in a node of their own, standing on the origin."""
    holder = NodePath("asset")
    inner = holder.attach_new_node("inner")
    for part, _, _ in parts:
        copy = part.copy_to(inner)
        copy.set_transform(part.get_transform(model))
    lo, hi = inner.get_tight_bounds()
    if trunk_centre:  # the trunk (the first part) on the origin
        _, tlo, thi = parts[0]
        cx, cy = 0.5 * (tlo[0] + thi[0]), 0.5 * (tlo[1] + thi[1])
    else:
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    inner.set_pos(-cx, -cy, -lo[2])
    holder.flatten_strong()
    _base_colour_only(holder)
    for node in holder.find_all_matches("**/+GeomNode"):
        if node.get_net_state().has_attrib(TransparencyAttrib.get_class_slot()):
            # Leaves: a binary alpha test, no sorting.
            node.set_transparency(TransparencyAttrib.M_binary, 1)
    return holder


def _base_colour_only(node: NodePath) -> None:
    """Keep each part's base colour texture alone (on its own stage): the
    game's shaders read it as p3d_Texture0 (the packs' other maps, normals
    and metal-roughness, would otherwise take that slot on some parts)."""
    for geom_node in node.find_all_matches("**/+GeomNode"):
        gn = geom_node.node()
        for k in range(gn.get_num_geoms()):
            state = gn.get_geom_state(k)
            attrib = state.get_attrib(TextureAttrib)
            if attrib is None:
                continue
            colour = None
            for s in range(attrib.get_num_on_stages()):
                stage = attrib.get_on_stage(s)
                tex = attrib.get_on_texture(stage)
                if (
                    "basecolor" in tex.get_name().lower()
                    or stage.get_name() == "Base Color"
                ):
                    colour = (stage, tex)
            if colour is None:
                state = state.remove_attrib(TextureAttrib.get_class_slot())
            else:
                # Its own stage (its texture coordinates' name), alone.
                state = state.set_attrib(TextureAttrib.make().add_on_stage(*colour))
            gn.set_geom_state(k, state)


def main() -> None:
    index = []
    for pack in ("forest", "birch", "snowy"):
        model = gltf.load_model(str(SOURCES / pack / "scene.gltf"))
        model = model if isinstance(model, NodePath) else NodePath(model)
        counts: dict[str, int] = {}
        for kind, parts in _groups(pack, _parts(model)):
            k = counts.get(kind, 0)
            counts[kind] = k + 1
            name = f"{pack}_{kind}_{k}"
            asset = _asset(model, parts, trunk_centre=kind == "tree_near")
            asset.set_name(name)
            path = TREES_PATH / f"{name}.bam"
            asset.write_bam_file(Filename.from_os_specific(str(path)))
            lo, hi = asset.get_tight_bounds()
            index.append(
                {
                    "name": name,
                    "kind": kind,
                    "pack": pack,
                    "file": path.name,
                    "triangles": _triangles(asset),
                    "size": [round(float(hi[i] - lo[i]), 2) for i in range(3)],
                }
            )
            print(
                f"{name:<24} {index[-1]['triangles']:5d} tris  size {index[-1]['size']}"
            )
    (TREES_PATH / "trees.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"{len(index)} assets -> {TREES_PATH}")


if __name__ == "__main__":
    main()
