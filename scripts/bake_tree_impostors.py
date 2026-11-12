"""
Bake the trees' impostors (world_builder.terrain.forest draws them past the
meshes' range): each tree asset (world_builder.datafiles) seen from
FRAMES x FRAMES directions over the upper hemisphere (a hemi-octahedral
grid), each view a TILE-pixel tile of one atlas per asset: its colour (as
authored, alpha its coverage) and its normal (the asset's frame, encoded in
[0, 1], alpha its coverage). Written to ``datafiles/trees/impostors``:
``albedo_NN.png``, ``normal_NN.png`` (NN: the layer, in the index's order)
and ``impostors.json`` (per asset: its layer, the views' width (its
bounding sphere's diameter, metres) and centre height). Also measures every
asset's crown (trees, plants and rocks), written to ``trees/crowns.json``:
its area seen from straight above (m^2, the covered pixels of a top view)
and its height (m): world_builder.terrain.forest derives its grids' spacing
and the far forest's cover from them.

Renders offscreen with Panda3D (the GPU)::

    python scripts/bake_tree_impostors.py [--tile PX] [--frames N] [--crowns-only]

A view's direction d (toward the camera, the asset's frame, z up) sits at
the hemi-octahedral point o = (p.x + p.y, p.x - p.y), p = d.xy / (|d.x| +
|d.y| + d.z), in [-1, 1]^2: frame (i, j) is the one nearest (o + 1) / 2 *
FRAMES - 0.5. Its up is the vertical's projection (the y axis, seen from
straight above). ``tree_impostor.vert`` picks the frame the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from panda3d.core import (
    CullFaceAttrib,
    LVecBase4f,
    OrthographicLens,
    PNMImage,
    Point3,
    Shader,
    Texture,
    Vec3,
    loadPrcFileData,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from world_builder.datafiles import TREES_PATH, tree_assets  # noqa: E402

OUT = TREES_PATH / "impostors"
KINDS = ("tree", "tree_near", "tree_far")

VERT = """
#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
out vec3 vNormal;
out vec2 vUv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vNormal = p3d_Normal;
    vUv = p3d_MultiTexCoord0;
}
"""
FRAG = """
#version 330
uniform sampler2D p3d_Texture0;
uniform int uPass;  // 0: the colour, 1: the normal
in vec3 vNormal;
in vec2 vUv;
out vec4 fragColor;
void main() {
    vec4 base = texture(p3d_Texture0, vUv);
    if (base.a < 0.5) discard;
    vec3 n = normalize(vNormal) * (gl_FrontFacing ? 1.0 : -1.0);
    fragColor = uPass == 0 ? vec4(base.rgb, 1.0) : vec4(0.5 + 0.5 * n, 1.0);
}
"""


def frame_direction(i: int, j: int, frames: int) -> np.ndarray:
    """:returns: the unit direction (toward the camera) of frame (i, j)"""
    o = (np.array([i, j], dtype=np.float64) + 0.5) / frames * 2.0 - 1.0
    p = np.array([(o[0] + o[1]) * 0.5, (o[0] - o[1]) * 0.5])
    d = np.array([p[0], p[1], 1.0 - abs(p[0]) - abs(p[1])])
    return d / np.linalg.norm(d)


def _dilate(rgba: np.ndarray, steps: int = 6) -> np.ndarray:
    """Spread the covered texels' colour over the empty ones around them (the
    mipmaps' edges stay the tree's colour, not black); alpha untouched."""
    rgb = rgba[..., :3].astype(np.float64)
    known = rgba[..., 3] > 127
    for _ in range(steps):
        acc = np.zeros_like(rgb)
        cnt = np.zeros(known.shape)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k = np.roll(np.roll(known, dy, 0), dx, 1)
                acc += np.roll(np.roll(rgb, dy, 0), dx, 1) * k[..., None]
                cnt += k
        grow = ~known & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        known = known | grow
    out = rgba.copy()
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def measure_crown(base, model, lens) -> dict:
    """:returns: the model's crown: its area seen from straight above (m^2)
    and its height (m), from a top view (the window's tile, scaled)"""
    lo, hi = model.get_tight_bounds()
    width = 1.02 * max(hi[0] - lo[0], hi[1] - lo[1])
    lens.set_film_size(width, width)
    lens.set_near_far(0.1, 4.0 * (hi[2] - lo[2]) + 10.0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    base.camera.set_pos(cx, cy, hi[2] + 2.0 * (hi[2] - lo[2]) + 5.0)
    base.camera.look_at(Point3(cx, cy, 0.0), Vec3(0, 1, 0))
    model.set_shader_input("uPass", 0)
    base.graphicsEngine.render_frame()
    base.graphicsEngine.render_frame()
    tex = base.win.get_screenshot()
    img = np.frombuffer(tex.get_ram_image_as("RGBA"), np.uint8)
    covered = float((img[3::4] > 127).mean())
    return {"area": round(covered * width * width, 3), "height": round(float(hi[2]), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--crowns-only",
        action="store_true",
        help="measure the crowns (crowns.json) only",
    )
    args = parser.parse_args()
    tile, frames = args.tile, args.frames
    loadPrcFileData(
        "",
        f"window-type offscreen\nwin-size {tile} {tile}\nframebuffer-alpha 1\n"
        "framebuffer-multisample 0\nsync-video 0\naudio-library-name null\n",
    )
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase(windowType="offscreen")
    base.set_background_color(0, 0, 0, 0)
    base.win.set_clear_color(LVecBase4f(0, 0, 0, 0))
    base.disable_mouse()
    shader = Shader.make(Shader.SL_GLSL, VERT, FRAG)
    lens = OrthographicLens()
    base.cam.node().set_lens(lens)
    OUT.mkdir(parents=True, exist_ok=True)
    crowns = {}
    for asset in tree_assets():
        model = asset.load()
        model.reparent_to(base.render)
        model.set_shader(shader, 100)
        model.set_attrib(CullFaceAttrib.make(CullFaceAttrib.M_cull_none))
        crowns[asset.name] = measure_crown(base, model, lens)
        model.remove_node()
        print(
            f"{asset.name:<22} crown {crowns[asset.name]['area']:6.1f} m^2, "
            f"height {crowns[asset.name]['height']:5.1f} m"
        )
    (TREES_PATH / "crowns.json").write_text(json.dumps(crowns, indent=1) + "\n")
    if args.crowns_only:
        return
    index = []
    assets = [a for a in tree_assets() if a.kind in KINDS]
    for layer, asset in enumerate(assets):
        model = asset.load()
        model.reparent_to(base.render)
        model.set_shader(shader, 100)
        model.set_attrib(CullFaceAttrib.make(CullFaceAttrib.M_cull_none))
        lo, hi = model.get_tight_bounds()
        centre = 0.5 * (np.array(lo) + np.array(hi))
        diameter = float(np.linalg.norm(np.array(hi) - np.array(lo)))
        lens.set_film_size(diameter, diameter)
        lens.set_near_far(0.1, 4.0 * diameter)
        atlases = []
        for pass_ in (0, 1):
            model.set_shader_input("uPass", pass_)
            atlas = np.zeros((frames * tile, frames * tile, 4), dtype=np.uint8)
            for j in range(frames):
                for i in range(frames):
                    d = frame_direction(i, j, frames)
                    eye = centre + d * 2.0 * diameter
                    base.camera.set_pos(*eye)
                    up = (0, 1, 0) if d[2] > 0.999 else (0, 0, 1)
                    base.camera.look_at(Point3(*centre), Vec3(*up))
                    base.graphicsEngine.render_frame()
                    base.graphicsEngine.render_frame()
                    tex = base.win.get_screenshot()
                    img = np.frombuffer(tex.get_ram_image_as("RGBA"), np.uint8)
                    img = img.reshape(tile, tile, 4)[::-1]  # rows from the top
                    row = (frames - 1 - j) * tile  # v up: frame row j from the bottom
                    atlas[row : row + tile, i * tile : (i + 1) * tile] = img
            if pass_ == 0:
                atlas = _dilate(atlas)
            atlases.append(atlas)
        for name, atlas in zip(("albedo", "normal"), atlases):
            image = PNMImage(atlas.shape[1], atlas.shape[0], 4)
            tex = Texture()
            tex.setup_2d_texture(
                atlas.shape[1], atlas.shape[0], Texture.T_unsigned_byte, Texture.F_rgba8
            )
            tex.set_ram_image_as(atlas[::-1].tobytes(), "RGBA")
            tex.store(image)
            image.write(str(OUT / f"{name}_{layer:02d}.png"))
        model.remove_node()
        index.append(
            {
                "name": asset.name,
                "layer": layer,
                "diameter": round(diameter, 3),
                "centre": [round(float(c), 3) for c in centre],
            }
        )
        print(f"{asset.name:<22} layer {layer:2d}  diameter {diameter:5.1f} m")
    (OUT / "impostors.json").write_text(
        json.dumps({"tile": tile, "frames": frames, "assets": index}, indent=1) + "\n"
    )
    print(f"{len(index)} impostors -> {OUT}")


if __name__ == "__main__":
    main()
