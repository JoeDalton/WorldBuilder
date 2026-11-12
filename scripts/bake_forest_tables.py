"""
Bake the far forest's tables (VEGETATION_PLAN.md, part V2): Bruneton and
Neyret's G, T, D and E [BN12 §6, Eqs. 14-21], measured on patches of our
own trees, per far group of world_builder.terrain.forest.

For each far group and tree density lambda (the share of the trees' cells
holding one of the group's trees: forest.TREE_CELL, the jittered grid and
the group's assets, spawn ratios and scales, as the game places them), a
flat patch of trees is rendered offscreen (Panda3D, the GPU), tiled, from
16 view zenith angles x 16 azimuths (relative to the sun's), each pixel's
first hit (tree or ground) and its position; and from the sun at 16 zenith
angles, its first hits (a shadow map). Per pixel, lit or not by each sun;
per view and sun, the fractions:

- kg: the visible and sunlit ground; kg + kgs: the visible ground [Eqs. 14-15]
- T = kt / (kt + kts): the sunlit share of the visible crowns [Eq. 16]
- E: the visible crowns' sky visibility (uniform over the upper hemisphere:
  foliage scatters isotropically) [Eq. 21]
- D: the ground's sky visibility under the trees (cosine weighted) [Eq. 18]

and the crowns' mean albedo seen from above (the far colour, linear: the
packs' textures hold linear albedo).

Written to ``datafiles/trees/forest_tables.npz``: ``table`` (groups, 8
lambda, 16 phi, 16 theta_l, 16 theta_v, 4: kg, kg + kgs, T, E), ``D``
(groups, 8), ``albedo`` (groups, 3), ``crown_cover`` (groups: the mean crown
area per tree cell), and the axes. Angles: theta in [0, 90] degrees and phi
(the view's azimuth from the sun's) in [0, 180], linear; lambda in [0, 1]::

    python scripts/bake_forest_tables.py [--res PX] [--sun-res PX] [--cells N]

Takes a few minutes. Re-run after a change of the tree assets, the far
groups or the trees' grid (VEGETATION_PLAN.md, decisions on data).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from panda3d.core import (
    CardMaker,
    CullFaceAttrib,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    LVecBase4f,
    NodePath,
    OrthographicLens,
    Point3,
    Shader,
    Texture,
    Vec3,
    WindowProperties,
    loadPrcFileData,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from world_builder.datafiles import TREES_PATH, tree_assets  # noqa: E402
from world_builder.terrain import forest  # noqa: E402

OUT = TREES_PATH / "forest_tables.npz"
N_LAMBDA = 8
N_ANGLE = 16
#: The views' zenith angle is clamped to this (a horizontal view sees no
#: ground and runs off the tiled patch), degrees.
MAX_VIEW_ZENITH = 87.0
#: The sun's zenith angle is clamped to this for its maps (on the horizon
#: the ground is edge-on: the lit ground is set to none there), degrees.
MAX_SUN_ZENITH = 87.0
#: The shadow test's bias, metres.
BIAS = 0.3

VERT = """
#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec3 vWorld;
out vec2 vUv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vWorld = (p3d_ModelMatrix * p3d_Vertex).xyz;
    vUv = p3d_MultiTexCoord0;
}
"""
FRAG = """
#version 330
uniform sampler2D p3d_Texture0;
uniform float uClass;  // 1 a tree, 0 the ground
uniform int uMode;     // 0 the hit's position, 1 its albedo
uniform float uLodBias;  // the sun's maps sample the leaves' mips as the views do
in vec3 vWorld;
in vec2 vUv;
out vec4 fragColor;
void main() {
    vec4 base = uClass > 0.5 ? texture(p3d_Texture0, vUv, uLodBias)
                             : vec4(0.3, 0.3, 0.3, 1.0);
    if (base.a < 0.5) discard;
    fragColor = uMode == 0 ? vec4(vWorld, uClass) : vec4(base.rgb, uClass);
}
"""


def angles() -> np.ndarray:
    """:returns: the zenith angles' samples, radians (theta_v's clamped)"""
    return np.radians(np.linspace(0.0, 90.0, N_ANGLE))


class Bench:
    """The offscreen buffers, cameras and shader."""

    def __init__(self, res: int, sun_res: int):
        loadPrcFileData(
            "",
            "window-type offscreen\nwin-size 16 16\nsync-video 0\n"
            "audio-library-name null\nframebuffer-multisample 0\n",
        )
        from direct.showbase.ShowBase import ShowBase

        self.base = ShowBase(windowType="offscreen")
        self.base.win.set_active(False)
        self.shader = Shader.make(Shader.SL_GLSL, VERT, FRAG)
        self.base.render.set_shader(self.shader, 100)
        self.base.render.set_shader_input("uMode", 0)
        self.base.render.set_shader_input("uClass", 1.0)
        self.base.render.set_shader_input("uLodBias", 0.0)
        self.base.render.set_attrib(CullFaceAttrib.make(CullFaceAttrib.M_cull_none))
        self.view = self._buffer("view", res)
        self.sun = self._buffer("sun", sun_res)

    def _buffer(self, name: str, res: int):
        fbp = FrameBufferProperties()
        fbp.set_rgba_bits(32, 32, 32, 32)
        fbp.set_float_color(True)
        fbp.set_depth_bits(24)
        buffer = self.base.graphicsEngine.make_output(
            self.base.pipe,
            name,
            -2,
            fbp,
            WindowProperties.size(res, res),
            GraphicsPipe.BF_refuse_window,
            self.base.win.get_gsg(),
            self.base.win,
        )
        tex = Texture(name)
        buffer.add_render_texture(tex, GraphicsOutput.RTM_copy_ram)
        buffer.set_clear_color(LVecBase4f(0, 0, 0, -1))
        buffer.set_active(False)
        lens = OrthographicLens()
        cam = self.base.make_camera(buffer, lens=lens)
        return {"buffer": buffer, "tex": tex, "lens": lens, "cam": cam, "res": res}

    def render(
        self,
        which: dict,
        centre: np.ndarray,
        toward: np.ndarray,
        film: tuple[float, float],
    ) -> tuple[np.ndarray, dict]:
        """Render along -toward (toward: from the scene to the eye) onto a
        film of (w, h) metres centred on *centre*.

        :returns: ((res * res, 4) float32 hits: position, class (-1 none)),
            the camera's frame (centre, right, up, forward, film)
        """
        cam, lens = which["cam"], which["lens"]
        lens.set_film_size(*film)
        lens.set_near_far(1.0, 8000.0)
        eye = centre + toward * 4000.0
        cam.set_pos(*eye)
        up = Vec3(0, 1, 0) if toward[2] > 0.999 else Vec3(0, 0, 1)
        cam.look_at(Point3(*centre), up)
        which["buffer"].set_active(True)
        self.base.graphicsEngine.render_frame()
        which["buffer"].set_active(False)
        img = np.frombuffer(which["tex"].get_ram_image_as("RGBA"), np.float32)
        q = cam.get_quat(self.base.render)
        frame = {
            "centre": centre,
            "right": np.array(q.get_right()),
            "up": np.array(q.get_up()),
            "forward": np.array(q.get_forward()),
            "film": film,
        }
        return img.reshape(-1, 4).copy(), frame


def patch_trees(group: int, lam: float, cells: int, seed: int = 3) -> list[tuple]:
    """The group's trees on a flat patch of cells x cells tree cells (the
    game's jittered grid and rules, one group, lambda per cell, no
    clumping: that is the vertex's density's business).

    :returns: [(asset name, x, y, yaw degrees, scale)]
    """
    c = forest.TREE_CELL
    cu, cv = np.meshgrid(np.arange(cells), np.arange(cells))
    s = forest.cell_samples("tree", cu.ravel(), cv.ravel(), 0, seed)
    grown = s["keep"] < lam
    s = forest.cell_details({k: a[grown] for k, a in s.items()})
    cu, cv = cu.ravel()[grown], cv.ravel()[grown]
    asset, scale, _ = forest._assets(s, np.full(len(cu), group))
    return [
        (forest.ASSETS[a], (u + ju) * c, (v + jv) * c, 360.0 * yaw, sc)
        for a, u, v, ju, jv, yaw, sc in zip(
            asset, cu, cv, s["ju"], s["jv"], s["yaw"], scale
        )
    ]


def build_scene(
    bench: Bench, models: dict, trees: list[tuple], period: float, tiles: int
) -> NodePath:
    """The patch, flattened, tiled (2 tiles + 1)^2 times, on a ground."""
    root = bench.base.render.attach_new_node("scene")
    patch = NodePath("patch")
    for name, x, y, yaw, scale in trees:
        m = models[name].copy_to(patch)
        m.set_pos(x, y, 0.0)
        m.set_h(yaw)
        m.set_scale(scale)
    patch.flatten_strong()
    for i in range(-tiles, tiles + 1):
        for j in range(-tiles, tiles + 1):
            # A node per tile: the instances share the patch's own node.
            tile = root.attach_new_node("tile")
            tile.set_pos(i * period, j * period, 0.0)
            patch.instance_to(tile)
    cm = CardMaker("ground")
    extent = (tiles + 1) * period
    cm.set_frame(-extent, extent + period, -extent, extent + period)
    ground = root.attach_new_node(cm.generate())
    ground.set_p(-90.0)
    ground.set_shader_input("uClass", 0.0)
    return root


def sun_dir(theta: float) -> np.ndarray:
    return np.array([math.sin(theta), 0.0, math.cos(theta)])


class SunMap:
    """The first hits seen from the sun: a depth map over the central patch
    (any point is wrapped into it: the scene is periodic)."""

    def __init__(
        self,
        bench: Bench,
        theta: float,
        period: float,
        height: float,
        view_pixel: float,
    ):
        self.period = period
        L = sun_dir(min(theta, math.radians(MAX_SUN_ZENITH)))
        centre = np.array([0.5 * period, 0.5 * period, 0.5 * height])
        # The film holds the central patch's box seen from the sun.
        corners = (
            np.array(
                [
                    [x, y, z]
                    for x in (0, period)
                    for y in (0, period)
                    for z in (0, height)
                ]
            )
            - centre
        )
        hits, frame = bench.render(bench.sun, centre, L, (1.0, 1.0))
        r, u = frame["right"], frame["up"]
        w = 2.05 * np.abs(corners @ r).max()
        h = 2.05 * np.abs(corners @ u).max()
        # The leaves are alpha-tested cards: their gaps match the views'
        # only at the same mip level (else the hotspot is lost).
        sun_pixel = max(w, h) / bench.sun["res"]
        bench.base.render.set_shader_input(
            "uLodBias", float(math.log2(max(view_pixel / sun_pixel, 1.0)))
        )
        hits, frame = bench.render(bench.sun, centre, L, (w, h))
        bench.base.render.set_shader_input("uLodBias", 0.0)
        self.frame = frame
        self.res = bench.sun["res"]
        self.depth = np.full(self.res * self.res, np.inf)
        valid = hits[:, 3] >= 0.0
        idx, d = self._project(hits[valid, :3])
        np.minimum.at(self.depth, idx, d)

    def _project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = self.frame
        rel = p - f["centre"]
        x = rel @ f["right"] / f["film"][0] + 0.5
        y = rel @ f["up"] / f["film"][1] + 0.5
        col = np.clip((x * self.res).astype(np.int64), 0, self.res - 1)
        row = np.clip((y * self.res).astype(np.int64), 0, self.res - 1)
        return row * self.res + col, rel @ f["forward"]

    def lit(self, p: np.ndarray) -> np.ndarray:
        """:returns: whether each point (n, 3) sees the sun"""
        q = p.copy()
        q[:, :2] = np.mod(q[:, :2], self.period)
        idx, d = self._project(q)
        return d <= self.depth[idx] + BIAS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--res", type=int, default=320, help="the views' pixels a side")
    parser.add_argument("--sun-res", type=int, default=1536)
    parser.add_argument(
        "--cells", type=int, default=24, help="the patch's cells a side"
    )
    parser.add_argument("--tiles", type=int, default=6, help="tiles each way round it")
    args = parser.parse_args()
    bench = Bench(args.res, args.sun_res)
    by_name = {a.name: a for a in tree_assets()}
    theta = angles()
    theta_v = np.minimum(theta, math.radians(MAX_VIEW_ZENITH))
    phi = np.radians(np.linspace(0.0, 180.0, N_ANGLE))
    lambdas = np.linspace(0.0, 1.0, N_LAMBDA)
    period = args.cells * forest.TREE_CELL
    groups = forest.FAR_GROUPS
    table = np.zeros((len(groups), N_LAMBDA, N_ANGLE, N_ANGLE, N_ANGLE, 4), np.float32)
    table_d = np.ones((len(groups), N_LAMBDA), np.float32)
    albedo = np.zeros((len(groups), 3), np.float32)
    # Quadratures over the upper hemisphere's zenith angle (trapezoids).
    dt = np.gradient(theta)
    w_cos = np.sin(theta) * np.cos(theta) * dt
    w_cos /= w_cos.sum()
    w_uni = np.sin(theta) * dt
    w_uni /= w_uni.sum()
    gu, gv = np.meshgrid(
        (np.arange(48) + 0.5) / 48 * period, (np.arange(48) + 0.5) / 48 * period
    )
    ground_pts = np.stack([gu.ravel(), gv.ravel(), np.full(gu.size, 0.02)], axis=1)
    start = time.perf_counter()
    for gi, g in enumerate(groups):
        group = forest.SPECIES[g]
        models = {}
        for name in group.assets:
            m = by_name[name].load()
            m.flatten_strong()
            models[name] = m
        height = max(forest.crowns()[a][1] for a in group.assets) * group.scale[1] + 1.0
        centre = np.array([0.5 * period, 0.5 * period, 0.35 * height])
        # λ = 0: bare ground, all of it seen and lit.
        table[gi, 0, ..., 0] = 1.0
        table[gi, 0, ..., 1] = 1.0
        table[gi, 0, ..., 2] = 1.0
        table[gi, 0, ..., 3] = 1.0
        for li, lam in enumerate(lambdas):
            if li == 0:
                continue
            trees = patch_trees(g, float(lam), args.cells)
            scene = build_scene(bench, models, trees, period, args.tiles)
            suns = [
                SunMap(bench, float(t), period, height, period / args.res)
                for t in theta
            ]
            # The ground's sky visibility (cosine weighted).
            vis = np.array([s.lit(ground_pts).mean() for s in suns])
            table_d[gi, li] = float(vis @ w_cos)
            if li == N_LAMBDA - 1:
                bench.base.render.set_shader_input("uMode", 1)
                hits, _ = bench.render(
                    bench.view, centre, np.array([0.0, 0.0, 1.0]), (period, period)
                )
                bench.base.render.set_shader_input("uMode", 0)
                crown = hits[:, 3] > 0.5
                albedo[gi] = hits[crown, :3].mean(axis=0) if crown.any() else 0.0
            for vi, tv in enumerate(theta_v):
                sky = []
                for pi, ph in enumerate(phi):
                    V = np.array(
                        [
                            math.sin(tv) * math.cos(ph),
                            math.sin(tv) * math.sin(ph),
                            math.cos(tv),
                        ]
                    )
                    hits, _ = bench.render(bench.view, centre, V, (period, period))
                    valid = hits[:, 3] >= 0.0
                    p = hits[valid, :3]
                    crown = hits[valid, 3] > 0.5
                    n = max(int(valid.sum()), 1)
                    nc = int(crown.sum())
                    kgs = 1.0 - nc / n
                    lit = np.stack([s.lit(p) for s in suns])  # (suns, n)
                    kg = (lit & ~crown[None]).sum(axis=1) / n
                    t = (
                        (lit & crown[None]).sum(axis=1) / nc
                        if nc
                        else np.ones(len(suns))
                    )
                    table[gi, li, pi, :, vi, 0] = kg
                    table[gi, li, pi, :, vi, 1] = kgs
                    table[gi, li, pi, :, vi, 2] = t
                    if nc:
                        sky.append(float(lit[:, crown].mean(axis=1) @ w_uni))
                table[gi, li, :, :, vi, 3] = np.mean(sky) if sky else 1.0
            table[gi, li, :, -1, :, 0] = 0.0  # the sun on the horizon
            scene.remove_node()
            print(
                f"{group.name:<12} lambda {lam:4.2f}: {len(trees):4d} trees, "
                f"D {table_d[gi, li]:.2f}, "
                f"top view ground {table[gi, li, 0, 0, 0, 1]:.2f}"
                f"  ({time.perf_counter() - start:5.0f} s)",
                flush=True,
            )
    crown_cover = np.array(
        [
            forest.group_crown_area(forest.SPECIES[g]) / forest.TREE_CELL**2
            for g in groups
        ],
        np.float32,
    )
    np.savez_compressed(
        OUT,
        table=table,
        D=table_d,
        albedo=albedo,
        crown_cover=crown_cover,
        groups=np.array([forest.SPECIES[g].name for g in groups]),
        lambdas=lambdas.astype(np.float32),
        theta_deg=np.degrees(theta).astype(np.float32),
        phi_deg=np.degrees(phi).astype(np.float32),
        tree_cell=np.float32(forest.TREE_CELL),
    )
    print(f"-> {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
