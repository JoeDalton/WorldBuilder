"""
A gallery of rock prototypes, rendered to PNG: one row per prototype, one
column per level of detail with its vertex normals, then the 8- and
32-quad levels shaded with the normal map (what the field draws)::

    python scripts/asteroid_gallery.py [--out FILE] [--count N] [--seed S]
        [--ice FRACTION] [--size "1500 1000"] [--family NAME] [--views N]

    --family   only that family's prototypes (round, binary, bent, fragment;
               ShapeSpec.family)
    --views    instead of the levels of detail, N views of each prototype's
               finest mesh, normal-mapped, turned about two axes

Plain Lambert lighting from the upper left, no textures: the shapes only.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from panda3d.core import Filename, GeomNode, NodePath, Shader, Vec3

sys.path.insert(0, str(Path(__file__).parent))

from demo_terrain import window_config  # noqa: E402

from world_builder.asteroids.mesh import build_geom, normal_map_texture  # noqa: E402
from world_builder.asteroids.prototype import (  # noqa: E402
    PrototypeJob,
    generate_prototype,
)
from world_builder.asteroids.spec import ShapeSpec  # noqa: E402

_VERTEX = """
#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec3 face_uv;
out vec3 vNormal;
out vec3 vFaceUv;
out vec3 vDir;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vNormal = mat3(p3d_ModelMatrix) * p3d_Normal;
    vDir = p3d_Vertex.xyz;
    vFaceUv = face_uv;
}
"""

_FRAGMENT = """
#version 330
uniform mat4 p3d_ModelMatrix;
uniform sampler2DArray uNormals;
uniform int uLayer;     // the prototype's first layer
uniform int uMapped;    // 1: shade with the normal map
uniform vec3 uAlbedo;
in vec3 vNormal;
in vec3 vFaceUv;
in vec3 vDir;
out vec4 fragColor;
void main() {
    vec3 n = normalize(vNormal);
    if (uMapped == 1) {
        vec2 uv = (vFaceUv.xy + 1.0) * 0.5;
        int face = int(vFaceUv.z + 0.5);
        vec3 body = texture(uNormals, vec3(uv, float(uLayer + face))).rgb * 2.0 - 1.0;
        n = normalize(mat3(p3d_ModelMatrix) * body);
    }
    vec3 L = normalize(vec3(-0.6, -0.5, 0.7));
    float lambert = max(dot(n, L), 0.0);
    vec3 c = uAlbedo * (0.08 + 1.1 * lambert);
    fragColor = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("asteroid_gallery.png"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ice", type=float, default=0.0)
    parser.add_argument("--size", default="1500 1000")
    parser.add_argument("--family", choices=ShapeSpec.FAMILIES, default=None)
    parser.add_argument("--views", type=int, default=0)
    args = parser.parse_args()

    window_config("asteroid gallery", args.size, offscreen=True, vsync=False)
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    base.set_background_color(0.05, 0.05, 0.07, 1)
    spec = ShapeSpec()
    job = PrototypeJob(spec, seed=args.seed)
    keys = [
        k
        for k in range(spec.prototypes)
        if args.family is None or spec.family(k) == args.family
    ][: args.count]
    prototypes = [generate_prototype(job, k) for k in keys]
    normals = normal_map_texture(prototypes)
    shader = Shader.make(Shader.SL_GLSL, vertex=_VERTEX, fragment=_FRAGMENT)
    rock = Vec3(0.42, 0.39, 0.35)
    ice = Vec3(0.75, 0.85, 0.95)
    albedo = rock + (ice - rock) * args.ice
    finest = len(job.lods) - 1
    if args.views:
        columns = [(finest, True)] * args.views
    else:
        columns = [(i, False) for i in range(len(job.lods))] + [
            (1, True),
            (finest, True),
        ]
    root = base.render.attach_new_node("gallery")
    root.set_shader(shader)
    root.set_shader_input("uNormals", normals)
    root.set_shader_input("uAlbedo", albedo)
    spacing = 2.4
    for row, prototype in enumerate(prototypes):
        for col, (level, mapped) in enumerate(columns):
            geom = build_geom(prototype.meshes[level])
            node = GeomNode("rock")
            node.add_geom(geom)
            path: NodePath = root.attach_new_node(node)
            path.set_pos(
                (col - 0.5 * (len(columns) - 1)) * spacing,
                0.0,
                (0.5 * (len(prototypes) - 1) - row) * spacing,
            )
            if args.views:
                path.set_hpr(360.0 * col / args.views, 35.0 * (col % 2), 10.0)
            else:
                path.set_hpr(25.0, 20.0, 10.0)
            path.set_shader_input("uLayer", 6 * row)
            path.set_shader_input("uMapped", 1 if mapped else 0)
    lens = base.cam.node().get_lens()
    lens.set_fov(30.0)
    # far enough for the whole grid, across and down
    half_w = 0.5 * len(columns) * spacing
    half_h = 0.5 * len(prototypes) * spacing
    distance = 1.05 * max(
        half_w / math.tan(math.radians(0.5 * lens.get_hfov())),
        half_h / math.tan(math.radians(0.5 * lens.get_vfov())),
    )
    base.camera.set_pos(0.0, -distance, 0.0)
    base.camera.look_at(0.0, 0.0, 0.0)
    for _ in range(3):
        base.graphicsEngine.render_frame()
    base.win.save_screenshot(Filename.from_os_specific(str(args.out.resolve())))
    errors = ", ".join(
        f"{m.quads}: {m.error_to_finest:.3f}/{m.error:.3f}"
        for m in prototypes[0].meshes
    )
    print(f"{args.out}: errors of prototype 0 (to finest / to full detail) {errors}")
    for p in prototypes:
        print(f"  prototype {p.key} {spec.family(p.key)}: {p.gen_ms:.0f} ms")
    base.destroy()


if __name__ == "__main__":
    main()
