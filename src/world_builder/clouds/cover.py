"""
The clouds in front of a game's translucent surfaces.

The clouds are laid over the view as a premultiplied card that writes no depth
(``cloud_composite.frag``). A game's translucent surface that writes none
either (a shield bubble: depth write off, drawn after the clouds) would then
be blended over every cloud on screen, one in front of it included; drawn
before them, a cloud behind it would be laid over it. The march already keeps,
per pixel, the cloud's premultiplied colour and coverage and its distance from
the camera (its resolve's ``colour`` and ``distance`` targets): this hands a
game's nodes those two textures, the ones the clouds are composited from this
frame, and ``cloud_cover.glsl`` (in :data:`GLSL_DIR`, on Panda's model path
once a world is built: :func:`use_glsl_dir`) reads them back as the opacity of
the cloud in front of a fragment (``cloudCoverBefore``), which the surface
fades itself by.

:class:`CloudCover` holds the tracked nodes; :class:`~world_builder.world.World`
owns one (``track_translucent``) and gives it the inputs of the clouds the
camera is among every update.
"""

from __future__ import annotations

from pathlib import Path

from panda3d.core import (
    Camera,
    Filename,
    LVecBase2f,
    NodePath,
    PTA_int,
    PTA_LVecBase2f,
    RenderState,
    ShaderAttrib,
    Texture,
    getModelPath,
)

#: The folder of ``cloud_cover.glsl``: ``#pragma include "cloud_cover.glsl"``
#: finds it once a World is built (it is put on Panda's model path)
GLSL_DIR = Path(__file__).resolve().parent.parent / "shaders"
#: The tag of the tracked nodes, and the key of the tag state of a view the
#: cover is off in (:meth:`CloudCover.off_in`)
TAG = "world_builder_cloud_cover"


def _blank() -> Texture:
    """A 1 x 1 texture with no cloud: the samplers' input without clouds."""
    texture = Texture("cloud_cover_blank")
    texture.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_rgba8)
    texture.set_clear_color((0.0, 0.0, 0.0, 0.0))
    texture.make_ram_image()
    return texture


_BLANK: Texture | None = None


def no_cover(node: NodePath) -> None:
    """Give *node* the inputs of ``cloud_cover.glsl`` with no cloud: what a
    shader including it needs to draw at all where no world tracks it (or
    once untracked). :meth:`CloudCover.track` replaces them."""
    global _BLANK
    if _BLANK is None:
        _BLANK = _blank()
    node.set_shader_inputs(
        uCloudCover=_BLANK,
        uCloudCoverDistance=_BLANK,
        uCloudCoverDownscale=LVecBase2f(1.0, 1.0),
        uCloudCoverOn=0,
    )


def use_glsl_dir() -> None:
    """Put :data:`GLSL_DIR` on Panda's model path (once), where a shader's
    ``#pragma include "cloud_cover.glsl"`` looks."""
    model_path = getModelPath()
    name = Filename.from_os_specific(str(GLSL_DIR))
    known = model_path.get_value()
    if not any(
        known.get_directory(i) == name for i in range(known.get_num_directories())
    ):
        model_path.append_directory(name)


class CloudCover:
    """A game's translucent nodes and the clouds in front of them.

    :ivar nodes: the tracked nodes (their shaders include ``cloud_cover.glsl``)
    """

    def __init__(self) -> None:
        # A game's shaders may be compiled before any node is tracked
        use_glsl_dir()
        self.nodes: list[NodePath] = []
        self._blank = _blank()
        # Written in place: the nodes' render states change only with the
        # textures (two, alternating), not with these.
        self._on = PTA_int.empty_array(1)
        self._downscale = PTA_LVecBase2f.empty_array(1)
        self._downscale[0] = LVecBase2f(1.0, 1.0)
        self._textures: tuple[Texture, Texture] = (self._blank, self._blank)

    def track(self, node: NodePath) -> None:
        """Give *node* (and what is under it) the cover's inputs."""
        use_glsl_dir()
        if any(node == known for known in self.nodes):
            return
        self.nodes.append(node)
        node.set_tag(TAG, "1")
        self._apply(node)

    def untrack(self, node: NodePath) -> None:
        """No cover on *node* any more (its shader still draws: no cloud)."""
        self.nodes = [known for known in self.nodes if known != node]
        if not node.is_empty():
            node.clear_tag(TAG)
            no_cover(node)

    def update(
        self, inputs: tuple[Texture, Texture, tuple[float, float]] | None
    ) -> None:
        """Per frame: the clouds composited this frame.

        :param inputs: their colour and distance textures and the view's
            pixels per texel of theirs (``CloudMarch.cover_inputs``); None =
            no clouds
        """
        if inputs is None:
            self._on[0] = 0
            textures = (self._blank, self._blank)
        else:
            colour, distance, downscale = inputs
            self._on[0] = 1
            self._downscale[0] = LVecBase2f(*downscale)
            textures = (colour, distance)
        if textures != self._textures:
            self._textures = textures
            self.nodes = [node for node in self.nodes if not node.is_empty()]
            for node in self.nodes:
                self._apply(node)

    def _apply(self, node: NodePath) -> None:
        colour, distance = self._textures
        node.set_shader_inputs(
            uCloudCover=colour,
            uCloudCoverDistance=distance,
            uCloudCoverDownscale=self._downscale,
            uCloudCoverOn=self._on,
        )

    @staticmethod
    def off_in(camera: NodePath) -> None:
        """No cover in another view (a rear-view mirror's camera): its pixels
        are not the clouds' (they are marched for the main view alone). The
        camera draws the tracked nodes with ``uCloudCoverOn`` 0 (a tag state:
        it takes the camera's tag state key).

        :param camera: a camera NodePath
        """
        node: Camera = camera.node()
        node.set_tag_state_key(TAG)
        off = ShaderAttrib.make().set_shader_input("uCloudCoverOn", 0)
        node.set_tag_state("1", RenderState.make(off))

    def clear(self) -> None:
        """Untrack every node."""
        for node in list(self.nodes):
            self.untrack(node)


# ── a reference surface ─────────────────────────────────────────────────────

_BUBBLE_VERT = """
#version 140
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;
out vec3 vView;
out vec3 vNormal;
void main() {
    vView = (p3d_ModelViewMatrix * p3d_Vertex).xyz;
    vNormal = p3d_NormalMatrix * p3d_Normal;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""
_BUBBLE_FRAG = """
#version 140
#pragma include "cloud_cover.glsl"
uniform vec4 uBubbleColour;
in vec3 vView;
in vec3 vNormal;
out vec4 p3d_FragColor;
void main() {
    float facing = abs(dot(normalize(vNormal), normalize(-vView)));
    float rim = pow(1.0 - facing, 3.0);
    vec3 colour = uBubbleColour.rgb * (0.6 + 1.4 * rim);
    float alpha = clamp(uBubbleColour.a + rim, 0.0, 1.0);
    float cover = cloudCoverBefore(gl_FragCoord.xy, length(vView));
    p3d_FragColor = vec4(colour, alpha * (1.0 - cover));
}
"""


def translucent_bubble(
    radius: float,
    colour: tuple[float, float, float, float] = (0.3, 0.6, 1.0, 0.25),
    rings: int = 24,
) -> NodePath:
    """A translucent sphere drawn as a game's shield bubble is (depth write
    off, after the clouds, straight alpha) and faded by the clouds in front
    of it: the reference surface of ``cloud_cover.glsl``, for checks and
    demos (``scripts/demo_world.py --bubble``). Track it with
    ``World.track_translucent`` (unparented: the caller places it).

    :param radius: metres
    :param colour: its tint and the opacity of its middle (the rim is denser)
    :param rings: the sphere's rings (twice as many segments)
    """
    import math

    from panda3d.core import (
        Geom,
        GeomNode,
        GeomTriangles,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        Shader,
        TransparencyAttrib,
        Vec4,
    )

    use_glsl_dir()
    data = GeomVertexData("bubble", GeomVertexFormat.get_v3n3(), Geom.UH_static)
    vertex = GeomVertexWriter(data, "vertex")
    normal = GeomVertexWriter(data, "normal")
    segments = 2 * rings
    for i in range(rings + 1):
        theta = math.pi * i / rings
        for j in range(segments + 1):
            phi = 2.0 * math.pi * j / segments
            n = (
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            )
            normal.add_data3(*n)
            vertex.add_data3(*(radius * c for c in n))
    triangles = GeomTriangles(Geom.UH_static)
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            triangles.add_vertices(a, b, a + 1)
            triangles.add_vertices(a + 1, b, b + 1)
    geom = Geom(data)
    geom.add_primitive(triangles)
    node = GeomNode("translucent_bubble")
    node.add_geom(geom)
    bubble = NodePath(node)
    bubble.set_transparency(TransparencyAttrib.M_alpha)
    bubble.set_two_sided(True)
    bubble.set_depth_write(False)
    bubble.set_light_off()
    bubble.set_bin("fixed", 60)  # after the clouds' bin
    bubble.set_shader(Shader.make(Shader.SL_GLSL, _BUBBLE_VERT, _BUBBLE_FRAG))
    bubble.set_shader_input("uBubbleColour", Vec4(*colour))
    no_cover(bubble)
    return bubble
