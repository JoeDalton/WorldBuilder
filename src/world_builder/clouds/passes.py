"""
The clouds' passes' shared pieces: the render states a pass camera starts
from (everything but the clouds into depth only), a camera sharing a split
camera's lens, and a fullscreen card laying a premultiplied texture over one
depth range in the clouds' slot. The raymarch (:mod:`march`) draws its depth
passes and its composite with them. (They were the half-resolution
billboards' pass's, :mod:`half_res`, removed with the billboards.)
"""

from __future__ import annotations

from typing import Any

from panda3d.core import (
    BitMask32,
    Camera,
    CardMaker,
    ColorBlendAttrib,
    ColorWriteAttrib,
    CullBinManager,
    Lens,
    LVecBase2f,
    NodePath,
    OmniBoundingVolume,
    RenderState,
    Shader,
    ShaderAttrib,
    Texture,
    TransparencyAttrib,
)

from world_builder.depth import (
    PASS_TAG,
)
from world_builder.terrain.terrain import depth_only_shader

#: The composites' cull bin: after the sky (21), where the clouds draw
#: without this pass (22-28), before the transparent bin (30).
LAYER_BIN = "cloud_layer"
LAYER_BIN_SORT = 22

_VERTEX = """
#version 330
in vec4 p3d_Vertex;
uniform vec2 uScale;
out vec2 vUv;
void main() {
    // The card's x and z span -1..1: clip space directly, whatever the lens.
    vUv = (p3d_Vertex.xz * 0.5 + 0.5) * uScale;
    gl_Position = vec4(p3d_Vertex.x, p3d_Vertex.z, 0.0, 1.0);
}
"""

_FRAGMENT = """
#version 330
uniform sampler2D uClouds;
in vec2 vUv;
out vec4 fragColor;
void main() {
    fragColor = texture(uClouds, vUv);  // premultiplied
}
"""


def pass_states() -> tuple[RenderState, RenderState]:
    """The render states of a camera drawing the scene for a cloud pass:
    everything writes depth only (initial), the terrain through its
    depth-only fragment stage (its tag state, ``PASS_TAG`` "terrain").

    :returns: (initial, terrain)
    """
    initial = RenderState.make(ColorWriteAttrib.make(ColorWriteAttrib.C_off))
    terrain = RenderState.make(
        ShaderAttrib.make(depth_only_shader()),
        ColorWriteAttrib.make(ColorWriteAttrib.C_off),
        1,  # over the sea's own shader below the terrain's root
    )
    return initial, terrain


def pass_camera(
    app: Any,
    name: str,
    lens: Lens,
    bit: int,
    initial: RenderState,
    terrain: RenderState,
    point_scale: float = 1.0,
) -> NodePath:
    """A camera of a cloud pass: under the app's camera, sharing a split
    camera's very lens (so it follows its range, fov and aspect every
    frame), seeing what is shown to *bit* with :func:`pass_states`, the
    trees' impostors sized in its pixels (*point_scale*: its pixels per view
    pixel, across).

    :returns: the camera's NodePath
    """
    node = Camera(name, lens)
    node.set_camera_mask(BitMask32.bit(bit))
    node.set_initial_state(initial)
    node.set_tag_state_key(PASS_TAG)
    node.set_tag_state("terrain", terrain)
    from world_builder.terrain.trees import IMPOSTOR_TAG, pass_state

    node.set_tag_state(IMPOSTOR_TAG, pass_state(point_scale))
    return app.camera.attach_new_node(node)


def composite_card(app: Any, name: str, texture: Texture, shown_to: int) -> NodePath:
    """A fullscreen card laying *texture* (premultiplied) over one depth
    range's pass, in the clouds' slot (:data:`LAYER_BIN`), upsampled
    bilinearly. Its ``uScale`` input is the share of the texture in use.

    :param shown_to: the bit of the only main camera that draws it
    :returns: the card
    """
    _ensure_layer_bin()
    maker = CardMaker(name)
    maker.set_frame(-1.0, 1.0, -1.0, 1.0)
    card = app.camera.attach_new_node(maker.generate())
    card.node().set_bounds(OmniBoundingVolume())
    card.node().set_final(True)
    card.set_shader(_composite_shader())
    card.set_shader_input("uClouds", texture)
    card.set_shader_input("uScale", LVecBase2f(1.0, 1.0))
    card.set_bin(LAYER_BIN, 0)
    card.set_depth_test(False)
    card.set_depth_write(False)
    card.set_transparency(TransparencyAttrib.M_none)
    card.set_attrib(
        ColorBlendAttrib.make(
            ColorBlendAttrib.M_add,
            ColorBlendAttrib.O_one,
            ColorBlendAttrib.O_one_minus_incoming_alpha,
        )
    )
    card.set_two_sided(True)
    # Only its own range's main camera lays it on.
    card.hide(BitMask32.all_on() & ~BitMask32.bit(shown_to))
    return card


_SHADER: list[Shader] = []


def _composite_shader() -> Shader:
    if not _SHADER:
        _SHADER.append(Shader.make(Shader.SL_GLSL, vertex=_VERTEX, fragment=_FRAGMENT))
    return _SHADER[0]


def _ensure_layer_bin() -> None:
    manager = CullBinManager.get_global_ptr()
    if manager.find_bin(LAYER_BIN) == -1:
        manager.add_bin(LAYER_BIN, CullBinManager.BT_fixed, LAYER_BIN_SORT)
