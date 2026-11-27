"""
Depth: one camera over the whole view, reversed and infinite (#57).

A planet seen from orbit and a rock under the camera do not fit a standard
24-bit depth buffer: with the near plane at 1 m, depth steps are hundreds of
metres at 100 km. :class:`ReversedDepthCamera` draws the view once with a
reversed, infinite projection (near at 1, infinity at 0) into a 32-bit float
depth buffer, whose exponent keeps the depth error about 1e-7 of the
distance at every range. It replaced the split cameras (a far camera over
[split, far], then the near one over [near, split]: every object near the
split drawn twice, the scene culled twice, translucent volumes cut at the
split).

The convention: OpenGL's 0-1 depth through clip control
(``gl-depth-zero-to-one``, read when the GL context is made: set it before
the window opens), or, without it (macOS), its [-1, 1], which keeps about a
24-bit range's precision: far bodies' depths may then fight. The near plane
stays put either way (a float depth's precision does not depend on it, and
one following the altitude cut whatever came near: a game's cockpit, a
wingman tens of metres off).
Shaders that pin or read depth go through ``shaders/depth_convention.glsl``
and the shared ``uDepthMode`` (:func:`depth_mode_input`).

Masks: the view camera carries two bits, :data:`NEAR_BIT` and
:data:`FAR_BIT` (those of the split's two cameras, kept): a node hidden from
both (``hide(HIDE_FROM_NEAR | HIDE_FROM_FAR)``) is out of the view, e.g. the
clouds' decks, drawn in the sea's reflection alone. Panda draws a node for a
camera if any of the camera's bits is still shown on it.

Culling: Panda builds a lens's culling frustum (``Lens.make_bounds``) with its
far end cut at ``1 / lens-far-limit``, 10,000 km by default, whatever the far
distance, and the reversed lens has none. The camera gets explicit cull
bounds to the world's far distance instead (:func:`frustum_bounds`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from panda3d.core import (
    BitMask32,
    BoundingPlane,
    Camera,
    ConfigVariableBool,
    DepthTestAttrib,
    FrameBufferProperties,
    IntersectionBoundingVolume,
    Lens,
    LPlane,
    NodePath,
    PTA_int,
    RenderState,
)

#: Camera bit of the objects' shadow map's camera
#: (:class:`world_builder.shadows.casters.CasterMap`): render is hidden from
#: it, and only the casters show through.
CASTER_BIT = 20
#: The view camera's two bits (outside the ocean reflection's 0-19), those of
#: the split's near and far cameras it replaced.
NEAR_BIT = 21
FAR_BIT = 22
#: Hiding a node from both (``HIDE_FROM_NEAR | HIDE_FROM_FAR``) keeps it out
#: of the view.
HIDE_FROM_NEAR = BitMask32.bit(NEAR_BIT)
#: The ocean's reflection camera's mask: camera bits 0-19. Panda draws a node
#: for a camera if any of the camera's bits is still shown on it, so a node is
#: hidden from the reflection by hiding *all* of these
#: (``node.hide(REFLECTION_CAMERA_MASK)``); hiding one bit does nothing. The
#: main cameras keep other bits (21 and 22 here, or all of them by default)
#: and still see it.
REFLECTION_CAMERA_MASK = BitMask32(0xFFFFF)
HIDE_FROM_FAR = BitMask32.bit(FAR_BIT)
NEAR_CAMERA_MASK = BitMask32.bit(NEAR_BIT)
FAR_CAMERA_MASK = BitMask32.bit(FAR_BIT)


def frustum_bounds(lens: Lens) -> IntersectionBoundingVolume:
    """A perspective lens's frustum to its true far plane, as six planes.

    What ``lens.make_bounds()`` covers, without its far end cut at
    ``1 / lens-far-limit`` (10,000 km by default). No film offset. For a
    camera's ``set_cull_bounds``, which Panda then culls with instead.

    Planes, not a ``BoundingHexahedron``: Panda carries the cull volume into
    each node's frame and a hexahedron then rebuilds its planes from its
    corners, in single precision. With corners 20,000 km out its side planes
    came out a metre or two off near the camera, and culled a game's cockpit
    parts at the screen's edge, more or less as the far distance moved: a
    flickering cockpit. A plane carries over exactly; the side ones go
    through the camera.

    :param lens: a ``PerspectiveLens``
    :returns: the planes' intersection (each one's normal outward)
    """
    fov = lens.get_fov()
    half_x = math.tan(math.radians(fov[0]) / 2.0)
    half_z = math.tan(math.radians(fov[1]) / 2.0)
    volume = IntersectionBoundingVolume()

    def side(x: float, y: float, z: float, d: float = 0.0) -> None:
        n = math.sqrt(x * x + y * y + z * z)
        volume.add_component(BoundingPlane(LPlane(x / n, y / n, z / n, d)))

    side(0.0, 1.0, 0.0, -lens.get_far())  # far
    side(0.0, -1.0, 0.0, lens.get_near())  # near
    side(1.0, -half_x, 0.0)  # right
    side(-1.0, -half_x, 0.0)  # left
    side(0.0, -half_z, 1.0)  # top
    side(0.0, -half_z, -1.0)  # bottom
    return volume


def planet_far(
    radius: float,
    camera_distance: float,
    max_height: float = 9000.0,
    min_height: float = 0.0,
) -> float:
    """The far distance for a camera over a sphere: just past the horizon of
    its highest ground.

    :param radius: sea-level radius, metres
    :param camera_distance: camera distance from the planet centre, metres
    :param max_height: highest ground above sea level, metres
    :param min_height: lowest ground that can block the view (0 with an
        ocean), metres
    :returns: metres
    """
    floor = radius + min_height
    eye = math.sqrt(max(0.0, camera_distance**2 - floor**2))
    peak = math.sqrt(max(0.0, (radius + max_height) ** 2 - floor**2))
    return 1.05 * (eye + peak)


def system_far(bodies: Iterable[tuple[float, ...]]) -> float:
    """The far distance among several bodies: the farthest visible part of
    any (:func:`planet_far` each).

    :param bodies: (radius, camera distance from its centre, max height, min
        height) per body
    :returns: metres
    """
    return max(planet_far(*body) for body in bodies)


# ── one reversed-Z camera (FRAME_COST_PLAN.md, Phase 2; issue #57) ───────────

#: ``uDepthMode``: standard depth, near at 0 and far at 1 (no depth camera:
#: the app's own).
DEPTH_STANDARD = 0
#: Reversed depth with OpenGL's 0-1 range (``gl-depth-zero-to-one``, clip
#: control): near at 1, infinity at 0, a float depth buffer's full precision.
DEPTH_REVERSED_01 = 1
#: Reversed depth in OpenGL's [-1, 1] (no clip control: macOS): near at 1,
#: infinity at 0 after the window's remap, about a 24-bit range's precision.
DEPTH_REVERSED_11 = 2


#: The one ``uDepthMode`` every renderer binds (a node's own inputs win over
#: render's, so each binds this same array): the reversed camera writes it.
_DEPTH_MODE = PTA_int.empty_array(1)


def depth_mode_input() -> PTA_int:
    """:returns: the shared ``uDepthMode`` input (:data:`DEPTH_STANDARD` until
    a :class:`ReversedDepthCamera` is made): bind it on the nodes whose
    shaders pin or read depth"""
    return _DEPTH_MODE


def zero_to_one() -> bool:
    """:returns: whether OpenGL's depth runs 0-1 (``gl-depth-zero-to-one``,
    read by the GSG when it is made: set it before the window opens)"""
    return bool(ConfigVariableBool("gl-depth-zero-to-one", False).get_value())


def reversed_depth_mode() -> int:
    """:returns: the reversed camera's ``uDepthMode`` under this PRC setting"""
    return DEPTH_REVERSED_01 if zero_to_one() else DEPTH_REVERSED_11


def far_plane_test(mode: int) -> int:
    """:returns: the ``DepthTestAttrib`` mode for what sits on the far plane
    (the sky: ``depth_convention.glsl``'s ``depthFarZ``) in depth *mode*: the
    camera's own test with equality ("less or equal" against the standard
    clear of 1, "greater or equal" against the reversed clear of 0), as a
    24-bit buffer rounds the far plane onto the clear value"""
    if mode == DEPTH_STANDARD:
        return DepthTestAttrib.M_less_equal
    return DepthTestAttrib.M_greater_equal


def float_depth_fbprops() -> FrameBufferProperties:
    """:returns: Panda's default framebuffer properties with a 32-bit float
    depth buffer: what a game passes when it opens its window (or makes its
    scene buffer) for :class:`ReversedDepthCamera`'s full precision (Panda's
    default properties cannot be changed from code)"""
    props = FrameBufferProperties(FrameBufferProperties.get_default())
    props.set_depth_bits(32)
    props.set_float_depth(True)
    return props


def scene_buffer_props(host: FrameBufferProperties) -> FrameBufferProperties:
    """:returns: the properties of :class:`ReversedDepthCamera`'s scene buffer
    for a view that drew into an output of *host*'s: an FBO's (fresh: the
    defaults' back buffers refuse it), 8-bit colour, 32-bit float depth, and
    *host*'s multisamples (a game's anti-aliasing pipeline keeps its MSAA)"""
    props = FrameBufferProperties()
    props.set_rgba_bits(8, 8, 8, 8)
    props.set_depth_bits(32)
    props.set_float_depth(True)
    props.set_multisamples(host.get_multisamples())
    return props


def view_test(mode: int) -> int:
    """:returns: the ``DepthTestAttrib`` mode of the view's own cameras in
    depth *mode*: "less" (standard), "greater" (reversed)"""
    if mode == DEPTH_STANDARD:
        return DepthTestAttrib.M_less
    return DepthTestAttrib.M_greater


def reversed_lens(lens: Lens, near: float) -> Lens:
    """Make *lens* reversed and infinite: Panda takes near = inf, far = the
    near plane as that (near maps to 1, infinity to -1 in the lens's [-1, 1],
    to 0 after the GSG's 0-1 mapping or the window's remap).

    :returns: *lens*
    """
    lens.set_near_far(math.inf, near)
    return lens


class ReversedDepthCamera:
    """One camera over the whole depth range, reversed (#57): one cull and one
    draw of the scene (the split cameras it replaced made two).

    A camera of its own under ``app.camera``, with both view bits
    (:data:`NEAR_BIT`, :data:`FAR_BIT`), the
    "greater" depth test as its initial state, and a reversed, infinite copy
    of ``app.camLens`` (:func:`reversed_lens`) kept in step with its field of
    view; culling is cut at :attr:`far`. ``app.camLens`` stays a standard
    lens (a game's picking and ray maths keep working); the app camera's
    region is switched off. The shared ``uDepthMode``
    (:func:`depth_mode_input`, also bound on ``render``) becomes
    :func:`reversed_depth_mode` for the shaders that pin or read depth.
    :meth:`destroy` gives it all back.

    Full precision needs a float depth buffer, which a window's framebuffer
    does not have: the view is drawn into a scene buffer of its own (32-bit
    float depth, the window's size) and a card in the window presents it,
    under the window's other regions (the HUD). A game whose window or scene
    output already has float depth (:func:`float_depth_fbprops`) can pass
    ``scene_buffer=False``.

    :ivar camera: its NodePath (a child of ``app.camera``)
    :ivar lens: its lens
    :ivar region: its display region (in :attr:`output`)
    :ivar output: what it draws into: the scene buffer, or the app camera's
        output
    :ivar present: the window's region showing the scene buffer (None
        without one)
    :ivar app_region: the app camera's, switched off meanwhile
    :ivar near: the near plane, metres
    :ivar far: the cull distance, metres (0 before :meth:`update`)
    :ivar mode: ``uDepthMode``
    :ivar depth_texture: the view's depth, copied after each frame (the
        clouds read last frame's)
    """

    def __init__(
        self, app: Any, near: float = 1.0, scene_buffer: bool | None = None
    ) -> None:
        """
        :param app: the ShowBase (``camera``, ``cam``, ``camLens``, ``render``,
            ``graphicsEngine``)
        :param near: the near plane, metres
        :param scene_buffer: draw into a float-depth buffer presented in the
            window; None = unless the app camera's output has float depth
        """
        self.app = app
        self.near = near
        self.far = 0.0
        self.mode = reversed_depth_mode()
        self.lens = reversed_lens(app.camLens.make_copy(), near)
        self._cull_lens = app.camLens.make_copy()
        node = Camera("reversed_camera", self.lens)
        node.set_camera_mask(NEAR_CAMERA_MASK | FAR_CAMERA_MASK)
        node.set_initial_state(
            RenderState.make(DepthTestAttrib.make(DepthTestAttrib.M_greater))
        )
        self.camera = app.camera.attach_new_node(node)
        self.app_region = app.cam.node().get_display_region(0)
        window = self.app_region.get_window()
        if scene_buffer is None:
            scene_buffer = not window.get_fb_properties().get_float_depth()
        self.present = None
        self._window = window
        self._scene: NodePath | None = None
        self._texture = None
        self._card: NodePath | None = None
        if scene_buffer:
            self.output = self._make_scene_buffer(window)
            self.region = self.output.make_display_region()
        else:
            self.output = window
            left, right, bottom, top = self.app_region.get_dimensions()
            self.region = window.make_display_region(left, right, bottom, top)
            self.region.set_sort(self.app_region.get_sort())
        self.region.set_camera(self.camera)
        self.region.set_clear_depth_active(True)
        self.region.set_clear_depth(0.0)
        # The view's depth, copied after each frame (not bound: the clouds'
        # composite reads it while the view draws): the clouds read it the
        # next frame instead of drawing the scene's depth again.
        from panda3d.core import GraphicsOutput, Texture

        self.depth_texture = Texture("view_depth")
        self.output.add_render_texture(
            self.depth_texture,
            GraphicsOutput.RTM_copy_texture,
            GraphicsOutput.RTP_depth,
        )
        self.app_region.set_active(False)
        _DEPTH_MODE[0] = self.mode
        app.render.set_shader_input("uDepthMode", _DEPTH_MODE)

    def _make_scene_buffer(self, window: Any) -> Any:
        """The view's float-depth buffer at the window's size, and the
        window's card that presents it where the app camera's region was."""
        from panda3d.core import (
            CardMaker,
            GraphicsOutput,
            GraphicsPipe,
            OrthographicLens,
            Texture,
            WindowProperties,
        )

        props = scene_buffer_props(window.get_fb_properties())
        self._texture = Texture("reversed_scene")
        buffer = self.app.graphicsEngine.make_output(
            window.get_pipe(),
            "reversed_scene",
            window.get_sort() - 1,
            props,
            WindowProperties.size(window.get_x_size(), window.get_y_size()),
            GraphicsPipe.BF_refuse_window | GraphicsPipe.BF_resizeable,
            window.get_gsg(),
            window,
        )
        if buffer is None:
            raise RuntimeError("no float-depth scene buffer could be made")
        buffer.add_render_texture(
            self._texture, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_color
        )
        buffer.set_clear_color_active(True)
        buffer.set_clear_color(window.get_clear_color())
        self._scene = NodePath("reversed_present")
        lens = OrthographicLens()
        lens.set_film_size(2.0, 2.0)
        lens.set_near_far(-1.0, 1.0)
        camera = self._scene.attach_new_node(Camera("reversed_present", lens))
        maker = CardMaker("reversed_present")
        maker.set_frame(-1.0, 1.0, -1.0, 1.0)
        self._card = self._scene.attach_new_node(maker.generate())
        self._card.set_texture(self._texture)
        self._card.set_depth_test(False)
        self._card.set_depth_write(False)
        self._card.set_light_off()
        left, right, bottom, top = self.app_region.get_dimensions()
        self.present = window.make_display_region(left, right, bottom, top)
        self.present.set_sort(self.app_region.get_sort())
        self.present.set_camera(camera)
        return buffer

    @property
    def float_depth(self) -> bool:
        """Whether the output's depth buffer is a float one."""
        return bool(self.output.get_fb_properties().get_float_depth())

    def update(self, far: float) -> None:
        """Per frame: the app lens's field of view, the culling cut at *far*
        (the lens itself reaches infinity), the scene buffer at the window's
        size."""
        source = self.app.camLens
        self.far = max(far, 2.0 * self.near)
        for lens in (self.lens, self._cull_lens):
            lens.set_fov(source.get_fov())
            lens.set_aspect_ratio(source.get_aspect_ratio())
        self._cull_lens.set_near_far(self.near, self.far)
        self.camera.node().set_cull_bounds(frustum_bounds(self._cull_lens))
        if self.present is not None:
            from panda3d.core import TextureStage

            size = (self._window.get_x_size(), self._window.get_y_size())
            if size != (self.output.get_x_size(), self.output.get_y_size()):
                self.output.set_size(*size)
            # The texture may be padded to powers of two.
            self._card.set_tex_scale(
                TextureStage.get_default(), self._texture.get_tex_scale()
            )

    def destroy(self) -> None:
        """Remove the camera, its region and scene buffer; the app camera
        draws again and ``uDepthMode`` is standard."""
        if self.present is not None:
            self._window.remove_display_region(self.present)
            self.app.graphicsEngine.remove_window(self.output)
            self._scene.remove_node()
        else:
            self.output.remove_display_region(self.region)
            _remove_render_texture(self.output, self.depth_texture)
        self.camera.remove_node()
        self.app_region.set_active(True)
        _DEPTH_MODE[0] = DEPTH_STANDARD


def _remove_render_texture(output: Any, texture: Any) -> None:
    """Take *texture* off *output*'s render textures, keeping the others
    (Panda only clears them all)."""
    kept = [
        (output.get_texture(i), output.get_rtm_mode(i), output.get_texture_plane(i))
        for i in range(output.count_textures())
        if output.get_texture(i) != texture
    ]
    output.clear_render_textures()
    for tex, mode, plane in kept:
        output.add_render_texture(tex, mode, plane)
