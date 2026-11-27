"""
Depth partitioning: one view drawn by a far camera, then a near camera.

A planet seen from orbit and a rock under the camera do not fit one 24-bit
depth buffer: with the near plane at 1 m, depth steps are hundreds of metres
at 100 km. :class:`SplitDepthCameras` (the trick KSP uses with its scaled and
local cameras) draws the scene twice into the same window: first a far camera
covering [split, far], then the usual camera covering [near, split] after
clearing depth. Each range gets the whole depth buffer, so over one planet
precision stays under a metre at the far end of each at any altitude (see
:meth:`SplitDepthCameras.update`; :meth:`SplitDepthCameras.system_ranges`
trades some of it far away for reaching other bodies).

Both cameras see the whole scene graph and Panda culls each to its own
frustum, so only objects straddling the split are drawn twice. Nothing else
changes: standard depth, the window's multisampling, existing shaders.

Masks: each camera has one bit of its own, :data:`NEAR_BIT` and
:data:`FAR_BIT` (its mask, :data:`NEAR_CAMERA_MASK` or
:data:`FAR_CAMERA_MASK`, is that bit alone). ``node.hide(HIDE_FROM_NEAR)``
keeps a node out of the near pass (e.g. a sky dome, which belongs to the far
pass); other cameras (the ocean's reflection) are unaffected. Panda draws a
node for a camera if any of the camera's bits is still shown on it, so a
camera with every bit but one could never be hidden from by one bit: with
the near camera's mask "all but FAR_BIT", ``hide(HIDE_FROM_NEAR)`` did
nothing, and the sky was kept out of the near pass only by lying beyond its
range.

Culling: Panda builds a lens's culling frustum (``Lens.make_bounds``) with its
far end cut at ``1 / lens-far-limit``, 10,000 km by default, whatever the far
distance. Past it a whole body was culled while its depth range still
covered it (a planet seen from beyond its moon, 10,000 km away, vanished; its
clouds' deck and the sky's atmosphere, which are never culled, stayed). The
far camera gets explicit cull bounds to its true far plane instead
(:func:`frustum_bounds`).

Translucent volumes: the near camera reaches ``overlap`` times the split, so
the two ranges never leave a gap. An opaque surface drawn twice in that band
looks the same, but a big translucent one without depth write (a game's
shield bubble) is blended twice there, which shows as a ring around the
camera at the split distance. :meth:`SplitDepthCameras.update` takes such
volumes and moves the split off every one it would cut
(:func:`split_clear_of`): nearer the camera, in front of the volume (finer
depth steps), or past its far side when it is too close for that (the camera
inside or right next to it).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from panda3d.core import (
    BitMask32,
    BoundingHexahedron,
    BoundingSphere,
    Camera,
    ConfigVariableBool,
    DepthTestAttrib,
    FrameBufferProperties,
    Lens,
    NodePath,
    PandaNode,
    Point3,
    PTA_int,
    RenderState,
)

#: Camera bit of the objects' shadow map's camera
#: (:class:`world_builder.shadows.casters.CasterMap`): render is hidden from
#: it, and only the casters show through.
CASTER_BIT = 20
#: Camera bit only the near camera has (outside the ocean reflection's 0-19).
NEAR_BIT = 21
#: Camera bit only the far camera has.
FAR_BIT = 22
#: ``node.hide(HIDE_FROM_NEAR)``: drawn by the far camera only.
HIDE_FROM_NEAR = BitMask32.bit(NEAR_BIT)
#: The ocean's reflection camera's mask: camera bits 0-19. Panda draws a node
#: for a camera if any of the camera's bits is still shown on it, so a node is
#: hidden from the reflection by hiding *all* of these
#: (``node.hide(REFLECTION_CAMERA_MASK)``); hiding one bit does nothing. The
#: main cameras keep other bits (21 and 22 here, or all of them by default)
#: and still see it.
REFLECTION_CAMERA_MASK = BitMask32(0xFFFFF)
#: ``node.hide(HIDE_FROM_FAR)``: drawn by the near camera only.
HIDE_FROM_FAR = BitMask32.bit(FAR_BIT)
#: The near camera's mask: its bit alone, so :data:`HIDE_FROM_NEAR` hides
#: from it.
NEAR_CAMERA_MASK = BitMask32.bit(NEAR_BIT)
#: The far camera's mask: its bit alone, so :data:`HIDE_FROM_FAR` hides
#: from it.
FAR_CAMERA_MASK = BitMask32.bit(FAR_BIT)
#: Camera bits of the half-resolution cloud pass's near and far cameras
#: (``world_builder.clouds.half_res``).
CLOUD_NEAR_BIT = 23
CLOUD_FAR_BIT = 24
#: ``node.hide(HIDE_FROM_CLOUD_PASS)``: not drawn into the half-resolution
#: cloud pass (where everything but the clouds only writes depth), e.g. the
#: sky, which hides nothing.
HIDE_FROM_CLOUD_PASS = BitMask32.bit(CLOUD_NEAR_BIT) | BitMask32.bit(CLOUD_FAR_BIT)
#: The tag key the cloud pass's cameras give their own state by
#: (``Camera.set_tag_state``): ``"terrain"`` on a terrain's root draws it
#: there with a depth-only fragment stage.
PASS_TAG = "wb_pass"


#: The split is never moved nearer than this off a translucent volume, metres:
#: the near camera keeps a useful range
MIN_SPLIT_M = 50.0


def camera_span(node: NodePath, camera: NodePath) -> tuple[float, float] | None:
    """How far a drawn volume reaches from the camera.

    :param node: the volume
    :param camera: the camera
    :returns: the nearest and farthest distances of its bounding sphere from
        the camera, metres (the nearest 0 with the camera inside); None when
        it is not drawn (removed, hidden, off the scene graph or empty)
    """
    if node.is_empty() or node.is_hidden() or node.get_parent().is_empty():
        return None
    # A copy, in the parent's frame: the node's own transform is applied
    bounds = node.get_bounds()
    if bounds.is_empty() or bounds.is_infinite():
        return None
    sphere = BoundingSphere()
    sphere.extend_by(bounds)
    sphere.xform(node.get_parent().get_mat(camera))
    distance = float(np.linalg.norm(np.array(sphere.get_center())))
    radius = sphere.get_radius()
    return max(0.0, distance - radius), distance + radius


def split_clear_of(
    split: float,
    spans: Iterable[tuple[float, float]],
    overlap: float,
    min_split: float = MIN_SPLIT_M,
) -> float:
    """A split moved so that no span crosses the band both cameras draw,
    [split, split * overlap].

    It moves nearer the camera, to just in front of the span it would cut,
    for as long as that stays at or past *min_split*; past that it moves
    beyond the far side of the spans instead, and keeps going out.

    :param split: the wanted split, metres
    :param spans: (nearest, farthest) distances of each translucent volume
        from the camera, metres (:func:`camera_span`)
    :param overlap: how far past the split the near camera reaches (a factor)
    :param min_split: the nearest the split may move, metres
    :returns: the split, metres
    """
    spans = sorted(spans)
    nearer = True
    moved = True
    while moved:
        moved = False
        for near, far in spans:
            in_front = near / overlap
            # Crosses the band; tested in the very arithmetic of the moves
            # below, so a split just moved off a span never finds it crossing
            # again by a rounding error
            if in_front < split and far > split:
                if nearer and in_front >= min_split:
                    split = in_front
                else:
                    nearer = False
                    split = far
                moved = True
    return split


def frustum_bounds(lens: Lens) -> BoundingHexahedron:
    """A perspective lens's frustum to its true far plane.

    What ``lens.make_bounds()`` returns, without its far end cut at
    ``1 / lens-far-limit`` (10,000 km by default). No film offset. For a
    camera's ``set_cull_bounds``, which Panda then culls with instead.

    :param lens: a ``PerspectiveLens``
    """
    fov = lens.get_fov()
    half_x = np.tan(np.radians(fov[0]) / 2.0)
    half_z = np.tan(np.radians(fov[1]) / 2.0)

    def corners(distance):
        x, z = distance * half_x, distance * half_z
        return [
            Point3(-x, distance, -z),
            Point3(x, distance, -z),
            Point3(x, distance, z),
            Point3(-x, distance, z),
        ]

    return BoundingHexahedron(*corners(lens.get_far()), *corners(lens.get_near()))


class SplitDepthCameras:
    """A far camera under the app's camera, drawn first into the same window.

    Takes over the app's camera: its mask becomes :data:`NEAR_CAMERA_MASK`
    and its lens's near/far are set by :meth:`update` every frame.
    :meth:`destroy` restores its mask and depth clearing (the lens keeps the
    last near/far).

    The far camera draws into the same output as the app's camera, whatever
    it is: the window, or an offscreen buffer when the game renders the scene
    into a texture (a post-process pipeline, e.g. ``FilterManager``). Build
    this after such a pipeline, and destroy it before the pipeline is torn
    down.

    :ivar far_camera: the far camera's NodePath (a child of ``app.camera``)
    :ivar far_lens: its lens, a copy of ``app.camLens`` kept in step with it
    :ivar near_region: the app camera's display region (clears depth)
    :ivar far_region: the far camera's, on the same output, sorted just
        before it
    :ivar output: the window or buffer both cameras draw into
    :ivar split: the current split distance, metres (0 before :meth:`update`)
    :ivar far: the current far distance, metres
    """

    def __init__(self, app: Any, near: float = 1.0, overlap: float = 1.0001) -> None:
        """
        :param app: the ShowBase (``camera``, ``cam``, ``camLens``, ``win``)
        :param near: near plane of the near camera, metres
        :param overlap: the near camera reaches this factor past the split, so
            the two ranges overlap by a sliver and never leave a gap
        """
        self.app = app
        self.near = near
        self.overlap = overlap
        self.far_lens = app.camLens.make_copy()
        far_node = Camera("far_camera", self.far_lens)
        far_node.set_camera_mask(FAR_CAMERA_MASK)
        self.far_camera = app.camera.attach_new_node(far_node)
        app.cam.node().set_camera_mask(NEAR_CAMERA_MASK)

        self.near_region = app.cam.node().get_display_region(0)
        left, right, bottom, top = self.near_region.get_dimensions()
        # The near camera's own output: the window, or the offscreen buffer a
        # post-process pipeline moved it into
        self.output = self.near_region.get_window()
        self.far_region = self.output.make_display_region(left, right, bottom, top)
        self.far_region.set_sort(self.near_region.get_sort() - 1)
        self.far_region.set_camera(self.far_camera)
        self.near_region.set_clear_depth_active(True)
        self.near_region.set_clear_depth(1.0)
        self.split = 0.0
        self.far = 0.0

    def update(
        self, split: float, far: float, translucent: Iterable[NodePath] = ()
    ) -> None:
        """Set the ranges for this frame: near camera [near, split], far
        camera [split, far].

        Depth steps at the end of a range are about d^2 / (near * 2^24): pick
        *split* near the nearest thing the far camera must draw finely and
        *far* just past the farthest thing visible.

        :param split: metres; moved off *translucent* (see the module)
        :param far: metres; raised to 2 * split if smaller
        :param translucent: big translucent volumes (no depth write) the
            split must not cut, or they show a ring at the split
        """
        split = max(split, 2.0 * self.near)
        spans = (camera_span(volume, self.app.cam) for volume in translucent)
        split = split_clear_of(
            split,
            [span for span in spans if span is not None],
            self.overlap,
            min_split=max(MIN_SPLIT_M, 2.0 * self.near),
        )
        far = max(far, 2.0 * split)
        self.split, self.far = split, far
        lens = self.app.camLens
        lens.set_near_far(self.near, split * self.overlap)
        self.far_lens.set_near_far(split, far)
        # Follow window resizes and FOV changes.
        self.far_lens.set_fov(lens.get_fov())
        self.far_lens.set_aspect_ratio(lens.get_aspect_ratio())
        self.far_camera.node().set_cull_bounds(frustum_bounds(self.far_lens))

    @staticmethod
    def planet_ranges(
        radius: float,
        camera_distance: float,
        max_height: float = 9000.0,
        min_height: float = 0.0,
        near_split: float = 2000.0,
    ) -> tuple[float, float]:
        """Split and far distances for a camera over a sphere.

        The split sits halfway down to the highest possible ground, so the far
        camera starts well short of anything; the far plane sits just past the
        horizon of the highest ground.

        :param radius: sea-level radius, metres
        :param camera_distance: camera distance from the planet centre, metres
        :param max_height: highest ground above sea level, metres
        :param min_height: lowest ground that can block the view (0 with an
            ocean), metres
        :param near_split: smallest split, metres
        :returns: (split, far)
        """
        altitude = camera_distance - radius
        split = max(near_split, 0.5 * (altitude - max_height))
        floor = radius + min_height
        eye = math.sqrt(max(0.0, camera_distance**2 - floor**2))
        peak = math.sqrt(max(0.0, (radius + max_height) ** 2 - floor**2))
        return split, 1.05 * (eye + peak)

    @staticmethod
    def system_ranges(
        bodies: Iterable[tuple[float, ...]], near_split: float = 2000.0
    ) -> tuple[float, float]:
        """Split and far distances for a camera among several bodies.

        The split follows the nearest body (it bounds the nearest ground);
        the far plane reaches the farthest visible part of any body. Depth
        steps at a distance only depend on the split, so reaching a moon
        10,000 km away costs nothing up close: a few km of step on the moon
        itself, which is a few pixels wide per km anyway.

        :param bodies: (radius, camera distance from its centre, max height,
            min height) per body, as :meth:`planet_ranges` takes them
        :param near_split: smallest split, metres
        :returns: (split, far)
        """
        ranges = [
            SplitDepthCameras.planet_ranges(*body, near_split=near_split)
            for body in bodies
        ]
        return min(r[0] for r in ranges), max(r[1] for r in ranges)

    def destroy(self) -> None:
        """Remove the far camera and its display region."""
        self.output.remove_display_region(self.far_region)
        self.far_camera.remove_node()
        self.app.cam.node().set_camera_mask(PandaNode.get_all_camera_mask())
        self.near_region.set_clear_depth_active(False)


# ── one reversed-Z camera (FRAME_COST_PLAN.md, Phase 2; issue #57) ───────────

#: ``uDepthMode`` (on render): the split's standard depth, near at 0 and far
#: at 1 in each range.
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
    draw of the scene where :class:`SplitDepthCameras` made two.

    A camera of its own under ``app.camera`` (as the split's far camera),
    with both of the split's bits (a node kept to one range is drawn), the
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
    :ivar near: the current near plane, metres (:meth:`near_for`)
    :ivar far: the cull distance, metres (0 before :meth:`update`)
    :ivar mode: ``uDepthMode``
    """

    #: The near plane follows this share of the height above the ground ...
    NEAR_SHARE = 0.01
    #: ... up to this, metres: from orbit [-1, 1] depth keeps bodies apart.
    MAX_NEAR = 1000.0

    def __init__(
        self, app: Any, near: float = 1.0, scene_buffer: bool | None = None
    ) -> None:
        """
        :param app: the ShowBase (``camera``, ``cam``, ``camLens``, ``render``,
            ``graphicsEngine``)
        :param near: the nearest the near plane comes, metres
        :param scene_buffer: draw into a float-depth buffer presented in the
            window; None = unless the app camera's output has float depth
        """
        self.app = app
        self.min_near = near
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

        # Fresh properties: an FBO's (the defaults' back buffers refuse it).
        props = FrameBufferProperties()
        props.set_rgba_bits(8, 8, 8, 8)
        props.set_depth_bits(32)
        props.set_float_depth(True)
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

    @classmethod
    def near_for(cls, height_above_ground: float, near: float) -> float:
        """:returns: the near plane for a camera *height_above_ground* metres
        over the nearest ground: :attr:`NEAR_SHARE` of it, within
        [*near*, :attr:`MAX_NEAR`]"""
        return min(max(cls.NEAR_SHARE * height_above_ground, near), cls.MAX_NEAR)

    def update(self, far: float, height_above_ground: float) -> None:
        """Per frame: the near plane for the height, the app lens's field of
        view, the culling cut at *far* (the lens itself reaches infinity), the
        scene buffer at the window's size."""
        source = self.app.camLens
        self.near = self.near_for(height_above_ground, self.min_near)
        self.far = max(far, 2.0 * self.near)
        for lens in (self.lens, self._cull_lens):
            lens.set_fov(source.get_fov())
            lens.set_aspect_ratio(source.get_aspect_ratio())
        reversed_lens(self.lens, self.near)
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
        self.camera.remove_node()
        self.app_region.set_active(True)
        _DEPTH_MODE[0] = DEPTH_STANDARD
