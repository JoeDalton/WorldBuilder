// The depth convention (world_builder.depth): uDepthMode is the shared input
// every node that pins or reads depth binds (depth.depth_mode_input()); a
// ReversedDepthCamera sets it. Include once per shader with #pragma include.
//   DEPTH_STANDARD     standard depth (no depth camera): near 0, far 1
//   DEPTH_REVERSED_01  one reversed camera, OpenGL's 0-1 (clip control):
//                      near 1, infinity 0, a float buffer's full precision
//   DEPTH_REVERSED_11  the same in OpenGL's [-1, 1] (macOS): near 1, infinity
//                      -1 before the window's remap to 0

#ifndef DEPTH_CONVENTION_GLSL
#define DEPTH_CONVENTION_GLSL

const int DEPTH_STANDARD = 0;
const int DEPTH_REVERSED_01 = 1;
const int DEPTH_REVERSED_11 = 2;

uniform int uDepthMode;

// The clip-space z that puts a vertex of clip-space w on the far plane, just
// inside it (the clip test keeps it): what is drawn there (the sky, the
// stars) is behind everything else, whatever the camera's range.
float depthFarZ(float w) {
    if (uDepthMode == DEPTH_REVERSED_01) {
        return w * 1e-12;  // under any body's depth (near / distance)
    }
    if (uDepthMode == DEPTH_REVERSED_11) {
        return -w * (1.0 - 1e-7);
    }
    return w * 0.999999;
}

// The near plane in clip space is z = depthNearZ() * w: the standard
// projection's -1 (OpenGL's [-1, 1]), the reversed one's +1 (near maps to 1
// in both of its conventions, remapped or not).
float depthNearZ() {
    return uDepthMode == DEPTH_STANDARD ? -1.0 : 1.0;
}

// A depth pass's cleared value: what a texel holds where nothing was drawn.
float depthClear() {
    return uDepthMode == DEPTH_STANDARD ? 1.0 : 0.0;
}

// Whether a depth pass drew something at a texel of this depth.
bool depthDrawn(float depth) {
    return uDepthMode == DEPTH_STANDARD ? depth < 1.0 : depth > 0.0;
}

// The clip-space z (over w) of a point toward the view's far end, finite, for
// a ray's direction from the near plane: the standard far plane, or (reversed
// and infinite) twice the near distance, which reversed 0-1 depth puts at
// 0.5 and [-1, 1] at 0. "Clip space" is the view's own: under clip control
// (DEPTH_REVERSED_01) it holds the GSG's 0-1 remap, as p3d_*Matrix do and as
// the march's matrices are given (march.py, _MarchLayer.update).
float depthFarNdc() {
    if (uDepthMode == DEPTH_REVERSED_01) {
        return 0.5;
    }
    return uDepthMode == DEPTH_STANDARD ? 1.0 : 0.0;
}

// A depth texel's clip-space z (over w), in that same space: the texel
// itself under clip control (exact: a float buffer's precision kept), else
// OpenGL's window remap undone.
float depthNdc(float depth) {
    return uDepthMode == DEPTH_REVERSED_01 ? depth : depth * 2.0 - 1.0;
}

// The window depth of a clip-space point through p3d_ProjectionMatrix (which
// holds the GSG's 0-1 remap under clip control), for gl_FragDepth.
float depthWindow(vec4 clip) {
    float ndc = clip.z / clip.w;
    return uDepthMode == DEPTH_REVERSED_01 ? ndc : ndc * 0.5 + 0.5;
}

// A depth just off the cleared value toward the camera, so that what is drawn
// at the far end still passes the camera's test.
float depthOffClear(float depth) {
    return uDepthMode == DEPTH_STANDARD ? min(depth, 0.99999) : max(depth, 1e-7);
}

// *depth*, but no farther than the scene's depth *scene* (and just in front
// of it): the camera's nearer side.
float depthNoFartherThan(float depth, float scene) {
    return uDepthMode == DEPTH_STANDARD ? min(depth, scene - 1e-6)
                                        : max(depth, scene * (1.0 + 1e-4));
}

#endif  // DEPTH_CONVENTION_GLSL
