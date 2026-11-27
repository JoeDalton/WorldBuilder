#version 330

#pragma include "../../shaders/depth_convention.glsl"
// Lays the raymarch (clouds.march) over the view, premultiplied "over", at
// the cloud's own depth: the scene's depth test then hides it behind anything
// nearer, pixel by pixel, at the view's resolution (a game's model inside a
// cloud keeps the cloud in front of it, which the march put there, and loses
// any the history left behind it). In the clouds' slot among the view's bins:
// after the sky, before the cirrus.
uniform sampler2D uClouds;     // premultiplied
uniform sampler2D uDistance;   // the cloud's distance from the camera, km, 0 none
uniform vec2 uDownscale;       // the view's pixels per pixel of theirs, across and up
uniform int uDepthScale;       // view pixels per texel of the scene's depth (1)
// The scene's depth (the view's own, last frame's: the same depth space as
// gl_FragDepth here), and 1 to cut the cloud's depth at it.
uniform sampler2D uSceneDepth;
uniform int uSceneDepthCut;
uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ProjectionMatrixInverse;

in vec2 vNdc;
out vec4 fragColor;

void main() {
    // Upsampled bilinearly (exactly the texel at full resolution); the
    // textures may be padded to powers of two.
    vec2 uv = gl_FragCoord.xy / uDownscale / vec2(textureSize(uClouds, 0));
    vec4 cloud = texture(uClouds, uv);
    float distance = texture(uDistance, uv).r * 1000.0;
    if (cloud.a <= 0.0 || distance <= 0.0) {
        discard;
    }
    // (p3d_ProjectionMatrix holds the GSG's 0-1 remap under clip control:
    // its near plane is the convention's too, but at 0 in standard 0-1 depth,
    // which no camera uses.)
    vec4 near = p3d_ProjectionMatrixInverse * vec4(vNdc, depthNearZ(), 1.0);
    vec4 far = p3d_ProjectionMatrixInverse * vec4(vNdc, depthFarNdc(), 1.0);
    vec3 dir = normalize(far.xyz / far.w - near.xyz / near.w);
    vec4 clip = p3d_ProjectionMatrix * vec4(dir * distance, 1.0);
    // Just off the cleared depth, or a cloud at the far plane fails the test.
    float depth = depthOffClear(clamp(depthWindow(clip), 0.0, 1.0));
    // No farther than the scene the march stopped at: everything it returned
    // lies in front of it. The resolved distance lags the fresh samples (the
    // history's mix): flying forward the clouds come nearer each frame, the
    // lagged distance put them behind the ground they hang over, and the
    // depth test cut them (90 % of a stratus lost; the edge against the
    // mountains sliding). The scene's depth does not lag.
    if (uSceneDepthCut != 0) {
        float scene = texelFetch(uSceneDepth, ivec2(gl_FragCoord.xy) / uDepthScale, 0).r;
        depth = depthNoFartherThan(depth, scene);
    }
    gl_FragDepth = depth;
    fragColor = cloud;
}
