#version 330

// A star over at most 2x2 pixels: a tent of one pixel's radius around its
// true (sub-pixel) position, which sums to 1 over the pixels it touches, so
// a star keeps its brightness as it moves across them (no twinkling). Added
// to the sky behind it after the tonemap, as the difference it makes.

#pragma include "sky_common.glsl"

in vec3 vBase;
in vec3 vStar;
out vec4 fragColor;

void main() {
    vec2 offset = abs(gl_PointCoord - 0.5) * 3.0;  // pixels from the star
    float weight = max(0.0, 1.0 - offset.x) * max(0.0, 1.0 - offset.y);
    if (weight <= 0.0) {
        discard;
    }
    vec3 added = aces_tonemap(vBase + vStar * weight) - aces_tonemap(vBase);
    fragColor = vec4(max(added, vec3(0.0)), 0.0);
}
