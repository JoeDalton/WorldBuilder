#version 330
// The raymarch's full resolution (clouds.march), gathered over the frames:
// cloud_march.frag traces one pixel of every uBlock x uBlock block a frame;
// the rest come from the last frame's result, moved to where the camera now
// sees them (reprojected along the view ray to the cloud's distance), held
// within the range of this frame's samples around them so a change shows at
// once rather than trailing behind. A traced pixel blends into its history,
// which averages the steps' jitter away over the visits.
//
// The cloud is premultiplied; its distance (kilometres, 0: none) goes on with
// it, for the next frame and for the composite's depth test. The textures may be larger
// than what is drawn in them (padded to powers of two): pixels, not UVs,
// except for the history's filtered read.
uniform sampler2D uMarch;            // this frame's samples, one a block
uniform sampler2D uMarchDistance;
uniform sampler2D uHistory;          // the last frame's result, full resolution
uniform sampler2D uHistoryDistance;
uniform mat4  uClipToRel;            // clip space -> relative to the camera (world axes)
uniform mat4  uPrevRelToClip;        // relative to the camera -> the last frame's clip space
uniform int   uBlock;
uniform ivec2 uOffset;
uniform vec2  uViewSize;
uniform int   uHistoryValid;         // 0: no usable last frame (first, resized, jumped)
uniform float uFreshWeight;          // a traced pixel's share over its history

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragDistance;

// How far to look along a ray with no cloud on it (the sky at infinity).
const float NO_CLOUD = 1.0e7;

vec3 rayDir(vec2 ndc) {
    vec4 near = uClipToRel * vec4(ndc, -1.0, 1.0);
    vec4 far = uClipToRel * vec4(ndc, 1.0, 1.0);
    return normalize(far.xyz / far.w - near.xyz / near.w);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 block = pixel / uBlock;
    ivec2 last = ivec2(ceil(uViewSize / float(uBlock))) - 1;
    bool fresh = pixel - block * uBlock == uOffset;
    vec4 current = texelFetch(uMarch, min(block, last), 0);
    float distance = texelFetch(uMarchDistance, min(block, last), 0).r;

    // The range of this frame's four samples around the pixel.
    ivec2 corner = ivec2(floor((vec2(pixel) + 0.5) / float(uBlock) - 0.5));
    vec4 lo = current;
    vec4 hi = current;
    for (int i = 0; i < 4; ++i) {
        ivec2 at = clamp(corner + ivec2(i & 1, i >> 1), ivec2(0), last);
        vec4 v = texelFetch(uMarch, at, 0);
        lo = min(lo, v);
        hi = max(hi, v);
    }

    vec2 ndc = (vec2(pixel) + 0.5) / uViewSize * 2.0 - 1.0;
    vec3 point = rayDir(ndc) * (distance > 0.0 ? distance * 1000.0 : NO_CLOUD);
    vec4 previous = uPrevRelToClip * vec4(point, 1.0);
    vec2 uv = previous.xy / previous.w * 0.5 + 0.5;
    bool valid = uHistoryValid != 0 && previous.w > 0.0
              && all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));

    vec4 colour;
    float outDistance;
    if (!valid) {
        // Nothing to go on: this frame's nearest sample.
        colour = current;
        outDistance = distance;
    } else {
        vec2 at = uv * uViewSize / vec2(textureSize(uHistory, 0));
        vec4 history = clamp(texture(uHistory, at), lo, hi);
        float historyDistance = texture(uHistoryDistance, at).r;
        colour = fresh ? mix(history, current, uFreshWeight) : history;
        outDistance = fresh || historyDistance <= 0.0 ? distance : historyDistance;
    }
    if (colour.a <= 0.0) {
        outDistance = 0.0;
    }
    fragColor = colour;
    fragDistance = vec4(outDistance, 0.0, 0.0, 1.0);
}
