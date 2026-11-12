#version 330
// Lays the raymarch (clouds.march) over one depth range's pass, premultiplied
// "over", at the cloud's own depth: the scene's depth test then hides it
// behind anything nearer, pixel by pixel, at the view's resolution (a game's
// model inside a cloud keeps the cloud in front of it, which the march put
// there, and loses any the history left behind it).
//
// One march covers both ranges; each range's card lays the pixels that belong
// to it: the near range's card where its depth pass drew something (the near
// camera draws it over the far range's pass afterwards, so the clouds in
// front of it must come after it), the far range's card everywhere else (in
// its place among the far pass's bins: after the sky, before the cirrus).
uniform sampler2D uClouds;     // premultiplied
uniform sampler2D uDistance;   // the cloud's distance from the camera, km, 0 none
uniform vec2 uDownscale;       // the view's pixels per pixel of theirs, across and up
uniform sampler2D uNearDepth;  // the near range's depth pass
uniform int uNearDepthOn;      // 0: not drawn (nothing of the slab in the near range)
uniform int uDepthScale;       // view pixels per pixel of the depth pass, per axis
uniform int uNearCard;         // 1: the near range's card, 0: the far range's
// This card's own range's depth pass (its lens: the same depth space as
// gl_FragDepth here), and 1 to cut the cloud's depth at it.
uniform sampler2D uSceneDepth;
uniform int uSceneDepthCut;
uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ProjectionMatrixInverse;

in vec2 vNdc;
out vec4 fragColor;

// Whether the near camera may draw a surface at this view pixel. The depth
// pass is coarser than the view: tested at this pixel's own texel alone, a
// pixel at the edge of the near range's surfaces (the ground where it crosses
// the split, a tree there) could hold a surface the texel missed. The far card
// laid the cloud there, the near camera then drew its surface over it: a
// dotted line of unclouded ground along the split, seen through a cloud. Any
// texel about it counts; the near card, laying a pixel the near camera leaves
// empty, puts the same cloud over the same far pass.
bool nearSurfaceAt(ivec2 pixel) {
    if (uNearDepthOn == 0) {
        return false;
    }
    ivec2 last = textureSize(uNearDepth, 0) - 1;
    ivec2 texel = pixel / uDepthScale;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 tap = clamp(texel + ivec2(x, y), ivec2(0), last);
            if (texelFetch(uNearDepth, tap, 0).r < 1.0) {
                return true;
            }
        }
    }
    return false;
}

void main() {
    bool nearSurface = nearSurfaceAt(ivec2(gl_FragCoord.xy));
    if (nearSurface != (uNearCard != 0)) {
        discard;
    }
    // Upsampled bilinearly (exactly the texel at full resolution); the
    // textures may be padded to powers of two.
    vec2 uv = gl_FragCoord.xy / uDownscale / vec2(textureSize(uClouds, 0));
    vec4 cloud = texture(uClouds, uv);
    float distance = texture(uDistance, uv).r * 1000.0;
    if (cloud.a <= 0.0 || distance <= 0.0) {
        discard;
    }
    vec4 near = p3d_ProjectionMatrixInverse * vec4(vNdc, -1.0, 1.0);
    vec4 far = p3d_ProjectionMatrixInverse * vec4(vNdc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w - near.xyz / near.w);
    vec4 clip = p3d_ProjectionMatrix * vec4(dir * distance, 1.0);
    // Just short of the cleared depth, or a cloud at the far plane fails the test.
    float depth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 0.99999);
    // No farther than the scene the march stopped at: everything it returned
    // lies in front of it. The resolved distance lags the fresh samples (the
    // history's mix): flying forward the clouds come nearer each frame, the
    // lagged distance put them behind the ground they hang over, and the
    // depth test cut them (90 % of a stratus lost; the edge against the
    // mountains sliding). The scene's depth does not lag.
    if (uSceneDepthCut != 0) {
        float scene = texelFetch(uSceneDepth, ivec2(gl_FragCoord.xy) / uDepthScale, 0).r;
        depth = min(depth, scene - 1e-6);
    }
    gl_FragDepth = depth;
    fragColor = cloud;
}
