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
uniform int uNearCard;         // 1: the near range's card, 0: the far range's
uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ProjectionMatrixInverse;

in vec2 vNdc;
out vec4 fragColor;

void main() {
    bool nearSurface = uNearDepthOn != 0
                    && texelFetch(uNearDepth, ivec2(gl_FragCoord.xy), 0).r < 1.0;
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
    gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 0.99999);
    fragColor = cloud;
}
