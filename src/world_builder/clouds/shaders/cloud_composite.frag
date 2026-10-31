#version 330
// Lays the raymarch (clouds.march) over one depth range's pass, premultiplied
// "over", at the cloud's own depth: the scene's depth test then hides it
// behind anything nearer, pixel by pixel, at the view's resolution (a game's
// model inside a cloud keeps the cloud in front of it, which the march put
// there, and loses any the history left behind it).
uniform sampler2D uClouds;     // premultiplied
uniform sampler2D uDistance;   // the cloud's distance from the camera, km, 0 none
uniform float uDownscale;      // the view's pixels per pixel of theirs, per axis
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
    vec4 near = p3d_ProjectionMatrixInverse * vec4(vNdc, -1.0, 1.0);
    vec4 far = p3d_ProjectionMatrixInverse * vec4(vNdc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w - near.xyz / near.w);
    vec4 clip = p3d_ProjectionMatrix * vec4(dir * distance, 1.0);
    // Just short of the cleared depth, or a cloud at the far plane fails the test.
    gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 0.99999);
    fragColor = cloud;
}
