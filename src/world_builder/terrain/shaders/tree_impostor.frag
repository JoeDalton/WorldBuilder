#version 330

// A tree's impostor (tree_impostor.vert): its baked view's colour and normal
// at the sprite's point, cut out by their coverage, lit as its mesh is
// (tree_common.glsl's treeColor). The dithered bands give the meshes the
// near side and the ground's far look the far side.

#pragma include "tree_common.glsl"

uniform sampler2DArray uImpostorAlbedo;
uniform sampler2DArray uImpostorNormal;
uniform int uFrames;

flat in int vLayer;
flat in vec2 vTile;
flat in vec3 vT1;
flat in vec3 vT2;
flat in vec3 vUp;
flat in vec3 vSun;
flat in vec3 vSky;
flat in vec3 vAirT;
flat in vec3 vAirS;
flat in float vTint;
flat in vec2 vScreenUp;
flat in float vFadeIn;
flat in float vFadeOut;

out vec4 fragColor;

void main() {
    float dither = treeDither();
    if (dither > vFadeIn || dither > vFadeOut) {
        discard;
    }
    // The sprite's point, y up, in its view's tile: turned so that the
    // tile's up (the tree's vertical in its baked view) follows the
    // vertical on screen (tree_impostor.vert's vScreenUp: a rolled camera).
    // The tile is the tree's bounding sphere: turned, the tree stays in it.
    vec2 offset = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y) - 0.5;
    vec2 at = vec2(dot(offset, vec2(vScreenUp.y, -vScreenUp.x)), dot(offset, vScreenUp)) + 0.5;
    if (any(lessThan(at, vec2(0.0))) || any(greaterThan(at, vec2(1.0)))) {
        discard;
    }
    vec2 uv = (vTile + at) / float(uFrames);
    vec4 base = texture(uImpostorAlbedo, vec3(uv, float(vLayer)));
    if (base.a < 0.5) {
        discard;
    }
    vec3 n = texture(uImpostorNormal, vec3(uv, float(vLayer))).xyz * 2.0 - 1.0;
    vec3 N = normalize(vT1 * n.x + vT2 * n.y + vUp * n.z);
    vec3 albedo = treeAlbedo(base.rgb, vTint);
    fragColor = treeColor(albedo, N, vUp, vSun, vSky, vAirT, vAirS, at.y);
}
