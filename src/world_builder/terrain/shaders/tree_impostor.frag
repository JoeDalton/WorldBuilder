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
flat in float vFadeIn;
flat in float vFadeOut;

out vec4 fragColor;

void main() {
    float dither = treeDither();
    if (dither > vFadeIn || dither > vFadeOut) {
        discard;
    }
    // The sprite's point, y up, in its view's tile.
    vec2 at = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
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
