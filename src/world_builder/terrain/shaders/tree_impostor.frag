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
flat in float vSpriteScale;
flat in float vHalfTexel;

out vec4 fragColor;

uniform float uPointScale;  // the target's pixels per view pixel (tree_impostor.vert)

// The point at *offset* from the sprite's middle (in the tree's sprite, y
// up) in the baked view's tile, turned so that the tile's up (the tree's
// vertical in its baked view) follows the vertical on screen
// (tree_impostor.vert's vScreenUp: a rolled camera). The tile is the tree's
// bounding sphere: turned, the tree stays in it.
vec2 tileAt(vec2 offset) {
    return vec2(dot(offset, vec2(vScreenUp.y, -vScreenUp.x)), dot(offset, vScreenUp)) + 0.5;
}

bool inTile(vec2 at) {
    return all(greaterThanEqual(at, vec2(0.0))) && all(lessThanEqual(at, vec2(1.0)));
}

void main() {
    float dither = treeDither();
    if (dither > vFadeIn || dither > vFadeOut) {
        discard;
    }
    vec2 offset = (vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y) - 0.5) / vSpriteScale;
    if (vHalfTexel > 0.0) {
        // A coarser target (the clouds' depth passes): its pixel stands for
        // several of the view's, kept if the tree covers any of them (the
        // clouds' cards are told apart, and cut, by these passes: a view
        // pixel of the tree they missed stood unfogged in front of a cloud).
        // Its coverage read at the view's mip level: at this target's own,
        // coarser, a thin tip's coverage averaged under the cut and the
        // tree lost its top here, where the view still drew it.
        vec2 gx = (tileAt(offset + dFdx(offset) * uPointScale) - tileAt(offset)) / float(uFrames);
        vec2 gy = (tileAt(offset + dFdy(offset) * uPointScale) - tileAt(offset)) / float(uFrames);
        bool covered = false;
        for (int y = -1; y <= 1 && !covered; ++y) {
            for (int x = -1; x <= 1 && !covered; ++x) {
                vec2 at = tileAt(offset + vec2(x, y) * vHalfTexel);
                covered = inTile(at)
                       && textureGrad(uImpostorAlbedo,
                                      vec3((vTile + at) / float(uFrames), float(vLayer)),
                                      gx, gy).a >= 0.5;
            }
        }
        if (!covered) {
            discard;
        }
        fragColor = vec4(0.0);  // depth only
        return;
    }
    vec2 at = tileAt(offset);
    if (!inTile(at)) {
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
