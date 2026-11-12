#version 330

// A tree asset near the camera (tree_mesh.vert): its texture's colour (as
// authored: decoded to linear light), cut out by its alpha (the leaves'
// cards), lit by the light reckoned at its vertices. Crossed over to the
// impostors by a dithered band (tree_impostor.frag takes the rest).

#pragma include "tree_common.glsl"

uniform sampler2D p3d_Texture0;

in vec2 vUv;
in vec3 vNormal;
in vec3 vUp;
in vec3 vSun;
in vec3 vSky;
in vec3 vAirT;
in vec3 vAirS;
in float vFade;
in float vTint;
in float vHeight;

out vec4 fragColor;

void main() {
    vec4 base = texture(p3d_Texture0, vUv);
    if (base.a < 0.5 || treeDither() > vFade) {
        discard;
    }
    vec3 N = normalize(vNormal) * (gl_FrontFacing ? 1.0 : -1.0);
    vec3 albedo = treeAlbedo(base.rgb, vTint);
    fragColor = treeColor(albedo, N, normalize(vUp), vSun, vSky, vAirT, vAirS,
                          clamp(vHeight / 12.0, 0.0, 1.0));
}
