#version 330
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "cloud_field.glsl"
// The cirrus sheet's stamps (clouds.cirrus): quads lying on the sheet's sphere,
// positioned in the planet frame (the node sits at the planet's centre). The
// air in front and the sunlight's tint are worked out per vertex: they change
// little across a stamp, and per fragment they would be paid for every stamp
// overlapping a pixel.
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrixInverse;
uniform int uOwnAtmosphere;

in vec4 p3d_Vertex;           // on the sheet, planet frame
in vec2 p3d_MultiTexCoord0;   // in the atlas
in float i_opacity;

out vec3 vPosition;
// The camera drawing this pass, planet frame: the sea's mirror has its own,
// and which crossing of the sheet a fragment is depends on it.
flat out vec3 vEye;
out vec2 vUv;
out float vOpacity;
out vec3 vAtmoT;
out vec3 vAtmoS;
out vec3 vSunTint;
out float vSkyTint;
out vec3 vTwilight;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_ClipDistance[0] = 1.0;  // above the sea (see cloud_sphere.vert)
    vec3 p = p3d_Vertex.xyz;
    vPosition = p;
    vEye = (p3d_ModelViewMatrixInverse * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vUv = p3d_MultiTexCoord0;
    vOpacity = i_opacity;

    vec3 rel = p - vEye;
    float dist = max(length(rel), 1e-3);
    vAtmoT = vec3(1.0);
    vAtmoS = vec3(0.0);
    if ((uCloudDebug & 2) == 0) {
        atmosphereAlong(rel / dist, dist, sunDir, uAtmoVertexSteps, vAtmoT, vAtmoS);
    }
    vSunTint = vec3(1.0);
    vSkyTint = 1.0;
    vTwilight = vec3(0.0);
    if (uOwnAtmosphere >= 0) {
        // Through the planet's air at the cirrus's own height: lit longer than
        // the cumulus below, and reddened later (cloud_field.glsl).
        vSunTint = cloudSunTint(uOwnAtmosphere, p, sunDir);
        vTwilight = cloudTwilight(uOwnAtmosphere, p, sunDir, vSunTint);
        vSkyTint = smoothstep(-0.3, 0.1, dot(normalize(p), sunDir));
    }
}
