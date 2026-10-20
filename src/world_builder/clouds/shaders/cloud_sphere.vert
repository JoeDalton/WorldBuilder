#version 330
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "cloud_field.glsl"
// Planet-mode billboards (clouds.sphere): one tile's particles, positioned
// relative to the tile node, which sits on the planet. The quad faces the
// camera, upright against the local vertical. Outputs as cloud.vert, with
// worldPos in the PLANET frame (centre at the origin), which is where
// cloud.frag samples the density field.
uniform mat4 p3d_ModelMatrix;          // tile -> render
uniform mat4 p3d_ViewProjectionMatrix;
uniform vec3 uCamRender;               // camera, render frame
uniform vec3 uPlanetCentre;            // planet centre, render frame
uniform float nearFadeRadii;
uniform int uOwnAtmosphere;

in vec4 p3d_Vertex;                    // particle position relative to the tile
in vec2 p3d_MultiTexCoord0;            // quad corner [-1, +1]
in float i_radius;
in vec4  i_uv_st;
in float i_layer;

out vec2  vUv;
out vec3  worldPos;
out float vRadius;
out float vFade;
flat out int vLayer;
// The air in front of the billboard and its sunlight's tint, per vertex: they
// change little across a billboard, and per fragment they would be paid for
// every layer of overdraw.
out vec3 vAtmoT;
out vec3 vAtmoS;
out vec3 vSunTint;
out float vSkyTint;

// Samples of the air per vertex (8 per pixel elsewhere): it barely changes
// across a billboard.
const int AIR_STEPS = 4;

void main() {
    vLayer = int(i_layer + 0.5);
    vec3 centre = (p3d_ModelMatrix * vec4(p3d_Vertex.xyz, 1.0)).xyz;
    vec3 fwd = normalize(uCamRender - centre);
    vec3 upRef = normalize(centre - uPlanetCentre);
    if (abs(dot(fwd, upRef)) > 0.99) {
        upRef = abs(fwd.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    }
    vec3 right = normalize(cross(upRef, fwd));
    vec3 up = cross(fwd, right);

    vec4 shape = layerParams[vLayer * LAYER_VEC4S + 10];
    float aspect = shape.x;
    float stretch = aspect > 1.0001 ? sqrt(aspect) : 1.0;

    vec2 corner = p3d_MultiTexCoord0;
    vec3 wp = centre + right * (corner.x * i_radius * stretch)
                     + up * (corner.y * i_radius / stretch);
    gl_Position = p3d_ViewProjectionMatrix * vec4(wp, 1.0);
    worldPos = wp - uPlanetCentre;
    vUv = i_uv_st.xy + (corner * 0.5 + 0.5) * i_uv_st.zw;
    vRadius = i_radius;
    vec3 rel = wp - uCamRender;
    float dist = max(length(rel), 1e-3);
    vAtmoT = vec3(1.0);
    vAtmoS = vec3(0.0);
    if ((uCloudDebug & 2) == 0) {
        atmosphereAlong(rel / dist, dist, sunDir, AIR_STEPS, vAtmoT, vAtmoS);
    }
    vSunTint = vec3(1.0);
    vSkyTint = 1.0;
    if (uOwnAtmosphere >= 0) {
        // Relative to the noon sun, half the reddening: see cloud.frag.
        vec3 p = centre - uPlanetCentre;
        vec3 vertical = normalize(p);
        vec3 through = atmoSunlight(uOwnAtmosphere, p, sunDir)
                     / max(atmoSunlight(uOwnAtmosphere, p, vertical), vec3(1e-3));
        vSunTint = sqrt(min(through, vec3(1.0)));
        vSkyTint = smoothstep(-0.3, 0.1, dot(vertical, sunDir));  // see cloud.frag
    }
    vFade = 1.0;
    if (nearFadeRadii > 0.0) {
        vFade *= smoothstep(0.0, nearFadeRadii * i_radius, length(uCamRender - centre));
    }
}
