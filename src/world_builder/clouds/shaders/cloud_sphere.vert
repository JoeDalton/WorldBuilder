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
out vec3 vTwilight;

// Samples of the air per vertex (8 per pixel elsewhere): it barely changes
// across a billboard.

void main() {
    vLayer = int(i_layer + 0.5);
    vec3 centre = (p3d_ModelMatrix * vec4(p3d_Vertex.xyz, 1.0)).xyz;
    vec4 shape = layerParams[vLayer * LAYER_VEC4S + 10];
    float aspect = shape.x;
    float stretch = aspect > 1.0001 ? sqrt(aspect) : 1.0;

    float centreDistance = length(uCamRender - centre);
    vFade = 1.0;
    if (nearFadeRadii > 0.0) {
        vFade = smoothstep(0.0, nearFadeRadii * i_radius, centreDistance);
    }
    vec3 fwd = normalize(uCamRender - centre);
    vec3 upRef = normalize(centre - uPlanetCentre);
    if (abs(dot(fwd, upRef)) > 0.99) {
        upRef = abs(fwd.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    }
    vec3 right = normalize(cross(upRef, fwd)) * (i_radius * stretch);
    vec3 up = cross(fwd, normalize(right)) * (i_radius / stretch);

    // Drop a billboard that would draw nothing, before the per-vertex air
    // and sunlight (every pass runs this for every billboard of every shown
    // tile: the ocean's mirror sees few of them). Exact, cloud.frag would
    // discard every fragment:
    // * wholly outside its shell's distance band, or faded out near the
    //   camera (a coarse shell's tile around the camera holds many such,
    //   large on screen);
    // * wholly outside the view: all four corners beyond one clip plane.
    float reach = i_radius * stretch * 1.5;  // past the quad's corners
    vec4 band = layerParams[vLayer * LAYER_VEC4S + 11];
    bool dropped = centreDistance + reach <= band.x
                || centreDistance - reach >= band.w || vFade <= 0.0;
    if (!dropped) {
        vec4 c0 = p3d_ViewProjectionMatrix * vec4(centre - right - up, 1.0);
        vec4 c1 = p3d_ViewProjectionMatrix * vec4(centre + right - up, 1.0);
        vec4 c2 = p3d_ViewProjectionMatrix * vec4(centre + right + up, 1.0);
        vec4 c3 = p3d_ViewProjectionMatrix * vec4(centre - right + up, 1.0);
        vec4 x = vec4(c0.x, c1.x, c2.x, c3.x);
        vec4 y = vec4(c0.y, c1.y, c2.y, c3.y);
        vec4 z = vec4(c0.z, c1.z, c2.z, c3.z);
        vec4 w = vec4(c0.w, c1.w, c2.w, c3.w);
        dropped = all(greaterThan(x, w)) || all(lessThan(x, -w))
               || all(greaterThan(y, w)) || all(lessThan(y, -w))
               || all(greaterThan(z, w)) || all(lessThan(z, -w));
    }
    if (dropped) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        gl_ClipDistance[0] = 1.0;
        vUv = vec2(0.0);
        worldPos = vec3(0.0);
        vRadius = 0.0;
        vFade = 0.0;
        vAtmoT = vec3(1.0);
        vAtmoS = vec3(0.0);
        vSunTint = vec3(1.0);
        vSkyTint = 1.0;
        vTwilight = vec3(0.0);
        return;
    }

    vec2 corner = p3d_MultiTexCoord0;
    vec3 wp = centre + right * corner.x + up * corner.y;
    gl_Position = p3d_ViewProjectionMatrix * vec4(wp, 1.0);
    // The clouds are all above the sea: the ocean's reflection pass (whose
    // clip plane is on) keeps them whole.
    gl_ClipDistance[0] = 1.0;
    worldPos = wp - uPlanetCentre;
    vUv = i_uv_st.xy + (corner * 0.5 + 0.5) * i_uv_st.zw;
    vRadius = i_radius;
    vec3 rel = wp - uCamRender;
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
        // See cloud.frag.
        vec3 p = centre - uPlanetCentre;
        vSunTint = cloudSunTint(uOwnAtmosphere, p, sunDir);
        vTwilight = cloudTwilight(uOwnAtmosphere, p, sunDir, vSunTint);
        vSkyTint = smoothstep(-0.3, 0.1, dot(normalize(p), sunDir));
    }
    // Other bodies eclipsing the sun, at the billboard's centre: the sun,
    // the lit air above (twilight) and the sky dim together.
    float eclipse = eclipseVisibility(centre - uCamRender, sunDir, uShadowSelf);
    // And the mountains between the cloud and a low sun (the body's map).
    vec3 planet = centre - uPlanetCentre;
    vSunTint *= eclipse * terrainShadow(normalize(planet), length(planet) - planetRadius);
    // And the clouds far along it, past the fragment's own march: the long
    // shadows of a low sun across the deck (only their slow part, here).
    if (uCloudFarReach > 0.0) {
        vSunTint *= exp2(-cloudFarDepth(getField(vLayer), planet, sunDir, uCloudFarReach));
    }
    vTwilight *= eclipse;
    vSkyTint *= mix(ECLIPSE_SKY_FLOOR, 1.0, eclipse);
}
