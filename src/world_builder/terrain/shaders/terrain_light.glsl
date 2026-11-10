// The light on the terrain's surfaces, shared by the ground (terrain_frag.glsl)
// and what grows on it (canopy.frag): the sun through the planet's own air
// and every shadow on it (other bodies, the clouds, the mountains, the
// game's objects, the rings), the sky light, the aerial perspective and the
// display encoding. Include once per shader with #pragma include, before
// anything else.

#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "../../clouds/shaders/cloud_field.glsl"
#pragma include "cube_faces.glsl"
#pragma include "object_light.glsl"

uniform vec3  uCamPos;            // the main camera, render frame
uniform vec3  uSunDir;            // toward the sun, render frame
uniform vec3  uSunColor;          // sunlight above any air
uniform vec3  uAmbient;           // sky light (near none on an airless body)
uniform float uExposure;
uniform int   uTonemap;           // 1: aces_tonemap, 0: a plain display encoding
uniform int   uRadialLight;       // 1: a planet's day and night, by its vertical;
                                  // 0: an object's (a rock): each face as it
                                  // faces the sun, the sky light from all round
uniform int   uOwnAtmosphere;     // this planet's atmosphere index, -1 = airless
// The clouds' shadow on the view's scene, a pass of the march's
// (clouds.march.GroundShadow): r the shade, g the scene's distance (km), -1
// where none, at the view's resolution over uGroundShadowScale. Off (0)
// without one: then shaded per pixel.
uniform sampler2D uGroundShadow;
uniform int   uGroundShadowOn;
uniform int   uGroundShadowScale;

// The share of the sun the planet's clouds leave here: their shadow, marched
// through the deck's field toward the sun (both families' with genera), or
// its mean where the pixel's footprint (metres) spans several clouds
// (cloud_field.glsl's planetCloudShadow; uCloudShadowLayer and Other, -1
// none: CloudVolume.bind_shadow_inputs). Clouds only come with air: the
// planet frame is the air's. Read from the ground shadow pass where it holds
// this pixel (the main view's), per pixel elsewhere (the sea's reflection).
// A texel of the ground shadow agrees with this pixel when their distances
// differ by less than this share: a slope's neighbouring texel does, the
// ground behind a silhouette (or a model in front) does not.
const float GROUND_SHADOW_AGREE = 0.08;

// The ground shadow pass's shade at this pixel: its four texels around it,
// bilinear, those of another surface left out. False where none agrees (a
// silhouette's edge, the near range when its depth pass is off).
// TERRAIN_LIGHT_VERTEX: included by a vertex stage (no pixel: never).
bool groundShadowAt(float dist, out float shade) {
#ifdef TERRAIN_LIGHT_VERTEX
    shade = 1.0;
    return false;
#else
    vec2 at = gl_FragCoord.xy / float(uGroundShadowScale) - 0.5;
    ivec2 base = ivec2(floor(at));
    vec2 f = at - vec2(base);
    ivec2 last = textureSize(uGroundShadow, 0) - 1;
    float km = dist / 1000.0;
    float total = 0.0;
    shade = 0.0;
    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            vec2 texel = texelFetch(uGroundShadow, clamp(base + ivec2(x, y), ivec2(0), last), 0).rg;
            vec2 w2 = mix(vec2(1.0) - f, f, vec2(x, y));
            float w = w2.x * w2.y + 1e-3;
            if (texel.g >= 0.0 && abs(texel.g - km) <= GROUND_SHADOW_AGREE * km) {
                shade += w * texel.r;
                total += w;
            }
        }
    }
    if (total <= 0.0) {
        return false;
    }
    shade /= total;
    return true;
#endif
}

// relPos: the point from the camera (render frame); mirror: the sea's
// reflection pass (no ground shadow pass there).
float cloudShade(vec3 L, float footprint, vec3 relPos, bool mirror) {
    if ((uCloudShadowLayer < 0 && uCloudShadowOther < 0) || uCloudShadowSteps <= 0
            || uOwnAtmosphere < 0) {
        return 1.0;
    }
    float shade;
    if (uGroundShadowOn != 0 && !mirror && groundShadowAt(length(relPos), shade)) {
        return shade;
    }
    vec3 planet = uAtmoCam[uOwnAtmosphere].xyz + relPos;
    return planetCloudShadow(planet, L, uCloudShadowSteps, footprint);
}

// The sun and sky light arriving at a surface point:
//   relPos     from the camera, render frame
//   planetPos  from the planet's centre, its own axes (the mountains' map's)
//   height     metres above sea level
//   N          the geometric normal (the objects' shadow bias)
//   surfaceN   the shading normal (whether the sun lights it at all)
//   up         the local vertical
//   footprint  the ground a pixel spans, metres
// Returns the sunlight (every shadow and the air's reddening in), the sky
// light, the sun's share past any terminator (sunDay) and the shadows'
// product alone (shade: the sea's near look takes it).
void surfaceLight(vec3 relPos, vec3 planetPos, float height, vec3 N, vec3 surfaceN,
                  vec3 up, vec3 L, float footprint, bool mirror,
                  out vec3 sunlight, out vec3 skylight, out float sunDay,
                  out float shade) {
    // Planet-scale day factor: past the terminator the sun does not light the
    // ground, whatever the local slope says. Used on an airless body only;
    // through an atmosphere, atmoSunlight below also does the reddening.
    float day = uRadialLight == 1 ? smoothstep(-0.12, 0.08, dot(up, L)) : 1.0;
    // Sunlight through this planet's own air: reddened low, gone in its
    // shadow. An airless body just has the day/night line. Other bodies
    // eclipse it.
    float eclipse = eclipseVisibility(relPos, L, uShadowSelf);
    // The clouds' shadow only where the sun can light this pixel at all
    // (objectLight's terms go as N.L): the march is the dear part.
    shade = dot(surfaceN, L) > 0.0 && dot(up, L) > -0.05
          ? cloudShade(L, footprint, relPos, mirror) : 1.0;
    // The mountains' (one fetch): what the sun clears from this height.
    shade *= terrainShadow(normalize(planetPos), height);
    // The game's objects' (their map: nothing to do without one), and the
    // rings'.
    shade *= casterShadow(relPos, N, L) * ringVisibility(relPos, L);
    sunlight = uSunColor * (eclipse * shade);
    sunDay = day;
    if (uOwnAtmosphere >= 0) {
        sunlight *= atmoSunlight(uOwnAtmosphere, uAtmoCam[uOwnAtmosphere].xyz + relPos, L);
        sunDay = 1.0;
    }
    // Sky light fades through twilight, and dims in an eclipse.
    skylight = (uRadialLight == 1 ? uAmbient * smoothstep(-0.2, 0.2, dot(up, L))
                                  : uAmbient)
             * mix(ECLIPSE_SKY_FLOOR, 1.0, eclipse);
}

// The air between the camera and a point: C becomes C * transmittance +
// inscatter (applyAtmospheres is linear in C).
void airAlong(vec3 relPos, vec3 L, out vec3 transmittance, out vec3 inscatter) {
    float dist = length(relPos);
    vec3 dir = relPos / max(dist, 1e-3);
#ifdef TERRAIN_LIGHT_VERTEX
    // A vertex just off the screen reads the aerial cells at its edge (a
    // march per vertex out of view cost more than the per-pixel light saved).
    vec4 clip = uAerialToClip * vec4(dir, 1.0);
    if (uAtmoCount > 0 && uAerialOn != 0 && clip.w > 0.0
            && dist <= uAerialRange.x + uAerialRange.y) {
        vec2 ndc = clamp(clip.xy / clip.w, vec2(-1.0), vec2(1.0));
        vec2 cell = clamp((ndc * 0.5 + 0.5) * AERIAL_CELLS, 0.5, AERIAL_CELLS - 0.5);
        float sl = sqrt(max(dist - uAerialRange.x, 0.0) / uAerialRange.y) * AERIAL_CELLS - 1.0;
        sl = clamp(sl, -1.0, AERIAL_CELLS - 1.0);
        float k = floor(sl);
        vec3 t0 = vec3(1.0), s0 = vec3(0.0);
        if (k >= 0.0) aerialSlice(k, cell, t0, s0);
        vec3 t1 = t0, s1 = s0;
        if (k + 1.0 < AERIAL_CELLS) aerialSlice(k + 1.0, cell, t1, s1);
        transmittance = mix(t0, t1, sl - k);
        inscatter = mix(s0, s1, sl - k);
        return;
    }
#endif
    inscatter = applyAtmospheres(vec3(0.0), dir, dist, L);
    transmittance = applyAtmospheres(vec3(1.0), dir, dist, L) - inscatter;
}

// The exposure and the display encoding of a radiance seen through the air.
vec4 displayColor(vec3 C) {
    C *= uExposure;
    return vec4(uTonemap == 1 ? aces_tonemap(C)
                              : pow(clamp(C, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}

// Aerial perspective (the air between the camera and the point, of every
// atmosphere the sight line crosses), the exposure and the display encoding.
vec4 surfaceColor(vec3 C, vec3 relPos, vec3 L) {
    float dist = length(relPos);
    C = applyAtmospheres(C, relPos / max(dist, 1e-3), dist, L);
    C *= uExposure;
    return vec4(uTonemap == 1 ? aces_tonemap(C)
                              : pow(clamp(C, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
