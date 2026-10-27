// The terrain's fragment shader, shared by terrain.frag (land, and the sea's
// far look) and terrain_sea.frag (SEA_NEAR: the sea's near look too, for
// the chunks with water while a Sea shows it).

#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "../../clouds/shaders/cloud_field.glsl"
#pragma include "../../ocean/shaders/water.glsl"
#pragma include "cube_faces.glsl"
#pragma include "terrain_material.glsl"
#pragma include "object_light.glsl"

// Terrain fragment shader: biome texture blending, lighting (sunlight through
// the planet's own air), the sea (seabed lifted to the water by terrain.vert),
// aerial perspective through every atmosphere, and the ACES tonemap. "Up" is
// the local vertical of the sphere (vUp), never the render frame's +Z.
//
// The sea has two looks: near the camera the full one of sea.glsl (waves,
// the planar reflection, foam), from high up a flat water lit per pixel,
// crossfaded by uSeaNear (world_builder.ocean.Sea sets it by altitude; 0
// without a Sea).

in vec3 vRelPos;
in vec3 vPlanetPos;  // from the planet's centre, its own axes (its shadow map's)
in vec2 vUv;
in float vHeight;
in float vSeabed;
in vec3 vNormal;
in vec3 vUp;
in vec4 vBiomeA;   // snow, rock, grass, forest
in vec4 vBiomeB;   // desert, beach, the chunk's slot; skirt flag (a >= 0.5)
in vec4 vSwell;    // the swell: gradient per metre (planet axes), field
flat in int vFace;
flat in vec3 vFaceU;   // the face's texture axes, render frame
flat in vec3 vFaceV;
flat in int vLevel;    // the chunk's level
flat in float vStep;   // its vertex spacing (uv metres per step), metres

uniform sampler2D uNoise;         // four tiling noises, breaks up biome edges
uniform vec3  uMeanAlbedo[6];     // linear; drawn instead of textures far away
uniform vec3  uFlatAlbedo[6];     // linear; untextured mode and the biome view
uniform int   uTextured;
uniform float uDetailFar;         // textures fade to the mean colour by here

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
uniform int   uSeaSurface;
uniform vec3  uShallowWater;      // the sea surface's shallow tint, linear
uniform float uShallowDepth;      // and its e-folding depth, metres

uniform int   uDebugMode;  // 0 shaded, 1 LOD, 2 biomes, 3 slope, 4 normals, 5 skirts
// The planet's clouds' layer in layerParams whose shadow falls here, -1 none
// (SphericalClouds.bind_shadow_inputs).
uniform int   uCloudShadowLayer;
uniform int   uMaxLevel;

out vec4 fragColor;

#ifdef SEA_NEAR
#pragma include "../../ocean/shaders/swell.glsl"
#pragma include "../../ocean/shaders/sea.glsl"
#endif


const vec3 WATER_SHALLOW = vec3(0.03, 0.09, 0.16);  // linear; ocean.py water_color is deep
const vec3 WATER_DEEP = vec3(0.02, 0.06, 0.14);


vec3 hue(float t) {
    return clamp(abs(fract(t + vec3(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
}

// The share of the sun the planet's clouds leave here: their shadow, marched
// through the deck's field toward the sun, or its mean where the pixel's
// footprint (metres) spans several clouds (cloud_field.glsl's cloudShadow).
// Clouds only come with air: the planet frame is the air's.
float cloudShade(vec3 L, float footprint) {
    if (uCloudShadowLayer < 0 || uCloudShadowSteps <= 0 || uOwnAtmosphere < 0) {
        return 1.0;
    }
    vec3 planet = uAtmoCam[uOwnAtmosphere].xyz + vRelPos;
    return cloudShadow(getField(uCloudShadowLayer), planet, L, uCloudShadowSteps,
                       footprint);
}

void main() {
    float dist = length(vRelPos);
    vec3 N = normalize(vNormal);
    vec3 up = normalize(vUp);
    float upN = dot(N, up);
    vec3 L = normalize(uSunDir);
    // Planet-scale day factor: past the terminator the sun does not light the
    // ground, whatever the local slope says. Used on an airless body only;
    // through an atmosphere, atmoSunlight below also does the reddening.
    float day = uRadialLight == 1 ? smoothstep(-0.12, 0.08, dot(up, L)) : 1.0;
    // The ground a pixel spans, metres (taken here, out of any branch).
    float footprint = max(length(dFdx(vRelPos)), length(dFdy(vRelPos)));

#ifdef SEA_NEAR
    // Water wholly in its near look: none of the land's or the far water's
    // work below.
    if (uSeaSurface == 1 && vSeabed < 0.0 && uSeaNear >= 1.0 && uDebugMode == 0) {
        bool raw;
        float sunShade = cloudShade(L, footprint)
                       * terrainShadow(normalize(vPlanetPos), vHeight)
                       * casterShadow(vRelPos, up, L);
        vec3 near = seaShade(uCamPos + vRelPos, vSwell,
                             vRelPos / max(dist, 1e-3), dist,
                             -vSeabed, vStep, sunShade, raw);
        if (!raw) {
            near = aces_tonemap(applyAtmospheres(near, vRelPos / max(dist, 1e-3), dist, L)
                                * uExposure);
        }
        fragColor = vec4(near, 1.0);
        return;
    }
#endif

    float w[6] = float[6](vBiomeA.r, vBiomeA.g, vBiomeA.b, vBiomeA.a, vBiomeB.r, vBiomeB.g);

    // World-anchored noise wobbles the weights, so biome edges are ragged
    // rather than following the vertex grid. Multiplicative: a pure biome
    // stays pure. Mipmapping averages it out far away, where it would alias.
    vec4 nearNoise = texture(uNoise, vUv / 64.0);
    vec4 farNoise = texture(uNoise, vUv / 512.0 + vec2(0.37, 0.71));
    float noise[6] = float[6](
        nearNoise.r + farNoise.b, nearNoise.g + farNoise.a, nearNoise.b + farNoise.r,
        nearNoise.a + farNoise.g, nearNoise.r + farNoise.a, nearNoise.g + farNoise.b);
    float total = 0.0;
    for (int i = 0; i < 6; ++i) {
        w[i] *= max(0.0, 1.0 + 0.9 * (noise[i] - 1.0));
        total += w[i];
    }
    for (int i = 0; i < 6; ++i) {
        w[i] /= max(total, 1e-4);
    }

    if (uDebugMode == 1) {
        fragColor = vec4(hue(float(vLevel) / float(uMaxLevel + 1)) * (0.55 + 0.45 * upN), 1.0);
        return;
    }
    if (uDebugMode == 3) {
        float slope = degrees(acos(clamp(upN, 0.0, 1.0)));
        vec3 ramp = slope < 30.0 ? mix(vec3(0.1, 0.5, 0.1), vec3(0.9, 0.9, 0.1), slope / 30.0)
                                 : mix(vec3(0.9, 0.5, 0.1), vec3(0.8, 0.0, 0.0), clamp((slope - 30.0) / 18.0, 0.0, 1.0));
        fragColor = vec4(ramp, 1.0);
        return;
    }
    if (uDebugMode == 4) {
        fragColor = vec4(0.5 + 0.5 * N, 1.0);
        return;
    }
    if (uDebugMode == 5 && vBiomeB.a > 0.5) {
        fragColor = vec4(1.0, 0.0, 1.0, 1.0);
        return;
    }

    vec3 albedo = vec3(0.0);
    vec3 meanAlbedo = vec3(0.0);
    for (int i = 0; i < 6; ++i) {
        meanAlbedo += w[i] * (uTextured == 1 ? uMeanAlbedo[i] : uFlatAlbedo[i]);
    }
    vec3 surfaceN = N;
    float roughness = 0.85;
    float ao = 1.0;
    float far = smoothstep(0.6 * uDetailFar, uDetailFar, dist);
    vec3 Tu = normalize(vFaceU - up * dot(up, vFaceU));
    vec3 Tv = normalize(vFaceV - up * dot(up, vFaceV));
    bool water = uSeaSurface == 1 && vSeabed < 0.0;

    if (water) {
        // The far look: flat water, tinted by depth, lit below.
        albedo = mix(WATER_SHALLOW, WATER_DEEP, clamp(-vSeabed / 3000.0, 0.0, 1.0));
        albedo = mix(uShallowWater, albedo,
                     deepWaterShare(-vSeabed, abs(dot(normalize(vRelPos), up)), uShallowDepth));
        surfaceN = up;
        roughness = 0.08;
    } else if (uDebugMode == 2) {
        albedo = vec3(0.0);
        for (int i = 0; i < 6; ++i) albedo += w[i] * uFlatAlbedo[i];
    } else if (uTextured == 0 || far >= 1.0) {
        albedo = meanAlbedo;
    } else {
        layeredMaterial(w, vUv, vHeight, N, Tu, Tv, upN, dist, meanAlbedo, far,
                        albedo, surfaceN, roughness, ao);
    }

    vec3 V = normalize(-vRelPos);
    // Sunlight through this planet's own air: reddened low, gone in its
    // shadow. An airless body just has the day/night line. Other bodies
    // eclipse it.
    float eclipse = eclipseVisibility(vRelPos, L, uShadowSelf);
    // The clouds' shadow only where the sun can light this pixel at all
    // (objectLight's terms go as N.L): the march is the dear part.
    float shade = dot(surfaceN, L) > 0.0 && dot(up, L) > -0.05 ? cloudShade(L, footprint)
                                                              : 1.0;
    // The mountains' (one fetch): what the sun clears from this height.
    shade *= terrainShadow(normalize(vPlanetPos), vHeight);
    // The game's objects' (their map: nothing to do without one).
    shade *= casterShadow(vRelPos, N, L);
    vec3 sunlight = uSunColor * (eclipse * shade);
    float sunDay = day;
    if (uOwnAtmosphere >= 0) {
        sunlight *= atmoSunlight(uOwnAtmosphere, uAtmoCam[uOwnAtmosphere].xyz + vRelPos, L);
        sunDay = 1.0;
    }
    // Sky light fades through twilight, and dims in an eclipse.
    vec3 skylight = (uRadialLight == 1 ? uAmbient * smoothstep(-0.2, 0.2, dot(up, L))
                                       : uAmbient)
                  * mix(ECLIPSE_SKY_FLOOR, 1.0, eclipse);
    vec3 C = objectLight(albedo, surfaceN, roughness, ao,
                         uRadialLight == 1 ? up : surfaceN, L, V, sunlight, sunDay, skylight);
    if (water) {
        // Sky reflected at grazing angles (Schlick).
        float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(surfaceN, V), 0.0), 5.0);
        C += fresnel * skylight * 0.8;
#ifdef SEA_NEAR
        if (uSeaNear > 0.0) {
            // The near look, crossfaded in below the sea's altitude band.
            bool raw;
            vec3 near = seaShade(uCamPos + vRelPos, vSwell,
                                 vRelPos / max(dist, 1e-3), dist,
                                 -vSeabed, vStep, shade, raw);
            if (raw) {
                fragColor = vec4(near, 1.0);
                return;
            }
            C = mix(C, near, uSeaNear);
        }
#endif
    }

    if (uDebugMode == 5) {
        C = mix(C, hue(float(vLevel) / float(uMaxLevel + 1)), 0.5);
    }
    // Aerial perspective: the air between the camera and the ground, of
    // every atmosphere the sight line crosses.
    C = applyAtmospheres(C, vRelPos / max(dist, 1e-3), dist, L);
    C *= uExposure;
    fragColor = vec4(uTonemap == 1 ? aces_tonemap(C)
                                   : pow(clamp(C, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
