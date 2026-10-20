#version 330

// Planet mode (uSpherical == 1, see clouds.sphere): the slab is a height above
// the sea-level sphere, the noise is sampled at planet-frame positions, a
// weather field shifts the fbm by direction, and the atmospheres replace the
// flat distance haze. uDeck == 1 draws the far-field deck surface instead of a
// billboard. The flat field leaves all of it off.
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
// Cloud colour pass. Each billboard is a SAMPLE OF A VOLUME, not a sprite: its
// outline and colour are functions of world position alone, while the sprite only
// supplies how much volume sits behind each pixel (a thickness profile). Types mix
// by superposition in the blender, so a fragment only evaluates its own field.
//
// Premultiplied "over", back to front: with alpha = 1 - exp2(-opticalDepth), N
// overlapping billboards leave transmittance exp2(-sum of depths), i.e. the
// billboards along a view ray ARE the steps of a raymarch. See docs/scenes.md.
uniform sampler3D cloudNoise;    // one tileable octave of value noise, R8
uniform sampler2D p3d_Texture0;  // puff atlas: alpha = per-quad THICKNESS profile
uniform vec3  camPos;
uniform vec3  noiseOffset;      // wind advection, in METRES
uniform float noiseSize;        // volume resolution, for the coord → uvw divide

// LAYER_VEC4S per cloud type, mirroring noise.pack_layer_params. Size is
// MAX_LAYERS * LAYER_VEC4S (a test checks it): too small reads out of bounds, and a
// mismatch with cloud.vert fails to link.
uniform vec4 layerParams[96];

uniform vec3  sunDir;           // FROM the scene TOWARD the sun, normalised
uniform vec3  sunColor;
uniform vec3  skyColor;
uniform vec3  hazeColor;        // what distant cloud tends to
uniform float sunBrightness;
uniform float skyStrength;
uniform float skyOcclusion;     // share of sky light reaching a deck's underside
uniform float exposure;
uniform float horizonFade;      // 1 / metres over which cloud tends fully to haze
uniform float earthCurvature;   // 1 / (2R); undone here, the field is flat

uniform int   uSpherical;       // 1: planet mode (see the top)
uniform int   uDeck;            // 1: the deck surface, not a billboard
uniform float planetRadius;     // sea-level radius (planet mode)
uniform sampler3D weatherNoise; // large-scale cloudiness by direction
uniform float weatherGain;      // fbm shift, clearest to cloudiest weather
uniform int   uOwnAtmosphere;   // the planet's atmosphere (reddens sunlight), -1 none
uniform vec2  uDeckStats;       // the fbm's mean and sigma (the far deck's coverage)
uniform float uPixelAngle;      // radians per pixel (the far deck's footprint)

in  vec2  vUv;
in  vec3  worldPos;
in  float vRadius;
in  float vFade;
flat in int vLayer;
in vec3 vAtmoT;      // planet-mode billboards: the air in front, per vertex
in vec3 vAtmoS;
in vec3 vSunTint;
in float vSkyTint;
out vec4 fragColor;

const int LAYER_VEC4S = 12;

// The type's field and optics. The layer index is dynamically uniform over a draw,
// so these are cached uniform reads.
struct Field {
    vec3  noiseScale;
    vec3  noiseOrigin;    // decorrelating offset plus wind advection
    vec4  octaveScales;
    vec4  octaveWeights;
    float threshLo;
    float threshHi;
    float slabBase;
    float slabThickness;
    float baseRamp;
    float topErosion;
    float topExponent;
    float shadowGain;       // noise.shadow_calibration
    float shadowBias;
    float extinction;       // per BILLBOARD
    float fieldExtinction;  // the field's own, for the sun march and powder
    vec4  shellFade;        // (fadeInLo, fadeInHi, fadeOutLo, fadeOutHi), metres
    int   sunSteps;
    float sunStepLength;
    float shadowStrength;
    float multipleScattering;
    float powderStrength;
    float powderLength;
    float maxOpticalDepth;
    float phaseForward;     // pre-solved by noise.phase_weights
    float phaseBackward;
    float phaseIsotropic;
    float forwardAnisotropy;
};

Field getField(int layer) {
    int b = layer * LAYER_VEC4S;
    vec4 v0 = layerParams[b + 0];
    vec4 v3 = layerParams[b + 3];
    vec4 v4 = layerParams[b + 4];
    vec4 v5 = layerParams[b + 5];
    vec4 v6 = layerParams[b + 6];
    vec4 v7 = layerParams[b + 7];
    vec4 v8 = layerParams[b + 8];
    vec4 v9 = layerParams[b + 9];
    Field f;
    f.noiseScale      = v0.xyz;
    f.threshLo        = v0.w;
    f.octaveScales    = layerParams[b + 1];
    f.octaveWeights   = layerParams[b + 2];
    // Wind arrives in metres and is scaled per layer, so every layer stays locked
    // to its own field whatever its feature size.
    f.noiseOrigin     = v3.xyz + noiseOffset * f.noiseScale;
    f.threshHi        = v3.w;
    f.slabBase        = v4.x;
    f.slabThickness   = v4.y;
    f.baseRamp        = v4.z;
    f.topErosion      = v4.w;
    f.extinction      = v5.x;
    f.fieldExtinction = v5.y;
    f.topExponent     = v6.x;
    f.shadowGain      = v6.y;
    f.shadowBias      = v6.z;
    f.maxOpticalDepth = v6.w;
    f.sunSteps        = int(v7.x + 0.5);
    f.sunStepLength   = v7.y;
    f.shadowStrength  = v7.z;
    f.multipleScattering = v7.w;
    f.powderStrength  = v8.x;
    f.powderLength    = v8.y;
    f.phaseForward    = v8.z;
    f.phaseBackward   = v8.w;
    f.phaseIsotropic  = v9.x;
    f.forwardAnisotropy = v9.y;
    f.shellFade       = layerParams[b + 11];
    return f;
}

float sampleNoise(vec3 coord) {
    return texture(cloudNoise, coord / noiseSize).r;
}

// Constant per-octave offsets (drifting ones would make the clouds boil). MUST
// match OCTAVE_OFFSETS in noise.py, which drives placement; a test reads this file.
const vec3 OCTAVE_2_OFFSET = vec3( 0.31,  0.73,  0.19);
const vec3 OCTAVE_3_OFFSET = vec3(-0.57,  0.11,  0.83);
const vec3 OCTAVE_4_OFFSET = vec3( 0.67, -0.29,  0.41);

float lowOctaves(Field f, vec3 coord) {
    return sampleNoise(coord * f.octaveScales.x)                   * f.octaveWeights.x
         + sampleNoise(coord * f.octaveScales.y + OCTAVE_2_OFFSET) * f.octaveWeights.y;
}

float cloudFbm(Field f, vec3 coord) {
    return lowOctaves(f, coord)
         + sampleNoise(coord * f.octaveScales.z + OCTAVE_3_OFFSET) * f.octaveWeights.z
         + sampleNoise(coord * f.octaveScales.w + OCTAVE_4_OFFSET) * f.octaveWeights.w;
}

// The sun march's two octaves, calibrated onto the full field's distribution.
float shadowFbm(Field f, vec3 coord) {
    return lowOctaves(f, coord) * f.shadowGain + f.shadowBias;
}

// Normalised height in the slab: 0 at the base, 1 at the ceiling.
float slabHeight(Field f, float z) {
    return (z - f.slabBase) / max(f.slabThickness, 1e-3);
}

// Height of a field position: z on the flat field, above the sea-level
// sphere on a planet.
float heightOf(vec3 p) {
    return uSpherical == 1 ? length(p) - planetRadius : p.z;
}

// The weather's fbm shift at a planet-frame position (none on the flat field).
float weatherShift(vec3 p) {
    if (uSpherical == 0) {
        return 0.0;
    }
    return weatherGain * (texture(weatherNoise, normalize(p) * 0.5 + 0.5).r - 0.5);
}

// Flat bases scale DENSITY; bulging tops RAISE THE THRESHOLD (noise.threshold_lift).
// Shared by the view sample and the sun march so both agree where the cloud ends.
float slabDensity(Field f, float h, float raw) {
    float lift = f.topErosion * pow(clamp(h, 0.0, 1.0), f.topExponent);
    return smoothstep(f.threshLo + lift, f.threshHi + lift, raw)
         * (1.0 - exp2(-f.baseRamp * h));
}

float cloudDensity(Field f, vec3 p) {
    float h = slabHeight(f, heightOf(p));
    if (h <= 0.0 || h >= 1.0) return 0.0;
    return slabDensity(f, h, cloudFbm(f, p * f.noiseScale + f.noiseOrigin) + weatherShift(p));
}

float hgPhase(float x, float g) {
    float g2 = g * g;
    return 0.25 * ((1.0 - g2) * pow(1.0 + g2 - 2.0 * g * x, -1.5));
}

// The pre-solved weights put 1 at 90°, forwardGain at 0° and backwardGain at 180°,
// so no normalisation is needed here.
const float BACKWARD_G = -0.4;

float phaseFunction(Field f, float x) {
    // Clamped as a last guard: a negative radiance renders as a dark hole.
    return max(0.0,
               f.phaseForward   * hgPhase(x, f.forwardAnisotropy)
             + f.phaseBackward  * hgPhase(x, BACKWARD_G)
             + f.phaseIsotropic * 0.25);
}

// Darkens optically THIN cloud. Fed the LOCAL density over a fixed length: the
// sunward depth would invert it, and the quad's chord would tie it to quad size.
float powder(Field f, float density) {
    return 1.0 - exp2(-f.fieldExtinction * density * f.powderLength * 2.0);
}

// Optical depth toward the sun through the same field, which is what lets clouds
// shadow themselves AND each other. A depth, not a transmittance, so everything
// derived from it stays a function of world position only.
float sunOpticalDepth(Field f, vec3 p) {
    if (f.sunSteps <= 0) {
        return 0.0;
    }
    vec3 increment = sunDir * f.sunStepLength;
    vec3 position = p + increment * 0.5;
    float density = 0.0;
    for (int i = 0; i < f.sunSteps; i++, position += increment) {
        float h = slabHeight(f, heightOf(position));
        if (h <= 0.0 || h >= 1.0) {
            continue;
        }
        density += slabDensity(
            f, h, shadowFbm(f, position * f.noiseScale + f.noiseOrigin) + weatherShift(position));
    }
    return density * f.sunStepLength * f.fieldExtinction * f.shadowStrength;
}

void main() {
    Field f = getField(vLayer);

    vec3 toCamera = worldPos - camPos;
    float distance = length(toCamera);
    vec3 V = -toCamera / distance;

    float puff;
    float chord;
    vec3 samplePos;
    float density;
    // The deck far away: 1 = the field's own detail, 0 = its expected
    // coverage (the share of a pixel's columns that hold cloud), once a
    // pixel spans more than the clouds' features (they would alias to
    // speckle from orbit).
    float detail = 1.0;
    float farCoverage = 0.0;
    if (uDeck == 1) {
        // The slab seen as a surface. The sight line crosses thickness / |cos|
        // of it (capped near grazing); the density is the mean of two samples
        // through the slab's depth.
        puff = 1.0;
        vec3 up = normalize(worldPos);
        float cosView = max(abs(dot(V, up)), 0.04);
        chord = f.slabThickness / cosView;
        float r = planetRadius + f.slabBase;
        samplePos = up * (r + 0.5 * f.slabThickness);
        float feature = 4.0 / (f.noiseScale.x * f.octaveScales.x);  // NOISE_FEATURE
        detail = 1.0 - smoothstep(0.15, 0.6, distance * uPixelAngle / feature);
        density = 0.0;
        if (detail > 0.0) {
            vec3 low = up * (r + 0.3 * f.slabThickness);
            vec3 high = up * (r + 0.7 * f.slabThickness);
            density = 0.5 * (cloudDensity(f, low) + cloudDensity(f, high));
        }
        if (detail < 1.0) {
            // fbm ~ normal(mean, sigma): the share above the threshold window's
            // middle (logistic approximation of the normal CDF).
            float x = (uDeckStats.x + weatherShift(samplePos)
                       - 0.5 * (f.threshLo + f.threshHi)) / max(uDeckStats.y, 1e-4);
            farCoverage = 1.0 / (1.0 + exp(-1.702 * x));
        }
        if (density <= 0.0 && farCoverage * (1.0 - detail) < 0.003) discard;
    } else {
        puff = texture(p3d_Texture0, vUv).a;
        if (puff < 0.01) discard;                // the sprite's own soft edge
        chord = 2.0 * vRadius * puff;
        // Sampled ON the quad plane (not mid-chord, which would make overlapping
        // quads disagree); on the flat field with the vertex stage's droop added
        // back.
        samplePos = uSpherical == 1 ? worldPos : vec3(
            worldPos.xy,
            worldPos.z + dot(toCamera.xy, toCamera.xy) * earthCurvature
        );
        density = cloudDensity(f, samplePos);
        if (density <= 0.0) discard;             // THE crisp silhouette
    }

    // Aerial perspective tends to the haze COLOUR, never to transparent (a fading
    // deck would show the ground behind it). Fully hazed cloud is sky: skip it.
    // (On a planet the atmospheres do it, in linear light, below.)
    float haze = uSpherical == 1 ? 0.0 : clamp(distance * horizonFade, 0.0, 1.0);
    if (haze >= 0.999) discard;
    if (vFade <= 0.0) discard;

    // Capped so every billboard stays a thin veil, never a hard-edged opaque quad.
    float opticalDepth = min(f.extinction * density * chord, f.maxOpticalDepth);

    // LOD shell crossfade: adjacent shells' weights sum to 1. Applied to OPTICAL
    // DEPTH (exact for any depth, unlike alpha) and AFTER the cap (or two shells
    // could each reach it).
    float shellWeight = smoothstep(f.shellFade.x, f.shellFade.y, distance)
                      * (1.0 - smoothstep(f.shellFade.z, f.shellFade.w, distance));
    opticalDepth *= shellWeight;

    float alpha = 1.0 - exp2(-opticalDepth);
    if (uDeck == 1) {
        // Each cloudy column is close to opaque; the far deck is their share.
        alpha = mix(0.92 * farCoverage * shellWeight, alpha, detail);
    }
    alpha *= vFade;
    if (alpha <= 0.0) discard;

    // Lighting last, after every discard. The far deck's self-shadowing and
    // powder are their averages, not the aliasing per-pixel values.
    // Shading detail goes before coverage detail: a cloud's self-shadow is
    // smaller than the cloud and aliases sooner.
    float shading = detail;
    if (uDeck == 1) {
        float feature = 4.0 / (f.noiseScale.x * f.octaveScales.x);
        shading = 1.0 - smoothstep(0.04, 0.2, distance * uPixelAngle / feature);
    }
    float sunVisibility = shading > 0.0 ? exp2(-sunOpticalDepth(f, samplePos)) : 1.0;
    sunVisibility = mix(0.75, sunVisibility, shading);
    float powderTerm = mix(1.0, powder(f, density), f.powderStrength * shading);
    // Skylight comes from above: occlude the SKY term toward the underside only.
    float skyReach = mix(skyOcclusion, 1.0, clamp(slabHeight(f, heightOf(samplePos)), 0.0, 1.0));
    // Multiple-scattering fill: sky-coloured and NOT phase-weighted, or dusk
    // shadows turn orange and sunward shadow outshines side-lit cloud.
    float ambient = skyStrength * skyReach
                  + f.multipleScattering * sunBrightness * (1.0 - sunVisibility);

    float phase = phaseFunction(f, dot(sunDir, -V));
    vec3 sunlight = sunColor;
    vec3 skylight = skyColor;
    if (uSpherical == 1 && uDeck == 0) {
        sunlight *= vSunTint;  // per vertex (cloud_sphere.vert)
        skylight *= vSkyTint;
    } else if (uSpherical == 1 && uOwnAtmosphere >= 0) {
        // Through the planet's air: reddened low, gone in its shadow (the sky
        // light fades with it). Relative to the noon sun, so a cloud lit from
        // high up stays white, and only half the reddening (a square root):
        // what multiple scattering in and around the cloud, missing here,
        // gives back. A sunset still turns it orange.
        vec3 up = normalize(samplePos);
        vec3 through = atmoSunlight(uOwnAtmosphere, samplePos, sunDir)
                     / max(atmoSunlight(uOwnAtmosphere, samplePos, up), vec3(1e-3));
        sunlight *= sqrt(min(through, vec3(1.0)));
        skylight *= smoothstep(-0.2, 0.2, dot(up, sunDir));
    }
    vec3 colour = sunlight * (sunBrightness * sunVisibility * powderTerm * phase)
                + skylight * ambient;
    if (uSpherical == 1) {
        // The air between the camera and the cloud, in linear light. Its
        // in-scattered light is in front of the cloud, so it is weighted by
        // the cloud's alpha like the cloud itself: the background behind
        // carries the same air's light for the uncovered part.
        vec3 transmittance = vAtmoT;
        vec3 inscatter = vAtmoS;
        if (uDeck == 1) {  // one layer: per fragment is affordable
            atmosphereAlong(-V, distance, sunDir, transmittance, inscatter);
        }
        colour = colour * transmittance + inscatter;
    }

    // Soft knee on the PEAK channel, scaling the colour, which preserves hue (a
    // per-channel knee maps a bright orange sun to white). Colours are never
    // negative, so peak / sqrt(peak² + 1) / peak needs no zero guard.
    colour *= exposure;
    float peak = max(max(colour.r, colour.g), colour.b);
    colour *= inversesqrt(peak * peak + 1.0);

    // Haze acts on the displayed colour: it sits between the eye and the cloud.
    colour = mix(colour, hazeColor, haze);

    fragColor = vec4(colour * alpha, alpha);   // premultiplied "over"
}
