// The cloud density field and the sun march through it, shared by the
// fragment stage (cloud.frag) and the planet billboards' vertex stage
// (cloud_sphere.vert: the layer parameters, sunDir, uCloudDebug). Include with
// #pragma include. noise.py mirrors the density on the CPU (fbm, density,
// density_on_sphere), which is what billboard placement runs on, and the sun
// march (sun_optical_depth_on_sphere), which the far deck's lighting table is
// averaged from (deck_light_table).
uniform sampler3D cloudNoise;    // one tileable octave of value noise, R8
uniform vec3  noiseOffset;       // wind advection, in METRES
uniform float noiseSize;         // volume resolution, for the coord → uvw divide
uniform vec3  sunDir;            // FROM the scene TOWARD the sun, normalised

// LAYER_VEC4S per cloud type, mirroring noise.pack_layer_params. Size is
// MAX_LAYERS * LAYER_VEC4S (a test checks it): too small reads out of bounds,
// and a mismatch with cloud.vert fails to link.
uniform vec4 layerParams[96];

uniform int   uSpherical;        // 1: planet mode (see cloud.frag)
uniform float planetRadius;      // sea-level radius (planet mode)
// The weather (clouds.weather), by direction from the planet centre:
// r = the target coverage, g = the fbm shift that gives it.
uniform samplerCube weatherMap;
// A/B switches for profiling (bits, 0 = normal): 1 no sun march, 2 no
// atmosphere, 4 two octaves only, 8 flat colour after the density.
uniform int   uCloudDebug;
// The far clouds' averages, by the sun's cosine (x, from deckLightCosMin to
// 1) and the weather's coverage (y): the deck's lighting
// (noise.deck_light_table; r: the sun term's shadowing, with the powder,
// g: the shadowing alone), and b: the clouds' shadow on the ground
// (noise.ground_shadow_table).
uniform sampler2D deckLight;
uniform float deckLightCosMin;

const int LAYER_VEC4S = 12;
// A noise lattice cell, in noise coordinates (noise.NOISE_FEATURE).
const float NOISE_FEATURE = 4.0;
// The ground shadow's march steps, in the layer's sunStepLength
// (noise.CLOUD_SHADOW_STEP_SCALE): within 0.004 of a 10 m march on average.
const float CLOUD_SHADOW_STEP_SCALE = 2.0;

// The type's field and optics. The layer index is dynamically uniform over a draw,
// so these are cached uniform reads.
struct Field {
    vec3  noiseScale;
    vec3  noiseOrigin;    // decorrelating offset plus wind advection
    vec4  octaveScales;
    vec4  octaveWeights;
    int   octaves;        // octaves evaluated; the rest count as their mean
    float octaveMean;     // the noise volume's mean
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
    f.octaves         = int(v9.z + 0.5);
    f.octaveMean      = v9.w;
    if ((uCloudDebug & 4) != 0) {
        f.octaves = min(f.octaves, 2);
    }
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

// The two finest octaves; one not evaluated counts as its mean, so dropping
// it keeps the coverage (noise.fbm's octaves).
float highOctaves(Field f, vec3 coord) {
    float third = f.octaves >= 3
        ? sampleNoise(coord * f.octaveScales.z + OCTAVE_3_OFFSET) : f.octaveMean;
    float fourth = f.octaves >= 4
        ? sampleNoise(coord * f.octaveScales.w + OCTAVE_4_OFFSET) : f.octaveMean;
    return third * f.octaveWeights.z + fourth * f.octaveWeights.w;
}

// The most highOctaves can add (the noise is at most 1).
float highOctavesMax(Field f) {
    return (f.octaves >= 3 ? 1.0 : f.octaveMean) * f.octaveWeights.z
         + (f.octaves >= 4 ? 1.0 : f.octaveMean) * f.octaveWeights.w;
}

float cloudFbm(Field f, vec3 coord) {
    return lowOctaves(f, coord) + highOctaves(f, coord);
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
    return texture(weatherMap, p).g;
}

// How far the threshold has risen at slab height h (noise.threshold_lift).
float thresholdLift(Field f, float h) {
    return f.topErosion * pow(clamp(h, 0.0, 1.0), f.topExponent);
}

// Flat bases scale DENSITY; bulging tops RAISE THE THRESHOLD (noise.threshold_lift).
// Shared by the view sample and the sun march so both agree where the cloud ends.
float slabDensity(Field f, float h, float raw) {
    float lift = thresholdLift(f, h);
    return smoothstep(f.threshLo + lift, f.threshHi + lift, raw)
         * (1.0 - exp2(-f.baseRamp * h));
}

float cloudDensity(Field f, vec3 p) {
    float h = slabHeight(f, heightOf(p));
    if (h <= 0.0 || h >= 1.0) return 0.0;
    vec3 coord = p * f.noiseScale + f.noiseOrigin;
    // Exact early-outs: the octaves still to come add at most their maxima,
    // and at or below the threshold the smoothstep is 0. Most of a billboard
    // quad is clear air, which then costs one or two noise fetches, not four
    // (plus the weather's, on a planet).
    float floor_ = f.threshLo + thresholdLift(f, h);
    float rest = highOctavesMax(f);
    float raw = sampleNoise(coord * f.octaveScales.x) * f.octaveWeights.x
              + weatherShift(p);
    if (raw + f.octaveWeights.y + rest <= floor_) return 0.0;
    raw += sampleNoise(coord * f.octaveScales.y + OCTAVE_2_OFFSET) * f.octaveWeights.y;
    if (raw + rest <= floor_) return 0.0;
    return slabDensity(f, h, raw + highOctaves(f, coord));
}

// Optical depth along a straight stretch of the field: steps samples
// stepLength apart along dir (unit) from start, the first half a step out,
// through the two lowest octaves calibrated onto the full field
// (noise.shadow_calibration). The weather varies over kilometres: one fetch,
// at start, for every step. noise.slab_optical_depth on the CPU.
float slabOpticalDepth(Field f, vec3 start, vec3 dir, float stepLength, int steps) {
    vec3 increment = dir * stepLength;
    vec3 position = start + increment * 0.5;
    float density = 0.0;
    float shift = weatherShift(start) + f.shadowBias;
    for (int i = 0; i < steps; i++, position += increment) {
        float h = slabHeight(f, heightOf(position));
        if (h <= 0.0 || h >= 1.0) {
            continue;
        }
        vec3 coord = position * f.noiseScale + f.noiseOrigin;
        float first = sampleNoise(coord * f.octaveScales.x) * f.octaveWeights.x;
        // The same exact early-out as cloudDensity's (shadowGain > 0).
        if ((first + f.octaveWeights.y) * f.shadowGain + shift
                <= f.threshLo + thresholdLift(f, h)) {
            continue;
        }
        float second = sampleNoise(coord * f.octaveScales.y + OCTAVE_2_OFFSET)
                     * f.octaveWeights.y;
        density += slabDensity(f, h, (first + second) * f.shadowGain + shift);
    }
    return density * stepLength * f.fieldExtinction * f.shadowStrength;
}

// Optical depth toward the sun through the same field, which is what lets clouds
// shadow themselves AND each other. A depth, not a transmittance, so everything
// derived from it stays a function of world position only.
float sunOpticalDepth(Field f, vec3 p) {
    if (f.sunSteps <= 0) {
        return 0.0;
    }
    return slabOpticalDepth(f, p, sunDir, f.sunStepLength, f.sunSteps);
}

// The chord a ray from p (planet frame) toward dir (unit) cuts through the
// slab's shell: where it enters it (or p, inside it) and its length to where
// it leaves the top; false above the slab (nothing to cross).
bool cloudShadowChord(Field f, vec3 p, vec3 dir, out vec3 start, out float chord) {
    float inner = planetRadius + f.slabBase;
    float outer = inner + f.slabThickness;
    float r = length(p);
    start = p;
    chord = 0.0;
    if (r >= outer) {
        return false;
    }
    // Where the ray leaves each sphere: -b + sqrt(b^2 + R^2 - r^2), the
    // difference of squares factored (centimetres in float32 at 600 km).
    float b = dot(p, dir);
    float leave = -b + sqrt(max((outer - r) * (outer + r) + b * b, 0.0));
    float enter = r < inner ? -b + sqrt(max((inner - r) * (inner + r) + b * b, 0.0))
                            : 0.0;
    chord = leave - enter;
    start = p + dir * enter;
    return chord > 0.0;
}

// The steps of a chord: about CLOUD_SHADOW_STEP_SCALE sunStepLength long,
// at most maxSteps (a long, low chord takes longer steps).
int cloudShadowSteps(Field f, float chord, int maxSteps) {
    float target = max(f.sunStepLength * CLOUD_SHADOW_STEP_SCALE, 1.0);
    return clamp(int(ceil(chord / target)), 1, maxSteps);
}

// The planet's clouds between a point (planet frame) and the sun toward dir
// (unit): the optical depth of their chord (cloudShadowChord), marched in
// cloudShadowSteps. 0 above the slab, or where the weather leaves no cloud.
// What a cloud's shadow on the ground is: exp2(-depth).
// noise.cloud_shadow_on_sphere on the CPU.
float cloudShadowDepth(Field f, vec3 p, vec3 dir, int maxSteps) {
    vec3 start;
    float chord;
    if (maxSteps <= 0 || !cloudShadowChord(f, p, dir, start, chord)) {
        return 0.0;
    }
    // Exact: where the two octaves at their most stay under the threshold,
    // the march would find nothing (clear weather: most of a fair sky).
    if ((f.octaveWeights.x + f.octaveWeights.y) * f.shadowGain + weatherShift(start)
            + f.shadowBias <= f.threshLo) {
        return 0.0;
    }
    int steps = cloudShadowSteps(f, chord, maxSteps);
    return slabOpticalDepth(f, start, dir, chord / float(steps), steps);
}

// The clouds' shadow on what is under them (planet frame p), as a share of
// the sun toward dir: the march (cloudShadowDepth) where it can follow the
// clouds' shapes, their mean for the weather there (deckLight's b) where it
// cannot, which costs nothing:
// * where a pixel's footprint (metres) spans several clouds (the march
//   would sparkle);
// * where the steps are longer than the clouds (a grazing sun's long chord:
//   the march would band, each height of ground stepping onto other
//   clouds).
// Faded over the second octave's cell.
float cloudShadow(Field f, vec3 p, vec3 dir, int maxSteps, float footprint) {
    vec3 start;
    float chord;
    if (maxSteps <= 0 || !cloudShadowChord(f, p, dir, start, chord)) {
        return 1.0;
    }
    float cell = NOISE_FEATURE / (f.noiseScale.x * f.octaveScales.y);
    float step = chord / float(cloudShadowSteps(f, chord, maxSteps));
    float far = max(smoothstep(0.25 * cell, cell, footprint),
                    smoothstep(0.5 * cell, cell, step));
    float mean = 1.0;
    if (far > 0.0) {
        vec2 size = vec2(textureSize(deckLight, 0));
        float cosSun = dot(normalize(p), dir);
        vec2 at = vec2((cosSun - deckLightCosMin) / (1.0 - deckLightCosMin),
                       texture(weatherMap, p).r);
        mean = texture(deckLight, (clamp(at, 0.0, 1.0) * (size - 1.0) + 0.5) / size).b;
    }
    if (far >= 1.0) {
        return mean;
    }
    return mix(exp2(-cloudShadowDepth(f, p, dir, maxSteps)), mean, far);
}

// ── Sunlight on a planet's clouds (planet mode, through its own air) ──────────
// For the shaders that also include atmosphere.glsl (atmoSunlight).

// Height above a cloud of the air whose light still reaches it once it is in
// the planet's shadow (cloudTwilight), metres.
const float TWILIGHT_AIR_HEIGHT = 6000.0;

// The share of the sunlight toward dir that gets through where its path dips
// below the altitude of p: there it crosses the haze, the terrain and the
// clouds around, more the lower it passes, and nothing gets through along
// the ground. 1 with the sun above p's horizon, 0 at the planet's shadow,
// smooth between. Without it a cloud's light fell only to some 15 % of red
// before the planet's shadow cut it to black: a hard line on the clouds.
float grazingLight(vec3 p, vec3 dir) {
    float r = length(p);
    float cosChi = dot(p, dir) / r;
    if (cosChi >= 0.0) {
        return 1.0;
    }
    float lowest = r * sqrt(max(0.0, 1.0 - cosChi * cosChi)) - planetRadius;
    return smoothstep(0.0, 1.0, sqrt(max(lowest, 0.0) / max(r - planetRadius, 1.0)));
}

// The sunlight on a cloud at p (planet frame) through the planet's air
// (atmosphere atmo), as a tint of the sun's colour: relative to the noon
// sun, so a cloud lit from high up stays white, and with the air's whole
// reddening, so a sunset turns it orange through the display gamma (half of
// it, a square root, left sunset clouds pale gold).
vec3 cloudSunTint(int atmo, vec3 p, vec3 sunDir) {
    vec3 through = atmoSunlight(atmo, p, sunDir)
                 / max(atmoSunlight(atmo, p, normalize(p)), vec3(1e-3));
    return min(through, vec3(1.0)) * grazingLight(p, sunDir);
}

// Twilight on a cloud: once it is in the planet's shadow, the air above it is
// still sunlit for a few degrees more and lights it, warm, fading as that air
// darkens in turn. Taken as the sunlight the air TWILIGHT_AIR_HEIGHT above
// receives and the cloud does not (tint: cloudSunTint at p), so by day it is
// nothing. A tint of the sun's colour, like cloudSunTint's.
vec3 cloudTwilight(int atmo, vec3 p, vec3 sunDir, vec3 tint) {
    vec3 above = p + normalize(p) * TWILIGHT_AIR_HEIGHT;
    return max(cloudSunTint(atmo, above, sunDir) - tint, vec3(0.0));
}
