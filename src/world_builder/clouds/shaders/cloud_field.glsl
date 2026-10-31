// The cloud density field and the sun march through it, shared by the
// fragment stage (cloud.frag) and the planet billboards' vertex stage
// (cloud_sphere.vert: the layer parameters, sunDir, uCloudDebug); the cirrus
// (cirrus.vert/.frag) takes its sunlight and phase helpers. Include with
// #pragma include. noise.py mirrors the density on the CPU (fbm,
// density_on_sphere), which is what billboard placement runs on, and the sun
// march (sun_optical_depth_on_sphere), which the far deck's lighting table is
// averaged from (deck_light_table).
uniform sampler3D cloudNoise;    // one tileable octave of value noise, R8
uniform vec3  noiseOffset;       // wind advection, in METRES
uniform float noiseSize;         // volume resolution, for the coord → uvw divide
uniform vec3  sunDir;            // FROM the scene TOWARD the sun, normalised

// LAYER_VEC4S per cloud type, mirroring noise.pack_layer_params. Size is
// MAX_LAYERS * LAYER_VEC4S (a test checks it): too small reads out of bounds,
// and a mismatch with cloud_sphere.vert fails to link.
uniform vec4 layerParams[104];

uniform float planetRadius;      // sea-level radius
// The weather (clouds.weather), by direction from the planet centre:
// r = the target coverage, g = the fbm shift that gives it, b = the
// convective depth, a = the anvil shelf's spread (clouds.genera; 0 in a
// two-channel map).
uniform samplerCube weatherMap;
// A/B switches for profiling (bits, 0 = normal): 1 no sun march, 2 no
// atmosphere, 4 two octaves only, 8 flat colour after the density, 16 (the
// raymarch) its steps per pixel, red over the budget.
uniform int   uCloudDebug;

const int LAYER_VEC4S = 13;

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
    // A convective field's growth with depth (noise.DensityField), neutral
    // by default: its top at depth 0 as a share of the slab, its anvil (and
    // shelf), its cell octave's weight, the shelf's fbm shift
    // (noise.with_shelf).
    float minTop;
    float anvil;
    float cellWeight;
    float shelfShift;
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
    vec4 v12 = layerParams[b + 12];
    f.minTop          = v12.x;
    f.anvil           = v12.y;
    f.cellWeight      = v12.z;
    f.shelfShift      = layerParams[b + 10].y;
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
// A convective field's cell octave: MUST match noise.CELL_SCALE / CELL_OFFSET.
const float CELL_SCALE = 0.25;
const vec3 CELL_OFFSET = vec3(0.13, 0.59, 0.37);

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

float hgPhase(float x, float g) {
    float g2 = g * g;
    return 0.25 * ((1.0 - g2) * pow(1.0 + g2 - 2.0 * g * x, -1.5));
}

// The pre-solved weights put 1 at 90°, forwardGain at 0° and backwardGain at 180°,
// so no normalisation is needed here.
const float BACKWARD_G = -0.4;

// weights: (forward, backward, isotropic, forward anisotropy), as
// noise.phase_weights solves them. Shared by the billboards and the cirrus.
float lobePhase(vec4 weights, float x) {
    // Clamped as a last guard: a negative radiance renders as a dark hole.
    return max(0.0,
               weights.x * hgPhase(x, weights.w)
             + weights.y * hgPhase(x, BACKWARD_G)
             + weights.z * 0.25);
}

float phaseFunction(Field f, float x) {
    return lobePhase(
        vec4(f.phaseForward, f.phaseBackward, f.phaseIsotropic, f.forwardAnisotropy), x);
}

// Normalised height in the slab: 0 at the base, 1 at the ceiling.
float slabHeight(Field f, float z) {
    return (z - f.slabBase) / max(f.slabThickness, 1e-3);
}

// Height of a field position (planet frame) above the sea-level sphere.
float heightOf(vec3 p) {
    return length(p) - planetRadius;
}

// The weather's fbm shift at a planet-frame position.
float weatherShift(vec3 p) {
    return texture(weatherMap, p).g;
}

// The weather at a planet-frame position: (its fbm shift, the convective
// depth, the anvil shelf's spread), one fetch.
vec3 weatherAt(vec3 p) {
    return texture(weatherMap, p).gba;
}

// The clouds' local top, as a share of the slab (noise.top_fraction).
float topFraction(Field f, float depth) {
    return f.minTop + (1.0 - f.minTop) * clamp(depth, 0.0, 1.0);
}

// Above a convective cloud's local top: no cloud (an "infinite" lift).
const float NO_CLOUD_LIFT = 1.0e9;

// How far the threshold has risen at slab height h (noise.threshold_lift),
// following the local top at the convective depth, and falling again just
// under it in a deep cloud: the anvil (noise.ANVIL_DEPTH, ANVIL_BAND).
float thresholdLift(Field f, float h, float depth) {
    float local = h / topFraction(f, depth);
    if (f.minTop < 1.0 && local >= 1.0) return NO_CLOUD_LIFT;
    float lift = f.topErosion * pow(clamp(local, 0.0, 1.0), f.topExponent);
    if (f.anvil > 0.0) {
        lift -= f.anvil * smoothstep(0.6, 1.0, depth) * smoothstep(0.7, 0.95, local);
    }
    return lift;
}

float thresholdLift(Field f, float h) {
    return thresholdLift(f, h, 0.0);
}

// A cumulonimbus's anvil shelf (noise.SHELF_BAND and the rest): a smooth
// lens under the slab's top, spread past the tower over the weather's
// spread (the map's alpha, 0 to 1). At spread 1 it rises in over (x, y) of
// the slab, its threshold falling by SHELF_CLEAR; its underside rises with
// the spread falling, to SHELF_EDGE at 0 (gone): thick by the tower, thin
// at its rim. None under its underside. Its DENSITY fades out over (z, w):
// a flat top cut by the threshold would cut every billboard along one
// line. Smooth, not cauliflower: the two coarse octaves only (the others
// their mean).
const vec4 SHELF_BAND = vec4(0.6, 0.72, 0.76, 1.0);
const float SHELF_EDGE = 1.0;
const float SHELF_CLEAR = 0.6;

// The shelf's underside at a spread (noise.shelf_underside).
float shelfUnderside(float spread) {
    return mix(SHELF_EDGE, SHELF_BAND.x, clamp(spread, 0.0, 1.0));
}

float shelfWeight(float h, float spread) {
    float under = shelfUnderside(spread);
    return smoothstep(under, under + SHELF_BAND.y - SHELF_BAND.x, h);
}

// What the fbm is raised by against the threshold at slab height h
// (noise._slab_density): x the column's (the weather's shift less the
// threshold's lift), y its base's fill (flat bases scale DENSITY; bulging
// tops RAISE THE THRESHOLD: noise.threshold_lift), z the anvil shelf's
// (noise.shelf_lift). -NO_CLOUD_LIFT where there is none. Shared by the
// view sample and the sun march so both agree where the cloud ends.
vec3 fieldBias(Field f, float h, vec3 weather) {
    float lift = thresholdLift(f, h, weather.y);
    float column = lift >= NO_CLOUD_LIFT ? -NO_CLOUD_LIFT : weather.x - lift;
    float fill = 1.0 - exp2(-f.baseRamp * h / topFraction(f, weather.y));
    float shelf = -NO_CLOUD_LIFT;
    if (f.anvil > 0.0 && h > shelfUnderside(weather.z)) {
        shelf = f.shelfShift - thresholdLift(f, h, 1.0)
              - SHELF_CLEAR * (1.0 - shelfWeight(h, weather.z));
    }
    return vec3(column, fill, shelf);
}

// The shelf's density at slab height h for its coarse fbm v (the two
// coarse octaves, the others' mean, the cell octave's shift), its bias
// given (noise._shelf_density).
float shelfDensity(Field f, float h, float v, float bias) {
    return smoothstep(f.threshLo, f.threshHi, v + bias)
         * (1.0 - smoothstep(SHELF_BAND.z, SHELF_BAND.w, h));
}

// What a convective field's cell octave adds at depth (noise.cell_shift):
// its noise about its mean, so the coverage barely moves.
float cellShift(Field f, vec3 coord, float depth) {
    if (f.cellWeight <= 0.0) return 0.0;
    return f.cellWeight * clamp(depth, 0.0, 1.0)
         * (sampleNoise(coord * CELL_SCALE + CELL_OFFSET) - f.octaveMean);
}

// The density at p, the weather there given (weatherAt: it varies over
// kilometres, so a raymarch reads it once for many samples): the column's
// or the anvil shelf's, whichever is denser.
float cloudDensityAt(Field f, vec3 p, vec3 weather) {
    float h = slabHeight(f, heightOf(p));
    if (h <= 0.0 || h >= 1.0) return 0.0;
    vec3 bias = fieldBias(f, h, weather);
    if (max(bias.x, bias.z) <= -NO_CLOUD_LIFT) return 0.0;  // no cloud here
    vec3 coord = p * f.noiseScale + f.noiseOrigin;
    // Exact early-outs: the octaves still to come add at most their maxima
    // (the shelf, their mean), and at or below the threshold the smoothstep
    // is 0. Most of a billboard quad is clear air, which then costs one or
    // two noise fetches, not four (plus the weather's, on a planet).
    float rest = highOctavesMax(f);
    float mean = f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w);
    float reach = max(bias.x + rest, bias.z + mean);
    float v = sampleNoise(coord * f.octaveScales.x) * f.octaveWeights.x
            + cellShift(f, coord, weather.y);
    if (v + f.octaveWeights.y + reach <= f.threshLo) return 0.0;
    v += sampleNoise(coord * f.octaveScales.y + OCTAVE_2_OFFSET) * f.octaveWeights.y;
    if (v + reach <= f.threshLo) return 0.0;
    float shelf = shelfDensity(f, h, v + mean, bias.z);
    if (v + bias.x + rest <= f.threshLo) return shelf;
    float column = smoothstep(f.threshLo, f.threshHi, v + highOctaves(f, coord) + bias.x);
    return max(column * bias.y, shelf);
}

// The density at p with only the weather's shift (no convective depth, no
// shelf).
float cloudDensityShifted(Field f, vec3 p, float shift) {
    return cloudDensityAt(f, p, vec3(shift, 0.0, -NO_CLOUD_LIFT));
}

float cloudDensity(Field f, vec3 p) {
    return cloudDensityAt(f, p, weatherAt(p));
}

// Optical depth toward the sun through the same field, which is what lets clouds
// shadow themselves AND each other. A depth, not a transmittance, so everything
// derived from it stays a function of world position only. Two octaves,
// calibrated onto the full field's distribution (noise.shadow_calibration).
// The weather at p given (weatherAt).
float sunOpticalDepthAt(Field f, vec3 p, vec3 weather) {
    if (f.sunSteps <= 0) {
        return 0.0;
    }
    vec3 increment = sunDir * f.sunStepLength;
    vec3 position = p + increment * 0.5;
    float density = 0.0;
    // The weather varies over kilometres, the march spans a few hundred
    // metres: one fetch for all its steps; the cell octave (eight
    // kilometres) too.
    float cell = cellShift(f, p * f.noiseScale + f.noiseOrigin, weather.y);
    // The column's two octaves are calibrated onto the full field; the
    // shelf's are its own, the rest their mean.
    float shift = f.shadowBias + cell;
    float mean = f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w) + cell;
    for (int i = 0; i < f.sunSteps; i++, position += increment) {
        float h = slabHeight(f, heightOf(position));
        if (h <= 0.0 || h >= 1.0) {
            continue;
        }
        vec3 bias = fieldBias(f, h, weather);
        if (max(bias.x, bias.z) <= -NO_CLOUD_LIFT) {
            continue;
        }
        vec3 coord = position * f.noiseScale + f.noiseOrigin;
        float first = sampleNoise(coord * f.octaveScales.x) * f.octaveWeights.x;
        // The same exact early-out as cloudDensity's (shadowGain > 0).
        float most = first + f.octaveWeights.y;
        if (max(most * f.shadowGain + shift + bias.x, most + mean + bias.z)
                <= f.threshLo) {
            continue;
        }
        float pair = first + sampleNoise(coord * f.octaveScales.y + OCTAVE_2_OFFSET)
                           * f.octaveWeights.y;
        float column = smoothstep(f.threshLo, f.threshHi, pair * f.shadowGain + shift + bias.x);
        density += max(column * bias.y, shelfDensity(f, h, pair + mean, bias.z));
    }
    return density * f.sunStepLength * f.fieldExtinction * f.shadowStrength;
}

// With only the weather's shift (no convective depth, no shelf).
float sunOpticalDepthShifted(Field f, vec3 p, float shift) {
    return sunOpticalDepthAt(f, p, vec3(shift, 0.0, -NO_CLOUD_LIFT));
}

float sunOpticalDepth(Field f, vec3 p) {
    return sunOpticalDepthAt(f, p, weatherAt(p));
}

// ── Sunlight on a planet's clouds (through its own air) ───────────────────────
// For the shaders that also include atmosphere.glsl (atmoSunlight).

// The lit air's light on a cloud in the planet's shadow, as a share of the
// sunlight it would take (cloudTwilight).
const float TWILIGHT_GLOW = 0.25;

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
