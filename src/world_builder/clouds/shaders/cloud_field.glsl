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

// GENUS_VEC4S per genus of the union march (noise.pack_genus_params: its
// field's LAYER_VEC4S rows, then its coverage row), MAX_GENERA of them, then
// the deck's. Size (MAX_GENERA + 1) * GENUS_VEC4S (a test checks it): too
// small reads out of bounds.
uniform vec4 layerParams[176];
// A genus's detail levels (the march's octave bands by distance, then its far
// one): each level's threshold offset (genusLevelOffset, genus * DETAIL_LEVELS
// + level, four a vec4: the coverage holds with fewer octaves) and octaves
// (detailOctaves, four a vec4). getField(layer) reads layer = genus *
// DETAIL_LEVELS + level; DECK_LAYER the deck's own entry, its level baked.
uniform vec4 genusLevelOffset[20];
uniform vec4 detailOctaves[2];

uniform float planetRadius;      // sea-level radius
// The weather (clouds.weather), by direction from the planet centre:
// r = the target coverage, g = the fbm shift that gives it, b = the
// convective depth, a = the anvil shelf's spread (clouds.genera; 0 in a
// two-channel map).
uniform samplerCube weatherMap;
// A/B switches for profiling (bits, 0 = normal): 1 no sun march, 2 no
// atmosphere, 4 two octaves only, 8 flat colour after the density; the
// raymarch's: 16 its steps per pixel, red over the budget, 32 no billows, 64 the
// clouds' light from the sun march, not the shadow maps, 128 where the steps go
// (red empty air, green cloud, blue the budget), 256 no lighting, 512 the
// clouds' light from the shadow maps and the sky's by the height ramp, not
// the light volume, 1024 the light volume's whole chord toward the sun, not
// the sun march's window.
uniform int   uCloudDebug;
// The far clouds' averages, by the sun's cosine (x, from deckLightCosMin to
// 1) and the weather's coverage (y): the deck's lighting
// (noise.deck_light_table; r: the sun term's shadowing, with the powder,
// g: the shadowing alone), and b: the clouds' shadow on the ground
// (noise.ground_shadow_table).
uniform sampler2D deckLight;
uniform float deckLightCosMin;
// The cloud shadow maps (clouds.shadow_map): cloudShadowDepth by where the
// sun's ray enters the slab's base, near (fine) and far (coarse), each on
// an azimuthal equidistant chart of the base's sphere (its centre's up and
// the base's radius, east and the spacing, north and the nodes per side);
// 0: none (the march per pixel).
uniform sampler2D cloudShadowNear;
uniform sampler2D cloudShadowFar;
uniform vec4  cloudNearChart[4];
uniform vec4  cloudFarChart[4];
uniform int   cloudShadowMapOn;
// The clouds' shadow's level: genus 0's layer in layerParams (-1: none), for
// its receivers (the terrain, the models: planetCloudShadow) and the light
// resources' bakes. Genus g's shadow is marched in getField(g *
// DETAIL_LEVELS + uCloudShadowLayer) (genusShadowLayer).
uniform int   uCloudShadowLayer;
// How many genera cast it (genera 0 to uCloudGenera - 1 of layerParams).
uniform int   uCloudGenera;
// The sky bands (march.sky_bands: the genera's slabs merged), lowest first:
// (base, top) above sea level, metres, then the genera in the band as a bit
// mask (bit g: genus g), 0. The light resources are baked through them, a
// gap between two crossed in no steps.
uniform vec4  uSkyBands[4];
uniform int   uSkyBandCount;
// The other genera's weather (genusWeatherAt): coverage cube A
// (weather.build_coverage_texture: a genus's coverage in one channel, row
// 13's y), turned into its fbm shift by its row of genusShift (row 13's z;
// x the coverage, weather.build_shift_texture). Their far tables
// (genusFarTable): a page each of genusDeckLight, page = genus
// (noise.build_deck_light_array). Genus 0 reads weatherMap and deckLight.
uniform samplerCube genusCoverA;
uniform sampler2D genusShift;
uniform sampler2DArray genusDeckLight;

const int LAYER_VEC4S = 13;
const int GENUS_VEC4S = 16;           // noise.GENUS_VEC4S
const int MAX_GENERA = 10;            // noise.MAX_GENERA
const int DETAIL_LEVELS = 8;          // noise.DETAIL_LEVELS
const int DECK_LAYER = MAX_GENERA * DETAIL_LEVELS;
const int MAX_SKY_BANDS = 4;          // march.MAX_SKY_BANDS
const int SHIFT_TABLE_SIZE = 256;     // weather.SHIFT_TABLE_SIZE: genusShift's width
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
    // A second order of scattering (noise.CloudOptics.octave_gain): its share
    // of the sunlight, and the share of the sunward depth it sees.
    float octaveGain;
    float octaveAttenuation;
    // A convective field's growth with depth (noise.DensityField), neutral
    // by default: its top at depth 0 as a share of the slab, its anvil (and
    // shelf), its cell octave's weight, the shelf's fbm shift
    // (noise.with_shelf).
    float minTop;
    float anvil;
    float cellWeight;
    float shelfShift;
    // The threshold's rise at depth 1 (noise.deep_profile: linear in depth
    // from topExponent and topErosion), and the cell octave's growth with
    // depth, depth ** cellPower.
    float deepExponent;
    float deepErosion;
    float cellPower;
    // The share of the clouds' depth beyond its sun window that shades it
    // (noise.GenusField.far_shadow; row 13's w): 1 for the stratus under a
    // cumulonimbus, 0 for the convective clouds.
    float farShadow;
};

Field getField(int layer) {
    int genus = layer / DETAIL_LEVELS;
    int level = layer - genus * DETAIL_LEVELS;
    int b = genus * GENUS_VEC4S;
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
    f.octaveGain      = v5.z;
    f.octaveAttenuation = v5.w;
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
    if (genus < MAX_GENERA) {
        // The level's octaves and the threshold offset that keeps the
        // coverage with them (the deck's entry has its own baked in).
        int k = genus * DETAIL_LEVELS + level;
        float offset = genusLevelOffset[k / 4][k - (k / 4) * 4];
        f.threshLo += offset;
        f.threshHi += offset;
        f.octaves = int(detailOctaves[level / 4][level - (level / 4) * 4] + 0.5);
    }
    if ((uCloudDebug & 4) != 0) {
        f.octaves = min(f.octaves, 2);
    }
    f.shellFade       = layerParams[b + 11];
    vec4 v12 = layerParams[b + 12];
    f.minTop          = v12.x;
    f.anvil           = v12.y;
    f.cellWeight      = v12.z;
    vec4 v10 = layerParams[b + 10];
    f.shelfShift      = v10.y;
    f.deepExponent = v10.z;
    f.deepErosion = v10.w;
    f.cellPower = v12.w;
    f.farShadow = layerParams[b + LAYER_VEC4S].w;
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
// The two finest octaves turned off the noise's axes (its lattice's planes
// drew slices and interference lines on a tower's face). MUST match
// OCTAVE_TURNS in noise.py (columns listed; a test reads this file).
const mat3 OCTAVE_3_TURN = mat3(0.36, 0.48, -0.8, -0.8, 0.6, 0.0, 0.48, 0.64, 0.6);
const mat3 OCTAVE_4_TURN = mat3(0.6, 0.0, 0.8, 0.64, 0.6, -0.48, -0.48, 0.8, 0.36);
// A convective field's cell octave: MUST match noise.CELL_SCALE / CELL_OFFSET.
const float CELL_SCALE = 0.25;
const vec3 CELL_OFFSET = vec3(0.13, 0.59, 0.37);

// The two finest octaves; one not evaluated counts as its mean, so dropping
// it keeps the coverage (noise.fbm's octaves).
float highOctaves(Field f, vec3 coord) {
    float third = f.octaves >= 3
        ? sampleNoise(OCTAVE_3_TURN * (coord * f.octaveScales.z) + OCTAVE_3_OFFSET)
        : f.octaveMean;
    float fourth = f.octaves >= 4
        ? sampleNoise(OCTAVE_4_TURN * (coord * f.octaveScales.w) + OCTAVE_4_OFFSET)
        : f.octaveMean;
    return third * f.octaveWeights.z + fourth * f.octaveWeights.w;
}

// The most highOctaves can add (the noise is at most 1).
float highOctavesMax(Field f) {
    return (f.octaves >= 3 ? 1.0 : f.octaveMean) * f.octaveWeights.z
         + (f.octaves >= 4 ? 1.0 : f.octaveMean) * f.octaveWeights.w;
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
    float d = clamp(depth, 0.0, 1.0);
    float lift = mix(f.topErosion, f.deepErosion, d)
               * pow(clamp(local, 0.0, 1.0), mix(f.topExponent, f.deepExponent, d));
    if (f.anvil > 0.0) {
        lift -= f.anvil * smoothstep(0.6, 1.0, depth) * smoothstep(0.7, 0.95, local);
    }
    return lift;
}

// How far into its anvil a point is, 0 to 1 (noise.anvil_weight), over a
// band ahead of the threshold's fall (noise.ANVIL_SMOOTH_BAND) at slab height
// h and convective depth; 0 for a field without one. The fine octaves (and the raymarch's billows) fade to
// their means by it, and the threshold's window widens by ANVIL_SOFTEN of
// itself: the anvil smooth and soft-edged, not the tower's cauliflower
// spread sideways.
const float ANVIL_SOFTEN = 1.0;

float anvilWeight(Field f, float h, float depth) {
    if (f.anvil <= 0.0) return 0.0;
    float local = h / topFraction(f, depth);
    return smoothstep(0.6, 1.0, depth) * smoothstep(0.5, 0.75, local);
}

// The anvil's (and its shelf's) two coarse octaves, read at ANVIL_SCALE of
// the field's frequency from ANVIL_OFFSET (noise.anvil_coarse): features four
// times wider, smooth over kilometres.
const float ANVIL_SCALE = 0.25;
const vec3 ANVIL_OFFSET = vec3(0.71, 0.29, 0.53);

float anvilLow(Field f, vec3 coord) {
    vec3 c = coord * ANVIL_SCALE + ANVIL_OFFSET;
    return sampleNoise(c * f.octaveScales.x) * f.octaveWeights.x
         + sampleNoise(c * f.octaveScales.y + OCTAVE_2_OFFSET) * f.octaveWeights.y;
}

// The column's fbm (noise.anvil_fbm), v its two coarse octaves (the cell's
// shift in them), low the anvil's (the cell's in it too), mean the fine
// octaves': its whole fbm, faded into the anvil's by its weight aw.
float anvilColumnValue(Field f, vec3 coord, float v, float low, float mean, float aw) {
    if (aw >= 1.0) return low + mean;
    float full = v + highOctaves(f, coord);
    return aw <= 0.0 ? full : mix(full, low + mean, aw);
}

// A column's raw value (its fbm less the threshold's lift, against threshLo)
// with the window widened by ANVIL_SOFTEN in the anvil, as a value pulled
// toward threshLo: smoothstep(lo, hi', x) == smoothstep(lo, hi, lo + (x - lo) / k).
float anvilSoftened(Field f, float raw, float h, float depth) {
    return f.threshLo + (raw - f.threshLo) / (1.0 + ANVIL_SOFTEN * anvilWeight(f, h, depth));
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
    // The anvil's soft edge: its window ANVIL_SOFTEN wider.
    float hi = f.threshLo + (f.threshHi - f.threshLo) * (1.0 + ANVIL_SOFTEN);
    return smoothstep(f.threshLo, hi, v + bias)
         * (1.0 - smoothstep(SHELF_BAND.z, SHELF_BAND.w, h));
}

// What a convective field's cell octave adds at depth (noise.cell_shift):
// its noise about its mean, so the coverage barely moves.
float cellShift(Field f, vec3 coord, float depth) {
    if (f.cellWeight <= 0.0) return 0.0;
    return f.cellWeight * pow(clamp(depth, 0.0, 1.0), f.cellPower)
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
    // Exact early-outs: the octaves still to come add at most their maxima,
    // and at or below the threshold the smoothstep is 0. Clear air then costs
    // one or two noise fetches, not four (plus the weather's, on a planet).
    // The anvil and its shelf read their own low-frequency octaves (anvilLow),
    // which the first two do not bound: no early-out in the anvil, and the
    // shelf, where there is one, is evaluated whole.
    float rest = highOctavesMax(f);
    float mean = f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w);
    float cell = cellShift(f, coord, weather.y);
    float aw = anvilWeight(f, h, weather.y);
    bool sheet = bias.z > -NO_CLOUD_LIFT;
    float low = aw > 0.0 || sheet ? anvilLow(f, coord) + cell : 0.0;
    float shelf = sheet ? shelfDensity(f, h, low + mean, bias.z) : 0.0;
    if (bias.x <= -NO_CLOUD_LIFT) return shelf;
    float v = sampleNoise(coord * f.octaveScales.x) * f.octaveWeights.x + cell;
    if (aw <= 0.0 && v + f.octaveWeights.y + bias.x + rest <= f.threshLo) return shelf;
    v += sampleNoise(coord * f.octaveScales.y + OCTAVE_2_OFFSET) * f.octaveWeights.y;
    if (aw <= 0.0 && v + bias.x + rest <= f.threshLo) return shelf;
    float column = smoothstep(f.threshLo, f.threshHi, anvilSoftened(
        f, anvilColumnValue(f, coord, v, low, mean, aw) + bias.x, h, weather.y));
    return max(column * bias.y, shelf);
}

// cloudDensityAt in its pieces, for a caller that moves the threshold after the
// noise is fetched (the raymarch's billows): the column's density is
// smoothstep(threshLo, threshHi, raw) * fill, the density the denser of that and
// shelf. False where the density is 0 (the same early-outs as cloudDensityAt's,
// at the weather given: a caller that lowers raw later gives the weather its
// highest shift).
bool cloudColumnAt(Field f, vec3 p, vec3 weather, out float raw, out float fill,
                   out float shelf) {
    raw = -NO_CLOUD_LIFT;
    fill = 0.0;
    shelf = 0.0;
    float h = slabHeight(f, heightOf(p));
    if (h <= 0.0 || h >= 1.0) return false;
    vec3 bias = fieldBias(f, h, weather);
    if (max(bias.x, bias.z) <= -NO_CLOUD_LIFT) return false;
    vec3 coord = p * f.noiseScale + f.noiseOrigin;
    // The early-outs and the anvil's octaves as cloudDensityAt's.
    float rest = highOctavesMax(f);
    float mean = f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w);
    float cell = cellShift(f, coord, weather.y);
    float aw = anvilWeight(f, h, weather.y);
    bool sheet = bias.z > -NO_CLOUD_LIFT;
    float low = aw > 0.0 || sheet ? anvilLow(f, coord) + cell : 0.0;
    shelf = sheet ? shelfDensity(f, h, low + mean, bias.z) : 0.0;
    if (bias.x <= -NO_CLOUD_LIFT) return shelf > 0.0;
    float v = sampleNoise(coord * f.octaveScales.x) * f.octaveWeights.x + cell;
    if (aw <= 0.0 && v + f.octaveWeights.y + bias.x + rest <= f.threshLo) return shelf > 0.0;
    v += sampleNoise(coord * f.octaveScales.y + OCTAVE_2_OFFSET) * f.octaveWeights.y;
    if (aw <= 0.0 && v + bias.x + rest <= f.threshLo) return shelf > 0.0;
    raw = anvilSoftened(f, anvilColumnValue(f, coord, v, low, mean, aw) + bias.x, h,
                        weather.y);
    fill = bias.y;
    return true;
}

float cloudDensity(Field f, vec3 p) {
    return cloudDensityAt(f, p, weatherAt(p));
}

// Optical depth along a straight stretch of the field: steps samples
// stepLength apart along dir (unit) from start, the first half a step out,
// through the two lowest octaves calibrated onto the full field
// (noise.shadow_calibration). The weather varies over kilometres: given
// once (weatherAt at start) for every step, and the cell octave (eight
// kilometres, cellShift at start) too. noise._march_density on the CPU
// (sun_optical_depth_on_sphere, cloud_shadow_on_sphere).
// The density (a sum of the column's and the shelf's, before the step's
// length) at one sample of the march through the calibrated two octaves:
// shift and mean as slabOpticalDepthCell's.
float slabSampleDensity(Field f, vec3 position, vec3 weather, float shift, float mean) {
    float h = slabHeight(f, heightOf(position));
    if (h <= 0.0 || h >= 1.0) {
        return 0.0;
    }
    vec3 bias = fieldBias(f, h, weather);
    if (max(bias.x, bias.z) <= -NO_CLOUD_LIFT) {
        return 0.0;
    }
    vec3 coord = position * f.noiseScale + f.noiseOrigin;
    // The anvil's and the shelf's octaves as cloudDensityAt's (mean and shift
    // carry the cell's here).
    float aw = anvilWeight(f, h, weather.y);
    bool sheet = bias.z > -NO_CLOUD_LIFT;
    float low = aw > 0.0 || sheet ? anvilLow(f, coord) : 0.0;
    float shelf = sheet ? shelfDensity(f, h, low + mean, bias.z) : 0.0;
    if (bias.x <= -NO_CLOUD_LIFT) {
        return shelf;
    }
    float first = sampleNoise(coord * f.octaveScales.x) * f.octaveWeights.x;
    // The same exact early-out as cloudDensity's (shadowGain > 0).
    if (aw <= 0.0
            && (first + f.octaveWeights.y) * f.shadowGain + shift + bias.x <= f.threshLo) {
        return shelf;
    }
    float pair = first + sampleNoise(coord * f.octaveScales.y + OCTAVE_2_OFFSET)
                       * f.octaveWeights.y;
    pair = mix(pair, low, aw);
    float column = smoothstep(f.threshLo, f.threshHi,
                              anvilSoftened(f, pair * f.shadowGain + shift + bias.x, h, weather.y));
    return max(column * bias.y, shelf);
}

float slabOpticalDepthCell(Field f, vec3 start, vec3 dir, float stepLength,
                           int steps, vec3 weather, float cell) {
    vec3 increment = dir * stepLength;
    vec3 position = start + increment * 0.5;
    float density = 0.0;
    // The column's two octaves are calibrated onto the full field; the
    // shelf's are its own, the rest their mean.
    float shift = f.shadowBias + cell;
    float mean = f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w) + cell;
    for (int i = 0; i < steps; i++, position += increment) {
        density += slabSampleDensity(f, position, weather, shift, mean);
    }
    return density * stepLength * f.fieldExtinction * f.shadowStrength;
}

float slabOpticalDepthAt(Field f, vec3 start, vec3 dir, float stepLength, int steps,
                         vec3 weather) {
    float cell = cellShift(f, start * f.noiseScale + f.noiseOrigin, weather.y);
    return slabOpticalDepthCell(f, start, dir, stepLength, steps, weather, cell);
}

// Optical depth toward the sun through the same field, which is what lets clouds
// shadow themselves AND each other. A depth, not a transmittance, so everything
// derived from it stays a function of world position only. The weather at p
// given (weatherAt).
float sunOpticalDepthAt(Field f, vec3 p, vec3 weather) {
    if (f.sunSteps <= 0) {
        return 0.0;
    }
    return slabOpticalDepthAt(f, p, sunDir, f.sunStepLength, f.sunSteps, weather);
}

// The chord a ray from p (planet frame) toward dir (unit) cuts through the
// shell between the spheres of radii inner and outer: how far along the ray
// it enters it (0: p inside it) and its length to where it leaves the outer
// sphere; false above the shell (nothing to cross).
bool shellChord(float inner, float outer, vec3 p, vec3 dir, out float enter,
                out float chord) {
    float r = length(p);
    enter = 0.0;
    chord = 0.0;
    if (r >= outer) {
        return false;
    }
    // Where the ray leaves each sphere: -b + sqrt(b^2 + R^2 - r^2), the
    // difference of squares factored (centimetres in float32 at 600 km).
    float b = dot(p, dir);
    float leave = -b + sqrt(max((outer - r) * (outer + r) + b * b, 0.0));
    enter = r < inner ? -b + sqrt(max((inner - r) * (inner + r) + b * b, 0.0)) : 0.0;
    chord = leave - enter;
    return chord > 0.0;
}

// The chord through the field's slab: where the ray enters it (or p, inside
// it) and its length to where it leaves the top; false above the slab.
bool cloudShadowChord(Field f, vec3 p, vec3 dir, out vec3 start, out float chord) {
    float inner = planetRadius + f.slabBase;
    float enter;
    bool crosses = shellChord(inner, inner + f.slabThickness, p, dir, enter, chord);
    start = p + dir * enter;
    return crosses;
}

// A chord that starts at *start* (on the slab's base, or inside it), cut at the
// clouds' own top there: a convective field's column reaches its local top
// only (its depth: topFraction), or the slab's when an anvil's shelf spreads
// there. Steps and levels spread over the cloud, not over the 10 km of slab
// a fair-weather cumulus fills a tenth of. The slab's own chord otherwise.
// noise._cloud_shadow_samples on the CPU.
float cloudChordToTop(Field f, vec3 start, vec3 dir, vec3 weather, float chord) {
    if (f.minTop >= 1.0) {
        return chord;
    }
    float fraction = f.anvil > 0.0 && weather.z > 0.0 ? 1.0 : topFraction(f, weather.y);
    float outer = planetRadius + f.slabBase + f.slabThickness * fraction;
    float r = length(start);
    float b = dot(start, dir);
    return min(chord, -b + sqrt(max((outer - r) * (outer + r) + b * b, 0.0)));
}

// The steps of a chord: about CLOUD_SHADOW_STEP_SCALE sunStepLength long,
// at most maxSteps (a long, low chord takes longer steps).
int cloudShadowSteps(Field f, float chord, int maxSteps) {
    float target = max(f.sunStepLength * CLOUD_SHADOW_STEP_SCALE, 1.0);
    return clamp(int(ceil(chord / target)), 1, maxSteps);
}

// The planet's clouds between a point (planet frame) and the sun toward dir
// (unit): the optical depth of their chord (cloudShadowChord), marched in
// cloudShadowSteps, the weather (the field's own map) taken where the chord
// enters the slab. 0 above the slab, or where the weather leaves no cloud.
// What a cloud's shadow on the ground is: exp2(-depth).
// noise.cloud_shadow_on_sphere on the CPU.
// The chord and the weather of a cloud shadow's march from p: false above the
// slab, or where the weather leaves no cloud (exact: where the two octaves at
// their most stay under the threshold at its lowest, an anvil's lowering it,
// its shelf's rest at their mean, the march would find nothing: clear
// weather, most of a fair sky).
bool cloudShadowMarch(Field f, samplerCube weatherCube, vec3 p, vec3 dir, int maxSteps,
                      out vec3 start, out float chord, out vec3 weather,
                      out float cell) {
    weather = vec3(0.0);
    cell = 0.0;
    if (maxSteps <= 0 || !cloudShadowChord(f, p, dir, start, chord)) {
        return false;
    }
    weather = texture(weatherCube, start).gba;
    chord = cloudChordToTop(f, start, dir, weather, chord);
    if (chord <= 0.0) {
        return false;
    }
    cell = cellShift(f, start * f.noiseScale + f.noiseOrigin, weather.y);
    float most = f.octaveWeights.x + f.octaveWeights.y;
    float column = most * f.shadowGain + f.shadowBias + cell + weather.x
                 + f.anvil * smoothstep(0.6, 1.0, weather.y);
    float shelf = f.anvil > 0.0
        ? most + f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w) + cell
              + f.shelfShift + f.anvil
        : -NO_CLOUD_LIFT;
    return max(column, shelf) > f.threshLo;
}

float cloudShadowDepthIn(Field f, samplerCube weatherCube, vec3 p, vec3 dir,
                         int maxSteps) {
    vec3 start, weather;
    float chord, cell;
    if (!cloudShadowMarch(f, weatherCube, p, dir, maxSteps, start, chord, weather, cell)) {
        return 0.0;
    }
    int steps = cloudShadowSteps(f, chord, maxSteps);
    return slabOpticalDepthCell(f, start, dir, chord / float(steps), steps, weather,
                                cell);
}

float cloudShadowDepth(Field f, vec3 p, vec3 dir, int maxSteps) {
    return cloudShadowDepthIn(f, weatherMap, p, dir, maxSteps);
}

// ── The genera's summed shadow (the light volume's and the maps' bakes) ──────
// One light volume and one pair of shadow maps hold the optical depth of all
// the genera together (clouds.light_volume, clouds.shadow_map): a depth is a
// sum, so the sum of the genera's is the depth of their summed density.

// Genus g's layer for the shadows (its far level).
int genusShadowLayer(int genus) {
    return genus * DETAIL_LEVELS + uCloudShadowLayer;
}

// A genus's coverage at p (unit or not): genus 0's the weather's, the
// others' their channel of coverage cube A.
float genusCoverageAt(int genus, vec3 p) {
    if (genus == 0) {
        return texture(weatherMap, p).r;
    }
    int channel = int(layerParams[genus * GENUS_VEC4S + LAYER_VEC4S].y + 0.5);
    return texture(genusCoverA, p)[channel];
}

// A genus's weather at p (unit or not): r its coverage, then its fbm shift,
// its convective depth, its anvil's spread (weatherMap's channels). Genus 0
// reads the weather cube; the others their coverage (genusCoverageAt), the
// shift their table gives it (the coverage filtered, then turned into a
// shift: genera.genus_weather_at), no depth, no spread.
vec4 genusWeatherAt(int genus, vec3 p) {
    if (genus == 0) {
        return texture(weatherMap, p);
    }
    vec4 row = layerParams[genus * GENUS_VEC4S + LAYER_VEC4S];
    float coverage = texture(genusCoverA, p)[int(row.y + 0.5)];
    vec2 at = vec2((coverage * float(SHIFT_TABLE_SIZE - 1) + 0.5) / float(SHIFT_TABLE_SIZE),
                   (row.z + 0.5) / float(MAX_GENERA));
    return vec4(coverage, texture(genusShift, at).r, 0.0, 0.0);
}

// A genus's far table's texel at (u, v) in [0, 1]^2 (deckLight's layout).
vec4 genusFarTable(int genus, vec2 at) {
    if (genus == 0) {
        vec2 size = vec2(textureSize(deckLight, 0));
        return texture(deckLight, (clamp(at, 0.0, 1.0) * (size - 1.0) + 0.5) / size);
    }
    vec2 size = vec2(textureSize(genusDeckLight, 0).xy);
    return texture(genusDeckLight,
                   vec3((clamp(at, 0.0, 1.0) * (size - 1.0) + 0.5) / size, float(genus)));
}

bool genusInBand(int genus, int band) {
    return ((int(uSkyBands[band].z + 0.5) >> genus) & 1) != 0;
}

// The sky bands' shell: the lowest base's radius and the highest top's.
float skyBaseRadius() {
    return planetRadius + uSkyBands[0].x;
}

float skyTopRadius() {
    return planetRadius + uSkyBands[max(uSkyBandCount - 1, 0)].y;
}

// A band's chord ahead of p toward dir (shellChord), cut, where the ray
// enters the band, at its genera's highest own top there (cloudChordToTop:
// a band of one genus is cut as its own slab's chord). 0 for none.
float skyBandChord(int band, vec3 p, vec3 dir, bool cut, out float enter) {
    float chord;
    if (!shellChord(planetRadius + uSkyBands[band].x, planetRadius + uSkyBands[band].y,
                    p, dir, enter, chord)) {
        return 0.0;
    }
    if (!cut) {
        return chord;
    }
    vec3 start = p + dir * enter;
    float top = 0.0;
    for (int g = 0; g < MAX_GENERA; ++g) {
        if (g >= uCloudGenera) {
            break;
        }
        if (genusInBand(g, band)) {
            top = max(top, cloudChordToTop(getField(genusShadowLayer(g)), start, dir,
                                           genusWeatherAt(g, start).gba, chord));
        }
    }
    return top;
}

// The genera's summed optical depth from p toward the sun (unit dir), through
// the sky bands ahead of it (skyBandChord, cut at the clouds' own tops when
// cut), each band in its own steps (cloudShadowSteps of its genera's shortest
// sun step, times stepScale; at most maxSteps), every genus of the band
// marched at the same samples by its own field and weather: the weather and
// the cell octave taken where the band's chord starts (atStart: the ground's
// march and the maps', cloudShadowMarch's early-out with them) or at every
// step (the light volume's). levels: the depth a quarter, a half and three
// quarters of the way along the bands' summed chord (cloudLevelDepth: what a
// map's texel holds; for one genus noise.cloud_shadow_on_sphere's levels).
// 0 without genera or clouds.
float skyShadowLevels(vec3 p, vec3 dir, int maxSteps, float stepScale, bool cut,
                      bool atStart, out vec3 levels) {
    levels = vec3(0.0);
    if (uCloudShadowLayer < 0 || maxSteps <= 0) {
        return 0.0;
    }
    float enters[MAX_SKY_BANDS];
    float chords[MAX_SKY_BANDS];
    float summed = 0.0;
    for (int band = 0; band < MAX_SKY_BANDS; ++band) {
        chords[band] = 0.0;
        enters[band] = 0.0;
        if (band < uSkyBandCount) {
            chords[band] = max(skyBandChord(band, p, dir, cut, enters[band]), 0.0);
            summed += chords[band];
        }
    }
    if (summed <= 0.0) {
        return 0.0;
    }
    float total = 0.0;
    float offset = 0.0;  // the summed chord before this band
    for (int band = 0; band < MAX_SKY_BANDS; ++band) {
        float chord = chords[band];
        if (chord <= 0.0) {
            continue;
        }
        vec3 start = p + dir * enters[band];
        float shortest = 1e30;
        for (int g = 0; g < MAX_GENERA; ++g) {
            if (g >= uCloudGenera) {
                break;
            }
            if (genusInBand(g, band)) {
                shortest = min(shortest, getField(genusShadowLayer(g)).sunStepLength);
            }
        }
        float target = max(shortest * stepScale * CLOUD_SHADOW_STEP_SCALE, 1.0);
        int steps = clamp(int(ceil(chord / target)), 1, maxSteps);
        float stepLength = chord / float(steps);
        vec3 increment = dir * stepLength;
        // Each level's boundary, in this band's steps from its start.
        vec3 boundary = (vec3(0.25, 0.5, 0.75) * summed - offset) / stepLength;
        for (int g = 0; g < MAX_GENERA; ++g) {
            if (g >= uCloudGenera) {
                break;
            }
            if (!genusInBand(g, band)) {
                continue;
            }
            Field f = getField(genusShadowLayer(g));
            vec3 weather = genusWeatherAt(g, start).gba;
            float cell = cellShift(f, start * f.noiseScale + f.noiseOrigin, weather.y);
            if (atStart) {
                // cloudShadowMarch's exact early-out: the weather leaves no cloud.
                float most = f.octaveWeights.x + f.octaveWeights.y;
                float column = most * f.shadowGain + f.shadowBias + cell + weather.x
                             + f.anvil * smoothstep(0.6, 1.0, weather.y);
                float shelf = f.anvil > 0.0
                    ? most + f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w) + cell
                          + f.shelfShift + f.anvil
                    : -NO_CLOUD_LIFT;
                if (max(column, shelf) <= f.threshLo) {
                    continue;
                }
            }
            float rest = f.octaveMean * (f.octaveWeights.z + f.octaveWeights.w);
            float scale = stepLength * f.fieldExtinction * f.shadowStrength;
            vec3 position = start + increment * 0.5;
            float own = 0.0;
            for (int i = 0; i < steps; i++, position += increment) {
                if (!atStart) {
                    weather = genusWeatherAt(g, position).gba;
                    cell = cellShift(f, position * f.noiseScale + f.noiseOrigin, weather.y);
                }
                float density = slabSampleDensity(f, position, weather, f.shadowBias + cell,
                                                  rest + cell) * scale;
                for (int k = 0; k < 3; k++) {
                    if (float(i) < boundary[k] && boundary[k] <= float(i + 1)) {
                        levels[k] += own + density * (boundary[k] - float(i));
                    }
                }
                own += density;
            }
            for (int k = 0; k < 3; k++) {
                if (boundary[k] > float(steps)) {
                    levels[k] += own;  // the whole band before the boundary
                }
            }
            total += own;
        }
        offset += chord;
    }
    return total;
}

// The genera's summed shadow depth from p toward dir: the ground's march
// (each band cut at the clouds' own top, the weather where it starts),
// cloudShadowDepthIn summed over the genera.
float skyShadowDepth(vec3 p, vec3 dir, int maxSteps) {
    vec3 levels;
    return skyShadowLevels(p, dir, maxSteps, 1.0, true, true, levels);
}

// Where along the bands' summed chord from start (on the bands' base) two
// distances along the ray lie (each the chord's length before it), and the
// summed chord: the map's levels' measure (skyShadowLevels).
vec2 skyChordAt(vec3 start, vec3 dir, vec2 distance, out float summed) {
    vec2 before = vec2(0.0);
    summed = 0.0;
    for (int band = 0; band < MAX_SKY_BANDS; ++band) {
        if (band >= uSkyBandCount) {
            break;
        }
        float enter;
        float chord = max(skyBandChord(band, start, dir, true, enter), 0.0);
        before += clamp(distance - enter, 0.0, chord);
        summed += chord;
    }
    return before;
}

// The clouds far along the sun's ray from a cloud at p (planet frame): the
// optical depth from where the cloud's own march stops (sunSteps steps of
// sunStepLength) out to reach metres, or to where the ray leaves the
// slab's top if sooner, in steps of about CLOUD_FAR_STEP (at most
// CLOUD_FAR_STEPS): a low sun's ray runs along the slab for kilometres,
// through the clouds sunward. The weather is taken where the segment
// starts. Slowly varying across a billboard, so taken at its centre
// (cloud_sphere.vert). noise.far_cloud_depth on the CPU.
const float CLOUD_FAR_STEP = 600.0;
const int CLOUD_FAR_STEPS = 12;
float cloudFarDepth(Field f, vec3 p, vec3 dir, float reach) {
    float near = float(max(f.sunSteps, 0)) * f.sunStepLength;
    vec3 start;
    float chord;
    if (reach <= near || !cloudShadowChord(f, p, dir, start, chord)) {
        return 0.0;
    }
    // From p, inside or under the slab: the ray leaves the top at its
    // chord's end (enter + chord from p).
    float leave = length(start - p) + chord;
    float end = min(reach, leave);
    if (end <= near) {
        return 0.0;
    }
    int steps = clamp(int(ceil((end - near) / CLOUD_FAR_STEP)), 1, CLOUD_FAR_STEPS);
    vec3 from = p + dir * near;
    return slabOpticalDepthAt(f, from, dir, (end - near) / float(steps), steps,
                              weatherAt(from));
}

// One cloud shadow map at a point of the base's sphere (unit dir): its
// texel (the depth toward the sun, then its levels: skyShadowLevels), and how much of it to keep (1 inside, 0 outside, faded over the
// last eighth of its half-width).
vec4 cloudMapTexel(sampler2D map, vec4 chart[4], vec3 dir, out float keep) {
    vec2 across = vec2(dot(dir, chart[1].xyz), dot(dir, chart[2].xyz));
    float sine = length(across);
    float angle = atan(sine, dot(dir, chart[0].xyz));
    vec2 xy = across * (sine > 1e-9 ? chart[0].w * angle / sine : chart[0].w);
    float size = chart[2].w;
    float reach = 0.5 * (size - 1.0) * chart[1].w;
    keep = 1.0 - smoothstep(0.75 * reach, 0.875 * reach, max(abs(xy.x), abs(xy.y)));
    return texture(map, (xy / chart[1].w + 0.5 * size) / size);
}

// The texel of a field's maps at a point of the base's sphere (unit dir), the
// near map's where it holds it, the far one's beyond, blended between, and how
// much the maps hold it: 1 inside the far map's solid part, fading to 0 over its
// last eighth (and 0 with the maps off).
float cloudMapsTexelWeightIn(int on, sampler2D nearMap, vec4 nearChart[4],
                             sampler2D farMap, vec4 farChart[4], vec3 dir,
                             out vec4 texel, out float keepNear) {
    texel = vec4(0.0);
    keepNear = 0.0;
    if (on == 0) {
        return 0.0;
    }
    float keepFar;
    vec4 far = cloudMapTexel(farMap, farChart, dir, keepFar);
    vec4 near = cloudMapTexel(nearMap, nearChart, dir, keepNear);
    texel = keepNear >= 1.0 ? near : mix(far, near, keepNear);
    return max(keepNear, keepFar);
}

// As cloudMapsTexelWeightIn, true where the maps hold the texel wholly (or the
// near map does at all): the ground's.
bool cloudMapsTexelIn(int on, sampler2D nearMap, vec4 nearChart[4], sampler2D farMap,
                      vec4 farChart[4], vec3 dir, out vec4 texel) {
    float keepNear;
    float weight = cloudMapsTexelWeightIn(on, nearMap, nearChart, farMap, farChart, dir,
                                          texel, keepNear);
    return on != 0 && (keepNear > 0.0 || weight >= 1.0);
}

// How many steps of the sun march a cloud keeps when it takes the clouds
// beyond them from the shadow map (cloudMapSunDepth).
const int MAP_LOCAL_STEPS = 2;

// The depth accumulated up to a point of the baked chord (u, its fraction along
// it, 0 to 1), from a map texel's levels (the total, then the depth at a
// quarter, a half and three quarters of the chord), linearly between them.
float cloudLevelDepth(vec4 levels, float u) {
    float x = 4.0 * clamp(u, 0.0, 1.0);
    int k = int(min(floor(x), 3.0));
    float t = x - float(k);
    float lower = k == 0 ? 0.0 : (k == 1 ? levels.g : (k == 2 ? levels.b : levels.a));
    float upper = k == 0 ? levels.g : (k == 1 ? levels.b : (k == 2 ? levels.a : levels.r));
    return mix(lower, upper, t);
}

// The genera's summed depth toward the sun (unit dir) from the shadow maps
// along the line through a point p of the sky bands' shell: where the line
// left the bands' base (back along it), the map's texel there (its total and
// levels: skyShadowLevels), and the depth it accumulates out to two distances
// from p (reach, and reach + window), linearly between the levels by the
// bands' summed chord (skyChordAt). Returns how much the maps hold the line
// (1: wholly, fading to 0 at the far map's edge; 0 beyond it, or for a sun too
// low for the line to have left the base).
float cloudMapLevelsAt(vec3 p, vec3 dir, float reach, float window, out float near,
                       out float far, out float total) {
    float inner = skyBaseRadius();
    float outer = skyTopRadius();
    float r = length(p);
    float b = dot(p, dir);
    // Back along the line to where it left the base's sphere: a negative
    // distance for a point in the bands under a sun above the horizon.
    float root = (inner - r) * (inner + r) + b * b;
    near = 0.0;
    far = 0.0;
    total = 0.0;
    if (cloudShadowMapOn == 0 || uSkyBandCount <= 0 || root <= 0.0 || r <= inner
            || r >= outer) {
        return 0.0;
    }
    float back = -b + sqrt(root);
    if (back >= 0.0) {
        return 0.0;
    }
    vec3 entry = p + dir * back;
    vec4 levels;
    float keepNear;
    float held = cloudMapsTexelWeightIn(cloudShadowMapOn, cloudShadowNear, cloudNearChart,
                                        cloudShadowFar, cloudFarChart, normalize(entry),
                                        levels, keepNear);
    if (held <= 0.0) {
        return 0.0;
    }
    // The chord the bake marched: the bands' from the entry, each cut at the
    // clouds' own top where the ray enters it.
    float summed;
    vec2 along = skyChordAt(entry, dir, vec2(reach - back, reach - back + window), summed);
    // The depth the chord accumulates out to each point, linearly between the
    // levels. (A line in transmittance blows up near a level that holds almost
    // all the depth.)
    near = cloudLevelDepth(levels, along.x / max(summed, 1.0));
    far = cloudLevelDepth(levels, along.y / max(summed, 1.0));
    total = levels.r;
    return held;
}

// The optical depth toward the sun (unit dir) from a cloud at p of genus
// field f, from the shadow maps: the sun march's first localSteps steps
// through the field itself (the nearby clouds' shape), then what the maps'
// chord accumulates between where they ended and where the sun march would
// have (its steps: the clouds count out to there, no farther: a tower's
// interior is dark but its sunlit face is lit, the light that reaches a
// sample coming from the clouds near it; counting the whole 10 km chord left
// every tower black), and farShadow of what lies beyond (cloudMapLevelsAt).
// Returns how much the maps hold the line: the sun march fills in the rest.
// noise.sun_depth_from_map on the CPU (one genus).
float cloudMapSunDepth(Field f, vec3 p, vec3 dir, vec3 weather, int localSteps,
                       out float depth) {
    float reach = float(localSteps) * f.sunStepLength;
    float window = float(max(f.sunSteps - localSteps, 0)) * f.sunStepLength;
    float near, far, total;
    float held = cloudMapLevelsAt(p, dir, reach, window, near, far, total);
    depth = 0.0;
    if (held <= 0.0) {
        return 0.0;
    }
    depth = slabOpticalDepthAt(f, p, dir, f.sunStepLength, localSteps, weather)
          + far - near + f.farShadow * max(total - far, 0.0);
    return held;
}

// The clouds' light volume's chart (clouds.light_volume; RENDERING_PLAN.md
// C.1): voxels of a field's slab around the camera, an atlas of slices.
//
// Horizontally the cloud shadow maps' azimuthal equidistant chart on the
// slab base's sphere (its centre's up, east, north), warped radially: a
// texel's node at t in [-1, 1]^2 (from its tile's centre) lies |t|^2 * reach
// metres from the centre along t, finer near the camera (EVE-Redux's light
// volume is a paraboloid). Vertically `slices` shells through the slab, the
// voxels at their middles; with several sky bands (uSkyBands) each band has
// its share of the slices (light_volume.SLICES each, through its own
// thickness) and a gap between two none: a 500 m stratus deck keeps the
// slices a thin slab of its own had. Tile k (a slice, 0 the lowest) sits at
// column k % columns, row k / columns of the atlas.
//
// uLightChart-style arrays: [0] the centre's up, the base's radius; [1] east,
// the reach; [2] north, the tile's texels; [3] the slab's thickness, the
// slices, the atlas's columns, the bands (0 or 1: one, the slab's).

// The height above the chart's base of a slice's middle (a fraction of the
// slices: its index plus 0.5), its band's share of the slices through it.
float lightSliceHeight(vec4 chart[4], float slice) {
    int bands = int(chart[3].w + 0.5);
    if (bands <= 1) {
        return slice / chart[3].y * chart[3].x;
    }
    float per = chart[3].y / float(bands);
    int band = clamp(int(floor(slice / per)), 0, bands - 1);
    float base = uSkyBands[band].x - uSkyBands[0].x;
    float top = uSkyBands[band].y - uSkyBands[0].x;
    return base + (slice - float(band) * per) / per * (top - base);
}

// A point's slice coordinate at height h above the chart's base (0 at the
// lowest slice's middle, slices - 1 at the highest's), within its band's
// slices: a point under a band's lowest middle or over its highest takes
// that slice (as under a lone slab's lowest), one in a gap the nearer band's.
// Read across a gap, a tower's base mixed in the slice under it, whose sun
// ray crosses the tower a kilometre aside: streaks along the bases.
float lightSliceAt(vec4 chart[4], float h) {
    int bands = int(chart[3].w + 0.5);
    if (bands <= 1) {
        return h / chart[3].x * chart[3].y - 0.5;
    }
    float per = chart[3].y / float(bands);
    float lowest = uSkyBands[0].x;
    for (int band = 0; band < MAX_SKY_BANDS; ++band) {
        float base = uSkyBands[band].x - lowest;
        float top = uSkyBands[band].y - lowest;
        float dz = (top - base) / per;
        float first = float(band) * per;
        float s = first + (h - base) / dz - 0.5;
        if (band >= bands - 1) {
            return band > 0 ? max(s, first) : s;  // over the top: the fade
        }
        // Up to the middle of the gap over it.
        if (h < 0.5 * (top + uSkyBands[band + 1].x - lowest)) {
            return clamp(s, band > 0 ? first : s, first + per - 1.0);
        }
    }
    return chart[3].y - 1.0;
}

// The planet-frame position of the voxel at a pixel of the atlas.
vec3 lightVoxelAt(vec4 chart[4], vec2 pixel, out int slice) {
    float tile = chart[2].w;
    int columns = int(chart[3].z + 0.5);
    ivec2 at = ivec2(floor(pixel / tile));
    slice = at.y * columns + at.x;
    vec2 t = (pixel - vec2(at) * tile) / tile * 2.0 - 1.0;
    float rho = length(t);
    float distance = rho * rho * chart[1].w;
    float angle = distance / chart[0].w;
    vec2 toward = rho > 0.0 ? t / rho : vec2(0.0);
    vec3 dir = cos(angle) * chart[0].xyz
             + sin(angle) * (toward.x * chart[1].xyz + toward.y * chart[2].xyz);
    return dir * (chart[0].w + lightSliceHeight(chart, float(slice) + 0.5));
}

// A point's place in the chart: its tile coordinates t ([-1, 1]^2 inside
// the volume) and its slice coordinate s (lightSliceAt).
void lightChartCoords(vec4 chart[4], vec3 p, out vec2 t, out float s) {
    float r = length(p);
    vec3 dir = p / r;
    vec2 across = vec2(dot(dir, chart[1].xyz), dot(dir, chart[2].xyz));
    float sine = length(across);
    float angle = atan(sine, dot(dir, chart[0].xyz));
    float distance = angle * chart[0].w;
    float rho = sqrt(distance / chart[1].w);
    t = sine > 1e-9 ? across / sine * rho : vec2(0.0);
    s = lightSliceAt(chart, r - chart[0].w);
}

#ifdef CLOUD_MAP_LIGHTING
// The raymarch's light volume (clouds.light_volume: the genera's summed
// depths): the atlas, its chart, on once whole.
uniform sampler2D cloudLight;
uniform vec4  cloudLightChart[4];
uniform int   cloudLightOn;
// Where in its texel and slice a sample reads the volumes, in [-0.5, 0.5]
// of each, per pixel and visit (the raymarch sets it from its blue noise),
// moved on at each sample along the ray (cloudSunDepth):
// read at their nodes, the depths' linear interpolation between texels
// hundreds of metres wide drew their grid on a tower's face, a few tens of
// pixels whatever the distance; jittered, it is noise the resolve averages.
// The same for both of a sample's reads (their difference is the light).
vec3 cloudLightJitter = vec3(0.0);
// The jitter's reach, texels and slices either way per unit of it: a full
// one. At half (the first choice) the volume's coarse texels at a distance
// showed as dark specks on far towers and silhouettes in motion, where the
// history no longer averages them (2.7 % of the cloud more than 10 levels off
// its still look, against 0.94 % now; 5.3 % with none). The grid stays gone.
const float LIGHT_JITTER = 2.0;
// Whether the sample takes the sun march's local steps before the volume
// (the raymarch clears it for samples too far for a pixel to show the
// nearby clouds' shape, RENDERING_PLAN.md C.3's alternate path: the volume
// alone, from the sample itself).
bool cloudLightLocal = true;

// A light volume's depths at p (r: toward the sun, g: up), linear between
// its texels and its slices; keep: how much the volume holds p (1 inside,
// fading over the last eighth of its tiles' half-width, 0 beyond). Above the
// slab's top 0 (no cloud above), fading to it from the highest slice's
// middle; below the lowest slice's middle, the lowest's.
vec2 cloudLightAt(sampler2D atlas, vec4 chart[4], vec3 p, out float keep) {
    vec2 t;
    float s;
    lightChartCoords(chart, p, t, s);
    float tile = chart[2].w;
    float slices = chart[3].y;
    // The jitter keeps to the point's sky band (lightSliceAt): jittered into
    // the band under it, a tower's base read the slice whose sun ray crosses
    // the tower a kilometre aside, and streaked.
    float bands = max(floor(chart[3].w + 0.5), 1.0);
    float per = slices / bands;
    float first = clamp(floor((s + 0.5) / per), 0.0, bands - 1.0) * per;
    float lowest = first > 0.0 ? first : -1e30;
    float highest = first + per < slices ? first + per - 1.0 : 1e30;
    t += cloudLightJitter.xy * (LIGHT_JITTER * 2.0 / tile);
    s = clamp(s + cloudLightJitter.z * LIGHT_JITTER, lowest, highest);
    keep = 1.0 - smoothstep(0.875, 1.0 - 1.0 / tile, max(abs(t.x), abs(t.y)));
    if (keep <= 0.0 || s >= slices - 0.5) {
        return vec2(0.0);
    }
    float top = clamp((slices - 0.5 - s) / 0.5, 0.0, 1.0);
    s = clamp(s, 0.0, slices - 1.0);
    float k0 = min(floor(s), slices - 2.0);
    float k1 = k0 + 1.0;
    float columns = chart[3].z;
    vec2 size = vec2(columns, slices / columns) * tile;
    vec2 local = clamp((t * 0.5 + 0.5) * tile, 0.5, tile - 0.5);
    vec2 a0 = vec2(mod(k0, columns), floor(k0 / columns)) * tile + local;
    vec2 a1 = vec2(mod(k1, columns), floor(k1 / columns)) * tile + local;
    vec2 lower = texture(atlas, a0 / size).rg;
    vec2 upper = texture(atlas, a1 / size).rg;
    return mix(lower, upper, s - k0) * (s > slices - 1.0 ? top : 1.0);
}

// The optical depth toward the sun (unit dir) from a point p of the slab, by
// the light volume: the sun march's first localSteps steps through the field
// itself (the nearby clouds' shape), then the volume's depth between where
// they ended and where the sun march would have (its steps: the window
// cloudMapSunDepth keeps, a tower's lit face lit), the difference of the two
// points' depths to the top (along one ray: linear), and the genus's
// farShadow of the depth beyond the window (the volume's at its end: all the
// genera's above, the stratus under a cumulonimbus in its shadow). Debug bit
// 1024: to the top, the whole chord. Returns how much the volume holds both
// points (the sun march fills in the rest).
float cloudLightSunDepth(Field f, sampler2D atlas, vec4 chart[4], vec3 p, vec3 dir,
                         vec3 weather, int localSteps, out float depth) {
    depth = 0.0;
    if (!cloudLightLocal) {
        localSteps = 0;
    }
    float reach = float(localSteps) * f.sunStepLength;
    float keepNear;
    float near = cloudLightAt(atlas, chart, p + dir * reach, keepNear).r;
    float far = 0.0;
    float keepFar = 1.0;
    if ((uCloudDebug & 1024) == 0) {
        float window = float(max(f.sunSteps - localSteps, 0)) * f.sunStepLength;
        far = cloudLightAt(atlas, chart, p + dir * (reach + window), keepFar).r;
    }
    float held = min(keepNear, keepFar);
    if (held <= 0.0) {
        return 0.0;
    }
    depth = max(near - far, 0.0) + f.farShadow * far;
    if (localSteps > 0) {
        depth += slabOpticalDepthAt(f, p, dir, f.sunStepLength, localSteps, weather);
    }
    return held;
}
#endif

// The clouds' optical depth straight up from the sample being lit, from a
// light volume (its g), and how much the volume holds the sample (0: none,
// the sky's light by the height ramp, cloud_shading.glsl's skyReach): set by
// cloudSunDepth with the sun's depth, read and cleared by cloudRadianceFrom.
float cloudSkyDepth = 0.0;
float cloudSkyHeld = 0.0;

// The optical depth toward the sun from a cloud at p of genus field f (its
// weather there given): from the light volume where the shader takes its
// light from it (CLOUD_MAP_LIGHTING: the raymarch's, whose nodes it is bound
// to) and it holds p, else from the shadow maps where they hold it, else the
// sun march's own (and, for a genus that takes it, farShadow of the maps'
// depth beyond the march).
#ifdef CLOUD_MAP_LIGHTING
float cloudSunDepth(Field f, vec3 p, vec3 weather) {
    float depth = 0.0;
    float held = 0.0;
    if (f.sunSteps > 0 && (uCloudDebug & 64) == 0) {
        int localSteps = min(f.sunSteps, MAP_LOCAL_STEPS);
        // The light volume where it is whole (debug bit 512: the maps').
        if (cloudLightOn != 0 && (uCloudDebug & 512) == 0) {
            held = cloudLightSunDepth(f, cloudLight, cloudLightChart, p, sunDir, weather,
                                      localSteps, depth);
            cloudSkyDepth = cloudLightAt(cloudLight, cloudLightChart, p, cloudSkyHeld).g;
            // Moved on for the ray's next sample (R3, low discrepancy): one
            // draw for the whole ray put all its reads off the same way, and
            // the pixel's light with them; moved on, they average along it
            // (the error over grey clouds in motion 1.55 to 1.41 levels,
            // performance.md §28). The step: Roberts' R3, (1/g, 1/g^2, 1/g^3)
            // with g = 1.2207440846 the real root of x^4 = x + 1, which
            // spreads its successive points over the cube most evenly (not
            // cloud_march.frag's per-frame R2 step: sharing a direction, the
            // offsets would repeat); the +- 0.5 keeps them in [-0.5, 0.5].
            cloudLightJitter = fract(cloudLightJitter + 0.5
                                     + vec3(0.8191725, 0.6710436, 0.5497005)) - 0.5;
        } else {
            held = cloudMapSunDepth(f, p, sunDir, weather, localSteps, depth);
        }
    }
    if (held >= 1.0) {
        return depth;
    }
    float march = sunOpticalDepthAt(f, p, weather);
    if (f.farShadow > 0.0 && f.sunSteps > 0) {
        float near, far, total;
        float window = float(f.sunSteps) * f.sunStepLength;
        float mapHeld = cloudMapLevelsAt(p, sunDir, 0.0, window, near, far, total);
        march += f.farShadow * mapHeld * max(total - far, 0.0);
    }
    return held > 0.0 ? mix(march, depth, held) : march;
}
#else
float cloudSunDepth(Field f, vec3 p, vec3 weather) {
    return sunOpticalDepthAt(f, p, weather);
}
#endif

// The planet's clouds' shadow on what is under them (planet frame p), as a
// share of the sun toward dir: the genera's summed depth (one map pair holds
// it; the product of their shades), marched (skyShadowDepth) where no map
// holds it, where it can follow the clouds' shapes; the product of their
// means for the weather there (each genus's far table's b) where it cannot,
// which costs nothing:
// * where a pixel's footprint (metres) spans several clouds (the march
//   would sparkle);
// * where the steps are longer than the clouds (a grazing sun's long chord:
//   the march would band, each height of ground stepping onto other
//   clouds).
// Faded over the second octave's cell, by the genus that fades first. 1
// where no clouds are bound (uCloudShadowLayer -1).
float planetCloudShadow(vec3 p, vec3 dir, int maxSteps, float footprint) {
    if (uCloudShadowLayer < 0 || maxSteps <= 0) {
        return 1.0;
    }
    float far = 0.0;
    bool any = false;
    for (int g = 0; g < MAX_GENERA; ++g) {
        if (g >= uCloudGenera) {
            break;
        }
        Field f = getField(genusShadowLayer(g));
        vec3 start;
        float chord;
        if (!cloudShadowChord(f, p, dir, start, chord)) {
            continue;
        }
        if (f.minTop < 1.0) {
            // A convective field's chord ends at its clouds' own top there, as
            // the march's (cloudShadowMarch) and the maps' do: its steps are then
            // short enough to follow a cumulus, not a 10 km slab's.
            chord = cloudChordToTop(f, start, dir, genusWeatherAt(g, start).gba, chord);
            if (chord <= 0.0) {
                continue;
            }
        }
        any = true;
        float cell = NOISE_FEATURE / (f.noiseScale.x * f.octaveScales.y);
        float step = chord / float(cloudShadowSteps(f, chord, maxSteps));
        far = max(far, max(smoothstep(0.25 * cell, cell, footprint),
                           smoothstep(0.5 * cell, cell, step)));
    }
    if (!any) {
        return 1.0;
    }
    float mean = 1.0;
    if (far > 0.0) {
        float cosSun = dot(normalize(p), dir);
        for (int g = 0; g < MAX_GENERA; ++g) {
            if (g >= uCloudGenera) {
                break;
            }
            vec3 start;
            float chord;
            if (cloudShadowChord(getField(genusShadowLayer(g)), p, dir, start, chord)) {
                vec2 at = vec2((cosSun - deckLightCosMin) / (1.0 - deckLightCosMin),
                               genusCoverageAt(g, p));
                mean *= genusFarTable(g, at).b;
            }
        }
    }
    if (far >= 1.0) {
        return mean;
    }
    // The maps where they hold the ray (it enters the bands' base: ground
    // under them), the near one first; the march elsewhere.
    float enter, chord;
    vec4 texel;
    float depth;
    if (length(p) < skyBaseRadius()
            && shellChord(skyBaseRadius(), skyTopRadius(), p, dir, enter, chord)
            && cloudMapsTexelIn(cloudShadowMapOn, cloudShadowNear, cloudNearChart,
                                cloudShadowFar, cloudFarChart, normalize(p + dir * enter),
                                texel)) {
        depth = texel.r;
    } else {
        depth = skyShadowDepth(p, dir, maxSteps);
    }
    return mix(exp2(-depth), mean, far);
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
