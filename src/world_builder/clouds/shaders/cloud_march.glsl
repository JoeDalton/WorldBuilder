// A planet's low clouds, raymarched (clouds.march): the view (the reversed
// camera's whole range) in one march, at a fraction of its resolution. The
// density field, its lighting and the far field are the billboards' and the
// deck's (cloud_field.glsl, cloud_shading.glsl); the march replaces their
// overdraw: each pixel steps along its own ray through the slab, from the near
// plane to the scene (the view's depth, last frame's), and stops once the
// cloud is opaque: a
// cloud near the camera spares the march the clouds behind it.
//
// Each frame it traces one pixel of every uBlock.x x uBlock.y block of the view
// (uOffset, in turn), which cloud_resolve.frag gathers into the full
// resolution over the frames. Out: the premultiplied cloud and its
// (transmittance-weighted) distance in kilometres (a half float keeps it to
// a part in two thousand), 0 where there is none. Its geometry
// (march.slab_segments, march.step_scale) has a NumPy mirror the tests run.
// The clouds' own light from the shadow maps (cloud_field.glsl's cloudSunDepth).
#define CLOUD_MAP_LIGHTING
// Included by cloud_march.frag, and by cloud_march_recipes.frag with
// CLOUD_RECIPES defined: the recipe genera's density in the view
// (cloud_field.glsl's genusRecipeDensity), compiled only for a march that has
// one (CloudMarch picks the shader): its branch in the sample loop cost the
// convective clouds 0.15-0.2 ms at 2560 x 1440 even never taken. Likewise
// CLOUD_HIGH (cloud_march_high.frag, cloud_march_recipes_high.frag): the high
// layer's thin-layer crossings (highCrossing), only for a march with a high
// genus covered (march.march_fragment).
#pragma include "../../shaders/depth_convention.glsl"
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "cloud_field.glsl"
#pragma include "cloud_shading.glsl"

const int MAX_MARCH_LAYERS = DETAIL_LEVELS;  // noise.DETAIL_LEVELS
const float NOISE_FEATURE_UNITS = 4.0;  // noise.NOISE_FEATURE
const float KM = 1000.0;                // the distance's unit out

// The scene's depth, at the view's resolution over uDepthScale.
uniform sampler2D uSceneDepth;
uniform int   uDepthScale;       // view pixels per depth texel, per axis (1: the view's)
uniform mat4  uClipToRender;     // the view's clip space -> the render frame
uniform float uSplit;            // metres: where the near step budget ends (0: one range)
uniform vec3  uCamRender;        // camera, render frame
uniform vec3  camPos;            // camera, planet frame
uniform float exposure;
uniform int   uOwnAtmosphere;    // the planet's atmosphere (reddens sunlight), -1 none
uniform float uPixelAngle;
// The light along a ray (cloudSunAndSky) is taken anew at a cloud sample
// past max(y x where it was last taken, that + x), metres (march.py's
// CloudSettings.light_refresh_*): a far cloud behind a near one is lit by
// its own light (its sunset colour), not the near one's. Taken once a pixel,
// a thin veil nearby coloured every cloud behind it. y 0: once a pixel.
uniform vec2 uLightRefresh;       // radians per pixel of the march (a block's area's side)
uniform int   uMarchSteps;       // the step budget per pixel
uniform float uMinStep;          // metres: the shortest step, near the camera
uniform float uStepGrowth;       // a step's length per metre of distance, farther
// Layer i (its octaves) is sampled up to uLayerFar[i] metres from the camera;
// the last one beyond.
uniform float uLayerFar[MAX_MARCH_LAYERS];
uniform int   uGenusMask;        // the genera drawn, a bit each, the high genera's past MAX_GENERA (set_genus_visible)
uniform int   uLayerCount;       // a genus's detail levels (its bands, then its far one)
uniform sampler2D uBlueNoise;    // march.blue_noise, 64 x 64, the steps' jitter
uniform int   uFrame;            // this pixel's visits: moves its steps' jitter (march.visit_of)
// The edges' billows (what the billboards' sprites carry): a fine billowy
// noise, uDetailScale times the field's first octave, moving the threshold
// by up to uDetailErosion of its window, about its mean uBillowMean.
uniform float uDetailScale;
uniform float uDetailErosion;
uniform float uBillowMean;
uniform float uBillowGain;       // march.billow_stats: the blobs' spread, as the ridged noise's
uniform float uBillowRawMean;    // the noise volume's own mean
// The billows bent by a curl noise (noise.curl_noise_volume, RGB as (c + 1) / 2):
// by up to uCurlStrength of the billows' noise units, its features uCurlFrequency
// times the noise volume's tile.
uniform sampler3D uCurlNoise;
uniform float uCurlStrength;
uniform float uCurlFrequency;
uniform ivec2 uBlock;            // view pixels per march pixel, across and up
uniform ivec2 uOffset;           // the pixel traced in each block this frame
uniform vec2  uViewSize;         // the view, pixels

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragDistance;

// What a pixel of the view resolves: a feature spanning fewer view pixels
// than this fades to its mean (it would alias, a different value every
// frame's jitter and every move), over to twice as many.
const vec2 RESOLVED = vec2(2.0, 4.0);

// The view pixel alone decides what a sample resolves, not the step's length
// as well. A fade by a share of the step's length (uStepFade, by the march's
// block, 0.1-0.25) once kept far clouds from shimmering; with the resolve's
// history it only cost them their lumps (far clouds read as smooth blobs),
// and without it they shimmer less against a full-resolution reference
// (cloud_bench --jitter: error 6.15 -> 2.51, flicker 3.28 -> 2.32 at
// `start`), for 0.1-0.2 ms (performance.md §37).

// Past this view footprint (a pixel's share of the clouds' feature, as
// shading's) a sample takes its sunlight from the light volume alone, without
// the sun march's local steps (cloud_field.glsl's cloudLightLocal).
uniform float uLightLocalFootprint;

// How much of a feature of this many metres a view pixel of footprint
// metres resolves: 1 wholly, 0 not at all.
float resolved(float feature, float footprint) {
    return smoothstep(RESOLVED.x, RESOLVED.y, feature / max(footprint, 1e-3));
}

// The billows' spread (the ridged noise's they stand for: march.billow_gain;
// the value noise's, uNoiseSigma, and edgeWidening are cloud_field.glsl's).
uniform float uBillowSigma;

// The light reaching cloud at p, once a pixel: the air's (planetCloudLight:
// its reddening, twilight, the sky's fade), then what else hides the sun
// there, as the deck (cloud.frag) and every other receiver take it: other
// bodies (eclipses: the sky's light dims with them, never to nothing), the
// planet's mountains (terrainShadow) and its rings (shadows.glsl; inputs
// from the world's Shadows and ShadowMap, defaults: none).
void cloudSunAndSky(vec3 p, out vec3 sunlight, out vec3 skylight, out vec3 twilight) {
    planetCloudLight(uOwnAtmosphere, p, sunlight, skylight, twilight);
    float eclipse = eclipseVisibility(p - camPos, sunDir, uShadowSelf);
    sunlight *= eclipse * terrainShadow(normalize(p), length(p) - planetRadius)
              * ringVisibility(p - camPos, sunDir);
    twilight *= eclipse;
    skylight *= mix(ECLIPSE_SKY_FLOOR, 1.0, eclipse);
}

// The billows' noise is turned off the field's axes: value noise has its
// lattice's planes in it (flat spots and creases), which the ridged billows
// show as parallel streaks across every cloud, all one way. A rotation
// (orthonormal, columns listed), so no scale changes.
const mat3 BILLOW_TURN = mat3(0.36, 0.48, -0.8, -0.8, 0.6, 0.0, 0.48, 0.64, 0.6);

// The finer billow layer: its scale over the first's (turned about), its
// offset and its weight.
const float BILLOW_FINER_SCALE = 2.3;
const vec3 BILLOW_FINER_OFFSET = vec3(0.37, 0.81, 0.23);
const float BILLOW_FINER_WEIGHT = 0.5;

// The density, its edges carved by billows (march.DETAIL_SCALE): the
// threshold moves by the billowy noise about its mean, in units of its
// window, so a cloud's edge breaks into rounded lumps, its core stays, and
// the coverage holds on average. Clear air is found first with the most the
// billows could add: cloud alone costs the noise twice over. The two fine
// octaves and the billows fade to their means where a view pixel (footprint
// metres) cannot resolve them: the coverage holds (as an octave left out
// does), the sparkle goes.
//
// What they no longer show widens the threshold's window instead
// (edgeWidening): the expected density over the noise a pixel spans, not
// the density of its mean. Thresholding the mean drew far clouds as smooth
// sharp blobs (their outline the coarse octaves' alone); widening it leaves
// their edges soft and frayed, as the detail averaged over a pixel would.
float marchDensity(Field f, vec3 p, vec3 weather, float footprint) {
    float unit = NOISE_FEATURE_UNITS / f.noiseScale.x;
    vec2 kept = vec2(resolved(unit / f.octaveScales.z, footprint),
                     resolved(unit / f.octaveScales.w, footprint));
    float shift = weather.x
               + dot((1.0 - kept) * f.octaveWeights.zw, vec2(f.octaveMean));
    // The faded octaves' spread (independent samples of one noise)
    float octaveSpread = length((1.0 - kept) * f.octaveWeights.zw) * uNoiseSigma;
    f.octaveWeights.zw *= kept;
    float billows = resolved(unit / (f.octaveScales.x * uDetailScale), footprint);
    float fullWindow = (uCloudDebug & 32) != 0 ? 0.0  // no billows (profiling)
                     : (f.threshHi - f.threshLo) * uDetailErosion;
    // None in a cumulonimbus's anvil (cloud_field.glsl's anvilWeight): it
    // is smooth, not the tower's lumps.
    fullWindow *= 1.0 - anvilWeight(f, slabHeight(f, heightOf(p)), weather.y);
    float window = fullWindow * billows;
    if ((uCloudDebug & 4096) == 0) {  // 4096: no widening (comparisons)
        vec2 widened = edgeWidening(f.threshLo, f.threshHi,
                                    length(vec2(octaveSpread,
                                                (1.0 - billows) * fullWindow * uBillowSigma)));
        f.threshLo = widened.x;  // the early-outs' too (cloudColumnAt)
        f.threshHi = widened.y;
    }
    // The field once, at the billows' highest threshold shift: the billows only
    // lower the column's value afterwards, so their pass costs no more noise.
    float raw, fill, shelf;
    if (!cloudColumnAt(f, p, vec3(shift + window * uBillowMean, weather.yz), raw, fill,
                       shelf)) {
        return 0.0;
    }
    float most = max(smoothstep(f.threshLo, f.threshHi, raw) * fill, shelf);
    if (most <= 0.0 || window <= 0.0) {
        return most;
    }
    vec3 coord = BILLOW_TURN * ((p * f.noiseScale + f.noiseOrigin)
                                * (f.octaveScales.x * uDetailScale));
    // Bent by the curl noise: the lumps swirl instead of sitting on the value
    // noise's lattice (divergence-free, so their mean and spread hold).
    if (uCurlStrength > 0.0) {
        vec3 bend = texture(uCurlNoise, coord * (uCurlFrequency / noiseSize)).rgb;
        coord += uCurlStrength * (2.0 * bend - 1.0);
    }
    // Round lumps: two scales of the noise itself, carving where it is high.
    // (The ridged noise, 1 - |2n - 1|, carved along sheets, which show as
    // streaks across a cloud seen edge-on.) Its mean and spread are the ridged
    // noise's, so the coverage and the erosion's depth stay.
    float finer = sampleNoise(coord.yzx * BILLOW_FINER_SCALE + BILLOW_FINER_OFFSET);
    finer = mix(uBillowRawMean, finer,
                resolved(unit / (f.octaveScales.x * uDetailScale * BILLOW_FINER_SCALE),
                         footprint));
    float billow = uBillowMean + uBillowGain * (sampleNoise(coord) - uBillowRawMean
                                                + BILLOW_FINER_WEIGHT * (finer - uBillowRawMean));
    return max(smoothstep(f.threshLo, f.threshHi, raw - window * billow) * fill, shelf);
}

#ifdef CLOUD_HIGH
// ── The high layer (clouds.md, "The high layer") ───────────────────────────
// Its slab: x its base and y its thickness above sea level, metres.
uniform vec4 uHighSlab;
// Each high genus's optics (genera.HIGH_OPTICS), x the cirrus's, y the
// cirrostratus's, z the cirrocumulus's (coverage cube B's channels 1-3):
// the optical depth per metre of a full veil (density 1), and what a
// crossing's optical depth saturates toward (march.high_optical_depth).
uniform vec3 uHighExtinction;
uniform vec3 uHighCap;
// x: the cirrocumulus's grains' mean (march.cirrocumulus_grain_mean), y:
// the cirrostratus's halo and z the cirrus's (genera.HighOptics.halo), w:
// the cirrus's tufts' mean (march.cirrus_means).
uniform vec4 uHighShape;
// The cirrostratus's and the cirrocumulus's stretch axis: the upper wind
// at the genera's focus, planet frame, unit (genera.high_jet).
uniform vec3 uHighJet;
uniform vec4 uHighPhase;  // noise.phase_weights of its optics, and its forward g
uniform vec4 uHighLight;  // x: its multiple scattering, y: its shadow strength

// The longest path through the slab, in slab depths
// (march.THIN_GRAZING_PATH: past it a veil saturates toward the horizon into
// a flat plate).
const float THIN_GRAZING_PATH = 4.0;
// The footprint on the sheet: a view pixel's over the cosine of the ray's
// angle to the vertical, at least this (march.slab_footprint).
const float HIGH_MIN_INCIDENCE = 0.05;

// The cirrostratus (march.CS_*): a veil with features CS_FEATURE metres
// across the jet, CS_STRETCH times longer along it, its density moved about
// 1 by CS_VEIL_GAIN of the noise's spread; faint fibres CS_FIBRE metres
// across, CS_FIBRE_STRETCH times longer, CS_FIBRE_GAIN.
const float CS_FEATURE = 25000.0;
const float CS_STRETCH = 6.0;
const float CS_FIBRE = 1500.0;
const float CS_FIBRE_STRETCH = 12.0;
const float CS_VEIL_GAIN = 1.2;
const float CS_FIBRE_GAIN = 0.5;
// The cirrocumulus (march.CC_*): grains of Worley cells CC_CELL metres
// across, smoothstep(CC_GRAIN, 1 - F1); their fill moved by a ripple, its
// crests across the jet CC_RIPPLE metres apart (CC_RIPPLE_GAIN), bent and
// thinned by a slow noise with features CC_FILL metres across (CC_FILL_GAIN).
const float CC_CELL = 200.0;
const vec2 CC_GRAIN = vec2(0.5, 0.75);
const float CC_RIPPLE = 600.0;
const float CC_RIPPLE_GAIN = 1.0;
const float CC_FILL = 4000.0;
const float CC_FILL_GAIN = 2.0;

// p squeezed along the jet by stretch: features stretch times longer
// along it (as recipePosition does along the recipe genera's axis).
vec3 alongJet(vec3 p, float stretch) {
    return p - dot(p, uHighJet) * (1.0 - 1.0 / stretch) * uHighJet;
}

// ── The cirrus (march.CI_*; clouds.md, "The cirrus") ──
// The upper wind (genera.high_wind_cube): xyz the wind (planet frame, the
// jets' fastest 1), w its stream function (metres; its contours are the
// streamlines). A cube's six faces as the layers of a 2D array, read by
// highWindAt: a cube map's hardware filter weighs its texels in 8 bits, so
// the stream function stepped by 1/256 of a 7 km texel (about 30 m):
// stairs along the fibres up close.
uniform sampler2DArray highWind;

// The upper wind at direction d: its face's four texels around d blended
// in float (weather.sample_cube, its CPU copy: the same faces as
// sky.starfield.cube_directions, clamped at a face's edge).
vec4 highWindAt(vec3 d) {
    vec3 a = abs(d);
    int face;
    float major;
    vec3 sAxis, tAxis;
    if (a.x >= a.y && a.x >= a.z) {
        major = a.x;
        face = d.x >= 0.0 ? 0 : 1;
        sAxis = vec3(0.0, 0.0, d.x >= 0.0 ? -1.0 : 1.0);
        tAxis = vec3(0.0, -1.0, 0.0);
    } else if (a.y >= a.z) {
        major = a.y;
        face = d.y >= 0.0 ? 2 : 3;
        sAxis = vec3(1.0, 0.0, 0.0);
        tAxis = vec3(0.0, 0.0, d.y >= 0.0 ? 1.0 : -1.0);
    } else {
        major = a.z;
        face = d.z >= 0.0 ? 4 : 5;
        sAxis = vec3(d.z >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
        tAxis = vec3(0.0, -1.0, 0.0);
    }
    int size = textureSize(highWind, 0).x;
    float n = float(size);
    // Column (s) and row (t) in texels, centres at .5
    vec2 st = (vec2(dot(d, sAxis), dot(d, tAxis)) / major + 1.0) * 0.5 * n;
    vec2 xy = clamp(st - 0.5, vec2(0.0), vec2(n - 1.0));
    ivec2 i = min(ivec2(floor(xy)), ivec2(size - 2));
    vec2 f = xy - vec2(i);
    vec4 c00 = texelFetch(highWind, ivec3(i, face), 0);
    vec4 c10 = texelFetch(highWind, ivec3(i.x + 1, i.y, face), 0);
    vec4 c01 = texelFetch(highWind, ivec3(i.x, i.y + 1, face), 0);
    vec4 c11 = texelFetch(highWind, ivec3(i + 1, face), 0);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}
// What the cirrus's parts fade to where a pixel does not resolve them
// (march.cirrus_means): x the streaks' detail, y and z the patches' at
// coverage 0 and their slope, w the strands'.
uniform vec4 uHighCirrus;
// Its noise in the flow's frame (cirrusFlow): CI_*_WIDTH metres across the
// wind, CI_*_LENGTH along it. The fibres; the bundles they come in; their
// strands (the fibrous texture); the tufts (the heads); the patches the
// streaks come in.
const float CI_FIBRE_WIDTH = 180.0;
const float CI_FIBRE_LENGTH = 6000.0;
const float CI_CLUMP_WIDTH = 1200.0;
const float CI_CLUMP_LENGTH = 4500.0;
const float CI_SHEEN_WIDTH = 70.0;
const float CI_SHEEN_LENGTH = 900.0;
const float CI_TUFT_WIDTH = 400.0;
const float CI_TUFT_LENGTH = 650.0;
const float CI_PATCH_WIDTH = 9000.0;
const float CI_PATCH_LENGTH = 30000.0;
// Each one's smoothstep window on its noise; the patches' noise moved up
// by CI_PATCH_COVER x the coverage (their share of the sheet about it).
const vec2 CI_FIBRE = vec2(0.45, 0.80);
const vec2 CI_CLUMP = vec2(0.45, 0.70);
const vec2 CI_STRAND = vec2(0.48, 0.68);
const vec2 CI_TUFT = vec2(0.66, 0.90);
const vec2 CI_PATCH = vec2(0.70, 0.85);
const float CI_PATCH_COVER = 0.55;
// The cirrus's density fades in over this much coverage (where a feature's
// or the weather's coverage begins): the coverage thins the patches, it
// does not scale the density.
const float CI_COVER_EDGE = 0.1;
// The strands' least share of the fibrous texture, the fibres' and the
// tails' gains against a head's 1, the most a streak's density.
const float CI_STRAND_FLOOR = 0.15;
const float CI_FIBRE_GAIN = 1.0;
const float CI_TAIL_GAIN = 1.6;
const float CI_MAX_DENSITY = 1.3;
// The heads: CI_HEAD the window on the tuft noise (wide: a soft, fraying
// edge), moved by CI_HEAD_FRAY x the fibre noise about its mean (the
// outline frays along the fibres), CI_HEAD_FLOOR of it between the strands,
// at most CI_HEAD_GAIN.
const vec2 CI_HEAD = vec2(0.60, 1.00);
const float CI_HEAD_FRAY = 0.25;
const float CI_HEAD_FLOOR = 0.2;
const float CI_HEAD_GAIN = 1.2;
// The taps' fan toward the head: at most this many metres across the
// wind at tap 0, a fibre's share by its noise.
const float CI_FAN = 500.0;
// The tails: CI_TAPS taps CI_TAIL_STEP metres apart downwind (each the tufts
// there, the tail upwind of its head), bent across the wind by
// CI_HOOK x k^2 metres at tap k (the fall streak's shear: a hook), each
// weighted (1 - k / CI_TAPS)^CI_TAIL_FADE.
const int CI_TAPS = 6;
const float CI_TAIL_STEP = 380.0;
const float CI_HOOK = 35.0;
const float CI_TAIL_FADE = 1.3;
// The domain warp: the curl noise, features CI_CURL metres across, moving
// the flow's frame by up to CI_WARP metres (the streaks meander, their
// spacing varies); and finely, CI_CURL_FINE and CI_WARP_FINE (the fibres,
// strands and tufts wiggle, merge and split).
const float CI_CURL = 12000.0;
const float CI_WARP = 1500.0;
const float CI_CURL_FINE = 2500.0;
const float CI_WARP_FINE = 180.0;
const float CI_CURL_FEATURES = 4.0;  // the curl volume's features a tile (noise.CURL_LATTICE)
// The axis the stream function moves the noise along (any, off the
// noise's lattice: no repeat across the wind).
const vec3 CI_PSI_AXIS = vec3(0.5257, 0.6180, 0.5848);

// The noise coordinates (sampleNoise's) of the point q + delta of the flow
// whose stream function at q is psi: features width metres across the
// wind, length along it. The stream function is the across coordinate
// (constant along a streamline, so the features follow the wind as it
// turns); the position the along one (and a little of the across).
vec3 cirrusFlow(vec3 q, float psi, vec3 across, float speed, vec3 delta, float width,
                float length, vec3 offset) {
    float stream = psi - speed * dot(across, delta);  // grad psi = -speed across
    return ((q + delta) / length + CI_PSI_AXIS * (stream / width)) * NOISE_FEATURE_UNITS
         + offset * noiseSize;
}

// A view pixel's footprint on the sheet along the sheet's unit direction
// axis, metres: pixel metres across the view, stretched by the slant
// (incidence: the cosine of the ray's angle to the vertical, at least
// HIGH_MIN_INCIDENCE) along the view's own direction on the sheet, view
// (march.footprint_along).
float footprintAlong(vec3 axis, vec3 view, float pixel, float incidence) {
    float a = dot(axis, view);
    return pixel * sqrt(a * a / (incidence * incidence) + max(1.0 - a * a, 0.0));
}

// The cirrus's density at the sheet point q seen along dir, a view pixel
// pixel metres across there, its coverage cover: streaks along the upper
// wind (highWind) in patches, each fibres with strands, tufts at their
// heads with tails trailing upwind and hooking across. Detail a pixel does
// not resolve is its mean (uHighCirrus, uHighShape.w), so a far patch
// keeps its optical depth; the tails' taps are skipped there. The fibres
// fade by the footprint across the wind and along it apart (a fibre seen
// across the view at a grazing angle stays resolved: the slant stretches
// the footprint along the view, not across it), the tufts by the slant's.
float cirrusStreaks(vec3 q, vec3 dir, float pixel, float cover) {
    vec4 wind = highWindAt(q);
    vec3 up = normalize(q);
    vec3 flow = wind.xyz - dot(wind.xyz, up) * up;
    float speed = length(flow);
    vec3 along = speed > 1e-4 ? flow / speed : uHighJet;
    vec3 across = cross(up, along);
    float psi = wind.w;
    // The footprints: the slant's, across the wind's and along it
    float incidence = max(abs(dot(up, dir)), HIGH_MIN_INCIDENCE);
    vec3 view = dir - dot(dir, up) * up;
    float level = length(view);
    view = level > 1e-5 ? view / level : along;
    float footprint = pixel / incidence;
    float footAcross = footprintAlong(across, view, pixel, incidence);
    float footAlong = footprintAlong(along, view, pixel, incidence);
    // The domain warp: the frame moved by the curl noise
    vec3 curl = texture(uCurlNoise, q * (1.0 / (CI_CURL * CI_CURL_FEATURES))
                                    + vec3(0.29, 0.61, 0.17)).rgb;
    vec3 warp = CI_WARP * (2.0 * curl - 1.0);
    warp -= dot(warp, up) * up;
    // The patches
    float patchNoise = sampleNoise(cirrusFlow(q, psi, across, speed, warp, CI_PATCH_WIDTH,
                                              CI_PATCH_LENGTH, vec3(0.13, 0.71, 0.37)));
    float patchy = smoothstep(CI_PATCH.x, CI_PATCH.y, patchNoise + CI_PATCH_COVER * cover);
    float patchMean = clamp(uHighCirrus.y + uHighCirrus.z * cover, 0.0, 1.0);
    patchy = mix(patchMean, patchy, resolved(CI_PATCH_WIDTH, footAcross)
                                    * resolved(CI_PATCH_LENGTH, footAlong));
    if (patchy <= 0.0) {
        return 0.0;
    }
    float detailSeen = resolved(CI_FIBRE_WIDTH, footAcross)
                     * resolved(CI_FIBRE_LENGTH, footAlong);
    float detail = uHighCirrus.x;
    if (detailSeen > 0.0) {
        // And finely, the fibres and the tufts' edges (they wiggle, merge
        // and split: no parallel lines)
        vec3 fine = texture(uCurlNoise, q * (1.0 / (CI_CURL_FINE * CI_CURL_FEATURES))
                                        + vec3(0.71, 0.13, 0.53)).rgb;
        vec3 fineWarp = warp + CI_WARP_FINE * (2.0 * fine - 1.0);
        fineWarp -= dot(fineWarp, up) * up;
        float fibreNoise = sampleNoise(cirrusFlow(q, psi, across, speed, fineWarp,
                                                  CI_FIBRE_WIDTH, CI_FIBRE_LENGTH,
                                                  vec3(0.47, 0.19, 0.83)));
        float fibre = smoothstep(CI_FIBRE.x, CI_FIBRE.y, fibreNoise);
        // The fibres in bundles that taper out along the wind
        fibre *= smoothstep(CI_CLUMP.x, CI_CLUMP.y,
            sampleNoise(cirrusFlow(q, psi, across, speed, warp, CI_CLUMP_WIDTH,
                                   CI_CLUMP_LENGTH, vec3(0.23, 0.89, 0.61))));
        // The strands: thin lines along the wind, the fibrous texture of
        // the tails, the fibres and the heads (their mean where a pixel does
        // not resolve them)
        float strandNoise = sampleNoise(cirrusFlow(q, psi, across, speed, fineWarp,
                                                   CI_SHEEN_WIDTH, CI_SHEEN_LENGTH,
                                                   vec3(0.91, 0.37, 0.53)));
        float strandSeen = resolved(CI_SHEEN_WIDTH, footAcross)
                         * resolved(CI_SHEEN_LENGTH, footAlong);
        float strands = mix(uHighCirrus.w, smoothstep(CI_STRAND.x, CI_STRAND.y, strandNoise),
                            strandSeen);
        float fibrous = CI_STRAND_FLOOR + (1.0 - CI_STRAND_FLOOR) * strands;
        // The head and the tail. The taps fan out toward the head: each
        // fibre shifts them across the wind by its own share of CI_FAN
        // (its noise about its mean, signed; the fibres' 180 m, not the
        // strands' 70 m, keeps neighbouring pixels' taps together in the
        // texture cache: +0.4 ms overhead with the strands'), most at the
        // head (tap 0), none at the tail's end, so the fibres meet the head at different
        // places and splay out there like a broom ("explode" at the comma),
        // and run together downstream. The head: a wide window on the tuft
        // noise (no outline), frayed along the fibres, striated by the
        // strands, at most CI_HEAD_GAIN (translucent). The tail: the tufts
        // downwind averaged along the hooked line (a line integral).
        float tuftSeen = resolved(CI_TUFT_WIDTH, footprint);
        float tufts = uHighShape.w;
        if (tuftSeen > 0.0) {
            float fan = CI_FAN * 2.0 * (fibreNoise - uBillowRawMean);
            float head = 0.0;
            float tail = 0.0;
            float weights = 0.0;
            for (int k = 0; k < CI_TAPS; ++k) {
                float fk = float(k);
                float spread = fan * (1.0 - fk / float(CI_TAPS));
                vec3 delta = fineWarp + (CI_TAIL_STEP * fk) * along
                           + (CI_HOOK * fk * fk + spread) * across;
                float n = sampleNoise(cirrusFlow(q, psi, across, speed, delta,
                                                 CI_TUFT_WIDTH, CI_TUFT_LENGTH,
                                                 vec3(0.67, 0.43, 0.09)));
                if (k == 0) {
                    head = CI_HEAD_GAIN
                         * smoothstep(CI_HEAD.x, CI_HEAD.y,
                                      n + CI_HEAD_FRAY * (fibreNoise - uBillowRawMean))
                         * (CI_HEAD_FLOOR + (1.0 - CI_HEAD_FLOOR) * strands);
                } else {
                    float w = pow(1.0 - fk / float(CI_TAPS), CI_TAIL_FADE);
                    tail += w * smoothstep(CI_TUFT.x, CI_TUFT.y, n);
                    weights += w;
                }
            }
            tufts = mix(tufts, head + CI_TAIL_GAIN * tail / weights * fibrous, tuftSeen);
        }
        float streaks = tufts + CI_FIBRE_GAIN * fibre * fibrous;
        detail = mix(detail, min(streaks, CI_MAX_DENSITY), detailSeen);
    }
    return patchy * detail;
}

// The cirrostratus's veil at the sheet point q, footprint metres there
// (march.slab_footprint): nearly uniform, about 1, slow swells and faint
// fibres along the jet, each faded out where the footprint does not
// resolve it.
float cirrostratusVeil(vec3 q, float footprint) {
    float swell = sampleNoise(alongJet(q, CS_STRETCH) * (NOISE_FEATURE_UNITS / CS_FEATURE)
                              + vec3(0.71, 0.23, 0.47) * noiseSize) - uBillowRawMean;
    float fibre = sampleNoise(alongJet(q, CS_FIBRE_STRETCH) * (NOISE_FEATURE_UNITS / CS_FIBRE)
                              + vec3(0.17, 0.83, 0.29) * noiseSize) - uBillowRawMean;
    return max(0.0, 1.0 + CS_VEIL_GAIN * swell * resolved(CS_FEATURE, footprint))
         * max(0.0, 1.0 + CS_FIBRE_GAIN * fibre * resolved(CS_FIBRE, footprint));
}

// The cirrocumulus's grains at the sheet point q, footprint metres there:
// round flakes, sky between (no shading: a thin sheet's light), in rows
// across the jet; a grain a pixel does not resolve is the grains' mean
// (march.cirrocumulus_grains), so a far patch keeps its optical depth.
float cirrocumulusGrains(vec3 q, float footprint) {
    float grain = smoothstep(CC_GRAIN.x, CC_GRAIN.y,
                             1.0 - texture(cloudCells, q / (CC_CELL * WORLEY_CELLS)).r);
    grain = mix(uHighShape.x, grain, resolved(CC_CELL, footprint));
    float slow = sampleNoise(q * (NOISE_FEATURE_UNITS / CC_FILL)
                             + vec3(0.43, 0.11, 0.67) * noiseSize) - uBillowRawMean;
    // Narrow crests, empty troughs: (1 + sin)^2 / 4 scaled to a mean of 1
    float crest = 0.5 + 0.5 * sin(6.2831853 * (dot(q, uHighJet) / CC_RIPPLE + 2.0 * slow));
    float ripple = mix(1.0, crest * crest * (8.0 / 3.0),
                       CC_RIPPLE_GAIN * resolved(CC_RIPPLE, footprint));
    float fill = max(0.0, 1.0 + CC_FILL_GAIN * slow * resolved(CC_FILL, footprint));
    return grain * ripple * fill;
}
#endif

// The fewest samples a stretch of the slab takes (march.SEGMENT_SAMPLES).
const float SEGMENT_SAMPLES = 8.0;
// The march stops at this transmittance, and counts the cloud opaque.
const float OPAQUE = 0.04;
// The steps do not lengthen as the light is spent (they did, by 1.5 times the
// share lost): a step's length then depended on what the samples before it
// found, so the steps' boundaries were no fixed comb the jitter shifts evenly
// over the visits, and what the visits averaged kept the contours of where
// the comb met a cloud's surface: bands across a tower's face, strongest
// behind a veil (its light spent, every step after longer). Without, the
// settled bands are gone, for 0.36 ms more over a stratus deck and nothing
// in convective cloud at 2560 x 1440 (performance.md §29).
// The light a ray lets through once composited (march.opaque_residual): its
// transmittance eased to 0 over [OPAQUE, 2 OPAQUE], value and slope
// continuous. Snapped from OPAQUE to 0, a thin sheet seen at a grazing
// angle (its rays cross the cutoff gradually across the view) took a step
// of 0.04 in alpha: a contour across the altostratus.
float opaqueResidual(float transmittance) {
    if (transmittance <= OPAQUE) {
        return 0.0;
    }
    float x = transmittance - OPAQUE;
    return transmittance >= 2.0 * OPAQUE ? transmittance : 3.0 * x - x * x / OPAQUE;
}
// A cloudy column's alpha seen from afar: the deck's.
const float FAR_ALPHA = 0.92;

// Far clouds' edges (genusDensity's far path): where a pixel no longer
// resolves the clouds, a coverage c drawn as a slab of alpha 0.92 c was a
// translucent wash wherever c was partial, the ground (its coastlines)
// showing through, the weather's texels (39 km on Earth) soft brushes.
// Instead the coverage is moved by a noise of two octaves of the noise
// volume at these features, metres (a standard score times MACRO_AMOUNT of
// coverage), then stepped at a half: opaque masses with sharp, ragged
// edges at 14-40 km and clear gaps, their insides varying in thickness with
// the weather (macroThickness). Stepped only near the edges, translucent
// fringes and rings showed beside them. Each octave fades in as the view pixel resolves it (resolved()).
// Thresholding the noise at the cover's own probit (every partial area the
// same random spots, the weather's masses lost) was tried first.
const vec2 MACRO_FEATURES = vec2(40000.0, 14000.0);
const vec2 MACRO_WEIGHTS = vec2(0.7, 0.3);
// How far the noise moves the coverage (per standard score), and the
// step's half-width, in coverage.
const float MACRO_AMOUNT = 0.18;
const float MACRO_EDGE = 0.04;
// A mass's optical thickness follows the weather's coverage under it
// (macroThickness): this share of the full alpha where it barely covers,
// the full alpha where it covers wholly. A band's streaks and a swirl's arms
// show as thinner and thicker cloud, not as holes.
const float MACRO_THIN = 0.6;
// Off the field's own noise coordinates (decorrelated from its octaves).
const vec3 MACRO_OFFSET = vec3(0.53, 0.11, 0.87);

// A far mass's alpha relative to FAR_ALPHA (MACRO_THIN): thinner where
// the weather's coverage c barely passes a half, whole where it is 1.
float macroThickness(float c) {
    return (uCloudDebug & 16384) != 0 ? 1.0 : mix(MACRO_THIN, 1.0, smoothstep(0.5, 1.0, c));
}

// The share of a far pixel's area that holds cloud: coverage c with sharp,
// ragged edges (MACRO_*), c itself where the pixel resolves none of the
// noise. mean: the noise volume's (the field's octaveMean; uNoiseSigma its
// spread). Debug bit 16384: c (comparison).
float macroCover(float c, vec3 p, float footprint, float mean) {
    if ((uCloudDebug & 16384) != 0 || c <= 0.0 || c >= 1.0) {
        return c;
    }
    float k0 = MACRO_WEIGHTS.x * resolved(MACRO_FEATURES.x, footprint);
    float k1 = MACRO_WEIGHTS.y * resolved(MACRO_FEATURES.y, footprint);
    float kept = length(vec2(k0, k1));
    if (kept < 0.02) {
        return c;
    }
    vec3 q = p * (NOISE_FEATURE / MACRO_FEATURES.x) + MACRO_OFFSET;
    vec3 r = OCTAVE_3_TURN * (p * (NOISE_FEATURE / MACRO_FEATURES.y)) + OCTAVE_2_OFFSET;
    float z = (k0 * (sampleNoise(q) - mean) + k1 * (sampleNoise(r) - mean))
            / (max(uNoiseSigma, 1e-4) * kept);
    float sharp = smoothstep(0.5 - MACRO_EDGE, 0.5 + MACRO_EDGE, c + MACRO_AMOUNT * z);
    return mix(c, sharp, kept / length(MACRO_WEIGHTS));
}

// The weather is read once this far along the ray, metres (uWeatherStretch):
// the fair-weather shift varies over kilometres, but a convective cloud's depth
// (its top) changes over a fraction of a tower's width: read once a kilometre
// it shows as bands across the tower, one per stretch.
uniform float uWeatherStretch;

// Every band two segments. The high slab's two crossings are not among
// them (march.slab_segments of the high slab): a ray's height along it is convex and the
// slab is above every band, so its crossing on the way down comes before
// every band's and its crossing on the way up after. Kept out of the
// sorted segments and out of their loop, the high variant costs the low
// clouds nothing: sorted among them (two more segments, the crossing
// inlined in the segments' loop) it cost 0.1-0.2 ms more at 2560 x 1440,
// the crossings' own work about nothing (performance.md §37, Task 23).
const int MAX_SEGMENTS = 2 * MAX_SKY_BANDS;

#ifdef CLOUD_HIGH
vec2 sphereInterval(vec3 c, vec3 dir, float r);

// Where the crossing seg meets the slab's middle, the sphere of radius r
// (march.sheet_crossing): every high genus's pattern is read there, once,
// as a thin sheet's (jittered samples would wander a slant ray over many
// grains or fibres, and the visits blur them). Where it does not meet it
// (a camera inside the slab, looking away from its middle), -1: the ray
// crosses no sheet, and the crossing draws nothing. (It read the patterns
// at the crossing's middle, another height for every ray: a second copy
// of the cirrus, offset, seen from inside the slab.)
float sheetCrossing(vec2 seg, vec3 dir, float r) {
    vec2 s = sphereInterval(camPos, dir, r);
    if (s.x < s.y) {
        if (s.x >= seg.x && s.x <= seg.y) {
            return s.x;
        }
        if (s.y >= seg.x && s.y <= seg.y) {
            return s.y;
        }
    }
    return -1.0;
}

// One crossing seg of the high slab by the ray from camPos along dir, composited
// over what the ray holds so far (its crossing on the way down before every
// band's segment, on the way up after them: an anvil in front hides it, it
// veils a tower behind it). The coverage (cube B's channels 1-3, all at
// once) and every genus's pattern at the sheet point (sheetCrossing: a thin
// sheet's, the same at every visit). The genera's optical depths over the
// crossing's path add (march.high_optical_depth), lit as a thin slab
// (thinSlabRadiance, with the cirrus's and the cirrostratus's halo by their
// shares) at the sheet point. No sun march, no steps.
void highCrossing(vec2 seg, vec3 dir, inout float transmittance,
                  inout vec3 light, inout float weightedDistance) {
    float len = seg.y - seg.x;
    float ts = sheetCrossing(seg, dir, planetRadius + uHighSlab.x + 0.5 * uHighSlab.y);
    if (ts < 0.0) {
        return;  // no sheet on this crossing
    }
    vec3 q = camPos + dir * ts;
    vec3 cover = texture(genusCoverB, q).gba;
    // Hidden (CloudMarch.set_genus_visible), their bits
    // past the low genera's (march.HIGH_MASK_SHIFT)
    cover *= vec3((ivec3(uGenusMask) >> ivec3(MAX_GENERA, MAX_GENERA + 1, MAX_GENERA + 2)) & 1);
    if (cover.r + cover.g + cover.b <= 0.0) {
        return;
    }
    // A view pixel's footprint per metre of distance (genusDensity's,
    // without the step's: the samples are no comb of steps)
    float perMetre = uPixelAngle / sqrt(float(uBlock.x * uBlock.y));
    vec3 up = normalize(q);
    vec3 density = vec3(0.0);
    if (cover.r > 0.0) {
        // The coverage thins the patches (cirrusStreaks): its density only
        // fades in at the coverage's edge
        density.r = smoothstep(0.0, CI_COVER_EDGE, cover.r)
                  * cirrusStreaks(q, dir, ts * perMetre, cover.r);
    }
    // The sheet's footprint: stretched along the view by the slant
    float footprint = ts * perMetre / max(abs(dot(up, dir)), HIGH_MIN_INCIDENCE);
    if (cover.g > 0.0) {
        density.g = cover.g * cirrostratusVeil(q, footprint);
    }
    if (cover.b > 0.0) {
        density.b = cover.b * cirrocumulusGrains(q, footprint);
    }
    vec3 parts = uHighExtinction * density * min(len, THIN_GRAZING_PATH * uHighSlab.y);
    float total = parts.r + parts.g + parts.b;
    if (total <= 0.0) {
        return;
    }
    float cap = dot(parts, uHighCap) / total;
    float depth = cap * (1.0 - exp(-total / cap));
    float alpha = 1.0 - exp2(-depth);
    vec3 sunlight, skylight, twilight;
    cloudSunAndSky(q, sunlight, skylight, twilight);
    vec3 radiance = (uCloudDebug & 256) != 0 ? vec3(0.8)  // no lighting (profiling)
        : thinSlabRadiance(dot(uHighExtinction, density) * uHighSlab.y, up, dir,
                           cap, uHighPhase, uHighLight,
                           (uHighShape.z * parts.r + uHighShape.y * parts.g) / total,
                           sunlight, skylight, twilight);
    float weight = transmittance * alpha;
    light += weight * radiance;
    weightedDistance += weight * ts;
    transmittance *= 1.0 - alpha;
}
#endif

vec3 unproject(vec3 ndc) {
    vec4 p = uClipToRender * vec4(ndc, 1.0);
    return p.xyz / p.w;
}

// The stretch of the ray c + t dir inside the sphere of radius r around the
// planet's centre: (t0, t1), empty when t0 >= t1. Through the ray's closest
// point to the centre: float32 keeps a planet's radius to centimetres, its
// square only to tens of metres.
vec2 sphereInterval(vec3 c, vec3 dir, float r) {
    float b = dot(c, dir);
    float q = length(c - dir * b);
    if (q >= r) {
        return vec2(1.0, -1.0);
    }
    float h = sqrt((r - q) * (r + q));
    return vec2(-b - h, -b + h);
}

// About how many steps cover seg, uMinStep-long up to the knee and growing
// with the distance past it.
float stepsOver(vec2 seg, float dtMin, float growth) {
    if (seg.y <= seg.x) {
        return 0.0;
    }
    float knee = dtMin / growth;
    float n = 1.0;
    if (seg.x < knee) {
        n += (min(seg.y, knee) - seg.x) / dtMin;
    }
    if (seg.y > knee) {
        n += log(seg.y / max(seg.x, knee)) / growth;
    }
    return n;
}

int layerAt(float t) {
    for (int i = 0; i < MAX_MARCH_LAYERS - 1; ++i) {
        if (i >= uLayerCount - 1 || t < uLayerFar[i]) {
            return i;
        }
    }
    return uLayerCount - 1;
}

// The genera of a shared band whose weather the march keeps apart (main).
const int BAND_WEATHER = 3;

float interleavedGradient(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

// Genus g's density at a sample t metres along the ray at p (lf its field
// at the sample's level, weather its genusWeatherAt there): none where its
// coverage is 0; far away,
// where a pixel spans more than the clouds' features, their expected
// coverage (the deck's) as a homogeneous slab whose vertical crossing takes
// the deck's alpha. Out: how much of its own self-shadowing the sample keeps
// (shading: 0 its averages, cloudRadiance) and its far coverage.
float genusDensity(int g, Field lf, vec3 p, vec4 weather, float t,
                   out float shading, out float farCoverage) {
    if (((uGenusMask >> g) & 1) == 0) {
        // Hidden (CloudMarch.set_genus_visible: the showcase's toggles)
        shading = 1.0;
        farCoverage = 0.0;
        return 0.0;
    }
    float feature = NOISE_FEATURE_UNITS / (lf.noiseScale.x * lf.octaveScales.x);
#ifdef CLOUD_RECIPES
    if ((lf.recipe & RECIPE_WORLEY) != 0) {
        feature = min(feature, lf.cellSize);  // a recipe genus's cells
    }
#endif
    float footprint = t * uPixelAngle / feature;
    float detail = 1.0 - smoothstep(0.15, 0.6, footprint);
    // Its own self-shadowing and powder kept as far as its shapes (the far
    // table's averages only where the shapes go too): out there the sun's
    // depth is the light volume's alone (no local steps, below), smooth
    // enough not to alias, and it keeps a far cloud's shaded side and
    // creases. It used to fade from a quarter of that (0.04-0.2: about
    // 110 km at 2560 x 1440 for the cumulus).
    shading = 1.0 - smoothstep(0.15, 0.6, footprint);
    // The light volume alone (no local sun steps) where a pixel spans this
    // much of the clouds' feature (cloud_field.glsl).
    cloudLightLocal = footprint < uLightLocalFootprint;
    float density = 0.0;
    if (detail > 0.0 && weather.r > 0.0) {
        float pixel = t * uPixelAngle / sqrt(float(uBlock.x * uBlock.y));
        // A recipe genus its recipe (cloud_field.glsl), the others the field
        // with its billows.
#ifdef CLOUD_RECIPES
        density = lf.recipe != 0 ? genusRecipeDensity(lf, p, weather, pixel)
                                 : marchDensity(lf, p, weather.gba, pixel);
#else
        density = marchDensity(lf, p, weather.gba, pixel);
#endif
    }
    farCoverage = 0.0;
    if (shading < 1.0) {
        farCoverage = genusCoverageAt(g, p);
        // Patches, not a wash (macroCover); the far table's light keeps the
        // pixel's mean coverage
        float view = t * uPixelAngle / sqrt(float(uBlock.x * uBlock.y));
        float cover = macroCover(farCoverage, p, view, lf.octaveMean)
                    * macroThickness(farCoverage);
        float mean = -log2(1.0 - FAR_ALPHA * cover) / (lf.extinction * lf.slabThickness);
        density = mix(mean, density, detail);
    } else if (lf.recipe != 0) {
        // A recipe genus's sun term reads its far table at every distance
        // (cloudRadianceFrom); close up, the weather's coverage at hand
        farCoverage = weather.r;
    }
    return density;
}

// Whether the light taken litAt metres along the ray is stale at t
// (uLightRefresh): t past growth x litAt and past litAt + the least distance.
bool lightStale(float t, float litAt) {
    return uLightRefresh.y > 0.0
        && t > max(uLightRefresh.y * litAt, litAt + uLightRefresh.x);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy) * uBlock + uOffset;
    if (any(greaterThanEqual(vec2(pixel), uViewSize))) {
        discard;
    }
    // A recipe genus's far edges widen as the field's (edgeWidening)
    recipeWiden = (uCloudDebug & 4096) == 0;
    vec2 ndc = (vec2(pixel) + 0.5) / uViewSize * 2.0 - 1.0;
    vec3 nearPoint = unproject(vec3(ndc, depthNearZ()));
    vec3 farPoint = unproject(vec3(ndc, depthFarNdc()));
    vec3 dir = normalize(farPoint - nearPoint);
    if (uDepthMode != DEPTH_STANDARD) {
        // Reversed and infinite: the far end is beyond any band.
        farPoint = nearPoint + dir * 1e10;
    }
    float tStart = max(dot(nearPoint - uCamRender, dir), 0.0);
    // The scene: the view's surface where it drew one (its depth, last
    // frame's, one texel a view pixel: uDepthScale 1), else the far end.
    ivec2 depthPixel = pixel / uDepthScale;
    float sceneDepth = texelFetch(uSceneDepth, depthPixel, 0).r;
    bool drawn = depthDrawn(sceneDepth);
    vec3 end = drawn ? unproject(vec3(ndc, depthNdc(sceneDepth))) : farPoint;
    float tEnd = length(end - uCamRender);
    // The scene this sample stopped at, km from the near plane (0: none, the
    // sky), in the distance's g: cloud_resolve.frag weighs the samples by it
    // against each pixel's own (its sceneDistance, the same measure).
    float sceneKm = drawn ? length(end - nearPoint) / KM : 0.0;

    // Each sky band (cloud_field.glsl's uSkyBands: the genera's slabs merged)
    // between two spheres, less the planet: up to two segments each (a ray
    // from above, or skimming the band's underside, leaves and re-enters),
    // the bands' together sorted along the ray (they do not overlap). Gaps
    // between bands take no steps.
    vec2 ground = sphereInterval(camPos, dir, planetRadius);
    if (ground.x < ground.y && ground.y > 0.0) {
        tEnd = min(tEnd, max(ground.x, 0.0));
    }
    vec2 segments[MAX_SEGMENTS];
    int bandOf[MAX_SEGMENTS];
    int count = 0;
    for (int i = 0; i < MAX_SKY_BANDS; ++i) {
        if (i >= uSkyBandCount) {
            break;
        }
        float rBase = planetRadius + uSkyBands[i].x;
        vec2 top = sphereInterval(camPos, dir, planetRadius + uSkyBands[i].y);
        vec2 base = sphereInterval(camPos, dir, rBase);
        vec2 first = top;
        vec2 second = vec2(1.0, -1.0);
        if (base.x < base.y) {
            first = vec2(top.x, base.x);
            second = vec2(base.y, top.y);
        }
        first = vec2(max(first.x, tStart), min(first.y, tEnd));
        second = vec2(max(second.x, tStart), min(second.y, tEnd));
        if (first.y > first.x) {
            segments[count] = first;
            bandOf[count] = i;
            ++count;
        }
        if (second.y > second.x) {
            segments[count] = second;
            bandOf[count] = i;
            ++count;
        }
    }
#ifdef CLOUD_HIGH
    // The high layer's slab, above every band: its crossing on the way down
    // (before every band's segment) and on the way up (after every one;
    // march.slab_segments of the high slab). A ray that never goes below its base crosses
    // no band: its one crossing is the first.
    vec2 highDown, highUp;
    {
        float rBase = planetRadius + uHighSlab.x;
        vec2 top = sphereInterval(camPos, dir, rBase + uHighSlab.y);
        vec2 base = sphereInterval(camPos, dir, rBase);
        highDown = top;
        highUp = vec2(1.0, -1.0);
        if (base.x < base.y) {
            highDown = vec2(top.x, base.x);
            highUp = vec2(base.y, top.y);
        }
        highDown = vec2(max(highDown.x, tStart), min(highDown.y, tEnd));
        highUp = vec2(max(highUp.x, tStart), min(highUp.y, tEnd));
    }
    bool anyHigh = highDown.y > highDown.x || highUp.y > highUp.x;
#else
    bool anyHigh = false;
#endif
    if (count == 0 && !anyHigh) {
        // No cloud on this ray, but its scene for the resolve.
        fragColor = vec4(0.0);
        fragDistance = vec4(0.0, sceneKm, 0.0, 1.0);
        return;
    }
    // Sorted by where they start (at most eight).
    for (int i = 1; i < MAX_SEGMENTS; ++i) {
        for (int k = i; k > 0 && k < count; --k) {
            if (segments[k].x < segments[k - 1].x) {
                vec2 swapSegment = segments[k];
                segments[k] = segments[k - 1];
                segments[k - 1] = swapSegment;
                int swapBand = bandOf[k];
                bandOf[k] = bandOf[k - 1];
                bandOf[k - 1] = swapBand;
            }
        }
    }

    // Steps uMinStep long near the camera, a share of the distance farther,
    // all lengthened alike when the ray would need more than the budget. The
    // near range (to uSplit) and the far range each have the budget (as when
    // the split cameras' ranges had a march each): one budget for the whole
    // ray would stretch the near steps for far stretches the march seldom
    // reaches. Since the one reversed camera (#57) uSplit is 0: the far
    // budget covers the whole ray, as measured then; a near budget again is
    // a look and cost change of its own.
    float dtMin = uMinStep;
    float growth = uStepGrowth;
    vec2 needed = vec2(0.0);  // near, far
    for (int i = 0; i < MAX_SEGMENTS; ++i) {
        if (i < count) {
            vec2 seg = segments[i];
            needed.x += stepsOver(vec2(seg.x, min(seg.y, uSplit)), dtMin, growth);
            needed.y += stepsOver(vec2(max(seg.x, uSplit), seg.y), dtMin, growth);
        }
    }
    vec2 stretch = max(needed / float(uMarchSteps), vec2(1.0));
    // Where in its step each sample lies: no banding, a fine noise instead,
    // moved on by the golden ratio each visit of this pixel (not each frame:
    // traced once a block's cycle, it would skip through the sequence), so
    // its visits stratify the step and the resolve averages them away.
    // The noise is a blue noise at the march's pixel (the gradient noise at a
    // view pixel's stride of uBlock aliased its diagonals into a lattice of
    // stripes), shifted by which pixel of the block this frame traces (the
    // block's pixels differ: R2 over the offset) and moved on by the visit, so
    // a pixel's visits stratify the step. (A space-time blue noise, a slice a
    // visit or a frame as EVE-Redux reads it, measured the same: performance.md
    // §24.)
    float jitter = fract(texelFetch(uBlueNoise, ivec2(gl_FragCoord.xy) & 63, 0).r
                         + dot(vec2(uOffset), vec2(0.7548776662, 0.5698402909))
                         + float(uFrame) * 0.618034);
    ivec2 noiseAt = ivec2(gl_FragCoord.xy);
    // Where the steps' boundaries fall: each segment's first step is this
    // share of its length, so the boundaries after it move too, by a
    // different amount each pixel and visit. The jitter above moves a sample
    // within its step, not the steps: neighbouring pixels shared their
    // boundaries (from the slab's entry, growing smoothly with distance),
    // and a long step's light (not linear in its density) carries a bias
    // that depends on where a cloud's surface falls between two boundaries:
    // a sawtooth of bands up a tower's face seen level, where the steps are
    // longest. Moved, the bias is noise the resolve averages.
    float boundary = max(fract(texelFetch(uBlueNoise, (noiseAt + ivec2(7, 23)) & 63, 0).r
                               + float(uFrame) * 0.4142136), 0.05);

    float transmittance = 1.0;
    vec3 light = vec3(0.0);
    float weightedDistance = 0.0;
    vec3 sunlight = vec3(0.0);
    vec3 skylight = vec3(0.0);
    vec3 twilight = vec3(0.0);
    bool lit = false;
    float litAt = 0.0;  // where the light was last taken (uLightRefresh)
    int current = -1;
    Field lf = getField(0);  // replaced at the first sample (current -1)
    int steps = 0;
    int emptySteps = 0;  // steps whose density was 0 (uCloudDebug 128)
    // A genus's weather (its coverage, its shift, the convective depth, the
    // anvil's spread: genusWeatherAt), read once a uWeatherStretch of ray: it
    // varies over kilometres. Kept for the last genus read: with genera the
    // stretch is shorter than a step (WEATHER_STRETCH_GENERA), so a sample
    // reads each of its genera's anyway.
    vec4 weather = vec4(0.0);
#ifdef CLOUD_HIGH
    if (highDown.y > highDown.x) {
        highCrossing(highDown, dir, transmittance, light, weightedDistance);
    }
#endif
    for (int s = 0; s < MAX_SEGMENTS; ++s) {
        if (s >= count) {
            break;
        }
        vec2 seg = segments[s];
        int genera = int(uSkyBands[bandOf[s]].z + 0.5);  // the band's, a bit each
        float weatherUntil = -1.0;
        float a = seg.x;
        float least = (seg.y - seg.x) / SEGMENT_SAMPLES;  // the fewest a segment takes
        bool first = true;  // its first step: shortened by boundary
        // A band of one genus (every band today) takes its samples alone, in
        // a loop of its own: the same light through the genus loop below, or
        // a test of the band per sample, cost 0.1-0.6 ms more between the
        // towers (cloud_bench --genera, 2560 x 1440).
        bool alone = (genera & (genera - 1)) == 0;
        int only = 0;  // its genus
        while (only < MAX_GENERA - 1 && ((genera >> only) & 1) == 0) {
            ++only;
        }
        while (alone && a < seg.y && steps < 2 * uMarchSteps + 8 && transmittance > OPAQUE) {
            float k = a < uSplit ? stretch.x : stretch.y;
            float dt = min(min(k * max(dtMin, a * growth), least), seg.y - a);
            if (first) {
                dt *= boundary;
                first = false;
            }
            float t = a + dt * jitter;
            steps++;
            int level = layerAt(t);
            vec3 p = camPos + dir * t;
            float shading, farCoverage;
            int layer = only * DETAIL_LEVELS + level;
            if (layer != current) {
                lf = getField(layer);
                current = layer;
            }
            if (t > weatherUntil) {
                weather = genusWeatherAt(only, p);
                weatherUntil = t + uWeatherStretch;
            }
            float density = genusDensity(only, lf, p, weather, t, shading, farCoverage);
            a += dt;
            if (density <= 0.0) {
                ++emptySteps;
                continue;
            }
            float alpha = 1.0 - exp2(-lf.extinction * density * dt);
            if (!lit || lightStale(t, litAt)) {
                // The light (the air's, eclipses, mountains, rings), at the
                // first cloud and again as the ray goes on (lightStale).
                cloudSunAndSky(p, sunlight, skylight, twilight);
                lit = true;
                litAt = t;
            }
            vec3 radiance = (uCloudDebug & 256) != 0 ? vec3(0.8)  // no lighting (profiling)
                : cloudRadianceAt(only, lf, p, density, -dir, shading, farCoverage, sunlight,
                                  skylight, twilight, weather.gba);
            float weight = transmittance * alpha;
            light += weight * radiance;
            weightedDistance += weight * t;
            transmittance *= 1.0 - alpha;
        }
        // Genera sharing a band: every genus whose slab holds the sample adds
        // its extinction (sigma), its light weighted by it. Each of the
        // band's first BAND_WEATHER genera keeps its own weather a
        // uWeatherStretch (one cache for all re-read it at every sample, the
        // genera alternating), and a genus without coverage there is skipped
        // before its field is read: a deep slab sharing a band with others
        // (a nimbostratus feature's, 0.6-5 km), mostly empty, cost 3 ms at
        // the ground with no cloud anywhere (synoptic weather plan, Task 7).
        vec4 bandWeather[BAND_WEATHER];
        float bandUntil[BAND_WEATHER];
        for (int w = 0; w < BAND_WEATHER; ++w) {
            bandWeather[w] = vec4(0.0);
            bandUntil[w] = -1.0;
        }
        while (!alone && a < seg.y && steps < 2 * uMarchSteps + 8 && transmittance > OPAQUE) {
            float k = a < uSplit ? stretch.x : stretch.y;
            float dt = min(min(k * max(dtMin, a * growth), least), seg.y - a);
            if (first) {
                dt *= boundary;
                first = false;
            }
            float t = a + dt * jitter;
            steps++;
            int level = layerAt(t);
            vec3 p = camPos + dir * t;
            float shading, farCoverage;
            float height = heightOf(p);
            float sigma = 0.0;
            vec3 radiance = vec3(0.0);
            int slot = -1;  // the genus's place among the band's
            for (int g = 0; g < MAX_GENERA; ++g) {
                if (g >= uCloudGenera) {
                    break;
                }
                if (((genera >> g) & 1) == 0) {
                    continue;
                }
                ++slot;
                vec2 slab = layerParams[g * GENUS_VEC4S + 4].xy;  // its slab (getField's)
                if (height <= slab.x || height >= slab.x + slab.y) {
                    continue;
                }
                if (slot < BAND_WEATHER) {
                    if (t > bandUntil[slot]) {
                        bandWeather[slot] = genusWeatherAt(g, p);
                        bandUntil[slot] = t + uWeatherStretch;
                    }
                    weather = bandWeather[slot];
                } else {
                    weather = genusWeatherAt(g, p);
                }
                if (weather.r <= 0.0) {
                    continue;  // no cover here: nothing to read
                }
                int layer = g * DETAIL_LEVELS + level;
                if (layer != current) {
                    lf = getField(layer);
                    current = layer;
                }
                float density = genusDensity(g, lf, p, weather, t, shading, farCoverage);
                if (density <= 0.0) {
                    continue;
                }
                if (!lit || lightStale(t, litAt)) {
                    cloudSunAndSky(p, sunlight, skylight, twilight);
                    lit = true;
                    litAt = t;
                }
                float own = lf.extinction * density;
                sigma += own;
                radiance += own * ((uCloudDebug & 256) != 0
                    ? vec3(0.8)
                    : cloudRadianceAt(g, lf, p, density, -dir, shading, farCoverage,
                                      sunlight, skylight, twilight, weather.gba));
            }
            a += dt;
            if (sigma <= 0.0) {
                ++emptySteps;
                continue;
            }
            float alpha = 1.0 - exp2(-sigma * dt);
            float weight = transmittance * alpha;
            light += weight * (radiance / sigma);
            weightedDistance += weight * t;
            transmittance *= 1.0 - alpha;
        }
    }
#ifdef CLOUD_HIGH
    if (highUp.y > highUp.x && transmittance > OPAQUE) {
        highCrossing(highUp, dir, transmittance, light, weightedDistance);
    }
#endif
    if ((uCloudDebug & 128) != 0) {
        // Where the steps go: red the share in empty air, green in cloud, blue
        // the steps over the budget.
        float taken = max(float(steps), 1.0);
        fragColor = vec4(float(emptySteps) / taken, float(steps - emptySteps) / taken,
                         float(steps) / float(uMarchSteps), 1.0);
        fragDistance = vec4(0.001, sceneKm, 0.0, 1.0);
        return;
    }
    if ((uCloudDebug & 16) != 0) {
        // The march's work: red its steps over the budget.
        fragColor = vec4(float(steps) / float(uMarchSteps), 0.0, 0.0, 1.0);
        fragDistance = vec4(0.001, sceneKm, 0.0, 1.0);
        return;
    }
    float alpha = 1.0 - opaqueResidual(transmittance);
    if (alpha < 1.0 / 512.0) {
        fragColor = vec4(0.0);
        fragDistance = vec4(0.0, sceneKm, 0.0, 1.0);
        return;
    }
    float covered = 1.0 - transmittance;
    vec3 colour = light / covered;  // the cloud's mean radiance
    if ((uCloudDebug & 8) != 0) {
        colour = vec3(0.8);
    } else {
        // No air in front of the cloud, unlike the billboards': a marched
        // cloud keeps its own colour at any distance (atmosphere.md, "Clouds
        // at the horizon"). The sun's reddening is in its light already.
        colour = displayColour(colour * exposure);
    }
    fragColor = vec4(colour * alpha, alpha);  // premultiplied "over"
    fragDistance = vec4(weightedDistance / covered / KM, sceneKm, 0.0, 1.0);
}
