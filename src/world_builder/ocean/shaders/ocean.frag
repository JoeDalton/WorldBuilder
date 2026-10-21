#version 140

// Optional: seen through the atmospheres when they are bound (uAtmoCount > 0;
// the default, set by Ocean, is none, and the distance haze below applies).
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
// The seabed (colour by depth, foam, the swell's amplitude) and the colour
// by depth shared with the terrain's sea surface.
#pragma include "seabed.glsl"
#pragma include "water.glsl"

uniform float     iTime;
uniform vec3      iCameraPos;
uniform vec3      uCamForward;     // the view's axis, base frame
uniform float     uPixelAngle;     // a pixel's angle at the screen's centre, radians
uniform vec3      iWaterColor;
uniform vec3      iWaterLight;     // daylight 1; dims it by night (Ocean.set_light)
uniform sampler2D iReflectionTex;
uniform float     iRippleStrength;
uniform vec2      uReflUVScale;
uniform mat4      uReflMVP;       // reflection view-projection (used per-fragment)
uniform vec2      iWindDir;       // normalised wind direction in XY, set from Python
uniform float     iWindStrength;  // 0=fully random waves, 1=all waves follow wind
uniform int       iIterationsNormal;  // wave-detail iterations (quality knob, set from Python)
// Per-iteration wave directions, precomputed on the CPU.  They depend only on
// the iteration index and the wind (not on the pixel), so computing them here
// would repeat the same sin/cos/normalize/mix for every pixel every iteration.
#define MAX_WAVE_ITER 64
uniform vec2      iWaveDirs[MAX_WAVE_ITER];
// Screen-space filtering of the waves: an octave fades out as its phase
// changes by uAAStart..uAAEnd radians from one pixel to the next (2*pi / the
// pixels per wavelength; Nyquist is pi). What it loses is kept as slope
// variance, shaded by the rough fresnel below, so the sea's average colour
// survives its waves going sub-pixel.
uniform float     uAAStart;
uniform float     uAAEnd;
// Sum of the octave weights over iIterationsNormal octaves: every pixel
// normalises by it, however many octaves it resolves, so dropping the fine
// ones does not inflate the others.
uniform float     uWaveWeightSum;
// Slope variance of octaves i.. (iIterationsNormal - 1), normalised as the
// normal is: what a pixel that stops at octave i leaves unresolved.
uniform float     uTailVar[MAX_WAVE_ITER];
// Fresnel averaged over unresolved slopes, by (N.V, slope spread / its max);
// 0 = plain Schlick on the resolved normal (the cheaper quality presets).
uniform int       uRoughFresnel;
uniform sampler2D uFresnelLUT;
// Reflection blur by that spread, through the reflection's mipmaps: the
// ripple offset the spread stands for, in texels (0 = sharp, no mipmaps).
uniform float     uReflBlur;
// Swell spatial frequency and scroll speed — set from Python so the vertex and
// fragment shaders share one source of truth (the wavelength is load-bearing:
// the geometric-swell mesh must sample it without aliasing).
uniform float     iSwellScale;
uniform float     iSwellDrift;
// The swell's deep-water height: the shading tilt follows the geometry's
// (SWELL_STRENGTH is the tilt of SWELL_REF_AMPLITUDE).
uniform float     uSwellAmplitude;
// Frequency-modulation depth (radians). The low-frequency swell field is added
// to the small-wave phase, so the wave field drifts over the swell's scale and
// the exact tiling of the dominant (lowest-frequency) octave is broken up. 0
// disables FM.
uniform float     uFmDepth;
uniform int       uDebugMode;  // 0 normal, 1 N, 2 reflUV, 3 fresnel, 4 worldgrid,
                               // 5 reflUV clamp, 6 raw reflection, 7 pre-tonemap
                               // clipping, 8 FM phase field, 9 octaves resolved,
                               // 10 unresolved slope spread, 11 octaves evaluated,
                               // 12 depth
uniform int       uWaveOff;    // debug: 1 = skip small-wave normal, swell only
uniform float     uExposure;   // pre-tonemap exposure
// Aerial perspective: distant water loses contrast to the air in front of it and
// tends to the horizon's own colour. Must agree with what the cloud field fades
// to, or the sea and the sky above it disagree at the horizon.
uniform mat3      uBaseToWorld;   // base frame -> world axes, for the atmospheres
uniform vec3      uAtmoSunDir;    // toward the sun, world axes
uniform vec3      uHazeColor;
uniform float     uHazeDistance;  // metres over which the water reaches it fully
// Colour by depth: the shallow tint, its e-folding depth (water.glsl).
uniform int       uDepthColourOn;
uniform vec3      uShallowColor;
uniform float     uShallowDepth;
// Shoreline foam: a band under uFoamDepth metres of water, solid at the
// waterline, then lines along the depth contours (uFoamSpacing metres of
// depth apart) running shoreward at uFoamSpeed radians a second, broken up by
// noise (uFoamNoiseOn).
uniform int       uFoamOn;
uniform int       uFoamNoiseOn;
uniform vec3      uFoamColor;     // linear, in daylight (iWaterLight dims it)
uniform float     uFoamDepth;
uniform float     uFoamSpacing;
uniform float     uFoamSpeed;
uniform float     uFoamStrength;

in  vec3 vWorldPos;
out vec4 fragColor;

#define DRAG_MULT         0.38
#define WAVE_HEIGHT       10.0
#define WAVE_SCALE        0.5
#define SWELL_STRENGTH    0.5    // how strongly the swell tilts the normal (frag-only)
#define SWELL_REF_AMPLITUDE 20.0 // the swell height SWELL_STRENGTH is the tilt of
#define WARP_SCALE        0.05
#define WARP_STRENGTH     1.0
#define NOISE_SCALE       0.01
#define NOISE_STRENGTH    0.75
// Means of fields that are averaged once sub-pixel (measured over the fields):
// |swell gradient|^2 (independent of iSwellScale), and fbmNoise.
#define SWELL_SLOPE_MS    0.1458
#define NOISE_MEAN        0.532
// Foam: its break-up noise's scale (1 / metres), and the lines' mean coverage
// (smoothstep(0.55, 0.95, 0.5 + 0.5 sin) over a period), what they fade to
// once sub-pixel.
#define FOAM_NOISE_SCALE  0.08
#define FOAM_LINE_MEAN    0.329
// The seabed texels over which foam fades out (metres): a band a metre or two
// deep cannot be placed by coarser ones, and would show as a wide grey sheet.
#define FOAM_TEXEL_FULL   12.0
#define FOAM_TEXEL_GONE   48.0

// Angle-based detail falloff (sine of the view-ray elevation angle).
// Below GRAZE_HI the surface fades to a flat near-mirror; full detail above it.
// Keyed on angle (not distance) so the look is consistent across altitude.
#define GRAZE_LO          0.0
#define GRAZE_HI          0.12
// Mean square of one octave's slope, (exp(sin x - 1) cos x)^2 over a period
// (ocean.py's C_WAVE: the CPU's tail variances use it too).
#define C_WAVE            0.1076346
// Slope spread at the fresnel table's last row (ocean.py's FRESNEL_LUT_SLOPE_MAX).
#define FRESNEL_SLOPE_MAX 2.0
#define FRESNEL_LUT_SIZE  32.0

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------
float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i + vec2(0,0)), hash(i + vec2(1,0)), u.x),
        mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
        u.y);
}

float fbmNoise(vec2 p) {
    return smoothNoise(p) * 0.6
         + smoothNoise(p * 2.1 + vec2(5.3, 1.7)) * 0.4;
}

// Large-scale swell field: the product of two value-noise layers drifting in
// different directions (and at different scales/speeds).  The product makes the
// pattern interfere and continuously change shape over time instead of rigidly
// scrolling, so the swell never looks like one fixed shape sliding past.
float swellField(vec2 w) {
    vec2 d1 = iWindDir;
    vec2 d2 = vec2(-iWindDir.y, iWindDir.x);  // perpendicular to the wind
    vec2 a = w * iSwellScale        + iTime * iSwellDrift        * d1;
    vec2 b = w * iSwellScale * 1.7  + iTime * iSwellDrift * 0.6  * d2;
    return fbmNoise(a) * fbmNoise(b);
}

// ---------------------------------------------------------------------------
// Waves
// ---------------------------------------------------------------------------
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
    float x    = dot(direction, position) * frequency + timeshift;
    float wave = exp(sin(x) - 1.0);
    float dx   = wave * cos(x);
    return vec2(wave, -dx);
}

float getwaves(vec2 position, int iterations) {
    float wavePhaseShift = length(position) * 0.1;
    float frequency      = 1.0;
    float timeMultiplier = 2.0;
    float weight         = 1.0;
    float sumOfValues    = 0.0;
    float sumOfWeights   = 0.0;
    for (int i = 0; i < iterations && i < MAX_WAVE_ITER; i++) {
        vec2 p = iWaveDirs[i];  // precomputed (wind-biased) direction

        vec2 res = wavedx(position, p, frequency,
                          iTime * timeMultiplier + wavePhaseShift);
        position      += p * res.y * weight * DRAG_MULT;
        sumOfValues   += res.x * weight;
        sumOfWeights  += weight;
        weight         = mix(weight, 0.0, 0.2);
        frequency     *= 1.18;
        timeMultiplier *= 1.07;
    }
    return sumOfValues / sumOfWeights;
}

vec2 domainWarp(vec2 pos, int iterations, float strength) {
    float wx = getwaves(pos * WARP_SCALE,                   iterations);
    float wy = getwaves(pos * WARP_SCALE + vec2(3.7, 1.3), iterations);
    return vec2(wx, wy) * strength;
}

// Analytic wave-height gradient (∂H/∂x, ∂H/∂y) accumulated in a SINGLE pass.
// Each wave term is wave = exp(sin(x) - 1) with x = freq * dot(p, pos) + t, so
// its slope is d(wave)/dpos = wave * cos(x) * freq * p — known in closed form.
// One pass instead of a 3x getwaves finite-difference normal, and no height
// accumulation (the surface is a flat plane; only the slope matters).
// Approximation: the loop advects `position` to sharpen crests, and we keep
// that advection so the field matches getwaves(), but we don't differentiate
// through it — the standard, visually-faithful real-time simplification.
//
// fmPhase is the swell-driven frequency-modulation phase (constant per fragment,
// varying slowly across the surface). Added to every octave's phase, it shifts
// each octave's crests — most in world units for the longest-wavelength octave —
// so the dominant tiling drifts with the swell. It is treated as locally
// constant for the gradient (same simplification already applied to
// wavePhaseShift and the advection); the swell is low-frequency, so its spatial
// derivative is negligible next to freq*p and omitting it keeps the normal stable.
//
// Filtering: dPdx / dPdy are the pixel's footprint in the waves' domain. Each
// octave fades by how far its phase moves across the pixel ALONG ITS OWN
// direction (a wave running across the view stays resolved longer than one
// running along it), scaling both its slope and its advection, and what it
// loses goes to residualVar. Evaluation stops once no direction can resolve
// an octave any more; the rest of the octaves' variance comes from uTailVar.
float gResolved;   // octaves resolved, summed by their fade (debug mode 9)
float gEvaluated;  // octaves evaluated: the loop's cost (debug mode 11)

// One octave, faded by a: its slope into sumGrad, its advection into position.
void waveOctave(vec2 p, float frequency, float weight, float phase, float a,
                inout vec2 position, inout vec2 sumGrad) {
    float x     = dot(p, position) * frequency + phase;
    float wave  = exp(sin(x) - 1.0);
    float dwave = wave * cos(x);  // d(wave)/dx
    sumGrad  += dwave * frequency * weight * a * p;
    position += p * (-dwave) * weight * DRAG_MULT * a;  // same advection as getwaves
}

vec2 waveGradient(vec2 position, int iterations, float fmPhase,
                  vec2 dPdx, vec2 dPdy, out float residualVar) {
    float wavePhaseShift = length(position) * 0.1 + fmPhase;
    float frequency      = 1.0;
    float timeMultiplier = 2.0;
    float weight         = 1.0;
    vec2  sumGrad        = vec2(0.0);
    float lost           = 0.0;
    int   n              = min(iterations, MAX_WAVE_ITER);
    // Octave i's frequency is 1.18^i, so the bounds of the footprint give the
    // octave ranges up front: below maxFoot's limit every direction is fully
    // resolved (a plain loop, as cheap as unfiltered); past minFoot's (a lower
    // bound of max(|p.dPdx|, |p.dPdy|) over unit p: |det| over the larger
    // singular value, bounded by sqrt(2) * the Frobenius norm) none is.
    float maxFoot = max(length(dPdx), length(dPdy));
    float frob    = sqrt(dot(dPdx, dPdx) + dot(dPdy, dPdy));
    float minFoot = abs(dPdx.x * dPdy.y - dPdx.y * dPdy.x)
                  / max(1.41421356 * frob, 1e-20);
    const float INV_LOG_GROWTH = 6.0398;  // 1 / ln(1.18)
    int nFull = clamp(int(floor(log(uAAStart / max(maxFoot, 1e-20)) * INV_LOG_GROWTH)) + 1, 0, n);
    int nEnd  = clamp(int(ceil(log(uAAEnd / max(minFoot, 1e-20)) * INV_LOG_GROWTH)), nFull, n);

    int i = 0;
    for (; i < nFull; i++) {
        waveOctave(iWaveDirs[i], frequency, weight,
                   iTime * timeMultiplier + wavePhaseShift, 1.0, position, sumGrad);
        weight         = mix(weight, 0.0, 0.2);
        frequency     *= 1.18;
        timeMultiplier *= 1.07;
    }
    gResolved = float(nFull);
    for (; i < nEnd; i++) {
        vec2  p  = iWaveDirs[i];
        float a  = 1.0 - smoothstep(uAAStart, uAAEnd,
                       frequency * max(abs(dot(p, dPdx)), abs(dot(p, dPdy))));
        float fw = frequency * weight;
        lost      += (1.0 - a * a) * C_WAVE * fw * fw;
        gResolved += a;
        if (a > 0.0) {
            waveOctave(p, frequency, weight,
                       iTime * timeMultiplier + wavePhaseShift, a, position, sumGrad);
        }
        weight         = mix(weight, 0.0, 0.2);
        frequency     *= 1.18;
        timeMultiplier *= 1.07;
    }
    gEvaluated  = float(nEnd);
    float norm  = WAVE_HEIGHT / uWaveWeightSum;
    residualVar = lost * norm * norm + (nEnd < MAX_WAVE_ITER ? uTailVar[nEnd] : 0.0);
    return sumGrad / uWaveWeightSum;
}

// What normalize(mix(vertical, N, t)) does to the magnitude s of N's slope:
// t*s for gentle slopes, less for steep ones (applied to the unresolved
// spread so it is flattened as the resolved normal is).
float flattenSlope(float s, float t) {
    return t * s / max((1.0 - t) * sqrt(1.0 + s * s) + t, 1e-6);
}

// ---------------------------------------------------------------------------
// Tonemapping
// ---------------------------------------------------------------------------
vec3 aces_tonemap(vec3 color) {
    mat3 m1 = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    mat3 m2 = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    vec3 v = m1 * color;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

// Exact inverse of aces_tonemap (before its exposure): display colour back to
// the linear light it came from. The matrices are m1 and m2 inverted; the
// rational fit w = a(v) / b(v) is solved per channel as a quadratic in v. A
// display of 1 decodes to ~25, so the clamped highlights stay bright.
vec3 aces_inverse(vec3 display) {
    mat3 m1i = mat3(
         1.764741, -0.147028, -0.036337,
        -0.675778,  1.160252, -0.162436,
        -0.088963, -0.013224,  1.198773
    );
    mat3 m2i = mat3(
        0.643038, 0.059269, 0.005962,
        0.311187, 0.931436, 0.063929,
        0.045775, 0.009295, 0.930118
    );
    vec3 w = min(m2i * pow(clamp(display, 0.0, 1.0), vec3(2.2)), vec3(1.0));
    vec3 qa = 1.0 - 0.983729 * w;
    vec3 qb = 0.0245786 - 0.4329510 * w;
    vec3 qc = -(0.000090537 + 0.238081 * w);
    vec3 v = (-qb + sqrt(max(qb * qb - 4.0 * qa * qc, 0.0))) / (2.0 * qa);
    return max(m1i * v, vec3(0.0));
}

// ---------------------------------------------------------------------------
void main() {
    float dist = length(vWorldPos - iCameraPos);
    vec3  V    = normalize(vWorldPos - iCameraPos);
    // The pixel's footprint on the water, world units: what every fade below
    // keys on. From the lens, not dFdx/dFdy: on the displaced swell mesh
    // those jump at every triangle edge, and each triangle would fade its
    // waves differently (facets). Across the view it is the distance times
    // the pixel's angle (cos(off-axis) narrower off the screen's centre);
    // along it, that over the sine of the elevation.
    float across  = dist * uPixelAngle * max(dot(V, uCamForward), 0.2);
    float along   = across / max(abs(V.z), 1e-3);
    vec2  toward  = length(V.xy) > 1e-6 ? normalize(V.xy) : vec2(1.0, 0.0);
    vec2  dWx     = toward * along;
    vec2  dWy     = vec2(-toward.y, toward.x) * across;
    float footMax = along;
    // Metres of water above the seabed (negative on land), and how much that
    // changes across the pixel (the foam lines' filtering).
    float depthTexel;
    float depth      = -seabedHeight(vWorldPos.xy, depthTexel);
    float depthWidth = fwidth(depth);

    // Angle-based detail (sine of the view-ray elevation angle): 1 looking
    // straight down, 0 at the horizon.  Flattens the normal toward a mirror at
    // grazing angles, consistently across altitude.
    float detailAngle = smoothstep(GRAZE_LO, GRAZE_HI, abs(V.z));

    // --- Swell field (sampled once; drives both the small-wave frequency
    //     modulation below and the swell normal-tilt further down) -----------
    // Faded like an octave by its finer noise layer: sub-pixel swell keeps
    // only its mean slope variance, and skips its two gradient samples.
    float swellFade = 1.0 - smoothstep(uAAStart, uAAEnd,
                                       6.2831853 * iSwellScale * 2.1 * footMax);
    float epsW  = 0.5 / iSwellScale;  // world step ≈ 0.5 in primary noise space
    float s0    = swellField(vWorldPos.xy);
    vec2  sgrad = vec2(0.0);
    if (swellFade > 0.0) {
        sgrad = vec2(swellField(vWorldPos.xy + vec2(epsW, 0.0)) - s0,
                     swellField(vWorldPos.xy + vec2(0.0, epsW)) - s0) / 0.5;
    }
    // FM phase for the small waves: the swell recentred (so the phase wanders
    // both ways rather than only advancing) and scaled by the depth knob. World-
    // anchored, view-independent — so crests stay put as the camera moves.
    float fmPhase = uFmDepth * (s0 - 0.1);

    // --- Small waves (filtered by their size on screen) ----------------------
    // waveVar: their slope variance left unresolved, before the flattening.
    vec3  N;
    float waveVar;
    vec2  dPdx = dWx * WAVE_SCALE;
    vec2  dPdy = dWy * WAVE_SCALE;
    // How resolved the longest small wave is along the view (the worst
    // axis): the domain warp only serves the waves, so it fades with them.
    float lodFactor = detailAngle
                    * (1.0 - smoothstep(uAAStart, uAAEnd, footMax * WAVE_SCALE));
    gResolved  = 0.0;
    gEvaluated = 0.0;
    // Skipped only once even the longest wave running across the view is
    // gone: waveGradient fades each octave by its own direction, and cutting
    // it off by the worst axis drew a hard line where they vanished at once.
    if (detailAngle < 0.01 || across * WAVE_SCALE > uAAEnd) {
        // Grazing, or every octave sub-pixel: skip them, and keep their
        // whole slope variance for the fresnel.
        N       = vec3(0.0, 0.0, 1.0);
        waveVar = uTailVar[0];
    } else if (uWaveOff == 1) {
        // Debug: no small waves, start flat and let the swell gradient tilt it.
        N       = vec3(0.0, 0.0, 1.0);
        waveVar = 0.0;
    } else {
        // Domain warp (iterations and strength fade with the waves)
        int  warpIter  = int(mix(1.0, 4.0, lodFactor));
        vec2 warp      = domainWarp(vWorldPos.xy, warpIter, WARP_STRENGTH * lodFactor);
        vec2 warpedPos = vWorldPos.xy + warp;

        // Detail normal (WAVE_SCALE sets the wave size, applied at the call site)
        vec2 g = waveGradient(warpedPos * WAVE_SCALE, iIterationsNormal, fmPhase,
                              dPdx, dPdy, waveVar) * WAVE_HEIGHT;
        N = normalize(vec3(-g, 1.0));
    }
    // Flatten small-wave normal toward vertical at grazing angles.
    N = normalize(mix(vec3(0.0, 0.0, 1.0), N, detailAngle));

    // --- Swell tilt (faded above) -------------------------------------------
    // What the fade takes away goes to the spread: the local slope while it
    // is still resolved, the field's mean as it goes (the local value aliases).
    // Its tilt follows the displaced geometry's height (amplitude and depth).
    float swellTilt  = SWELL_STRENGTH * uSwellAmplitude / SWELL_REF_AMPLITUDE
                     * swellDepthFactor(depth);
    vec2  swellSlope = sgrad * swellTilt;
    float swellVar   = (1.0 - swellFade * swellFade) * swellTilt * swellTilt
                     * mix(SWELL_SLOPE_MS, dot(sgrad, sgrad), swellFade);
    N = normalize(N - vec3(swellSlope * swellFade, 0.0));

    // Large-scale noise mask, tending to its mean where it is sub-pixel.
    float noiseFade = 1.0 - smoothstep(uAAStart, uAAEnd,
                                       6.2831853 * NOISE_SCALE * 2.1 * footMax);
    float noise = NOISE_MEAN;
    if (noiseFade > 0.0) {
        noise = mix(NOISE_MEAN,
                    fbmNoise(vWorldPos.xy * NOISE_SCALE + iTime * 0.002), noiseFade);
    }
    float calm  = 1.0 - noise * NOISE_STRENGTH;
    N = normalize(mix(vec3(0.0, 0.0, 1.0), N, calm));

    // Final angle flatten on the fully combined normal — keeps horizon a flat mirror.
    N = normalize(mix(vec3(0.0, 0.0, 1.0), N, detailAngle));

    // The unresolved slopes, flattened as the resolved ones were.
    float spread = flattenSlope(sqrt(waveVar), detailAngle);
    spread = sqrt(spread * spread + swellVar);
    spread = flattenSlope(flattenSlope(spread, calm), detailAngle);

    float NdotV = max(0.0, dot(N, -V));
    float fresnel;
    if (uRoughFresnel == 1) {
        // The fresnel averaged over the facets the pixel covers: distant
        // water settles on a steady blend of sky and water colour instead
        // of flickering between them.
        vec2 lut = clamp(vec2(NdotV, spread / FRESNEL_SLOPE_MAX), 0.0, 1.0);
        fresnel  = texture(uFresnelLUT,
                           (lut * (FRESNEL_LUT_SIZE - 1.0) + 0.5) / FRESNEL_LUT_SIZE).r;
    } else {
        fresnel = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);
    }

    // Reflection coordinate computed per-fragment from the flat surface
    // position (z=0).  Doing the projective divide here — rather than
    // interpolating a clip-space coord from the vertices — keeps it consistent
    // when the geometry is displaced (the vertex clip-w would otherwise skew
    // the divide per cell, sampling the wrong reflection texel).
    vec4 rc          = uReflMVP * vec4(vWorldPos.xy, 0.0, 1.0);
    vec2 reflUV      = ((rc.xy / rc.w) * 0.5 + 0.5) * uReflUVScale;
    // Ripple perturbation fades with detail so grazing pixels sample the
    // undistorted (near-mirror) reflection instead of clamped edge texels.
    vec2 perturbedUV = reflUV + N.xy * iRippleStrength * detailAngle * uReflUVScale;

    // Clamp to the rendered region [0, uReflUVScale] — beyond it is the
    // texture's power-of-two padding (clear colour), so the ripple must never
    // push the sample out there or it picks up bright/garbage edge texels.
    // Sub-pixel ripples blur the reflection as far as they would have moved
    // its sample (the ripple offset of the unresolved spread).
    float reflLod = 0.0;
    if (uReflBlur > 0.0) {
        // The ripple offsets by N.xy: the sine of the tilt, not its slope.
        float tilt   = spread / sqrt(1.0 + spread * spread);
        float texels = tilt * detailAngle * iRippleStrength
                     * float(textureSize(iReflectionTex, 0).x) * uReflBlur;
        reflLod = log2(max(texels, 1.0));
    }
    vec3 reflSample = textureLod(iReflectionTex,
                                 clamp(perturbedUV, vec2(0.0), uReflUVScale), reflLod).rgb;
    // The reflection holds DISPLAY colours: the sky and terrain tonemap
    // themselves, with this same exposure. Decode them to linear light before
    // the fresnel blend and the tonemap below, or that tonemap applies twice
    // and lifts the dark (a night sky's reflection came out twice as bright as
    // the sky). A full mirror then shows exactly the sky's own colour.
    vec3 reflected = aces_inverse(reflSample) / uExposure;
    // The water's own colour, tinted over shallow seabed.
    vec3 waterColor = iWaterColor;
    if (uDepthColourOn == 1) {
        waterColor = mix(uShallowColor, iWaterColor,
                         deepWaterShare(depth, abs(V.z), uShallowDepth));
    }
    vec3 scattered = waterColor * iWaterLight * (1.0 - fresnel);

    vec3 C = fresnel * reflected + scattered;

    // --- Shoreline foam -----------------------------------------------------
    float foam = 0.0;
    // Also where the seabed says land: water drawn there is at the waterline
    // (the seabed's texels are coarser than the terrain), and where the
    // terrain stands above it, it hides the foam.
    float foamSure = 1.0 - smoothstep(FOAM_TEXEL_FULL, FOAM_TEXEL_GONE, depthTexel);
    if (uFoamOn == 1 && depth < uFoamDepth && foamSure > 0.0) {
        float k     = 6.2831853 / uFoamSpacing;
        // Shoreward: a line's phase holds where depth falls as time passes.
        // The swell field shifts it along the coast, so the lines wander.
        float phase = max(depth, 0.0) * k + iTime * uFoamSpeed + 6.2831853 * s0;
        float lines = smoothstep(0.55, 0.95, 0.5 + 0.5 * sin(phase));
        // Lines closer than a pixel apart: their mean coverage.
        lines = mix(FOAM_LINE_MEAN, lines,
                    1.0 - smoothstep(0.7854, 1.5708, depthWidth * k));
        float wash = 1.0 - smoothstep(0.0, 0.5 * uFoamDepth, depth);  // at the waterline
        foam = (1.0 - smoothstep(0.5 * uFoamDepth, uFoamDepth, depth)) * max(lines, wash);
        if (uFoamNoiseOn == 1) {
            float fade = 1.0 - smoothstep(uAAStart, uAAEnd,
                                          6.2831853 * FOAM_NOISE_SCALE * 2.1 * footMax);
            float n = NOISE_MEAN;
            if (fade > 0.0) {
                n = mix(NOISE_MEAN,
                        fbmNoise(vWorldPos.xy * FOAM_NOISE_SCALE + iTime * 0.3 * iWindDir),
                        fade);
            }
            foam *= smoothstep(0.3, 0.7, n);
        }
        foam *= uFoamStrength * foamSure;
    }
    C = mix(C, uFoamColor * iWaterLight, foam);

    if (uDebugMode == 1) {        // surface normal (xy encoded)
        fragColor = vec4(0.5 + 0.5 * N.x, 0.5 + 0.5 * N.y, N.z, 1.0); return;
    } else if (uDebugMode == 2) { // reflection UV
        fragColor = vec4(reflUV / uReflUVScale, 0.0, 1.0); return;
    } else if (uDebugMode == 3) { // fresnel
        fragColor = vec4(vec3(fresnel), 1.0); return;
    } else if (uDebugMode == 4) { // world-position grid (continuity check)
        vec2 g = fract(vWorldPos.xy / 31.25);
        fragColor = vec4(g, 0.0, 1.0); return;
    } else if (uDebugMode == 5) { // reflUV clamp indicator (red = clamped)
        vec2 cl = clamp(perturbedUV, vec2(0.0), uReflUVScale);
        bool clamped = (cl != perturbedUV);
        fragColor = vec4(clamped ? 1.0 : 0.0, reflSample.g, 0.0, 1.0); return;
    } else if (uDebugMode == 6) { // raw reflection sample (display colour)
        fragColor = vec4(reflSample, 1.0); return;
    } else if (uDebugMode == 7) { // pre-tonemap C*2 > 1 saturation (red=clipped chan)
        vec3 lin = C * 2.0;
        fragColor = vec4(step(1.0, lin.r), step(1.0, lin.g), step(1.0, lin.b), 1.0); return;
    } else if (uDebugMode == 8) { // FM phase field (swell-driven), wrapped to [0,1]
        fragColor = vec4(vec3(0.5 + 0.5 * sin(fmPhase)), 1.0); return;
    } else if (uDebugMode == 9) { // share of the octaves resolved (white = all)
        fragColor = vec4(vec3(gResolved / float(max(iIterationsNormal, 1))), 1.0); return;
    } else if (uDebugMode == 10) { // unresolved slope spread (white = the table's max)
        fragColor = vec4(vec3(spread / FRESNEL_SLOPE_MAX), 1.0); return;
    } else if (uDebugMode == 11) { // octaves evaluated, of the 36 HIGH has (cost)
        fragColor = vec4(vec3(gEvaluated / 36.0), 1.0); return;
    } else if (uDebugMode == 12) { // depth: black at the waterline, white at 50 m;
                                   // red on land, blue where no cascade reaches
        if (depth > 0.5 * -SEABED_DEEP) { fragColor = vec4(0.0, 0.0, 1.0, 1.0); return; }
        if (depth < 0.0) { fragColor = vec4(1.0, 0.0, 0.0, 1.0); return; }
        fragColor = vec4(vec3(sqrt(clamp(depth / 50.0, 0.0, 1.0))), 1.0); return;
    }

    // Haze applied to the DISPLAYED colour, after the tonemap: the air sits
    // between the eye and the water, so it is not something the water scatters.
    // The cloud field does the same, which is what lets the two meet cleanly at
    // the horizon despite using different tonemap operators.
    if (uAtmoCount > 0) {
        // The atmospheres replace the haze (in linear light, before the
        // tonemap, as the terrain does).
        vec3 seen = applyAtmospheres(C, uBaseToWorld * V, dist, uAtmoSunDir);
        fragColor = vec4(aces_tonemap(seen * uExposure), 1.0);
        return;
    }
    vec3 display = aces_tonemap(C * uExposure);
    fragColor = vec4(
        mix(display, uHazeColor, clamp(dist / uHazeDistance, 0.0, 1.0)), 1.0);
}
