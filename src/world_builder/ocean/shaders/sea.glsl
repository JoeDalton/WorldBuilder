// The sea's surface shading near the camera: waves filtered by their size on
// screen, the swell's tilt, a rough fresnel, the planar reflection, colour by
// depth and shoreline foam. Included by terrain.frag, which draws the sea on
// its own chunks (seabed vertices lifted to the water); world_builder.ocean.Sea
// sets these inputs (terrain.Terrain binds harmless defaults).
//
// The waves live on the tangent plane at the camera's nadir (uSeaEast,
// uSeaNorth, uSeaUp, at uSeaOrigin in the render frame), in the coordinates
// of a chart anchored near it (uSeaChart is the origin's). Their fine
// octaves' phases at the origin are computed in float64 on the CPU
// (uWavePhase): float32 would lose them tens of kilometres from the anchor.
//
// Needs water.glsl, swell.glsl and the terrain's uShallowWater, uShallowDepth
// and uExposure declared before it.

uniform float     iTime;
uniform vec3      uSeaOrigin;      // the nadir's sea-level point, render frame
uniform vec3      uSeaEast;        // the tangent plane's axes, render frame
uniform vec3      uSeaNorth;
uniform vec3      uSeaUp;
uniform vec2      uSeaChart;       // the origin's chart coordinates, metres
uniform vec3      uSeaCamForward;  // the view's axis, render frame
uniform float     uSeaPixelAngle;  // a pixel's angle at the screen's centre, radians
uniform float     uSeaNear;        // 1: this shading; 0: the far (terrain) water
uniform vec3      iWaterColor;
uniform vec3      iWaterLight;     // daylight 1; dims it by night (Sea.set_light)
uniform sampler2D iReflectionTex;
uniform float     iRippleStrength;
uniform vec2      uReflUVScale;
uniform mat4      uSeaReflMVP;     // render frame -> reflection clip space
uniform vec2      iWindDir;        // normalised wind direction, chart axes
uniform int       iIterationsNormal;  // wave-detail iterations (quality knob)
// Per-iteration wave directions, precomputed on the CPU (they depend only on
// the iteration and the wind).
#define MAX_WAVE_ITER 64
uniform vec2      iWaveDirs[MAX_WAVE_ITER];
// Each octave's phase at the origin: frequency * its direction . the origin's
// (scaled) chart position, plus the time term, modulo 2 pi.
uniform float     uWavePhase[MAX_WAVE_ITER];
// Screen-space filtering of the waves: an octave fades out as its phase
// changes by uAAStart..uAAEnd radians from one pixel to the next (2*pi / the
// pixels per wavelength; Nyquist is pi). What it loses is kept as slope
// variance, shaded by the rough fresnel below, so the sea's average colour
// survives its waves going sub-pixel.
uniform float     uAAStart;
uniform float     uAAEnd;
// Sum of the octave weights over iIterationsNormal octaves.
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
// Frequency-modulation depth (radians) of the small waves by the swell.
uniform float     uFmDepth;
uniform int       uSeaDebug;   // 0 normal, 1 N, 2 reflUV, 3 fresnel, 4 worldgrid,
                               // 5 reflUV clamp, 6 raw reflection, 7 pre-tonemap
                               // clipping, 8 FM phase field, 9 octaves resolved,
                               // 10 unresolved slope spread, 11 octaves evaluated,
                               // 12 depth
uniform int       uWaveOff;    // debug: 1 = skip small-wave normal, swell only
// Colour by depth: on/off (the tint and its depth are the terrain's
// uShallowWater and uShallowDepth).
uniform int       uDepthColourOn;
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
#define DRAG_MULT         0.38
#define WAVE_HEIGHT       10.0
#define WAVE_SCALE        0.5
#define SWELL_STRENGTH    0.5    // how strongly the swell tilts the normal
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
// The vertex spacing over which foam fades out (metres): a band a metre or
// two deep cannot be placed by coarser chunks, and would show as a wide grey
// sheet.
#define FOAM_TEXEL_FULL   12.0
#define FOAM_TEXEL_GONE   48.0

// Angle-based detail falloff (sine of the view-ray elevation angle).
// Below GRAZE_HI the surface fades to a flat near-mirror; full detail above it.
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

// Analytic wave-height gradient (dH/dx, dH/dy) accumulated in a single pass:
// each wave term is wave = exp(sin(x) - 1) with x = freq * dot(p, pos) + t,
// so its slope is wave * cos(x) * freq * p. The loop advects the position to
// sharpen crests (as getwaves) without differentiating through it.
//
// The position is split: `position` (the part near the origin, plus the
// advection) is what float32 sees; the origin's own share of every octave's
// phase is uWavePhase, from the CPU. wavePhaseShift is the slow radial shift
// of the whole pattern (length(full) * 0.1, from the full position, coarse
// enough for float32) plus the swell's frequency modulation.
//
// Filtering: dPdx / dPdy are the pixel's footprint in the waves' domain. Each
// octave fades by how far its phase moves across the pixel along its own
// direction, scaling both its slope and its advection, and what it loses goes
// to residualVar. Evaluation stops once no direction can resolve an octave
// any more; the rest of the octaves' variance comes from uTailVar.
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

vec2 waveGradient(vec2 position, vec2 full, int iterations, float fmPhase,
                  vec2 dPdx, vec2 dPdy, out float residualVar) {
    float wavePhaseShift = length(full) * 0.1 + fmPhase;
    float frequency      = 1.0;
    float weight         = 1.0;
    vec2  sumGrad        = vec2(0.0);
    float lost           = 0.0;
    int   n              = min(iterations, MAX_WAVE_ITER);
    // Octave i's frequency is 1.18^i, so the bounds of the footprint give the
    // octave ranges up front: below maxFoot's limit every direction is fully
    // resolved; past minFoot's none is.
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
                   uWavePhase[i] + wavePhaseShift, 1.0, position, sumGrad);
        weight     = mix(weight, 0.0, 0.2);
        frequency *= 1.18;
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
                       uWavePhase[i] + wavePhaseShift, a, position, sumGrad);
        }
        weight     = mix(weight, 0.0, 0.2);
        frequency *= 1.18;
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

// Exact inverse of aces_tonemap (before its exposure): display colour back to
// the linear light it came from. A display of 1 decodes to ~25, so the
// clamped highlights stay bright.
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
// The sea's colour at a water fragment, in linear light before the
// atmospheres and the tonemap (or, with a debug view, the view's display
// colour, and raw set).
//
// P: the fragment, render frame; planet: the same, planet frame; Vr: from
// the camera to it, unit, render frame; dist: its distance; depth: metres of
// water under it; spacing: the chunk's vertex spacing (how finely the depth
// is known).
// ---------------------------------------------------------------------------
vec3 seaShade(vec3 P, vec3 planet, vec3 Vr, float dist, float depth, float spacing,
              out bool raw) {
    raw = false;
    // The fragment on the tangent plane, in chart coordinates: local, near
    // the origin (precise), and the full position (the chart's).
    vec3  rel   = P - uSeaOrigin;
    vec2  local = vec2(dot(rel, uSeaEast), dot(rel, uSeaNorth));
    vec2  wp    = uSeaChart + local;
    vec3  V     = vec3(dot(Vr, uSeaEast), dot(Vr, uSeaNorth), dot(Vr, uSeaUp));
    // The pixel's footprint on the water, world units: what every fade below
    // keys on. From the lens, not dFdx/dFdy (those jump at triangle edges).
    // Across the view it is the distance times the pixel's angle (narrower
    // off the screen's centre); along it, that over the sine of the elevation.
    float across  = dist * uSeaPixelAngle * max(dot(Vr, uSeaCamForward), 0.2);
    float along   = across / max(abs(V.z), 1e-3);
    vec2  toward  = length(V.xy) > 1e-6 ? normalize(V.xy) : vec2(1.0, 0.0);
    vec2  dWx     = toward * along;
    vec2  dWy     = vec2(-toward.y, toward.x) * across;
    float footMax = along;
    // How much the depth changes across the pixel (the foam lines' filtering).
    float depthWidth = fwidth(depth);

    // Angle-based detail: 1 looking straight down, 0 at the horizon.
    // Flattens the normal toward a mirror at grazing angles.
    float detailAngle = smoothstep(GRAZE_LO, GRAZE_HI, abs(V.z));

    // --- Swell field (the geometry's, sampled once; drives both the small
    //     waves' frequency modulation and the swell's tilt) ------------------
    // Its tilt faded like an octave by its finer noise layer: sub-pixel swell
    // keeps only its mean slope variance.
    float swellFade = 1.0 - smoothstep(uAAStart, uAAEnd,
                                       6.2831853 * iSwellScale * 2.1 * footMax);
    vec3  swellGrad;
    float s0    = swellField(planet, 0.0, swellGrad);
    // Per unit of the first layer's noise, on the plane's axes.
    vec2  sgrad = vec2(dot(swellGrad, uSeaEast), dot(swellGrad, uSeaNorth))
                / iSwellScale;
    // FM phase for the small waves: the swell recentred and scaled by the
    // depth knob, world-anchored.
    float fmPhase = uFmDepth * (s0 - 0.1);

    // --- Small waves (filtered by their size on screen) ----------------------
    vec3  N;
    float waveVar;
    vec2  dPdx = dWx * WAVE_SCALE;
    vec2  dPdy = dWy * WAVE_SCALE;
    // How resolved the longest small wave is along the view: the domain warp
    // only serves the waves, so it fades with them.
    float lodFactor = detailAngle
                    * (1.0 - smoothstep(uAAStart, uAAEnd, footMax * WAVE_SCALE));
    gResolved  = 0.0;
    gEvaluated = 0.0;
    if (detailAngle < 0.01 || across * WAVE_SCALE > uAAEnd) {
        // Grazing, or every octave sub-pixel: skip them, and keep their
        // whole slope variance for the fresnel.
        N       = vec3(0.0, 0.0, 1.0);
        waveVar = uTailVar[0];
    } else if (uWaveOff == 1) {
        N       = vec3(0.0, 0.0, 1.0);
        waveVar = 0.0;
    } else {
        int  warpIter = int(mix(1.0, 4.0, lodFactor));
        vec2 warp     = domainWarp(wp, warpIter, WARP_STRENGTH * lodFactor);
        vec2 g = waveGradient((local + warp) * WAVE_SCALE, (wp + warp) * WAVE_SCALE,
                              iIterationsNormal, fmPhase, dPdx, dPdy, waveVar)
               * WAVE_HEIGHT;
        N = normalize(vec3(-g, 1.0));
    }
    N = normalize(mix(vec3(0.0, 0.0, 1.0), N, detailAngle));

    // --- Swell tilt (faded above) -------------------------------------------
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
        noise = mix(NOISE_MEAN, fbmNoise(wp * NOISE_SCALE + iTime * 0.002), noiseFade);
    }
    float calm  = 1.0 - noise * NOISE_STRENGTH;
    N = normalize(mix(vec3(0.0, 0.0, 1.0), N, calm));
    N = normalize(mix(vec3(0.0, 0.0, 1.0), N, detailAngle));

    // The unresolved slopes, flattened as the resolved ones were.
    float spread = flattenSlope(sqrt(waveVar), detailAngle);
    spread = sqrt(spread * spread + swellVar);
    spread = flattenSlope(flattenSlope(spread, calm), detailAngle);

    float NdotV = max(0.0, dot(N, -V));
    float fresnel;
    if (uRoughFresnel == 1) {
        // The fresnel averaged over the facets the pixel covers.
        vec2 lut = clamp(vec2(NdotV, spread / FRESNEL_SLOPE_MAX), 0.0, 1.0);
        fresnel  = texture(uFresnelLUT,
                           (lut * (FRESNEL_LUT_SIZE - 1.0) + 0.5) / FRESNEL_LUT_SIZE).r;
    } else {
        fresnel = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);
    }

    // The reflection: the fragment's footprint on the mirror plane, projected
    // per fragment.
    vec4 rc          = uSeaReflMVP * vec4(P - uSeaUp * dot(rel, uSeaUp), 1.0);
    vec2 reflUV      = ((rc.xy / rc.w) * 0.5 + 0.5) * uReflUVScale;
    vec2 perturbedUV = reflUV + N.xy * iRippleStrength * detailAngle * uReflUVScale;
    // Clamp to the rendered region; sub-pixel ripples blur the reflection as
    // far as they would have moved its sample.
    float reflLod = 0.0;
    if (uReflBlur > 0.0) {
        float tilt   = spread / sqrt(1.0 + spread * spread);
        float texels = tilt * detailAngle * iRippleStrength
                     * float(textureSize(iReflectionTex, 0).x) * uReflBlur;
        reflLod = log2(max(texels, 1.0));
    }
    vec3 reflSample = textureLod(iReflectionTex,
                                 clamp(perturbedUV, vec2(0.0), uReflUVScale), reflLod).rgb;
    // The reflection holds display colours: decode them to linear light.
    vec3 reflected = aces_inverse(reflSample) / uExposure;
    vec3 waterColor = iWaterColor;
    if (uDepthColourOn == 1) {
        waterColor = mix(uShallowWater, iWaterColor,
                         deepWaterShare(depth, abs(V.z), uShallowDepth));
    }
    vec3 scattered = waterColor * iWaterLight * (1.0 - fresnel);
    vec3 C = fresnel * reflected + scattered;

    // --- Shoreline foam -----------------------------------------------------
    float foam = 0.0;
    float foamSure = 1.0 - smoothstep(FOAM_TEXEL_FULL, FOAM_TEXEL_GONE, spacing);
    if (uFoamOn == 1 && depth < uFoamDepth && foamSure > 0.0) {
        float k     = 6.2831853 / uFoamSpacing;
        // Shoreward: a line's phase holds where depth falls as time passes.
        // The swell field shifts it along the coast, so the lines wander.
        float phase = max(depth, 0.0) * k + iTime * uFoamSpeed + 6.2831853 * s0;
        float lines = smoothstep(0.55, 0.95, 0.5 + 0.5 * sin(phase));
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
                        fbmNoise(wp * FOAM_NOISE_SCALE + iTime * 0.3 * iWindDir), fade);
            }
            foam *= smoothstep(0.3, 0.7, n);
        }
        foam *= uFoamStrength * foamSure;
    }
    C = mix(C, uFoamColor * iWaterLight, foam);

    raw = uSeaDebug != 0;
    if (uSeaDebug == 1) {        // surface normal (xy encoded)
        return vec3(0.5 + 0.5 * N.x, 0.5 + 0.5 * N.y, N.z);
    } else if (uSeaDebug == 2) { // reflection UV
        return vec3(reflUV / uReflUVScale, 0.0);
    } else if (uSeaDebug == 3) { // fresnel
        return vec3(fresnel);
    } else if (uSeaDebug == 4) { // world-position grid (continuity check)
        return vec3(fract(wp / 31.25), 0.0);
    } else if (uSeaDebug == 5) { // reflUV clamp indicator (red = clamped)
        vec2 cl = clamp(perturbedUV, vec2(0.0), uReflUVScale);
        return vec3(cl != perturbedUV ? 1.0 : 0.0, reflSample.g, 0.0);
    } else if (uSeaDebug == 6) { // raw reflection sample (display colour)
        return reflSample;
    } else if (uSeaDebug == 7) { // pre-tonemap C*2 > 1 saturation (red=clipped chan)
        return step(vec3(1.0), C * 2.0);
    } else if (uSeaDebug == 8) { // FM phase field
        return vec3(0.5 + 0.5 * sin(fmPhase));
    } else if (uSeaDebug == 9) { // share of the octaves resolved (white = all)
        return vec3(gResolved / float(max(iIterationsNormal, 1)));
    } else if (uSeaDebug == 10) { // unresolved slope spread (white = the table's max)
        return vec3(spread / FRESNEL_SLOPE_MAX);
    } else if (uSeaDebug == 11) { // octaves evaluated, of the 36 HIGH has (cost)
        return vec3(gEvaluated / 36.0);
    } else if (uSeaDebug == 12) { // depth: black at the waterline, white at 50 m
        return vec3(sqrt(clamp(depth / 50.0, 0.0, 1.0)));
    }
    return C;
}
