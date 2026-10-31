#version 330

// The cirrus sheet (clouds.cirrus): fibrous ice cloud over a planet, one
// surface at the slab's mid-height. A sight line crosses thickness / |cos| of
// the slab; the density is sampled where it meets the sphere. cirrus.py
// mirrors the density on the CPU (cirrus_density); change them together.
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "cloud_field.glsl"

#define MAX_VORTICES 16

// 0: the behind node, keeping the crossings where sight lines leave the
// sphere (behind every cumulus); 1: the front node, where they enter it (in
// front of every cumulus). See cirrus.py.
uniform int   uFront;
uniform float uSheetRadius;     // planet centre to the sheet, metres
uniform float uThickness;       // the slab's depth, metres
uniform float uPixelAngle;      // radians per pixel
uniform int   uOwnAtmosphere;   // the planet's atmosphere, -1 none

// r: the cover (share of the sheet with cirrus), g: the raw density's
// threshold giving it (cirrus.calibrate).
uniform samplerCube cirrusMap;
// The zonal upper wind's stream function, metres, at latitudes evenly from
// -90 to 90 degrees (weather.stream_table); float32, interpolated here.
uniform sampler2D streamTable;
// The weather systems' terms (weather.stream_vortices): a: (centre, 1/width²)
// b.x: amplitude, metres.
uniform vec4  uVortexA[MAX_VORTICES];
uniform vec4  uVortexB[MAX_VORTICES];
uniform int   uVortexCount;

// Noise coordinates per metre (1 / length): x across the streaks
// (per stream-function metre), y along them, z the veil, w the meanders.
uniform vec4  uScales;
// x across the fibrils, y along them, z the meanders' amplitude (stream-
// function metres), w the streaks' share against the veil.
uniform vec4  uFibril;
uniform vec3  uOffsetStreak;
uniform vec3  uOffsetVeil;
uniform vec3  uOffsetFibril;
uniform vec3  uOffsetMeander;
// x: the density's rise above its threshold (raw units), y: extinction per
// metre, z: the optical depth cap, w: the mean density where cloudy (the far
// sheet's).
uniform vec4  uShape;
// x: streak spacing, y: fibril spacing (metres; how far each is drawn before
// fading), z: the noise's mean.
uniform vec4  uFar;
uniform vec4  uPhase;           // noise.phase_weights, and the forward g
// x: multiple scattering, y: shadow strength, z: fibril strength.
uniform vec4  uLight;

uniform vec3  sunColor;
uniform vec3  skyColor;
uniform float sunBrightness;
uniform float skyStrength;
uniform float exposure;

in vec3 vPosition;
flat in vec3 vEye;
out vec4 fragColor;

const float STREAK_W1 = 0.65;   // cirrus.STREAK_WEIGHTS
const float STREAK_W2 = 0.35;
const float ALONG_MIX = 0.7;    // cirrus.ALONG_MIX

// Hashed value noise, quintic (C2): one feature per unit, in [0, 1).
// cirrus.value_noise mirrors it. Computed, not fetched: a streak magnifies
// its noise ~50 times, and through the density's threshold a 3D texture's
// linear filter showed its creases as facets.
uint hashCell(ivec3 c) {
    uvec3 q = uvec3(c);
    uint h = (q.x * 0x8DA6B343u) ^ (q.y * 0xD8163841u);
    h ^= q.z * 0xCB1AB31Fu;
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return h;
}

float cellValue(ivec3 c) {
    return float(hashCell(c) >> 8u) * (1.0 / 16777216.0);
}

float hashNoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = x - i;
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    ivec3 c = ivec3(i);
    float x00 = mix(cellValue(c),                   cellValue(c + ivec3(1, 0, 0)), u.x);
    float x10 = mix(cellValue(c + ivec3(0, 1, 0)), cellValue(c + ivec3(1, 1, 0)), u.x);
    float x01 = mix(cellValue(c + ivec3(0, 0, 1)), cellValue(c + ivec3(1, 0, 1)), u.x);
    float x11 = mix(cellValue(c + ivec3(0, 1, 1)), cellValue(c + ivec3(1, 1, 1)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// The slow coordinates along the fibres (cirrus.along): p in the noise's y
// and z only, so x carries the stream function alone. Added to x, they
// tilted the fibres off the wind by (spacing / length) / speed, 15 degrees
// where it blows at a third of the jets'. Blind along (1, -0.7, -0.7),
// never an east-west wind.
vec3 along(vec3 p) {
    return vec3(0.0, p.y + ALONG_MIX * p.x, p.z + ALONG_MIX * p.x);
}

// The upper wind's stream function at sheet point p (planet frame), with
// the meanders: the fibres run along its contours (cirrus.fibre_coordinates).
float fibreStream(vec3 p, vec3 up) {
    int size = textureSize(streamTable, 0).x;
    float x = (degrees(asin(clamp(up.z, -1.0, 1.0))) + 90.0) / 180.0 * float(size - 1);
    int i = clamp(int(floor(x)), 0, size - 2);
    float psi = mix(texelFetch(streamTable, ivec2(i, 0), 0).r,
                    texelFetch(streamTable, ivec2(i + 1, 0), 0).r,
                    x - float(i));
    for (int v = 0; v < MAX_VORTICES; ++v) {
        if (v >= uVortexCount) break;
        vec3 offset = up - uVortexA[v].xyz;
        psi += uVortexB[v].x * exp(-dot(offset, offset) * uVortexA[v].w);
    }
    float wander = hashNoise(p * uScales.w + uOffsetMeander);
    return psi + uFibril.z * 2.0 * (wander - uFar.z);
}

void main() {
    // Which crossing of the sheet's sphere this fragment is, seen from this
    // pass's camera; shaded where the sight line meets the true sphere (the
    // mesh's facets sag under it).
    vec3 dir = normalize(vPosition - vEye);
    float eyeR = length(vEye);
    float b = dot(vEye, dir);
    float c = (eyeR - uSheetRadius) * (eyeR + uSheetRadius);
    float disc = b * b - c;
    if (disc <= 0.0) discard;
    float root = sqrt(disc);
    float t;
    if (c <= 0.0) {
        // From inside the sphere a sight line only leaves it.
        if (uFront == 1) discard;
        t = -b + root;
    } else {
        float enter = -b - root;
        float leave = -b + root;
        bool entering = dot(vPosition - vEye, dir) < 0.5 * (enter + leave);
        if (entering != (uFront == 1)) discard;
        t = entering ? enter : leave;
    }
    if (t <= 0.0) discard;
    vec3 p = vEye + dir * t;
    vec3 up = p / uSheetRadius;
    float distance = t;

    // A veil from inside the slab: fade near the camera.
    float nearFade = smoothstep(0.5 * uThickness, 2.0 * uThickness, distance);
    vec2 cover = texture(cirrusMap, up).rg;
    if (nearFade <= 0.0 || cover.r < 0.002) discard;

    float cosView = max(abs(dot(dir, up)), 0.04);
    float chord = uThickness / cosView;
    // Far away the streaks fade to the cover (they would alias to speckle),
    // the fibrils first, to their mean. Seen grazing, a pixel spans
    // 1 / cos more of the sheet one way: the geometric mean of the two
    // ways, or the streaks broke into one-pixel dashes at the horizon (all
    // of the 1 / cos would fade the long bands seen along the horizon too).
    float footprint = distance * uPixelAngle * inversesqrt(cosView);
    float detail = 1.0 - smoothstep(0.15, 0.6, footprint / uFar.x);
    float fibrilDetail = 1.0 - smoothstep(0.25, 1.0, footprint / uFar.y);

    float density = 0.0;
    if (detail > 0.0) {
        float psi = fibreStream(p, up);
        vec3 streakAt = vec3(psi * uScales.x, 0.0, 0.0) + along(p) * uScales.y + uOffsetStreak;
        float streak = STREAK_W1 * hashNoise(streakAt)
                     + STREAK_W2 * hashNoise(streakAt * 2.0 + OCTAVE_2_OFFSET);
        float veil = hashNoise(p * uScales.z + uOffsetVeil);
        float raw = mix(veil, streak, uFibril.w);
        density = smoothstep(cover.g, cover.g + uShape.x, raw);
        if (density > 0.0) {
            float fibril = uFar.z;
            if (fibrilDetail > 0.0) {
                vec3 fibrilAt = vec3(psi * uFibril.x, 0.0, 0.0) + along(p) * uFibril.y
                              + uOffsetFibril;
                fibril = mix(uFar.z, hashNoise(fibrilAt), fibrilDetail);
            }
            density *= 1.0 - uLight.z * (1.0 - fibril);
        }
    }
    float alphaNear = 1.0 - exp2(-min(uShape.y * density * chord, uShape.z));
    float alphaFar = cover.r * (1.0 - exp2(-min(uShape.y * uShape.w * chord, uShape.z)));
    float alpha = mix(alphaFar, alphaNear, detail) * nearFade;
    if (alpha < 1.0 / 512.0) discard;

    // Lighting last, after every discard.
    if ((uCloudDebug & 8) != 0) {
        fragColor = vec4(vec3(0.8) * alpha, alpha);
        return;
    }
    // Thin: no march, the sunward half of the slab at this density.
    float meanDensity = mix(uShape.w, density, detail);
    float cosSun = dot(up, sunDir);
    float sunDepth = uShape.y * meanDensity * 0.5 * uThickness / max(cosSun, 0.1);
    float sunVisibility = exp2(-min(sunDepth * uLight.y, uShape.z));
    float ambient = skyStrength + uLight.x * sunBrightness * (1.0 - sunVisibility);
    float phase = lobePhase(uPhase, dot(sunDir, dir));

    vec3 sunlight = sunColor;
    vec3 skylight = skyColor;
    vec3 twilight = vec3(0.0);
    if (uOwnAtmosphere >= 0) {
        // Through the planet's air at the cirrus's own height: it is lit
        // longer than the cumulus below, and reddened later.
        vec3 tint = cloudSunTint(uOwnAtmosphere, p, sunDir);
        sunlight *= tint;
        twilight = sunColor * cloudTwilight(uOwnAtmosphere, p, sunDir, tint);
        skylight *= smoothstep(-0.3, 0.1, cosSun);
    }
    vec3 colour = sunlight * (sunBrightness * sunVisibility * phase)
                + skylight * ambient
                + twilight * (sunBrightness * TWILIGHT_GLOW);

    // The air between the camera and the sheet, as on the cumulus's deck:
    // its in-scattered light is weighted by the alpha like the cirrus itself.
    if ((uCloudDebug & 2) == 0) {
        vec3 transmittance;
        vec3 inscatter;
        atmosphereAlong(dir, distance, sunDir, transmittance, inscatter);
        colour = colour * transmittance + inscatter;
    }
    colour = displayColour(colour * exposure);
    fragColor = vec4(colour * alpha, alpha);   // premultiplied "over"
}
