// The sea's swell: a height field of the planet-frame point, the same for
// every camera, which terrain.vert raises the water by and sea.glsl tilts its
// normal by. The native library computes the very same field on the CPU
// (noise.c's wb_swell, from swell.h), so collisions follow the drawn water.
//
// The product of two layers of two-octave value noise, each layer turning
// slowly about its own axis through the planet's centre (uSwellRotA, B, from
// the CPU in float64): the pattern drifts with the wind and changes shape
// instead of sliding past. Octaves finer than a chunk's vertex spacing can
// carry fade to their mean, as the terrain's own noise does, so the swell
// flattens with distance through the terrain's LOD.

uniform float uSwellAmplitude;  // the water's height per unit of field, deep water, metres
uniform float iSwellScale;      // the first layer's frequency, cycles per metre
uniform mat3  uSwellRotA;       // each layer's drift: a rotation of the point
uniform mat3  uSwellRotB;
// The swell's share of its deep-water amplitude by depth.
uniform int   uSwellByDepth;
uniform float uSwellShallow;
uniform float uSwellDeep;

// The lattice's value at an integer cell, in [0, 1): integer arithmetic, so
// the CPU's copy (swell.h) gives the same bits.
float swellLattice(ivec3 c) {
    uvec3 u = uvec3(c);
    uint h = (u.x * 0x8da6b343u) ^ (u.y * 0xd8163841u) ^ (u.z * 0xcb1ab31fu);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return float(h >> 8u) * (1.0 / 16777216.0);
}

// Value noise in [0, 1] at q (lattice units), smoothstep-interpolated, and its
// gradient per lattice unit.
float swellNoise(vec3 q, out vec3 grad) {
    vec3 i = floor(q);
    vec3 f = q - i;
    vec3 u = f * f * (3.0 - 2.0 * f);
    vec3 du = 6.0 * f * (1.0 - f);
    ivec3 c = ivec3(i);
    float v000 = swellLattice(c);
    float v100 = swellLattice(c + ivec3(1, 0, 0));
    float v010 = swellLattice(c + ivec3(0, 1, 0));
    float v110 = swellLattice(c + ivec3(1, 1, 0));
    float v001 = swellLattice(c + ivec3(0, 0, 1));
    float v101 = swellLattice(c + ivec3(1, 0, 1));
    float v011 = swellLattice(c + ivec3(0, 1, 1));
    float v111 = swellLattice(c + ivec3(1, 1, 1));
    float x00 = mix(v000, v100, u.x), x10 = mix(v010, v110, u.x);
    float x01 = mix(v001, v101, u.x), x11 = mix(v011, v111, u.x);
    float y0 = mix(x00, x10, u.y), y1 = mix(x01, x11, u.y);
    grad.x = du.x * mix(mix(v100 - v000, v110 - v010, u.y),
                        mix(v101 - v001, v111 - v011, u.y), u.z);
    grad.y = du.y * mix(x10 - x00, x11 - x01, u.z);
    grad.z = du.z * (y1 - y0);
    return mix(y0, y1, u.z);
}

// One layer at q: two octaves, each faded toward its mean (1/2) by its
// weight in w.
float swellLayer(vec3 q, vec2 w, out vec3 grad) {
    vec3 g0, g1;
    float n0 = swellNoise(q, g0);
    float n1 = swellNoise(q * 2.1 + vec3(5.3, 1.7, 3.1), g1);
    grad = 0.6 * w.x * g0 + 0.4 * w.y * 2.1 * g1;
    return 0.6 * mix(0.5, n0, w.x) + 0.4 * mix(0.5, n1, w.y);
}

// An octave's share at this frequency for a resolution (noise.octave_weight):
// all of it below half the cut-off frequency, none at it. maxFreq <= 0 = no
// limit.
float swellBand(float freq, float maxFreq) {
    if (maxFreq <= 0.0) {
        return 1.0;
    }
    float t = clamp((2.0 * maxFreq - 2.0 * freq) / maxFreq, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// The swell field at a planet-frame point P (metres), in [0, 1], its octaves
// above maxFreq (cycles per metre) faded to their mean; grad: its gradient
// per metre, planet axes.
float swellField(vec3 P, float maxFreq, out vec3 grad) {
    float s = iSwellScale;
    vec2 wa = vec2(swellBand(s, maxFreq), swellBand(2.1 * s, maxFreq));
    vec2 wb = vec2(swellBand(1.7 * s, maxFreq), swellBand(3.57 * s, maxFreq));
    vec3 ga, gb;
    float a = swellLayer((uSwellRotA * P) * s, wa, ga);
    float b = swellLayer((uSwellRotB * P) * (1.7 * s), wb, gb);
    grad = transpose(uSwellRotA) * ga * (s * b)
         + transpose(uSwellRotB) * gb * (1.7 * s * a);
    return a * b;
}

// The swell's share of its deep-water amplitude over this depth (metres):
// none on the beach, all of it past uSwellDeep. 1 when off.
float swellDepthFactor(float depth) {
    return uSwellByDepth == 1 ? smoothstep(uSwellShallow, uSwellDeep, depth) : 1.0;
}

// The field's mean, which faded octaves tend to (each layer's is 1/2).
#define SWELL_MEAN 0.25

// The water's height above sea level at a point over seabed this deep, for a
// resolution (maxFreq; terrain.vert's is its chunk's): about the sea level,
// and exactly at it where the swell is too fine for the chunk.
float swellHeight(vec3 P, float depth, float maxFreq) {
    vec3 g;
    return uSwellAmplitude * swellDepthFactor(depth)
         * (swellField(P, maxFreq, g) - SWELL_MEAN);
}
