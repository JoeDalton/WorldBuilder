#version 330
// The mountains' shadow ceiling, baked on the GPU for one sun: one cube
// face per card (world_builder.terrain.shadow_bake.CeilingBake). Each texel
// marches toward the sun over the ground (shadow_map.march_ceiling is the
// same march in numpy): from the first counted step on, a node k half
// texels toward the sun along the great circle bounds the ceiling at its
// ground less the rise of the sun's ray from here; the march stops once no
// ground can rise above the ceiling found, or at the sub-solar point. Past
// the angle no ground sees the sun at, the full shadow. Written in km, as
// shadows.glsl's terrainShadow reads it.

uniform samplerCube uHeights;  // the ground at the texel centres, metres
// The highest ground within uReach of each block (shadow_map.local_tops):
// the march stops on it while that near, on the planet's beyond.
uniform samplerCube uTops;
uniform float uReach;          // radians
uniform vec3 uSun;             // toward the sun, the planet's frame (unit)
uniform vec4 uBake;            // radius, highest ground, step, last angle
uniform ivec2 uSteps;          // the first step counted, the most steps
uniform float uSize;           // texels per face edge
uniform int uFace;

out vec4 fragColor;

const float NO_CEILING = -1.0e5;   // shadow_map.NO_CEILING
const float FULL_SHADOW = 1.0e5;   // shadow_map.FULL_SHADOW
const float HALF_PI = 1.5707963267948966;

// sky.starfield._FACES: (major, s, t) per face, OpenGL order.
const vec3 MAJOR[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                              vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));
const vec3 S_AXIS[6] = vec3[6](vec3(0, 0, -1), vec3(0, 0, 1), vec3(1, 0, 0),
                               vec3(1, 0, 0), vec3(1, 0, 0), vec3(-1, 0, 0));
const vec3 T_AXIS[6] = vec3[6](vec3(0, -1, 0), vec3(0, -1, 0), vec3(0, 0, 1),
                               vec3(0, 0, -1), vec3(0, -1, 0), vec3(0, -1, 0));

void main() {
    // The texel's centre: columns along s, rows (bottom-up) along t.
    vec2 c = gl_FragCoord.xy / uSize * 2.0 - 1.0;
    vec3 d = normalize(MAJOR[uFace] + c.x * S_AXIS[uFace] + c.y * T_AXIS[uFace]);
    float radius = uBake.x;
    float top = uBake.y;
    float step = uBake.z;
    float cosZ = clamp(dot(d, uSun), -1.0, 1.0);
    float zenith = acos(cosZ);
    float ceiling = NO_CEILING;
    float distance = float(uSteps.x) * step * radius;
    if (zenith >= uBake.w) {
        fragColor = vec4(FULL_SHADOW / 1000.0, step * radius / 1000.0, 0.0, 1.0);
        return;
    }
    vec3 toward = uSun - cosZ * d;
    float length_ = length(toward);
    if (length_ > 1e-9) {
        toward /= length_;
        float lifted = radius + max(texture(uHeights, d).r, 0.0);
        float nearTop = texture(uTops, d).r;
        for (int k = uSteps.x; k < uSteps.y; ++k) {
            float angle = float(k) * step;
            if (angle >= zenith) {
                break;
            }
            vec3 node = cos(angle) * d + sin(angle) * toward;
            // shadow_map.ray_gain: r (sin z / sin(z - a) - 1), without the
            // cancellation: 2 cos(z - a/2) sin(a/2) / sin(z - a).
            float gain = lifted * 2.0 * cos(zenith - 0.5 * angle) * sin(0.5 * angle)
                       / max(sin(zenith - angle), 1e-9);
            float candidate = texture(uHeights, node).r - gain;
            if (candidate > ceiling) {
                ceiling = candidate;
                distance = angle * radius;
            }
            // Past the ray's lowest point, nothing can rise above it.
            float highest = angle < uReach ? nearTop : top;
            if (angle >= 2.0 * (zenith - HALF_PI) && highest - gain <= ceiling) {
                break;
            }
        }
    }
    fragColor = vec4(ceiling / 1000.0, distance / 1000.0, 0.0, 1.0);
}
