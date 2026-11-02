#version 330
// The nearby ridges' shadow ceiling, baked on the GPU for one sun and one
// square of ground (world_builder.terrain.shadow_bake.LocalBake): each node
// of the chart around the camera marches toward the sun over the chart's
// ground, the occluders between near and far (shadow_map.march_local is the
// same march in numpy). Ground outside the square is unknown: no occluder.
// Written in metres (shadows.glsl's localShadow reads it so).

uniform sampler2D uHeights;  // the ground at the chart's nodes, metres
uniform vec3 uSun;           // toward the sun, the planet's frame (unit)
// The chart (terrain.chart.AzimuthalEquidistantChart): its centre's up and
// the radius, east and the node spacing, north and the nodes per side.
uniform vec4 uChart[3];
uniform ivec2 uSteps;        // the first spacing counted, past the last
uniform float uFar;          // metres
uniform float uTop;          // the square's highest ground, metres: the
                             // march stops once nothing can rise above

out vec4 fragColor;

const float NO_CEILING = -1.0e5;   // shadow_map.NO_CEILING

// A direction's place on the chart, metres (chart.from_dirs).
vec2 chartOf(vec3 d) {
    vec2 across = vec2(dot(d, uChart[1].xyz), dot(d, uChart[2].xyz));
    float sine = length(across);
    float angle = atan(sine, dot(d, uChart[0].xyz));
    return across * (sine > 1e-12 ? uChart[0].w * angle / sine : uChart[0].w);
}

// The direction of a place on the chart (chart.to_dirs).
vec3 dirOf(vec2 xy) {
    float rho = length(xy);
    float angle = rho / uChart[0].w;
    vec3 across = rho > 1e-9 ? (xy.x * uChart[1].xyz + xy.y * uChart[2].xyz) / rho
                             : vec3(0.0);
    return cos(angle) * uChart[0].xyz + sin(angle) * across;
}

// The ground there (bilinear between nodes), and whether it is known.
float groundAt(vec2 xy, out bool inside) {
    float size = uChart[2].w;
    float half_ = 0.5 * (size - 1.0) * uChart[1].w;
    inside = max(abs(xy.x), abs(xy.y)) <= half_;
    return texture(uHeights, (xy / uChart[1].w + 0.5 * size) / size).r;
}

void main() {
    float radius = uChart[0].w;
    float spacing = uChart[1].w;
    float size = uChart[2].w;
    // This node: (i - (n - 1) / 2) spacings from the centre.
    vec2 xy = (gl_FragCoord.xy - 0.5 * size) * spacing;
    vec3 d = normalize(dirOf(xy));
    float cosZ = clamp(dot(d, uSun), -1.0, 1.0);
    float zenith = acos(cosZ);
    vec3 toward = normalize(uSun - cosZ * d);
    bool inside;
    float lifted = radius + max(groundAt(xy, inside), 0.0);
    float ceiling = NO_CEILING;
    float distance = uFar;
    for (int k = uSteps.x; k < uSteps.y; ++k) {
        float angle = float(k) * spacing / radius;
        vec2 at = chartOf(cos(angle) * d + sin(angle) * toward);
        float ground = groundAt(at, inside);
        if (!inside) {
            continue;
        }
        // shadow_map.ray_gain, without the cancellation.
        float gain = lifted * 2.0 * cos(zenith - 0.5 * angle) * sin(0.5 * angle)
                   / max(sin(zenith - angle), 1e-9);
        if (ground - gain > ceiling) {
            ceiling = ground - gain;
            distance = float(k) * spacing;
        }
        if (angle >= 2.0 * (zenith - 1.5707963) && uTop - gain <= ceiling) {
            break;
        }
    }
    fragColor = vec4(ceiling, distance, 0.0, 1.0);
}
