// Linear radiance to display colour: the tonemap the terrain, the sky, the
// stars and the clouds share. Include once per shader with #pragma include.

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

// The brightest channel of a linear colour (never 0: it divides).
float displayPeak(vec3 colour) {
    return max(max(colour.r, colour.g), max(colour.b, 1e-6));
}

// A linear colour's hue at its brightest channel's tonemapped level. Per
// channel, the tonemap flattens whatever is brighter than white toward
// white: an orange light far past white maps to white. Scaled by its peak
// instead, it keeps its hue.
vec3 keptHue(vec3 colour, float peak) {
    return pow(colour / peak, vec3(1.0 / 2.2)) * aces_tonemap(vec3(peak)).g;
}

// Where displayColour starts keeping the hue (brightest channel, after
// exposure), and where it keeps it fully.
const float KEEP_HUE_FROM = 0.4;
const float KEEP_HUE_TO = 5.0;

// The tonemap per channel up to KEEP_HUE_FROM, fading to the kept hue by
// KEEP_HUE_TO: below white, a light tint desaturates as it does on everything
// else; far above, a coloured light keeps its colour.
vec3 displayColour(vec3 colour) {
    float peak = displayPeak(colour);
    float keep = smoothstep(KEEP_HUE_FROM, KEEP_HUE_TO, peak);
    vec3 display = aces_tonemap(colour);
    return keep > 0.0 ? mix(display, keptHue(colour, peak), keep) : display;
}
