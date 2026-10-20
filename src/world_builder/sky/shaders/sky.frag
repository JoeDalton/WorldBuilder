#version 330

// The sky: stars and the sun's disc, seen through every atmosphere. Drawn
// first, behind everything; what is drawn over it applies the atmospheres
// itself. Tonemapped like the terrain.

#pragma include "../../atmosphere/shaders/atmosphere.glsl"

in vec3 vDir;
out vec4 fragColor;

uniform samplerCube uStars;
uniform mat3  uStarRotation;     // world -> star map
uniform float uStarIntensity;    // linear radiance of a full-white texel
uniform vec3  uSunDir;           // toward the sun
uniform vec3  uSunRadiance;      // the disc's linear radiance (before the air)
uniform float uSunCos;           // cosine of the disc's angular radius
uniform float uExposure;
uniform float uStarAdaptation;   // how fast a bright sky hides the stars

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

void main() {
    vec3 dir = normalize(vDir);
    vec3 stars = pow(texture(uStars, uStarRotation * dir).rgb, vec3(2.2)) * uStarIntensity;
    // A soft-edged disc, a pixel or two of antialiasing.
    float cosSun = dot(dir, uSunDir);
    float edge = (1.0 - uSunCos) * 0.15;
    float disc = smoothstep(uSunCos - edge, uSunCos + edge, cosSun);
    vec3 transmittance, inscatter;
    atmosphereAlong(dir, 1e30, uSunDir, transmittance, inscatter);
    // The eye adapts to a bright sky: stars fade out behind it, not just by
    // the little light the air absorbs.
    float skyLuminance = dot(inscatter, vec3(0.2126, 0.7152, 0.0722));
    stars *= exp(-skyLuminance * uStarAdaptation);
    vec3 background = stars + uSunRadiance * disc;
    vec3 colour = background * transmittance + inscatter;
    fragColor = vec4(aces_tonemap(colour * uExposure), 1.0);
}
