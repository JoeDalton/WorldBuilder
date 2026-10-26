// What the sky shader and the star points share: the background's colour
// and the tonemap. Include with #pragma include.

#pragma include "../../atmosphere/shaders/display.glsl"

uniform samplerCube uStars;      // painted colours, or the glow's parameters
uniform mat3  uStarRotation;     // world -> star map
uniform float uStarIntensity;    // painted map: linear radiance of full white
// Procedural glow (starfield.make_glow: r band, g nebula density, b palette
// coordinate), coloured here so the look is live (starfield.NebulaStyle).
uniform int   uGlowMode;         // 1: procedural glow, 0: painted cube map
uniform vec3  uMilkyWay;
uniform vec3  uNebulaPalette[3];
uniform float uNebulaIntensity;
uniform vec3  uSunDir;           // toward the sun
uniform float uExposure;
uniform float uStarAdaptation;   // how fast a bright sky hides the stars

// The background's linear radiance from a texel of uStars.
vec3 skyBackground(vec3 texel) {
    if (uGlowMode == 0) {
        return pow(texel, vec3(2.2)) * uStarIntensity;
    }
    float k = texel.b * 2.0;
    vec3 tint = k < 1.0 ? mix(uNebulaPalette[0], uNebulaPalette[1], k)
                        : mix(uNebulaPalette[1], uNebulaPalette[2], k - 1.0);
    return texel.r * uMilkyWay + texel.g * uNebulaIntensity * tint;
}

// The eye adapts to a bright sky: stars fade out behind it, not just by the
// little light the air absorbs.
float starAdaptation(vec3 inscatter) {
    return exp(-dot(inscatter, vec3(0.2126, 0.7152, 0.0722)) * uStarAdaptation);
}
