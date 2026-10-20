#version 330

// Procedural stars (starfield.make_stars) as points of 3x3 pixels, of which
// stars.frag lights at most 2x2. Each carries the sky behind it, so the
// fragment can add exactly what the star adds once tonemapped.

#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "sky_common.glsl"

uniform mat4 p3d_ModelViewProjectionMatrix;  // the sky sphere's
uniform float uStarBrightness;

in vec4 p3d_Vertex;   // direction in the star map
in vec4 star;         // rgb: tint, a: flux

out vec3 vBase;       // the sky behind, exposed
out vec3 vStar;       // the star's radiance on a pixel it covers whole, exposed

const int AIR_STEPS = 4;

void main() {
    // The map's rotation is orthonormal: its transpose maps it back.
    vec3 dir = transpose(uStarRotation) * p3d_Vertex.xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(dir, 1.0);
    gl_PointSize = 3.0;
    vec3 transmittance, inscatter;
    atmosphereAlong(dir, 1e30, uSunDir, AIR_STEPS, transmittance, inscatter);
    float adapt = starAdaptation(inscatter);
    vec3 glow = skyBackground(textureLod(uStars, p3d_Vertex.xyz, 0.0).rgb);
    vBase = (glow * adapt * transmittance + inscatter) * uExposure;
    vStar = star.rgb * star.a * uStarBrightness * adapt * transmittance * uExposure;
}
