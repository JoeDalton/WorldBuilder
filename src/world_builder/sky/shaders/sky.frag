#version 330

// The sky: the background (painted, or the procedural glow) and the sun's
// disc, seen through every atmosphere. Drawn first, behind everything; what
// is drawn over it applies the atmospheres itself. Tonemapped like the
// terrain. Procedural stars are points drawn just after it (stars.vert).

#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "sky_common.glsl"

in vec3 vDir;
out vec4 fragColor;

uniform vec3  uSunRadiance;      // the disc's linear radiance (before the air)
uniform float uSunCos;           // cosine of the disc's angular radius

void main() {
    vec3 dir = normalize(vDir);
    vec3 background = skyBackground(texture(uStars, uStarRotation * dir).rgb);
    // A soft-edged disc, a pixel or two of antialiasing.
    float cosSun = dot(dir, uSunDir);
    float edge = (1.0 - uSunCos) * 0.15;
    float disc = smoothstep(uSunCos - edge, uSunCos + edge, cosSun);
    vec3 transmittance, inscatter;
    atmosphereAlong(dir, 1e30, uSunDir, transmittance, inscatter);
    background = background * starAdaptation(inscatter) + uSunRadiance * disc;
    vec3 colour = background * transmittance + inscatter;
    fragColor = vec4(aces_tonemap(colour * uExposure), 1.0);
}
