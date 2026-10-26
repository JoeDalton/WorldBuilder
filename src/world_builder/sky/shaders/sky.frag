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

// The low sun (see main): how much further its hue saturates at the rim and
// in the glare, how much less at the centre (a power of its hue: > 1 redder,
// < 1 paler and brighter), and its glare's radiance (a share of the disc's)
// and width (in disc radii).
const float SUN_REDDEN = 0.6;
const float SUN_CORE = 0.9;
const float SUN_GLARE = 0.004;
const float SUN_GLARE_RADII = 2.0;

void main() {
    vec3 dir = normalize(vDir);
    vec3 background = skyBackground(texture(uStars, uStarRotation * dir).rgb);
    // A soft-edged disc, a pixel or two of antialiasing.
    float cosSun = dot(dir, uSunDir);
    float edge = (1.0 - uSunCos) * 0.15;
    float disc = smoothstep(uSunCos - edge, uSunCos + edge, cosSun);
    vec3 transmittance, inscatter;
    atmosphereAlong(dir, 1e30, uSunDir, transmittance, inscatter);
    background *= starAdaptation(inscatter);
    // The sun as it reaches the camera along this ray. Low, its light is
    // no brighter than the glow around it on screen (both past white, see
    // below), and the disc was lost in it. So a low sun is drawn as an
    // overexposed one looks: a hotter core (its own colour, half way to the
    // pale one of an overexposed sun), a redder rim, and a glare of its own
    // light around it. How low is read off its light (green against red), so
    // it holds on any planet and leaves a high sun as it was.
    vec3 sunLight = uSunRadiance * transmittance;
    float sunPeak = max(max(sunLight.r, sunLight.g), max(sunLight.b, 1e-12));
    float low = smoothstep(0.75, 0.45, sunLight.g / sunPeak);
    float angle = sqrt(max(2.0 * (1.0 - cosSun), 0.0));  // small angles
    float radius = sqrt(2.0 * (1.0 - uSunCos));
    float glare = SUN_GLARE * low * exp(-pow(angle / (SUN_GLARE_RADII * radius), 2.0));
    float rim = clamp(angle / radius, 0.0, 1.0);
    float hue = 1.0 + low * mix(SUN_CORE - 1.0, SUN_REDDEN, rim * rim);
    vec3 sun = sunPeak * pow(sunLight / sunPeak, vec3(hue)) * (disc + glare);
    if (atmoHitsGround(dir)) {
        background = vec3(0.0);  // the ground is in front (when not drawn yet)
        sun = vec3(0.0);
    }
    vec3 colour = (background * transmittance + sun + inscatter) * uExposure;
    vec3 display = aces_tonemap(colour);
    // The sun and the glow around it keep their hue. Per channel, the
    // tonemap flattens whatever is brighter than white toward white: the
    // disc (thousands of times brighter) saturated to white whatever its
    // light, and the glow around it paled. Scaled by their brightest channel
    // instead (display.glsl's keptHue), they show the colour of the
    // sunlight reaching the camera (the air's transmittance along this very
    // ray): orange at sunset, as the light they cast. Faded in above white,
    // so the sky away from the sun is as before. The sea's reflection
    // follows (it mirrors these display colours).
    float peak = displayPeak(colour);
    float keep = max(disc, smoothstep(KEEP_HUE_FROM, KEEP_HUE_TO, peak));
    if (keep > 0.0) {
        display = mix(display, keptHue(colour, peak), keep);
    }
    fragColor = vec4(display, 1.0);
}
