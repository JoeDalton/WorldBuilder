#version 330

// Bakes the sky-view table (sky_view.glsl): each texel's direction marched
// through every atmosphere (atmosphereAlong, ATMO_MAX_STEPS samples: the
// eclipses, the rings, the planet's shadow as the sky's own march), its
// in-scatter in the lower rows, its transmittance in the upper.

// The air with the god rays: the clouds' and the mountains' shadows on it
// (god_rays.glsl), across the sky as in front of the ground.
#define ATMO_GOD_RAYS
#define CLOUD_MAP_LIGHTING
#pragma include "atmosphere.glsl"
#pragma include "../../clouds/shaders/cloud_field.glsl"
#pragma include "god_rays.glsl"
#pragma include "sky_view.glsl"

uniform vec3 uSunDir;
out vec4 colour;

void main() {
    int a = skyViewAtmosphere();
    if (a < 0) {
        colour = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 pixel = gl_FragCoord.xy;
    bool through = pixel.y >= SKY_VIEW_SIZE.y;
    if (through) {
        pixel.y -= SKY_VIEW_SIZE.y;
    }
    vec3 dir = skyViewDirection(a, pixel / SKY_VIEW_SIZE, uSunDir);
    vec3 transmittance, inscatter;
    atmosphereAlong(dir, 1e30, uSunDir, ATMO_MAX_STEPS, transmittance, inscatter);
    colour = vec4(through ? transmittance : inscatter, 1.0);
}
