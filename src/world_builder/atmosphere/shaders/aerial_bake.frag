#version 330

// Bakes the aerial-perspective cells (atmosphere.glsl's aerialAlong): each
// cell's direction (its centre through the main camera's lens) marched from
// the camera to the end of its slice by atmosphereAlong, ATMO_MAX_STEPS
// samples (the eclipses, the rings and the planet's shadow with it); the
// in-scatter in the lower half, the transmittance in the upper.

// The air with the god rays: the clouds' and the mountains' shadows on it
// (god_rays.glsl), the light volume's readers with the clouds' field.
#define ATMO_GOD_RAYS
#define CLOUD_MAP_LIGHTING
#pragma include "atmosphere.glsl"
#pragma include "../../clouds/shaders/cloud_field.glsl"
#pragma include "god_rays.glsl"

// The main camera's clip space to camera-relative world points.
uniform mat4 uAerialFromClip;
uniform vec3 uSunDir;
out vec4 colour;

void main() {
    vec2 pixel = gl_FragCoord.xy;
    bool through = pixel.y >= 0.5 * AERIAL_ATLAS.y;
    if (through) {
        pixel.y -= 0.5 * AERIAL_ATLAS.y;
    }
    vec2 tile = floor(pixel / AERIAL_CELLS);
    float k = tile.y * 8.0 + tile.x;
    vec2 ndc = (pixel - tile * AERIAL_CELLS) / AERIAL_CELLS * 2.0 - 1.0;
    vec4 point = uAerialFromClip * vec4(ndc, 0.0, 1.0);
    vec3 dir = normalize(point.xyz / point.w);
    float end = (k + 1.0) / AERIAL_CELLS;
    float dist = uAerialRange.x + uAerialRange.y * end * end;
    vec3 transmittance, inscatter;
    atmoThroughGround = true;
    atmosphereAlong(dir, dist, uSunDir, ATMO_MAX_STEPS, transmittance, inscatter);
    colour = vec4(through ? transmittance : inscatter, 1.0);
}
