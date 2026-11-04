#version 330

// A field's haze: the rocks not drawn, integrated along the pixel's ray
// (haze.glsl; world_builder.asteroids.haze.optical_depth is the same
// quadrature in numpy). Drawn where the ray leaves the field's hull: the
// interval that ends there, from this camera's near plane on (the far
// camera of a split starts at the split) to where it leaves the field, past
// the far plane if it must. The bodies' shadows across the field (a
// planet's on its ring) dim each sample's sunlight.

#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "../../shadows/shaders/shadows.glsl"

uniform mat4 p3d_ViewProjectionMatrix;
uniform vec3 uCamRender;
uniform mat3 uFieldRot;
uniform vec3 uSunDir;          // toward the sun, render frame
uniform vec3 uSunColor;
uniform vec3 uAmbient;
uniform float uExposure;
uniform int uTonemap;
// The pass: 1 what lies before the sphere of the planet's clouds' tops
// (drawn after the clouds), 2 what lies beyond it (before them); its
// radius from the field's origin, 0: no clouds (all in pass 1).
uniform int uHazePass;
uniform float uCloudTop;

#pragma include "haze.glsl"

in vec3 vRay;
in vec3 vNormal;

out vec4 fragColor;

const int SAMPLES = 32;
// The bodies' shadows are taken at this many points along a ray, each for
// its share of the samples: a shadow's edge is tens of km wide at a ring
// (the sun's angular size times the planet's distance), and taking them
// per sample cost up to 14 ms with a ring in a planet's shadow.
const int LIT_GROUPS = 4;

// hazeDepth (haze.glsl), and the share of it in the sun: each sample's
// depth weighted by what the bodies leave of the sun there (dRender: the
// ray, render frame, for the samples relative to the camera).
float hazeDepthLit(vec3 o, vec3 d, vec3 dRender, float a, float b, int samples,
                   out float lit) {
    lit = 1.0;
    a = max(a, uHazeStart);
    if (b <= a) return 0.0;
    a = max(a, 1e-3);
    vec3 L = normalize(uSunDir);
    float ratio = b / a;
    float tau = 0.0;
    float sunlit = 0.0;
    float prev = a;
    int group = max(samples / LIT_GROUPS, 1);
    float shade = 1.0;
    for (int i = 1; i <= samples; ++i) {
        if ((i - 1) % group == 0) {
            // At the middle of this group of samples (geometric, as they).
            float middle = float(i - 1) + 0.5 * float(group);
            shade = eclipseVisibility(dRender * (a * pow(ratio, middle / float(samples))),
                                      L, -1);
        }
        float t = a * pow(ratio, float(i) / float(samples));
        float mid = sqrt(prev * t);
        float dtau = hazeProfile(o + d * mid) * hazeMissing(mid) * (t - prev);
        tau += dtau;
        sunlit += dtau * shade;
        prev = t;
    }
    lit = tau > 0.0 ? sunlit / tau : 1.0;
    return tau;
}

void main() {
    vec3 dir = normalize(vRay);
    if (dot(dir, vNormal) <= 0.0) discard;  // where rays enter: nothing to close
    float tFrag = length(vRay);
    vec3 o = uCamField;

    // Where this camera's near plane cuts the ray.
    vec3 dirRender = uFieldRot * dir;
    vec4 A = p3d_ViewProjectionMatrix * vec4(uCamRender, 1.0);
    vec4 B = p3d_ViewProjectionMatrix * vec4(dirRender, 0.0);
    float tNear = -(A.z + A.w) / (B.z + B.w);

    // The interval this hull point closes: the one ending nearest it.
    vec2 s0, s1;
    int n = hazeIntervals(o, dir, s0, s1);
    if (n == 0) discard;
    vec2 span = (n == 2 && abs(s1.y - tFrag) < abs(s0.y - tFrag)) ? s1 : s0;

    // Where the ray enters the clouds' tops: from under them, at once.
    float split = 1e30;
    if (uCloudTop > 0.0) {
        float r = length(o);
        float c = (r - uCloudTop) * (r + uCloudTop);
        float b = dot(o, dir);
        if (c <= 0.0) {
            split = 0.0;
        } else if (b < 0.0 && b * b >= c) {
            split = -b - sqrt(b * b - c);
        }
    }
    float a = max(span.x, max(tNear, 0.0));
    float e = span.y;
    if (uHazePass == 2) {
        a = max(a, split);
    } else {
        e = min(e, split);
    }
    if (e <= a) discard;
    float lit;
    float tau = hazeDepthLit(o, dir, dirRender, a, e, SAMPLES, lit);
    float alpha = 1.0 - exp(-tau);
    if (alpha < 1.0 / 1024.0) discard;
    fragColor = vec4(hazeColourLit(dirRender, lit), alpha);
}
