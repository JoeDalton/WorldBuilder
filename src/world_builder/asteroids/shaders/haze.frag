#version 330

// A field's haze: the rocks not drawn, integrated along the pixel's ray
// (haze.glsl; world_builder.asteroids.haze.optical_depth is the same
// quadrature in numpy). Drawn where the ray leaves the field's hull: the
// interval that ends there, from this camera's near plane on (the far
// camera of a split starts at the split) to where it leaves the field, past
// the far plane if it must.

#pragma include "../../atmosphere/shaders/display.glsl"

uniform mat4 p3d_ViewProjectionMatrix;
uniform vec3 uCamRender;
uniform mat3 uFieldRot;
uniform vec3 uSunDir;          // toward the sun, render frame
uniform vec3 uSunColor;
uniform vec3 uAmbient;
uniform float uExposure;
uniform int uTonemap;

#pragma include "haze.glsl"

in vec3 vRay;
in vec3 vNormal;

out vec4 fragColor;

const int SAMPLES = 32;

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
    float tau = hazeDepth(o, dir, max(span.x, max(tNear, 0.0)), span.y, SAMPLES);
    float alpha = 1.0 - exp(-tau);
    if (alpha < 1.0 / 1024.0) discard;
    fragColor = vec4(hazeColour(dirRender), alpha);
}
