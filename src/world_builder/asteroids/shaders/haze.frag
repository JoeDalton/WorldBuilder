#version 330

// A field's haze: the rocks not drawn, integrated along the pixel's ray
// (world_builder.asteroids.haze, whose optical_depth is this quadrature in
// numpy). Drawn where the ray leaves the field's hull: the interval that
// ends there, from this camera's near plane on (the far camera of a split
// starts at the split: nearer, every rock is drawn) to where it leaves the
// field, past the far plane if it must.

#pragma include "../../atmosphere/shaders/display.glsl"

uniform mat4 p3d_ViewProjectionMatrix;
uniform vec3 uCamRender;
uniform mat3 uFieldRot;
uniform vec3 uCamField;
uniform vec3 uSunDir;          // toward the sun, render frame
uniform vec3 uSunColor;
uniform vec3 uAmbient;
uniform float uExposure;
uniform int uTonemap;
uniform float uPixelsPerRadian;
uniform float uMinPixels;      // the smallest diameter drawn (the budget's)
uniform float uFar;            // no rock drawn past it

uniform int uShape;            // 0 box, 1 sphere, 2 annulus
uniform vec3 uShapeSize;       // half sizes; radius; inner, outer, half thickness
uniform vec2 uEdges;           // edge, radial edge
uniform int uProfiled;
uniform float uProfile[256];   // an annulus's radial profile, inner to outer
uniform float uCrossSection[64];  // HazeModel.table, over log radius
uniform float uSizeLo;
uniform float uLogSpan;
uniform float uDensity;        // rocks per m3 at full profile
uniform float uCover;          // projected area over the bounding disc's
uniform float uDust;           // the dust's optical depth per metre
uniform vec3 uHazeAlbedo;      // linear

in vec3 vRay;
in vec3 vNormal;

out vec4 fragColor;

const int SAMPLES = 32;
const float FADE_MIDDLE = 1.125;
const float PI = 3.141592653589793;
const float INF = 1e30;

float smoothstep01(float t) {
    t = clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float lerpTable(float x, int n, int which) {
    float f = clamp(x, 0.0, 1.0) * float(n - 1);
    int i = min(int(floor(f)), n - 2);
    float a = which == 0 ? uCrossSection[i] : uProfile[i];
    float b = which == 0 ? uCrossSection[i + 1] : uProfile[i + 1];
    return a + (b - a) * (f - float(i));
}

// spec.Box/Sphere/Annulus.profile
float profile(vec3 p) {
    float e = uEdges.x;
    if (uShape == 0) {
        vec3 h = uShapeSize;
        if (e > 0.0) {
            vec3 d = (h - abs(p)) / e;
            return smoothstep01(d.x) * smoothstep01(d.y) * smoothstep01(d.z);
        }
        return all(lessThanEqual(abs(p), h)) ? 1.0 : 0.0;
    }
    if (uShape == 1) {
        float d = length(p);
        return e > 0.0 ? smoothstep01((uShapeSize.x - d) / e)
                       : (d <= uShapeSize.x ? 1.0 : 0.0);
    }
    float rho = length(p.xy);
    float inner = uShapeSize.x, outer = uShapeSize.y, h = uShapeSize.z;
    float vertical = e > 0.0 ? smoothstep01((h - abs(p.z)) / e)
                             : (abs(p.z) <= h ? 1.0 : 0.0);
    float re = uEdges.y;
    float radial = re > 0.0
        ? smoothstep01((rho - inner) / re) * smoothstep01((outer - rho) / re)
        : (rho >= inner && rho <= outer ? 1.0 : 0.0);
    if (uProfiled == 1)
        radial *= lerpTable((rho - inner) / (outer - inner), 256, 1);
    return vertical * radial;
}

// Where a ray is within radius of the z axis; false if never.
bool cylinder(vec3 o, vec3 d, float radius, out vec2 span) {
    float a = dot(d.xy, d.xy);
    float rho = length(o.xy);
    float c = (rho - radius) * (rho + radius);
    if (a < 1e-12) {
        span = vec2(-INF, INF);
        return c <= 0.0;
    }
    float b = dot(o.xy, d.xy);
    float disc = b * b - a * c;
    if (disc < 0.0) return false;
    float q = -(b + (b < 0.0 ? -1.0 : 1.0) * sqrt(disc));
    if (q == 0.0) {
        span = vec2(0.0);
        return true;
    }
    float t1 = q / a, t2 = c / q;
    span = vec2(min(t1, t2), max(t1, t2));
    return true;
}

bool slab(float o, float d, float h, out vec2 span) {
    if (abs(d) < 1e-12) {
        span = vec2(-INF, INF);
        return abs(o) <= h;
    }
    float t1 = (-h - o) / d, t2 = (h - o) / d;
    span = vec2(min(t1, t2), max(t1, t2));
    return true;
}

// haze.intervals: the one that ends nearest the hull point tFrag.
bool closing(vec3 o, vec3 d, float tFrag, out vec2 span) {
    if (uShape == 0) {
        vec2 sx, sy, sz;
        if (!slab(o.x, d.x, uShapeSize.x, sx) || !slab(o.y, d.y, uShapeSize.y, sy)
            || !slab(o.z, d.z, uShapeSize.z, sz))
            return false;
        span = vec2(max(max(sx.x, sy.x), max(sz.x, 0.0)), min(min(sx.y, sy.y), sz.y));
        return span.y > span.x;
    }
    if (uShape == 1) {
        float b = dot(o, d);
        float c = dot(o, o) - uShapeSize.x * uShapeSize.x;
        float disc = b * b - c;
        if (disc < 0.0) return false;
        float s = sqrt(disc);
        span = vec2(max(-b - s, 0.0), -b + s);
        return span.y > span.x;
    }
    vec2 z, outer, hole;
    if (!slab(o.z, d.z, uShapeSize.z, z) || !cylinder(o, d, uShapeSize.y, outer))
        return false;
    float lo = max(max(z.x, outer.x), 0.0), hi = min(z.y, outer.y);
    if (uShapeSize.x > 0.0 && cylinder(o, d, uShapeSize.x, hole)) {
        vec2 near = vec2(lo, min(hi, hole.x));
        vec2 far = vec2(max(lo, hole.y), hi);
        bool nearOk = near.y > near.x, farOk = far.y > far.x;
        if (nearOk && farOk)
            span = abs(near.y - tFrag) < abs(far.y - tFrag) ? near : far;
        else if (nearOk)
            span = near;
        else if (farOk)
            span = far;
        else
            return false;
        return true;
    }
    span = vec2(lo, hi);
    return hi > lo;
}

// HazeModel.extinction(r_cut(t))
float extinction(float t) {
    float cut = t > uFar ? INF : t * FADE_MIDDLE * uMinPixels / (2.0 * uPixelsPerRadian);
    float x = log(max(cut, uSizeLo) / uSizeLo) / uLogSpan;
    return uDensity * uCover * lerpTable(x, 64, 0) + uDust;
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

    vec2 span;
    if (!closing(o, dir, tFrag, span)) discard;
    float start = uDust > 0.0
        ? 0.0 : uSizeLo * 2.0 * uPixelsPerRadian / (FADE_MIDDLE * uMinPixels);
    float a = max(max(span.x, max(tNear, 0.0)), min(start, uFar));
    float b = span.y;
    if (b <= a) discard;

    float ratio = b / a;
    float tau = 0.0;
    float prev = a;
    for (int i = 1; i <= SAMPLES; ++i) {
        float t = a * pow(ratio, float(i) / float(SAMPLES));
        float mid = sqrt(prev * t);
        tau += profile(o + dir * mid) * extinction(mid) * (t - prev);
        prev = t;
    }
    float alpha = 1.0 - exp(-tau);
    if (alpha < 1.0 / 1024.0) discard;

    // Lambert spheres, lit by the sun and the sky light.
    vec3 L = normalize(uSunDir);
    float phaseAngle = acos(clamp(dot(L, -dirRender), -1.0, 1.0));
    float lambert = (sin(phaseAngle) + (PI - phaseAngle) * cos(phaseAngle)) / PI;
    vec3 C = uHazeAlbedo * (uSunColor * (2.0 / 3.0) * lambert + uAmbient) * uExposure;
    vec3 rgb = uTonemap == 1 ? aces_tonemap(C) : pow(clamp(C, 0.0, 1.0), vec3(1.0 / 2.2));
    fragColor = vec4(rgb, alpha);
}
