// A field's haze (world_builder.asteroids.haze): the rocks not drawn along
// a ray, as an optical depth and a colour. What is missing per metre is
// tabulated each frame over the distance (FieldHaze.update). Shared by the haze's own pass
// (haze.frag: where a ray leaves the field) and the rocks (asteroid.frag:
// the haze between the camera and a rock veils it). Include once per
// shader, after display.glsl and the declarations of uSunDir, uSunColor,
// uAmbient, uExposure, uTonemap and uFieldRot. Every name here starts with
// haze (or HAZE_).

uniform vec3 uCamField;          // the camera, field frame
uniform float uPixelsPerRadian;
uniform float uMinPixels;        // the smallest diameter drawn (the budget's)

uniform int uShape;              // 0 box, 1 sphere, 2 annulus
uniform vec3 uShapeSize;         // half sizes; radius; inner, outer, half thickness
uniform vec2 uEdges;             // edge, radial edge
uniform int uProfiled;
uniform float uProfile[256];     // an annulus's radial profile, inner to outer
uniform float uMissing[256];     // HazeModel.missing this frame, per metre,
uniform float uMissingT0;        // over log distance from uMissingT0 ...
uniform float uMissingLogSpan;   // ... spanning this (natural log)
uniform float uHazeStart;        // HazeModel.start this frame
uniform vec3 uHazeAlbedo;        // linear

const float HAZE_PI = 3.141592653589793;
const float HAZE_INF = 1e30;

float hazeSmooth(float t) {
    t = clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// spec.Box/Sphere/Annulus.profile
float hazeProfile(vec3 p) {
    float e = uEdges.x;
    if (uShape == 0) {
        vec3 h = uShapeSize;
        if (e > 0.0) {
            vec3 d = (h - abs(p)) / e;
            return hazeSmooth(d.x) * hazeSmooth(d.y) * hazeSmooth(d.z);
        }
        return all(lessThanEqual(abs(p), h)) ? 1.0 : 0.0;
    }
    if (uShape == 1) {
        float d = length(p);
        return e > 0.0 ? hazeSmooth((uShapeSize.x - d) / e)
                       : (d <= uShapeSize.x ? 1.0 : 0.0);
    }
    float rho = length(p.xy);
    float inner = uShapeSize.x, outer = uShapeSize.y, h = uShapeSize.z;
    float vertical = e > 0.0 ? hazeSmooth((h - abs(p.z)) / e)
                             : (abs(p.z) <= h ? 1.0 : 0.0);
    float re = uEdges.y;
    float radial = re > 0.0
        ? hazeSmooth((rho - inner) / re) * hazeSmooth((outer - rho) / re)
        : (rho >= inner && rho <= outer ? 1.0 : 0.0);
    if (uProfiled == 1) {
        float x = clamp((rho - inner) / (outer - inner), 0.0, 1.0) * 255.0;
        int i = min(int(floor(x)), 254);
        radial *= mix(uProfile[i], uProfile[i + 1], x - float(i));
    }
    return vertical * radial;
}

// Where a ray is within radius of the z axis; false if never.
bool hazeCylinder(vec3 o, vec3 d, float radius, out vec2 span) {
    float a = dot(d.xy, d.xy);
    float rho = length(o.xy);
    float c = (rho - radius) * (rho + radius);
    if (a < 1e-12) {
        span = vec2(-HAZE_INF, HAZE_INF);
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

bool hazeSlab(float o, float d, float h, out vec2 span) {
    if (abs(d) < 1e-12) {
        span = vec2(-HAZE_INF, HAZE_INF);
        return abs(o) <= h;
    }
    float t1 = (-h - o) / d, t2 = (h - o) / d;
    span = vec2(min(t1, t2), max(t1, t2));
    return true;
}

// haze.intervals: where the ray (unit d) is in the shape's bounds, ahead of
// o; returns how many (0-2).
int hazeIntervals(vec3 o, vec3 d, out vec2 s0, out vec2 s1) {
    s0 = s1 = vec2(0.0);
    if (uShape == 0) {
        vec2 sx, sy, sz;
        if (!hazeSlab(o.x, d.x, uShapeSize.x, sx) || !hazeSlab(o.y, d.y, uShapeSize.y, sy)
            || !hazeSlab(o.z, d.z, uShapeSize.z, sz))
            return 0;
        s0 = vec2(max(max(sx.x, sy.x), max(sz.x, 0.0)), min(min(sx.y, sy.y), sz.y));
        return s0.y > s0.x ? 1 : 0;
    }
    if (uShape == 1) {
        float b = dot(o, d);
        float c = dot(o, o) - uShapeSize.x * uShapeSize.x;
        float disc = b * b - c;
        if (disc < 0.0) return 0;
        float s = sqrt(disc);
        s0 = vec2(max(-b - s, 0.0), -b + s);
        return s0.y > s0.x ? 1 : 0;
    }
    vec2 z, outer, hole;
    if (!hazeSlab(o.z, d.z, uShapeSize.z, z) || !hazeCylinder(o, d, uShapeSize.y, outer))
        return 0;
    float lo = max(max(z.x, outer.x), 0.0), hi = min(z.y, outer.y);
    if (uShapeSize.x > 0.0 && hazeCylinder(o, d, uShapeSize.x, hole)) {
        vec2 near = vec2(lo, min(hi, hole.x));
        vec2 far = vec2(max(lo, hole.y), hi);
        int n = 0;
        if (near.y > near.x) {
            s0 = near;
            n = 1;
        }
        if (far.y > far.x) {
            if (n == 0) s0 = far; else s1 = far;
            n += 1;
        }
        return n;
    }
    s0 = vec2(lo, hi);
    return hi > lo ? 1 : 0;
}

// HazeModel.missing at a distance t, from this frame's table (FieldHaze):
// each class's rocks too small to be drawn, those its limit fades out, the
// dust; per metre at full profile.
float hazeMissing(float t) {
    float x = clamp(log(max(t, uMissingT0) / uMissingT0) / uMissingLogSpan, 0.0, 1.0) * 255.0;
    int i = min(int(floor(x)), 254);
    return mix(uMissing[i], uMissing[i + 1], x - float(i));
}

// haze.optical_depth's quadrature from a to b: midpoints spaced
// geometrically.
float hazeDepth(vec3 o, vec3 d, float a, float b, int samples) {
    a = max(a, uHazeStart);
    if (b <= a) return 0.0;
    a = max(a, 1e-3);
    float ratio = b / a;
    float tau = 0.0;
    float prev = a;
    for (int i = 1; i <= samples; ++i) {
        float t = a * pow(ratio, float(i) / float(samples));
        float mid = sqrt(prev * t);
        tau += hazeProfile(o + d * mid) * hazeMissing(mid) * (t - prev);
        prev = t;
    }
    return tau;
}

// The haze's colour seen along dirRender (render frame), for display: the
// rocks as Lambert spheres lit by the sun and the sky light.
vec3 hazeColour(vec3 dirRender) {
    vec3 L = normalize(uSunDir);
    float phaseAngle = acos(clamp(dot(L, -dirRender), -1.0, 1.0));
    float lambert = (sin(phaseAngle) + (HAZE_PI - phaseAngle) * cos(phaseAngle)) / HAZE_PI;
    vec3 C = uHazeAlbedo * (uSunColor * (2.0 / 3.0) * lambert + uAmbient) * uExposure;
    return uTonemap == 1 ? aces_tonemap(C) : pow(clamp(C, 0.0, 1.0), vec3(1.0 / 2.2));
}
