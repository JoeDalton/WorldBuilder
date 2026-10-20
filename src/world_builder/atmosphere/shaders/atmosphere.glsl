// Single-scattering atmospheres: aerial perspective, sky colour and sunlight.
//
// Include with #pragma include (relative to the including shader). Bind the
// uniforms with world_builder.atmosphere.Atmospheres.bind(node).
//
// Each atmosphere is an exponential Rayleigh + Mie medium around a sphere. A
// view ray is marched in ATMO_STEPS samples through each atmosphere it
// crosses; at every sample, the optical depth toward the sun is the analytic
// Chapman function (no inner march). The planet's shadow is found
// analytically along the ray, and each sample counts the lit share of its
// stretch, so the shadow's edge moves smoothly (all-or-nothing samples drew
// hard bands across the dusk sky). Single scattering, plus a flat share of
// multiply scattered light that fades through twilight (see docs: precomputed
// multiple scattering, Bruneton). Lengths in metres, positions relative to
// each planet's centre.

#define MAX_ATMOSPHERES 2
#define ATMO_STEPS 8

uniform int  uAtmoCount;
// xyz: camera relative to the planet centre; w: |cam|^2 - top^2 (float64 on
// the CPU: as a float32 difference of squares it would be noise).
uniform vec4 uAtmoCam[MAX_ATMOSPHERES];
// x: ground radius, y: top radius, z: Rayleigh scale height, w: Mie scale
// height.
uniform vec4 uAtmoShape[MAX_ATMOSPHERES];
// rgb: Rayleigh scattering per metre at the ground; a: Mie scattering.
uniform vec4 uAtmoScatter[MAX_ATMOSPHERES];
// x: Mie extinction per metre, y: Mie anisotropy g, z: sun intensity,
// w: |cam|^2 - ground^2 (float64 on the CPU).
uniform vec4 uAtmoMie[MAX_ATMOSPHERES];
// x: multiply scattered light, as a share of direct sunlight.
uniform vec4 uAtmoExtra[MAX_ATMOSPHERES];

const float ATMO_PI = 3.14159265;

// exp(y^2) erfc(y) for y >= 0, within 0.33 %.
float atmoErfcx(float y) {
    return 1.0 / (1.1635781 * y + sqrt(0.3707296 * y * y + 1.0));
}

// Chapman function for a ray leaving radius x (in scale heights) at an angle
// whose cosine is >= 0: optical depth / (scale height * local density).
float atmoChapman(float x, float cosChi) {
    return sqrt(1.5707963 * x) * atmoErfcx(sqrt(0.5 * x) * cosChi);
}

// Optical depth (density-weighted metres) from radius r toward a direction
// at cosine cosChi from the vertical, through a layer of scale height H over
// a ground of radius R. Huge if the planet is in the way.
float atmoDepthToward(float r, float cosChi, float R, float H) {
    float density = exp(-(r - R) / H);
    if (cosChi >= 0.0) {
        return H * density * atmoChapman(r / H, cosChi);
    }
    float sinChi = sqrt(max(0.0, 1.0 - cosChi * cosChi));
    float tangent = r * sinChi;  // closest approach to the centre
    if (tangent < R) {
        return 1e9;
    }
    // Both sides of the tangent point, minus the side behind the start.
    return H * (2.0 * exp(-(tangent - R) / H) * atmoChapman(tangent / H, 0.0)
                - density * atmoChapman(r / H, -cosChi));
}

// Entry and exit distances along a ray from o (relative to the centre) with
// |o|^2 - radius^2 = c; false if it misses.
bool atmoSphere(vec3 o, vec3 dir, float c, out float t0, out float t1) {
    float b = dot(o, dir);
    float disc = b * b - c;
    if (disc <= 0.0) {
        return false;
    }
    float q = b >= 0.0 ? -b - sqrt(disc) : -b + sqrt(disc);  // stable roots
    t0 = q;
    t1 = c / q;
    if (t0 > t1) {
        float t = t0; t0 = t1; t1 = t;
    }
    return true;
}

// Where a ray from o (relative to the centre) is in the planet's shadow: a
// cylinder of radius R along the sun's axis, behind the planet. Returns
// [s0, s1], empty when s0 >= s1.
vec2 atmoShadow(vec3 o, vec3 dir, vec3 sunDir, float R) {
    vec3 op = o - sunDir * dot(o, sunDir);
    vec3 dp = dir - sunDir * dot(dir, sunDir);
    float a = dot(dp, dp);
    float b = dot(op, dp);
    float c = dot(op, op) - R * R;
    vec2 s;
    if (a < 1e-10) {  // along the axis: in it everywhere or nowhere
        s = c < 0.0 ? vec2(-1e30, 1e30) : vec2(1e30, -1e30);
    } else {
        float disc = b * b - a * c;
        if (disc <= 0.0) {
            return vec2(1e30, -1e30);
        }
        float root = sqrt(disc);
        s = vec2(-b - root, -b + root) / a;
    }
    float along = dot(dir, sunDir);
    float start = dot(o, sunDir);
    if (abs(along) < 1e-9) {
        if (start >= 0.0) {
            s = vec2(1e30, -1e30);
        }
    } else if (along > 0.0) {
        s.y = min(s.y, -start / along);
    } else {
        s.x = max(s.x, -start / along);
    }
    return s;
}

// One atmosphere's effect on the segment [0, tmax] of a view ray: what the
// background is multiplied by, and the light scattered toward the eye. near is
// where the segment starts (for ordering several atmospheres).
void atmoSegment(int i, vec3 dir, float tmax, vec3 sunDir, int steps,
                 out vec3 transmittance, out vec3 inscatter, out float near) {
    transmittance = vec3(1.0);
    inscatter = vec3(0.0);
    near = 1e30;
    vec3 o = uAtmoCam[i].xyz;
    vec4 shape = uAtmoShape[i];
    float t0, t1;
    if (!atmoSphere(o, dir, uAtmoCam[i].w, t0, t1) || t1 <= 0.0) {
        return;
    }
    // Rays that reach the ground stop there (the sky below the horizon, and
    // anything drawn past it on the far side).
    float g0, g1;
    if (atmoSphere(o, dir, uAtmoMie[i].w, g0, g1) && g1 > 0.0 && g0 > 0.0) {
        tmax = min(tmax, g0);
    }
    t0 = max(t0, 0.0);
    t1 = min(t1, tmax);
    if (t1 <= t0) {
        return;
    }
    near = t0;
    float R = shape.x;
    float Hr = shape.z;
    float Hm = shape.w;
    vec3 betaR = uAtmoScatter[i].rgb;
    float betaM = uAtmoScatter[i].a;
    float betaMExt = uAtmoMie[i].x;

    float ds = (t1 - t0) / float(steps);
    vec2 shadow = atmoShadow(o, dir, sunDir, R);
    float multiple = uAtmoExtra[i].x;
    float depthR = 0.0;
    float depthM = 0.0;
    vec3 sumR = vec3(0.0);
    vec3 sumM = vec3(0.0);
    for (int k = 0; k < steps; ++k) {
        float ta = t0 + float(k) * ds;
        float tc = ta + 0.5 * ds;
        vec3 p = o + dir * tc;
        float r = length(p);
        float densityR = exp(-(r - R) / Hr) * ds;
        float densityM = exp(-(r - R) / Hm) * ds;
        depthR += 0.5 * densityR;
        depthM += 0.5 * densityM;
        // The lit share of this sample's stretch, and a lit point in it to
        // take the sun's depth at (just outside the shadow if the centre is
        // in it: there the sunlight grazes the ground, deep red, finite).
        float shaded = max(0.0, min(ta + ds, shadow.y) - max(ta, shadow.x));
        float lit = 1.0 - shaded / ds;
        float ts = tc;
        if (tc > shadow.x && tc < shadow.y) {
            ts = tc - shadow.x < shadow.y - tc ? shadow.x - 50.0 : shadow.y + 50.0;
        }
        vec3 ps = o + dir * ts;
        float rs = length(ps);
        float cosSun = dot(ps, sunDir) / rs;
        vec3 direct = vec3(0.0);
        if (lit > 0.0) {
            float sunR = atmoDepthToward(rs, cosSun, R, Hr);
            float sunM = atmoDepthToward(rs, cosSun, R, Hm);
            direct = exp(-(betaR * (depthR + sunR) + betaMExt * (depthM + sunM))) * lit;
        }
        // Light scattered more than once: a share of the sunlight, wherever
        // the sky above is lit (fading through twilight), seen through the
        // air back to the camera.
        float twilight = smoothstep(-0.3, 0.1, dot(p, sunDir) / r);
        vec3 fill = exp(-(betaR * depthR + betaMExt * depthM)) * multiple * twilight;
        sumR += (direct + fill) * densityR;
        sumM += (direct + fill) * densityM;
        depthR += 0.5 * densityR;
        depthM += 0.5 * densityM;
    }
    float mu = dot(dir, sunDir);
    float phaseR = 3.0 / (16.0 * ATMO_PI) * (1.0 + mu * mu);
    float g = uAtmoMie[i].y;
    float g2 = g * g;
    float phaseM = 3.0 / (8.0 * ATMO_PI) * (1.0 - g2) * (1.0 + mu * mu)
                 / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
    inscatter = uAtmoMie[i].z * (sumR * betaR * phaseR + sumM * betaM * phaseM);
    transmittance = exp(-(betaR * depthR + betaMExt * depthM));
}

// Whether a view ray from the camera meets the ground of a planet with an
// atmosphere (the stars and the sun are behind it).
bool atmoHitsGround(vec3 dir) {
    for (int i = 0; i < MAX_ATMOSPHERES; ++i) {
        float g0, g1;
        if (i < uAtmoCount
            && atmoSphere(uAtmoCam[i].xyz, dir, uAtmoMie[i].w, g0, g1)
            && g0 > 0.0) {
            return true;
        }
    }
    return false;
}

// What every atmosphere along a view ray does to whatever lies at distance
// dist in direction dir: seen = background * transmittance + inscatter.
// Atmospheres are composed far to near (they normally do not overlap).
// steps: samples per atmosphere (ATMO_STEPS per pixel; fewer per vertex).
void atmosphereAlong(vec3 dir, float dist, vec3 sunDir, int steps,
                     out vec3 transmittance, out vec3 inscatter) {
    transmittance = vec3(1.0);
    inscatter = vec3(0.0);
    if (uAtmoCount <= 0) {
        return;
    }
    vec3 t[MAX_ATMOSPHERES];
    vec3 s[MAX_ATMOSPHERES];
    float near[MAX_ATMOSPHERES];
    for (int i = 0; i < MAX_ATMOSPHERES; ++i) {
        if (i < uAtmoCount) {
            atmoSegment(i, dir, dist, sunDir, steps, t[i], s[i], near[i]);
        } else {
            t[i] = vec3(1.0);
            s[i] = vec3(0.0);
            near[i] = 1e30;
        }
    }
    int first = near[0] < near[1] ? 0 : 1;  // the nearer one
    int second = 1 - first;
    transmittance = t[first] * t[second];
    inscatter = s[first] + t[first] * s[second];
}

void atmosphereAlong(vec3 dir, float dist, vec3 sunDir,
                     out vec3 transmittance, out vec3 inscatter) {
    atmosphereAlong(dir, dist, sunDir, ATMO_STEPS, transmittance, inscatter);
}

// The colour seen through every atmosphere along a view ray: background
// (linear radiance) at distance dist in direction dir from the camera.
vec3 applyAtmospheres(vec3 background, vec3 dir, float dist, vec3 sunDir) {
    if (uAtmoCount <= 0) {
        return background;
    }
    vec3 transmittance[MAX_ATMOSPHERES];
    vec3 inscatter[MAX_ATMOSPHERES];
    float near[MAX_ATMOSPHERES];
    for (int i = 0; i < MAX_ATMOSPHERES; ++i) {
        if (i < uAtmoCount) {
            atmoSegment(i, dir, dist, sunDir, ATMO_STEPS, transmittance[i], inscatter[i],
                        near[i]);
        } else {
            transmittance[i] = vec3(1.0);
            inscatter[i] = vec3(0.0);
            near[i] = 1e30;
        }
    }
    vec3 colour = background;
    if (near[0] < near[1]) {  // the nearer atmosphere goes on last
        colour = colour * transmittance[1] + inscatter[1];
        colour = colour * transmittance[0] + inscatter[0];
    } else {
        colour = colour * transmittance[0] + inscatter[0];
        colour = colour * transmittance[1] + inscatter[1];
    }
    return colour;
}

// Sunlight reaching a point (relative to atmosphere i's centre) through that
// atmosphere: 1 above it, reddened near the terminator, 0 in the planet's
// shadow. Multiply the sun's colour by it.
vec3 atmoSunlight(int i, vec3 p, vec3 sunDir) {
    vec4 shape = uAtmoShape[i];
    float r = max(length(p), shape.x);
    float cosSun = dot(p, sunDir) / r;
    float depthR = atmoDepthToward(r, cosSun, shape.x, shape.z);
    float depthM = atmoDepthToward(r, cosSun, shape.x, shape.w);
    return exp(-(uAtmoScatter[i].rgb * depthR + uAtmoMie[i].x * depthM));
}
