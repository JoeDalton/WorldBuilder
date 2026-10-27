#pragma once
// World shadows: what hides the sun from a point. Multiply the sunlight by
// each term (world_builder.shadows: the CPU twin, and the uniforms, bound
// with Shadows.bind(node, body)).
//
// atmosphere.glsl includes this, so every shader that sees through air has
// it. Positions are relative to the camera, render frame (its axes are the
// world's), metres.

#define MAX_SHADOW_BODIES 8

// Spherical bodies: xyz their centre minus the camera (float64 on the CPU),
// w their radius.
uniform vec4  uShadowBodies[MAX_SHADOW_BODIES];
uniform int   uShadowCount;
// Per body, a bit per body whose penumbra can reach what it holds (its air,
// its ground): set on the CPU for the current sun, mostly none.
uniform int   uShadowCasters[MAX_SHADOW_BODIES];
// The sun's angular radius, radians (the sky's disc is as wide).
uniform float uSunAngularRadius;
// The receiver's own body, which never eclipses it (-1: none).
uniform int   uShadowSelf;
// The most samples a ground pixel takes through the planet's clouds toward
// the sun (cloud_field.glsl's cloudShadowDepth), 0 = no cloud shadows: a
// quality knob (world_builder.shadows.ShadowSettings).
uniform int   uCloudShadowSteps;

const float SHADOW_PI = 3.14159265;
// The sky's light left in a total eclipse, as a share of its light by day:
// multiply sky light by mix(ECLIPSE_SKY_FLOOR, 1.0, visibility).
const float ECLIPSE_SKY_FLOOR = 0.15;

// The share of a uniform disc of radius sunR that a disc of radius occR
// covers, their centres sep apart (angles, taken as flat lengths): the
// overlap's area over the first disc's.
float discCoverage(float sunR, float occR, float sep) {
    if (sep >= sunR + occR) {
        return 0.0;
    }
    if (sep <= abs(sunR - occR)) {
        return occR >= sunR ? 1.0 : (occR * occR) / (sunR * sunR);
    }
    float a2 = sunR * sunR;
    float b2 = occR * occR;
    float s2 = sep * sep;
    float alpha = acos(clamp((s2 + a2 - b2) / (2.0 * sep * sunR), -1.0, 1.0));
    float beta = acos(clamp((s2 + b2 - a2) / (2.0 * sep * occR), -1.0, 1.0));
    float kite = 0.5 * sqrt(max((-sep + sunR + occR) * (sep + sunR - occR)
                                * (sep - sunR + occR) * (sep + sunR + occR), 0.0));
    return clamp((a2 * alpha + b2 * beta - kite) / (SHADOW_PI * a2), 0.0, 1.0);
}

// Bodies whose shadow can reach a point of body self (-1: any point), as
// bits: none, most of the time.
int eclipseCasters(int self) {
    return self >= 0 ? uShadowCasters[self] : (1 << uShadowCount) - 1;
}

// Whether any body's shadow can reach a point of body self: a uniform test,
// to keep eclipseVisibility out of a hot loop that does not need it.
bool eclipseCanReach(int self) {
    return eclipseCasters(self) != 0;
}

// The share of the sun's disc the bodies leave visible from a point rel
// (relative to the camera) toward the sun L (unit): 1 in full sun, 0 in an
// umbra. Body self is skipped (-1: none), and for a point of a body, only
// the bodies whose shadow can reach it are looked at (none, most of the
// time: nothing to do).
float eclipseVisibility(vec3 rel, vec3 L, int self) {
    int casters = eclipseCasters(self);
    if (casters == 0) {
        return 1.0;
    }
    float visible = 1.0;
    float sunTan = tan(uSunAngularRadius);
    for (int k = 0; k < uShadowCount; ++k) {
        if ((casters & (1 << k)) == 0) {
            continue;
        }
        vec3 c = uShadowBodies[k].xyz - rel;  // the body's centre, from the point
        float R = uShadowBodies[k].w;
        float d2 = dot(c, c);
        if (d2 <= R * R) {
            return 0.0;  // inside it
        }
        float along = dot(c, L);
        float across = length(c - L * along);
        // Clear of the sun when the angle between them is at least the sum
        // of their radii: tan(occ + sun) from the two tangents (exact, so
        // no edge), before any inverse trigonometry.
        float occTan = R * inversesqrt(d2 - R * R);
        float sumCos = 1.0 - occTan * sunTan;
        if (sumCos > 0.0 && across * sumCos >= along * (occTan + sunTan)) {
            continue;
        }
        float occR = asin(min(R * inversesqrt(d2), 1.0));
        visible *= 1.0 - discCoverage(uSunAngularRadius, occR, atan(across, along));
    }
    return visible;
}
