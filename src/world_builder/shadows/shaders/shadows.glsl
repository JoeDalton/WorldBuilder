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
// The mountains' shadow (world_builder.terrain.shadow_map.ShadowMap): by
// direction from the receiver's body's centre, in its own frame, r the
// lowest altitude the sun clears every occluder from (the ceiling) and g
// the distance to the occluder that sets it, both km; off (0) until baked.
uniform samplerCube uTerrainShadow;
uniform int   uTerrainShadowOn;
// The nearby ridges' (shadow_map.LocalShadowMap): the same over a square
// chart around the camera, metres, occluders nearer than the global map's
// reach. uLocalChart: its centre's up (xyz) and the radius, east and the
// node spacing, north and the nodes per side, and where it fades out
// (from x to y metres from its centre, along either axis).
uniform sampler2D uLocalShadow;
uniform int   uLocalShadowOn;
uniform vec4  uLocalChart[4];
// The most samples a ground pixel takes through the planet's clouds toward
// the sun (cloud_field.glsl's cloudShadowDepth), 0 = no cloud shadows: a
// quality knob (world_builder.shadows.ShadowSettings).
uniform int   uCloudShadowSteps;
// The objects' shadow map (world_builder.shadows.casters.CasterMap): their
// depth down the sun's rays, compared by the hardware; uCasterMatrix takes
// a position from the camera (render frame) to its texture and depth;
// uCasterBias: the normal offset (metres), the depth bias, a texel (uv).
// Off (0) with no caster.
uniform sampler2DShadow uCasterMap;
uniform int   uCasterOn;
uniform mat4  uCasterMatrix;
uniform vec4  uCasterBias;
// How far along the sun's ray a cloud billboard looks for the clouds that
// shade it, past its own march (cloud_field.glsl's cloudFarDepth), metres;
// 0 = no farther (ShadowSettings.cloud_far_reach).
uniform float uCloudFarReach;

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

// The share of the sun body self leaves a point of its own (over its
// ground, or on it), rel from the camera: its horizon, with the sun's
// penumbra. Its limb is a straight edge across the sun's disc (discCoverage
// cancels away in float32 for a disc half the sky wide): the sun's
// elevation against the horizon's dip, none at or under the sea-level
// radius (a crater's floor is not inside the body). For what no air
// lights by day and night (atmoSunlight does through air).
// Shadows.own_shadow on the CPU.
float ownShadow(vec3 rel, vec3 L, int self) {
    vec3 p = rel - uShadowBodies[self].xyz;  // from its centre
    float d = length(p);
    float R = uShadowBodies[self].w;
    float elevation = asin(clamp(dot(p, L) / d, -1.0, 1.0));
    float dip = atan(sqrt(max((d - R) * (d + R), 0.0)), R);
    // How deep the limb cuts into the sun's disc, in its radii: the
    // covered segment's share.
    float x = clamp(-(elevation + dip) / uSunAngularRadius, -1.0, 1.0);
    return 0.5 - (x * sqrt(1.0 - x * x) + asin(x)) / SHADOW_PI;
}

// The share of the sun the mountains leave a point of the body's frame,
// planetDir (unit) from its centre, altitude metres above sea level: the
// ray toward the sun clears the occluder that sets the ceiling by
// (altitude - ceiling) / distance radians, and the penumbra is a smoothstep
// of that over the sun's angular radius. shadow_map.terrain_visibility on
// the CPU.
float globalTerrainShadow(vec3 planetDir, float altitude) {
    if (uTerrainShadowOn == 0) {
        return 1.0;
    }
    vec2 ceiling = texture(uTerrainShadow, planetDir).rg * 1000.0;
    float margin = (altitude - ceiling.x) / max(ceiling.y, 1.0);
    return smoothstep(-uSunAngularRadius, uSunAngularRadius, margin);
}

// The same from the local map: 1 outside it, faded out toward its edges
// (the ground beyond them is unknown). Its chart is azimuthal equidistant
// (terrain.chart): the angle from its centre by atan, precise near it.
float localShadow(vec3 planetDir, float altitude) {
    if (uLocalShadowOn == 0) {
        return 1.0;
    }
    vec2 across = vec2(dot(planetDir, uLocalChart[1].xyz), dot(planetDir, uLocalChart[2].xyz));
    float sine = length(across);
    float angle = atan(sine, dot(planetDir, uLocalChart[0].xyz));
    vec2 xy = across * (sine > 1e-9 ? uLocalChart[0].w * angle / sine : uLocalChart[0].w);
    float keep = 1.0 - smoothstep(uLocalChart[3].x, uLocalChart[3].y,
                                  max(abs(xy.x), abs(xy.y)));
    if (keep <= 0.0) {
        return 1.0;
    }
    float size = uLocalChart[2].w;
    vec2 ceiling = texture(uLocalShadow, (xy / uLocalChart[1].w + 0.5 * size) / size).rg;
    float margin = (altitude - ceiling.x) / max(ceiling.y, 1.0);
    return mix(1.0, smoothstep(-uSunAngularRadius, uSunAngularRadius, margin), keep);
}

// The share of the sun the objects (the casters) leave a point rel (from
// the camera, render frame) whose surface faces N, toward L: 1 outside
// their map or nearer the sun than it. Pushed out along its normal as it
// turns from the sun, and its depth toward the sun, against acne; 3x3 of
// the hardware's bilinear comparisons, a soft edge a few texels wide.
float casterShadow(vec3 rel, vec3 N, vec3 L) {
    if (uCasterOn == 0) {
        return 1.0;
    }
    float slope = 1.0 - clamp(dot(N, L), 0.0, 1.0);
    vec3 s = (uCasterMatrix * vec4(rel + N * (uCasterBias.x * slope), 1.0)).xyz;
    if (any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0)))) {
        return 1.0;
    }
    // Past the casters' far side: as at it (behind every one).
    float ref = min(s.z - uCasterBias.y, 1.0);
    if (ref <= 0.0) {
        return 1.0;
    }
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            lit += texture(uCasterMap, vec3(s.xy + vec2(x, y) * uCasterBias.z, ref));
        }
    }
    return lit / 9.0;
}

// Both: their occluders never overlap (the local map's are nearer than
// the global map's reach), and the higher horizon of the two hides the
// sun: the lesser share.
float terrainShadow(vec3 planetDir, float altitude) {
    return min(globalTerrainShadow(planetDir, altitude), localShadow(planetDir, altitude));
}
