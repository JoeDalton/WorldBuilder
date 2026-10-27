// The sunlight reaching a game's model, per pixel, in every world shadow
// (world_builder.objects.ModelLighting binds its inputs): the model's
// body's air, the other bodies' eclipses (an airless body's own horizon),
// its clouds and its mountains. Include once per shader with
// #pragma include, after atmosphere.glsl (and shadows.glsl with it) and
// cloud_field.glsl.

uniform vec3 uSunColor;       // sunlight above any air, linear
uniform int  uOwnAtmosphere;  // the body's atmosphere index, -1 = airless
uniform mat3 uPlanetAxes;     // the body's own axes to the render frame
                              // (Terrain.rotation): its shadow map's
// The body's clouds' layer in layerParams whose shadow falls here, -1 none
// (SphericalClouds.bind_shadow_inputs).
uniform int  uCloudShadowLayer;

// The body's vertical at a point rel (from the camera, render frame), or
// fallback where there is no body (open space).
vec3 modelUp(vec3 rel, vec3 fallback) {
    if (uShadowSelf < 0) {
        return fallback;
    }
    return normalize(rel - uShadowBodies[uShadowSelf].xyz);
}

// The sunlight at a point rel (from the camera, render frame), toward L
// (unit), footprint the metres a pixel spans there (the clouds' shadow
// fades to its mean over several clouds). World.lights_at's direct light
// on the CPU (an airless body's own horizon: Shadows.own_shadow).
vec3 modelSunlight(vec3 rel, vec3 L, float footprint) {
    int self = uShadowSelf;
    vec3 sunlight = uSunColor * eclipseVisibility(rel, L, self);
    if (self < 0) {
        return sunlight;  // open space: the other bodies alone
    }
    vec3 planet = rel - uShadowBodies[self].xyz;  // from its centre, world axes
    if (uOwnAtmosphere >= 0) {
        // Its air reddens the sun, and hides it past its terminator.
        sunlight *= atmoSunlight(uOwnAtmosphere, planet, L);
        if (uCloudShadowLayer >= 0 && uCloudShadowSteps > 0) {
            sunlight *= cloudShadow(getField(uCloudShadowLayer), planet, L,
                                    uCloudShadowSteps, footprint);
        }
    } else {
        sunlight *= ownShadow(rel, L, self);
    }
    float altitude = length(planet) - uShadowBodies[self].w;
    return sunlight * terrainShadow(normalize(transpose(uPlanetAxes) * planet), altitude);
}
