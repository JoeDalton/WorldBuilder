// God rays (RENDERING_PLAN.md C.5): the shadows of the clouds and the
// mountains on the air, taken by the bakes that march it per texel (the
// aerial-perspective cells; atmosphere.glsl's atmoGodRays in atmoSegment's
// direct sun). Include after atmosphere.glsl and the clouds' cloud_field.glsl
// (CLOUD_MAP_LIGHTING defined: the light volume's readers), with
// ATMO_GOD_RAYS defined before atmosphere.glsl.

// 1: the air takes the shadows (AtmosphereSettings.god_rays); 0: not.
uniform int uGodRays;
// The atmosphere whose planet the clouds and the mountains bound here
// belong to (-1: none).
uniform int uGodRaysAtmo;

// One family's clouds' optical depth from p toward dir: the light volume's
// where it holds the point (in the slab: itself; under it: where the sun's
// ray enters the base), else the cloud shadow maps' total there, else none.
// 0 above the slab.
float godRayCloudDepth(Field f, int mapsOn, sampler2D nearMap, vec4 nearChart[4],
                       sampler2D farMap, vec4 farChart[4], int lightOn,
                       sampler2D light, vec4 lightChart[4], vec3 p, vec3 dir) {
    vec3 start;
    float chord;
    if (!cloudShadowChord(f, p, dir, start, chord)) {
        return 0.0;  // above the slab, or no chord
    }
    float keep = 0.0;
    float depth = 0.0;
    if (lightOn != 0) {
        depth = cloudLightAt(light, lightChart, start, keep).r;
    }
    if (keep >= 1.0) {
        return depth;
    }
    vec4 texel;
    float keepNear;
    float held = cloudMapsTexelWeightIn(mapsOn, nearMap, nearChart, farMap, farChart,
                                        normalize(start), texel, keepNear);
    return mix(held * texel.r, depth, keep);
}

float atmoGodRays(int i, vec3 p, vec3 dir) {
    if (uGodRays == 0 || i != uGodRaysAtmo) {
        return 1.0;
    }
    float r = length(p);
    // The mountains: the point's altitude against the ceiling they set
    // toward the sun there (any point, not only the ground).
    float visible = terrainShadow(p / r, max(r - uAtmoShape[i].x, 0.0));
    float depth = 0.0;
    if (uCloudShadowLayer >= 0) {
        depth += godRayCloudDepth(getField(uCloudShadowLayer), cloudShadowMapOn,
                                  cloudShadowNear, cloudNearChart, cloudShadowFar,
                                  cloudFarChart, cloudLightOn, cloudLight,
                                  cloudLightChart, p, dir);
    }
    if (uCloudShadowOther >= 0) {
        depth += godRayCloudDepth(getField(uCloudShadowOther), otherShadowMapOn,
                                  otherShadowNear, otherNearChart, otherShadowFar,
                                  otherFarChart, otherLightOn, otherLight,
                                  otherLightChart, p, dir);
    }
    return visible * exp2(-depth);
}
