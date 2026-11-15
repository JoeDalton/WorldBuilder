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

// The clouds' optical depth from p toward dir (all the genera's: one light
// volume and one map pair hold their sum): the light volume's where it holds
// the point (in a sky band: itself; under or between them: where the sun's
// ray enters the next band), else the cloud shadow maps' total there, else
// none. 0 above the bands.
float godRayCloudDepth(vec3 p, vec3 dir) {
    float enter = 0.0;
    bool ahead = false;
    for (int band = 0; band < MAX_SKY_BANDS; ++band) {
        float chord;
        if (band < uSkyBandCount
                && shellChord(planetRadius + uSkyBands[band].x,
                              planetRadius + uSkyBands[band].y, p, dir, enter, chord)) {
            ahead = true;
            break;
        }
    }
    if (!ahead) {
        return 0.0;  // above the bands, or no chord
    }
    vec3 start = p + dir * enter;
    float keep = 0.0;
    float depth = 0.0;
    if (cloudLightOn != 0) {
        depth = cloudLightAt(cloudLight, cloudLightChart, start, keep).r;
    }
    if (keep >= 1.0) {
        return depth;
    }
    vec4 texel;
    float keepNear;
    float held = cloudMapsTexelWeightIn(cloudShadowMapOn, cloudShadowNear, cloudNearChart,
                                        cloudShadowFar, cloudFarChart, normalize(start),
                                        texel, keepNear);
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
    float depth = uCloudShadowLayer >= 0 ? godRayCloudDepth(p, dir) : 0.0;
    return visible * exp2(-depth);
}
