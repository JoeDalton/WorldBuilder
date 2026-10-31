// The light a sample of a planet's cloud sends toward the camera, shared by
// the billboards and the deck (cloud.frag) and the raymarch
// (cloud_march.frag). Include with #pragma include, after atmosphere.glsl and
// cloud_field.glsl. See docs/source/clouds.md, "Lighting".
uniform vec3  sunColor;
uniform vec3  skyColor;
uniform float sunBrightness;
uniform float skyStrength;
uniform float skyOcclusion;     // share of sky light reaching a slab's underside
// The far deck's lighting (noise.deck_light_table): x the sun's cosine at the
// cloud, from deckLightCosMin to 1; y the weather's coverage. r: the sun
// term's shadowing (with the powder), g: the shadowing alone.
uniform sampler2D deckLight;
uniform float deckLightCosMin;

// Darkens optically THIN cloud. Fed the LOCAL density over a fixed length: the
// sunward depth would invert it, and the quad's chord would tie it to quad size.
float powder(Field f, float density) {
    return 1.0 - exp2(-f.fieldExtinction * density * f.powderLength * 2.0);
}

// The sunlight, skylight and twilight on a cloud at p (planet frame) through
// the planet's air atmo (-1: none), per fragment: the billboards work the
// same out per vertex (cloud_sphere.vert).
void planetCloudLight(int atmo, vec3 p, out vec3 sunlight, out vec3 skylight,
                      out vec3 twilight) {
    sunlight = sunColor;
    skylight = skyColor;
    twilight = vec3(0.0);
    if (atmo >= 0) {
        // Through the planet's air: reddened low, fading out toward its
        // shadow (cloud_field.glsl's cloudSunTint).
        vec3 tint = cloudSunTint(atmo, p, sunDir);
        sunlight *= tint;
        twilight = sunColor * cloudTwilight(atmo, p, sunDir, tint);
        // The sky's light fades through twilight as the air's own fill
        // does (atmosphere.glsl), so clouds in the planet's shadow stay as
        // bright as the air around them.
        skylight *= smoothstep(-0.3, 0.1, dot(normalize(p), sunDir));
    }
}

// The radiance a sample of cloud at p sends toward the camera (V: from the
// cloud toward it), linear, before the air in front and the display
// transform: what an opaque cloud there would look like.
//   density      the field's density at p (cloudDensity)
//   shading      1 = its own self-shadowing and powder, 0 = their averages
//                over the clouds' columns (the deck's light table, for the
//                sun and the weather's coverage farCoverage), between: mixed
//   sunlight     the sun's colour as it reaches p (planetCloudLight)
//   skylight     the sky's
//   twilight     the lit air's above p once p is in the planet's shadow
//   weather      the weather at p: its fbm shift, the convective depth, the
//                anvil shelf's shift (weatherAt)
vec3 cloudRadianceAt(Field f, vec3 p, float density, vec3 V, float shading,
                     float farCoverage, vec3 sunlight, vec3 skylight,
                     vec3 twilight, vec3 weather) {
    if ((uCloudDebug & 1) != 0) shading = 0.0;
    float sunVisibility = shading > 0.0 ? exp2(-sunOpticalDepthAt(f, p, weather)) : 1.0;
    // What the sun term sees: the shadowing and the powder.
    float sunShadowing = sunVisibility * mix(1.0, powder(f, density), f.powderStrength);
    if (shading < 1.0) {
        // Far away: the averages of both over the clouds' columns, which
        // the aliasing per-pixel values would not give, for this sun and
        // coverage. They are what the clouds show close up, blurred, so
        // nothing changes colour closing in. A low sun marches along the
        // slab, through the clouds around: the terminator darkens gradually,
        // in the (reddened) sunlight's colour.
        vec2 size = vec2(textureSize(deckLight, 0));
        float cosSun = dot(normalize(p), sunDir);
        vec2 at = vec2((cosSun - deckLightCosMin) / (1.0 - deckLightCosMin), farCoverage);
        vec2 averaged = texture(deckLight, (clamp(at, 0.0, 1.0) * (size - 1.0) + 0.5) / size).rg;
        sunVisibility = mix(averaged.g, sunVisibility, shading);
        sunShadowing = mix(averaged.r, sunShadowing, shading);
    }
    // Skylight comes from above: occlude the SKY term toward the underside
    // only (of the cloud's own height, at its convective depth; over its
    // column's top, of the anvil shelf's).
    float h = slabHeight(f, heightOf(p));
    float local = h / topFraction(f, weather.y);
    if (local > 1.0 && f.anvil > 0.0) {
        local = (h - SHELF_BAND.x) / (SHELF_BAND.w - SHELF_BAND.x);
    }
    float skyReach = mix(skyOcclusion, 1.0, clamp(local, 0.0, 1.0));
    // Multiple-scattering fill: sky-coloured and NOT phase-weighted, or dusk
    // shadows turn orange and sunward shadow outshines side-lit cloud.
    float ambient = skyStrength * skyReach
                  + f.multipleScattering * sunBrightness * (1.0 - sunVisibility);

    float phase = phaseFunction(f, dot(sunDir, -V));
    // The lit air above a cloud in the planet's shadow (cloudTwilight): light
    // from the sky, so neither phase-weighted nor self-shadowed, and reaching
    // the underside as the sky's does. This is what spreads the terminator
    // over the clouds instead of a line from lit to dark.
    return sunlight * (sunBrightness * sunShadowing * phase)
         + skylight * ambient
         + twilight * (sunBrightness * TWILIGHT_GLOW * skyReach);
}

// With only the weather's shift (no convective depth, no shelf).
vec3 cloudRadianceShifted(Field f, vec3 p, float density, vec3 V, float shading,
                          float farCoverage, vec3 sunlight, vec3 skylight,
                          vec3 twilight, float shift) {
    return cloudRadianceAt(f, p, density, V, shading, farCoverage, sunlight,
                           skylight, twilight, vec3(shift, 0.0, -NO_CLOUD_LIFT));
}

vec3 cloudRadiance(Field f, vec3 p, float density, vec3 V, float shading,
                   float farCoverage, vec3 sunlight, vec3 skylight, vec3 twilight) {
    return cloudRadianceAt(f, p, density, V, shading, farCoverage, sunlight,
                           skylight, twilight, weatherAt(p));
}
