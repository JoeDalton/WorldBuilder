// The light a sample of a planet's cloud sends toward the camera, shared by
// the deck (cloud.frag) and the raymarch (cloud_march.frag); and the thin
// slab's (thinSlabRadiance), the march's high layer's. Include with
// #pragma include, after atmosphere.glsl and cloud_field.glsl. See
// docs/source/clouds.md, "Lighting".
uniform vec3  sunColor;
uniform vec3  skyColor;
uniform float sunBrightness;
uniform float skyStrength;
uniform float skyOcclusion;     // share of sky light reaching a slab's underside
// The far clouds' lighting: each genus's far table (cloud_field.glsl's
// genusFarTable: deckLight for genus 0) and deckLightCosMin.

// How the light diffused through a cloud fades with the column above it
// (its optical depth straight up, exp2 units, from a light volume: the
// raymarch's): 1 / (1 + DIFFUSE_FADE * depth), a diffusion's slow fall, not
// the exponential of the light that crosses unscattered. It scales the
// multiple scattering's fill and the second order: a thin cumulus keeps
// them, a cumulonimbus's base under kilometres of tower loses them (it was
// nearly as bright as its lit top at noon). The column up, not toward the
// sun: toward a low sun it crosses tens of kilometres of other clouds and
// the gaps between, which diffusion through one cloud does not model (towers
// went brown at dusk). Not the sky's floor under the clouds: a tower's side
// sees the open sky around it, and without that blue the low sun's orange
// browned it at dusk too. Debug bit 2048: without (the old look).
const float DIFFUSE_FADE = 0.2;

float diffused(float depth) {
    return 1.0 / (1.0 + DIFFUSE_FADE * max(depth, 0.0));
}

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
//   genus        whose clouds: its far table (genusFarTable), sunDepth the
//                optical depth toward the sun from p (cloudSunDepth; needed
//                only with shading > 0)
vec3 cloudRadianceFrom(int genus, float sunDepth, Field f, vec3 p,
                       float density, vec3 V, float shading, float farCoverage,
                       vec3 sunlight, vec3 skylight, vec3 twilight, vec3 weather) {
    if ((uCloudDebug & 1) != 0) shading = 0.0;
    float sunVisibility = shading > 0.0 ? exp2(-sunDepth) : 1.0;
    // What the sun term sees: the shadowing and the powder.
    float powdered = mix(1.0, powder(f, density), f.powderStrength);
    float sunShadowing = sunVisibility * powdered;
    // The second order of scattering (Wrenninge's octaves; EVE-Redux's clouds
    // light theirs with a multiple-scattering lobe of their own): sunlight that
    // scattered once more on its way, so it sees only octaveAttenuation of the
    // sunward depth and its phase is half-way to isotropic. Sunlight-coloured,
    // it lights a cloud's shadowed side where the flat fill below is the sky's.
    float octaveShadowing = shading > 0.0
        ? exp2(-f.octaveAttenuation * sunDepth) * powdered : 1.0;
    if (shading < 1.0) {
        // Far away: the averages of both over the clouds' columns, which
        // the aliasing per-pixel values would not give, for this sun and
        // coverage. They are what the clouds show close up, blurred, so
        // nothing changes colour closing in. A low sun marches along the
        // slab, through the clouds around: the terminator darkens gradually,
        // in the (reddened) sunlight's colour.
        float cosSun = dot(normalize(p), sunDir);
        vec2 at = vec2((cosSun - deckLightCosMin) / (1.0 - deckLightCosMin), farCoverage);
        vec4 averaged = genusFarTable(genus, at);
        sunVisibility = mix(averaged.g, sunVisibility, shading);
        sunShadowing = mix(averaged.r, sunShadowing, shading);
        octaveShadowing = mix(averaged.a, octaveShadowing, shading);
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
    // Where a light volume holds the sample (the raymarch's): by the clouds
    // actually above it, its optical depth up (cloudSkyDepth), between the
    // same floor and the open sky, in place of the ramp; and the diffused
    // light's share left by that column (DIFFUSE_FADE).
    float fade = 1.0;
    if (cloudSkyHeld > 0.0 && shading > 0.0) {
        float diffuse = (uCloudDebug & 2048) == 0 ? diffused(cloudSkyDepth) : 1.0;
        fade = mix(1.0, diffuse, cloudSkyHeld);
        skyReach = mix(skyReach, mix(skyOcclusion, 1.0, exp2(-cloudSkyDepth)),
                       cloudSkyHeld);
    }
    cloudSkyHeld = 0.0;
    // Multiple-scattering fill: sky-coloured and NOT phase-weighted, or dusk
    // shadows turn orange and sunward shadow outshines side-lit cloud.
    float fill = f.multipleScattering * sunBrightness * (1.0 - sunVisibility) * fade;
    octaveShadowing *= fade;
    float ambient = skyStrength * skyReach + fill;

    float phase = phaseFunction(f, dot(sunDir, -V));
    // The lit air above a cloud in the planet's shadow (cloudTwilight): light
    // from the sky, so neither phase-weighted nor self-shadowed, and reaching
    // the underside as the sky's does. This is what spreads the terminator
    // over the clouds instead of a line from lit to dark.
    // Its share comes out of the first order's, so a side-lit cloud in full sun
    // keeps its brightness and only the shadowed parts gain.
    float sunTerm = mix(sunShadowing * phase, octaveShadowing * mix(phase, 1.0, 0.5),
                        f.octaveGain);
    return sunlight * (sunBrightness * sunTerm)
         + skylight * ambient
         + twilight * (sunBrightness * TWILIGHT_GLOW * skyReach);
}

// The radiance of a sample of genus g's clouds (f its field at the
// sample's level): one path for every genus, its far table its own, its sun
// depth from the one light volume (whose depth beyond its window, its
// farShadow, is the deeper clouds' shadow on it).
vec3 cloudRadianceAt(int genus, Field f, vec3 p, float density, vec3 V, float shading,
                     float farCoverage, vec3 sunlight, vec3 skylight,
                     vec3 twilight, vec3 weather) {
    float sunDepth = shading > 0.0 && (uCloudDebug & 1) == 0
                   ? cloudSunDepth(f, p, weather) : 0.0;
    return cloudRadianceFrom(genus, sunDepth, f, p, density, V, shading,
                             farCoverage, sunlight, skylight, twilight, weather);
}

vec3 cloudRadiance(Field f, vec3 p, float density, vec3 V, float shading,
                   float farCoverage, vec3 sunlight, vec3 skylight, vec3 twilight) {
    return cloudRadianceAt(0, f, p, density, V, shading, farCoverage, sunlight,
                           skylight, twilight, weatherAt(p));
}

// The radiance of a thin slab of ice cloud (the march's high layer), linear,
// before the display transform: no sun march, the sunward half of the slab's
// own depth instead; a phase of its own (strongly forward, ice); the sky's
// light, and the sunlight the slab scatters about, as its fill; the lit air's
// twilight once the slab is in the planet's shadow.
//   verticalDepth  the slab's optical depth straight through it (exp2 units)
//   up             the slab's up at the sample (unit)
//   dir            the view ray's direction, toward the slab (unit)
//   maxDepth       what an optical depth saturates toward, softly
//   phase          noise.phase_weights of its optics, and its forward g
//   light          x: its multiple scattering, y: its shadow strength
//   halo           the 22 degree halo's strength in its phase (haloRing),
//                  0 = none (the cirrostratus's, by its share of the depth)
//   sunlight       the sun's colour as it reaches the slab (reddened:
//                  planetCloudLight's)
//   skylight       the sky's (faded through twilight)
//   twilight       the lit air's above it (sunColor * cloudTwilight)
// The 22 degree halo of hexagonal ice (light refracted through the
// crystals' 60 degree prisms, the least deviation 22 degrees): a ring about
// the sun, 1 at its peak, its inner edge sharp (no light refracts less: the
// sky inside it darker), fading out over a few degrees outside. One acos a
// crossing: cheaper than any sample.
float haloRing(float cosAngle) {
    float x = degrees(acos(clamp(cosAngle, -1.0, 1.0))) - 22.0;
    return exp(-x * x / (x < 0.0 ? 0.5 : 8.0));
}

vec3 thinSlabRadiance(float verticalDepth, vec3 up, vec3 dir, float maxDepth, vec4 phase,
                      vec4 light, float halo, vec3 sunlight, vec3 skylight,
                      vec3 twilight) {
    float cosSun = dot(up, sunDir);
    float sunDepth = 0.5 * verticalDepth / max(cosSun, 0.1);
    float sunVisibility = exp2(-maxDepth * (1.0 - exp(-sunDepth * light.y / maxDepth)));
    float ambient = skyStrength + light.x * sunBrightness * (1.0 - sunVisibility);
    float scattered = lobePhase(phase, dot(sunDir, dir));
    if (halo > 0.0) {
        scattered += halo * haloRing(dot(sunDir, dir));
    }
    return sunlight * (sunBrightness * sunVisibility * scattered)
         + skylight * ambient
         + twilight * (sunBrightness * TWILIGHT_GLOW);
}
