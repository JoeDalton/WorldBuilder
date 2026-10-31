#version 330

// A planet's clouds (clouds.sphere): the slab is a height above the sea-level
// sphere, the noise is sampled at planet-frame positions, a weather field
// shifts the fbm by direction, and the atmospheres haze them. uDeck == 1 draws
// the far-field deck surface instead of a billboard.
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "cloud_field.glsl"
// Cloud colour pass. Each billboard is a SAMPLE OF A VOLUME, not a sprite: its
// outline and colour are functions of world position alone, while the sprite only
// supplies how much volume sits behind each pixel (a thickness profile). Types mix
// by superposition in the blender, so a fragment only evaluates its own field.
//
// Premultiplied "over", back to front: with alpha = 1 - exp2(-opticalDepth), N
// overlapping billboards leave transmittance exp2(-sum of depths), i.e. the
// billboards along a view ray ARE the steps of a raymarch. See docs/source/clouds.md.
uniform sampler2D p3d_Texture0;  // puff atlas: alpha = per-quad THICKNESS profile
uniform vec3  camPos;

uniform vec3  sunColor;
uniform vec3  skyColor;
uniform float sunBrightness;
uniform float skyStrength;
uniform float skyOcclusion;     // share of sky light reaching a deck's underside
uniform float exposure;

uniform int   uDeck;            // 1: the deck surface, not a billboard
uniform int   uOwnAtmosphere;   // the planet's atmosphere (reddens sunlight), -1 none
uniform float uPixelAngle;      // radians per pixel (the far deck's footprint)
// The far deck's lighting (noise.deck_light_table): x the sun's cosine at the
// cloud, from deckLightCosMin to 1; y the weather's coverage. r: the sun
// term's shadowing (with the powder), g: the shadowing alone.
uniform sampler2D deckLight;
uniform float deckLightCosMin;

in  vec2  vUv;
in  vec3  worldPos;
in  float vRadius;
in  float vFade;
flat in int vLayer;
in vec3 vAtmoT;      // planet-mode billboards: the air in front, per vertex
in vec3 vAtmoS;
in vec3 vSunTint;
in float vSkyTint;
in vec3 vTwilight;
out vec4 fragColor;

// Darkens optically THIN cloud. Fed the LOCAL density over a fixed length: the
// sunward depth would invert it, and the quad's chord would tie it to quad size.
float powder(Field f, float density) {
    return 1.0 - exp2(-f.fieldExtinction * density * f.powderLength * 2.0);
}

void main() {
    Field f = getField(vLayer);

    vec3 toCamera = worldPos - camPos;
    float distance = length(toCamera);
    vec3 V = -toCamera / distance;

    // LOD shell crossfade: adjacent shells' weights sum to 1. Applied to OPTICAL
    // DEPTH (exact for any depth, unlike alpha) and AFTER the cap (or two shells
    // could each reach it), below. Known before the field is: a fragment
    // weighing nothing (another shell's range) costs nothing.
    float shellWeight = smoothstep(f.shellFade.x, f.shellFade.y, distance)
                      * (1.0 - smoothstep(f.shellFade.z, f.shellFade.w, distance));
    if (shellWeight <= 0.0 || vFade <= 0.0) discard;

    float puff;
    float chord;
    vec3 samplePos;
    float density;
    // The deck far away: 1 = the field's own detail, 0 = its expected
    // coverage (the share of a pixel's columns that hold cloud), once a
    // pixel spans more than the clouds' features (they would alias to
    // speckle from orbit).
    float detail = 1.0;
    // The same for the self-shadowing and powder, which go first: a cloud's
    // self-shadow is smaller than the cloud and aliases sooner. 0 = their
    // averages (the deck's light table).
    float shading = 1.0;
    float farCoverage = 0.0;
    if (uDeck == 1) {
        // The slab seen as a surface. The sight line crosses thickness / |cos|
        // of it (capped near grazing); the density is the mean of two samples
        // through the slab's depth.
        puff = 1.0;
        vec3 up = normalize(worldPos);
        float cosView = max(abs(dot(V, up)), 0.04);
        chord = f.slabThickness / cosView;
        float r = planetRadius + f.slabBase;
        samplePos = up * (r + 0.5 * f.slabThickness);
        float feature = 4.0 / (f.noiseScale.x * f.octaveScales.x);  // NOISE_FEATURE
        float footprint = distance * uPixelAngle / feature;
        detail = 1.0 - smoothstep(0.15, 0.6, footprint);
        shading = 1.0 - smoothstep(0.04, 0.2, footprint);  // <= detail
        density = 0.0;
        if (detail > 0.0) {
            vec3 low = up * (r + 0.3 * f.slabThickness);
            vec3 high = up * (r + 0.7 * f.slabThickness);
            density = 0.5 * (cloudDensity(f, low) + cloudDensity(f, high));
        }
        if (shading < 1.0) {
            // The weather map's coverage: what the field's columns give
            // with its shift there, by construction (clouds.weather).
            farCoverage = texture(weatherMap, up).r;
        }
        if (density <= 0.0 && farCoverage * (1.0 - detail) < 0.003) discard;
    } else {
        puff = texture(p3d_Texture0, vUv).a;
        if (puff < 0.01) discard;                // the sprite's own soft edge
        chord = 2.0 * vRadius * puff;
        // Sampled ON the quad plane (not mid-chord, which would make overlapping
        // quads disagree).
        samplePos = worldPos;
        density = cloudDensity(f, samplePos);
        if (density <= 0.0) discard;             // THE crisp silhouette
    }

    // Capped so every billboard stays a thin veil, never a hard-edged opaque quad.
    float opticalDepth = min(f.extinction * density * chord, f.maxOpticalDepth);

    opticalDepth *= shellWeight;

    float alpha = 1.0 - exp2(-opticalDepth);
    if (uDeck == 1) {
        // Each cloudy column is close to opaque; the far deck is their share.
        alpha = mix(0.92 * farCoverage * shellWeight, alpha, detail);
    }
    alpha *= vFade;
    if (alpha <= 0.0) discard;

    // Lighting last, after every discard.
    if ((uCloudDebug & 8) != 0) {
        fragColor = vec4(vec3(0.8) * alpha, alpha);
        return;
    }
    if ((uCloudDebug & 1) != 0) shading = 0.0;
    float sunVisibility = shading > 0.0 ? exp2(-sunOpticalDepth(f, samplePos)) : 1.0;
    // What the sun term sees: the shadowing and the powder.
    float sunShadowing = sunVisibility * mix(1.0, powder(f, density), f.powderStrength);
    if (shading < 1.0) {
        // The far deck: the averages of both over the clouds' columns, which
        // the aliasing per-pixel values would not give, for this sun and
        // coverage. They are what the deck shows close up, blurred, so
        // nothing changes colour closing in. A low sun marches along the
        // slab, through the clouds around: the terminator darkens gradually,
        // in the (reddened) sunlight's colour.
        vec2 size = vec2(textureSize(deckLight, 0));
        float cosSun = dot(normalize(samplePos), sunDir);
        vec2 at = vec2((cosSun - deckLightCosMin) / (1.0 - deckLightCosMin), farCoverage);
        vec2 averaged = texture(deckLight, (clamp(at, 0.0, 1.0) * (size - 1.0) + 0.5) / size).rg;
        sunVisibility = mix(averaged.g, sunVisibility, shading);
        sunShadowing = mix(averaged.r, sunShadowing, shading);
    }
    // Skylight comes from above: occlude the SKY term toward the underside only.
    float skyReach = mix(skyOcclusion, 1.0, clamp(slabHeight(f, heightOf(samplePos)), 0.0, 1.0));
    // Multiple-scattering fill: sky-coloured and NOT phase-weighted, or dusk
    // shadows turn orange and sunward shadow outshines side-lit cloud.
    float ambient = skyStrength * skyReach
                  + f.multipleScattering * sunBrightness * (1.0 - sunVisibility);

    float phase = phaseFunction(f, dot(sunDir, -V));
    vec3 sunlight = sunColor;
    vec3 skylight = skyColor;
    vec3 twilight = vec3(0.0);
    if (uDeck == 0) {
        sunlight *= vSunTint;  // per vertex (cloud_sphere.vert)
        skylight *= vSkyTint;
        twilight = sunColor * vTwilight;
    } else if (uOwnAtmosphere >= 0) {
        // Through the planet's air: reddened low, fading out toward its
        // shadow (cloud_field.glsl's cloudSunTint).
        vec3 tint = cloudSunTint(uOwnAtmosphere, samplePos, sunDir);
        sunlight *= tint;
        twilight = sunColor * cloudTwilight(uOwnAtmosphere, samplePos, sunDir, tint);
        // The sky's light fades through twilight as the air's own fill
        // does (atmosphere.glsl), so clouds in the planet's shadow stay as
        // bright as the air around them.
        skylight *= smoothstep(-0.3, 0.1, dot(normalize(samplePos), sunDir));
    }
    // The lit air above a cloud in the planet's shadow (cloudTwilight): light
    // from the sky, so neither phase-weighted nor self-shadowed, and reaching
    // the underside as the sky's does. This is what spreads the terminator
    // over the clouds instead of a line from lit to dark.
    vec3 colour = sunlight * (sunBrightness * sunShadowing * phase)
                + skylight * ambient
                + twilight * (sunBrightness * TWILIGHT_GLOW * skyReach);
    // The air between the camera and the cloud, in linear light. Its
    // in-scattered light is in front of the cloud, so it is weighted by the
    // cloud's alpha like the cloud itself: the background behind carries the
    // same air's light for the uncovered part.
    vec3 transmittance = vAtmoT;
    vec3 inscatter = vAtmoS;
    if (uDeck == 1 && (uCloudDebug & 2) == 0) {  // one layer: per fragment is affordable
        atmosphereAlong(-V, distance, sunDir, transmittance, inscatter);
    }
    colour = colour * transmittance + inscatter;

    // The sky's display transform, so cloud hazed by the air meets the sky
    // behind it. Per channel below white, which softens the warm cast a long
    // low path of air leaves on lit tops (kept, it turned the horizon's clouds
    // orange at noon); hue kept above white, or a dusk sun's orange light maps
    // to white. See docs: atmosphere.md, clouds at the horizon.
    colour = displayColour(colour * exposure);

    fragColor = vec4(colour * alpha, alpha);   // premultiplied "over"
}
