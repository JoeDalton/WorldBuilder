#version 330

// A planet's cloud decks (clouds.deck.deck_nodes): a genus's slab seen as a
// surface at its height above the sea-level sphere, for the sea's reflection
// (the march draws the view's clouds), one deck a genus (uDeckGenus). The noise is sampled at planet-frame positions, a
// weather field shifts the fbm by direction, and far away the deck is the
// weather's expected coverage. (It was the billboards' fragment stage too:
// removed with them.)
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "cloud_field.glsl"
#pragma include "cloud_shading.glsl"
// Premultiplied "over". See docs/source/clouds.md.
uniform vec3  camPos;

uniform float exposure;

uniform int   uOwnAtmosphere;   // the planet's atmosphere (reddens sunlight), -1 none
uniform float uPixelAngle;      // radians per pixel (the far deck's footprint)
// Whose deck this is (deck.deck_nodes): genus 0 reads its own entry
// (vLayer: the deck's), every other genus its far level, its weather
// (genusWeatherAt) and its far table; the genera hidden (uGenusMask, the
// march's) draw none.
uniform int   uDeckGenus;
uniform int   uGenusMask;

in  vec3  worldPos;
flat in int vLayer;
out vec4 fragColor;

void main() {
    if (((uGenusMask >> uDeckGenus) & 1) == 0) discard;
    Field f = getField(uDeckGenus == 0 ? vLayer
                                       : genusShadowLayer(uDeckGenus));

    vec3 toCamera = worldPos - camPos;
    float distance = length(toCamera);
    vec3 V = -toCamera / distance;

    // The layer's distance fade (its shellFade), on optical depth, after the cap.
    float shellWeight = smoothstep(f.shellFade.x, f.shellFade.y, distance)
                      * (1.0 - smoothstep(f.shellFade.z, f.shellFade.w, distance));
    if (shellWeight <= 0.0) discard;

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
    {
        // The slab seen as a surface. The sight line crosses thickness / |cos|
        // of it (capped near grazing); the density is the mean of two samples
        // through the slab's depth.
        vec3 up = normalize(worldPos);
        // The column's weather: coverage, shift, convective depth, anvil
        // shelf. The clouds there reach the local top (a convective field's
        // grows with the depth; every other's is the slab's).
        vec4 weather = genusWeatherAt(uDeckGenus, up);
        float thickness = f.slabThickness * topFraction(f, weather.b);
        float cosView = max(abs(dot(V, up)), 0.04);
        chord = thickness / cosView;
        float r = planetRadius + f.slabBase;
        samplePos = up * (r + 0.5 * thickness);
        float feature = 4.0 / (f.noiseScale.x * f.octaveScales.x);  // NOISE_FEATURE
        if ((f.recipe & RECIPE_WORLEY) != 0) {
            feature = min(feature, f.cellSize);  // a recipe genus's cells
        }
        float footprint = distance * uPixelAngle / feature;
        detail = 1.0 - smoothstep(0.15, 0.6, footprint);
        shading = 1.0 - smoothstep(0.04, 0.2, footprint);  // <= detail
        density = 0.0;
        if (detail > 0.0 && weather.r > 0.0) {
            vec3 low = up * (r + 0.3 * thickness);
            vec3 high = up * (r + 0.7 * thickness);
            // A recipe genus's own density (genusRecipeDensity: its octaves
            // finer than the pixel's metres fade to their means); the others
            // the field's.
            density = f.recipe != 0
                ? 0.5 * (genusRecipeDensity(f, low, weather, distance * uPixelAngle)
                       + genusRecipeDensity(f, high, weather, distance * uPixelAngle))
                : 0.5 * (cloudDensityAt(f, low, weather.gba)
                       + cloudDensityAt(f, high, weather.gba));
            if (f.anvil > 0.0) {
                // An anvil's shelf, over its column's top: sampled in the
                // middle of its band.
                float core = 0.5 * (SHELF_BAND.y + SHELF_BAND.z);
                vec3 shelf = up * (planetRadius + f.slabBase + core * f.slabThickness);
                density = max(density, cloudDensityAt(f, shelf, weather.gba));
            }
        }
        if (shading < 1.0) {
            // The weather map's coverage: what the field's columns give
            // with its shift there, by construction (clouds.weather).
            farCoverage = weather.r;
        }
        if (density <= 0.0 && farCoverage * (1.0 - detail) < 0.003) discard;
    }

    float opticalDepth = min(f.extinction * density * chord, f.maxOpticalDepth);

    opticalDepth *= shellWeight;

    float alpha = 1.0 - exp2(-opticalDepth);
    // Each cloudy column is close to opaque; the far deck is their share.
    alpha = mix(0.92 * farCoverage * shellWeight, alpha, detail);
    if (alpha <= 0.0) discard;

    // Lighting last, after every discard.
    if ((uCloudDebug & 8) != 0) {
        fragColor = vec4(vec3(0.8) * alpha, alpha);
        return;
    }
    vec3 sunlight, skylight, twilight;
    planetCloudLight(uOwnAtmosphere, samplePos, sunlight, skylight, twilight);
    // Other bodies eclipsing the sun.
    float eclipse = eclipseVisibility(samplePos - camPos, sunDir, uShadowSelf);
    sunlight *= eclipse
              * terrainShadow(normalize(samplePos), length(samplePos) - planetRadius)
              * ringVisibility(samplePos - camPos, sunDir);
    twilight *= eclipse;
    skylight *= mix(ECLIPSE_SKY_FLOOR, 1.0, eclipse);
    vec3 colour = cloudRadianceAt(uDeckGenus, f, samplePos, density, V, shading,
                                  farCoverage, sunlight, skylight, twilight,
                                  genusWeatherAt(uDeckGenus, normalize(samplePos)).gba);
    // No air in front of the cloud: it keeps its own colour at any distance
    // (atmosphere.md, "Clouds at the horizon").

    // The sky's display transform, so cloud hazed by the air meets the sky
    // behind it. Per channel below white, which softens the warm cast a long
    // low path of air leaves on lit tops (kept, it turned the horizon's clouds
    // orange at noon); hue kept above white, or a dusk sun's orange light maps
    // to white. See docs: atmosphere.md, clouds at the horizon.
    colour = displayColour(colour * exposure);

    fragColor = vec4(colour * alpha, alpha);   // premultiplied "over"
}
