#version 330
// A planet's low clouds, raymarched (clouds.march): both depth ranges of the
// view (the split cameras') in one march, at a fraction of its resolution. The
// density field, its lighting and the far field are the billboards' and the
// deck's (cloud_field.glsl, cloud_shading.glsl); the march replaces their
// overdraw: each pixel steps along its own ray through the slab, from the near
// range's near plane to the scene (the near range's depth where it holds
// something, else the far range's), and stops once the cloud is opaque: a
// cloud near the camera spares the march the clouds behind it.
//
// Each frame it traces one pixel of every uBlock x uBlock block of the view
// (uOffset, in turn), which cloud_resolve.frag gathers into the full
// resolution over the frames. Out: the premultiplied cloud and its
// (transmittance-weighted) distance in kilometres (a half float keeps it to
// a part in two thousand), 0 where there is none. Its geometry
// (march.slab_segments, march.step_scale) has a NumPy mirror the tests run.
// The clouds' own light from the shadow maps (cloud_field.glsl's cloudSunDepth).
#define CLOUD_MAP_LIGHTING
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "cloud_field.glsl"
#pragma include "cloud_shading.glsl"

const int MAX_MARCH_LAYERS = 8;        // noise.MAX_LAYERS
const float NOISE_FEATURE_UNITS = 4.0;  // noise.NOISE_FEATURE
const float KM = 1000.0;                // the distance's unit out

// The scene's depth in each range, at the view's resolution. The near one is
// read only while it is drawn (uNearDepthOn: the slab reaches the near range).
uniform sampler2D uSceneDepthNear;
uniform sampler2D uSceneDepthFar;
uniform int   uNearDepthOn;
uniform mat4  uClipToRender;     // the near range's clip space -> the render frame
uniform mat4  uClipToRenderFar;  // the far range's
uniform float uSplit;            // metres: where the near range ends (its step budget)
uniform vec3  uCamRender;        // camera, render frame
uniform vec3  camPos;            // camera, planet frame
uniform float exposure;
uniform int   uOwnAtmosphere;    // the planet's atmosphere (reddens sunlight), -1 none
uniform float uPixelAngle;       // radians per pixel of the march
uniform int   uMarchSteps;       // the step budget per pixel
uniform float uMinStep;          // metres: the shortest step, near the camera
uniform float uStepGrowth;       // a step's length per metre of distance, farther
// Layer i (its octaves) is sampled up to uLayerFar[i] metres from the camera;
// the last one beyond.
uniform float uLayerFar[MAX_MARCH_LAYERS];
uniform int   uLayerCount;       // the layers of one family (its bands, then its far one)
uniform int   uFamilies;         // 1, or 2 with genera: the convective clouds, the stratus under them
uniform sampler2D uBlueNoise;    // march.blue_noise, 64 x 64, the steps' jitter
uniform int   uFrame;            // this pixel's visits: moves its steps' jitter (march.visit_of)
// The edges' billows (what the billboards' sprites carry): a fine billowy
// noise, uDetailScale times the field's first octave, moving the threshold
// by up to uDetailErosion of its window, about its mean uBillowMean.
uniform float uDetailScale;
uniform float uDetailErosion;
uniform float uBillowMean;
uniform float uBillowGain;       // march.billow_stats: the blobs' spread, as the ridged noise's
uniform float uBillowRawMean;    // the noise volume's own mean
uniform int   uBlock;            // view pixels per march pixel, per axis
uniform ivec2 uOffset;           // the pixel traced in each block this frame
uniform vec2  uViewSize;         // the view, pixels

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragDistance;

// What a pixel of the view resolves: a feature spanning fewer view pixels
// than this fades to its mean (it would alias, a different value every
// frame's jitter and every move), over to twice as many.
const vec2 RESOLVED = vec2(2.0, 4.0);

// A step is a sample along the ray as a pixel is across it: features finer than
// about this share of the step's length (over RESOLVED) fade to their means as
// the sub-pixel ones do, or they alias into the jitter (the shimmer of far
// clouds, whatever the steps' count). It costs the clouds' detail at a distance
// (their lumps): march.STEP_FADES gives the least that keeps the shimmer down at
// each march resolution.
uniform float uStepFade;

// How much of a feature of this many metres a view pixel of footprint
// metres resolves: 1 wholly, 0 not at all.
float resolved(float feature, float footprint) {
    return smoothstep(RESOLVED.x, RESOLVED.y, feature / max(footprint, 1e-3));
}

// The billows' noise is turned off the field's axes: value noise has its
// lattice's planes in it (flat spots and creases), which the ridged billows
// show as parallel streaks across every cloud, all one way. A rotation
// (orthonormal, columns listed), so no scale changes.
const mat3 BILLOW_TURN = mat3(0.36, 0.48, -0.8, -0.8, 0.6, 0.0, 0.48, 0.64, 0.6);

// The finer billow layer: its scale over the first's (turned about), its
// offset and its weight.
const float BILLOW_FINER_SCALE = 2.3;
const vec3 BILLOW_FINER_OFFSET = vec3(0.37, 0.81, 0.23);
const float BILLOW_FINER_WEIGHT = 0.5;

// The density, its edges carved by billows (march.DETAIL_SCALE): the
// threshold moves by the billowy noise about its mean, in units of its
// window, so a cloud's edge breaks into rounded lumps, its core stays, and
// the coverage holds on average. Clear air is found first with the most the
// billows could add: cloud alone costs the noise twice over. The two fine
// octaves and the billows fade to their means where a view pixel (footprint
// metres) cannot resolve them: the coverage holds (as an octave left out
// does), the sparkle goes.
float marchDensity(Field f, vec3 p, vec3 weather, float footprint) {
    float unit = NOISE_FEATURE_UNITS / f.noiseScale.x;
    vec2 kept = vec2(resolved(unit / f.octaveScales.z, footprint),
                     resolved(unit / f.octaveScales.w, footprint));
    float shift = weather.x
               + dot((1.0 - kept) * f.octaveWeights.zw, vec2(f.octaveMean));
    f.octaveWeights.zw *= kept;
    float billows = resolved(unit / (f.octaveScales.x * uDetailScale), footprint);
    float window = (uCloudDebug & 32) != 0 ? 0.0  // no billows (profiling)
                 : (f.threshHi - f.threshLo) * uDetailErosion * billows;
    // The field once, at the billows' highest threshold shift: the billows only
    // lower the column's value afterwards, so their pass costs no more noise.
    float raw, fill, shelf;
    if (!cloudColumnAt(f, p, vec3(shift + window * uBillowMean, weather.yz), raw, fill,
                       shelf)) {
        return 0.0;
    }
    float most = max(smoothstep(f.threshLo, f.threshHi, raw) * fill, shelf);
    if (most <= 0.0 || window <= 0.0) {
        return most;
    }
    vec3 coord = BILLOW_TURN * ((p * f.noiseScale + f.noiseOrigin)
                                * (f.octaveScales.x * uDetailScale));
    // Round lumps: two scales of the noise itself, carving where it is high.
    // (The ridged noise, 1 - |2n - 1|, carved along sheets, which show as
    // streaks across a cloud seen edge-on.) Its mean and spread are the ridged
    // noise's, so the coverage and the erosion's depth stay.
    float finer = sampleNoise(coord.yzx * BILLOW_FINER_SCALE + BILLOW_FINER_OFFSET);
    finer = mix(uBillowRawMean, finer,
                resolved(unit / (f.octaveScales.x * uDetailScale * BILLOW_FINER_SCALE),
                         footprint));
    float billow = uBillowMean + uBillowGain * (sampleNoise(coord) - uBillowRawMean
                                                + BILLOW_FINER_WEIGHT * (finer - uBillowRawMean));
    return max(smoothstep(f.threshLo, f.threshHi, raw - window * billow) * fill, shelf);
}

// The fewest samples a stretch of the slab takes (march.SEGMENT_SAMPLES).
const float SEGMENT_SAMPLES = 8.0;
// The march stops at this transmittance, and counts the cloud opaque.
const float OPAQUE = 0.04;
// A step lengthens by this share of the transmittance already lost (march.STEP_DEEPENING):
// in a dense cloud the light is spent in a few steps.
const float STEP_DEEPENING = 1.5;
// A cloudy column's alpha seen from afar: the deck's.
const float FAR_ALPHA = 0.92;
// The weather is read once this far along the ray, metres (uWeatherStretch):
// the fair-weather shift varies over kilometres, but a convective cloud's depth
// (its top) changes over a fraction of a tower's width: read once a kilometre
// it shows as bands across the tower, one per stretch.
uniform float uWeatherStretch;

vec3 unproject(vec3 ndc) {
    vec4 p = uClipToRender * vec4(ndc, 1.0);
    return p.xyz / p.w;
}

vec3 unprojectFar(vec3 ndc) {
    vec4 p = uClipToRenderFar * vec4(ndc, 1.0);
    return p.xyz / p.w;
}

// The stretch of the ray c + t dir inside the sphere of radius r around the
// planet's centre: (t0, t1), empty when t0 >= t1. Through the ray's closest
// point to the centre: float32 keeps a planet's radius to centimetres, its
// square only to tens of metres.
vec2 sphereInterval(vec3 c, vec3 dir, float r) {
    float b = dot(c, dir);
    float q = length(c - dir * b);
    if (q >= r) {
        return vec2(1.0, -1.0);
    }
    float h = sqrt((r - q) * (r + q));
    return vec2(-b - h, -b + h);
}

// About how many steps cover seg, uMinStep-long up to the knee and growing
// with the distance past it.
float stepsOver(vec2 seg, float dtMin, float growth) {
    if (seg.y <= seg.x) {
        return 0.0;
    }
    float knee = dtMin / growth;
    float n = 1.0;
    if (seg.x < knee) {
        n += (min(seg.y, knee) - seg.x) / dtMin;
    }
    if (seg.y > knee) {
        n += log(seg.y / max(seg.x, knee)) / growth;
    }
    return n;
}

int layerAt(float t) {
    for (int i = 0; i < MAX_MARCH_LAYERS - 1; ++i) {
        if (i >= uLayerCount - 1 || t < uLayerFar[i]) {
            return i;
        }
    }
    return uLayerCount - 1;
}

float interleavedGradient(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy) * uBlock + uOffset;
    if (any(greaterThanEqual(vec2(pixel), uViewSize))) {
        discard;
    }
    vec2 ndc = (vec2(pixel) + 0.5) / uViewSize * 2.0 - 1.0;
    vec3 nearPoint = unproject(vec3(ndc, -1.0));
    vec3 farPoint = unprojectFar(vec3(ndc, 1.0));
    vec3 dir = normalize(farPoint - nearPoint);
    float tStart = max(dot(nearPoint - uCamRender, dir), 0.0);
    // The scene: the near range's surface where it has one, else the far
    // range's, else the far plane (EVE's distance buffer of both cameras).
    float nearDepth = uNearDepthOn != 0 ? texelFetch(uSceneDepthNear, pixel, 0).r : 1.0;
    float farDepth = texelFetch(uSceneDepthFar, pixel, 0).r;
    vec3 end = nearDepth < 1.0 ? unproject(vec3(ndc, nearDepth * 2.0 - 1.0))
             : farDepth < 1.0 ? unprojectFar(vec3(ndc, farDepth * 2.0 - 1.0))
             : farPoint;
    float tEnd = length(end - uCamRender);

    // Each family's slab between two spheres, less the planet: up to two
    // segments (a ray from above, or skimming the slab's underside, leaves and
    // re-enters), the families' together sorted along the ray (the slabs do not
    // overlap).
    Field f = getField(0);  // the first family's: every layer of it shares the slab
    vec2 ground = sphereInterval(camPos, dir, planetRadius);
    if (ground.x < ground.y && ground.y > 0.0) {
        tEnd = min(tEnd, max(ground.x, 0.0));
    }
    vec2 segments[4];
    int family[4];
    int count = 0;
    for (int i = 0; i < 2; ++i) {
        if (i >= uFamilies) {
            break;
        }
        Field slabField = getField(i * uLayerCount);
        float rBase = planetRadius + slabField.slabBase;
        vec2 top = sphereInterval(camPos, dir, rBase + slabField.slabThickness);
        vec2 base = sphereInterval(camPos, dir, rBase);
        vec2 first = top;
        vec2 second = vec2(1.0, -1.0);
        if (base.x < base.y) {
            first = vec2(top.x, base.x);
            second = vec2(base.y, top.y);
        }
        first = vec2(max(first.x, tStart), min(first.y, tEnd));
        second = vec2(max(second.x, tStart), min(second.y, tEnd));
        if (first.y > first.x) {
            segments[count] = first;
            family[count] = i;
            ++count;
        }
        if (second.y > second.x) {
            segments[count] = second;
            family[count] = i;
            ++count;
        }
    }
    if (count == 0) {
        discard;
    }
    // Sorted by where they start (at most four).
    for (int i = 1; i < 4; ++i) {
        for (int k = i; k > 0 && k < count; --k) {
            if (segments[k].x < segments[k - 1].x) {
                vec2 swapSegment = segments[k];
                segments[k] = segments[k - 1];
                segments[k - 1] = swapSegment;
                int swapFamily = family[k];
                family[k] = family[k - 1];
                family[k - 1] = swapFamily;
            }
        }
    }

    // Steps uMinStep long near the camera, a share of the distance farther,
    // all lengthened alike when the ray would need more than the budget. The
    // near range (to uSplit) and the far range each have the budget (as when
    // each range had a march of its own): one budget for the whole ray would
    // stretch the near steps for far stretches the march seldom reaches (an
    // opaque cloud near the camera stops it first).
    float dtMin = uMinStep;
    float growth = uStepGrowth;
    vec2 needed = vec2(0.0);  // near, far
    for (int i = 0; i < 4; ++i) {
        if (i < count) {
            vec2 seg = segments[i];
            needed.x += stepsOver(vec2(seg.x, min(seg.y, uSplit)), dtMin, growth);
            needed.y += stepsOver(vec2(max(seg.x, uSplit), seg.y), dtMin, growth);
        }
    }
    vec2 stretch = max(needed / float(uMarchSteps), vec2(1.0));
    // Where in its step each sample lies: no banding, a fine noise instead,
    // moved on by the golden ratio each visit of this pixel (not each frame:
    // traced once a block's cycle, it would skip through the sequence), so
    // its visits stratify the step and the resolve averages them away.
    // The noise is a blue noise at the march's pixel (the gradient noise at a
    // view pixel's stride of uBlock aliased its diagonals into a lattice of
    // stripes), shifted by which pixel of the block this frame traces (the
    // block's pixels differ: R2 over the offset) and moved on by the visit, so
    // a pixel's visits stratify the step.
    float jitter = fract(texelFetch(uBlueNoise, ivec2(gl_FragCoord.xy) & 63, 0).r
                         + dot(vec2(uOffset), vec2(0.7548776662, 0.5698402909))
                         + float(uFrame) * 0.618034);

    float transmittance = 1.0;
    vec3 light = vec3(0.0);
    float weightedDistance = 0.0;
    vec3 sunlight = vec3(0.0);
    vec3 skylight = vec3(0.0);
    vec3 twilight = vec3(0.0);
    bool lit = false;
    int current = -1;
    Field lf = f;
    int steps = 0;
    int emptySteps = 0;  // steps whose density was 0 (uCloudDebug 128)
    // The weather (the shift, the convective depth, the anvil's spread), read
    // once a WEATHER_STRETCH of ray: it varies over kilometres.
    vec3 weather = vec3(0.0);
    float weatherUntil = -1.0;
    for (int s = 0; s < 4; ++s) {
        if (s >= count) {
            break;
        }
        vec2 seg = segments[s];
        int fam = family[s];
        weatherUntil = -1.0;
        float a = seg.x;
        float least = (seg.y - seg.x) / SEGMENT_SAMPLES;  // the fewest a segment takes
        while (a < seg.y && steps < 2 * uMarchSteps + 8 && transmittance > OPAQUE) {
            float k = a < uSplit ? stretch.x : stretch.y;
            float dt = min(min(k * max(dtMin, a * growth) * (1.0 + STEP_DEEPENING * (1.0 - transmittance)), least), seg.y - a);
            float t = a + dt * jitter;
            steps++;
            int layer = fam * uLayerCount + layerAt(t);
            if (layer != current) {
                lf = getField(layer);
                current = layer;
            }
            vec3 p = camPos + dir * t;
            if (t > weatherUntil) {
                weather = fam == 0 ? texture(weatherMap, p).gba
                                   : texture(otherWeather, p).gba;
                weatherUntil = t + uWeatherStretch;
            }
            // Far away a pixel spans more than the clouds' features: their
            // expected coverage (the deck's), as a homogeneous slab whose
            // vertical crossing takes the deck's alpha; the self-shadowing
            // goes to its averages first (cloudRadiance).
            float feature = NOISE_FEATURE_UNITS / (lf.noiseScale.x * lf.octaveScales.x);
            float footprint = t * uPixelAngle / feature;
            float detail = 1.0 - smoothstep(0.15, 0.6, footprint);
            float shading = 1.0 - smoothstep(0.04, 0.2, footprint);
            float density = detail > 0.0
                          ? marchDensity(lf, p, weather,
                                         max(t * uPixelAngle / float(uBlock), uStepFade * dt))
                          : 0.0;
            float farCoverage = 0.0;
            if (shading < 1.0) {
                farCoverage = fam == 0 ? texture(weatherMap, p).r
                                       : texture(otherWeather, p).r;
                float mean = -log2(1.0 - FAR_ALPHA * farCoverage)
                           / (lf.extinction * lf.slabThickness);
                density = mix(mean, density, detail);
            }
            a += dt;
            if (density <= 0.0) {
                ++emptySteps;
                continue;
            }
            float alpha = 1.0 - exp2(-lf.extinction * density * dt);
            if (!lit) {
                // The air's colouring of the light, once a pixel.
                planetCloudLight(uOwnAtmosphere, p, sunlight, skylight, twilight);
                lit = true;
            }
            vec3 radiance = (uCloudDebug & 256) != 0 ? vec3(0.8)  // no lighting (profiling)
                          : fam == 0
                ? cloudRadianceAt(lf, p, density, -dir, shading, farCoverage, sunlight,
                                  skylight, twilight, weather)
                : cloudRadianceOther(lf, getField(uLayerCount - 1), p, density, -dir,
                                     shading, farCoverage, sunlight, skylight,
                                     twilight, weather);
            float weight = transmittance * alpha;
            light += weight * radiance;
            weightedDistance += weight * t;
            transmittance *= 1.0 - alpha;
        }
    }
    if ((uCloudDebug & 128) != 0) {
        // Where the steps go: red the share in empty air, green in cloud, blue
        // the steps over the budget.
        float taken = max(float(steps), 1.0);
        fragColor = vec4(float(emptySteps) / taken, float(steps - emptySteps) / taken,
                         float(steps) / float(uMarchSteps), 1.0);
        fragDistance = vec4(0.001, 0.0, 0.0, 1.0);
        return;
    }
    if ((uCloudDebug & 16) != 0) {
        // The march's work: red its steps over the budget.
        fragColor = vec4(float(steps) / float(uMarchSteps), 0.0, 0.0, 1.0);
        fragDistance = vec4(0.001, 0.0, 0.0, 1.0);
        return;
    }
    float alpha = transmittance <= OPAQUE ? 1.0 : 1.0 - transmittance;
    if (alpha < 1.0 / 512.0) {
        discard;
    }
    float covered = 1.0 - transmittance;
    vec3 colour = light / covered;  // the cloud's mean radiance
    if ((uCloudDebug & 8) != 0) {
        colour = vec3(0.8);
    } else {
        // No air in front of the cloud, unlike the billboards': a marched
        // cloud keeps its own colour at any distance (atmosphere.md, "Clouds
        // at the horizon"). The sun's reddening is in its light already.
        colour = displayColour(colour * exposure);
    }
    fragColor = vec4(colour * alpha, alpha);  // premultiplied "over"
    fragDistance = vec4(weightedDistance / covered / KM, 0.0, 0.0, 1.0);
}
