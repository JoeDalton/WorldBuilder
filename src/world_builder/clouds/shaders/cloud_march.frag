#version 330
// A planet's low clouds, raymarched (clouds.march): one depth range of the
// view (the split cameras'), at a fraction of its resolution. The density
// field, its lighting and the far field are the billboards' and the deck's
// (cloud_field.glsl, cloud_shading.glsl); the march replaces their overdraw:
// each pixel steps along its own ray through the slab, from the range's near
// plane to the scene's depth, and stops once the cloud is opaque.
//
// Each frame it traces one pixel of every uBlock x uBlock block of the view
// (uOffset, in turn), which cloud_resolve.frag gathers into the full
// resolution over the frames. Out: the premultiplied cloud and its
// (transmittance-weighted) distance in kilometres (a half float keeps it to
// a part in two thousand), 0 where there is none. Its geometry
// (march.slab_segments, march.step_scale) has a NumPy mirror the tests run.
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "cloud_field.glsl"
#pragma include "cloud_shading.glsl"

const int MAX_MARCH_LAYERS = 8;        // noise.MAX_LAYERS
const float NOISE_FEATURE_UNITS = 4.0;  // noise.NOISE_FEATURE
const float KM = 1000.0;                // the distance's unit out

uniform sampler2D uSceneDepth;   // the range's depth, at the view's resolution
uniform mat4  uClipToRender;     // the range's clip space -> the render frame
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
uniform int   uLayerCount;
uniform int   uFrame;            // moves the steps' jitter (0: fixed)
// The edges' billows (what the billboards' sprites carry): a fine billowy
// noise, uDetailScale times the field's first octave, moving the threshold
// by up to uDetailErosion of its window, about its mean uBillowMean.
uniform float uDetailScale;
uniform float uDetailErosion;
uniform float uBillowMean;
uniform int   uBlock;            // view pixels per march pixel, per axis
uniform ivec2 uOffset;           // the pixel traced in each block this frame
uniform vec2  uViewSize;         // the view, pixels

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragDistance;

// cloudDensityShifted's, its edges carved by billows (march.DETAIL_SCALE):
// the threshold moves by the billowy noise about its mean, in units of its
// window, so a cloud's edge breaks into rounded lumps, its core stays, and
// the coverage holds on average. Cloud only costs the noise twice over: clear
// air is found first with the most the billows could add.
float marchDensity(Field f, vec3 p, float shift) {
    float window = (f.threshHi - f.threshLo) * uDetailErosion;
    float most = cloudDensityShifted(f, p, shift + window * uBillowMean);
    if (most <= 0.0 || window <= 0.0) {
        return most;
    }
    vec3 coord = (p * f.noiseScale + f.noiseOrigin) * (f.octaveScales.x * uDetailScale);
    float billow = 1.0 - abs(2.0 * sampleNoise(coord) - 1.0);
    return cloudDensityShifted(f, p, shift + window * (uBillowMean - billow));
}

// The march stops at this transmittance, and counts the cloud opaque.
const float OPAQUE = 0.01;
// A cloudy column's alpha seen from afar: the deck's.
const float FAR_ALPHA = 0.92;
// Clear air's steps are up to this many times the planned ones.
const float COARSE = 4.0;
// The weather's shift is read once this far along the ray, metres.
const float WEATHER_STRETCH = 1000.0;

vec3 unproject(vec3 ndc) {
    vec4 p = uClipToRender * vec4(ndc, 1.0);
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
    vec3 farPoint = unproject(vec3(ndc, 1.0));
    vec3 dir = normalize(farPoint - nearPoint);
    float depth = texelFetch(uSceneDepth, pixel, 0).r;
    float tStart = max(dot(nearPoint - uCamRender, dir), 0.0);
    vec3 end = depth < 1.0 ? unproject(vec3(ndc, depth * 2.0 - 1.0)) : farPoint;
    float tEnd = length(end - uCamRender);

    // The slab between two spheres, less the planet: up to two segments (a
    // ray from above, or skimming the slab's underside, leaves and re-enters).
    Field f = getField(0);  // every layer shares the slab
    float rBase = planetRadius + f.slabBase;
    vec2 top = sphereInterval(camPos, dir, rBase + f.slabThickness);
    vec2 base = sphereInterval(camPos, dir, rBase);
    vec2 ground = sphereInterval(camPos, dir, planetRadius);
    if (ground.x < ground.y && ground.y > 0.0) {
        tEnd = min(tEnd, max(ground.x, 0.0));
    }
    vec2 seg0 = top;
    vec2 seg1 = vec2(1.0, -1.0);
    if (base.x < base.y) {
        seg0 = vec2(top.x, base.x);
        seg1 = vec2(base.y, top.y);
    }
    seg0 = vec2(max(seg0.x, tStart), min(seg0.y, tEnd));
    seg1 = vec2(max(seg1.x, tStart), min(seg1.y, tEnd));
    if (seg0.y <= seg0.x && seg1.y <= seg1.x) {
        discard;
    }

    // Steps uMinStep long near the camera, a share of the distance farther,
    // all lengthened alike when the ray would need more than the budget.
    float dtMin = uMinStep;
    float growth = uStepGrowth;
    float needed = stepsOver(seg0, dtMin, growth) + stepsOver(seg1, dtMin, growth);
    if (needed > float(uMarchSteps)) {
        float k = needed / float(uMarchSteps);
        dtMin *= k;
        growth *= k;
    }
    // Where in its step each sample lies: no banding, a fine noise instead,
    // moved on by the golden ratio each frame so the frames average it away.
    float jitter = fract(interleavedGradient(vec2(pixel)) + float(uFrame) * 0.618034);

    float feature = NOISE_FEATURE_UNITS / (f.noiseScale.x * f.octaveScales.x);
    float transmittance = 1.0;
    vec3 light = vec3(0.0);
    float weightedDistance = 0.0;
    vec3 sunlight, skylight, twilight;
    bool lit = false;
    int current = -1;
    Field lf = f;
    int steps = 0;
    // The weather's shift, read once a WEATHER_STRETCH of ray: it varies
    // over kilometres.
    float shift = 0.0;
    float shiftUntil = -1.0;
    for (int s = 0; s < 2; ++s) {
        vec2 seg = s == 0 ? seg0 : seg1;
        float a = seg.x;
        float least = 0.5 * (seg.y - seg.x);  // two samples a segment at least
        // Clear air is crossed in longer steps (up to COARSE times), the
        // cloud in the planned ones: a step that lands in cloud after a long
        // one is taken again from its start.
        float coarse = 1.0;
        while (a < seg.y && steps < uMarchSteps + 8 && transmittance > OPAQUE) {
            float dt = min(min(max(dtMin, a * growth), least) * coarse, seg.y - a);
            float t = a + dt * jitter;
            steps++;
            int layer = layerAt(t);
            if (layer != current) {
                lf = getField(layer);
                current = layer;
            }
            vec3 p = camPos + dir * t;
            if (t > shiftUntil) {
                shift = weatherShift(p);
                shiftUntil = t + WEATHER_STRETCH;
            }
            // Far away a pixel spans more than the clouds' features: their
            // expected coverage (the deck's), as a homogeneous slab whose
            // vertical crossing takes the deck's alpha; the self-shadowing
            // goes to its averages first (cloudRadiance).
            float footprint = t * uPixelAngle / feature;
            float detail = 1.0 - smoothstep(0.15, 0.6, footprint);
            float shading = 1.0 - smoothstep(0.04, 0.2, footprint);
            float density = detail > 0.0 ? marchDensity(lf, p, shift) : 0.0;
            float farCoverage = 0.0;
            if (shading < 1.0) {
                farCoverage = texture(weatherMap, p).r;
                float mean = -log2(1.0 - FAR_ALPHA * farCoverage)
                           / (lf.extinction * lf.slabThickness);
                density = mix(mean, density, detail);
            }
            if (density <= 0.0) {
                a += dt;
                coarse = min(coarse * 2.0, COARSE);
                continue;
            }
            if (coarse > 1.0) {
                coarse = 1.0;  // back to the start of this step, finely
                continue;
            }
            a += dt;
            float alpha = 1.0 - exp2(-lf.extinction * density * dt);
            if (!lit) {
                // The air's colouring of the light, once a pixel.
                planetCloudLight(uOwnAtmosphere, p, sunlight, skylight, twilight);
                lit = true;
            }
            vec3 radiance = cloudRadianceShifted(lf, p, density, -dir, shading,
                                                 farCoverage, sunlight, skylight,
                                                 twilight, shift);
            float weight = transmittance * alpha;
            light += weight * radiance;
            weightedDistance += weight * t;
            transmittance *= 1.0 - alpha;
        }
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
        // The air in front, once, at the cloud's (transmittance-weighted)
        // distance; its in-scattered light weighted by the cloud's alpha, as
        // the billboards'.
        if ((uCloudDebug & 2) == 0) {
            vec3 airT, airS;
            // As few samples as the billboards' per vertex: it is smooth.
            atmosphereAlong(dir, weightedDistance / covered, sunDir, uAtmoVertexSteps,
                            airT, airS);
            colour = colour * airT + airS;
        }
        colour = displayColour(colour * exposure);
    }
    fragColor = vec4(colour * alpha, alpha);  // premultiplied "over"
    fragDistance = vec4(weightedDistance / covered / KM, 0.0, 0.0, 1.0);
}
