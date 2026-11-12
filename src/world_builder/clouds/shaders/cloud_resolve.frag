#version 330
// The raymarch's full resolution (clouds.march), gathered over the frames:
// cloud_march.frag traces one pixel of every uBlock.x x uBlock.y block a frame;
// every pixel here blends this frame's samples around it (reconstructed
// bilinearly from the four nearest) into the last frame's result, moved to
// where the camera now sees it (reprojected along the view ray to the pixel's
// own cloud distance) and held within the spread of this frame's samples
// around it (widened by a floor while the view is still, or the steps' own
// noise would clamp it), so a change shows at once rather than trailing
// behind. How much a pixel takes from this frame falls
// with how many samples it has gathered (its count), down to uMinWeight:
// still, the steps' jitter averages away over many visits; in motion down to
// uMovingWeight only, about a cycle of the block. Every pixel of a
// block takes the same share, whatever its distance to a traced one, or the
// traced pixels would show as a grid.
//
// Depth-aware: each sample weighs by how near its scene (where its march
// stopped: cloud_march.frag keeps its distance in the distance's g) is to
// this pixel's own (the depth passes here, the same measure). A
// sample that stopped at a thin thing in front (a tree, a ridge's rim) holds
// only the air before it: blended into the pixels around, it showed as a halo
// of sky round the tree in a cloud; those pixels now take the samples that
// went on behind it, and the tree's own pixels the one that stopped at it.
//
// The cloud is premultiplied; its distance (kilometres, 0: none) and its
// count go on with it (r and g), for the next frame and for the composite's
// depth test. The textures may be larger than what is drawn in them (padded
// to powers of two): pixels, not UVs, except for the history's filtered
// read.
uniform sampler2D uMarch;            // this frame's samples, one a block
uniform sampler2D uMarchDistance;
uniform sampler2D uHistory;          // the last frame's result, full resolution
uniform sampler2D uHistoryDistance;  // r: distance, km; g: samples gathered
uniform mat4  uClipToRel;            // clip space -> relative to the camera (world axes)
uniform mat4  uPrevRelToClip;        // relative to the camera -> the last frame's clip space
uniform ivec2 uBlock;
uniform ivec2 uOffset;
uniform vec2  uViewSize;
uniform int   uHistoryValid;         // 0: no usable last frame (first, resized, jumped)
uniform float uStill;               // 1: the camera still, 0: moving (the clamp's floor)
uniform float uSmooth;               // how far a pixel on fresh samples takes their smooth mean (0 to 1)
uniform float uMinWeight;            // this frame's least share, with a full count
uniform float uMovingWeight;         // the same while the camera moves (by uStill): march.MOVING_WEIGHTS
// The scene's depth in each range (the march's: cloud_march.frag), and each
// range's clip space -> the render frame.
uniform sampler2D uSceneDepthNear;
uniform sampler2D uSceneDepthFar;
uniform int   uNearDepthOn;
uniform int   uDepthScale;           // view pixels per depth pixel, per axis
uniform mat4  uClipToRender;
uniform mat4  uClipToRenderFar;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragDistance;

// How far to look along a ray with no cloud on it (the sky at infinity).
const float NO_CLOUD = 1.0e7;
// The history is held within this many standard deviations of this frame's
// samples around the pixel.
const float CLAMP_SIGMA = 1.25;
// The reach of the samples' statistics, in samples: their weight falls to
// nothing this far from the pixel (a radial bump, (1 - (d / reach)^2)^2).
const float STATS_REACH = 2.0;
// The share a pixel takes from the frame (uSmooth's ramp): none below the first,
// all of uSmooth at the second.
const float SMOOTH_FROM = 0.08;
const float SMOOTH_TO = 0.4;
// The clamp's width, over the samples' spread: a floor and a share of their
// mean, so a pixel whose neighbours happen to agree is not clamped by noise, while
// the view is still (uStill): a floor in motion keeps a cloud as it was, thin,
// as it comes closer.
const float CLAMP_FLOOR = 0.04;
const float CLAMP_SHARE = 0.15;
// The count a clamp may take a pixel's history down to, at most: a change
// still starts it over, but a pixel does not take a whole noisy sample.
const float COUNT_KEPT = 4.0;
// The share of the clamp's floor kept in motion (1 still). With none, the
// clamp's range in motion was the samples' own spread: where they
// agree it was their noisy mean, which moved every frame and the history
// with it: a tower grainy in motion, smooth still (motion_shots.py's grain,
// 1.47 to 1.11 rising past a tower). A floor keeps a little of what a cloud
// was as it comes closer: a little more lag (the error over the clouds 2.2
// to 2.6 levels there), traded for the grain (performance.md §28).
const float MOTION_FLOOR = 0.25;
// A sample's weight by its scene's distance against the pixel's own:
// exp(-DEPTH_SHARPNESS |ln(ratio)|): 10 % off weighs 0.56, 30 % off 0.2;
// the sky (no scene) against any surface, nothing. And the least a sample of
// the statistics keeps (a window all of other scenes still has a mean).
const float DEPTH_SHARPNESS = 6.0;
const float DEPTH_FLOOR = 1e-3;
// The distance of a pixel with no scene (the sky), metres.
const float SKY_DISTANCE = 1.0e9;

// The distance of the scene at a view pixel, metres, as the march stopped at
// it (the near range's surface, else the far range's, else none): from the
// near range's near plane, the same for both ranges (cloud_march.frag's
// sceneKm, there for each sample).
float sceneDistance(ivec2 viewPixel) {
    ivec2 depthPixel = viewPixel / uDepthScale;
    vec2 ndc = (vec2(viewPixel) + 0.5) / uViewSize * 2.0 - 1.0;
    vec4 near = uClipToRender * vec4(ndc, -1.0, 1.0);
    float nearDepth = uNearDepthOn != 0 ? texelFetch(uSceneDepthNear, depthPixel, 0).r : 1.0;
    vec4 p;
    if (nearDepth < 1.0) {
        p = uClipToRender * vec4(ndc, nearDepth * 2.0 - 1.0, 1.0);
    } else {
        float farDepth = texelFetch(uSceneDepthFar, depthPixel, 0).r;
        if (farDepth >= 1.0) {
            return SKY_DISTANCE;
        }
        p = uClipToRenderFar * vec4(ndc, farDepth * 2.0 - 1.0, 1.0);
    }
    return length(p.xyz / p.w - near.xyz / near.w);
}

float depthWeight(float other, float own) {
    return exp(-DEPTH_SHARPNESS * abs(log(max(other, 0.1) / max(own, 0.1))));
}

vec3 rayDir(vec2 ndc) {
    vec4 near = uClipToRel * vec4(ndc, -1.0, 1.0);
    vec4 far = uClipToRel * vec4(ndc, 1.0, 1.0);
    return normalize(far.xyz / far.w - near.xyz / near.w);
}

// Where the point along this pixel's ray at distance km was in the last
// frame's view, as a texel coordinate of the history; false off it.
bool reproject(vec3 dir, float km, out vec2 at) {
    vec3 point = dir * (km > 0.0 ? km * 1000.0 : NO_CLOUD);
    vec4 previous = uPrevRelToClip * vec4(point, 1.0);
    vec2 uv = previous.xy / previous.w * 0.5 + 0.5;
    at = uv * uViewSize;
    return previous.w > 0.0 && all(greaterThanEqual(uv, vec2(0.0)))
        && all(lessThanEqual(uv, vec2(1.0)));
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 last = ivec2(ceil(uViewSize / vec2(uBlock))) - 1;
    vec2 block = vec2(uBlock);

    // The march's samples around: sample k sits at view pixel k * block +
    // uOffset. The four nearest, bilinearly (read with the window below,
    // which holds them); the 4 x 4 around, for the clamp.
    vec2 f = (vec2(pixel) - vec2(uOffset)) / block;
    ivec2 base = ivec2(floor(f));
    vec2 w = f - vec2(base);
    vec4 reconstructed = vec4(0.0);
    float reconstructedWeight = 0.0;
    float own = sceneDistance(pixel);
    // The sample of the window whose scene is nearest this pixel's: what the
    // pixel takes when none of the four nearest shares its scene.
    vec4 closest = vec4(0.0);
    float closestWeight = -1.0;
    // The samples' mean and spread around this pixel, for the clamp: the 4 x 4
    // about it, each weighted by a bump of its distance that falls to nothing,
    // flat, STATS_REACH samples away. A sample enters and leaves the window at
    // no weight, so the statistics run on unbroken from one of the block's
    // cells to the next. The 3 x 3 about the nearest sample (a Gaussian, wider
    // in motion) jumped at each cell's edge, a quarter of its weight swapped
    // in motion; where the clamp cut the history, its box printed the jumps:
    // a mosaic of the block's cells over a cloud's shaded side whenever the
    // camera moved (Karis' "3x3 box artifacts", performance.md §30). The
    // corners beyond the reach weigh nothing and are not read.
    vec4 mean = vec4(0.0);
    vec4 square = vec4(0.0);
    float total = 0.0;
    float kmSum = 0.0;
    float kmTotal = 0.0;
    // The fresh samples' distances' range: what the history's may be.
    float kmLo = 1e9;
    float kmHi = 0.0;
    for (int y = -1; y <= 2; ++y) {
        for (int x = -1; x <= 2; ++x) {
            ivec2 tap = base + ivec2(x, y);
            vec2 d = vec2(tap) - f;
            float bump = 1.0 - dot(d, d) / (STATS_REACH * STATS_REACH);
            if (bump <= 0.0) {
                continue;
            }
            ivec2 clamped = clamp(tap, ivec2(0), last);
            vec4 v = texelFetch(uMarch, clamped, 0);
            vec2 tapDistance = texelFetch(uMarchDistance, clamped, 0).rg;
            float tapScene = tapDistance.g > 0.0 ? tapDistance.g * 1000.0 : SKY_DISTANCE;
            float dw = depthWeight(tapScene, own);
            float g = bump * bump * max(dw, DEPTH_FLOOR);
            if (dw > closestWeight) {
                closestWeight = dw;
                closest = v;
            }
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {  // one of the four nearest
                vec2 bw = mix(vec2(1.0) - w, w, vec2(x, y));
                reconstructed += bw.x * bw.y * dw * v;
                reconstructedWeight += bw.x * bw.y * dw;
            }
            mean += g * v;
            square += g * v * v;
            total += g;
            if (v.a > 0.0) {
                float tapKm = tapDistance.r;
                if (tapKm > 0.0) {
                    kmSum += g * v.a * tapKm;
                    kmTotal += g * v.a;
                    kmLo = min(kmLo, tapKm);
                    kmHi = max(kmHi, tapKm);
                }
            }
        }
    }
    // Clear sky is most of the view: no cloud in the samples around
    // and none where the history was (at the sky's distance) is no cloud,
    // without the rest. (With the four nearest only, a pixel at a cloud's edge
    // flipped between that and the full resolve as the samples' offset moved:
    // its count zeroed, a blink in motion.)
    if (mean.a <= 0.0 && uHistoryValid != 0) {
        vec2 clearAt;
        vec3 clearDir = rayDir((vec2(pixel) + 0.5) / uViewSize * 2.0 - 1.0);
        if (reproject(clearDir, 0.0, clearAt)
            && texture(uHistory, clearAt / vec2(textureSize(uHistory, 0))).a <= 0.0) {
            fragColor = vec4(0.0);
            fragDistance = vec4(0.0, 1.0, 0.0, 1.0);
            return;
        }
    }
    // The distance the history is moved by, from the same smooth weights: a
    // bilinear mix of four samples changes with the pixel's place in its block
    // where an edge is, and the reprojection with it (a lattice of offsets).
    float reconstructedKm = kmTotal > 0.0 ? kmSum / kmTotal : 0.0;
    reconstructed = reconstructedWeight > 1e-4 ? reconstructed / reconstructedWeight : closest;
    mean /= total;
    vec4 sigma = sqrt(max(square / total - mean * mean, vec4(0.0)))
               + max(uStill, MOTION_FLOOR) * (CLAMP_FLOOR + CLAMP_SHARE * mean);
    vec4 lo = mean - CLAMP_SIGMA * sigma;
    vec4 hi = mean + CLAMP_SIGMA * sigma;

    vec2 ndc = (vec2(pixel) + 0.5) / uViewSize * 2.0 - 1.0;
    vec3 dir = rayDir(ndc);
    // Reprojected by this frame's distance first, then by the history's own
    // there (the pixel's cloud as it was): an edge keeps its place.
    vec2 at = vec2(0.0);
    bool valid = uHistoryValid != 0 && reproject(dir, reconstructedKm, at);
    vec2 historyKm = vec2(0.0);
    if (valid) {
        historyKm = texelFetch(uHistoryDistance, ivec2(at), 0).rg;
        // The history's distance no farther or nearer than this frame's
        // samples around: mixed like the colour, it lags them by up to
        // 1 / uMinWeight frames, and flying forward (the clouds nearer each
        // frame) it moved the history from too far, where the clamp threw it
        // away (small clouds lost in a descent).
        if (historyKm.r > 0.0 && kmHi > 0.0) {
            historyKm.r = clamp(historyKm.r, kmLo, kmHi);
        }
        if (historyKm.r > 0.0) {
            valid = reproject(dir, historyKm.r, at);
            historyKm = valid ? texelFetch(uHistoryDistance, ivec2(at), 0).rg : historyKm;
            if (historyKm.r > 0.0 && kmHi > 0.0) {
                historyKm.r = clamp(historyKm.r, kmLo, kmHi);
            }
        }
    }

    vec4 colour;
    float km;
    float count;
    if (!valid) {
        // Nothing to go on: this frame's samples, a sample's worth.
        colour = mix(reconstructed, mean, uSmooth);
        km = reconstructedKm;
        count = 1.0;
    } else {
        vec4 history = texture(uHistory, at / vec2(textureSize(uHistory, 0)));
        vec4 held = clamp(history, lo, hi);
        // What the clamp took away counts as samples lost: a change starts
        // the count over, where the scene moved or appeared.
        float moved = length(held - history) / max(length(history), 0.05);
        count = historyKm.g * clamp(1.0 - 2.0 * moved, 0.0, 1.0);
        count = max(count, min(historyKm.g, COUNT_KEPT));
        // In motion at least about a cycle of the block's share (each pixel's
        // own sample once): a history of 1 / uMinWeight frames lagged the view
        // of a cloud whose look changes fast as the camera flies (a lumpy
        // anvil's base seen at a grazing angle from inside the slab), and the
        // clamp cut that lag wherever its box of noisy samples happened to be
        // narrow: grain over the cloud's shaded side (performance.md §30).
        float least = mix(max(uMovingWeight, uMinWeight), uMinWeight, uStill);
        float weight = max(1.0 / (count + 1.0), least);
        // The bilinear mix of four samples shows each one's noise as a dot of
        // its own: where the pixel takes most of this frame's (its history lost
        // or short) it takes their smooth mean instead.
        // In motion, wholly the smooth mean: the bilinear mix of four samples
        // printed each one's noise as a speck as the view moved.
        vec4 fresh = mix(reconstructed, mean,
                         uSmooth * max(smoothstep(SMOOTH_FROM, SMOOTH_TO, weight),
                                       1.0 - uStill));
        colour = mix(held, fresh, weight);
        km = historyKm.r > 0.0 && reconstructedKm > 0.0
           ? mix(historyKm.r, reconstructedKm, weight)
           : (weight >= 0.5 || historyKm.r <= 0.0 ? reconstructedKm : historyKm.r);
        count = min(count + 1.0, 1.0 / uMinWeight);
    }
    if (colour.a <= 0.0) {
        km = 0.0;
    }
    fragColor = colour;
    fragDistance = vec4(km, count, 0.0, 1.0);
}
