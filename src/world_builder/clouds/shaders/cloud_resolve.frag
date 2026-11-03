#version 330
// The raymarch's full resolution (clouds.march), gathered over the frames:
// cloud_march.frag traces one pixel of every uBlock x uBlock block a frame;
// every pixel here blends this frame's samples around it (reconstructed
// bilinearly from the four nearest, weighted by how near the nearest is)
// into the last frame's result, moved to where the camera now sees it
// (reprojected along the view ray to the pixel's own cloud distance) and
// held within the spread of this frame's samples around it, so a change
// shows at once rather than trailing behind. How much a pixel takes from
// this frame falls with how many samples it has gathered (its count), down
// to uMinWeight: still, the steps' jitter averages away over many visits;
// no pixel of a block is treated apart from the others, so no grid shows.
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
uniform int   uBlock;
uniform ivec2 uOffset;
uniform vec2  uViewSize;
uniform int   uHistoryValid;         // 0: no usable last frame (first, resized, jumped)
uniform float uMinWeight;            // this frame's least share, with a full count

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragDistance;

// How far to look along a ray with no cloud on it (the sky at infinity).
const float NO_CLOUD = 1.0e7;
// The history is held within this many standard deviations of this frame's
// samples around the pixel.
const float CLAMP_SIGMA = 1.25;
// A sample's reach, in its spacing (uBlock pixels): the Gaussian weight a
// pixel gives the nearest one.
const float REACH = 0.3;

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
    ivec2 last = ivec2(ceil(uViewSize / float(uBlock))) - 1;
    float block = float(uBlock);

    // The march's samples around: sample k sits at view pixel k * block +
    // uOffset. The four nearest, bilinearly; the 3 x 3 around, for the clamp.
    vec2 f = (vec2(pixel) - vec2(uOffset)) / block;
    ivec2 base = ivec2(floor(f));
    vec2 w = f - vec2(base);
    vec4 reconstructed = vec4(0.0);
    float reconstructedKm = 0.0;
    float kmWeight = 0.0;
    float nearest = 1e9;
    for (int i = 0; i < 4; ++i) {
        ivec2 corner = ivec2(i & 1, i >> 1);
        ivec2 at = clamp(base + corner, ivec2(0), last);
        vec4 v = texelFetch(uMarch, at, 0);
        float km = texelFetch(uMarchDistance, at, 0).r;
        vec2 bw = mix(vec2(1.0) - w, w, vec2(corner));
        float weight = bw.x * bw.y;
        reconstructed += weight * v;
        if (km > 0.0) {
            reconstructedKm += weight * v.a * km;
            kmWeight += weight * v.a;
        }
        nearest = min(nearest, length(vec2(base + corner) * block + vec2(uOffset)
                                      - vec2(pixel)));
    }
    reconstructedKm = kmWeight > 0.0 ? reconstructedKm / kmWeight : 0.0;
    ivec2 centre = clamp(ivec2(floor(f + 0.5)), ivec2(0), last);
    vec4 mean = vec4(0.0);
    vec4 square = vec4(0.0);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec4 v = texelFetch(uMarch, clamp(centre + ivec2(x, y), ivec2(0), last), 0);
            mean += v;
            square += v * v;
        }
    }
    mean /= 9.0;
    vec4 sigma = sqrt(max(square / 9.0 - mean * mean, vec4(0.0)));
    vec4 lo = mean - CLAMP_SIGMA * sigma;
    vec4 hi = mean + CLAMP_SIGMA * sigma;

    // How much this pixel takes from this frame: its nearest sample's
    // Gaussian weight (1 on a traced pixel, and at block 1 everywhere).
    float reach = REACH * block;
    float near = block <= 1.0 ? 1.0 : exp(-0.5 * nearest * nearest / (reach * reach));

    vec2 ndc = (vec2(pixel) + 0.5) / uViewSize * 2.0 - 1.0;
    vec3 dir = rayDir(ndc);
    // Reprojected by this frame's distance first, then by the history's own
    // there (the pixel's cloud as it was): an edge keeps its place.
    vec2 at;
    bool valid = uHistoryValid != 0 && reproject(dir, reconstructedKm, at);
    vec2 historyKm = vec2(0.0);
    if (valid) {
        historyKm = texelFetch(uHistoryDistance, ivec2(at), 0).rg;
        if (historyKm.r > 0.0) {
            valid = reproject(dir, historyKm.r, at);
            historyKm = valid ? texelFetch(uHistoryDistance, ivec2(at), 0).rg : historyKm;
        }
    }

    vec4 colour;
    float km;
    float count;
    if (!valid) {
        // Nothing to go on: this frame's samples, a sample's worth.
        colour = reconstructed;
        km = reconstructedKm;
        count = near;
    } else {
        vec2 uv = at / vec2(textureSize(uHistory, 0));
        vec4 history = texture(uHistory, uv);
        vec4 held = clamp(history, lo, hi);
        // What the clamp took away counts as samples lost: a change starts
        // the count over, where the scene moved or appeared.
        float moved = length(held - history) / max(length(history), 0.05);
        count = historyKm.g * clamp(1.0 - 2.0 * moved, 0.0, 1.0);
        float weight = near * max(1.0 / (count + 1.0), uMinWeight);
        colour = mix(held, reconstructed, weight);
        km = historyKm.r > 0.0 && reconstructedKm > 0.0
           ? mix(historyKm.r, reconstructedKm, weight)
           : (weight >= 0.5 || historyKm.r <= 0.0 ? reconstructedKm : historyKm.r);
        count = min(count + near, 1.0 / uMinWeight);
    }
    if (colour.a <= 0.0) {
        km = 0.0;
    }
    fragColor = colour;
    fragDistance = vec4(km, count, 0.0, 1.0);
}
