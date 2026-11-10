// The canopy's march (world_builder.terrain.vegetation): what grows in a
// chunk's shell (canopy.vert), along the view ray from where it enters the
// shell (canopy.frag) or from the camera inside it (canopy_inside.frag).
//
// The plants are vegetation_cells.glsl's: one at most per cell, each an
// ellipsoid crown (a tree's on a trunk near the camera). The ray walks the
// cells it crosses (a 2D DDA in the face's texture coordinates), each the
// exact hit on its cell's crown, until a hit, the ground, the shell's top,
// the chunk's edge (the next chunk's shell takes the ray from its wall) or
// uVegSteps cells. On a hit the next cell is tested too: a crown reaches a
// little into its neighbours (VEG_REACH).
//
// The march runs in the chunk's tangent frame at its middle vertex O (sea
// level): Q a point from O, planet axes; its grid coordinate g = 32 +
// (Fu.Q, Fv.Q) (Fu, Fv: the grid's dual axes, steps per metre); its
// height above sea level h = Q.up + |Q_t|^2 / 2R (the sphere's drop). The
// ground at g is the chunk's own (its canopy layer, bilinear), and its
// biome weights the nearest vertex's.

#pragma include "terrain_light.glsl"
#pragma include "vegetation_cells.glsl"

uniform sampler2DArray uCanopy;  // the canopy layers of a page: per chunk
                                 // 65 x 66: ground, forest, grass, desert +
                                 // 256 rock (bytes); row 65 its frame
uniform sampler2D uNoise;        // four tiling noises (the crowns' leaves)
uniform int   uVegOn;
uniform float uVegRange;         // plants drawn out to here, metres (a dithered fade)
uniform float uVegDetail;        // trunks and leafy normals out to here
uniform float uVegShadowRange;   // the crowns' shadows on each other out to here
#define uVegShadow uVegShadowRange
uniform int   uVegSteps;         // the most cells a ray walks
uniform float uVegSeaFloor;      // nothing grows on ground below this (the sea)
uniform mat4  p3d_ModelViewProjectionMatrix;
uniform mat4  p3d_ModelMatrix;

in vec3 vLocal;
in vec3 vRay;
in float vLift;
in vec3 vSun;
in vec3 vSky;
in vec3 vAirT;
in vec3 vAirS;
flat in vec4 vEu;
flat in vec3 vEv;
flat in vec3 vFu;
flat in vec3 vFv;
flat in vec3 vOffset;
flat in vec3 vUpO;
flat in vec3 vCentre;
flat in vec3 vUvStep;
flat in ivec4 vIds;

out vec4 fragColor;

// This fragment's chunk: its canopy layer and frame.
int gLayer;
vec2 gUv0;        // the chunk's first vertex's uv, metres
float gStep;      // uv metres per grid step
vec3 gEu, gEv;    // metres per grid step along u and v (tangent), planet axes
vec3 gFu, gFv;    // grid steps per metre (the dual axes)
vec3 gUp;         // the vertical at O
float gRadius;    // the planet's, metres

float canopyGround(vec2 g) {
    g = clamp(g, vec2(0.0), vec2(64.0));
    return texture(uCanopy, vec3((g + 0.5) / vec2(65.0, 131.0), float(gLayer))).r;
}

// The quad at g: its lowest ground and the shell's highest top (the drawn
// shell never leaves them).
vec2 canopyQuad(vec2 g) {
    ivec2 q = clamp(ivec2(floor(g)), ivec2(0), ivec2(63));
    return texelFetch(uCanopy, ivec3(q.x, 66 + q.y, gLayer), 0).gb;
}

vec4 canopyWeights(vec2 g) {
    ivec2 c = clamp(ivec2(floor(g + 0.5)), ivec2(0), ivec2(64));
    vec4 t = texelFetch(uCanopy, ivec3(c, gLayer), 0);
    float rockDesert = t.a;
    float rock = floor(rockDesert / 256.0);
    return vec4(t.g, t.b, (rockDesert - 256.0 * rock) / 255.0, rock / 255.0);
}

vec2 canopyGrid(vec3 Q) {
    return vec2(32.0) + vec2(dot(gFu, Q), dot(gFv, Q));
}

float canopyHeight(vec3 Q) {
    float v = dot(Q, gUp);
    return v + (dot(Q, Q) - v * v) / (2.0 * gRadius);
}

// The point at grid coordinate g, h metres above sea level, from O.
vec3 canopyPoint(vec2 g, float h) {
    vec3 t = gEu * (g.x - 32.0) + gEv * (g.y - 32.0);
    return t + gUp * (h - dot(t, t) / (2.0 * gRadius));
}

// The nearest hit of the ray (o, d) on an ellipsoid at c with horizontal
// semi-axis r and vertical b, past tMin; 1e9 = none (or the camera inside:
// a crown is not seen from within).
float hitEllipsoid(vec3 o, vec3 d, vec3 c, float r, float b, float tMin) {
    vec3 q = o - c;
    float qv = dot(q, gUp), dv = dot(d, gUp);
    vec3 qh = q - gUp * qv, dh = d - gUp * dv;
    float ir2 = 1.0 / (r * r), ib2 = 1.0 / (b * b);
    float A = dot(dh, dh) * ir2 + dv * dv * ib2;
    float B = dot(qh, dh) * ir2 + qv * dv * ib2;
    float C = dot(qh, qh) * ir2 + qv * qv * ib2 - 1.0;
    float disc = B * B - A * C;
    if (disc < 0.0 || C < 0.0) {
        return 1e9;
    }
    float s = sqrt(disc);
    float t = (-B - s) / A;
    // The march starting inside it (the shell's top through a crown): there.
    return t >= tMin ? t : ((-B + s) / A >= tMin ? tMin : 1e9);
}

// A vertical trunk of radius r from the base c0 up by len; 1e9 = none.
float hitTrunk(vec3 o, vec3 d, vec3 c0, float r, float len, float tMin) {
    vec3 q = o - c0;
    float qv = dot(q, gUp), dv = dot(d, gUp);
    vec3 qh = q - gUp * qv, dh = d - gUp * dv;
    float A = dot(dh, dh);
    float B = dot(qh, dh);
    float disc = B * B - A * (dot(qh, qh) - r * r);
    if (disc < 0.0 || A < 1e-8) {
        return 1e9;
    }
    float t = (-B - sqrt(disc)) / A;
    float v = qv + dv * t;
    return t < tMin || v < 0.0 || v > len ? 1e9 : t;
}

// A cell's plant, all the march and the shading need of it.
struct Grown {
    Plant plant;
    vec3 centre;   // its crown's, from O
    vec2 crown;    // plantCrown's
    float base;    // its ground, metres above sea level
    float forest;  // the forest weight at its cell
};

bool grownAt(ivec2 cell, ivec2 period, int face, out Grown g) {
    vec2 centreG = ((vec2(cell) + 0.5) * VEG_CELL - gUv0) / gStep;
    vec4 w = canopyWeights(centreG);
    g.plant = plantAt(cell, period, face, w);
    if (g.plant.kind == VEG_NONE) {
        return false;
    }
    vec2 at = (g.plant.uv - gUv0) / gStep;
    g.base = canopyGround(at);
    g.crown = plantCrown(g.plant);
    g.centre = canopyPoint(at, g.base + g.crown.x);
    g.forest = w.x;
    return g.base >= uVegSeaFloor;
}

// The ray's hit on a cell's plant past tMin (1e9: none); trunk: on its
// trunk (detail: trunks are tested).
float testCell(ivec2 cell, ivec2 period, int face, vec3 o, vec3 d, float tMin,
               bool detail, out bool trunk) {
    trunk = false;
    Grown g;
    if (!grownAt(cell, period, face, g)) {
        return 1e9;
    }
    float t = hitEllipsoid(o, d, g.centre, g.plant.radius, g.crown.y, tMin);
    if (detail && g.plant.kind == VEG_TREE) {
        float tt = hitTrunk(o, d, g.centre - gUp * (g.crown.x + 0.3),
                            0.12 + 0.012 * g.plant.height, g.crown.x + 0.3, tMin);
        if (tt < t) {
            t = tt;
            trunk = true;
        }
    }
    return t;
}

// Interleaved gradient noise: a stable dither per pixel.
float dither() {
    return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
}

void main() {
    // The shell on the ground where nothing grows.
    if (uVegOn == 0 || vLift < 0.05) {
        discard;
    }
    gLayer = vIds.x;
    gUv0 = vUvStep.xy;
    gStep = vUvStep.z;
    int face = vIds.y;
    ivec2 period = vIds.zw;
    gEu = vEu.xyz;
    gRadius = vEu.w;
    gEv = vEv;
    gFu = vFu;
    gFv = vFv;
    gUp = vUpO;
    vec3 offset = vOffset;  // O from the chunk's centre

    float entry = length(vRay);
    vec3 d = vRay / max(entry, 1e-6);
    vec3 cam = vLocal - offset - vRay;  // the camera, from O
#ifdef CANOPY_INSIDE
    // From the camera to the shell's far side.
    float t0 = 0.2;
    float t1 = entry;
#else
    float t0 = entry;
    float t1 = 1e7;
#endif
    if (t0 > uVegRange) {
        discard;
    }

    // The walk, in cells (uv / VEG_CELL) along t.
    vec2 uvCam = gUv0 + canopyGrid(cam) * gStep;
    vec2 dir = gStep * vec2(dot(gFu, d), dot(gFv, d)) / VEG_CELL;
    vec2 p = (uvCam / VEG_CELL) + dir * t0;
    ivec2 cell = ivec2(floor(p));
    ivec2 stepDir = ivec2(dir.x >= 0.0 ? 1 : -1, dir.y >= 0.0 ? 1 : -1);
    vec2 tDelta = vec2(abs(dir.x) > 1e-9 ? 1.0 / abs(dir.x) : 1e9,
                       abs(dir.y) > 1e-9 ? 1.0 / abs(dir.y) : 1e9);
    vec2 tMax = t0 + abs(vec2(cell) + vec2(greaterThanEqual(dir, vec2(0.0))) - p) * tDelta;
    float reach = (VEG_REACH - 0.5) / max(abs(dir.x), abs(dir.y));  // in t

    float bestT = 1e9;
    ivec2 bestCell = cell;
    bool bestTrunk = false;
    bool detail = t0 < uVegDetail;
    bool last = false;  // a hit: the next cell is the last tested
    for (int i = 0; i < uVegSteps; ++i) {
        bool trunk;
        float t = testCell(cell, period, face, cam, d, t0, detail, trunk);
        if (t < bestT) {
            bestT = t;
            bestCell = cell;
            bestTrunk = trunk;
        }
        if (last) break;
        float tNext = min(tMax.x, tMax.y);
        if (bestT < tNext) {
            // A hit inside this cell: the next cell's crown may reach
            // nearer, if the hit lies within its reach of the next cell.
            if (tNext - bestT > reach) break;
            last = true;
        } else {
            if (tNext > t1 || tNext > uVegRange) break;
            vec3 q = cam + d * tNext;
            vec2 g = canopyGrid(q);
            // Past the chunk's edge: the next chunk's shell draws it.
            if (any(lessThan(g, vec2(-0.5))) || any(greaterThan(g, vec2(64.5)))) break;
            // Under the ground, or out of the shell (over its lift): a ray
            // entering it again meets it again, its own fragment then.
            vec2 shell = canopyQuad(g);  // the quad's lowest ground, top
            float h = canopyHeight(q);
            if (h < shell.x - 0.5 || h > shell.y + 0.3) break;
        }
        if (tMax.x < tMax.y) {
            tMax.x += tDelta.x;
            cell.x += stepDir.x;
        } else {
            tMax.y += tDelta.y;
            cell.y += stepDir.y;
        }
    }
    if (bestT > 1e8 || bestT > t1) {
        discard;
    }
    // Faded out by the range, dithered.
    if (dither() > 1.0 - smoothstep(0.7 * uVegRange, uVegRange, bestT)) {
        discard;
    }

    Grown g;
    grownAt(bestCell, period, face, g);
    Plant plant = g.plant;
    vec3 hit = cam + d * bestT;
    vec3 q = hit - g.centre;
    float qv = dot(q, gUp);
    vec3 qh = q - gUp * qv;
    vec3 nP = bestTrunk ? normalize(qh)
                        : normalize(qh / (plant.radius * plant.radius)
                                    + gUp * (qv / (g.crown.y * g.crown.y)));
    mat3 rotation = mat3(p3d_ModelMatrix);
    float height = canopyHeight(hit);
    vec3 up = normalize(rotation * gUp);
    vec3 L = normalize(uSunDir);
    vec3 Lp = transpose(rotation) * L;
    float leafy = 1.0 - smoothstep(0.5 * uVegDetail, uVegDetail, bestT);
    if (!bestTrunk && leafy > 0.0) {
        // Leaf clumps: the normal tilted by the noise around the crown.
        vec2 hitUv = gUv0 + canopyGrid(hit) * gStep;
        vec4 nz = texture(uNoise, hitUv / 3.1 + vec2(0.0, height / 2.3));
        vec3 bump = (nz.xyz - 0.5) * 1.6;
        nP = normalize(nP + leafy * (bump - gUp * dot(bump, gUp) * 0.5));
    }
    vec3 N = normalize(rotation * nP);
    float inTree = clamp((height - g.base) / plant.height, 0.0, 1.0);
    vec3 albedo = bestTrunk ? vec3(0.045, 0.035, 0.025) : plantColor(plant, g.forest);
    float ao = plant.kind == VEG_TREE ? mix(0.3, 1.0, inTree) : mix(0.5, 1.0, inTree);
    // The crown's inner side (facing down) sees little sky.
    ao *= 0.6 + 0.4 * clamp(dot(nP, gUp) + 0.5, 0.0, 1.0);
    // The neighbours' crowns between the hit and the sun, as the far look
    // reckons them (vegetation_far.glsl): the deeper in the canopy and the
    // lower the sun, the more of it in their shade.
    float muL = max(dot(Lp, gUp), 0.12);
    float cover = vegCoverage(vec4(g.forest, 1.0 - g.forest, 0.0, 0.0)).y;
    float shaded = exp(-2.5 * cover * (1.0 - inTree) / muL);
    // Foliage lets light through: the sun wraps past the terminator.
    float wrap = bestTrunk ? max(dot(N, L), 0.0)
                           : clamp((dot(N, L) + 0.4) / 1.4, 0.0, 1.0);
    // And through the leaves: a crown between the camera and the sun glows.
    vec3 V = -d;
    float through = bestTrunk ? 0.0
                  : 0.45 * pow(max(dot(-Lp, V), 0.0), 3.0) * mix(0.4, 1.0, inTree);
    vec3 C = albedo * (vSun * (wrap * shaded + through)
                       + vSky * (0.55 + 0.45 * dot(N, up)) * ao);
    fragColor = displayColor(C * vAirT + vAirS);
    vec4 clip = p3d_ModelViewProjectionMatrix * vec4(offset + hit, 1.0);
    gl_FragDepth = 0.5 * clip.z / clip.w + 0.5;
}
