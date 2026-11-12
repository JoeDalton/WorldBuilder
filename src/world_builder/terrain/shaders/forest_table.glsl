// The far forest's tables and their lookup (vegetation_far.glsl's
// shader-map), shared by the ground's vertex stage (terrain.vert: the
// fractions, per vertex: the angles and the densities vary slowly over the
// ground) and its fragment stage (the radiance). Include once per shader
// with #pragma include.

uniform int   uVegOn;            // 0: no vegetation (a barren planet, off)
// The far forest's tables (world_builder.terrain.far_forest).
uniform sampler3D uForestTable;  // x θv, y θl, z (group 8 + λ) 16 + φ:
                                 // kg, kg + kḡ, T, E
uniform float uForestD[24];      // D per group and λ (8 each)
uniform vec3  uForestAlbedo[3];  // the crowns' mean albedo per group, linear
uniform float uForestCrown[3];   // the mean crown area per tree cell
uniform int   uForestGroups;     // how many (0: no tables, no far forest)

const float FOREST_LAMBDAS = 8.0;
const float FOREST_ANGLES = 16.0;
// The leaves' hotspot's amplitude and inverse width [Eq. 8].
const float FOREST_A1 = 0.35;
const float FOREST_A2 = 5.0;

// One group's (kg, kg + kḡ, T, E) at the angles (θv, θl in [0, 1] of a
// right angle, φ in [0, 1] of a half turn) and density lambda.
vec4 forestFractions(int g, float lambda, vec3 angles) {
    float l = clamp(lambda, 0.0, 1.0) * (FOREST_LAMBDAS - 1.0);
    float i = min(floor(l), FOREST_LAMBDAS - 2.0);
    float f = l - i;
    vec2 xy = (angles.xy * (FOREST_ANGLES - 1.0) + 0.5) / FOREST_ANGLES;
    float slices = float(uForestGroups) * FOREST_LAMBDAS * FOREST_ANGLES;
    float phi = angles.z * (FOREST_ANGLES - 1.0) + 0.5;
    float z0 = ((float(g) * FOREST_LAMBDAS + i) * FOREST_ANGLES + phi) / slices;
    float z1 = z0 + FOREST_ANGLES / slices;
    return mix(textureLod(uForestTable, vec3(xy, z0), 0.0),
               textureLod(uForestTable, vec3(xy, z1), 0.0), f);
}

float forestD(int g, float lambda) {
    float l = clamp(lambda, 0.0, 1.0) * (FOREST_LAMBDAS - 1.0);
    int i = min(int(l), int(FOREST_LAMBDAS) - 2);
    return mix(uForestD[g * 8 + i], uForestD[g * 8 + i + 1], l - float(i));
}

// The trees' expected cover (Λ A: summed crown area per area) of a vertex's
// forest bytes.
float forestCover(vec4 forest) {
    float cover = 0.0;
    for (int g = 0; g < uForestGroups; ++g) {
        cover += forest[g + 1] * uForestCrown[g];
    }
    return cover;
}

// The forest's shares at a point, all groups mixed (independent: the
// ground seen through each, the crowns' terms by their shares of the view).
//   forest   the vertex's forest bytes (cover / 2, the far groups' λ)
//   up, V, L the vertical, toward the camera, toward the sun
// Returns k = (kg, kg + kḡ, T, E), albedoD = (the crowns' albedo, D),
// hotCover = (the leaves' hotspot F [Eq. 8], the expected cover Λ A).
void forestShares(vec4 forest, vec3 up, vec3 V, vec3 L,
                  out vec4 k, out vec4 albedoD, out vec2 hotCover) {
    k = vec4(1.0, 1.0, 0.0, 0.0);
    albedoD = vec4(0.0, 0.0, 0.0, 1.0);
    hotCover = vec2(1.0, 0.0);
    if (uVegOn == 0 || uForestGroups == 0 || forest.y + forest.z + forest.w <= 0.0) {
        return;
    }
    float muV = clamp(dot(V, up), 0.0, 1.0);
    float muL = dot(L, up);
    float sinV = sqrt(max(1.0 - muV * muV, 0.0));
    float sinL = sqrt(max(1.0 - muL * muL, 0.0));
    float cosPhi = sinV * sinL > 1e-4 ? (dot(V, L) - muV * muL) / (sinV * sinL) : 1.0;
    vec3 angles = vec3(acos(muV), acos(clamp(muL, 0.0, 1.0)), acos(clamp(cosPhi, -1.0, 1.0)))
                / vec3(1.5707963, 1.5707963, 3.1415927);
    float cover = forestCover(forest);
    // Near the transition, the actual trees' cover for the expected one
    // (Γ / A for Λ [§7]).
    float actual = forest.x * 2.0;  // forest.COVER_MAX
    float scale = cover > 1e-3 ? clamp(actual / cover, 0.0, 4.0) : 0.0;
    float crowns = 0.0;
    vec2 sums = vec2(0.0);
    vec3 albedo = vec3(0.0);
    float D = 1.0;
    for (int g = 0; g < uForestGroups; ++g) {
        float l = min(forest[g + 1] * scale, 1.0);
        if (l <= 1e-3) continue;
        vec4 kk = forestFractions(g, l, angles);
        k.xy *= kk.xy;
        D *= forestD(g, l);
        float c = max(1.0 - kk.y, 1e-4);
        crowns += c;
        sums += c * kk.zw;
        albedo += c * uForestAlbedo[g];
    }
    if (crowns > 0.0) {
        k.zw = sums / crowns;
        albedo /= crowns;
    }
    if (muL <= 0.0) {
        k.x = 0.0;
        k.z = 0.0;
    }
    albedoD = vec4(albedo, D);
    hotCover = vec2(1.0 - FOREST_A1 * (1.0 - exp(-FOREST_A2 * acos(clamp(dot(V, L), -1.0, 1.0)))),
                    cover);
}
