#version 330

// Terrain fragment shader: biome texture blending, lighting, ACES tonemap and
// haze. The tonemap and the post-tonemap haze match ocean.frag, so terrain,
// sea and cloud meet cleanly at the horizon. "Up" is the local vertical of
// the sphere (vUp), never the render frame's +Z.

in vec3 vRelPos;
in vec2 vUv;
in float vHeight;
in float vSeabed;
in vec3 vNormal;
in vec3 vUp;
in vec4 vBiomeA;   // snow, rock, grass, forest
in vec4 vBiomeB;   // desert, beach, -, skirt flag
flat in int vFace;

uniform sampler2DArray uAlbedoH;  // sRGB albedo + height
uniform sampler2DArray uNormRA;   // normal xy, roughness, ao
uniform sampler2D uNoise;         // four tiling noises, breaks up biome edges
uniform float uTileM[6];          // metres per texture repeat
uniform vec3  uMeanAlbedo[6];     // linear; drawn instead of textures far away
uniform vec3  uFlatAlbedo[6];     // linear; untextured mode and the biome view
uniform int   uTextured;
uniform float uDetailFar;         // textures fade to the mean colour by here
uniform float uBlendDepth;        // biome transition softness: ~0.1 crisp
                                  // height-cut edges, >= 1 smooth crossfades

uniform vec3  uSunDir;            // toward the sun, render frame
uniform vec3  uSunColor;
uniform vec3  uAmbient;
uniform float uExposure;
uniform vec3  uHazeColor;
uniform float uHazeDistance;      // metres of sight line in the haze shell to full haze
uniform vec3  uCamFromCentre;     // camera relative to the planet centre
uniform float uCamShellC;         // |camera|^2 - shell radius^2 (float64 on the CPU)
uniform int   uSeaSurface;

uniform int   uDebugMode;  // 0 shaded, 1 LOD, 2 biomes, 3 slope, 4 normals, 5 skirts
uniform int   uLevel;      // per chunk
uniform int   uMaxLevel;

out vec4 fragColor;

const int ROCK = 1;
const float MACRO_SCALE = 8.0;  // must match materials.MACRO_SCALE

// Each cube face's u and v axes (cubesphere.FACE_FRAMES): the texture's
// directions, once projected on the tangent plane.
const vec3 FACE_U[6] = vec3[6](
    vec3(0, 1, 0), vec3(0, -1, 0), vec3(-1, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 1, 0));
const vec3 FACE_V[6] = vec3[6](
    vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1), vec3(-1, 0, 0), vec3(1, 0, 0));

const vec3 WATER_SHALLOW = vec3(0.03, 0.09, 0.16);  // linear; ocean.py water_color is deep
const vec3 WATER_DEEP = vec3(0.02, 0.06, 0.14);

vec3 aces_tonemap(vec3 color) {
    mat3 m1 = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    mat3 m2 = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    vec3 v = m1 * color;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

// Samples inside the per-layer branches below are in non-uniform control flow,
// where implicit derivatives are undefined: pass them explicitly.
vec4 sampleLayer(sampler2DArray s, vec2 uv, vec2 gx, vec2 gy, float scale, int layer) {
    return textureGrad(s, vec3(uv * scale, float(layer)), gx * scale, gy * scale);
}

vec3 hue(float t) {
    return clamp(abs(fract(t + vec3(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
}

// Length of the sight line from the camera to this fragment that lies inside
// the haze shell. Near the ground (camera inside the shell) it is the whole
// distance; from orbit only the part through the shell counts, so the limb
// glows and the disc below stays clear.
float hazePath(float dist) {
    vec3 d = vRelPos / max(dist, 1e-3);
    float b = dot(uCamFromCentre, d);
    float c = uCamShellC;
    float disc = b * b - c;
    if (disc <= 0.0) return 0.0;
    // Roots of t^2 + 2bt + c, the stable way round.
    float q = b >= 0.0 ? -b - sqrt(disc) : -b + sqrt(disc);
    float t0 = q;
    float t1 = c / q;
    if (t0 > t1) {
        float t = t0; t0 = t1; t1 = t;
    }
    return max(0.0, min(t1, dist) - max(t0, 0.0));
}

void main() {
    float dist = length(vRelPos);
    vec3 N = normalize(vNormal);
    vec3 up = normalize(vUp);
    float upN = dot(N, up);
    vec3 L = normalize(uSunDir);
    // Planet-scale day factor: past the terminator neither the sun nor the
    // blue sky light the ground, whatever the local slope says.
    float day = smoothstep(-0.12, 0.08, dot(up, L));

    float w[6] = float[6](vBiomeA.r, vBiomeA.g, vBiomeA.b, vBiomeA.a, vBiomeB.r, vBiomeB.g);

    // World-anchored noise wobbles the weights, so biome edges are ragged
    // rather than following the vertex grid. Multiplicative: a pure biome
    // stays pure. Mipmapping averages it out far away, where it would alias.
    vec4 nearNoise = texture(uNoise, vUv / 64.0);
    vec4 farNoise = texture(uNoise, vUv / 512.0 + vec2(0.37, 0.71));
    float noise[6] = float[6](
        nearNoise.r + farNoise.b, nearNoise.g + farNoise.a, nearNoise.b + farNoise.r,
        nearNoise.a + farNoise.g, nearNoise.r + farNoise.a, nearNoise.g + farNoise.b);
    float total = 0.0;
    for (int i = 0; i < 6; ++i) {
        w[i] *= max(0.0, 1.0 + 0.9 * (noise[i] - 1.0));
        total += w[i];
    }
    for (int i = 0; i < 6; ++i) {
        w[i] /= max(total, 1e-4);
    }

    if (uDebugMode == 1) {
        fragColor = vec4(hue(float(uLevel) / float(uMaxLevel + 1)) * (0.55 + 0.45 * upN), 1.0);
        return;
    }
    if (uDebugMode == 3) {
        float slope = degrees(acos(clamp(upN, 0.0, 1.0)));
        vec3 ramp = slope < 30.0 ? mix(vec3(0.1, 0.5, 0.1), vec3(0.9, 0.9, 0.1), slope / 30.0)
                                 : mix(vec3(0.9, 0.5, 0.1), vec3(0.8, 0.0, 0.0), clamp((slope - 30.0) / 18.0, 0.0, 1.0));
        fragColor = vec4(ramp, 1.0);
        return;
    }
    if (uDebugMode == 4) {
        fragColor = vec4(0.5 + 0.5 * N, 1.0);
        return;
    }
    if (uDebugMode == 5 && vBiomeB.a > 0.5) {
        fragColor = vec4(1.0, 0.0, 1.0, 1.0);
        return;
    }

    vec3 albedo = vec3(0.0);
    vec3 meanAlbedo = vec3(0.0);
    for (int i = 0; i < 6; ++i) {
        meanAlbedo += w[i] * (uTextured == 1 ? uMeanAlbedo[i] : uFlatAlbedo[i]);
    }
    vec3 surfaceN = N;
    float roughness = 0.85;
    float ao = 1.0;
    float far = smoothstep(0.6 * uDetailFar, uDetailFar, dist);
    vec3 Tu = normalize(FACE_U[vFace] - up * dot(up, FACE_U[vFace]));
    vec3 Tv = normalize(FACE_V[vFace] - up * dot(up, FACE_V[vFace]));
    bool water = uSeaSurface == 1 && vSeabed < 0.0;

    if (water) {
        albedo = mix(WATER_SHALLOW, WATER_DEEP, clamp(-vSeabed / 3000.0, 0.0, 1.0));
        surfaceN = up;
        roughness = 0.08;
    } else if (uDebugMode == 2) {
        albedo = vec3(0.0);
        for (int i = 0; i < 6; ++i) albedo += w[i] * uFlatAlbedo[i];
    } else if (uTextured == 0 || far >= 1.0) {
        albedo = meanAlbedo;
    } else {
        // More of the large repeat with distance: hides tiling, then takes over.
        float macro = mix(0.3, 0.75, smoothstep(20.0, 600.0, dist));
        float normalFade = 1.0 - smoothstep(60.0, 600.0, dist);
        // Steep ground: rock is projected from the sides too, not stretched.
        vec3 tri = pow(abs(vec3(dot(N, Tu), dot(N, Tv), upN)), vec3(4.0));
        tri /= tri.x + tri.y + tri.z;

        vec2 gx = dFdx(vUv), gy = dFdy(vUv);
        vec2 sideX = vec2(vUv.y, vHeight), sideY = vec2(vUv.x, vHeight);
        vec2 sxGx = dFdx(sideX), sxGy = dFdy(sideX);
        vec2 syGx = dFdx(sideY), syGy = dFdy(sideY);

        vec4 layerAlbedoH[6];
        vec4 layerNRA[6];
        float best = -1.0;
        for (int i = 0; i < 6; ++i) {
            if (w[i] < 0.02) {
                layerAlbedoH[i] = vec4(0.0);
                layerNRA[i] = vec4(0.5, 0.5, 0.85, 1.0);
                continue;
            }
            float scale = 1.0 / uTileM[i];
            vec4 ah = mix(sampleLayer(uAlbedoH, vUv, gx, gy, scale, i),
                          sampleLayer(uAlbedoH, vUv, gx, gy, scale / MACRO_SCALE, i), macro);
            vec4 nra = sampleLayer(uNormRA, vUv, gx, gy, scale, i);
            if (i == ROCK && tri.z < 0.98) {
                ah = ah * tri.z
                   + sampleLayer(uAlbedoH, sideX, sxGx, sxGy, scale, i) * tri.x
                   + sampleLayer(uAlbedoH, sideY, syGx, syGy, scale, i) * tri.y;
                nra = mix(vec4(0.5, 0.5, nra.b, nra.a), nra, tri.z);
            }
            layerAlbedoH[i] = ah;
            layerNRA[i] = nra;
            best = max(best, ah.a + w[i]);
        }
        // Height blend: where two biomes meet, the locally taller texels win
        // (rock pokes through snow, sand fills the hollows). A large depth
        // lets every layer within it contribute, in proportion to its weight:
        // a smooth crossfade with only a hint of the height structure.
        float sum = 0.0;
        vec3 color = vec3(0.0);
        vec4 nra = vec4(0.0);
        for (int i = 0; i < 6; ++i) {
            if (w[i] < 0.02) continue;
            float b = max(layerAlbedoH[i].a + w[i] - (best - uBlendDepth), 0.0) * w[i];
            color += b * layerAlbedoH[i].rgb;
            nra += b * layerNRA[i];
            sum += b;
        }
        color /= max(sum, 1e-5);
        nra /= max(sum, 1e-5);
        albedo = mix(color, meanAlbedo, far);
        roughness = nra.b;
        ao = mix(nra.a, 1.0, far);

        vec2 txy = (nra.xy * 2.0 - 1.0) * normalFade;
        vec3 tn = vec3(txy, sqrt(max(0.0, 1.0 - dot(txy, txy))));
        vec3 T = normalize(Tu - N * dot(N, Tu));
        vec3 B = cross(N, T);
        surfaceN = normalize(T * tn.x + B * tn.y + N * tn.z);
    }

    vec3 V = normalize(-vRelPos);
    vec3 H = normalize(L + V);
    float lambert = max(dot(surfaceN, L), 0.0) * day;
    float gloss = 1.0 - roughness;
    float specular = gloss * gloss * 0.3 * pow(max(dot(surfaceN, H), 0.0), mix(8.0, 96.0, gloss));
    vec3 C = albedo * (uSunColor * lambert + uAmbient * (0.55 + 0.45 * dot(surfaceN, up)) * ao * day)
           + uSunColor * specular * lambert;
    if (water) {
        // Sky reflected at grazing angles (Schlick).
        float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(surfaceN, V), 0.0), 5.0);
        C += fresnel * uAmbient * 0.8 * day;
    }

    if (uDebugMode == 5) {
        C = mix(C, hue(float(uLevel) / float(uMaxLevel + 1)), 0.5);
    }
    vec3 display = aces_tonemap(C * uExposure);
    float haze = clamp(hazePath(dist) / uHazeDistance, 0.0, 1.0);
    fragColor = vec4(mix(display, uHazeColor * mix(0.02, 1.0, day), haze), 1.0);
}
