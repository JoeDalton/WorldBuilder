#version 330

// A tree past the meshes' range (world_builder.terrain.trees: the
// impostors): one point, a sprite as wide on screen as its asset's bounding
// sphere at its distance, centred on the asset's middle. Its view is the
// baked one (scripts/bake_tree_impostors.py) nearest the direction it is
// seen from in its own frame (its turn about the local vertical): the
// hemi-octahedral grid of uFrames x uFrames. The light is reckoned here,
// once per tree.

#define TERRAIN_LIGHT_VERTEX
#pragma include "terrain_light.glsl"

in vec4 p3d_Vertex;   // its base, from its chunk's centre (planet axes)
in vec4 plant;        // turn, scale, asset, tint

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;     // the chunk -> render
uniform vec3 uPlanetCentre;       // render frame
uniform float uPlanetRadius;
uniform float uMeshRange;         // the impostors start over its dithered band
uniform float uTreeRange;         // and end here (another band)
uniform float uPixelScale[1];     // pixels per radian
uniform vec4 uImpostors[32];      // per asset: layer (-1 none), diameter, centre z
uniform int uFrames;

flat out int vLayer;
flat out vec2 vTile;      // its view's tile, in tiles
flat out vec3 vT1;        // its frame, render frame
flat out vec3 vT2;
flat out vec3 vUp;
flat out vec3 vSun;
flat out vec3 vSky;
flat out vec3 vAirT;
flat out vec3 vAirS;
flat out float vTint;
flat out float vFadeIn;   // the meshes' band: 0 meshes, 1 impostor
flat out float vFadeOut;  // the far band: 1 impostor, 0 the ground's look

vec2 hemiOct(vec3 d) {
    vec2 p = d.xy / (abs(d.x) + abs(d.y) + d.z);
    return vec2(p.x + p.y, p.x - p.y);
}

void main() {
    int asset = int(plant.z + 0.5);
    vec4 info = uImpostors[asset];
    float scale = plant.y;
    mat3 rotation = mat3(p3d_ModelMatrix);
    vec3 base = (p3d_ModelMatrix * vec4(p3d_Vertex.xyz, 1.0)).xyz;
    vec3 up = normalize(base - uPlanetCentre);
    vec3 centre = base + up * info.z * scale;
    vec3 relPos = centre - uCamPos;
    float dist = length(relPos);
    vec3 local = transpose(rotation) * (centre - p3d_ModelMatrix[3].xyz);
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(local, 1.0);
    if (info.x < 0.0 || dist < 0.85 * uMeshRange || dist > uTreeRange) {
        gl_Position = vec4(0.0, 0.0, 0.0, -1.0);  // the meshes' or none
    }
    gl_PointSize = max(info.y * scale / max(dist, 1.0) * uPixelScale[0], 1.0);
    vFadeIn = smoothstep(0.85 * uMeshRange, uMeshRange, dist);
    vFadeOut = 1.0 - smoothstep(0.8 * uTreeRange, uTreeRange, dist);

    vec3 ref = abs(up.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 east = normalize(cross(ref, up));
    vec3 north = cross(up, east);
    float c = cos(plant.x), s = sin(plant.x);
    vec3 t1 = c * east + s * north;
    vec3 t2 = -s * east + c * north;
    vT1 = t1;
    vT2 = t2;
    vUp = up;
    // The view's direction (toward the camera) in its frame, over the horizon.
    vec3 V = -relPos / max(dist, 1e-3);
    vec3 d = vec3(dot(V, t1), dot(V, t2), max(dot(V, up), 0.02));
    vec2 o = hemiOct(normalize(d));
    vTile = clamp(floor((o * 0.5 + 0.5) * float(uFrames)), 0.0, float(uFrames - 1));
    vLayer = int(info.x + 0.5);
    vTint = plant.w;

    vec3 L = normalize(uSunDir);
    vec3 planetPos = transpose(rotation) * (centre - uPlanetCentre);
    float height = length(planetPos) - uPlanetRadius;
    treeLight(relPos, planetPos, height, up, L, vSun, vSky);
    airAlong(relPos, L, vAirT, vAirS);
}
