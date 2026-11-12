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
uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ModelMatrix;     // the chunk -> render
uniform vec3 uPlanetCentre;       // render frame
uniform float uPlanetRadius;
uniform float uMeshRange;         // the impostors start over its dithered band
uniform float uTreeRange;         // and end here (another band)
uniform float uPixelScale[1];     // pixels per radian
uniform float uPointScale;        // the target's pixels per view pixel (1 in
                                  // the view; the clouds' depth passes: theirs)
uniform vec4 uImpostors[32];      // per asset: layer (-1 none), diameter, centre z
uniform int uFrames;

flat out int vLayer;
flat out vec2 vTile;      // its view's tile, in tiles
flat out float vFadeIn;   // the meshes' band: 0 meshes, 1 impostor
flat out float vFadeOut;  // the far band: 1 impostor, 0 the ground's look
flat out vec3 vT1;        // its frame, render frame
flat out vec3 vT2;
flat out vec3 vUp;
flat out vec3 vSun;
flat out vec3 vSky;
flat out vec3 vAirT;
flat out vec3 vAirS;
flat out float vTint;
flat out vec2 vScreenUp;  // the tree's vertical on screen (pixels, unit): the
                          // sprite is turned to it (a rolled camera)
flat out float vSpriteScale;  // the tree's share of the sprite, across (1 in
                              // the view; less in a coarser target's wider one)
flat out float vHalfTexel;    // half a target pixel, in the tree's sprite (0 in
                              // the view: one sample per pixel)

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
    // Its vertical on screen, in pixels (the projection's aspect: square
    // pixels): a point sprite stands upright on the screen, the tree on its
    // ground, whatever the camera's roll.
    vec4 above = p3d_ModelViewProjectionMatrix
               * vec4(local + transpose(rotation) * up * (info.y * scale * 0.25), 1.0);
    vec2 screenUp = (above.xy / above.w - gl_Position.xy / gl_Position.w)
                  * vec2(p3d_ProjectionMatrix[1][1] / p3d_ProjectionMatrix[0][0], 1.0);
    vScreenUp = dot(screenUp, screenUp) > 1e-12 ? normalize(screenUp) : vec2(0.0, 1.0);
    if (info.x < 0.0 || dist < 0.85 * uMeshRange || dist > uTreeRange) {
        gl_Position = vec4(0.0, 0.0, 0.0, -1.0);  // the meshes' or none
    }
    // In a coarser target (the clouds' depth passes) one pixel wider: it
    // covers every view pixel the tree does (the clouds' near and far cards
    // are told apart by it: a tree it missed stood unfogged over the far
    // card), and at most a pixel more (more, a halo of the tree's depth).
    float size = info.y * scale / max(dist, 1.0) * uPixelScale[0];
    gl_PointSize = max(size * uPointScale + (uPointScale < 1.0 ? 1.0 : 0.0), 1.0);
    // The tree keeps its size in that wider sprite (the view's, max(size, 1)
    // pixels, in the target's): drawn over the whole of it, a small tree
    // shrank to its middle, the target's pixels sampled the air at its
    // corners and lost it (a tree unfogged in a cloud).
    vSpriteScale = uPointScale < 1.0 ? max(size, 1.0) * uPointScale / gl_PointSize : 1.0;
    vHalfTexel = uPointScale < 1.0 ? 0.5 / (gl_PointSize * vSpriteScale) : 0.0;
    vFadeIn = smoothstep(0.85 * uMeshRange, uMeshRange, dist);
    vFadeOut = 1.0 - smoothstep(0.8 * uTreeRange, uTreeRange, dist);

    vec3 ref = abs(up.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 east = normalize(cross(ref, up));
    vec3 north = cross(up, east);
    float c = cos(plant.x), s = sin(plant.x);
    vec3 t1 = c * east + s * north;
    vec3 t2 = -s * east + c * north;
    // The view's direction (toward the camera) in its frame, over the horizon.
    vec3 V = -relPos / max(dist, 1e-3);
    vec3 d = vec3(dot(V, t1), dot(V, t2), max(dot(V, up), 0.02));
    vec2 o = hemiOct(normalize(d));
    vTile = clamp(floor((o * 0.5 + 0.5) * float(uFrames)), 0.0, float(uFrames - 1));
    vLayer = int(info.x + 0.5);
    vT1 = t1;
    vT2 = t2;
    vUp = up;
    vTint = plant.w;

    vec3 L = normalize(uSunDir);
    vec3 planetPos = transpose(rotation) * (centre - uPlanetCentre);
    float height = length(planetPos) - uPlanetRadius;
    treeLight(relPos, planetPos, height, up, L, vSun, vSky);
    airAlong(relPos, L, vAirT, vAirS);
}
