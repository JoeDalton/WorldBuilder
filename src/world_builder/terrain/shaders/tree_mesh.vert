#version 330

// A tree asset drawn once per plant near the camera (world_builder.terrain.
// trees: the meshes): each instance's base (from the terrain root's origin,
// its frame), turn about the local vertical and scale from uPlants. The
// model stands on its base, z up; here up is the planet's vertical there.
// The light is reckoned per vertex (terrain_light.glsl's treeLight, and the
// air to the camera).

#define TERRAIN_LIGHT_VERTEX
#pragma include "terrain_light.glsl"

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;     // the trees' root (the terrain root's) -> render
uniform vec3 uPlanetCentre;       // the planet's centre, render frame
uniform float uPlanetRadius;
uniform float uMeshRange;         // the meshes end here (a dithered band)
uniform samplerBuffer uPlants;    // per plant: (base, turn), (scale, tint, 0, 0)

out vec2 vUv;
out float vFade;      // 1 within the meshes' range, 0 past it
out vec3 vNormal;     // render frame
out vec3 vUp;
out vec3 vSun;
out vec3 vSky;
out vec3 vAirT;
out vec3 vAirS;
out float vTint;
out float vHeight;    // the vertex's height in the model, metres

void main() {
    vec4 a = texelFetch(uPlants, 2 * gl_InstanceID);
    vec4 b = texelFetch(uPlants, 2 * gl_InstanceID + 1);
    mat3 rotation = mat3(p3d_ModelMatrix);
    vec3 base = (p3d_ModelMatrix * vec4(a.xyz, 1.0)).xyz;  // render frame
    vec3 up = normalize(base - uPlanetCentre);
    vec3 ref = abs(up.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 east = normalize(cross(ref, up));
    vec3 north = cross(up, east);
    float c = cos(a.w), s = sin(a.w);
    vec3 t1 = c * east + s * north;
    vec3 t2 = -s * east + c * north;
    float scale = b.x;
    vec3 world = base + (t1 * p3d_Vertex.x + t2 * p3d_Vertex.y + up * p3d_Vertex.z) * scale;
    float dist = length(base - uCamPos);
    vFade = 1.0 - smoothstep(0.85 * uMeshRange, uMeshRange, dist);
    // Back to the root's frame for its projection.
    vec3 local = transpose(rotation) * (world - p3d_ModelMatrix[3].xyz);
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(local, 1.0);
    if (dist > uMeshRange) {
        gl_Position = vec4(0.0, 0.0, 0.0, -1.0);  // the impostors' (clipped whole)
    }
    vUv = p3d_MultiTexCoord0;
    vNormal = t1 * p3d_Normal.x + t2 * p3d_Normal.y + up * p3d_Normal.z;
    vUp = up;
    vTint = b.y;
    vHeight = p3d_Vertex.z * scale;

    vec3 relPos = world - uCamPos;
    vec3 L = normalize(uSunDir);
    vec3 planetPos = transpose(rotation) * (world - uPlanetCentre);
    float height = length(planetPos) - uPlanetRadius;
    treeLight(relPos, planetPos, height, up, L, vSun, vSky);
    airAlong(relPos, L, vAirT, vAirS);
}
