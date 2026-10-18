#version 330

// Terrain chunk vertex shader: camera-relative planet curvature, coastline depth
// separation, and the clip distance the ocean's reflection pass needs.

in vec4 p3d_Vertex;     // chunk-local: centre at the origin, z = height (metres)
in vec3 p3d_Normal;
in float morph_z;
in vec4 biome_a;        // snow, rock, grass, forest
in vec4 biome_b;        // desert, beach, -, skirt flag

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;   // chunk -> render; pure translation
uniform vec4 p3d_ClipPlane[1];  // reflection pass only (view space)

uniform vec3  uCamPos;          // MAIN camera in the render frame (also in the mirror pass)
uniform float uEarthCurvature;  // 1 / (2 R), as in cloud.vert; 0 = flat
uniform float uSeaLevel;        // render-frame z of sea level
uniform float uDepthBias;       // coastline separation, in depth-buffer steps
uniform vec2  uChunkUvOrigin;   // chunk centre modulo the UV period, metres

out vec3 vRelPos;     // dropped position relative to the camera, render frame
out vec2 vUv;         // world-anchored texture coordinate, metres
out float vHeight;    // height above sea level, metres (undropped)
out vec3 vNormal;
out vec4 vBiomeA;
out vec4 vBiomeB;

void main() {
    vec4 world = p3d_ModelMatrix * p3d_Vertex;
    vec2 horizontal = world.xy - uCamPos.xy;
    // The surface falls away from under the camera by d^2 / 2R. Chunks are
    // pure translations, so a world-z offset is a local-z offset.
    float drop = dot(horizontal, horizontal) * uEarthCurvature;
    vec4 local = p3d_Vertex;
    local.z -= drop;

    gl_Position = p3d_ModelViewProjectionMatrix * local;

    // Land just above the sea and seabed just below it sit within one depth
    // step of the ocean at range; nudging land nearer and seabed farther by a
    // fixed number of steps keeps the coastline stable at every distance.
    float side = clamp(world.z - uSeaLevel, -1.0, 1.0);
    gl_Position.z -= side * uDepthBias * (2.0 / 16777216.0) * gl_Position.w;

    gl_ClipDistance[0] = dot(p3d_ClipPlane[0], p3d_ModelViewMatrix * local);

    vRelPos = vec3(horizontal, world.z - drop - uCamPos.z);
    vUv = uChunkUvOrigin + p3d_Vertex.xy;
    vHeight = world.z - uSeaLevel;
    vNormal = p3d_Normal;
    vBiomeA = biome_a;
    vBiomeB = biome_b;
}
