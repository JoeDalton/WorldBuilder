#version 330
// Planet-mode far-field deck (clouds.sphere): a sphere at mid-slab height,
// shaded by cloud.frag with uDeck = 1. worldPos is in the planet frame.
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform int uDeckLayer;

in vec4 p3d_Vertex;                    // planet frame (the node sits at its centre)

out vec2  vUv;
out vec3  worldPos;
out float vRadius;
out float vFade;
flat out int vLayer;
// Planet-mode billboards' per-vertex air (cloud_sphere.vert); unused here.
out vec3 vAtmoT;
out vec3 vAtmoS;
out vec3 vSunTint;
out float vSkyTint;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_ClipDistance[0] = 1.0;  // above the sea (see cloud_sphere.vert)
    worldPos = p3d_Vertex.xyz;
    vUv = vec2(0.5);
    vRadius = 0.0;
    vFade = 1.0;
    vLayer = uDeckLayer;
    vAtmoT = vec3(1.0);
    vAtmoS = vec3(0.0);
    vSunTint = vec3(1.0);
    vSkyTint = 1.0;
}
