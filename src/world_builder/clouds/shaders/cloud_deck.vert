#version 330
// A planet's far-field deck (clouds.deck): a sphere at mid-slab height,
// shaded by cloud.frag. worldPos is in the planet frame.
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform int uDeckLayer;

in vec4 p3d_Vertex;                    // planet frame (the node sits at its centre)

out vec3  worldPos;
flat out int vLayer;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    gl_ClipDistance[0] = 1.0;  // above the sea: the reflection clips below it
    worldPos = p3d_Vertex.xyz;
    vLayer = uDeckLayer;
}
