#version 330
// The aerial-perspective bake's full-screen card (atmosphere.aerial):
// its x and z span -1..1, clip space directly, whatever the lens.

in vec4 p3d_Vertex;

void main() {
    gl_Position = vec4(p3d_Vertex.x, p3d_Vertex.z, 0.0, 1.0);
}
