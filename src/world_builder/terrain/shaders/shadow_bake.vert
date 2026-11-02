#version 330
// A full-screen card of the shadow bake (world_builder.terrain.shadow_bake):
// its x and z span -1..1, clip space directly, whatever the lens.

in vec4 p3d_Vertex;

void main() {
    gl_Position = vec4(p3d_Vertex.x, p3d_Vertex.z, 0.0, 1.0);
}
