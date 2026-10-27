#version 330
// The cloud shadow map's bake (clouds.shadow_map): a card over the whole
// buffer, its vertices already in clip space (CardMaker's x, z).
in vec4 p3d_Vertex;

void main() {
    gl_Position = vec4(p3d_Vertex.x, p3d_Vertex.z, 0.0, 1.0);
}
